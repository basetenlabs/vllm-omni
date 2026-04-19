"""Wrapper for Qwen3-ForcedAligner-0.6B word-level timestamp alignment.

The aligner takes synthesized audio and its corresponding text, and
produces word-level start/end timestamps.  It runs in a dedicated
thread-pool so alignment never blocks the async event loop.

Activation:
    Set the ``FORCED_ALIGNER_MODEL`` environment variable (e.g.
    ``Qwen/Qwen3-ForcedAligner-0.6B`` or a local path).  The model is
    loaded lazily on the first alignment request.

Configuration env vars:
    FORCED_ALIGNER_MODEL   — HF repo id or local path (required to enable)
    FORCED_ALIGNER_DEVICE  — torch device string (default ``cuda:0``)
    FORCED_ALIGNER_DTYPE   — ``bfloat16`` | ``float16`` | ``float32``
                             (default ``bfloat16``)
"""

import asyncio
import os
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)


# The underlying ``Qwen3ForcedAligner`` tokenizes the input text by (a)
# splitting on any whitespace, (b) stripping every character that isn't a
# letter / digit / apostrophe, and (c) dropping pure-punctuation segments
# entirely.  That means its ``words`` output cannot be joined back into the
# original text, which breaks downstream consumers that key a cache on the
# synthesized text (they need ``" ".join(words) == original_text``).
#
# The helpers below re-derive the original whitespace-delimited tokens and
# attach the matching start/end times produced by the aligner so the invariant
# holds.  They mirror the tokenization logic in
# ``qwen_asr.Qwen3ForceAlignProcessor`` so the mapping is exact.


def _is_kept_char(ch: str) -> bool:
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # Extension A
        or 0x20000 <= code <= 0x2A6DF  # Extension B
        or 0x2A700 <= code <= 0x2B73F  # Extension C
        or 0x2B740 <= code <= 0x2B81F  # Extension D
        or 0x2B820 <= code <= 0x2CEAF  # Extension E
        or 0xF900 <= code <= 0xFAFF  # Compatibility Ideographs
    )


def _clean_token(token: str) -> str:
    return "".join(ch for ch in token if _is_kept_char(ch))


def _split_cjk_subtokens(cleaned: str) -> list[str]:
    """Split a cleaned token into the sub-tokens the aligner would emit.

    For pure-Latin tokens this is a single-element list; CJK characters are
    each emitted as their own sub-token.
    """
    if not cleaned:
        return []
    out: list[str] = []
    buf: list[str] = []
    for ch in cleaned:
        if _is_cjk_char(ch):
            if buf:
                out.append("".join(buf))
                buf = []
            out.append(ch)
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def _attach_timestamps_to_original_tokens(
    text: str,
    aligner_starts: list[float],
    aligner_ends: list[float],
) -> tuple[list[str], list[float], list[float]]:
    """Map aligner output onto the whitespace tokens of ``text``.

    Returns ``(tokens, starts, ends)`` such that ``" ".join(tokens) == text``.
    Pure-punctuation tokens (and empties from runs of spaces) receive a
    zero-duration timestamp anchored at the previous token's end time so the
    sequence stays monotonic.

    Only handles space-separated languages correctly; languages whose aligner
    tokenization doesn't preserve whitespace boundaries (Japanese, Korean) are
    still returned best-effort but may not satisfy the join invariant.
    """
    tokens = text.split(" ")
    out_starts: list[float] = [0.0] * len(tokens)
    out_ends: list[float] = [0.0] * len(tokens)

    ai = 0
    n = len(aligner_starts)
    prev_end = 0.0

    for i, tok in enumerate(tokens):
        cleaned = _clean_token(tok)
        if not cleaned:
            out_starts[i] = prev_end
            out_ends[i] = prev_end
            continue
        sub = _split_cjk_subtokens(cleaned)
        k = len(sub)
        if k == 0 or ai + k > n:
            out_starts[i] = prev_end
            out_ends[i] = prev_end
            continue
        out_starts[i] = aligner_starts[ai]
        out_ends[i] = aligner_ends[ai + k - 1]
        prev_end = out_ends[i]
        ai += k

    if ai != n:
        logger.debug(
            "Forced-aligner produced %d tokens but original text needed %d; "
            "timestamps may be approximate.",
            n,
            ai,
        )

    return tokens, out_starts, out_ends


class ForcedAligner:
    """Thread-safe wrapper around ``Qwen3ForcedAligner``."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._dtype = dtype
        self._model: Any = None
        self._load_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            import torch
            from qwen_asr import Qwen3ForcedAligner

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(self._dtype, torch.bfloat16)

            logger.info(
                "Loading ForcedAligner %s on %s (%s)",
                self._model_name,
                self._device,
                self._dtype,
            )
            self._model = Qwen3ForcedAligner.from_pretrained(
                self._model_name,
                dtype=torch_dtype,
                device_map=self._device,
            )
            logger.info("ForcedAligner loaded successfully")

    # ------------------------------------------------------------------
    # Synchronous alignment (runs inside the thread-pool)
    # ------------------------------------------------------------------

    def _align_sync(
        self,
        audio: np.ndarray,
        sample_rate: int,
        text: str,
        language: str = "Auto",
    ) -> dict[str, Any]:
        """Run forced alignment and return a ``timestamp_info`` dict."""
        self._ensure_loaded()

        results = self._model.align(
            audio=(audio, sample_rate),
            text=text,
            language=language,
        )

        empty: dict[str, Any] = {
            "word_alignment": {
                "words": [],
                "word_start_times_seconds": [],
                "word_end_times_seconds": [],
            }
        }

        if not results or not results[0]:
            return empty

        aligner_starts: list[float] = []
        aligner_ends: list[float] = []
        for item in results[0]:
            aligner_starts.append(item.start_time)
            aligner_ends.append(item.end_time)

        # Re-attach timestamps to the original whitespace tokens so that
        # " ".join(words) == text (required by text-keyed caches upstream).
        tokens, token_starts, token_ends = _attach_timestamps_to_original_tokens(
            text, aligner_starts, aligner_ends
        )

        return {
            "word_alignment": {
                "words": tokens,
                "word_start_times_seconds": [round(t, 4) for t in token_starts],
                "word_end_times_seconds": [round(t, 4) for t in token_ends],
            }
        }

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def align(
        self,
        audio: np.ndarray,
        sample_rate: int,
        text: str,
        language: str = "Auto",
    ) -> dict[str, Any]:
        """Run alignment without blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._align_sync,
            audio,
            sample_rate,
            text,
            language,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def create_aligner_from_env() -> ForcedAligner | None:
    """Create a ``ForcedAligner`` from environment variables, or ``None``."""
    model = os.environ.get("FORCED_ALIGNER_MODEL")
    if not model:
        return None

    device = os.environ.get("FORCED_ALIGNER_DEVICE", "cuda:0")
    dtype = os.environ.get("FORCED_ALIGNER_DTYPE", "bfloat16")
    logger.info(
        "Timestamps enabled — ForcedAligner model=%s device=%s dtype=%s",
        model,
        device,
        dtype,
    )
    return ForcedAligner(model_name=model, device=device, dtype=dtype)
