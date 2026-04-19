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
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)


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

        words: list[str] = []
        start_times: list[float] = []
        end_times: list[float] = []
        for item in results[0]:
            words.append(item.text)
            start_times.append(round(item.start_time, 4))
            end_times.append(round(item.end_time, 4))

        return {
            "word_alignment": {
                "words": words,
                "word_start_times_seconds": start_times,
                "word_end_times_seconds": end_times,
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
