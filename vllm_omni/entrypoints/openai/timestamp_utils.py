# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for forced-alignment timestamp handling.

Used by both the HTTP SSE streaming path (``serving_speech.py``) and the
WebSocket streaming handler (``serving_speech_stream.py``) so the two
transports stay consistent.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np


def pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw 16-bit signed PCM bytes to float32 numpy array in [-1, 1]."""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def offset_timestamps(ts_info: dict[str, Any], offset: float) -> None:
    """Shift all word timestamps by ``offset`` seconds (in-place).

    Used to make per-sentence timestamps global relative to the start of a
    concatenated audio stream.
    """
    if offset == 0.0:
        return
    wa = ts_info.get("word_alignment")
    if not wa:
        return
    wa["word_start_times_seconds"] = [
        round(t + offset, 4) for t in wa["word_start_times_seconds"]
    ]
    wa["word_end_times_seconds"] = [
        round(t + offset, 4) for t in wa["word_end_times_seconds"]
    ]


def decode_audio_to_pcm(audio_bytes: bytes, response_format: str) -> bytes:
    """Decode encoded audio (WAV, FLAC, …) to raw 16-bit signed PCM bytes.

    For ``pcm`` the bytes are returned as-is. For other formats soundfile is
    used to decode to int16 PCM.
    """
    if response_format == "pcm":
        return audio_bytes
    try:
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError("soundfile is required to decode non-PCM audio for alignment") from e
    data, _ = sf.read(io.BytesIO(audio_bytes), dtype="int16", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=-1).astype(np.int16)
    return data.tobytes()
