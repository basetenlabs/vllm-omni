#!/usr/bin/env python3
"""Minimal WebSocket client for Voxtral TTS voice cloning via /v1/audio/speech/stream.

Requires a vLLM-Omni server running the Voxtral TTS stage config, for example:

    vllm serve mistralai/Voxtral-4B-TTS-2603 \\
        --stage-configs-path vllm_omni/model_executor/stage_configs/voxtral_tts.yaml \\
        --omni --port 8091 --trust-remote-code --enforce-eager

Two cloning flows (same protocol as docs/serving/speech_api.md):

1) Reference audio in session (one-shot)
   - Send session.config with ref_audio (HTTPS URL or data:audio/...;base64,...),
     task_type \"Base\", optional ref_text, and no preset voice.

2) Upload then synthesize (reuses stored sample on the server)
   - Send voice.upload (base64 audio + consent + name, optional ref_text).
   - Send session.config with voice set to that name (server maps it to clone audio).

Dependencies:
    pip install websockets
    # optional, nicer WAV writing:
    pip install soundfile numpy

Examples:
    python ws_voice_clone_demo.py \\
        --text \"Hello, this is a quick clone test.\" \\
        --ref-audio ./my_voice.wav \\
        --ref-text \"Exact transcript of the reference clip.\"

    python ws_voice_clone_demo.py \\
        --text \"Hello again.\" \\
        --upload ./my_voice.wav \\
        --voice-name my_clone \\
        --consent local_dev_consent_001 \\
        --ref-text \"Exact transcript of the reference clip.\"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import struct
import sys
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets", file=sys.stderr)
    raise SystemExit(1)

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    np = None  # type: ignore[assignment]
    sf = None  # type: ignore[assignment]

SAMPLE_RATE = 24000


def _file_to_data_url(path: Path) -> str:
    """Encode a local file as a data URL (keeps cloning self-contained for the client)."""
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "application/octet-stream"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _write_wav_pcm16(path: str, pcm: "np.ndarray", sample_rate: int) -> None:
    data = pcm.astype("int16").tobytes()
    n_channels = 1
    sample_width = 2
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(
            struct.pack(
                "<IHHIIHH",
                16,
                1,
                n_channels,
                sample_rate,
                sample_rate * n_channels * sample_width,
                n_channels * sample_width,
                16,
            )
        )
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


async def _recv_json_until(ws, *types: str) -> dict:
    while True:
        msg = await ws.recv()
        if isinstance(msg, bytes):
            continue
        data = json.loads(msg)
        t = data.get("type")
        if t == "error":
            raise RuntimeError(data.get("message", data))
        if t in types:
            return data


async def _upload_voice_ws(
    ws,
    *,
    path: Path,
    name: str,
    consent: str,
    ref_text: str | None,
) -> None:
    raw = path.read_bytes()
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "audio/wav"
    payload: dict = {
        "type": "voice.upload",
        "name": name,
        "consent": consent,
        "audio_data": base64.b64encode(raw).decode("ascii"),
        "mime_type": mime,
        "filename": path.name,
    }
    if ref_text:
        payload["ref_text"] = ref_text
    await ws.send(json.dumps(payload))
    res = await _recv_json_until(ws, "voice.upload.result")
    if not res.get("success", True):
        raise RuntimeError(f"voice.upload failed: {res}")


async def run_clone(
    url: str,
    text: str,
    output: str,
    *,
    ref_audio: str | None,
    ref_text: str | None,
    language: str | None,
    upload_path: Path | None,
    upload_name: str | None,
    consent: str | None,
) -> None:
    if bool(upload_path) != bool(upload_name) or (upload_path and not consent):
        raise ValueError("Upload mode requires --upload, --voice-name, and --consent")

    if upload_path and ref_audio:
        raise ValueError("Use either --upload or --ref-audio, not both")

    if not upload_path:
        if not ref_audio:
            raise ValueError("Provide --ref-audio (URL or file) or use --upload")
        if ref_audio.startswith("http://") or ref_audio.startswith("https://"):
            ref_payload: str | None = ref_audio
        else:
            p = Path(ref_audio).expanduser().resolve()
            if not p.is_file():
                raise FileNotFoundError(p)
            ref_payload = _file_to_data_url(p)
    else:
        ref_payload = None

    config: dict = {
        "response_format": "wav",
        "speed": 1.0,
        "split_granularity": "sentence",
    }
    if language:
        config["language"] = language

    if upload_path:
        config["voice"] = upload_name
    else:
        config["task_type"] = "Base"
        config["ref_audio"] = ref_payload
        if ref_text:
            config["ref_text"] = ref_text

    async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
        if upload_path:
            await _upload_voice_ws(
                ws,
                path=upload_path,
                name=upload_name or "",
                consent=consent or "",
                ref_text=ref_text,
            )

        await ws.send(json.dumps({"type": "session.config", **config}))

        await ws.send(json.dumps({"type": "input.text", "text": text}))
        await ws.send(json.dumps({"type": "input.done"}))

        chunks: list[bytes] = []
        pcm_parts: list["np.ndarray"] = []

        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                chunks.append(msg)
                continue
            data = json.loads(msg)
            mtype = data.get("type")
            if mtype == "audio.start":
                chunks = []
            elif mtype == "audio.done":
                raw = b"".join(chunks)
                if data.get("error"):
                    raise RuntimeError(f"audio.done flagged error for sentence {data.get('sentence_index')}")
                if sf is not None and np is not None:
                    import io

                    pcm, sr = sf.read(io.BytesIO(raw), dtype="float32")
                    if pcm.ndim > 1:
                        pcm = pcm[:, 0]
                    pcm_parts.append((pcm * 32767.0).clip(-32768, 32767).astype("int16"))
                elif np is not None:
                    usable = len(raw) - (len(raw) % 2)
                    if usable:
                        pcm_parts.append(np.frombuffer(raw[:usable], dtype="<i2").copy())
                else:
                    raise SystemExit("Install numpy (and soundfile) to decode WAV frames: pip install numpy soundfile")
                chunks = []
            elif mtype == "session.done":
                break
            elif mtype == "error":
                raise RuntimeError(data.get("message", data))

    if not pcm_parts:
        print("No audio received.", file=sys.stderr)
        raise SystemExit(1)

    if np is None:
        raise SystemExit("numpy required to concatenate output")
    combined = np.concatenate(pcm_parts)
    out_path = Path(output)
    if sf is not None:
        if out_path.suffix.lower() not in {".wav", ".flac"}:
            out_path = out_path.with_suffix(".wav")
        sf.write(str(out_path), combined.astype("float32") / 32768.0, SAMPLE_RATE)
    else:
        if out_path.suffix.lower() != ".wav":
            out_path = out_path.with_suffix(".wav")
        _write_wav_pcm16(str(out_path), combined, SAMPLE_RATE)

    print(f"Wrote {out_path.resolve()} ({len(combined) / SAMPLE_RATE:.2f}s)")


def main() -> None:
    p = argparse.ArgumentParser(description="Voxtral TTS voice clone over WebSocket")
    p.add_argument(
        "--url",
        default="ws://127.0.0.1:8091/v1/audio/speech/stream",
        help="WebSocket URL",
    )
    p.add_argument("--text", required=True, help="Text to synthesize")
    p.add_argument("-o", "--output", default="voxtral_clone_out.wav", help="Output WAV path")

    p.add_argument(
        "--ref-audio",
        help="Reference clip: https URL or path to local wav/mp3/etc. (embedded as data URL)",
    )
    p.add_argument(
        "--ref-text",
        help="Transcript of the reference audio (recommended for in-context cloning when supported)",
    )
    p.add_argument(
        "--language",
        default="English",
        help="Language hint, e.g. English (default). Omit with --no-language.",
    )
    p.add_argument(
        "--no-language",
        action="store_true",
        help="Do not send language (server uses Auto)",
    )

    u = p.add_argument_group("upload-based clone")
    u.add_argument("--upload", type=Path, help="Local audio file to voice.upload before TTS")
    u.add_argument("--voice-name", help="Name registered on the server (used as session voice)")
    u.add_argument("--consent", help="Consent / recording id required by the API")

    args = p.parse_args()
    lang = None if args.no_language else args.language

    asyncio.run(
        run_clone(
            args.url,
            args.text,
            args.output,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            language=lang,
            upload_path=args.upload,
            upload_name=args.voice_name,
            consent=args.consent,
        )
    )


if __name__ == "__main__":
    main()
