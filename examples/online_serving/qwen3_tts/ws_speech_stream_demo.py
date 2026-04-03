#!/usr/bin/env python3
"""Demo client for the /v1/audio/speech/stream WebSocket endpoint.

Connects to the streaming TTS WebSocket, sends text (optionally word-by-word
to simulate real-time STT input), receives audio per sentence, and saves the
concatenated result. Supports both per-sentence WAV delivery and progressive
PCM chunk streaming.

Examples:
    # Basic usage — sends text, saves concatenated WAV
    python ws_speech_stream_demo.py \
        --text "Hello world. How are you today? I'm doing great."

    # Progressive PCM streaming (lower latency per sentence)
    python ws_speech_stream_demo.py \
        --text "Hello world. How are you today?" \
        --stream-audio

    # Simulate real-time STT: drip-feed words with 100ms delay
    python ws_speech_stream_demo.py \
        --text "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs." \
        --simulate-stt --stt-delay 0.08

    # VoiceDesign task (voice from a natural-language description)
    python ws_speech_stream_demo.py \
        --text "Today is absolutely wonderful." \
        --task-type VoiceDesign \
        --instructions "A cheerful young female voice with a warm tone"

    # Clause-level splitting for faster first audio
    python ws_speech_stream_demo.py \
        --text "Well, I think so, but let me check. Yes, that's correct." \
        --split-granularity clause --stream-audio

    # Voice cloning (Base task) with a reference audio URL
    python ws_speech_stream_demo.py \
        --text "Hello world. How are you?" \
        --task-type Base \
        --ref-audio "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav" \
        --ref-text "Okay. Yeah. I resent you. I love you."

Requirements:
    pip install websockets soundfile numpy
"""

import argparse
import asyncio
import io
import json
import struct
import time

import numpy as np

try:
    import websockets
except ImportError:
    raise SystemExit("Missing dependency: pip install websockets")

try:
    import soundfile as sf
except ImportError:
    sf = None

SAMPLE_RATE = 24000


async def run_session(
    url: str,
    text: str,
    config: dict,
    output: str,
    simulate_stt: bool = False,
    stt_delay: float = 0.1,
) -> None:
    """Open a WebSocket session, send text, receive audio, and save output."""

    all_pcm: list[np.ndarray] = []
    sentence_count = 0
    first_audio_time = None
    t0 = time.perf_counter()

    is_pcm = config.get("response_format", "wav") == "pcm" or config.get("stream_audio", False)

    async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
        # ── 1. Send session.config ────────────────────────────────
        config_msg = {"type": "session.config", **config}
        await ws.send(json.dumps(config_msg))
        print(f"[config] Connected to {url}")
        print(f"[config] voice={config.get('voice', 'default')}  "
              f"format={config.get('response_format', 'wav')}  "
              f"stream_audio={config.get('stream_audio', False)}  "
              f"split={config.get('split_granularity', 'sentence')}")

        # ── 2. Send text ──────────────────────────────────────────
        async def send_text():
            if simulate_stt:
                words = text.split(" ")
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    await ws.send(json.dumps({"type": "input.text", "text": chunk}))
                    await asyncio.sleep(stt_delay)
                print(f"[input]  Sent {len(words)} words (simulated STT, {stt_delay}s delay)")
            else:
                await ws.send(json.dumps({"type": "input.text", "text": text}))
                print(f"[input]  Sent {len(text)} chars")

            await ws.send(json.dumps({"type": "input.done"}))

        sender = asyncio.create_task(send_text())

        # ── 3. Receive audio ──────────────────────────────────────
        current_chunks: list[bytes] = []
        current_format = "wav"
        current_sentence = ""

        try:
            while True:
                msg = await ws.recv()

                if isinstance(msg, bytes):
                    if first_audio_time is None:
                        first_audio_time = time.perf_counter()
                    current_chunks.append(msg)
                    continue

                data = json.loads(msg)
                mtype = data.get("type")

                if mtype == "audio.start":
                    current_chunks = []
                    current_format = data.get("format", "wav")
                    current_sentence = data.get("sentence_text", "")
                    idx = data["sentence_index"]
                    print(f"[sent {idx}] \"{current_sentence}\"")

                elif mtype == "audio.done":
                    idx = data["sentence_index"]
                    total_bytes = data.get("total_bytes", sum(len(c) for c in current_chunks))
                    errored = data.get("error", False)

                    raw = b"".join(current_chunks)
                    if errored:
                        print(f"[sent {idx}] ERROR generating audio")
                    elif is_pcm:
                        usable = len(raw) - (len(raw) % 2)
                        if usable > 0:
                            pcm = np.frombuffer(raw[:usable], dtype=np.int16)
                            all_pcm.append(pcm)
                        dur = usable / 2 / SAMPLE_RATE
                        print(f"[sent {idx}] {total_bytes:,} bytes  ({dur:.2f}s audio)")
                    else:
                        try:
                            pcm_arr, sr = sf.read(io.BytesIO(raw))
                            if pcm_arr.ndim > 1:
                                pcm_arr = pcm_arr[:, 0]
                            pcm_int16 = (np.clip(pcm_arr, -1, 1) * 32767).astype(np.int16)
                            all_pcm.append(pcm_int16)
                            dur = len(pcm_int16) / sr
                            print(f"[sent {idx}] {total_bytes:,} bytes  ({dur:.2f}s audio, {current_format})")
                        except Exception as e:
                            print(f"[sent {idx}] {total_bytes:,} bytes  (decode error: {e})")

                    sentence_count += 1
                    current_chunks = []

                elif mtype == "session.done":
                    print(f"\n[done]   {data['total_sentences']} sentence(s)")
                    break

                elif mtype == "error":
                    print(f"[error]  {data['message']}")

        finally:
            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass

    # ── 4. Save concatenated output ───────────────────────────────
    elapsed = time.perf_counter() - t0
    ttfa = (first_audio_time - t0) if first_audio_time else None

    if not all_pcm:
        print("\nNo audio received.")
        return

    combined = np.concatenate(all_pcm)
    audio_duration = len(combined) / SAMPLE_RATE

    if sf is not None:
        if not output.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
            output += ".wav"
        sf.write(output, combined.astype(np.float32) / 32767.0, SAMPLE_RATE)
    else:
        if not output.lower().endswith(".wav"):
            output += ".wav"
        _write_wav(output, combined, SAMPLE_RATE)

    print(f"\n{'─' * 50}")
    print(f"  Output:     {output}")
    print(f"  Duration:   {audio_duration:.2f}s  ({sentence_count} sentences)")
    print(f"  Wall time:  {elapsed:.2f}s")
    if ttfa is not None:
        print(f"  TTFA:       {ttfa * 1000:.0f}ms")
    if audio_duration > 0:
        rtf = elapsed / audio_duration
        print(f"  RTF:        {rtf:.3f}x  ({1/rtf:.1f}x realtime)")
    print(f"{'─' * 50}")


def _write_wav(path: str, pcm: np.ndarray, sample_rate: int) -> None:
    """Write a 16-bit mono WAV file without soundfile."""
    data = pcm.astype(np.int16).tobytes()
    n_channels = 1
    sample_width = 2
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                            sample_rate * n_channels * sample_width, n_channels * sample_width, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


def main():
    p = argparse.ArgumentParser(
        description="Demo client for /v1/audio/speech/stream WebSocket endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url", default="ws://localhost:8091/v1/audio/speech/stream",
                    help="WebSocket URL (default: ws://localhost:8091/v1/audio/speech/stream)")
    p.add_argument("--text", required=True, help="Text to synthesize")
    p.add_argument("--output", "-o", default="ws_tts_output.wav", help="Output file (default: ws_tts_output.wav)")

    g = p.add_argument_group("session config")
    g.add_argument("--voice", default="Vivian", help="Speaker name (default: Vivian)")
    g.add_argument("--task-type", default="CustomVoice",
                    choices=["CustomVoice", "VoiceDesign", "Base"])
    g.add_argument("--language", default=None, help="Language hint (Auto, English, Chinese, ...)")
    g.add_argument("--instructions", default=None,
                    help="Style/emotion instructions (required for VoiceDesign)")
    g.add_argument("--response-format", default="wav",
                    choices=["wav", "pcm", "flac", "mp3", "aac", "opus"])
    g.add_argument("--speed", type=float, default=1.0, help="Speed 0.25-4.0 (default: 1.0)")
    g.add_argument("--stream-audio", action="store_true",
                    help="Progressive PCM streaming per sentence (forces format=pcm)")
    g.add_argument("--split-granularity", default="sentence",
                    choices=["sentence", "clause"],
                    help="Text splitting granularity (default: sentence)")

    g2 = p.add_argument_group("voice cloning (Base task)")
    g2.add_argument("--ref-audio", default=None, help="Reference audio URL or local path")
    g2.add_argument("--ref-text", default=None, help="Reference audio transcript")
    g2.add_argument("--x-vector-only", action="store_true", help="Use speaker embedding only")

    g3 = p.add_argument_group("STT simulation")
    g3.add_argument("--simulate-stt", action="store_true",
                     help="Send text word-by-word to simulate real-time input")
    g3.add_argument("--stt-delay", type=float, default=0.1,
                     help="Delay between words in seconds (default: 0.1)")

    args = p.parse_args()

    config: dict = {
        "voice": args.voice,
        "task_type": args.task_type,
        "response_format": args.response_format,
        "speed": args.speed,
        "split_granularity": args.split_granularity,
    }
    if args.stream_audio:
        config["stream_audio"] = True
        config["response_format"] = "pcm"
    if args.language:
        config["language"] = args.language
    if args.instructions:
        config["instructions"] = args.instructions
    if args.ref_audio:
        config["ref_audio"] = args.ref_audio
    if args.ref_text:
        config["ref_text"] = args.ref_text
    if args.x_vector_only:
        config["x_vector_only_mode"] = True

    asyncio.run(run_session(
        url=args.url,
        text=args.text,
        config=config,
        output=args.output,
        simulate_stt=args.simulate_stt,
        stt_delay=args.stt_delay,
    ))


if __name__ == "__main__":
    main()
