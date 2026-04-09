"""Concurrency sweep benchmark for TTS streaming endpoints.

Runs the same workload at increasing concurrency levels, each for a fixed
duration with steady-state workers, and reports TTFP, E2E latency, RTF,
and audio throughput at every level.  Output is JSON-compatible with
plot_results.py from bench_tts_serve.

Examples:
    python concurrency_sweep.py \
        --host 127.0.0.1 --port 8000 \
        --levels 1,2,4,8,16 --duration 60 --warmup 10 \
        --task-type Base --voice clone_2

    python concurrency_sweep.py \
        --host 127.0.0.1 --port 8000 \
        --levels 1,4,16,64 --duration 30 --warmup 5 \
        --task-type Base --voice clone_2 --config-name latency
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np

PROMPTS = [
    "Hello, welcome to the voice synthesis benchmark test.",
    "She said she would be here by noon, but nobody showed up.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "I can't believe how beautiful the sunset looks from up here on the mountain.",
    "Please remember to bring your identification documents to the appointment tomorrow morning.",
    "Have you ever wondered what it would be like to travel through time and visit ancient civilizations?",
    "The restaurant on the corner serves the best pasta I have ever tasted in my entire life.",
    "After the meeting, we should discuss the quarterly results and plan for the next phase.",
    "Learning a new language takes patience, practice, and a genuine curiosity about other cultures.",
    "The train leaves at half past seven, so we need to arrive at the station before then.",
    "Could you please turn down the music a little bit, I'm trying to concentrate on my work.",
    "It was a dark and stormy night when the old lighthouse keeper heard a knock at the door.",
]

VOICES = {
    "clone_2": {
        "ref_audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav",
        "ref_text": "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
    },
}

INSTRUCT = "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."


# ── helpers ─────────────────────────────────────────────────────────────────

def pcm_bytes_to_duration(
    num_bytes: int, sample_rate: int = 24000, sample_width: int = 2,
) -> float:
    return (num_bytes / sample_width) / sample_rate


def create_payload(
    prompt: str,
    task_type: str = "Base",
    voice: str = "clone_2",
    language: str = "English",
    include_ref_audio: bool = False,
) -> dict:
    payload = {
        "input": prompt,
        "language": language,
        "stream": True,
        "response_format": "pcm",
        "task_type": task_type,
    }
    if task_type == "Base":
        if include_ref_audio and voice in VOICES:
            payload["ref_audio"] = VOICES[voice]["ref_audio"]
            payload["ref_text"] = VOICES[voice]["ref_text"]
        else:
            payload["voice"] = voice
    elif task_type == "CustomVoice":
        payload["voice"] = voice
    elif task_type == "VoiceDesign":
        payload["instructions"] = INSTRUCT
    return payload


def percentile(sorted_data: list[float], pct: float) -> float:
    if not sorted_data:
        return 0.0
    idx = (pct / 100) * (len(sorted_data) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


async def upload_voice_to_server(
    session: aiohttp.ClientSession,
    base_url: str,
    voice_name: str,
) -> bool:
    voice_cfg = VOICES.get(voice_name)
    if not voice_cfg:
        return False

    voices_url = f"{base_url}/v1/audio/voices"

    async with session.get(voices_url) as resp:
        if resp.status == 200:
            data = await resp.json()
            existing = {v.lower() for v in data.get("voices", [])}
            if voice_name.lower() in existing:
                print(f"  Voice '{voice_name}' already exists on server.")
                return True

    print(f"  Downloading ref audio for voice '{voice_name}'...")
    async with session.get(voice_cfg["ref_audio"]) as resp:
        if resp.status != 200:
            print(f"  Failed to download ref audio: HTTP {resp.status}")
            return False
        audio_bytes = await resp.read()

    print(f"  Uploading voice '{voice_name}' to server...")
    form = aiohttp.FormData()
    form.add_field("audio_sample", audio_bytes, filename=f"{voice_name}.wav", content_type="audio/wav")
    form.add_field("consent", "agreed")
    form.add_field("name", voice_name)
    if voice_cfg.get("ref_text"):
        form.add_field("ref_text", voice_cfg["ref_text"])

    async with session.post(voices_url, data=form) as resp:
        if resp.status == 200:
            print(f"  Voice '{voice_name}' uploaded successfully.")
            return True
        text = await resp.text()
        print(f"  Voice upload failed: HTTP {resp.status}: {text[:200]}")
        return False


# ── dataclasses (compatible with bench_tts_serve / plot_results.py) ─────────

@dataclass
class RequestResult:
    success: bool = False
    ttfp: float = 0.0
    e2e: float = 0.0
    audio_bytes: int = 0
    audio_duration: float = 0.0
    rtf: float = 0.0
    prompt: str = ""
    error: str = ""


@dataclass
class BenchmarkResult:
    config_name: str = ""
    concurrency: int = 0
    num_prompts: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    mean_ttfp_ms: float = 0.0
    median_ttfp_ms: float = 0.0
    std_ttfp_ms: float = 0.0
    p90_ttfp_ms: float = 0.0
    p95_ttfp_ms: float = 0.0
    p99_ttfp_ms: float = 0.0
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    std_e2e_ms: float = 0.0
    p90_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    mean_rtf: float = 0.0
    median_rtf: float = 0.0
    std_rtf: float = 0.0
    p99_rtf: float = 0.0
    mean_audio_duration_s: float = 0.0
    total_audio_duration_s: float = 0.0
    audio_throughput: float = 0.0
    request_throughput: float = 0.0
    per_request: list = field(default_factory=list)


# ── async worker ────────────────────────────────────────────────────────────

async def worker(
    session: aiohttp.ClientSession,
    api_url: str,
    task_type: str,
    voice: str,
    language: str,
    results: list[RequestResult],
    stop_event: asyncio.Event,
    prompt_idx: list[int],
):
    """Steady-state worker: sends streaming requests in a loop until stopped."""
    while not stop_event.is_set():
        idx = prompt_idx[0] % len(PROMPTS)
        prompt_idx[0] += 1
        prompt = PROMPTS[idx]
        payload = create_payload(prompt, task_type, voice, language)

        result = RequestResult(prompt=prompt)
        st = time.perf_counter()

        try:
            async with session.post(api_url, json=payload) as response:
                if response.status != 200:
                    result.error = f"HTTP {response.status}"
                    result.e2e = time.perf_counter() - st
                    results.append(result)
                    continue

                first_chunk = True
                total_bytes = 0

                async for chunk in response.content.iter_any():
                    if first_chunk and len(chunk) > 0:
                        result.ttfp = time.perf_counter() - st
                        first_chunk = False
                    total_bytes += len(chunk)

                result.e2e = time.perf_counter() - st
                result.audio_bytes = total_bytes
                result.audio_duration = pcm_bytes_to_duration(total_bytes)
                if result.audio_duration > 0:
                    result.rtf = result.e2e / result.audio_duration
                result.success = True

        except Exception as e:
            result.error = str(e)
            result.e2e = time.perf_counter() - st

        results.append(result)


# ── single-level run ────────────────────────────────────────────────────────

async def run_level(
    session: aiohttp.ClientSession,
    api_url: str,
    concurrency: int,
    duration: float,
    warmup: float,
    task_type: str,
    voice: str,
    language: str,
) -> tuple[list[RequestResult], float]:
    """Run steady-state workers at a given concurrency for *duration* seconds.

    Returns (measured_results, wall_time).
    """
    results: list[RequestResult] = []
    stop_event = asyncio.Event()
    prompt_counter: list[int] = [0]

    workers = [
        asyncio.create_task(
            worker(session, api_url, task_type, voice, language,
                   results, stop_event, prompt_counter)
        )
        for _ in range(concurrency)
    ]

    if warmup > 0:
        await asyncio.sleep(warmup)
    results.clear()

    bench_start = time.perf_counter()
    await asyncio.sleep(duration)
    stop_event.set()
    await asyncio.gather(*workers)
    measure_wall = time.perf_counter() - bench_start

    return results, measure_wall


def compute_stats(
    results: list[RequestResult],
    wall_time: float,
    concurrency: int,
    config_name: str,
) -> BenchmarkResult:
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    bench = BenchmarkResult(
        config_name=config_name,
        concurrency=concurrency,
        num_prompts=len(results),
        completed=len(successful),
        failed=len(failed),
        duration_s=wall_time,
    )

    if not successful:
        return bench

    ttfps = [r.ttfp * 1000 for r in successful]
    e2es = [r.e2e * 1000 for r in successful]
    rtfs = [r.rtf for r in successful]
    audio_durs = [r.audio_duration for r in successful]

    bench.mean_ttfp_ms = float(np.mean(ttfps))
    bench.median_ttfp_ms = float(np.median(ttfps))
    bench.std_ttfp_ms = float(np.std(ttfps))
    bench.p90_ttfp_ms = float(np.percentile(ttfps, 90))
    bench.p95_ttfp_ms = float(np.percentile(ttfps, 95))
    bench.p99_ttfp_ms = float(np.percentile(ttfps, 99))

    bench.mean_e2e_ms = float(np.mean(e2es))
    bench.median_e2e_ms = float(np.median(e2es))
    bench.std_e2e_ms = float(np.std(e2es))
    bench.p90_e2e_ms = float(np.percentile(e2es, 90))
    bench.p95_e2e_ms = float(np.percentile(e2es, 95))
    bench.p99_e2e_ms = float(np.percentile(e2es, 99))

    bench.mean_rtf = float(np.mean(rtfs))
    bench.median_rtf = float(np.median(rtfs))
    bench.std_rtf = float(np.std(rtfs))
    bench.p99_rtf = float(np.percentile(rtfs, 99))

    bench.mean_audio_duration_s = float(np.mean(audio_durs))
    bench.total_audio_duration_s = float(np.sum(audio_durs))
    bench.audio_throughput = bench.total_audio_duration_s / wall_time
    bench.request_throughput = len(successful) / wall_time

    bench.per_request = [
        {
            "ttfp_ms": r.ttfp * 1000,
            "e2e_ms": r.e2e * 1000,
            "rtf": r.rtf,
            "audio_duration_s": r.audio_duration,
            "prompt": r.prompt,
        }
        for r in successful
    ]

    return bench


# ── display ─────────────────────────────────────────────────────────────────

def print_level_stats(bench: BenchmarkResult) -> None:
    W = 50
    print(f"{'=' * W}")
    print(f"{'Serving Benchmark Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Successful requests:':<40}{bench.completed:<10}")
    print(f"{'Failed requests:':<40}{bench.failed:<10}")
    print(f"{'Maximum request concurrency:':<40}{bench.concurrency:<10}")
    print(f"{'Benchmark duration (s):':<40}{bench.duration_s:<10.2f}")
    print(f"{'Request throughput (req/s):':<40}{bench.request_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'End-to-end Latency':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean E2EL (ms):':<40}{bench.mean_e2e_ms:<10.2f}")
    print(f"{'Median E2EL (ms):':<40}{bench.median_e2e_ms:<10.2f}")
    print(f"{'P90 E2EL (ms):':<40}{bench.p90_e2e_ms:<10.2f}")
    print(f"{'P99 E2EL (ms):':<40}{bench.p99_e2e_ms:<10.2f}")
    print(f"{'=' * W}")
    print(f"{'Audio Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Total audio duration generated (s):':<40}{bench.total_audio_duration_s:<10.2f}")
    print(f"{'Audio throughput (audio duration/s):':<40}{bench.audio_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Time to First Packet':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_TTFP (ms):':<40}{bench.mean_ttfp_ms:<10.2f}")
    print(f"{'Median AUDIO_TTFP (ms):':<40}{bench.median_ttfp_ms:<10.2f}")
    print(f"{'P90 AUDIO_TTFP (ms):':<40}{bench.p90_ttfp_ms:<10.2f}")
    print(f"{'P99 AUDIO_TTFP (ms):':<40}{bench.p99_ttfp_ms:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Real Time Factor':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_RTF:':<40}{bench.mean_rtf:<10.3f}")
    print(f"{'Median AUDIO_RTF:':<40}{bench.median_rtf:<10.3f}")
    print(f"{'P99 AUDIO_RTF:':<40}{bench.p99_rtf:<10.3f}")
    print(f"{'=' * W}")
    print()


def print_summary_table(all_stats: list[BenchmarkResult]) -> None:
    print()
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    header = (
        f"{'Conc':>5}  {'Reqs':>5}  {'Err':>4}  {'req/s':>7}  {'aud/wall':>8}  "
        f"{'TTFP p50':>8}  {'TTFP p90':>8}  {'TTFP p99':>8}  "
        f"{'E2E p50':>8}  {'E2E p90':>8}  {'E2E p99':>8}  "
        f"{'RTF p50':>7}  {'RTF p99':>7}"
    )
    print(header)
    print("-" * 120)
    for s in all_stats:
        row = (
            f"{s.concurrency:>5}  {s.completed:>5}  {s.failed:>4}  "
            f"{s.request_throughput:>7.2f}  "
            f"{s.audio_throughput:>8.2f}  "
            f"{s.median_ttfp_ms:>8.1f}  {s.p90_ttfp_ms:>8.1f}  {s.p99_ttfp_ms:>8.1f}  "
            f"{s.median_e2e_ms:>8.1f}  {s.p90_e2e_ms:>8.1f}  {s.p99_e2e_ms:>8.1f}  "
            f"{s.median_rtf:>7.3f}  {s.p99_rtf:>7.3f}"
        )
        print(row)
    print("=" * 120)


# ── main ────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    base_url = f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/audio/speech"
    levels = sorted(int(x.strip()) for x in args.levels.split(","))

    print(f"Host:       {args.host}:{args.port}")
    print(f"Task type:  {args.task_type}")
    print(f"Voice:      {args.voice}")
    print(f"Levels:     {levels}")
    print(f"Duration:   {args.duration}s per level (+ {args.warmup}s warmup)")
    print()

    connector = aiohttp.TCPConnector(
        limit=max(levels) + 10,
        limit_per_host=max(levels) + 10,
        keepalive_timeout=60,
    )
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=600),
    )

    # Upload voice once before any benchmarking
    if args.task_type == "Base" and args.voice in VOICES:
        uploaded = await upload_voice_to_server(session, base_url, args.voice)
        if not uploaded:
            print("  WARNING: Could not upload voice, requests will include ref_audio.")

    all_stats: list[BenchmarkResult] = []

    for level in levels:
        print(f"\n{'=' * 60}")
        print(f"  concurrency={level}")
        print(f"{'=' * 60}")
        if args.warmup > 0:
            print(f"  Warming up for {args.warmup}s...")

        results, wall_time = await run_level(
            session=session,
            api_url=api_url,
            concurrency=level,
            duration=args.duration,
            warmup=args.warmup,
            task_type=args.task_type,
            voice=args.voice,
            language=args.language,
        )

        bench = compute_stats(results, wall_time, level, args.config_name)
        all_stats.append(bench)
        print_level_stats(bench)

    await session.close()

    print_summary_table(all_stats)

    # Save JSON (same format as bench_tts_serve for plot_results.py)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"sweep_{args.config_name}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump([asdict(s) for s in all_stats], f, indent=2)
    print(f"\nResults saved to {result_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TTS concurrency sweep benchmark (streaming, steady-state)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--levels", "-l", type=str, default="1,2,4,8,16",
                        help="Comma-separated concurrency levels (default: 1,2,4,8,16)")
    parser.add_argument("--duration", "-d", type=int, default=60,
                        help="Measurement duration per level in seconds (default: 60)")
    parser.add_argument("--warmup", "-w", type=int, default=10,
                        help="Warmup seconds at each level before measuring (default: 10)")
    parser.add_argument("--task-type", "-t", type=str, default="Base",
                        choices=["CustomVoice", "VoiceDesign", "Base"])
    parser.add_argument("--voice", type=str, default="clone_2",
                        help="Voice name (must exist in VOICES dict for Base task)")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--config-name", type=str, default="sweep",
                        help="Label for this config (used in filenames)")
    parser.add_argument("--result-dir", type=str, default="results",
                        help="Directory to save JSON results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
