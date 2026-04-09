"""Benchmark client for Qwen3-TTS via /v1/audio/speech endpoint.

Measures TTFP (Time-to-First-Packet), E2E latency, and RTF (Real-Time Factor)
across configurable concurrency levels. Saves results as JSON for plotting.

Usage:
    python bench_tts_serve.py \
        --host 127.0.0.1 --port 8000 \
        --num-prompts 50 \
        --max-concurrency 1 4 10 \
        --result-dir results/
"""

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

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


@dataclass
class RequestResult:
    success: bool = False
    ttfp: float = 0.0  # Time to first audio packet (seconds)
    e2e: float = 0.0  # End-to-end latency (seconds)
    audio_bytes: int = 0  # Total audio bytes received
    audio_duration: float = 0.0  # Audio duration in seconds (estimated from PCM)
    rtf: float = 0.0  # Real-time factor = e2e / audio_duration
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
    # TTFP stats (ms)
    mean_ttfp_ms: float = 0.0
    median_ttfp_ms: float = 0.0
    std_ttfp_ms: float = 0.0
    p90_ttfp_ms: float = 0.0
    p95_ttfp_ms: float = 0.0
    p99_ttfp_ms: float = 0.0
    # E2E stats (ms)
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    std_e2e_ms: float = 0.0
    p90_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    # RTF stats
    mean_rtf: float = 0.0
    median_rtf: float = 0.0
    std_rtf: float = 0.0
    p99_rtf: float = 0.0
    # Audio stats
    mean_audio_duration_s: float = 0.0
    total_audio_duration_s: float = 0.0
    audio_throughput: float = 0.0  # audio_duration / wall_time
    request_throughput: float = 0.0  # requests / second
    # Per-request details
    per_request: list = field(default_factory=list)


def pcm_bytes_to_duration(num_bytes: int, sample_rate: int = 24000, sample_width: int = 2) -> float:
    """Convert raw PCM byte count to duration in seconds."""
    num_samples = num_bytes / sample_width
    return num_samples / sample_rate


def create_payload(
    prompt: str,
    task_type: str = "CustomVoice",
    voice: str = "vivian",
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


async def send_tts_request(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt: str,
    task_type: str = "CustomVoice",
    voice: str = "vivian",
    language: str = "English",
    include_ref_audio: bool = False,
    pbar: tqdm | None = None,
) -> RequestResult:
    """Send a streaming TTS request and measure latency metrics."""
    payload = create_payload(prompt, task_type, voice, language, include_ref_audio=include_ref_audio)

    result = RequestResult(prompt=prompt)
    st = time.perf_counter()

    try:
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
                result.success = False
                return result

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
        result.success = False
        result.e2e = time.perf_counter() - st

    if pbar:
        pbar.update(1)
    return result


async def upload_voice_to_server(
    session: aiohttp.ClientSession,
    base_url: str,
    voice_name: str,
) -> bool:
    """Upload a voice to the server so benchmark requests can reference it by name.

    Downloads the ref_audio locally, then POSTs it to /v1/audio/voices.
    Returns True if the voice is ready to use.
    """
    voice_cfg = VOICES.get(voice_name)
    if not voice_cfg:
        return False

    voices_url = f"{base_url}/v1/audio/voices"

    # Check if voice already exists
    async with session.get(voices_url) as resp:
        if resp.status == 200:
            data = await resp.json()
            existing = {v.lower() for v in data.get("voices", [])}
            if voice_name.lower() in existing:
                print(f"  Voice '{voice_name}' already exists on server, skipping upload.")
                return True

    # Download the ref audio
    print(f"  Downloading ref audio for voice '{voice_name}'...")
    async with session.get(voice_cfg["ref_audio"]) as resp:
        if resp.status != 200:
            print(f"  Failed to download ref audio: HTTP {resp.status}")
            return False
        audio_bytes = await resp.read()

    # Upload to the server
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
        else:
            text = await resp.text()
            print(f"  Voice upload failed: HTTP {resp.status}: {text[:200]}")
            return False


async def run_benchmark(
    host: str,
    port: int,
    num_prompts: int,
    max_concurrency: int,
    num_warmups: int = 3,
    task_type: str = "CustomVoice",
    voice: str = "vivian",
    language: str = "English",
) -> BenchmarkResult:
    """Run benchmark at a given concurrency level."""
    base_url = f"http://{host}:{port}"
    api_url = f"{base_url}/v1/audio/speech"

    connector = aiohttp.TCPConnector(
        limit=max_concurrency,
        limit_per_host=max_concurrency,
        keepalive_timeout=60,
    )
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=600),
    )

    # Upload voice if using Base task with a known voice config
    if task_type == "Base" and voice in VOICES:
        uploaded = await upload_voice_to_server(session, base_url, voice)
        if not uploaded:
            print(f"  WARNING: Could not upload voice '{voice}', falling back to sending ref_audio per request.")

    # Warmup (send ref_audio on warmup to prime any remaining caches)
    if num_warmups > 0:
        print(f"  Warming up with {num_warmups} requests...")
        warmup_tasks = []
        for i in range(num_warmups):
            prompt = PROMPTS[i % len(PROMPTS)]
            warmup_tasks.append(
                send_tts_request(session, api_url, prompt, task_type, voice, language, include_ref_audio=True)
            )
        await asyncio.gather(*warmup_tasks)
        print("  Warmup done.")

    # Build request list
    request_prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_prompts)]

    # Run benchmark
    print(f"  Running {num_prompts} requests with concurrency={max_concurrency}...")
    semaphore = asyncio.Semaphore(max_concurrency)
    pbar = tqdm(total=num_prompts, desc=f"  concurrency={max_concurrency}")

    async def limited_request(prompt):
        async with semaphore:
            return await send_tts_request(session, api_url, prompt, task_type, voice, language, pbar=pbar)

    start_time = time.perf_counter()
    tasks = [asyncio.create_task(limited_request(p)) for p in request_prompts]
    results: list[RequestResult] = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time
    pbar.close()

    await session.close()

    # Compute stats
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    bench = BenchmarkResult(
        concurrency=max_concurrency,
        num_prompts=num_prompts,
        completed=len(successful),
        failed=len(failed),
        duration_s=duration,
    )

    if successful:
        ttfps = [r.ttfp * 1000 for r in successful]  # convert to ms
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
        bench.audio_throughput = bench.total_audio_duration_s / duration
        bench.request_throughput = len(successful) / duration

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

    # Print summary in standardized performance template
    W = 50
    print("")
    print(f"{'=' * W}")
    print(f"{'Serving Benchmark Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Successful requests:':<40}{bench.completed:<10}")
    print(f"{'Failed requests:':<40}{bench.failed:<10}")
    print(f"{'Maximum request concurrency:':<40}{max_concurrency:<10}")
    print(f"{'Benchmark duration (s):':<40}{duration:<10.2f}")
    print(f"{'Request throughput (req/s):':<40}{bench.request_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'End-to-end Latency':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean E2EL (ms):':<40}{bench.mean_e2e_ms:<10.2f}")
    print(f"{'Median E2EL (ms):':<40}{bench.median_e2e_ms:<10.2f}")
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
    print(f"{'P99 AUDIO_TTFP (ms):':<40}{bench.p99_ttfp_ms:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Real Time Factor':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_RTF:':<40}{bench.mean_rtf:<10.3f}")
    print(f"{'Median AUDIO_RTF:':<40}{bench.median_rtf:<10.3f}")
    print(f"{'P99 AUDIO_RTF:':<40}{bench.p99_rtf:<10.3f}")
    print(f"{'=' * W}")
    print("")

    if failed:
        for r in failed[:3]:
            print(f"  [ERROR] {r.error[:200]}")

    return bench


async def main(args):
    all_results = []

    for concurrency in args.max_concurrency:
        result = await run_benchmark(
            host=args.host,
            port=args.port,
            num_prompts=args.num_prompts,
            max_concurrency=concurrency,
            num_warmups=args.num_warmups,
            task_type=args.task_type,
            voice=args.voice,
            language=args.language,
        )
        result.config_name = args.config_name
        all_results.append(asdict(result))

    # Save results
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"bench_{args.config_name}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {result_file}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Benchmark Client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-prompts", type=int, default=50, help="Number of prompts per concurrency level")
    parser.add_argument(  # noqa: E501
        "--max-concurrency", type=int, nargs="+", default=[1, 4, 10], help="Concurrency levels to test"
    )
    parser.add_argument("--num-warmups", type=int, default=3)
    parser.add_argument("--task-type", type=str, default="Base", choices=["CustomVoice", "VoiceDesign", "Base"])
    parser.add_argument("--voice", type=str, default="clone_2", help="Voice name (must exist in VOICES dict for Base task)")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument(
        "--config-name", type=str, default="async_chunk", help="Label for this config (used in filenames)"
    )
    parser.add_argument("--result-dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
