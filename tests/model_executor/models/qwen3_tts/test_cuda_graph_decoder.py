# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for CUDA Graph decoder wrapper numerical equivalence.

Verifies that CUDA Graph-accelerated decoding produces results equivalent
to eager mode, with special attention to padding cases where zero-padding
may introduce small numerical differences due to attention and convolution.

Architecture note: the real Qwen3TTSTokenizerV2Decoder uses causal
convolutions, so zero-padding on the right has minimal impact (~2e-3).
The synthetic decoder here uses standard (non-causal) Conv1d for a
worst-case test of the wrapper mechanism.
"""

import importlib.util
import os

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

DEVICE = torch.device("cuda:0")
NUM_QUANTIZERS = 8
TOTAL_UPSAMPLE = 4

# Load CUDAGraphDecoderWrapper: try package import first, fall back to direct file load
try:
    from vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper import CUDAGraphDecoderWrapper
except Exception:
    _WRAPPER_PATH = os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
        "vllm_omni",
        "model_executor",
        "models",
        "qwen3_tts",
        "cuda_graph_decoder_wrapper.py",
    )
    _spec = importlib.util.spec_from_file_location("cuda_graph_decoder_wrapper", os.path.abspath(_WRAPPER_PATH))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    CUDAGraphDecoderWrapper = _mod.CUDAGraphDecoderWrapper


class SyntheticDecoder(nn.Module):
    """A small decoder mimicking Qwen3TTSTokenizerV2Decoder's interface.

    Uses Conv1d layers so that zero-padding can affect neighboring positions
    via the receptive field, providing a worst-case test for padding effects.
    """

    def __init__(self, num_quantizers=NUM_QUANTIZERS, total_upsample=TOTAL_UPSAMPLE):
        super().__init__()
        hidden = 32
        self.total_upsample = total_upsample
        self.embed = nn.Conv1d(num_quantizers, hidden, kernel_size=3, padding=1)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.upsample = nn.ConvTranspose1d(hidden, hidden, kernel_size=total_upsample, stride=total_upsample)
        self.out = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, codes):
        x = codes.float()
        x = torch.relu(self.embed(x))
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.upsample(x)
        return self.out(x).clamp(min=-1, max=1)


@pytest.fixture(scope="module")
def decoder():
    """Create a synthetic decoder on CUDA with fixed weights."""
    torch.manual_seed(42)
    return SyntheticDecoder().to(DEVICE).eval()


@pytest.fixture(scope="module")
def wrapper(decoder):
    """Create a warmed-up CUDAGraphDecoderWrapper."""
    w = CUDAGraphDecoderWrapper(
        decoder=decoder,
        capture_sizes=[25, 50, 100],
        num_quantizers=NUM_QUANTIZERS,
        enabled=True,
    )
    w.warmup(DEVICE)
    return w


def _random_codes(seq_len, device=DEVICE):
    return torch.randint(0, 100, (1, NUM_QUANTIZERS, seq_len), dtype=torch.long, device=device)


# ──────────────────────────────────────────────────────────────────
# 1. Exact-size inputs (no padding needed) → bit-identical
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seq_len", [25, 50, 100])
def test_exact_size_numerical_equivalence(decoder, wrapper, seq_len):
    """When input exactly matches a capture size, output must be bit-identical."""
    codes = _random_codes(seq_len)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


# ──────────────────────────────────────────────────────────────────
# 2. Padded inputs (zero-padding to nearest capture size)
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seq_len", [10, 30, 47, 73, 99])
def test_padded_output_shape_and_length(decoder, wrapper, seq_len):
    """Padded decode must return output trimmed to actual input length."""
    codes = _random_codes(seq_len)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)
    expected_len = seq_len * TOTAL_UPSAMPLE
    assert graph_out.shape == eager_out.shape
    assert graph_out.shape[-1] == expected_len


@pytest.mark.parametrize("seq_len", [10, 30, 47, 73, 99])
def test_padded_interior_positions_close(decoder, wrapper, seq_len):
    """Interior positions (away from padding boundary) should be very close.

    The conv receptive field is at most 5 (kernel_size=5), so positions
    more than 2 timesteps from the end (times the upsample factor) should
    be nearly identical between eager and graph modes.
    """
    codes = _random_codes(seq_len)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)

    # Exclude the last (receptive_field * upsample) positions from strict check
    boundary = 3 * TOTAL_UPSAMPLE  # conservative: 3 positions * 4x upsample
    if eager_out.shape[-1] > boundary:
        interior_eager = eager_out[..., :(-boundary)]
        interior_graph = graph_out[..., :(-boundary)]
        torch.testing.assert_close(interior_graph, interior_eager, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seq_len", [10, 30, 47, 73, 99])
def test_padded_output_bounded(decoder, wrapper, seq_len):
    """Padded output values must remain in [-1, 1] and max diff should be bounded."""
    codes = _random_codes(seq_len)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)

    assert graph_out.min() >= -1.0 and graph_out.max() <= 1.0
    max_diff = (graph_out - eager_out).abs().max().item()
    # With non-causal conv, boundary diffs can be large (~0.5).
    # The real causal decoder shows ~2e-3.
    assert max_diff < 1.0, f"Max diff {max_diff} exceeds bound"


# ──────────────────────────────────────────────────────────────────
# 3. Fallback to eager (size exceeds all capture sizes) → bit-identical
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seq_len", [101, 150, 200])
def test_fallback_eager_exact_match(decoder, wrapper, seq_len):
    """Input larger than all capture sizes falls back to eager -> bit-identical."""
    codes = _random_codes(seq_len)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


# ──────────────────────────────────────────────────────────────────
# 4. Chunked decode equivalence
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("total_len", [60, 100, 150, 250])
def test_chunked_decode_shape_match(decoder, wrapper, total_len):
    """Chunked decode output shape must match between eager and graph modes."""
    codes = _random_codes(total_len)
    chunk_size, ctx = 50, 10

    with torch.no_grad():
        eager_out = _eager_chunked(decoder, codes, chunk_size, ctx)
        graph_out = wrapper.chunked_decode_with_cudagraph(codes, chunk_size=chunk_size, left_context_size=ctx)

    assert eager_out.shape == graph_out.shape


@pytest.mark.parametrize("total_len", [50, 100])
def test_chunked_decode_exact_size_equivalence(decoder, wrapper, total_len):
    """Chunked decode with chunks matching capture sizes should be bit-identical."""
    codes = _random_codes(total_len)
    # chunk_size=50 matches a capture size exactly, no context overlap
    chunk_size, ctx = 50, 0

    with torch.no_grad():
        eager_out = _eager_chunked(decoder, codes, chunk_size, ctx)
        graph_out = wrapper.chunked_decode_with_cudagraph(codes, chunk_size=chunk_size, left_context_size=ctx)

    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


def _eager_chunked(decoder, codes, chunk_size, left_context_size):
    """Eager chunked decode matching the real decoder's chunked_decode logic."""
    wavs = []
    start = 0
    total_len = codes.shape[-1]
    while start < total_len:
        end = min(start + chunk_size, total_len)
        ctx = left_context_size if start - left_context_size > 0 else start
        chunk = codes[..., start - ctx : end]
        wav = decoder(chunk)
        wavs.append(wav[..., ctx * decoder.total_upsample :])
        start = end
    return torch.cat(wavs, dim=-1)


# ──────────────────────────────────────────────────────────────────
# 5. Edge cases and control tests
# ──────────────────────────────────────────────────────────────────


def test_single_frame(decoder, wrapper):
    """Single-frame input (seq_len=1) should work with padding."""
    codes = _random_codes(1)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)
    assert graph_out.shape == eager_out.shape
    assert graph_out.shape[-1] == TOTAL_UPSAMPLE


def test_disabled_wrapper_matches_eager(decoder, wrapper):
    """Disabled wrapper should produce bit-identical output to eager."""
    codes = _random_codes(30)
    wrapper.enabled = False
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)
    wrapper.enabled = True
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


def test_batch_size_gt1_falls_back_on_decode(decoder, wrapper):
    """decode() is the single-request API and must fall back to eager on bs>1."""
    codes = torch.randint(0, 100, (2, NUM_QUANTIZERS, 25), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


# ──────────────────────────────────────────────────────────────────
# 5a. Batched decode (new)
# ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def batched_wrapper(decoder):
    """A wrapper that captures multi-bs graphs in addition to bs=1."""
    from vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper import (
        CUDAGraphDecoderWrapper as _W,
    )

    w = _W(
        decoder=decoder,
        capture_sizes=[25, 50, 100],
        batch_capture_sizes=[1, 2, 4],
        num_quantizers=NUM_QUANTIZERS,
        enabled=True,
    )
    w.warmup(DEVICE)
    return w


@pytest.mark.parametrize("batch,seq_len", [(1, 25), (2, 50), (4, 100)])
def test_batched_decode_exact_bucket_bit_identical(decoder, batched_wrapper, batch, seq_len):
    """When (bs, seq_len) matches a captured bucket exactly, output is bit-identical."""
    codes = torch.randint(0, 100, (batch, NUM_QUANTIZERS, seq_len), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = batched_wrapper.batched_decode(codes)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


@pytest.mark.parametrize("batch,bs_bucket,seq_len", [(3, 4, 100), (5, 8, 50)])
def test_batched_decode_padded_bs_numerically_close(decoder, batched_wrapper, batch, bs_bucket, seq_len):
    """Batch-padded replay may differ from eager by small FP epsilon due to cuDNN
    algorithm selection varying with batch size. It must remain numerically close
    and shape-compatible for the requested sub-slice.
    """
    if bs_bucket not in batched_wrapper.batch_capture_sizes and bs_bucket != 8:
        pytest.skip("bucket not captured in this fixture")
    codes = torch.randint(0, 100, (batch, NUM_QUANTIZERS, seq_len), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = batched_wrapper.batched_decode(codes)
    assert graph_out.shape == eager_out.shape
    torch.testing.assert_close(graph_out, eager_out, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_batched_decode_padded_bs_and_seq(decoder, batched_wrapper, batch):
    """Batch and seq padding should return a correctly shaped, bounded output."""
    seq_len = 47  # not exact size → pads to 50
    codes = torch.randint(0, 100, (batch, NUM_QUANTIZERS, seq_len), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = batched_wrapper.batched_decode(codes)
    assert graph_out.shape == eager_out.shape
    # Padded output must still be clamped to [-1, 1].
    assert graph_out.min() >= -1.0 and graph_out.max() <= 1.0


def test_batched_decode_bs_no_graph_falls_back_to_bs1(decoder, batched_wrapper):
    """A batch size exceeding all multi-bs buckets falls back to eager
    (padded_bs is None path)."""
    codes = torch.randint(0, 100, (5, NUM_QUANTIZERS, 25), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = batched_wrapper.batched_decode(codes)
    assert graph_out.shape == eager_out.shape


def test_batched_decode_missing_graph_below_threshold_uses_eager(decoder, batched_wrapper):
    """When (padded_bs, padded_size) has no captured graph and bs is at or
    below ``eager_fallback_max_bs``, we take a single batched eager call.

    This exercises the "fast" fallback branch that avoids the per-request
    replay throughput trap at small/mid batch sizes.
    """
    # (bs=2, seq=25) is NOT in the captured grid (preferred_seq=[100, 50])
    # but padded_bs=2 is in batch_capture_sizes=[1,2,4].
    assert (2, 25) not in batched_wrapper.graphs
    assert 2 in batched_wrapper.batch_capture_sizes
    codes = torch.randint(0, 100, (2, NUM_QUANTIZERS, 25), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = batched_wrapper.batched_decode(codes)
    assert graph_out.shape == eager_out.shape
    # Single batched eager forward must be bit-identical to a direct decoder call.
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


def test_batched_decode_above_threshold_uses_per_request_loop(decoder, batched_wrapper):
    """Above ``eager_fallback_max_bs``, fallback uses the memory-safe
    per-request bs=1 replay loop instead of a batched eager forward."""
    # Lower the cap to force the per-request path for bs=2.
    original_cap = batched_wrapper.eager_fallback_max_bs
    batched_wrapper.eager_fallback_max_bs = 1
    try:
        # (bs=2, seq=25): no multi-bs graph, bs=1 graph at seq=25 exists.
        assert (2, 25) not in batched_wrapper.graphs
        assert (1, 25) in batched_wrapper.graphs
        codes = torch.randint(0, 100, (2, NUM_QUANTIZERS, 25), dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            eager_out = decoder(codes)
            graph_out = batched_wrapper.batched_decode(codes)
        assert graph_out.shape == eager_out.shape
        # Per-request loop concatenation of bs=1 graph replays should match eager
        # at F=25 exactly (both use captured static tensors sized for seq=25).
        torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)
    finally:
        batched_wrapper.eager_fallback_max_bs = original_cap


def test_batched_decode_oversized_seq_falls_back_to_eager(decoder, batched_wrapper):
    """When seq_len exceeds all capture sizes, batched_decode falls back to eager."""
    codes = torch.randint(0, 100, (2, NUM_QUANTIZERS, 150), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = batched_wrapper.batched_decode(codes)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


def test_grid_selects_streaming_sizes_for_multi_bs():
    """In streaming mode, multi-bs graphs must target (chunk+left_context)
    and chunk, not the two largest seq sizes. This regression guard catches
    a past bug where preferred_seq defaulted to the top of the whole list
    and missed the streaming hot buckets entirely.
    """
    from vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper import (
        CUDAGraphDecoderWrapper as _W,
    )

    # seq_sizes modeled on compute_capture_sizes(chunk=25, left=72)
    seq_sizes = [2, 4, 8, 16, 25, 32, 64, 97, 128, 256, 325]
    bs_sizes = [1, 2, 4, 8, 16]

    w = _W(decoder=torch.nn.Identity(), capture_sizes=seq_sizes, batch_capture_sizes=bs_sizes)
    grid = w._select_batched_capture_grid(
        seq_sizes,
        bs_sizes,
        codec_chunk_frames=25,
        codec_left_context_frames=72,
    )

    multi_bs_pairs = {(bs, s) for bs, s in grid if bs > 1}
    # Steady-state hot streaming buckets must appear at every bs>1.
    for bs in [2, 4, 8, 16]:
        assert (bs, 97) in multi_bs_pairs, f"missing ({bs}, 97)"
        assert (bs, 25) in multi_bs_pairs, f"missing ({bs}, 25)"
    # IC-phase power-of-2 buckets STRICTLY BELOW codec_chunk_frames must also
    # appear at every bs>1: at high concurrency the IC burst would otherwise
    # fall into the per-request bs=1 replay loop at bs=16..32 for every
    # initial chunk, which is the primary stage-1 throughput ceiling.
    for bs in [2, 4, 8, 16]:
        for s in [2, 4, 8, 16]:
            assert (bs, s) in multi_bs_pairs, f"missing IC bucket ({bs}, {s})"
    # IC buckets AT OR ABOVE codec_chunk_frames (e.g. 32, 64 when chunk=25)
    # must NOT be captured at bs>1: they're not actually hit during IC (chunk
    # never exceeds codec_chunk_frames), and their activations blow up CUDA
    # graph private-pool memory at high bs (observed OOM during warmup at
    # bs=32, s=32 without this cap).
    for bs in [2, 4, 8, 16]:
        assert (bs, 32) not in multi_bs_pairs, f"unexpected IC bucket ({bs}, 32)"
        assert (bs, 64) not in multi_bs_pairs, f"unexpected IC bucket ({bs}, 64)"
    # And seq sizes at or above the steady-state primary (other than 97 and 25)
    # must NOT be captured at bs>1 — they're not part of the streaming hot path.
    assert (16, 325) not in multi_bs_pairs
    assert (16, 256) not in multi_bs_pairs
    assert (16, 128) not in multi_bs_pairs


def test_grid_falls_back_to_largest_when_not_streaming():
    """In non-streaming mode (no chunk/left_context), use the two largest
    seq sizes as a reasonable default."""
    from vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper import (
        CUDAGraphDecoderWrapper as _W,
    )

    seq_sizes = [8, 16, 64, 128, 256]
    bs_sizes = [1, 2, 4]
    w = _W(decoder=torch.nn.Identity(), capture_sizes=seq_sizes, batch_capture_sizes=bs_sizes)
    grid = w._select_batched_capture_grid(seq_sizes, bs_sizes)
    multi_bs_pairs = {(bs, s) for bs, s in grid if bs > 1}
    for bs in [2, 4]:
        assert (bs, 256) in multi_bs_pairs
        assert (bs, 128) in multi_bs_pairs


def test_deterministic_across_calls(decoder, wrapper):
    """Same input should produce identical CUDA graph output across calls."""
    codes = _random_codes(30)
    with torch.no_grad():
        out1 = wrapper.decode(codes)
        out2 = wrapper.decode(codes)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


# ──────────────────────────────────────────────────────────────────
# 6. compute_capture_sizes
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs,expected_in,not_expected",
    [
        ({}, [2, 4, 8, 16, 32, 64, 128, 256, 325], [512]),
        (
            {"codec_chunk_frames": 33, "codec_left_context_frames": 25},
            [2, 4, 8, 16, 32, 33, 58, 64, 128, 256, 325],
            [512],
        ),
        (
            {"codec_chunk_frames": 25, "codec_left_context_frames": 25},
            [2, 4, 8, 16, 25, 32, 50, 64, 128, 256, 325],
            [512],
        ),
    ],
    ids=["default", "streaming_c33", "streaming_c25"],
)
def test_compute_capture_sizes(kwargs, expected_in, not_expected):
    """compute_capture_sizes produces expected sizes capped by max useful size."""
    sizes = CUDAGraphDecoderWrapper.compute_capture_sizes(**kwargs)
    for val in expected_in:
        assert val in sizes, f"{val} not in {sizes}"
    for val in not_expected:
        assert val not in sizes, f"{val} should not be in {sizes}"


# ──────────────────────────────────────────────────────────────────
# 7. SnakeBeta Triton kernel vs eager equivalence
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "batch,channels,seq_len",
    [(2, 64, 1000), (1, 32, 1), (1, 32, 7), (1, 32, 128), (1, 32, 1024), (1, 32, 4096)],
)
def test_snakebeta_triton_vs_eager(batch, channels, seq_len):
    """Fused Triton SnakeBeta kernel must match eager PyTorch output."""
    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        SnakeBeta,
    )

    if not SnakeBeta._init_triton():
        pytest.skip("Triton not available")

    torch.manual_seed(42)
    snake = SnakeBeta(in_features=channels).to(DEVICE).eval()
    x = torch.randn(batch, channels, seq_len, device=DEVICE)

    with torch.no_grad():
        eager_out = snake._eager_forward(x)
        triton_out = snake._triton_forward(x)

    torch.testing.assert_close(triton_out, eager_out, atol=1e-5, rtol=1e-5)
