# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

This module provides CUDA Graph acceleration for the speech tokenizer decoder,
reducing kernel launch overhead during inference.
"""

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class CUDAGraphDecoderWrapper:
    """
    CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

    This wrapper captures the decoder forward pass for fixed input sizes
    and replays them during inference to reduce kernel launch overhead.

    The wrapper supports both single-request and batched replay. The graph
    pool is keyed by ``(batch_size, seq_len)``; callers that want batched
    replay must pass ``batch_capture_sizes`` at construction time. By
    default only ``batch_capture_sizes=[1]`` is captured, preserving the
    historical bs=1 behavior.

    Usage:
        wrapper = CUDAGraphDecoderWrapper(
            decoder,
            capture_sizes=[25, 50, 100, 200, 300],
            batch_capture_sizes=[1, 2, 4, 8, 16],
        )
        wrapper.warmup(device)

        # Single-request replay (back-compat):
        output = wrapper.decode(codes)

        # Batched replay: codes is [B, Q, F]
        output = wrapper.batched_decode(codes)
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        capture_sizes: list[int] | None = None,
        batch_capture_sizes: list[int] | None = None,
        num_quantizers: int = 8,
        enabled: bool = True,
        eager_fallback_max_bs: int = 8,
    ):
        self.decoder = decoder
        self._explicit_sizes = capture_sizes is not None
        self.capture_sizes = sorted(capture_sizes) if capture_sizes else []
        # bs=1 is always captured; additional batch-size buckets are opt-in.
        user_bs = sorted(set(int(b) for b in (batch_capture_sizes or []) if int(b) >= 1))
        if 1 not in user_bs:
            user_bs = [1] + user_bs
        self.batch_capture_sizes = user_bs
        self.num_quantizers = num_quantizers
        self.enabled = enabled
        # Upper bound on batch size for the single-batched eager fallback path
        # when no multi-bs graph is available. Below/at this threshold we use
        # ``decoder(codes)`` directly (fast, one launch); above it we fall
        # back to per-request bs=1 graph replay (slow but memory-safe).
        #
        # The cap exists because stage-0 and stage-1 co-reside on the same
        # GPU in the Qwen3-TTS pipeline; a large-bs eager forward in stage-1
        # can fragment the CUDA allocator and destabilize stage-0 bursts.
        # bs<=8 was observed to be safe and unlocks the conc=8/16 win.
        self.eager_fallback_max_bs = max(1, int(eager_fallback_max_bs))

        # Keyed by (bs, seq_len)
        self.graphs: dict[tuple[int, int], CUDAGraph] = {}
        self.static_inputs: dict[tuple[int, int], torch.Tensor] = {}
        self.static_outputs: dict[tuple[int, int], torch.Tensor] = {}

        self._warmed_up = False
        self._device = None

        # --- diagnostic counters for batched_decode path selection ---
        # Enables triage of throughput cliffs by revealing when we're on the
        # slow "per-request bs=1 loop" fallback vs. the fast captured-graph
        # replay path. Counters are printed every ``_stats_log_interval``
        # batched_decode calls and reset after each print, so the snapshot
        # reflects recent traffic only.
        self._stats: dict[str, int] = {
            "calls": 0,
            "graph_hit": 0,
            "bs1_graph_hit_via_batched": 0,
            "eager_small_bs": 0,
            "per_request_loop": 0,
            "eager_full_fallback": 0,
        }
        self._miss_keys: dict[tuple[int, int], int] = {}
        self._hit_keys: dict[tuple[int, int], int] = {}
        self._stats_log_interval = 500

    @staticmethod
    def compute_capture_sizes(
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
        decode_chunk_size: int = 300,
        decode_left_context: int = 25,
    ) -> list[int]:
        """Compute capture sizes from chunking config for high graph hit rate."""
        sizes: set[int] = set()

        # Streaming exact hits
        if codec_chunk_frames > 0:
            sizes.add(codec_chunk_frames)
            if codec_left_context_frames > 0:
                sizes.add(codec_chunk_frames + codec_left_context_frames)

        # Non-streaming chunked decode: full chunk + last-chunk buckets
        non_stream_max = decode_chunk_size + decode_left_context
        sizes.add(non_stream_max)

        # Power-of-2 buckets covering both streaming IC sizes and non-streaming last-chunk sizes
        for p2 in [2, 4, 8, 16, 32, 64, 128, 256]:
            if p2 <= non_stream_max:
                sizes.add(p2)

        return sorted(sizes)

    def _get_padded_size(self, actual_size: int) -> int | None:
        for size in self.capture_sizes:
            if actual_size <= size:
                return size
        return None

    def _get_padded_batch_size(self, actual_batch: int) -> int | None:
        for bs in self.batch_capture_sizes:
            if actual_batch <= bs:
                return bs
        return None

    def _select_batched_capture_grid(
        self,
        seq_sizes: list[int],
        bs_sizes: list[int],
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
    ) -> list[tuple[int, int]]:
        """Choose the (bs, seq_len) pairs to capture.

        We always capture the bs=1 row for backward compatibility. For bs>1
        we capture the "hot" streaming seq sizes so total graph memory and
        warmup time stay bounded:

        * In streaming mode (``codec_chunk_frames>0``), the hot sizes are:
            - ``codec_chunk_frames + codec_left_context_frames`` (steady-state)
            - ``codec_chunk_frames`` (saturated-context after we stop prepending
              ref_code / left_context)
            - the IC-phase power-of-2 buckets below the steady-state primary
              (e.g. {2, 4, 8, 16, 32} for chunk=25). These are dominant at
              high concurrency where many requests burst through IC together:
              falling back to a per-request bs=1 replay loop at bs=16..32 for
              every IC chunk is the primary stage-1 throughput ceiling.
          These are the only seq lengths that actually flow through Stage-1
          in streaming deployments, so capturing them at every bs bucket
          maximizes graph hit-rate.
        * Otherwise (non-streaming), we fall back to the two largest captured
          seq sizes for multi-bs — a reasonable proxy for full-utterance
          decode buckets.
        """
        grid: list[tuple[int, int]] = []
        for s in seq_sizes:
            grid.append((1, s))
        if len(bs_sizes) <= 1:
            return grid

        preferred_seq: list[int] = []
        # IC-transition buckets are the power-of-2 sizes between
        # codec_chunk_frames and primary. These are hit by every request
        # during left_context ramp-up (e.g. chunk 2 of every request at
        # chunk=25/left=72 has F=50 → pads to 64). They have larger
        # activation footprints and blew up CUDA graph private pools at
        # high bs during warmup (observed OOM at bs=32 size=32/64), so we
        # capture them only at bs <= ``ic_transition_cap_bs``.
        ic_transition: list[int] = []
        ic_transition_cap_bs = 16
        if codec_chunk_frames > 0:
            primary = codec_chunk_frames + max(0, codec_left_context_frames)
            if primary in seq_sizes:
                preferred_seq.append(primary)
            if codec_chunk_frames in seq_sizes and codec_chunk_frames not in preferred_seq:
                preferred_seq.append(codec_chunk_frames)
            # IC-phase power-of-2 buckets BELOW codec_chunk_frames (cheap
            # activations, safe at every bs). These are dominant at high
            # concurrency when requests burst through IC together.
            for s in seq_sizes:
                if s >= codec_chunk_frames:
                    continue
                if s in preferred_seq:
                    continue
                if s > 0 and (s & (s - 1)) == 0:
                    preferred_seq.append(s)
            # IC-transition power-of-2 buckets between codec_chunk_frames
            # and primary (e.g. 32 and 64 for chunk=25, left=72). Hot path
            # for "chunk N of every request during IC ramp-up"; observed
            # as >20% of Stage-1 calls falling into per-request serial
            # replay when absent. Capped at bs <= ic_transition_cap_bs.
            for s in seq_sizes:
                if s < codec_chunk_frames or s >= primary:
                    continue
                if s in preferred_seq or s in ic_transition:
                    continue
                if s > 0 and (s & (s - 1)) == 0:
                    ic_transition.append(s)

        if not preferred_seq:
            preferred_seq = sorted(set(seq_sizes), reverse=True)[:2]

        for bs in bs_sizes:
            if bs == 1:
                continue
            for s in preferred_seq:
                grid.append((bs, s))
            if bs <= ic_transition_cap_bs:
                for s in ic_transition:
                    grid.append((bs, s))
        return grid

    def warmup(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.long,
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
    ):
        if device.type != "cuda" or not self.enabled or self._warmed_up:
            return

        self._device = device
        self.decoder.eval()

        if not self._explicit_sizes:
            self.capture_sizes = self.compute_capture_sizes(
                codec_chunk_frames=codec_chunk_frames,
                codec_left_context_frames=codec_left_context_frames,
            )

        grid = self._select_batched_capture_grid(
            self.capture_sizes,
            self.batch_capture_sizes,
            codec_chunk_frames=codec_chunk_frames,
            codec_left_context_frames=codec_left_context_frames,
        )
        logger.info(
            "Starting CUDA Graph warmup: %d (bs, seq_len) pairs (bs=%s, seq=%s)",
            len(grid),
            self.batch_capture_sizes,
            self.capture_sizes,
        )

        # Warmup runs (eager) to ensure all CUDA caching allocator pools are
        # populated before capture. Without this, replay can see mismatched
        # pool allocations.
        seen_warmup: set[tuple[int, int]] = set()
        for bs, size in grid:
            key = (bs, size)
            if key in seen_warmup:
                continue
            dummy = torch.zeros(bs, self.num_quantizers, size, dtype=dtype, device=device)
            with torch.no_grad():
                _ = self.decoder(dummy)
            seen_warmup.add(key)

        torch.cuda.synchronize(device)

        for bs, size in grid:
            try:
                self._capture(bs, size, device, dtype)
                logger.info("  Captured CUDA Graph for bs=%d size=%d", bs, size)
            except Exception:
                logger.warning("  Failed to capture graph for bs=%d size=%d", bs, size, exc_info=True)

        self._warmed_up = True
        logger.info(
            "CUDA Graph warmup complete: %d/%d captured",
            len(self.graphs),
            len(grid),
        )

    def _capture(self, bs: int, size: int, device: torch.device, dtype: torch.dtype):
        static_input = torch.zeros(bs, self.num_quantizers, size, dtype=dtype, device=device)
        with torch.no_grad():
            _ = self.decoder(static_input)
        torch.cuda.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_output = self.decoder(static_input)

        key = (bs, size)
        self.graphs[key] = graph
        self.static_inputs[key] = static_input
        self.static_outputs[key] = static_output

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self._warmed_up or codes.shape[0] != 1:
            return self.decoder(codes)

        actual_size = codes.shape[-1]
        padded_size = self._get_padded_size(actual_size)

        key = (1, padded_size) if padded_size is not None else None
        if key is None or key not in self.graphs:
            return self.decoder(codes)

        self.static_inputs[key].zero_()
        self.static_inputs[key][:, :, :actual_size] = codes
        self.graphs[key].replay()

        actual_out_len = actual_size * self.decoder.total_upsample
        return self.static_outputs[key][..., :actual_out_len].clone()

    def batched_decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Replay a captured graph for a batched ``[B, Q, F]`` input.

        Falls back to eager when no matching ``(bs_bucket, seq_bucket)`` graph
        has been captured. The output is trimmed to the actual sequence length
        (``F * total_upsample``) to match eager output shape.
        """
        if not self.enabled or not self._warmed_up:
            return self.decoder(codes)

        actual_bs = int(codes.shape[0])
        actual_size = int(codes.shape[-1])
        padded_bs = self._get_padded_batch_size(actual_bs)
        padded_size = self._get_padded_size(actual_size)

        self._stats["calls"] += 1

        if padded_bs is None or padded_size is None:
            self._stats["eager_full_fallback"] += 1
            self._miss_keys[(int(actual_bs), int(actual_size))] = (
                self._miss_keys.get((int(actual_bs), int(actual_size)), 0) + 1
            )
            self._maybe_log_stats()
            return self.decoder(codes)
        key = (padded_bs, padded_size)
        if key not in self.graphs:
            # No captured multi-bs graph at this (bs, F). We have two fallback
            # options:
            #   (a) a single batched eager call on the full [B, Q, F]
            #   (b) a serial per-request replay of the bs=1 graph at this F
            #
            # (a) is ~O(bs) faster than (b) at small/mid bs but allocates
            # O(bs) more transient activation memory. In the Qwen3-TTS
            # pipeline stage-0 and stage-1 share the GPU, so a large-bs
            # eager forward in stage-1 can fragment the allocator and
            # destabilize a concurrent stage-0 prefill burst.
            #
            # We pick (a) up to ``eager_fallback_max_bs`` (covers the common
            # conc=2..8 IC-phase buckets that aren't in preferred_seq) and
            # fall back to (b) above that threshold for memory safety.
            # bs==1 always takes the bs=1 graph replay since it's strictly
            # better than eager (same work, no launch overhead).
            self._miss_keys[key] = self._miss_keys.get(key, 0) + 1
            if actual_bs == 1:
                bs1_key = (1, padded_size)
                if bs1_key in self.graphs:
                    self._stats["bs1_graph_hit_via_batched"] += 1
                    self._maybe_log_stats()
                    return self.decode(codes)
                self._stats["eager_full_fallback"] += 1
                self._maybe_log_stats()
                return self.decoder(codes)
            if actual_bs <= self.eager_fallback_max_bs:
                self._stats["eager_small_bs"] += 1
                self._maybe_log_stats()
                return self.decoder(codes)
            bs1_key = (1, padded_size)
            if bs1_key in self.graphs:
                self._stats["per_request_loop"] += 1
                self._maybe_log_stats()
                outs = [self.decode(codes[i : i + 1]) for i in range(actual_bs)]
                return torch.cat(outs, dim=0)
            self._stats["eager_full_fallback"] += 1
            self._maybe_log_stats()
            return self.decoder(codes)

        self._stats["graph_hit"] += 1
        self._hit_keys[key] = self._hit_keys.get(key, 0) + 1

        static_in = self.static_inputs[key]
        static_in.zero_()
        static_in[:actual_bs, :, :actual_size].copy_(codes)
        self.graphs[key].replay()

        self._maybe_log_stats()

        actual_out_len = actual_size * self.decoder.total_upsample
        return self.static_outputs[key][:actual_bs, :, :actual_out_len].clone()

    def _maybe_log_stats(self) -> None:
        """Log a snapshot of batched_decode path counters.

        Fires every ``self._stats_log_interval`` calls, then clears counters.
        The log exposes:
          * how calls are split across fast (graph_hit) vs. slow
            (per_request_loop) paths
          * the top (bs, F) buckets that missed captured graphs
          * the top (bs, F) buckets that hit captured graphs
        """
        if self._stats["calls"] < self._stats_log_interval:
            return

        total = self._stats["calls"]
        pct = lambda n: f"{(100.0 * n / total):5.1f}%" if total else "  0.0%"
        logger.info(
            "[CUDAGraphDecoder] last %d batched_decode calls: "
            "graph_hit=%d (%s), eager_small_bs=%d (%s), "
            "per_request_loop=%d (%s), bs1_graph_via_batched=%d (%s), "
            "eager_full_fallback=%d (%s)",
            total,
            self._stats["graph_hit"], pct(self._stats["graph_hit"]),
            self._stats["eager_small_bs"], pct(self._stats["eager_small_bs"]),
            self._stats["per_request_loop"], pct(self._stats["per_request_loop"]),
            self._stats["bs1_graph_hit_via_batched"], pct(self._stats["bs1_graph_hit_via_batched"]),
            self._stats["eager_full_fallback"], pct(self._stats["eager_full_fallback"]),
        )
        if self._miss_keys:
            top_misses = sorted(self._miss_keys.items(), key=lambda kv: kv[1], reverse=True)[:10]
            logger.info(
                "[CUDAGraphDecoder] top fallback (padded_bs,padded_size) buckets: %s",
                ", ".join(f"{k}:{v}" for k, v in top_misses),
            )
        if self._hit_keys:
            top_hits = sorted(self._hit_keys.items(), key=lambda kv: kv[1], reverse=True)[:5]
            logger.info(
                "[CUDAGraphDecoder] top graph-hit (padded_bs,padded_size) buckets: %s",
                ", ".join(f"{k}:{v}" for k, v in top_hits),
            )

        for k in self._stats:
            self._stats[k] = 0
        self._miss_keys.clear()
        self._hit_keys.clear()

    def chunked_decode_with_cudagraph(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        wavs = []
        start_index = 0
        total_len = codes.shape[-1]
        total_upsample = self.decoder.total_upsample

        while start_index < total_len:
            end_index = min(start_index + chunk_size, total_len)
            context_size = left_context_size if start_index - left_context_size > 0 else start_index

            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self.decode(codes_chunk)

            wavs.append(wav_chunk[..., context_size * total_upsample :])
            start_index = end_index

        return torch.cat(wavs, dim=-1)
