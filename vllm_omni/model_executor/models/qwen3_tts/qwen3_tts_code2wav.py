from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .qwen3_tts_tokenizer import Qwen3TTSTokenizer

logger = init_logger(__name__)


class Qwen3TTSCode2Wav(nn.Module):
    """Stage-1 code2wav model for Qwen3-TTS (GenerationModelRunner).
    Consumes frame-aligned codec tokens from input_ids and decodes waveform
    via the SpeechTokenizer decoder directly (bypassing HF wrapper overhead)."""

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._speech_tokenizer: Qwen3TTSTokenizer | None = None
        self._decoder: nn.Module | None = None
        self._num_quantizers: int | None = None
        self._output_sample_rate: int | None = None
        self._total_upsample: int | None = None
        self._decoder_sliding_window: int | None = None
        self._logged_codec_stats = False

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def _compute_batch_capture_sizes(self) -> list[int]:
        """Choose the batch-size buckets to capture CUDA graphs for.

        The set is capped by the scheduler's ``max_num_seqs`` so we don't
        capture graphs that can never be used, and truncated to a small
        power-of-two ladder to bound total capture memory. Each extra bucket
        at the largest F adds ~``bs * F * total_upsample * 4`` bytes of
        static-output memory, so overshooting the scheduler limit is pure
        waste.
        """
        try:
            scheduler_cfg = getattr(self.vllm_config, "scheduler_config", None)
            max_num_seqs = int(getattr(scheduler_cfg, "max_num_seqs", 1) or 1)
        except Exception:
            max_num_seqs = 1
        candidate = [1, 2, 4, 8, 16, 32, 64]
        buckets = [b for b in candidate if b <= max_num_seqs]
        if not buckets:
            buckets = [1]
        # Ensure the scheduler cap itself is a bucket so the biggest steps
        # actually hit a graph.
        if max_num_seqs not in buckets and max_num_seqs > buckets[-1]:
            buckets.append(int(max_num_seqs))
        return buckets

    def _ensure_speech_tokenizer_loaded(self) -> None:
        if self._decoder is not None:
            return

        cfg_path = cached_file(self.model_path, "speech_tokenizer/config.json")
        if cfg_path is None:
            raise ValueError(f"{self.model_path}/speech_tokenizer/config.json not found")
        speech_tokenizer_dir = os.path.dirname(cfg_path)

        prep_cfg = cached_file(self.model_path, "speech_tokenizer/preprocessor_config.json")
        if prep_cfg is None:
            raise ValueError(
                f"{self.model_path}/speech_tokenizer/preprocessor_config.json not found. "
                "Please make sure the checkpoint contains the required HF preprocessing files."
            )

        tok = Qwen3TTSTokenizer.from_pretrained(
            speech_tokenizer_dir,
            torch_dtype=torch.float32,
            load_feature_extractor=False,
        )

        if tok.model is not None:
            tok.model.to(device=self.vllm_config.device_config.device)
            tok.device = self._module_device(tok.model)

        dec_cfg = getattr(tok.model.config, "decoder_config", None)
        num_q = getattr(dec_cfg, "num_quantizers", None) if dec_cfg is not None else None
        if num_q is None:
            raise ValueError("speech_tokenizer decoder_config.num_quantizers not found")
        num_q = int(num_q)
        if num_q <= 0:
            raise ValueError(f"Invalid speech_tokenizer num_quantizers={num_q}")

        try:
            upsample = int(tok.get_decode_upsample_rate())
        except Exception as e:
            raise ValueError(f"Failed to get decode upsample rate: {e}") from e
        if upsample <= 0:
            raise ValueError(f"Invalid decode upsample rate: {upsample}")

        try:
            out_sr = int(tok.get_output_sample_rate())
        except Exception as e:
            raise ValueError(f"Failed to get output sample rate: {e}") from e

        decoder = tok.model.decoder
        decoder.eval()

        self._speech_tokenizer = tok
        self._decoder = decoder
        self._num_quantizers = num_q
        self._output_sample_rate = out_sr
        self._total_upsample = int(decoder.total_upsample)
        self._decoder_sliding_window = int(getattr(dec_cfg, "sliding_window", 0) or 0)

        # Precompute SnakeBeta exp caches (benefits both Triton and eager paths)
        if hasattr(decoder, "precompute_snake_caches"):
            decoder.precompute_snake_caches()

        if hasattr(decoder, "enable_cudagraph"):
            device = self._module_device(decoder)
            if device.type == "cuda":
                try:
                    chunk_frames = 0
                    left_frames = 0

                    model_cfg = getattr(self.vllm_config, "model_config", None)
                    connector_cfg = getattr(model_cfg, "stage_connector_config", None)
                    extra_cfg = (
                        connector_cfg.get("extra", connector_cfg)
                        if isinstance(connector_cfg, dict)
                        else getattr(connector_cfg, "extra", None)
                    )
                    if isinstance(extra_cfg, dict):
                        chunk_frames = int(extra_cfg.get("codec_chunk_frames") or 0)
                        left_frames = int(extra_cfg.get("codec_left_context_frames") or 0)
                        if (
                            chunk_frames > 0
                            and left_frames > 0
                            and self._decoder_sliding_window
                            and left_frames < self._decoder_sliding_window
                        ):
                            logger.warning(
                                "Qwen3-TTS streaming codec_left_context_frames=%d is smaller than "
                                "decoder sliding_window=%d; chunk-boundary distortion may occur. "
                                "Increase codec_left_context_frames to at least %d for streaming.",
                                left_frames,
                                self._decoder_sliding_window,
                                self._decoder_sliding_window,
                            )

                    # Compute batch-size buckets so that multi-request steps
                    # can replay a pre-captured graph and avoid serial bs=1
                    # replays. We keep the set small to bound capture memory.
                    batch_buckets = self._compute_batch_capture_sizes()

                    decoder.enable_cudagraph(
                        device=device,
                        codec_chunk_frames=chunk_frames,
                        codec_left_context_frames=left_frames,
                        batch_capture_sizes=batch_buckets,
                    )
                    logger.info(
                        "Code2Wav decoder CUDA Graph enabled (bs_buckets=%s)",
                        batch_buckets,
                    )
                except Exception:
                    logger.warning("Failed to enable CUDA Graph for Code2Wav decoder", exc_info=True)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # This stage ignores token embeddings. Keep a stable dummy embedding for vLLM runner.
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def _split_request_ids(self, ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        """Split concatenated input_ids into per-request segments.

        Uses seq_token_counts (injected by the runner via model_kwargs) when
        available, falling back to forward-context ubatch_slices when
        micro-batching is active. Returns [ids] for single-request batches.
        """
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + s)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode codec codes into audio waveform.

        input_ids layout per request: [codec_context_frames, *flat_codes]
        where flat_codes is codebook-major [q*F].

        Bypasses the HF Qwen3TTSTokenizer.decode() wrapper and calls the
        decoder.chunked_decode() directly to avoid GPU->CPU->GPU round-trips.
        Length management is done here instead of relying on HF's padding=-1
        sentinel logic.
        """
        self._ensure_speech_tokenizer_loaded()
        assert self._decoder is not None
        assert self._num_quantizers is not None
        assert self._total_upsample is not None

        decoder = self._decoder
        q = int(self._num_quantizers)
        upsample = int(self._total_upsample)
        sr_val = int(self._output_sample_rate)
        sr_tensor = torch.tensor(sr_val, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))

        parsed: list[tuple[int, int]] = []
        valid_codes_qf: list[torch.Tensor] = []
        valid_indices: list[int] = []
        left_context_size = [0] * len(request_ids_list)
        if runtime_additional_information is not None:
            for i, info in enumerate(runtime_additional_information):
                if i >= len(left_context_size):
                    break
                if "left_context_size" in info:
                    # left_context_size may come through serialization as an int, [int], or tensor([int]).
                    value = info["left_context_size"]
                    if isinstance(value, list):
                        value = value[0] if value else 0
                    if isinstance(value, torch.Tensor):
                        value = value.reshape(-1)[0].item() if value.numel() > 0 else 0
                    left_context_size[i] = int(value)
        for i, req_ids in enumerate(request_ids_list):
            if req_ids.numel() < 1:
                parsed.append((0, 0))
                continue
            ctx_frames = left_context_size[i]
            flat = req_ids
            n = flat.numel()
            if n == 0 or n % q != 0:
                if n > 0:
                    logger.warning(
                        "Code2Wav input_ids length %d not divisible by num_quantizers %d; skipping malformed request.",
                        n,
                        q,
                    )
                parsed.append((0, 0))
                continue
            frames = n // q
            # [q*F] -> [Q, F] for direct decoder call (decoder expects [B, Q, F])
            codes_qf = flat.reshape(q, frames)
            parsed.append((ctx_frames, frames))
            valid_codes_qf.append(codes_qf)
            valid_indices.append(i)

        num_req = len(request_ids_list)
        if not valid_codes_qf:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty] * num_req,
                    "sr": [sr_tensor] * num_req,
                },
            )

        if not self._logged_codec_stats:
            self._logged_codec_stats = True
            try:
                c = valid_codes_qf[0]
                logger.info(
                    "Code2Wav codec: frames=%d q=%d uniq=%d range=[%d,%d] batch=%d",
                    c.shape[1],
                    q,
                    int(torch.unique(c).numel()),
                    int(c.min().item()),
                    int(c.max().item()),
                    len(valid_codes_qf),
                )
            except Exception:
                pass

        # Batched decode path: stack all requests into a single [B, Q, F] and
        # replay a pre-captured (bs, F) graph when available. This avoids the
        # serial bs=1 loop and slashes per-request launch overhead.
        wav_tensors = self._decode_valid_codes(decoder, valid_codes_qf)

        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        for j, idx in enumerate(valid_indices):
            ctx_frames, actual_frames = parsed[idx]
            wav = wav_tensors[j]
            # Slice on exact codec-frame boundaries instead of proportionally.
            start = max(0, ctx_frames * upsample)
            end = max(start, actual_frames * upsample)
            if start >= wav.shape[0]:
                logger.warning(
                    "Context trim start %d >= decoded length %d; returning empty audio.",
                    start,
                    wav.shape[0],
                )
                continue
            wav = wav[start : min(end, wav.shape[0])]
            if wav.shape[0] > 0:
                audios[idx] = wav.to(dtype=torch.float32).reshape(-1)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def _decode_valid_codes(
        self,
        decoder: nn.Module,
        valid_codes_qf: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Run the speech-tokenizer decoder across a batch of valid requests.

        Attempts three paths in order of preference:
        1. Fast batched CUDA-graph replay via the wrapper's ``batched_decode``
           when all requests share a seq length below the largest capture
           bucket. Requires the decoder to have an active wrapper exposing
           ``batched_decode``.
        2. Single eager batched call (no CUDA graph) as a cheap fallback when
           the wrapper isn't present but requests can still be stacked.
        3. Per-request ``chunked_decode`` (legacy) when any single request's
           F exceeds the wrapper's largest captured size (non-streaming /
           oversized inputs).
        """
        assert valid_codes_qf, "caller should skip empty batches"
        actual_frames = [int(c.shape[-1]) for c in valid_codes_qf]
        max_frames = max(actual_frames)

        wrapper = getattr(decoder, "_cudagraph_wrapper", None)
        use_graph = (
            getattr(decoder, "_cudagraph_enabled", False)
            and wrapper is not None
            and hasattr(wrapper, "batched_decode")
            and getattr(wrapper, "_warmed_up", False)
            and wrapper.capture_sizes
            and max_frames <= int(wrapper.capture_sizes[-1])
        )

        if len(valid_codes_qf) == 1 and not use_graph:
            # Preserve the legacy single-request path (which internally uses
            # the wrapper's bs=1 graph via decode() if enabled).
            codes_bqf = valid_codes_qf[0].unsqueeze(0)
            wav = decoder.chunked_decode(codes_bqf)
            return [wav.squeeze(0).squeeze(0)]

        if not use_graph:
            # No wrapper available, or an input exceeds all captured sizes.
            # Fall back to serial chunked_decode; this preserves numerical
            # behavior for long non-streaming inputs.
            out: list[torch.Tensor] = []
            for codes_qf in valid_codes_qf:
                codes_bqf = codes_qf.unsqueeze(0)
                wav = decoder.chunked_decode(codes_bqf)
                out.append(wav.squeeze(0).squeeze(0))
            return out

        # Build a single [B, Q, F_max] batched input. We pad on the right
        # with zeros, which is safe for the causal Qwen3-TTS decoder (padding
        # at the tail cannot leak into kept output positions).
        q = int(self._num_quantizers)
        device = valid_codes_qf[0].device
        dtype = valid_codes_qf[0].dtype
        batched = torch.zeros(
            (len(valid_codes_qf), q, max_frames),
            dtype=dtype,
            device=device,
        )
        for i, codes_qf in enumerate(valid_codes_qf):
            f_i = int(codes_qf.shape[-1])
            batched[i, :, :f_i] = codes_qf

        wav_batched = wrapper.batched_decode(batched)  # [B, 1, F_max * upsample]

        upsample = int(self._total_upsample)
        out: list[torch.Tensor] = []
        for i, f_i in enumerate(actual_frames):
            # Slice to each request's actual decoded length before
            # context-trim runs in the caller.
            out.append(wav_batched[i, 0, : f_i * upsample])
        return out

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        if not (isinstance(model_outputs, tuple) and len(model_outputs) == 2):
            raise TypeError(f"Qwen3TTSCode2Wav expected (audio_tensor, sr) outputs, got {type(model_outputs)}")

        audio_tensor, sr = model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": audio_tensor,
                "sr": sr,
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # SpeechTokenizer weights live under `speech_tokenizer/` and are loaded
        # lazily from that directory. Ignore main checkpoint weights.
        return set()
