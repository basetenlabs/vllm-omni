# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import Qwen3TTSCode2Wav

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeDecoder(nn.Module):
    def __init__(self, total_upsample: int = 4):
        super().__init__()
        self.total_upsample = total_upsample

    def chunked_decode(self, codes: torch.Tensor) -> torch.Tensor:
        frames = codes.shape[-1]
        wav_len = frames * self.total_upsample + 6
        wav = torch.arange(wav_len, dtype=torch.float32)
        return wav.view(1, 1, -1)


def _make_model() -> Qwen3TTSCode2Wav:
    model = Qwen3TTSCode2Wav(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model="unused"),
            device_config=SimpleNamespace(device=torch.device("cpu")),
        )
    )
    model._decoder = _FakeDecoder()
    model._num_quantizers = 2
    model._output_sample_rate = 24000
    model._total_upsample = 4
    model._ensure_speech_tokenizer_loaded = lambda: None
    return model


def test_forward_trims_context_on_exact_frame_boundaries():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        runtime_additional_information=[{"left_context_size": 2}],
    )

    audio = out.multimodal_outputs["model_outputs"][0]
    expected = torch.arange(8, 24, dtype=torch.float32)
    torch.testing.assert_close(audio, expected)


def test_forward_trims_trailing_padding_without_context():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        runtime_additional_information=[{"left_context_size": 0}],
    )

    audio = out.multimodal_outputs["model_outputs"][0]
    expected = torch.arange(24, dtype=torch.float32)
    torch.testing.assert_close(audio, expected)


class _FakeBatchedWrapper:
    """Minimal stand-in for CUDAGraphDecoderWrapper that runs eager but keeps
    per-call metadata so tests can verify how Code2Wav drives batched replay.
    """

    def __init__(self, decoder: nn.Module, capture_size: int):
        self.decoder = decoder
        self._warmed_up = True
        self.capture_sizes = [capture_size]
        self.batch_capture_sizes = [1, 2, 4]
        self.calls: list[tuple[int, int]] = []

    def batched_decode(self, codes: torch.Tensor) -> torch.Tensor:
        self.calls.append((int(codes.shape[0]), int(codes.shape[-1])))
        return self.decoder(codes)


class _FakeBatchDecoder(nn.Module):
    """Batched-capable decoder for testing ``_decode_valid_codes``."""

    def __init__(self, total_upsample: int = 4):
        super().__init__()
        self.total_upsample = total_upsample
        self._cudagraph_enabled = True
        self._cudagraph_wrapper: _FakeBatchedWrapper | None = None

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        # Produce a deterministic per-request waveform so we can tell outputs apart.
        bs, _, f = codes.shape
        upsample = self.total_upsample
        wav_len = f * upsample
        out = torch.zeros((bs, 1, wav_len), dtype=torch.float32)
        for i in range(bs):
            base = float(int(codes[i, 0, 0].item())) if codes[i, 0, 0].numel() > 0 else 0.0
            out[i, 0] = torch.arange(wav_len, dtype=torch.float32) + base
        return out

    def chunked_decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.forward(codes)


def test_decode_valid_codes_batches_requests_when_wrapper_available():
    """With a CUDA-graph wrapper exposing batched_decode, multiple requests
    must be stacked into a single [B, Q, F_max] call rather than looped bs=1.
    """
    model = _make_model()
    decoder = _FakeBatchDecoder()
    wrapper = _FakeBatchedWrapper(decoder, capture_size=16)
    decoder._cudagraph_wrapper = wrapper
    model._decoder = decoder
    model._num_quantizers = 2
    model._total_upsample = 4

    # Three valid requests with different tail markers in codes.
    req_a = torch.tensor([[11, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.long)  # [Q=2, F=4]
    req_b = torch.tensor([[22, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.long)  # [Q=2, F=5]
    req_c = torch.tensor([[33, 0, 0], [0, 0, 0]], dtype=torch.long)  # [Q=2, F=3]
    wavs = model._decode_valid_codes(decoder, [req_a, req_b, req_c])

    assert len(wavs) == 3
    # Exactly one batched call into the wrapper.
    assert len(wrapper.calls) == 1
    bs, f_max = wrapper.calls[0]
    assert bs == 3
    assert f_max == 5  # pad to the largest request

    # Each request's waveform is trimmed to its actual frame count.
    assert wavs[0].shape[0] == 4 * 4
    assert wavs[1].shape[0] == 5 * 4
    assert wavs[2].shape[0] == 3 * 4

    # Per-request waveforms reflect their input's first token (our fake decoder
    # uses codes[:,0,0] as an offset). This confirms batch rows weren't mixed.
    assert torch.isclose(wavs[0][0], torch.tensor(11.0))
    assert torch.isclose(wavs[1][0], torch.tensor(22.0))
    assert torch.isclose(wavs[2][0], torch.tensor(33.0))


def test_decode_valid_codes_falls_back_per_request_when_oversized():
    """When max F exceeds the wrapper's largest capture size, we must fall
    back to per-request chunked_decode to preserve numerical behavior."""
    model = _make_model()
    decoder = _FakeBatchDecoder()
    wrapper = _FakeBatchedWrapper(decoder, capture_size=4)
    decoder._cudagraph_wrapper = wrapper
    model._decoder = decoder
    model._num_quantizers = 2
    model._total_upsample = 4

    req_a = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.long)  # F=4 (fits)
    req_b = torch.tensor([[2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.long)  # F=6 (too big)
    wavs = model._decode_valid_codes(decoder, [req_a, req_b])

    assert len(wavs) == 2
    # No batched_decode calls — we fell back entirely.
    assert wrapper.calls == []
