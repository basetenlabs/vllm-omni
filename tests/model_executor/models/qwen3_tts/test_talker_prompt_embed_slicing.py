# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the talker prompt-embedding host/device helpers.

The helpers batch the CPU-side prompt-embedding slice and the tts_pad_embed
padding into a single, async H2D copy — replacing the prior pageable->GPU
round-trip. These tests pin down the functional contract (shape, dtype, values)
without requiring the full talker model to be loaded.
"""
import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_prompt_embeds_to_host_returns_cpu_tensor():
    src = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    host = Qwen3TTSTalkerForConditionalGeneration._prompt_embeds_to_host(src)
    assert host.device.type == "cpu"
    assert host.shape == src.shape
    torch.testing.assert_close(host.to(torch.float32), src.to(torch.float32))


def test_slice_prompt_embeds_no_padding_needed():
    src = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    tts_pad = torch.full((4,), -1.0, dtype=torch.bfloat16)
    out = Qwen3TTSTalkerForConditionalGeneration._slice_prompt_embeds_to_device(
        src, start=1, span_len=3, tts_pad_embed=tts_pad, device=torch.device("cpu")
    )
    assert out.shape == (3, 4)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.to(torch.float32), src[1:4].to(torch.float32))


def test_slice_prompt_embeds_pads_with_tts_pad_at_tail():
    src = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    tts_pad = torch.full((4,), -7.0, dtype=torch.bfloat16)
    # start=2, span_len=5 reaches past the end (only 2 rows left) -> 3 pad rows.
    out = Qwen3TTSTalkerForConditionalGeneration._slice_prompt_embeds_to_device(
        src, start=2, span_len=5, tts_pad_embed=tts_pad, device=torch.device("cpu")
    )
    assert out.shape == (5, 4)
    torch.testing.assert_close(out[:2].to(torch.float32), src[2:4].to(torch.float32))
    torch.testing.assert_close(
        out[2:].to(torch.float32),
        torch.full((3, 4), -7.0, dtype=torch.float32),
    )


def test_slice_prompt_embeds_start_beyond_end_yields_all_padding():
    src = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    tts_pad = torch.full((4,), 9.0, dtype=torch.bfloat16)
    out = Qwen3TTSTalkerForConditionalGeneration._slice_prompt_embeds_to_device(
        src, start=10, span_len=3, tts_pad_embed=tts_pad, device=torch.device("cpu")
    )
    assert out.shape == (3, 4)
    torch.testing.assert_close(out.to(torch.float32), torch.full((3, 4), 9.0, dtype=torch.float32))
