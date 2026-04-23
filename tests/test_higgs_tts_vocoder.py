# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Higgs TTS vocoder stage (PR5).

Drives :func:`build_vocode_fn` with a deterministic stub codec so we can
exercise the routing / state-lookup / EOC-trim / empty-path logic
without loading the real tokenizer checkpoint.
"""

from __future__ import annotations

import numpy as np
import torch

from sglang_omni.models.higgs_tts.delay_pattern import (
    BOC_ID,
    EOC_ID,
    apply_delay_pattern,
)
from sglang_omni.models.higgs_tts.io import HiggsTtsState
from sglang_omni.models.higgs_tts.pipeline.stages import build_vocode_fn
from sglang_omni.models.higgs_tts.pipeline.state_io import store_state
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest


class _StubCodec:
    """Deterministic codec stand-in for vocoder stage tests.

    ``decode`` returns a waveform whose length is ``T * samples_per_frame``
    and whose values are ``0.01 * code_sum`` so the test can verify both
    the right frames were passed through and the right length emerged.
    """

    def __init__(self, samples_per_frame: int = 960, num_codebooks: int = 8):
        self.samples_per_frame = samples_per_frame
        self.num_codebooks = num_codebooks
        self.calls: list[torch.Tensor] = []

    def decode(self, codes_TN: torch.Tensor) -> torch.Tensor:
        self.calls.append(codes_TN.clone())
        T = codes_TN.shape[0]
        # Scale by code magnitudes so the test can distinguish different
        # inputs.
        mag = codes_TN.float().sum(dim=-1).mean().item()
        return torch.full((T * self.samples_per_frame,), 0.01 * mag)


def _payload_with_state(state: HiggsTtsState) -> StagePayload:
    payload = StagePayload(
        request_id="r0",
        request=OmniRequest(inputs={"text": "hi"}, params={}),
        data=None,
    )
    return store_state(payload, state)


def _state_with_codes_TN(codes_TN: torch.Tensor, **state_kwargs) -> HiggsTtsState:
    """Wrap raw ``[T, N]`` codes as the delayed stream the engine produces."""
    delayed = apply_delay_pattern(codes_TN)
    return HiggsTtsState(
        num_codebooks=codes_TN.shape[1],
        output_codes_delayed=delayed.tolist(),
        **state_kwargs,
    )


def test_vocode_decodes_delayed_codes():
    """Round-trip: apply delay → vocoder reverses it + calls codec.decode."""
    codec = _StubCodec()
    codes = torch.randint(0, 1024, (12, 8), dtype=torch.long)
    state = _state_with_codes_TN(codes)
    out = build_vocode_fn(codec)(_payload_with_state(state))

    assert len(codec.calls) == 1
    # Codec received the un-delayed codes.
    assert torch.equal(codec.calls[0], codes)
    # audio_data length == T * samples_per_frame.
    assert out.data["audio_data"].shape == (12 * 960,)
    assert out.data["sample_rate"] == 24_000
    assert out.data["modality"] == "audio"


def test_vocode_preserves_windown_tail_as_data():
    """``apply_delay_pattern`` naturally fills the last ``N-1`` rows of
    codebook-0 with EOC; the vocoder must feed those straight into
    ``reverse_delay_pattern`` so all ``T`` data rows round-trip. The
    sampler's wind-down is NOT junk — it carries the tail of
    codebooks 1..N-1."""
    codec = _StubCodec()
    real_codes = torch.randint(0, 1024, (5, 8), dtype=torch.long)
    delayed = apply_delay_pattern(real_codes)
    # Last N-1 = 7 rows should have cb0 == EOC by construction.
    assert (delayed[-7:, 0] == EOC_ID).all()

    state = HiggsTtsState(num_codebooks=8, output_codes_delayed=delayed.tolist())
    out = build_vocode_fn(codec)(_payload_with_state(state))

    assert len(codec.calls) == 1
    # All 5 real rows survive — no trimming.
    assert torch.equal(codec.calls[0], real_codes)
    assert out.data["audio_data"].shape == (5 * 960,)


def test_vocode_empty_output_graceful():
    """No engine output → zero-length waveform, no crash."""
    codec = _StubCodec()
    state = HiggsTtsState(num_codebooks=8, output_codes_delayed=None)
    out = build_vocode_fn(codec)(_payload_with_state(state))

    assert codec.calls == []
    assert out.data["audio_data"].shape == (0,)
    assert out.data["sample_rate"] == 24_000
    assert out.data["modality"] == "audio"


def test_vocode_too_short_graceful():
    """After trimming, if we have fewer than N rows, return empty rather
    than crash inside ``reverse_delay_pattern``."""
    codec = _StubCodec()
    # Only BOC prefix — no data rows at all.
    bad = torch.full((3, 8), BOC_ID, dtype=torch.long)
    state = HiggsTtsState(num_codebooks=8, output_codes_delayed=bad.tolist())
    out = build_vocode_fn(codec)(_payload_with_state(state))

    assert codec.calls == []
    assert out.data["audio_data"].shape == (0,)


def test_vocode_propagates_usage():
    codec = _StubCodec()
    codes = torch.randint(0, 1024, (4, 8), dtype=torch.long)
    state = _state_with_codes_TN(
        codes, prompt_tokens=32, completion_tokens=4, engine_time_s=0.123456
    )
    out = build_vocode_fn(codec)(_payload_with_state(state))

    assert "usage" in out.data
    usage = out.data["usage"]
    assert usage["prompt_tokens"] == 32
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == 36
    assert usage["engine_time_s"] == round(0.123456, 6)


def test_vocode_writes_numpy_float32():
    codec = _StubCodec()
    codes = torch.randint(0, 1024, (4, 8), dtype=torch.long)
    state = _state_with_codes_TN(codes)
    out = build_vocode_fn(codec)(_payload_with_state(state))

    arr = out.data["audio_data"]
    # Must be numpy float32 so the downstream WAV encoder can pass it
    # straight through (see ``sglang_omni/client/client.py::_set_audio_data``).
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32


def test_vocode_custom_sample_rate():
    codec = _StubCodec()
    codes = torch.randint(0, 1024, (3, 8), dtype=torch.long)
    state = _state_with_codes_TN(codes)
    out = build_vocode_fn(codec, sample_rate=16_000)(_payload_with_state(state))

    assert out.data["sample_rate"] == 16_000


def test_vocode_honours_state_num_codebooks():
    """A state with a custom N must route through reverse_delay_pattern
    with the right N, not the hardcoded default."""
    codec = _StubCodec(num_codebooks=4)
    codes = torch.randint(0, 1024, (6, 4), dtype=torch.long)
    state = _state_with_codes_TN(codes)  # num_codebooks=4 from shape
    out = build_vocode_fn(codec)(_payload_with_state(state))

    assert codec.calls[0].shape == (6, 4)
    assert out.data["audio_data"].shape == (6 * 960,)
