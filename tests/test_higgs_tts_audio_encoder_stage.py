# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Higgs TTS ``audio_encoder`` pipeline stage.

Drives :func:`build_audio_encode_fn` with a stub fused-embedding module so
the routing / state-lookup / zero-shot / shape-validation paths can be
exercised without loading the real 1.7 B TTS checkpoint.
"""

from __future__ import annotations

import os

import pytest
import torch

from sglang_omni.models.higgs_tts.io import HiggsTtsState
from sglang_omni.models.higgs_tts.pipeline.stages import build_audio_encode_fn
from sglang_omni.models.higgs_tts.pipeline.state_io import load_state, store_state
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest

_REAL_TTS_CKPT = "/hot-data/checkpoints/TTS/c66596d6cde44fb4ba8076cc2dc1e77c/step_35500"
_real_tts_missing = not os.path.isdir(_REAL_TTS_CKPT)


class _StubFusedEmbedding(torch.nn.Module):
    """Stub matching ``HiggsFusedMultiTextEmbedding.forward`` contract:
    ``[..., N] int -> [..., D]`` float (sum already applied).

    Returns deterministic values derived from the input so tests can
    check the right codes reached us.
    """

    def __init__(self, num_codebooks: int = 8, hidden_size: int = 16) -> None:
        super().__init__()
        self._num_codebooks = num_codebooks
        self._hidden_size = hidden_size
        self._sentinel = torch.nn.Parameter(torch.zeros(1))  # so .parameters() works
        self.calls: list[torch.Tensor] = []

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        self.calls.append(codes.clone())
        T = codes.shape[0]
        # Encode the sum of each row's codes into the first channel so
        # asserts can distinguish rows.
        out = torch.zeros(T, self._hidden_size, dtype=torch.float32)
        out[:, 0] = codes.float().sum(dim=-1)
        return out


def _payload_with_state(state: HiggsTtsState) -> StagePayload:
    payload = StagePayload(
        request_id="r0",
        request=OmniRequest(inputs={"text": "x"}, params={}),
        data=None,
    )
    return store_state(payload, state)


def test_encode_emits_reference_audio_embed():
    fused = _StubFusedEmbedding(num_codebooks=8, hidden_size=16)
    codes = torch.arange(8 * 4).reshape(4, 8).tolist()
    state = HiggsTtsState(num_codebooks=8, reference_codes_delayed=codes)

    out = build_audio_encode_fn(fused, num_codebooks=8)(_payload_with_state(state))

    new_state = load_state(out)
    assert new_state.reference_audio_embed is not None
    assert len(new_state.reference_audio_embed) == 4
    assert all(len(row) == 16 for row in new_state.reference_audio_embed)
    # First channel matches the sum of the row's codes (stub contract).
    for i, row in enumerate(new_state.reference_audio_embed):
        expected = sum(codes[i])
        assert row[0] == pytest.approx(expected)


def test_encode_zero_shot_passthrough():
    """No ref codes → nothing happens, ``reference_audio_embed`` stays ``None``."""
    fused = _StubFusedEmbedding()
    state = HiggsTtsState(num_codebooks=8, reference_codes_delayed=None)
    out = build_audio_encode_fn(fused, num_codebooks=8)(_payload_with_state(state))
    new_state = load_state(out)
    assert new_state.reference_audio_embed is None
    assert fused.calls == []  # fused never invoked


def test_encode_empty_list_is_zero_shot():
    """Empty codes list (``[]``) behaves the same as ``None``."""
    fused = _StubFusedEmbedding()
    state = HiggsTtsState(num_codebooks=8, reference_codes_delayed=[])
    out = build_audio_encode_fn(fused, num_codebooks=8)(_payload_with_state(state))
    assert load_state(out).reference_audio_embed is None
    assert fused.calls == []


def test_encode_shape_mismatch_raises():
    fused = _StubFusedEmbedding(num_codebooks=4)
    bad = torch.zeros(3, 5, dtype=torch.long).tolist()
    state = HiggsTtsState(num_codebooks=4, reference_codes_delayed=bad)
    with pytest.raises(ValueError, match=r"\[T, 4\]"):
        build_audio_encode_fn(fused, num_codebooks=4)(_payload_with_state(state))


def test_encode_routes_codes_to_fused_device():
    """Codes are moved to the fused module's device before lookup."""

    class _DeviceTracker(_StubFusedEmbedding):
        def forward(self, codes):  # type: ignore[override]
            # Ensures we received a tensor on our device.
            assert codes.device == self._sentinel.device
            return super().forward(codes)

    fused = _DeviceTracker(num_codebooks=8, hidden_size=8)
    codes = torch.zeros(3, 8, dtype=torch.long).tolist()
    state = HiggsTtsState(num_codebooks=8, reference_codes_delayed=codes)
    build_audio_encode_fn(fused, num_codebooks=8)(_payload_with_state(state))
    assert len(fused.calls) == 1


@pytest.mark.skipif(
    _real_tts_missing, reason=f"Real Higgs TTS ckpt not mounted at {_REAL_TTS_CKPT}"
)
def test_load_fused_embedding_from_real_tts_ckpt():
    """Sanity-check that we can pull the fused embedding weight out of a
    real TTS ckpt and run it end-to-end through the stage."""
    from sglang_omni.models.higgs_tts.pipeline.stages import (
        _load_fused_embedding_from_tts_ckpt,
    )

    fused = _load_fused_embedding_from_tts_ckpt(_REAL_TTS_CKPT, device="cpu")
    assert fused.num_codebooks == 8
    assert fused.vocab_size == 1026

    # Three distinct code rows must produce three distinct embeddings.
    # This is stronger than just checking "any non-zero output" — a
    # freshly-initialised ``torch.empty(...)`` parameter also typically
    # has non-zero values, so the bare ``abs().max() > 0`` check would
    # silently pass on a ``copy_()`` regression. Distinct rows in →
    # distinct rows out is what actually proves the loaded weights are
    # carrying real signal.
    codes = torch.stack(
        [
            torch.zeros(8, dtype=torch.long),
            torch.ones(8, dtype=torch.long),
            torch.full((8,), 42, dtype=torch.long),
        ],
        dim=0,
    )
    out = fused(codes)
    assert out.shape == (3, fused.weight.shape[1])
    # Pairwise distinct rows (L∞ distance > 0).
    for i in range(3):
        for j in range(i + 1, 3):
            diff = float((out[i] - out[j]).abs().max())
            assert diff > 0, f"rows {i} and {j} produced identical output"
