# SPDX-License-Identifier: Apache-2.0
"""Tests for the Higgs TTS preprocessing stage (PR3a).

Drives the ``build_preprocess_fn`` closure with a stub tokenizer so the
bulk of the suite runs without a real Higgs checkpoint on disk.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.delay_pattern import delayed_length
from sglang_omni.models.higgs_tts.io import HiggsTtsState
from sglang_omni.models.higgs_tts.pipeline.stages import build_preprocess_fn
from sglang_omni.models.higgs_tts.pipeline.state_io import load_state
from sglang_omni.models.higgs_tts.tokenizer import (
    AUDIO_PLACEHOLDER_ID,
    HiggsTokenizerAdapter,
)
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest

_SPECIAL_IDS = {
    "<|tts|>": 151667,
    "<|audio|>": 151670,
    "<|audio_end|>": 151671,
    "<|text|>": 151672,
    "<|text_end|>": 151673,
    "<|eoc|>": 151674,
    "<|ref_audio|>": 151679,
    "<|ref_text|>": 151680,
}


class _StubTokenizer:
    def __init__(self, added_vocab: dict[str, int]):
        self._added = dict(added_vocab)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [max(1, ord(c) - 32) for c in text]


def _make_adapter() -> HiggsTokenizerAdapter:
    return HiggsTokenizerAdapter(_StubTokenizer(_SPECIAL_IDS))


def _make_payload(inputs, params: dict | None = None) -> StagePayload:
    return StagePayload(
        request_id="r0",
        request=OmniRequest(inputs=inputs, params=params or {}),
        data=None,
    )


def _make_fn(num_codebooks: int = 8, codebook_size: int = 1026):
    return build_preprocess_fn(
        _make_adapter(),
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
    )


# ---------------------------------------------------------------------------
# Zero-shot (no reference audio)
# ---------------------------------------------------------------------------


def test_zero_shot_string_input_produces_prompt_with_no_placeholders():
    fn = _make_fn()
    payload = _make_payload("hi")
    out = fn(payload)

    state = load_state(out)
    assert AUDIO_PLACEHOLDER_ID not in state.prompt_token_ids
    assert state.num_ref_tokens == 0
    assert state.reference_codes_delayed is None
    # Prompt starts with <|text|> and ends with <|audio|>.
    assert state.prompt_token_ids[0] == 151672
    assert state.prompt_token_ids[-1] == 151670


def test_zero_shot_dict_input_uses_input_key():
    fn = _make_fn()
    payload = _make_payload({"input": "abc"})
    out = fn(payload)
    state = load_state(out)
    assert state.prompt_token_ids[0] == 151672
    assert state.num_ref_tokens == 0


# ---------------------------------------------------------------------------
# Voice cloning
# ---------------------------------------------------------------------------


def test_voice_cloning_applies_delay_and_inserts_placeholders():
    num_codebooks = 4
    T = 6
    codes_TN = torch.randint(0, 1024, (T, num_codebooks), dtype=torch.long).tolist()

    fn = _make_fn(num_codebooks=num_codebooks)
    payload = _make_payload({"input": "hi", "reference_codes": codes_TN})
    out = fn(payload)
    state = load_state(out)

    expected_len = delayed_length(T, num_codebooks)
    assert state.num_ref_tokens == expected_len
    assert state.reference_codes_delayed is not None
    assert len(state.reference_codes_delayed) == expected_len
    assert all(len(row) == num_codebooks for row in state.reference_codes_delayed)

    # Placeholder slice in the prompt matches num_ref_tokens.
    ids = state.prompt_token_ids
    assert ids[0] == 151679  # <|ref_audio|>
    assert ids[1 : 1 + expected_len] == [AUDIO_PLACEHOLDER_ID] * expected_len
    assert ids[1 + expected_len] == 151672  # <|text|>
    assert ids[-1] == 151670  # <|audio|>


def test_voice_cloning_accepts_transposed_codes_shape():
    """[N, T] should be auto-transposed to [T, N]."""
    num_codebooks = 4
    T = 5
    # Provide codes in [N, T] layout (matches HiggsAudioV2Tokenizer._encode
    # pre-transpose output).
    codes_NT = torch.randint(0, 1024, (num_codebooks, T), dtype=torch.long).tolist()

    fn = _make_fn(num_codebooks=num_codebooks)
    payload = _make_payload({"input": "x", "reference_codes": codes_NT})
    state = load_state(fn(payload))
    assert state.num_ref_tokens == delayed_length(T, num_codebooks)


def test_voice_cloning_rejects_ambiguous_shape():
    fn = _make_fn(num_codebooks=8)
    bad_shape = torch.zeros(3, 5, dtype=torch.long).tolist()  # neither 3 nor 5 == 8
    payload = _make_payload({"input": "x", "reference_codes": bad_shape})
    with pytest.raises(ValueError, match="num_codebooks=8"):
        fn(payload)


def test_voice_cloning_empty_codes_fall_back_to_zero_shot():
    fn = _make_fn()
    payload = _make_payload({"input": "x", "reference_codes": []})
    state = load_state(fn(payload))
    assert state.num_ref_tokens == 0
    assert state.reference_codes_delayed is None


# ---------------------------------------------------------------------------
# Generation params
# ---------------------------------------------------------------------------


def test_generation_params_propagated_to_state():
    fn = _make_fn()
    payload = _make_payload(
        {"input": "x"},
        params={
            "max_new_tokens": 512,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "seed": 42,
        },
    )
    state = load_state(fn(payload))
    assert state.max_new_tokens == 512
    assert state.temperature == 0.8
    assert state.top_p == 0.9
    assert state.top_k == 40
    assert state.seed == 42


def test_default_generation_params_when_absent():
    fn = _make_fn()
    payload = _make_payload("x")
    state = load_state(fn(payload))
    assert state.max_new_tokens == 2048
    assert state.temperature == 1.0
    assert state.top_p is None
    assert state.top_k is None
    assert state.seed is None


# ---------------------------------------------------------------------------
# State round-trip through payload.data
# ---------------------------------------------------------------------------


def test_state_roundtrips_through_payload_data():
    fn = _make_fn(num_codebooks=4)
    codes = torch.randint(0, 1024, (3, 4), dtype=torch.long).tolist()
    payload = _make_payload({"input": "hello", "reference_codes": codes})
    out = fn(payload)

    # payload.data must be a plain serialisable dict.
    assert isinstance(out.data, dict)
    assert "prompt_token_ids" in out.data

    restored = HiggsTtsState.from_dict(out.data)
    assert restored.prompt_token_ids == load_state(out).prompt_token_ids
    assert restored.num_ref_tokens == load_state(out).num_ref_tokens
