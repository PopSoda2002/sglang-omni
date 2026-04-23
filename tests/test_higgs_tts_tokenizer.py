# SPDX-License-Identifier: Apache-2.0
"""Tests for the Higgs TTS tokenizer adapter (PR3a)."""

from __future__ import annotations

import os

import pytest

from sglang_omni.models.higgs_tts.tokenizer import (
    AUDIO_PLACEHOLDER_ID,
    HiggsTokenizerAdapter,
)

_REAL_CKPT = "/hot-data/checkpoints/TTS/c66596d6cde44fb4ba8076cc2dc1e77c/step_35500"

# Expected ids from the reference Higgs checkpoint.
_EXPECTED_IDS = {
    "<|tts|>": 151667,
    "<|ref_audio|>": 151679,
    "<|text|>": 151672,
    "<|audio|>": 151670,
}


class _StubTokenizer:
    def __init__(self, added_vocab: dict[str, int]):
        self._added = dict(added_vocab)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [max(1, ord(c) - 32) for c in text]


def _make_adapter() -> HiggsTokenizerAdapter:
    return HiggsTokenizerAdapter(_StubTokenizer(_EXPECTED_IDS))


def test_special_ids_resolved():
    adapter = _make_adapter()
    assert adapter.tts_id == 151667
    assert adapter.ref_audio_id == 151679
    assert adapter.text_id == 151672
    assert adapter.audio_id == 151670


def test_missing_special_raises():
    incomplete = {k: v for k, v in _EXPECTED_IDS.items() if k != "<|audio|>"}
    with pytest.raises(ValueError, match=r"<\|audio\|>"):
        HiggsTokenizerAdapter(_StubTokenizer(incomplete))


def test_build_prompt_voice_cloning():
    adapter = _make_adapter()
    ids = adapter.build_prompt("hi", num_ref_tokens=5)
    # <|tts|> task token first, then the ref audio segment.
    assert ids[0] == adapter.tts_id
    assert ids[1] == adapter.ref_audio_id
    assert ids[2:7] == [AUDIO_PLACEHOLDER_ID] * 5
    assert ids[7] == adapter.text_id
    assert ids[-1] == adapter.audio_id
    assert ids[8:-1] == adapter.tokenizer.encode("hi", add_special_tokens=False)


def test_build_prompt_zero_shot():
    adapter = _make_adapter()
    ids = adapter.build_prompt("hello", num_ref_tokens=0)
    assert AUDIO_PLACEHOLDER_ID not in ids
    assert ids[0] == adapter.tts_id
    assert ids[1] == adapter.text_id
    assert ids[-1] == adapter.audio_id
    assert adapter.ref_audio_id not in ids


def test_build_prompt_rejects_negative():
    with pytest.raises(ValueError, match=">= 0"):
        _make_adapter().build_prompt("x", num_ref_tokens=-1)


@pytest.mark.skipif(
    not os.path.isdir(_REAL_CKPT), reason="real Higgs TTS checkpoint not mounted"
)
def test_real_checkpoint_ids_match():
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(_REAL_CKPT)
    except (AttributeError, TypeError, ValueError) as e:
        # Ckpt metadata is transformers_version=5.2.0; pinned transformers<5
        # can't load its ``extra_special_tokens`` schema. Skip rather than fail.
        pytest.skip(f"tokenizer load failed: {e}")
    for name, expected in _EXPECTED_IDS.items():
        assert tok.convert_tokens_to_ids(name) == expected, name
