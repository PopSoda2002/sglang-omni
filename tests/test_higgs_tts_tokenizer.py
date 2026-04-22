# SPDX-License-Identifier: Apache-2.0
"""Tests for the Higgs TTS tokenizer adapter (PR3a).

Uses the real Higgs checkpoint to validate special-token id resolution
against the shipped ``tokenizer.json`` when available; falls back to a
synthetic tokenizer for the build-prompt layout tests (so the bulk of the
suite passes without the 4 GB checkpoint mounted).
"""

from __future__ import annotations

import os

import pytest

from sglang_omni.models.higgs_tts.tokenizer import (
    AUDIO_PLACEHOLDER_ID,
    CODEBOOK_BOC_ID,
    CODEBOOK_EOC_ID,
    HiggsTokenizerAdapter,
)

_REAL_CKPT = "/hot-data/checkpoints/TTS/c66596d6cde44fb4ba8076cc2dc1e77c/step_35500"

# Special-token ids in the reference checkpoint. If the checkpoint isn't
# mounted we still verify these values in the synthetic tokenizer test.
_EXPECTED_SPECIAL_IDS = {
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
    """Minimal stand-in for ``PreTrainedTokenizerFast`` covering the surface
    :class:`HiggsTokenizerAdapter` uses."""

    def __init__(self, added_vocab: dict[str, int]):
        self._added = dict(added_vocab)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Trivial deterministic mapping: one token per character, id = ord - 32.
        return [max(1, ord(c) - 32) for c in text]


def _make_stub_adapter() -> HiggsTokenizerAdapter:
    return HiggsTokenizerAdapter(_StubTokenizer(_EXPECTED_SPECIAL_IDS))


# ---------------------------------------------------------------------------
# Special-token resolution
# ---------------------------------------------------------------------------


def test_stub_tokenizer_resolves_all_required_specials():
    adapter = _make_stub_adapter()
    assert adapter.special.tts == 151667
    assert adapter.special.ref_audio == 151679
    assert adapter.special.ref_text == 151680
    assert adapter.special.text == 151672
    assert adapter.special.audio == 151670
    assert adapter.special.audio_end == 151671
    assert adapter.special.text_end == 151673
    assert adapter.special.eoc == 151674
    assert adapter.stop_token_ids == [151674]


def test_missing_special_tokens_raises():
    incomplete = {k: v for k, v in _EXPECTED_SPECIAL_IDS.items() if k != "<|audio|>"}
    with pytest.raises(ValueError, match="<|audio|>"):
        HiggsTokenizerAdapter(_StubTokenizer(incomplete))


@pytest.mark.skipif(
    not os.path.isdir(_REAL_CKPT),
    reason="real Higgs TTS checkpoint not mounted",
)
def test_real_checkpoint_special_token_ids_match():
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(_REAL_CKPT)
    except (AttributeError, TypeError, ValueError) as e:
        # Higgs checkpoints ship ``transformers_version=5.2.0`` metadata; on
        # pinned transformers<5 the ``extra_special_tokens`` schema differs
        # (list in ckpt, dict in current tokenizer). Skip rather than fail —
        # the stub-based test above already validates the adapter surface.
        pytest.skip(f"tokenizer load failed on this transformers version: {e}")

    adapter = HiggsTokenizerAdapter(tok)
    for name, expected_id in _EXPECTED_SPECIAL_IDS.items():
        actual = tok.convert_tokens_to_ids(name)
        assert actual == expected_id, f"{name}: expected {expected_id}, got {actual}"
    # Spot-check adapter surface matches the resolved ids.
    assert adapter.special.ref_audio == 151679
    assert adapter.special.eoc == 151674


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def test_build_prompt_voice_cloning_layout():
    adapter = _make_stub_adapter()
    text = "hi"
    num_ref = 5
    ids = adapter.build_prompt(text, num_ref_tokens=num_ref)

    assert ids[0] == adapter.special.ref_audio
    assert ids[1 : 1 + num_ref] == [AUDIO_PLACEHOLDER_ID] * num_ref
    assert ids[1 + num_ref] == adapter.special.text
    assert ids[-1] == adapter.special.audio
    text_tokens = ids[2 + num_ref : -1]
    assert text_tokens == adapter.tokenizer.encode(text, add_special_tokens=False)


def test_build_prompt_zero_shot_omits_ref_section():
    adapter = _make_stub_adapter()
    ids = adapter.build_prompt("hello", num_ref_tokens=0)
    assert AUDIO_PLACEHOLDER_ID not in ids
    assert ids[0] == adapter.special.text
    assert ids[-1] == adapter.special.audio
    assert adapter.special.ref_audio not in ids


def test_build_prompt_rejects_negative_num_ref_tokens():
    adapter = _make_stub_adapter()
    with pytest.raises(ValueError, match=">= 0"):
        adapter.build_prompt("x", num_ref_tokens=-1)


def test_placeholder_id_matches_audio_token_id_convention():
    # -100 is the Higgs HF config's audio_token_id and matches vLLM /
    # transformers' IGNORE_INDEX convention for multimodal placeholders.
    assert AUDIO_PLACEHOLDER_ID == -100


def test_codebook_special_ids_match_higgs_default():
    # vocab_size = 1026; last two entries are <|boc|> and <|eoc|>.
    assert CODEBOOK_BOC_ID == 1024
    assert CODEBOOK_EOC_ID == 1025
