# SPDX-License-Identifier: Apache-2.0
"""Tokenizer adapter for HiggsMultimodalQwen3 TTS.

Resolves the three specials used by :meth:`HiggsTokenizerAdapter.build_prompt`
(``<|ref_audio|>``, ``<|text|>``, ``<|audio|>``) and assembles:

- voice-cloning:  ``<|ref_audio|> [-100]×N <|text|> tok(text) <|audio|>``
- zero-shot:      ``<|text|> tok(text) <|audio|>``

``-100`` is the position at which :class:`HiggsFusedMultiTextEmbedding` will
splice in the summed multi-codebook embeddings for the reference audio; the
caller supplies ``num_ref_tokens`` equal to the ref codes' row count *after*
the delay pattern (``T + num_codebooks - 1``).
"""

from __future__ import annotations

from typing import Any

# Matches the Higgs HF config's ``audio_token_id`` and transformers'
# ``IGNORE_INDEX`` convention for multimodal placeholders.
AUDIO_PLACEHOLDER_ID = -100

_REQUIRED_SPECIALS: tuple[str, ...] = ("<|ref_audio|>", "<|text|>", "<|audio|>")


class HiggsTokenizerAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer
        vocab = dict(tokenizer.get_added_vocab())
        missing = [t for t in _REQUIRED_SPECIALS if t not in vocab]
        if missing:
            raise ValueError(f"Tokenizer is missing Higgs TTS specials: {missing}")
        self.ref_audio_id: int = vocab["<|ref_audio|>"]
        self.text_id: int = vocab["<|text|>"]
        self.audio_id: int = vocab["<|audio|>"]

    @property
    def tokenizer(self) -> Any:
        return self._tok

    def build_prompt(self, prompt_text: str, *, num_ref_tokens: int = 0) -> list[int]:
        if num_ref_tokens < 0:
            raise ValueError(f"num_ref_tokens must be >= 0, got {num_ref_tokens}")
        ids: list[int] = []
        if num_ref_tokens > 0:
            ids.append(self.ref_audio_id)
            ids.extend([AUDIO_PLACEHOLDER_ID] * num_ref_tokens)
        ids.append(self.text_id)
        ids.extend(self._tok.encode(prompt_text, add_special_tokens=False))
        ids.append(self.audio_id)
        return ids


__all__ = ["AUDIO_PLACEHOLDER_ID", "HiggsTokenizerAdapter"]
