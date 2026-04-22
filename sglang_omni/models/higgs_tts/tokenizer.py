# SPDX-License-Identifier: Apache-2.0
"""Tokenizer adapter for HiggsMultimodalQwen3 TTS.

Wraps an HF ``PreTrainedTokenizerFast`` loaded from a Higgs checkpoint and
exposes:

- The special-token ids used in the TTS prompt template
  (``<|ref_audio|>``, ``<|text|>``, ``<|audio|>``, ``<|eoc|>``, ...).
- :meth:`HiggsTokenizerAdapter.build_prompt` which assembles
  ``<|ref_audio|>[-100] × N<|text|>{prompt_text}<|audio|>`` — the voice-cloning
  prompt — or its zero-shot variant ``<|text|>{prompt_text}<|audio|>`` when
  no reference audio is supplied.

The ``-100`` placeholders mark positions where
:class:`HiggsFusedMultiTextEmbedding` injects the summed multi-codebook
embeddings for the reference audio; the caller is responsible for applying
the delay pattern to the raw ref codes and passing the resulting row count
as ``num_ref_tokens``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ``audio_token_id`` in the Higgs HF config is -100; we reuse that value as
# the multimodal placeholder so any downstream ``audio_token_id``-aware code
# paths (e.g., vLLM's ``merge_multimodal_embeddings``) see a consistent value.
AUDIO_PLACEHOLDER_ID = -100


@dataclass(frozen=True)
class HiggsSpecialTokens:
    """IDs of the special tokens used in the TTS prompt template.

    Values resolved by content from the tokenizer's added vocab so they
    track any future checkpoint that shifts ids around.
    """

    tts: int  # <|tts|>
    ref_audio: int  # <|ref_audio|>
    ref_text: int  # <|ref_text|>
    text: int  # <|text|>
    text_end: int  # <|text_end|>
    audio: int  # <|audio|>
    audio_end: int  # <|audio_end|>
    eoc: int  # <|eoc|>


# Codec-vocab special tokens (internal to the ``[N*V]`` codebook space, NOT
# the text vocab). Concrete values match the default Higgs config
# (codebook_size=1024 data + 2 specials = 1026 total).
CODEBOOK_BOC_ID = 1024
CODEBOOK_EOC_ID = 1025


class HiggsTokenizerAdapter:
    """Wraps the HF tokenizer and builds the TTS prompt sequence.

    The adapter only touches the text tokenizer. Audio encoding (raw audio
    → multi-codebook codes) is the audio-codec layer's responsibility;
    this class just accepts the resulting row count to lay out placeholders.
    """

    _REQUIRED_SPECIALS: tuple[str, ...] = (
        "<|tts|>",
        "<|ref_audio|>",
        "<|ref_text|>",
        "<|text|>",
        "<|text_end|>",
        "<|audio|>",
        "<|audio_end|>",
        "<|eoc|>",
    )

    def __init__(self, tokenizer: Any) -> None:
        """
        Args:
            tokenizer: A ``PreTrainedTokenizerFast`` (or equivalent) loaded
                from a Higgs TTS checkpoint. Must have the required TTS
                special tokens in its added vocab.
        """
        self._tok = tokenizer
        self._added_vocab: dict[str, int] = dict(tokenizer.get_added_vocab())
        missing = [t for t in self._REQUIRED_SPECIALS if t not in self._added_vocab]
        if missing:
            raise ValueError(
                f"Tokenizer is missing required Higgs TTS special tokens: {missing}. "
                f"Check that the checkpoint's tokenizer.json includes them."
            )
        self.special = HiggsSpecialTokens(
            tts=self._added_vocab["<|tts|>"],
            ref_audio=self._added_vocab["<|ref_audio|>"],
            ref_text=self._added_vocab["<|ref_text|>"],
            text=self._added_vocab["<|text|>"],
            text_end=self._added_vocab["<|text_end|>"],
            audio=self._added_vocab["<|audio|>"],
            audio_end=self._added_vocab["<|audio_end|>"],
            eoc=self._added_vocab["<|eoc|>"],
        )

    @property
    def tokenizer(self) -> Any:
        return self._tok

    @property
    def stop_token_ids(self) -> list[int]:
        """Token ids the sampler should treat as stop-of-audio markers.

        The sampler consumes these *in addition* to its multi-codebook
        wind-down logic (see PR4): when codebook-0 emits ``<|eoc|>`` the
        sampler triggers wind-down, but the text-side ``<|eoc|>`` is also
        the canonical end-of-generation marker.
        """
        return [self.special.eoc]

    def build_prompt(
        self,
        prompt_text: str,
        *,
        num_ref_tokens: int = 0,
    ) -> list[int]:
        """Assemble the TTS prompt token ids.

        Voice-cloning:
            ``<|ref_audio|>`` [-100] × ``num_ref_tokens`` ``<|text|>`` tok(prompt_text) ``<|audio|>``

        Zero-shot (no reference audio):
            ``<|text|>`` tok(prompt_text) ``<|audio|>``

        Args:
            prompt_text: The text to synthesise.
            num_ref_tokens: Number of ``-100`` placeholders to reserve for
                the reference audio's multi-codebook embedding. Should equal
                the reference codes' row count **after** applying the delay
                pattern (i.e. ``valid_len + num_codebooks - 1``). Zero to
                produce a zero-shot prompt.

        Returns:
            A flat list of integer token ids.
        """
        if num_ref_tokens < 0:
            raise ValueError(f"num_ref_tokens must be >= 0, got {num_ref_tokens}")

        ids: list[int] = []
        if num_ref_tokens > 0:
            ids.append(self.special.ref_audio)
            ids.extend([AUDIO_PLACEHOLDER_ID] * num_ref_tokens)
        ids.append(self.special.text)
        ids.extend(self._tok.encode(prompt_text, add_special_tokens=False))
        ids.append(self.special.audio)
        return ids


__all__ = [
    "AUDIO_PLACEHOLDER_ID",
    "CODEBOOK_BOC_ID",
    "CODEBOOK_EOC_ID",
    "HiggsSpecialTokens",
    "HiggsTokenizerAdapter",
]
