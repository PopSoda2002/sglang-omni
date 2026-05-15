# SPDX-License-Identifier: Apache-2.0
"""Tokenizer adapter for HiggsMultimodalQwen3 TTS.

Resolves the five specials used by :meth:`HiggsTokenizerAdapter.build_prompt`
(``<|tts|>``, ``<|ref_text|>``, ``<|ref_audio|>``, ``<|text|>``, ``<|audio|>``)
and assembles the TTS prompt:

- voice-cloning + ref transcript:
    ``<|tts|> <|ref_text|> tok(ref) <|ref_audio|> [-100]×N <|text|> tok(text) <|audio|>``
- voice-cloning, no ref transcript:
    ``<|tts|> <|ref_audio|> [-100]×N <|text|> tok(text) <|audio|>``
- zero-shot:
    ``<|tts|> <|text|> tok(text) <|audio|>``

``<|tts|>`` is the task-mode token (ASR vs TTS); a missing task prefix
yields fluent-but-wrong speech.

``<|ref_text|>`` introduces the reference audio's transcript and noticeably
improves voice-clone quality vs the audio-only prompt.

``-100`` marks where :class:`HiggsFusedMultiTextEmbedding` splices the
summed multi-codebook embeddings; ``num_ref_tokens`` must equal the ref
codes' row count *after* delay (``T + num_codebooks - 1``).
"""

from __future__ import annotations

from typing import Any

# Matches the Higgs HF config's ``audio_token_id`` and transformers'
# ``IGNORE_INDEX`` convention for multimodal placeholders.
AUDIO_PLACEHOLDER_ID = -100

_REQUIRED_SPECIALS: tuple[str, ...] = (
    "<|tts|>",
    "<|ref_audio|>",
    "<|text|>",
    "<|audio|>",
)


class HiggsTokenizerAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer
        vocab = dict(tokenizer.get_added_vocab())
        missing = [t for t in _REQUIRED_SPECIALS if t not in vocab]
        if missing:
            raise ValueError(f"Tokenizer is missing Higgs TTS specials: {missing}")
        self.tts_id: int = vocab["<|tts|>"]
        self.ref_audio_id: int = vocab["<|ref_audio|>"]
        self.text_id: int = vocab["<|text|>"]
        self.audio_id: int = vocab["<|audio|>"]
        # ``<|ref_text|>`` is present in newer Higgs checkpoints and lets the
        # model condition on the reference audio's transcript. Older ckpts
        # may not have it — fall back to omitting the ref_text segment.
        self.ref_text_id: int | None = vocab.get("<|ref_text|>")

    @property
    def tokenizer(self) -> Any:
        return self._tok

    def build_prompt(
        self,
        prompt_text: str,
        *,
        num_ref_tokens: int = 0,
        reference_text: str | None = None,
    ) -> list[int]:
        """Assemble the TTS prompt token ids.

        Args:
            prompt_text: Target text to synthesize (placed after ``<|text|>``).
            num_ref_tokens: Number of ``-100`` placeholder positions to insert
                for the reference audio. Must equal the delayed ref-code row
                count (``T + num_codebooks - 1``). Set to ``0`` for zero-shot.
            reference_text: Transcript of the reference audio. When supplied
                (and the tokenizer has ``<|ref_text|>``), gets emitted as
                ``<|ref_text|> tok(ref) <|ref_audio|> [-100]×N``. Without
                it the prompt falls back to the audio-only voice-cloning form.
        """
        if num_ref_tokens < 0:
            raise ValueError(f"num_ref_tokens must be >= 0, got {num_ref_tokens}")
        ids: list[int] = [self.tts_id]
        if reference_text and num_ref_tokens > 0 and self.ref_text_id is not None:
            ids.append(self.ref_text_id)
            ids.extend(self._tok.encode(reference_text, add_special_tokens=False))
        if num_ref_tokens > 0:
            ids.append(self.ref_audio_id)
            ids.extend([AUDIO_PLACEHOLDER_ID] * num_ref_tokens)
        ids.append(self.text_id)
        ids.extend(self._tok.encode(prompt_text, add_special_tokens=False))
        ids.append(self.audio_id)
        return ids


__all__ = ["AUDIO_PLACEHOLDER_ID", "HiggsTokenizerAdapter"]
