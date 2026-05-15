# SPDX-License-Identifier: Apache-2.0
"""Per-request pipeline state for Higgs TTS.

Carried between stages via :class:`sglang_omni.proto.StagePayload.data`.
Mirrors :class:`sglang_omni.models.higgs_tts.io.HiggsTtsState` from the V0
package — same wire format so V0/V1 can exchange state in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HiggsTtsState:
    """Per-request state for the Higgs TTS pipeline."""

    prompt_token_ids: list[int] = field(default_factory=list)
    """Prompt ids with ``AUDIO_PLACEHOLDER_ID`` (-100) at ref-audio positions."""

    reference_codes_delayed: list[list[int]] | None = None
    """Delayed ref-audio codes, shape ``[num_ref_tokens, num_codebooks]`` as a
    nested list. ``None`` for zero-shot."""

    reference_audio_embed: Any | None = None
    """Pre-computed fused audio embedding, shape ``[num_ref_tokens, hidden_size]``,
    pasted at ``-100`` placeholder positions by the engine prefill. ``None`` for
    zero-shot. Stored as a CPU fp32 ``torch.Tensor`` — ``pipeline.relay_io.extract_tensors``
    ships any tensor in ``StagePayload.data`` over the raw tensor buffer instead
    of pickling it (the embed is MB-scale for long references).
    """

    num_codebooks: int = 8
    codebook_size: int = 1026  # 1024 data + <|boc|> + <|eoc|>

    # Generation parameters (consumed by tts_engine).
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None

    # Filled by tts_engine.
    output_codes_delayed: list[list[int]] | None = None
    """Multi-codebook codes produced by the engine, shape
    ``[num_steps, num_codebooks]`` serialised as nested list. The vocoder
    applies :func:`reverse_delay_pattern` before decoding."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0

    # Filled by vocoder.
    audio_samples: Any | None = None
    sample_rate: int = 24000

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "prompt_token_ids": list(self.prompt_token_ids),
            "num_codebooks": self.num_codebooks,
            "codebook_size": self.codebook_size,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        if self.reference_codes_delayed is not None:
            data["reference_codes_delayed"] = self.reference_codes_delayed
        if self.reference_audio_embed is not None:
            data["reference_audio_embed"] = self.reference_audio_embed
        for key in ("top_p", "top_k", "seed"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.output_codes_delayed is not None:
            data["output_codes_delayed"] = self.output_codes_delayed
        for key in ("prompt_tokens", "completion_tokens", "engine_time_s"):
            value = getattr(self, key)
            if value:
                data[key] = value
        if self.audio_samples is not None:
            data["audio_samples"] = self.audio_samples
            data["sample_rate"] = self.sample_rate
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HiggsTtsState:
        return cls(
            prompt_token_ids=list(data.get("prompt_token_ids", [])),
            reference_codes_delayed=data.get("reference_codes_delayed"),
            reference_audio_embed=data.get("reference_audio_embed"),
            num_codebooks=data.get("num_codebooks", 8),
            codebook_size=data.get("codebook_size", 1026),
            max_new_tokens=data.get("max_new_tokens", 2048),
            temperature=data.get("temperature", 1.0),
            top_p=data.get("top_p"),
            top_k=data.get("top_k"),
            seed=data.get("seed"),
            output_codes_delayed=data.get("output_codes_delayed"),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            engine_time_s=data.get("engine_time_s", 0.0),
            audio_samples=data.get("audio_samples"),
            sample_rate=data.get("sample_rate", 24000),
        )


__all__ = ["HiggsTtsState"]
