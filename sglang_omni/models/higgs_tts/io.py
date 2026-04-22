# SPDX-License-Identifier: Apache-2.0
"""Per-request pipeline state for Higgs TTS preprocessing output.

Carried between stages via :class:`sglang_omni.proto.StagePayload.data`. Only
fields produced by PR3a preprocessing + consumed by PR4 engine are defined
here; PR4/PR5 will extend this as their outputs land.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HiggsTtsState:
    prompt_token_ids: list[int] = field(default_factory=list)
    """Prompt ids with ``AUDIO_PLACEHOLDER_ID`` (-100) at ref-audio positions."""

    reference_codes_delayed: list[list[int]] | None = None
    """Delayed ref-audio codes, shape ``[num_ref_tokens, num_codebooks]`` as a
    nested list. ``None`` for zero-shot."""

    num_codebooks: int = 8
    codebook_size: int = 1026  # 1024 data + <|boc|> + <|eoc|>

    # Generation parameters (consumed by PR4).
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None

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
        for key in ("top_p", "top_k", "seed"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HiggsTtsState:
        return cls(
            prompt_token_ids=list(data.get("prompt_token_ids", [])),
            reference_codes_delayed=data.get("reference_codes_delayed"),
            num_codebooks=data.get("num_codebooks", 8),
            codebook_size=data.get("codebook_size", 1026),
            max_new_tokens=data.get("max_new_tokens", 2048),
            temperature=data.get("temperature", 1.0),
            top_p=data.get("top_p"),
            top_k=data.get("top_k"),
            seed=data.get("seed"),
        )


__all__ = ["HiggsTtsState"]
