# SPDX-License-Identifier: Apache-2.0
"""Per-request pipeline state for HiggsMultimodalQwen3 TTS.

Carried between preprocessing → tts_engine → vocoder via
:class:`sglang_omni.proto.StagePayload.data` (JSON-serialisable dict).
Tensors are flattened to nested lists on the way out and rebuilt on the way
in, matching the :class:`S2ProState` convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class HiggsTtsState:
    """Per-request state for the Higgs TTS pipeline."""

    # -- Filled by preprocessing -----------------------------------------------
    prompt_token_ids: list[int] = field(default_factory=list)
    """Prompt ids with ``-100`` placeholders for ref-audio positions."""

    reference_codes_delayed: list[list[int]] | None = None
    """Delayed multi-codebook codes for the reference audio, shape
    ``[num_ref_tokens, num_codebooks]`` serialised as a nested list. Absent
    (``None``) for zero-shot (no reference audio)."""

    num_ref_tokens: int = 0
    """Count of ``-100`` placeholders in ``prompt_token_ids``. Equals
    ``reference_codes_delayed.shape[0]`` when present, else 0."""

    num_codebooks: int = 8
    codebook_size: int = 1026  # 1024 data + <|boc|> + <|eoc|>

    # -- Generation parameters -------------------------------------------------
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None

    # -- Filled by tts_engine (PR4) --------------------------------------------
    output_codes_delayed: list[list[int]] | None = None
    """Model-generated delayed codes, shape ``[T + num_codebooks - 1, num_codebooks]``."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0

    # -- Filled by vocoder (PR5) -----------------------------------------------
    audio_samples: list[float] | None = None
    sample_rate: int = 24000  # higgs-audio-v2-tokenizer native rate

    # -- (De)serialisation -----------------------------------------------------
    @staticmethod
    def _to_list(t: Any) -> Any:
        if isinstance(t, torch.Tensor):
            return t.tolist()
        return t

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "prompt_token_ids": list(self.prompt_token_ids),
            "num_ref_tokens": self.num_ref_tokens,
            "num_codebooks": self.num_codebooks,
            "codebook_size": self.codebook_size,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "sample_rate": self.sample_rate,
        }
        if self.reference_codes_delayed is not None:
            data["reference_codes_delayed"] = self._to_list(
                self.reference_codes_delayed
            )
        if self.top_p is not None:
            data["top_p"] = self.top_p
        if self.top_k is not None:
            data["top_k"] = self.top_k
        if self.seed is not None:
            data["seed"] = self.seed
        if self.output_codes_delayed is not None:
            data["output_codes_delayed"] = self._to_list(self.output_codes_delayed)
        if self.prompt_tokens:
            data["prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens:
            data["completion_tokens"] = self.completion_tokens
        if self.engine_time_s:
            data["engine_time_s"] = self.engine_time_s
        if self.audio_samples is not None:
            data["audio_samples"] = self._to_list(self.audio_samples)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HiggsTtsState:
        return cls(
            prompt_token_ids=list(data.get("prompt_token_ids", [])),
            reference_codes_delayed=data.get("reference_codes_delayed"),
            num_ref_tokens=data.get("num_ref_tokens", 0),
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
