# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Higgs TTS pipeline.

- ``create_preprocessing_executor`` (PR3a): text + optional pre-encoded
  reference codes → prompt ids with ``-100`` placeholders, wrapped in a
  :class:`HiggsTtsState` on the payload. Server-side audio → codec encoding
  lives in PR3b.
- ``create_sglang_tts_engine_executor`` (PR4, stub).
- ``create_vocoder_executor`` (PR5, stub).
"""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from sglang_omni.models.higgs_tts.delay_pattern import apply_delay_pattern
from sglang_omni.models.higgs_tts.io import HiggsTtsState
from sglang_omni.models.higgs_tts.pipeline.state_io import store_state
from sglang_omni.models.higgs_tts.tokenizer import HiggsTokenizerAdapter
from sglang_omni.proto import StagePayload


def _to_codes_TN(raw: Any, num_codebooks: int) -> torch.Tensor | None:
    """Coerce request input to an ``[T, num_codebooks]`` int64 tensor."""
    if raw is None:
        return None
    t = raw if isinstance(raw, torch.Tensor) else torch.tensor(raw)
    if t.numel() == 0:
        return None
    if t.ndim != 2 or t.shape[1] != num_codebooks:
        raise ValueError(
            f"reference_codes must have shape [T, {num_codebooks}], got {tuple(t.shape)}"
        )
    return t.to(torch.long)


def build_preprocess_fn(
    adapter: HiggsTokenizerAdapter,
    *,
    num_codebooks: int,
    codebook_size: int,
):
    """Return the ``(payload) -> payload`` closure used by the preprocessing
    stage. Exposed so tests can drive it with a stub tokenizer adapter."""

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}

        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("input") or inputs.get("text") or ""
        ref_codes_TN = _to_codes_TN(inputs.get("reference_codes"), num_codebooks)

        if ref_codes_TN is None:
            prompt_ids = adapter.build_prompt(text, num_ref_tokens=0)
            ref_codes_delayed: list[list[int]] | None = None
        else:
            delayed = apply_delay_pattern(ref_codes_TN)
            prompt_ids = adapter.build_prompt(text, num_ref_tokens=delayed.shape[0])
            ref_codes_delayed = delayed.tolist()

        state = HiggsTtsState(
            prompt_token_ids=prompt_ids,
            reference_codes_delayed=ref_codes_delayed,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            max_new_tokens=int(params.get("max_new_tokens", 2048)),
            temperature=float(params.get("temperature", 1.0)),
            top_p=params.get("top_p"),
            top_k=params.get("top_k"),
            seed=params.get("seed"),
        )
        return store_state(payload, state)

    return _preprocess


def create_preprocessing_executor(
    model_path: str,
    *,
    num_codebooks: int = 8,
    codebook_size: int = 1026,
) -> PreprocessingExecutor:
    """Build the Higgs TTS preprocessing stage.

    ``model_path`` is a Higgs checkpoint directory or an HF repo id — passed
    straight to ``PreTrainedTokenizerFast.from_pretrained``.
    """
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    adapter = HiggsTokenizerAdapter(tokenizer)
    return PreprocessingExecutor(
        build_preprocess_fn(
            adapter, num_codebooks=num_codebooks, codebook_size=codebook_size
        )
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    max_new_tokens: int = 2048,
) -> EngineExecutor:
    del model_path, device, max_new_tokens
    raise NotImplementedError(
        "Higgs TTS sglang engine stage is not implemented yet (planned for PR4)."
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cpu",
) -> PreprocessingExecutor:
    del model_path, device
    raise NotImplementedError(
        "Higgs TTS vocoder stage is not implemented yet (planned for PR5)."
    )
