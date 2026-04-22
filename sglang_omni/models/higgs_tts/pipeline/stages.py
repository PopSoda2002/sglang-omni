# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Higgs TTS pipeline.

- ``create_preprocessing_executor`` (PR3a): text + optional pre-encoded
  reference codes → prompt ids with ``-100`` placeholders, wrapped in a
  :class:`HiggsTtsState` on the payload.
- ``create_sglang_tts_engine_executor`` (PR4, stub): sglang engine + custom
  multi-codebook sampler.
- ``create_vocoder_executor`` (PR5, stub): decode multi-codebook codes to WAV.

PR3a accepts the reference audio as **pre-encoded** codes (``reference_codes``
in the request inputs). Server-side audio → codec encoding lives in PR3b.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from sglang_omni.models.higgs_tts.delay_pattern import (
    apply_delay_pattern,
    delayed_length,
)
from sglang_omni.models.higgs_tts.io import HiggsTtsState
from sglang_omni.models.higgs_tts.pipeline.state_io import store_state
from sglang_omni.models.higgs_tts.tokenizer import (
    CODEBOOK_BOC_ID,
    CODEBOOK_EOC_ID,
    HiggsTokenizerAdapter,
)
from sglang_omni.proto import StagePayload


def _resolve_checkpoint(model_path: str) -> str:
    """Accept either a local dir or an HF repo id (already cached)."""
    if os.path.isdir(model_path):
        return model_path
    # Leave Hub resolution to ``PreTrainedTokenizerFast.from_pretrained``.
    return model_path


def _coerce_reference_codes(raw: Any, num_codebooks: int) -> torch.Tensor | None:
    """Normalise request ``reference_codes`` input to ``[T, num_codebooks]`` int64.

    Accepted shapes: ``[T, N]`` (preferred) or ``[N, T]`` (matches
    ``HiggsAudioV2Tokenizer._encode`` layout with a batch-squeeze). Returns
    ``None`` if ``raw`` is missing / empty.
    """
    if raw is None:
        return None
    tensor = raw if isinstance(raw, torch.Tensor) else torch.tensor(raw)
    if tensor.numel() == 0:
        return None
    if tensor.ndim != 2:
        raise ValueError(
            f"reference_codes must be 2-D, got shape {tuple(tensor.shape)}"
        )
    if tensor.shape[1] == num_codebooks:
        codes_TN = tensor
    elif tensor.shape[0] == num_codebooks:
        codes_TN = tensor.transpose(0, 1).contiguous()
    else:
        raise ValueError(
            f"reference_codes shape {tuple(tensor.shape)} does not align with "
            f"num_codebooks={num_codebooks}; expected [T, {num_codebooks}] or "
            f"[{num_codebooks}, T]"
        )
    return codes_TN.to(torch.long)


def build_preprocess_fn(
    adapter: HiggsTokenizerAdapter,
    *,
    num_codebooks: int,
    codebook_size: int,
):
    """Return the ``(payload) -> payload`` closure used by the preprocessing
    stage. Exposed so tests can build the closure with a stub adapter without
    loading a real HF tokenizer.
    """

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}

        # The /v1/audio/speech endpoint sends the text as a plain string.
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("input") or inputs.get("text") or ""

        # Reference audio: PR3a expects pre-encoded codes. PR3b will add
        # server-side encoding from raw audio / audio_path.
        ref_codes_TN = _coerce_reference_codes(
            inputs.get("reference_codes"), num_codebooks
        )

        if ref_codes_TN is None:
            prompt_ids = adapter.build_prompt(text, num_ref_tokens=0)
            ref_codes_delayed_list: list[list[int]] | None = None
            num_ref_tokens = 0
        else:
            valid_len = int(ref_codes_TN.shape[0])
            delayed = apply_delay_pattern(
                ref_codes_TN,
                valid_len=valid_len,
                boc_id=CODEBOOK_BOC_ID,
                eoc_id=CODEBOOK_EOC_ID,
            )
            num_ref_tokens = delayed_length(valid_len, num_codebooks)
            assert delayed.shape == (num_ref_tokens, num_codebooks)
            prompt_ids = adapter.build_prompt(text, num_ref_tokens=num_ref_tokens)
            ref_codes_delayed_list = delayed.tolist()

        state = HiggsTtsState(
            prompt_token_ids=prompt_ids,
            reference_codes_delayed=ref_codes_delayed_list,
            num_ref_tokens=num_ref_tokens,
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

    Args:
        model_path: Higgs checkpoint directory (or HF repo id) — used to
            load the text tokenizer.
        num_codebooks: Audio encoder's ``num_codebooks``. Must match the
            checkpoint's ``audio_encoder_config.num_codebooks`` (default 8).
        codebook_size: Codebook vocab size (data + boc + eoc; default 1026).
    """
    from transformers import PreTrainedTokenizerFast

    checkpoint_dir = _resolve_checkpoint(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_dir)
    adapter = HiggsTokenizerAdapter(tokenizer)

    preprocess_fn = build_preprocess_fn(
        adapter,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
    )
    return PreprocessingExecutor(preprocess_fn)


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    max_new_tokens: int = 2048,
) -> EngineExecutor:
    """TODO(PR4): wrap sglang engine + multi-codebook sampler."""
    del model_path, device, max_new_tokens
    raise NotImplementedError(
        "Higgs TTS sglang engine stage is not implemented yet (planned for PR4)."
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cpu",
) -> PreprocessingExecutor:
    """TODO(PR5): decode multi-codebook tokens → waveform via higgs-audio tokenizer."""
    del model_path, device
    raise NotImplementedError(
        "Higgs TTS vocoder stage is not implemented yet (planned for PR5)."
    )
