# SPDX-License-Identifier: Apache-2.0
"""Per-request data + StagePayload <-> scheduler adapters for Higgs TTS (V1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData


@dataclass
class HiggsSGLangRequestData(SGLangARRequestData):
    """Per-request state for the Higgs TTS scheduler.

    Carries the precomputed reference-audio embedding from the
    ``audio_encoder`` stage plus the running list of multi-codebook codes
    emitted by :class:`HiggsTTSModel`.
    """

    reference_audio_embed: torch.Tensor | None = None
    """Pre-computed fused audio embedding, shape ``[num_ref_tokens,
    hidden_size]``. Pasted at ``-100`` placeholder positions in the
    scheduler's prepare_prefill hook."""

    num_ref_codes_consumed: int = 0
    """Count of rows from ``reference_audio_embed`` already overlaid onto
    earlier prefill chunks. Advanced by the model runner so chunked prefill
    slices the embed tensor correctly across multiple extend calls."""

    num_codebooks: int = 8
    codebook_size: int = 1026

    output_codes: list[torch.Tensor] = field(default_factory=list)
    """One ``[num_codebooks]`` long tensor per AR step, captured from
    :class:`HiggsTTSModel`'s per-request slot in ``post_prefill`` /
    ``post_decode``."""

    generation_done: bool = False


def _to_tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if value.dtype != dtype else value
    return torch.tensor(value, dtype=dtype)


def build_sglang_higgs_request(
    state: HiggsTtsState, *, request_id: str = ""
) -> HiggsSGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    input_ids_list = list(state.prompt_token_ids)
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    sp_kwargs: dict[str, Any] = {
        "max_new_tokens": int(state.max_new_tokens),
        "temperature": float(state.temperature),
    }
    if state.top_p is not None:
        sp_kwargs["top_p"] = float(state.top_p)
    if state.top_k is not None:
        sp_kwargs["top_k"] = int(state.top_k)
    sampling_params = SamplingParams(**sp_kwargs)

    # ``Req`` consumes the int vocab size for codebook-0 padding tracking;
    # use the backbone text vocab so primary cb0 can be threaded through
    # sglang's regular sampling infra.
    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=151_936,
    )
    # V1's prefill manager probes these attrs on the Req. Fish sets them too;
    # without them the scheduler crashes on the first prefill cycle with
    # ``AttributeError: '_input_embeds_are_projected'``.
    req._codec_suppress_tokens = None
    req._input_embeds_are_projected = False

    return HiggsSGLangRequestData(
        input_ids=input_ids,
        req=req,
        reference_audio_embed=_to_tensor(state.reference_audio_embed),
        num_codebooks=int(state.num_codebooks),
        codebook_size=int(state.codebook_size),
        max_new_tokens=int(state.max_new_tokens),
        temperature=float(state.temperature),
        top_p=float(state.top_p) if state.top_p is not None else 1.0,
        top_k=int(state.top_k) if state.top_k is not None else -1,
    )


def apply_higgs_result(state: HiggsTtsState, data: HiggsSGLangRequestData) -> None:
    if data.output_codes:
        codes = torch.stack(data.output_codes, dim=0).to(torch.long)
        state.output_codes_delayed = codes.tolist()
        state.completion_tokens = int(codes.shape[0])
    else:
        state.output_codes_delayed = None
    state.prompt_tokens = len(data.input_ids) if data.input_ids is not None else 0


def make_higgs_scheduler_adapters():
    """Build StagePayload <-> scheduler adapters for Higgs TTS."""

    def request_builder(payload: StagePayload) -> HiggsSGLangRequestData:
        state = HiggsTtsState.from_dict(payload.data)
        data = build_sglang_higgs_request(state, request_id=payload.request_id)
        data.stage_payload = payload
        return data

    def result_adapter(data: HiggsSGLangRequestData) -> StagePayload:
        payload = data.stage_payload
        state = HiggsTtsState.from_dict(payload.data)
        apply_higgs_result(state, data)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter


__all__ = [
    "HiggsSGLangRequestData",
    "apply_higgs_result",
    "build_sglang_higgs_request",
    "make_higgs_scheduler_adapters",
]
