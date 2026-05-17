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
    """Per-request state for the Higgs TTS scheduler."""

    reference_codes_delayed: list[list[int]] | None = None
    num_ref_codes_consumed: int = 0
    num_codebooks: int = 8
    codebook_size: int = 1026
    output_codes: list[torch.Tensor] = field(default_factory=list)
    generation_done: bool = False


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

    # vocab_size = backbone text vocab so cb0 rides sglang's standard sampler path.
    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=151_936,
    )
    # V1's prefill manager probes these attrs; absence triggers AttributeError.
    req._codec_suppress_tokens = None
    req._input_embeds_are_projected = False

    return HiggsSGLangRequestData(
        input_ids=input_ids,
        req=req,
        reference_codes_delayed=state.reference_codes_delayed,
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
