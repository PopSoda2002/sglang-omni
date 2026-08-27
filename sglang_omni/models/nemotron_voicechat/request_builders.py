from __future__ import annotations

from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

PAD_TOKEN_ID = 12
BOS_TOKEN_ID = 1

def build_thinker_request(payload: StagePayload) -> SGLangARRequestData:
    state = NemotronVoiceChatState.from_dict(payload.data)
    num_frames = state.num_frames
    input_ids = [BOS_TOKEN_ID] + [PAD_TOKEN_ID] * (num_frames - 1)
    sampling_params = SamplingParams(
        max_new_tokens=num_frames,
        temperature=0.0,
        ignore_eos=True,
    )
    sampling_params.normalize(tokenizer=None)
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=input_ids,
        sampling_params=sampling_params,
    )
    return SGLangARRequestData(
        req=req,
        input_ids=state.acoustic_frames.new_tensor(input_ids, dtype=None),
        stage_payload=payload,
        max_new_tokens=num_frames,
        temperature=0.0,
    )

def apply_thinker_result(data: SGLangARRequestData) -> StagePayload:
    payload = data.stage_payload
    state = NemotronVoiceChatState.from_dict(payload.data)
    state.text_ids = list(data.output_ids)
    state.function_ids = list(data.extra_model_outputs.get("function_ids", ()))
    payload.data = state.to_dict()
    return payload


