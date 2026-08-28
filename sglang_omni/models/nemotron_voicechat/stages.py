from __future__ import annotations

import json
from pathlib import Path

import torch
from einops import rearrange
from torch import nn

from sglang_omni.models.nemotron_voicechat.conformer import AudioPerception
from sglang_omni.models.nemotron_voicechat.engine_builder import NemotronVoiceChatEngineBuilder
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.models.weight_loader import load_module, resolve_dtype, resolve_model_path
from sglang_omni.preprocessing.transcription import prepare_audio
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.device import resolve_device_spec

PERCEPTION_PREFIX = "stt_model.perception."
SAMPLES_PER_FRAME = 1_280
INPUT_SAMPLE_RATE = 16_000


def _perception_config(model_path: str) -> dict:
    config_path = Path(resolve_model_path(model_path)) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config["model"]["stt"]["model"]["perception"]


def create_preprocessing_executor(model_path: str, **_):
    del model_path

    def preprocess(payload: StagePayload) -> StagePayload:
        prepared = prepare_audio(
            payload,
            source_name="VoiceChat",
            target_sample_rate=INPUT_SAMPLE_RATE,
        )
        waveform = torch.as_tensor(prepared.waveform, dtype=torch.float32)
        remainder = waveform.shape[-1] % SAMPLES_PER_FRAME
        if remainder:
            waveform = nn.functional.pad(waveform, (0, SAMPLES_PER_FRAME - remainder))

        state = NemotronVoiceChatState.from_dict(payload.data)
        state.waveform = waveform
        state.num_frames = waveform.shape[-1] // SAMPLES_PER_FRAME
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(preprocess)

def create_perception_executor(model_path: str, *, dtype=None, device=None):
    device = resolve_device_spec(device)
    module = AudioPerception(_perception_config(model_path))
    load_module(
        module,
        model_path,
        prefix=PERCEPTION_PREFIX,
        dtype=resolve_dtype(dtype),
        device=device,
        strict=True,
    )
    module.eval()
    parameter_dtype = module.proj.weight.dtype

    @torch.inference_mode()
    def encode(payload: StagePayload) -> StagePayload:
        state = NemotronVoiceChatState.from_dict(payload.data)
        waveform = state.waveform
        waveform_1S = rearrange(waveform, "s -> 1 s") if waveform.ndim == 1 else waveform

        frames = module(waveform_1S.to(device=device, dtype=parameter_dtype))
        assert frames.shape[1] == state.num_frames + 1, (
            f"Perception returned {frames.shape[1]} rows for {state.num_frames} "
            "frames of audio; expected one more than the frame count."
        )

        state.acoustic_frames = frames[0]
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(encode)

def create_thinker_executor(model_path, *, dtype=None, device=None, gpu_id=None, **overrides):
    builder = NemotronVoiceChatEngineBuilder(max_running_requests=1)
    return builder.build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype or "float32",
        server_args_overrides=overrides or None,
    )
