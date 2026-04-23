# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for HiggsMultimodalQwen3 TTS.

Provides the ``PipelineConfig`` subclass used by the registry to resolve
``architectures[0] == "HiggsMultimodalQwen3ForConditionalGeneration"`` in the
HuggingFace config.json.

Codec wiring: preprocessing (encode ref audio) and vocoder (decode
output codes) both need the Higgs Audio V2 tokenizer. The TTS ckpt
ships the codec weights inline (under
``tied.embedding.modality_embeddings.0.model.*``), so by default both
stages load the codec directly from ``model_path``. This means a user
only needs **one checkpoint directory** to serve Higgs TTS.

Override with ``HIGGS_AUDIO_CODEC_PATH`` (env var) to point at a
separate standalone codec ckpt — e.g. for A/B testing against a
different codec release. If unset, and the TTS ckpt doesn't have the
inline codec weights (older ckpts), fall back to the upstream HF id.
"""

from __future__ import annotations

import os
from typing import ClassVar

from sglang_omni.config import ExecutorConfig, PipelineConfig, RelayConfig, StageConfig
from sglang_omni.models.higgs_tts.pipeline.next_stage import (
    AGGREGATE_STAGE,
    AUDIO_ENCODER_STAGE,
    PREPROCESSING_STAGE,
    TTS_ENGINE_STAGE,
    VOCODER_STAGE,
)

_HIGGS_PKG = "sglang_omni.models.higgs_tts.pipeline"
_DEFAULT_CODEC_HF_ID = "bosonai/higgs-audio-v2-tokenizer"

# ``None`` means "load from ``model_path``" (the TTS ckpt). The
# ``_get_or_load_codec`` helper in stages.py auto-detects whether a
# given path is a TTS ckpt or a standalone codec. An operator can
# override with the env var to force a different codec source.
_CODEC_OVERRIDE = os.environ.get("HIGGS_AUDIO_CODEC_PATH") or None


class HiggsTtsPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "HiggsMultimodalQwen3ForConditionalGeneration"

    model_path: str
    entry_stage: str = PREPROCESSING_STAGE
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory=f"{_HIGGS_PKG}.stages.create_preprocessing_executor",
                args={
                    "audio_codec_path": _CODEC_OVERRIDE,
                    "audio_codec_device": "cpu",
                },
            ),
            get_next=f"{_HIGGS_PKG}.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=AUDIO_ENCODER_STAGE,
            executor=ExecutorConfig(
                factory=f"{_HIGGS_PKG}.stages.create_audio_encoder_executor",
                args={"device": "cpu"},
            ),
            get_next=f"{_HIGGS_PKG}.next_stage.audio_encoder_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=AGGREGATE_STAGE,
            executor=ExecutorConfig(
                factory=f"{_HIGGS_PKG}.stages.create_aggregate_executor",
            ),
            get_next=f"{_HIGGS_PKG}.next_stage.aggregate_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=TTS_ENGINE_STAGE,
            executor=ExecutorConfig(
                factory=f"{_HIGGS_PKG}.stages.create_sglang_tts_engine_executor",
                args={
                    "device": "cuda:0",
                    "max_new_tokens": 2048,
                },
            ),
            get_next=f"{_HIGGS_PKG}.next_stage.tts_engine_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=VOCODER_STAGE,
            executor=ExecutorConfig(
                factory=f"{_HIGGS_PKG}.stages.create_vocoder_executor",
                args={
                    "audio_codec_path": _CODEC_OVERRIDE,
                    "device": "cpu",
                },
            ),
            get_next=f"{_HIGGS_PKG}.next_stage.vocoder_next",
            relay=RelayConfig(device="cpu"),
        ),
    ]


EntryClass = HiggsTtsPipelineConfig
