# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for HiggsMultimodalQwen3 TTS.

Provides the ``PipelineConfig`` subclass used by the registry to resolve
``architectures[0] == "HiggsMultimodalQwen3ForConditionalGeneration"`` in the
HuggingFace config.json. Stage factories currently raise ``NotImplementedError``
(PR1 scaffolding) and will be implemented in follow-up PRs.
"""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import ExecutorConfig, PipelineConfig, RelayConfig, StageConfig
from sglang_omni.models.higgs_tts.pipeline.next_stage import (
    PREPROCESSING_STAGE,
    TTS_ENGINE_STAGE,
    VOCODER_STAGE,
)

_HIGGS_PKG = "sglang_omni.models.higgs_tts.pipeline"


class HiggsTtsPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "HiggsMultimodalQwen3ForConditionalGeneration"

    model_path: str
    entry_stage: str = PREPROCESSING_STAGE
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory=f"{_HIGGS_PKG}.stages.create_preprocessing_executor",
            ),
            get_next=f"{_HIGGS_PKG}.next_stage.preprocessing_next",
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
            ),
            get_next=f"{_HIGGS_PKG}.next_stage.vocoder_next",
            relay=RelayConfig(device="cpu"),
        ),
    ]


EntryClass = HiggsTtsPipelineConfig
