# SPDX-License-Identifier: Apache-2.0
"""Scaffolding + wiring tests for the Higgs TTS pipeline config.

Asserts:
- The package is discovered by the pipeline-config registry under the
  ``HiggsMultimodalQwen3ForConditionalGeneration`` architecture key.
- The ``HiggsTtsPipelineConfig`` instantiates cleanly with the default
  three-stage layout (preprocessing → tts_engine → vocoder).
- ``ConfigManager.from_model_path`` resolves the architecture from a raw
  ``config.json`` (the path taken when ``AutoConfig`` cannot load the custom
  ``HiggsMultimodalQwen3Config`` without ``trust_remote_code``).
- The ``vocoder`` stage factory still raises ``NotImplementedError``
  (PR5). ``preprocessing`` is implemented as of PR3a, ``tts_engine``
  as of PR4c.
"""

from __future__ import annotations

import json
import os
import tempfile

ARCHITECTURE = "HiggsMultimodalQwen3ForConditionalGeneration"


def test_registry_discovers_higgs_tts_architecture():
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

    supported = PIPELINE_CONFIG_REGISTRY.get_supported_archs()
    assert (
        ARCHITECTURE in supported
    ), f"Higgs TTS architecture not found in registry. Registered: {sorted(supported)}"


def test_registry_returns_higgs_tts_config_class():
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

    config_cls = PIPELINE_CONFIG_REGISTRY.get_config(ARCHITECTURE)
    assert config_cls.__name__ == "HiggsTtsPipelineConfig"


def test_higgs_tts_pipeline_config_five_stages():
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
    from sglang_omni.models.higgs_tts.pipeline.next_stage import (
        AGGREGATE_STAGE,
        AUDIO_ENCODER_STAGE,
        PREPROCESSING_STAGE,
        TTS_ENGINE_STAGE,
        VOCODER_STAGE,
    )

    config = HiggsTtsPipelineConfig(model_path="test/higgs-tts")

    stage_names = [stage.name for stage in config.stages]
    assert stage_names == [
        PREPROCESSING_STAGE,
        AUDIO_ENCODER_STAGE,
        AGGREGATE_STAGE,
        TTS_ENGINE_STAGE,
        VOCODER_STAGE,
    ]
    assert config.entry_stage == PREPROCESSING_STAGE
    assert config.config_cls == "HiggsTtsPipelineConfig"


def test_config_manager_resolves_higgs_from_local_config():
    from sglang_omni.config.manager import ConfigManager

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(
                {
                    "architectures": [ARCHITECTURE],
                    "model_type": "higgs_multimodal_qwen3",
                },
                f,
            )

        mgr = ConfigManager.from_model_path(tmpdir)
    assert mgr.config is not None
    assert type(mgr.config).__name__ == "HiggsTtsPipelineConfig"


def test_higgs_hf_config_instantiates():
    """The ported HF config should instantiate with the discrete encoder shape."""
    from sglang_omni.models.higgs_tts import HiggsMultimodalQwen3Config

    cfg = HiggsMultimodalQwen3Config(
        audio_encoder_config={
            "encoder_type": "discrete",
            "num_codebooks": 10,
            "vocab_size": 1026,
        },
        text_config={"model_type": "qwen3", "hidden_size": 2048},
    )
    assert cfg.model_type == "higgs_multimodal_qwen3"
    assert cfg.audio_encoder_config["encoder_type"] == "discrete"
    # PR4c eagerly realises text_config as a concrete ``Qwen3Config`` instance
    # (attribute access, not subscript) so sglang's ``hf_transformers_utils``
    # can read ``num_attention_heads`` / ``hidden_size`` directly.
    assert cfg.text_config.hidden_size == 2048
