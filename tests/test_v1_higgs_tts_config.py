# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the V1 Higgs TTS package (no-GPU)."""

from __future__ import annotations

from sglang_omni_v1.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni_v1.models.higgs_tts.hf_config import HiggsMultimodalQwen3Config

_TINY_TEXT_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 1024,
    "rope_theta": 1_000_000.0,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 128,
    "tie_word_embeddings": True,
}


def test_pipeline_config_resolves_stages() -> None:
    cfg = HiggsTtsPipelineConfig(model_path="/some/where")
    assert cfg.architecture == "HiggsMultimodalQwen3ForConditionalGeneration"
    assert [s.name for s in cfg.stages] == [
        "preprocessing",
        "audio_encoder",
        "tts_engine",
        "vocoder",
    ]
    assert cfg.resolved_entry_stage == "preprocessing"
    assert cfg.terminal_stages == ["vocoder"]


def test_pipeline_config_is_discoverable() -> None:
    """``import_pipeline_configs`` finds the Higgs entry from the V1 registry."""
    from sglang_omni_v1.models.registry import import_pipeline_configs

    cfgs = import_pipeline_configs("sglang_omni_v1.models", "config")
    assert "HiggsMultimodalQwen3ForConditionalGeneration" in cfgs
    assert (
        cfgs["HiggsMultimodalQwen3ForConditionalGeneration"] is HiggsTtsPipelineConfig
    )


def test_text_config_patches_null_rope_theta_to_qwen3_default() -> None:
    """Higgs ckpts ship ``rope_theta: null`` in the text sub-config; the V1
    config must rewrite it to the Qwen3-trained value of ``1e6`` before
    handing off to ``Qwen3Config``. Direct port of the V0 regression test;
    the original bug was the same on both stacks (both build
    ``Qwen3Config(**dict)`` and inherit transformers' wrong default of
    ``10000``)."""
    null_rope = dict(_TINY_TEXT_CONFIG)
    null_rope["rope_theta"] = None

    cfg = HiggsMultimodalQwen3Config(
        audio_encoder_config={
            "encoder_type": "discrete",
            "num_codebooks": 4,
            "vocab_size": 16,
            "out_dim": 64,
            "tie_word_embeddings": True,
        },
        text_config=null_rope,
    )
    assert cfg.get_text_config().rope_theta == 1_000_000


def test_payload_state_round_trip() -> None:
    from sglang_omni_v1.models.higgs_tts.payload_types import HiggsTtsState

    state = HiggsTtsState(
        prompt_token_ids=[1, 2, -100, -100, 5],
        reference_codes_delayed=[[0, 1, 2, 3, 4, 5, 6, 7]] * 3,
        num_codebooks=8,
        codebook_size=1026,
        max_new_tokens=512,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        seed=42,
    )
    round_tripped = HiggsTtsState.from_dict(state.to_dict())
    assert round_tripped.prompt_token_ids == state.prompt_token_ids
    assert round_tripped.reference_codes_delayed == state.reference_codes_delayed
    assert round_tripped.num_codebooks == state.num_codebooks
    assert round_tripped.max_new_tokens == state.max_new_tokens
    assert round_tripped.top_p == state.top_p
    assert round_tripped.seed == state.seed
