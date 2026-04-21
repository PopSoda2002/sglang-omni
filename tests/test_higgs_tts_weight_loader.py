# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Higgs discrete-TTS weight name mapping (PR2a).

Covers the prefixes listed in boson-vllm
(`vllm/model_executor/models/higgs_multimodal_qwen3.py:902-907, 953-972`):
text backbone (embed_tokens / layers / norm / lm_head), discrete modality
embedding, modality head (tied + untied), and the skip rule for the audio
tokenizer backbone.
"""

from __future__ import annotations

import pytest

from sglang_omni.models.higgs_tts.weight_loader import (
    HIGGS_TEXT_PREFIX_MAP_SGLANG,
    HIGGS_TEXT_PREFIX_MAP_VLLM,
    DiscreteWeightMapper,
    map_higgs_discrete_weight_name,
)


class TestTextBackboneMapping:
    @pytest.mark.parametrize(
        "src,dst",
        [
            (
                "tied.embedding.text_embedding.weight",
                "language_model.model.embed_tokens.weight",
            ),
            (
                "body.layers.0.self_attn.q_proj.weight",
                "language_model.model.layers.0.self_attn.q_proj.weight",
            ),
            ("body.norm.weight", "language_model.model.norm.weight"),
            ("tied.head.text_head.weight", "language_model.lm_head.weight"),
        ],
    )
    def test_vllm_destination(self, src, dst):
        mapper = DiscreteWeightMapper(text_prefix_map=HIGGS_TEXT_PREFIX_MAP_VLLM)
        assert mapper.map(src) == dst

    @pytest.mark.parametrize(
        "src,dst",
        [
            ("tied.embedding.text_embedding.weight", "embed_tokens.weight"),
            (
                "body.layers.15.mlp.gate_up_proj.weight",
                "layers.15.mlp.gate_up_proj.weight",
            ),
            ("body.norm.weight", "norm.weight"),
            ("tied.head.text_head.weight", "lm_head.weight"),
        ],
    )
    def test_sglang_destination(self, src, dst):
        mapper = DiscreteWeightMapper(text_prefix_map=HIGGS_TEXT_PREFIX_MAP_SGLANG)
        assert mapper.map(src) == dst


class TestModalityEmbedding:
    def test_fused_embedding_weight_is_remapped(self):
        mapper = DiscreteWeightMapper(text_prefix_map=HIGGS_TEXT_PREFIX_MAP_SGLANG)
        assert (
            mapper.map("tied.embedding.modality_embeddings.0.embedding.weight")
            == "multimodal_embedding.modality_embedding_0.weight"
        )

    def test_custom_embedding_destination(self):
        mapper = DiscreteWeightMapper(
            text_prefix_map=HIGGS_TEXT_PREFIX_MAP_SGLANG,
            embedding_dest="audio_tok_embed.",
        )
        assert (
            mapper.map("tied.embedding.modality_embeddings.0.embedding.weight")
            == "audio_tok_embed.weight"
        )

    def test_audio_tokenizer_backbone_is_skipped(self):
        """Checkpoints may ship the frozen audio tokenizer backbone; we ignore it."""
        mapper = DiscreteWeightMapper(text_prefix_map=HIGGS_TEXT_PREFIX_MAP_SGLANG)
        assert (
            mapper.map(
                "tied.embedding.modality_embeddings.0.model.encoder.blocks.0.weight"
            )
            is None
        )


class TestModalityHead:
    def test_tied_head_has_no_dedicated_mapping(self):
        """With ``tie_modality=True``, head weights do not appear under
        ``tied.head.modality_heads.0.*``; nothing to remap."""
        mapper = DiscreteWeightMapper(
            text_prefix_map=HIGGS_TEXT_PREFIX_MAP_SGLANG,
            tie_modality=True,
        )
        assert (
            mapper.map("tied.head.modality_heads.0.weight")
            == "tied.head.modality_heads.0.weight"
        )

    def test_untied_head_remapped_to_modality_head(self):
        mapper = DiscreteWeightMapper(
            text_prefix_map=HIGGS_TEXT_PREFIX_MAP_SGLANG,
            tie_modality=False,
        )
        assert mapper.map("tied.head.modality_heads.0.weight") == "modality_head.weight"


class TestUnknownPrefix:
    def test_unknown_name_passes_through(self):
        mapper = DiscreteWeightMapper(text_prefix_map=HIGGS_TEXT_PREFIX_MAP_SGLANG)
        assert mapper.map("some.unknown.param") == "some.unknown.param"


class TestConvenienceFunction:
    def test_defaults_to_sglang_destination(self):
        assert map_higgs_discrete_weight_name("body.norm.weight") == "norm.weight"

    def test_tie_modality_false(self):
        assert (
            map_higgs_discrete_weight_name(
                "tied.head.modality_heads.0.weight", tie_modality=False
            )
            == "modality_head.weight"
        )
