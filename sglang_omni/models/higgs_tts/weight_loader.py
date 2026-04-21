# SPDX-License-Identifier: Apache-2.0
"""Checkpoint weight-name remapping for HiggsMultimodalQwen3 (discrete TTS path).

Ported from boson-vllm's ``_map_higgs_to_vllm_name`` (discrete branch only).
Higgs checkpoints use prefixes like ``tied.embedding.text_embedding.`` and
``body.layers.``; downstream frameworks (vLLM, sglang) expect their own
parameter tree layout. The mapping function is parameterised by the
destination prefix so the PR2b sglang model and a future vLLM-style loader
can reuse the same rules.

The public surface is intentionally narrow: one module-level constant
(``HIGGS_TEXT_PREFIX_MAP``) plus :func:`map_higgs_discrete_weight_name`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Prefix substitutions that always apply to a Higgs checkpoint, regardless of
# whether the downstream framework is vLLM or sglang. The values use vLLM's
# nested ``language_model.model.*`` layout; callers with a different layout
# pass a custom ``text_prefix_map`` to :func:`map_higgs_discrete_weight_name`.
HIGGS_TEXT_PREFIX_MAP_VLLM: dict[str, str] = {
    "tied.embedding.text_embedding.": "language_model.model.embed_tokens.",
    "body.layers.": "language_model.model.layers.",
    "body.norm.": "language_model.model.norm.",
    "tied.head.text_head.": "language_model.lm_head.",
}

# sglang-native flat layout (matches the pattern used by fishaudio_s2_pro's
# ``S2ProSGLangTextModel``: no ``language_model.`` wrapper, ``lm_head`` at root
# when not tied, ``embed_tokens``/``layers``/``norm`` at root). PR2b will
# instantiate the loader with this map.
HIGGS_TEXT_PREFIX_MAP_SGLANG: dict[str, str] = {
    "tied.embedding.text_embedding.": "embed_tokens.",
    "body.layers.": "layers.",
    "body.norm.": "norm.",
    "tied.head.text_head.": "lm_head.",
}


@dataclass(frozen=True)
class DiscreteWeightMapper:
    """Weight name remapper for the discrete TTS path.

    Args:
        text_prefix_map: Maps Higgs prefixes for the text backbone to the
            downstream parameter tree. See ``HIGGS_TEXT_PREFIX_MAP_*`` above.
        embedding_dest: Destination prefix for
            ``tied.embedding.modality_embeddings.0.embedding.*`` (the fused
            multi-codebook embedding weight).
        head_dest: Destination prefix for ``tied.head.modality_heads.0.*`` (the
            fused multi-codebook head). Ignored when ``tie_modality=True``
            (the head shares the embedding weight).
        tie_modality: Whether the modality head weight is tied to the
            modality embedding weight. Must match the checkpoint's
            ``audio_encoder_config.tie_word_embeddings`` flag.
    """

    text_prefix_map: dict[str, str]
    embedding_dest: str = "multimodal_embedding.modality_embedding_0."
    head_dest: str = "modality_head."
    tie_modality: bool = True

    def _instance_prefix_map(self) -> dict[str, str]:
        """Build the per-instance prefix map (discrete-mode only)."""
        mapping = {
            "tied.embedding.modality_embeddings.0.embedding.": self.embedding_dest,
        }
        if not self.tie_modality:
            mapping["tied.head.modality_heads.0."] = self.head_dest
        return mapping

    def map(self, name: str) -> str | None:
        """Map a Higgs checkpoint parameter name to the downstream name.

        Returns ``None`` when the weight should be skipped (e.g., the audio
        tokenizer's backbone weights that live inside the checkpoint but are
        not part of the inference graph).
        """
        # Instance-specific mappings take priority.
        for higgs_prefix, dest_prefix in self._instance_prefix_map().items():
            if name.startswith(higgs_prefix):
                return dest_prefix + name[len(higgs_prefix) :]

        # Audio tokenizer backbone (frozen, not in the serving graph).
        if name.startswith("tied.embedding.modality_embeddings.0.model."):
            return None

        # Text backbone.
        for higgs_prefix, dest_prefix in self.text_prefix_map.items():
            if name.startswith(higgs_prefix):
                return dest_prefix + name[len(higgs_prefix) :]

        return name


def map_higgs_discrete_weight_name(
    name: str,
    *,
    text_prefix_map: dict[str, str] = HIGGS_TEXT_PREFIX_MAP_SGLANG,
    tie_modality: bool = True,
) -> str | None:
    """Convenience wrapper around :class:`DiscreteWeightMapper` for one-shot use."""
    return DiscreteWeightMapper(
        text_prefix_map=text_prefix_map,
        tie_modality=tie_modality,
    ).map(name)


__all__ = [
    "DiscreteWeightMapper",
    "HIGGS_TEXT_PREFIX_MAP_SGLANG",
    "HIGGS_TEXT_PREFIX_MAP_VLLM",
    "map_higgs_discrete_weight_name",
]
