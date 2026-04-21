# SPDX-License-Identifier: Apache-2.0
"""sglang-native Higgs Multimodal Qwen3 TTS model (PR2b).

Wraps sglang's built-in :class:`sglang.srt.models.qwen3.Qwen3ForCausalLM` as
the text backbone and attaches the fused multi-codebook embedding and head
from PR2a. The text forward path is fully functional; multi-codebook
embedding insertion into ``input_embeds`` and multi-codebook logits sampling
are deferred to PR4 (the engine / sampler PR).

The class is registered in sglang's ``ModelRegistry`` under the HF
architecture name ``HiggsMultimodalQwen3ForConditionalGeneration`` (see
:mod:`sglang_omni.models.sglang_registry`).
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn

from sglang_omni.models.higgs_tts.hf_config import HiggsMultimodalQwen3Config
from sglang_omni.models.higgs_tts.modeling import (
    HiggsFusedMultiTextEmbedding,
    HiggsFusedMultiTextHead,
)
from sglang_omni.models.higgs_tts.weight_loader import DiscreteWeightMapper

# Prefix map for the text backbone. Since we compose :class:`Qwen3ForCausalLM`
# under ``self.backbone``, all text-backbone weights live under
# ``backbone.model.*`` (and ``backbone.lm_head.*``). :func:`load_weights` then
# strips the ``backbone.`` prefix and hands the remaining names to
# ``Qwen3ForCausalLM.load_weights``, which handles qkv / gate_up stacking and
# tied-lm-head logic.
_BACKBONE_PREFIX_MAP: dict[str, str] = {
    "tied.embedding.text_embedding.": "backbone.model.embed_tokens.",
    "body.layers.": "backbone.model.layers.",
    "body.norm.": "backbone.model.norm.",
    "tied.head.text_head.": "backbone.lm_head.",
}


class _HiggsMultimodalEmbedding(nn.Module):
    """Container matching the Higgs checkpoint layout.

    The checkpoint stores the fused embedding under
    ``multimodal_embedding.modality_embedding_0.*``; the container name
    ``multimodal_embedding`` is preserved so weight loading is a straight
    prefix substitution (see :class:`DiscreteWeightMapper`).
    """

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.modality_embedding_0 = HiggsFusedMultiTextEmbedding(
            num_codebooks=num_codebooks,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )


class HiggsTTSModel(nn.Module):
    """Higgs Multimodal Qwen3 model (discrete TTS path) adapted for sglang.

    Composition over :class:`sglang.srt.models.qwen3.Qwen3ForCausalLM` —
    the backbone handles paged attention, KV cache, logits processing and
    standard text weight loading. This wrapper adds:

    - ``multimodal_embedding.modality_embedding_0``: the fused
      :class:`HiggsFusedMultiTextEmbedding` (shape ``[N*V, D]``).
    - ``modality_head``: the fused :class:`HiggsFusedMultiTextHead`, tied
      to the embedding weight when ``audio_encoder_config.tie_word_embeddings``.
    - :meth:`load_weights` that remaps Higgs checkpoint names and splits
      the stream between the backbone and the multimodal modules.

    Multi-codebook input embedding insertion (``-100`` placeholder overlay
    from reference audio codes) and multi-codebook sampling are deferred
    to PR4.
    """

    def __init__(
        self,
        config: HiggsMultimodalQwen3Config,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Late import so the test suite can exercise the registry entry
        # without eagerly importing sglang's heavyweight inference stack.
        from sglang.srt.models.qwen3 import Qwen3ForCausalLM

        self.config = config

        text_config = config.get_text_config()
        self.backbone = Qwen3ForCausalLM(
            text_config,
            quant_config=quant_config,
            prefix=prefix + "backbone" if prefix else "backbone",
        )

        enc_cfg = config.audio_encoder_config or {}
        encoder_type = enc_cfg.get("encoder_type", "discrete")
        if encoder_type != "discrete":
            raise NotImplementedError(
                f"HiggsTTSModel currently supports only the discrete "
                f"TTS path; got encoder_type={encoder_type!r}. Whisper/Qwen3-AUT "
                f"(ASR) encoders are planned for a future PR."
            )

        num_codebooks: int = int(enc_cfg["num_codebooks"])
        vocab_size: int = int(enc_cfg["vocab_size"])
        hidden_size: int = int(enc_cfg.get("out_dim", text_config.hidden_size))
        self._num_codebooks = num_codebooks
        self._codebook_vocab_size = vocab_size
        self._tie_modality = bool(enc_cfg.get("tie_word_embeddings", True))

        self.multimodal_embedding = _HiggsMultimodalEmbedding(
            num_codebooks=num_codebooks,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )
        self.modality_head = HiggsFusedMultiTextHead(
            num_codebooks=num_codebooks,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )
        if self._tie_modality:
            # Share storage with the fused embedding (matches boson-vllm behaviour).
            self.modality_head.weight = (
                self.multimodal_embedding.modality_embedding_0.weight
            )

    # Accessors used by downstream stages in later PRs -----------------------
    def get_input_embeddings(self) -> nn.Embedding:
        return self.backbone.get_input_embeddings()

    def get_multimodal_embedding(self) -> HiggsFusedMultiTextEmbedding:
        return self.multimodal_embedding.modality_embedding_0

    def get_modality_head(self) -> HiggsFusedMultiTextHead:
        return self.modality_head

    @property
    def num_codebooks(self) -> int:
        return self._num_codebooks

    @property
    def codebook_vocab_size(self) -> int:
        return self._codebook_vocab_size

    # Forward ----------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Text-only forward. PR4 will inject multi-codebook embeddings into
        ``input_embeds`` before calling this method when audio placeholders
        are present in the prompt.
        """
        return self.backbone(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            **kwargs,
        )

    # Weight loading ---------------------------------------------------------
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Remap Higgs checkpoint names then route to the right submodule.

        Returns the set of **own** parameter names this wrapper loaded (i.e.,
        the multimodal embedding and optionally the untied modality head).
        Text backbone weights are delegated to
        :meth:`sglang.srt.models.qwen3.Qwen3ForCausalLM.load_weights`, which
        performs its own stacking (qkv_proj, gate_up_proj) and tied-lm-head
        logic; those loads are not reflected in the returned set.
        """
        mapper = DiscreteWeightMapper(
            text_prefix_map=_BACKBONE_PREFIX_MAP,
            tie_modality=self._tie_modality,
        )

        backbone_weights: list[Tuple[str, torch.Tensor]] = []
        self_weights: list[Tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()
        own_names = self._own_param_names()

        for name, tensor in weights:
            mapped = mapper.map(name)
            if mapped is None:
                continue  # e.g., audio-tokenizer backbone, discarded by design
            if mapped.startswith("backbone."):
                # Hand off to sglang's Qwen3 loader with the ``backbone.`` prefix
                # stripped; it expects names starting with ``model.`` or ``lm_head.``.
                backbone_weights.append((mapped[len("backbone.") :], tensor))
            elif mapped in own_names:
                self_weights.append((mapped, tensor))
            # Names that survived remapping without a destination (e.g.,
            # `tied.head.modality_heads.0.weight` under tie_modality=True) are
            # dropped silently; the checkpoint value is redundant with the
            # tied embedding weight.

        # Delegate text backbone loading. Qwen3ForCausalLM handles qkv/gate_up
        # stacking and lm_head tying internally.
        self.backbone.load_weights(iter(backbone_weights))

        # Load fused embedding (and untied head, if applicable) directly.
        own_params = dict(self.named_parameters(remove_duplicate=False))
        for name, tensor in self_weights:
            param = own_params.get(name)
            if param is None:
                continue
            if param.shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: expected {tuple(param.shape)}, "
                    f"got {tuple(tensor.shape)}"
                )
            param.data.copy_(tensor.to(param.dtype))
            loaded.add(name)

        return loaded

    def _own_param_names(self) -> set[str]:
        """Names of parameters owned directly by this wrapper (not backbone)."""
        names: set[str] = set()
        for name, _ in self.named_parameters(remove_duplicate=False):
            if not name.startswith("backbone."):
                names.add(name)
        return names


__all__ = ["HiggsTTSModel"]
