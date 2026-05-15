# SPDX-License-Identifier: Apache-2.0
"""Bootstrap helpers for Higgs TTS SGLang execution."""

from __future__ import annotations

import json
import logging
import os

import torch

logger = logging.getLogger(__name__)


def register_higgs_tts_in_sglang() -> None:
    """Register :class:`HiggsTTSModel` and :class:`HiggsMultimodalQwen3Config`.

    Idempotent — safe to call multiple times.
    """
    from sglang.srt.models.registry import ModelRegistry
    from transformers import AutoConfig

    from sglang_omni.models.higgs_tts.hf_config import HiggsMultimodalQwen3Config
    from sglang_omni.models.higgs_tts.model import HiggsTTSModel

    ARCH = "HiggsMultimodalQwen3ForConditionalGeneration"
    if ModelRegistry.models.get(ARCH) is not HiggsTTSModel:
        ModelRegistry.models[ARCH] = HiggsTTSModel

    try:
        AutoConfig.register("higgs_multimodal_qwen3", HiggsMultimodalQwen3Config)
    except (ValueError, KeyError):
        # Already registered — fine.
        pass


def load_fused_embedding_from_tts_ckpt(
    tts_ckpt_path: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load only the fused multi-codebook embedding from a TTS ckpt for the
    audio_encoder stage (skips the full Qwen3 backbone). Materialises the
    module in ``dtype`` (default bf16) BEFORE the weight copy so the
    lookup-and-sum runs in bf16 — matching the engine backbone bitwise;
    fp32 here adds ~1 ULP that compounds across the AR loop.
    """
    from safetensors import safe_open

    from sglang_omni.models.higgs_tts.hf_config import HiggsMultimodalQwen3Config
    from sglang_omni.models.higgs_tts.modeling import HiggsFusedMultiTextEmbedding

    cfg = HiggsMultimodalQwen3Config.from_pretrained(tts_ckpt_path)
    enc_cfg = cfg.audio_encoder_config or {}
    num_codebooks = int(enc_cfg["num_codebooks"])
    vocab_size = int(enc_cfg["vocab_size"])
    hidden_size = int(enc_cfg.get("out_dim", cfg.get_text_config().hidden_size))

    module = HiggsFusedMultiTextEmbedding(
        num_codebooks=num_codebooks,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )

    key = "tied.embedding.modality_embeddings.0.embedding.weight"
    index_path = os.path.join(tts_ckpt_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            shard_name = json.load(f)["weight_map"].get(key)
        if shard_name is None:
            raise RuntimeError(
                f"Fused embedding weight {key!r} not found in TTS ckpt index"
            )
        shard_path = os.path.join(tts_ckpt_path, shard_name)
    else:
        shard_path = os.path.join(tts_ckpt_path, "model.safetensors")

    with safe_open(shard_path, framework="pt") as f:
        tensor = f.get_tensor(key)

    if tensor.shape != tuple(module.weight.shape):
        raise RuntimeError(
            f"Fused embedding shape mismatch: ckpt {tuple(tensor.shape)}, "
            f"expected {tuple(module.weight.shape)}"
        )

    module = module.to(dtype=dtype)
    with torch.no_grad():
        module.weight.copy_(tensor.to(dtype=dtype))

    module = module.to(device).eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def truncate_rope_to_bf16(model: torch.nn.Module) -> None:
    """Truncate sglang's fp32 ``cos_sin_cache`` to bf16 precision in-place
    (stored as fp32) to match Higgs's bf16 training-time RoPE — otherwise
    the fp32 frequencies drift logits at serving time.
    """
    for module in model.modules():
        if hasattr(module, "cos_sin_cache"):
            module.cos_sin_cache.data = module.cos_sin_cache.data.to(torch.bfloat16).to(
                torch.float32
            )


__all__ = [
    "load_fused_embedding_from_tts_ckpt",
    "register_higgs_tts_in_sglang",
    "truncate_rope_to_bf16",
]
