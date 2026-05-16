# SPDX-License-Identifier: Apache-2.0
"""Bootstrap helpers for Higgs TTS SGLang execution."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


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


__all__ = ["truncate_rope_to_bf16"]
