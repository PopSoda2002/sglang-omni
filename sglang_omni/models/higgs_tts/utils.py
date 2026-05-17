# SPDX-License-Identifier: Apache-2.0
"""Small utilities shared across the Higgs TTS pipeline.

- :func:`apply_delay_pattern` / :func:`reverse_delay_pattern` — multi-codebook
  delay shift used by preprocessing + audio_encoder + vocoder. Codebook ``c``
  is delayed by ``c`` steps; positions before the data segment are filled with
  :data:`BOC_ID`, positions after with :data:`EOC_ID`. Both live inside the
  codebook vocab (ids 1024 / 1025 for the default 1026 vocab).
- :func:`truncate_rope_to_bf16` — bf16-truncate sglang's fp32 RoPE cache to
  match Higgs's bf16 training-time RoPE; otherwise the fp32 frequencies drift
  logits at serving time.
"""

from __future__ import annotations

import torch

# Codec-vocab specials (inside the [N*V] codebook space, NOT the text vocab).
BOC_ID = 1024
EOC_ID = 1025


def apply_delay_pattern(codes_TN: torch.Tensor) -> torch.Tensor:
    """Shift codebook ``c`` by ``c`` steps; pad with ``BOC_ID`` / ``EOC_ID``.

    Args:
        codes_TN: Raw multi-codebook tokens, shape ``[T, N]``.

    Returns:
        Delayed tokens, shape ``[T + N - 1, N]``.
    """
    if codes_TN.ndim != 2:
        raise ValueError(
            f"codes_TN must be 2-D [T, N], got shape {tuple(codes_TN.shape)}"
        )
    T, N = codes_TN.shape
    out = torch.full(
        (T + N - 1, N), EOC_ID, device=codes_TN.device, dtype=codes_TN.dtype
    )
    t_idx = torch.arange(T + N - 1, device=codes_TN.device)
    for c in range(N):
        out[t_idx < c, c] = BOC_ID
        out[c : c + T, c] = codes_TN[:, c]
    return out


def reverse_delay_pattern(delayed_LN: torch.Tensor) -> torch.Tensor:
    """Undo :func:`apply_delay_pattern`.

    Given a delayed sequence of shape ``[L, N]`` (where ``L >= N - 1``),
    pull codebook ``c`` back by ``c`` steps and return the
    ``[L - (N - 1), N]`` data window.
    """
    if delayed_LN.ndim != 2:
        raise ValueError(
            f"delayed_LN must be 2-D [L, N], got shape {tuple(delayed_LN.shape)}"
        )
    L, N = delayed_LN.shape
    T = L - (N - 1)
    if T <= 0:
        raise ValueError(
            f"delayed_LN has L={L}, N={N}; need L >= N so at least one "
            f"data row can be recovered."
        )
    out = torch.empty((T, N), device=delayed_LN.device, dtype=delayed_LN.dtype)
    for c in range(N):
        out[:, c] = delayed_LN[c : c + T, c]
    return out


def truncate_rope_to_bf16(model: torch.nn.Module) -> None:
    """Truncate sglang's fp32 ``cos_sin_cache`` to bf16 precision in-place
    (stored as fp32) to match Higgs's bf16 training-time RoPE.
    """
    for module in model.modules():
        if hasattr(module, "cos_sin_cache"):
            module.cos_sin_cache.data = module.cos_sin_cache.data.to(torch.bfloat16).to(
                torch.float32
            )


__all__ = [
    "BOC_ID",
    "EOC_ID",
    "apply_delay_pattern",
    "reverse_delay_pattern",
    "truncate_rope_to_bf16",
]
