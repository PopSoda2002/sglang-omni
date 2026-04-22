# SPDX-License-Identifier: Apache-2.0
"""Higgs multi-codebook delay pattern.

Codebook ``i`` is shifted ``i`` steps relative to codebook 0. Positions before
the data segment are filled with ``<|boc|>``; positions after, with ``<|eoc|>``
— both live inside the codebook vocab (ids 1024 and 1025 for the default 1026
vocab).

Ported from boson-vllm's ``build_delay_pattern`` in
``vllm/model_executor/models/higgs_multimodal_qwen3.py:398-436``.
"""

from __future__ import annotations

import torch

# Codec-vocab specials (inside the [N*V] codebook space, NOT the text vocab).
# Match the Higgs default: vocab_size=1026, last two entries are <|boc|>/<|eoc|>.
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


__all__ = ["BOC_ID", "EOC_ID", "apply_delay_pattern"]
