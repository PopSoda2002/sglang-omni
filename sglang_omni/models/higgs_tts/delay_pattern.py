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


def reverse_delay_pattern(delayed_LN: torch.Tensor) -> torch.Tensor:
    """Undo :func:`apply_delay_pattern`.

    Given a delayed sequence of shape ``[L, N]`` (where ``L >= N - 1`` so
    the data segment has at least one row), pull codebook ``c`` back by
    ``c`` steps and return the ``[L - (N - 1), N]`` data window. The BOC
    prefix (first ``c`` rows of codebook ``c``) and the EOC tail (rows
    past the data region) are dropped.

    Mirrors boson-vllm's ``simple_tts_inference.py:65-91`` reconstruction
    step.

    Args:
        delayed_LN: Delayed tokens from :func:`apply_delay_pattern` or
            directly from the AR decoder, shape ``[L, N]``.

    Returns:
        Raw codes of shape ``[L - (N - 1), N]``. Rows with ``L < N - 1``
        can't be reconstructed — the function raises ``ValueError``.
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


__all__ = ["BOC_ID", "EOC_ID", "apply_delay_pattern", "reverse_delay_pattern"]
