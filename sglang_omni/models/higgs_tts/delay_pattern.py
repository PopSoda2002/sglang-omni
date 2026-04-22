# SPDX-License-Identifier: Apache-2.0
"""Delay-pattern utilities for Higgs multi-codebook TTS.

Codebook ``i`` is delayed by ``i`` steps relative to codebook 0. Positions
before the data segment are filled with ``<|boc|>`` (begin-of-codes, inside
the codebook vocab), positions after are filled with ``<|eoc|>`` (end-of-codes).

Forward pass (``apply_delay_pattern``) is applied to reference audio codes
before they enter :class:`HiggsFusedMultiTextEmbedding`; the inverse
(``reverse_delay_pattern``) is applied to model-generated codes before they
reach the vocoder.

Ported from boson-vllm's ``build_delay_pattern`` in
``vllm/model_executor/models/higgs_multimodal_qwen3.py:398-436``; kept as a
standalone torch helper so the preprocessing stage (CPU) and the vocoder
stage can share it.
"""

from __future__ import annotations

import torch


def apply_delay_pattern(
    codes_LN: torch.Tensor,
    valid_len: int,
    boc_id: int,
    eoc_id: int,
) -> torch.Tensor:
    """Apply the Higgs delay pattern.

    Args:
        codes_LN: Raw multi-codebook tokens, shape ``[L, N]``.
        valid_len: Number of valid (data) rows at the start of ``codes_LN``.
            Rows ``[valid_len, L)`` are ignored.
        boc_id: Begin-of-codes token id (inside codec vocab; typically 1024).
        eoc_id: End-of-codes token id (inside codec vocab; typically 1025).

    Returns:
        Delayed tokens, shape ``[valid_len + N - 1, N]``. Codebook ``c``
        has the data region at rows ``[c, c + valid_len)`` and ``boc_id`` /
        ``eoc_id`` padding outside.
    """
    if codes_LN.ndim != 2:
        raise ValueError(
            f"codes_LN must be 2-D [L, N], got shape {tuple(codes_LN.shape)}"
        )
    L, N = codes_LN.shape
    if valid_len < 0 or valid_len > L:
        raise ValueError(f"valid_len={valid_len} out of range [0, {L}]")

    new_L = valid_len + N - 1
    device = codes_LN.device
    dtype = codes_LN.dtype

    output = torch.full((new_L, N), eoc_id, device=device, dtype=dtype)
    t_idx = torch.arange(new_L, device=device)

    for c in range(N):
        # boc before the data segment of codebook c
        boc_mask = t_idx < c
        output[boc_mask, c] = boc_id
        # data region
        data_start = c
        data_end = c + valid_len
        if data_end > data_start:
            output[data_start:data_end, c] = codes_LN[:valid_len, c]

    return output


def reverse_delay_pattern(delayed_LN: torch.Tensor) -> torch.Tensor:
    """Undo :func:`apply_delay_pattern` — recover the square ``[T, N]`` codes.

    Given a delayed sequence shaped ``[T + N - 1, N]``, pick for codebook
    ``c`` the rows ``[c, c + T)`` so the output at row ``t`` contains
    codebook 0's step ``t``, codebook 1's step ``t``, ..., with the delay
    stripped.

    Args:
        delayed_LN: Delayed tokens, shape ``[T + N - 1, N]``.

    Returns:
        Recovered tokens, shape ``[T, N]``.
    """
    if delayed_LN.ndim != 2:
        raise ValueError(
            f"delayed_LN must be 2-D [T+N-1, N], got shape {tuple(delayed_LN.shape)}"
        )
    delayed_L, N = delayed_LN.shape
    T = delayed_L - N + 1
    if T < 0:
        raise ValueError(
            f"delayed_L={delayed_L} smaller than N-1={N - 1}; cannot be a delayed sequence"
        )

    out = torch.empty((T, N), device=delayed_LN.device, dtype=delayed_LN.dtype)
    for c in range(N):
        out[:, c] = delayed_LN[c : c + T, c]
    return out


def delayed_length(valid_len: int, num_codebooks: int) -> int:
    """Number of rows (and prompt ``-100`` placeholders) after delay is applied."""
    return valid_len + num_codebooks - 1


__all__ = [
    "apply_delay_pattern",
    "reverse_delay_pattern",
    "delayed_length",
]
