# SPDX-License-Identifier: Apache-2.0
"""Tests for the Higgs delay pattern (PR3a)."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.delay_pattern import (
    BOC_ID,
    EOC_ID,
    apply_delay_pattern,
    reverse_delay_pattern,
)


def test_output_shape():
    codes = torch.zeros(6, 4, dtype=torch.long)
    assert apply_delay_pattern(codes).shape == (6 + 4 - 1, 4)


def test_codebook_0_unshifted():
    codes = torch.tensor([[10, 20, 30], [11, 21, 31], [12, 22, 32]])
    out = apply_delay_pattern(codes)
    # codebook 0: data at rows [0, 3), eoc at rows [3, 5)
    assert out[:, 0].tolist() == [10, 11, 12, EOC_ID, EOC_ID]


def test_codebook_i_delayed_by_i():
    codes = torch.tensor([[10, 20, 30], [11, 21, 31], [12, 22, 32]])
    out = apply_delay_pattern(codes)
    # codebook 1: boc at row 0, data at rows [1, 4), eoc at row 4
    assert out[:, 1].tolist() == [BOC_ID, 20, 21, 22, EOC_ID]
    # codebook 2: boc at rows [0, 2), data at rows [2, 5)
    assert out[:, 2].tolist() == [BOC_ID, BOC_ID, 30, 31, 32]


def test_requires_2d():
    with pytest.raises(ValueError, match="2-D"):
        apply_delay_pattern(torch.zeros(4, dtype=torch.long))


# ---------------------------------------------------------------------------
# reverse_delay_pattern (PR5)
# ---------------------------------------------------------------------------


def test_reverse_roundtrip():
    """apply → reverse recovers the original codes."""
    torch.manual_seed(0)
    for T, N in [(1, 2), (3, 3), (6, 4), (17, 8)]:
        codes = torch.randint(0, 1024, (T, N), dtype=torch.long)
        recovered = reverse_delay_pattern(apply_delay_pattern(codes))
        assert recovered.shape == codes.shape
        assert torch.equal(recovered, codes)


def test_reverse_drops_boc_prefix():
    """First ``c`` rows of codebook ``c`` are BOC and must not leak through."""
    codes = torch.tensor([[10, 20, 30], [11, 21, 31], [12, 22, 32]])
    delayed = apply_delay_pattern(codes)
    recovered = reverse_delay_pattern(delayed)
    # None of the recovered values should be BOC/EOC.
    assert int(recovered.min().item()) >= 0
    assert BOC_ID not in recovered.unique().tolist()
    assert EOC_ID not in recovered.unique().tolist()


def test_reverse_requires_2d():
    with pytest.raises(ValueError, match="2-D"):
        reverse_delay_pattern(torch.zeros(4, dtype=torch.long))


def test_reverse_rejects_too_short():
    """L < N leaves no data rows — must raise rather than return empty."""
    # N=8, L=5 → T = 5 - 7 = -2 (invalid).
    with pytest.raises(ValueError, match="at least one"):
        reverse_delay_pattern(torch.zeros(5, 8, dtype=torch.long))
