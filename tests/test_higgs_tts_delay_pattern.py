# SPDX-License-Identifier: Apache-2.0
"""Tests for the Higgs delay pattern (PR3a)."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.delay_pattern import (
    BOC_ID,
    EOC_ID,
    apply_delay_pattern,
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
