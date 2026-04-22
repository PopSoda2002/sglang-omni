# SPDX-License-Identifier: Apache-2.0
"""Tests for Higgs delay-pattern helpers (PR3a)."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.delay_pattern import (
    apply_delay_pattern,
    delayed_length,
    reverse_delay_pattern,
)

BOC = 1024
EOC = 1025


class TestApplyDelayPattern:
    def test_output_shape(self):
        codes = torch.zeros(6, 4, dtype=torch.long)
        out = apply_delay_pattern(codes, valid_len=6, boc_id=BOC, eoc_id=EOC)
        assert out.shape == (6 + 4 - 1, 4)

    def test_codebook_0_data_region_unshifted(self):
        codes = torch.tensor([[10, 20, 30], [11, 21, 31], [12, 22, 32]])
        out = apply_delay_pattern(codes, valid_len=3, boc_id=BOC, eoc_id=EOC)
        # codebook 0 gets data at rows [0, 3), then eoc at rows [3, 5)
        assert out[:, 0].tolist() == [10, 11, 12, EOC, EOC]

    def test_codebook_i_delayed_by_i(self):
        codes = torch.tensor([[10, 20, 30], [11, 21, 31], [12, 22, 32]])
        out = apply_delay_pattern(codes, valid_len=3, boc_id=BOC, eoc_id=EOC)
        # codebook 1: boc at row 0, data at rows [1, 4), eoc at row 4
        assert out[:, 1].tolist() == [BOC, 20, 21, 22, EOC]
        # codebook 2: boc at rows [0, 2), data at rows [2, 5)
        assert out[:, 2].tolist() == [BOC, BOC, 30, 31, 32]

    def test_valid_len_smaller_than_L(self):
        codes = torch.arange(12).reshape(6, 2).long()  # [6, 2]
        # Only the first 3 rows are "real" data; rest should be ignored.
        out = apply_delay_pattern(codes, valid_len=3, boc_id=BOC, eoc_id=EOC)
        assert out.shape == (3 + 2 - 1, 2)  # 4 rows
        # Codebook 0: data at rows 0..2, eoc at row 3
        assert out[:, 0].tolist() == [0, 2, 4, EOC]

    def test_invalid_valid_len(self):
        codes = torch.zeros(4, 2, dtype=torch.long)
        with pytest.raises(ValueError, match="valid_len"):
            apply_delay_pattern(codes, valid_len=5, boc_id=BOC, eoc_id=EOC)
        with pytest.raises(ValueError, match="valid_len"):
            apply_delay_pattern(codes, valid_len=-1, boc_id=BOC, eoc_id=EOC)

    def test_requires_2d(self):
        codes = torch.zeros(4, dtype=torch.long)
        with pytest.raises(ValueError, match="2-D"):
            apply_delay_pattern(codes, valid_len=4, boc_id=BOC, eoc_id=EOC)


class TestReverseDelayPattern:
    def test_roundtrip(self):
        torch.manual_seed(0)
        T, N = 7, 5
        codes = torch.randint(0, 1024, (T, N), dtype=torch.long)
        delayed = apply_delay_pattern(codes, valid_len=T, boc_id=BOC, eoc_id=EOC)
        recovered = reverse_delay_pattern(delayed)
        assert torch.equal(recovered, codes)

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="2-D"):
            reverse_delay_pattern(torch.zeros(5, dtype=torch.long))
        # delayed_L < N - 1 is malformed.
        with pytest.raises(ValueError, match="smaller than"):
            reverse_delay_pattern(torch.zeros(2, 5, dtype=torch.long))


class TestDelayedLength:
    @pytest.mark.parametrize(
        "valid_len,num_codebooks,expected",
        [(10, 8, 17), (1, 4, 4), (0, 8, 7), (100, 1, 100)],
    )
    def test_values(self, valid_len, num_codebooks, expected):
        assert delayed_length(valid_len, num_codebooks) == expected
