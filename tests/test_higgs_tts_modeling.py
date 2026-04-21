# SPDX-License-Identifier: Apache-2.0
"""Numerical tests for the fused multi-codebook modules (PR2a)."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.modeling import (
    HiggsFusedMultiTextEmbedding,
    HiggsFusedMultiTextHead,
)


class TestHiggsFusedMultiTextEmbedding:
    def test_output_shape(self):
        emb = HiggsFusedMultiTextEmbedding(
            num_codebooks=8, vocab_size=1026, hidden_size=2048
        )
        codes = torch.randint(0, 1024, (4, 8), dtype=torch.long)
        out = emb(codes)
        assert out.shape == (4, 2048)

    def test_output_shape_with_leading_dims(self):
        emb = HiggsFusedMultiTextEmbedding(
            num_codebooks=8, vocab_size=1026, hidden_size=128
        )
        codes = torch.randint(0, 1024, (2, 3, 8), dtype=torch.long)
        out = emb(codes)
        assert out.shape == (2, 3, 128)

    def test_offset_math_matches_manual_lookup(self):
        """Each codebook i indexes into rows [i*V, (i+1)*V); output is the row sum."""
        torch.manual_seed(0)
        N, V, D = 3, 16, 8
        emb = HiggsFusedMultiTextEmbedding(num_codebooks=N, vocab_size=V, hidden_size=D)
        torch.nn.init.normal_(emb.weight)

        codes = torch.tensor([[5, 10, 3], [0, 15, 7]], dtype=torch.long)
        fused_out = emb(codes)

        expected = torch.stack(
            [
                emb.weight[5] + emb.weight[V + 10] + emb.weight[2 * V + 3],
                emb.weight[0] + emb.weight[V + 15] + emb.weight[2 * V + 7],
            ]
        )
        torch.testing.assert_close(fused_out, expected)

    def test_weight_is_single_contiguous_parameter(self):
        """The fused embedding holds ONE weight of shape [N*V, D]."""
        emb = HiggsFusedMultiTextEmbedding(
            num_codebooks=8, vocab_size=1026, hidden_size=2048
        )
        params = dict(emb.named_parameters())
        assert list(params.keys()) == ["weight"]
        assert params["weight"].shape == (8 * 1026, 2048)


class TestHiggsFusedMultiTextHead:
    def test_output_shape(self):
        head = HiggsFusedMultiTextHead(
            num_codebooks=8, vocab_size=1026, hidden_size=2048
        )
        hidden = torch.randn(5, 2048)
        out = head.generate(hidden)
        assert out.shape == (5, 8, 1026)

    def test_generate_matches_linear_then_reshape(self):
        torch.manual_seed(0)
        N, V, D = 4, 32, 16
        head = HiggsFusedMultiTextHead(num_codebooks=N, vocab_size=V, hidden_size=D)
        torch.nn.init.normal_(head.weight)

        hidden = torch.randn(7, D)
        expected = torch.nn.functional.linear(hidden, head.weight).reshape(7, N, V)
        torch.testing.assert_close(head.generate(hidden), expected)


class TestWeightTying:
    """When ``tie_word_embeddings=True``, head shares embedding weight."""

    def test_tied_weight_shares_storage(self):
        N, V, D = 4, 8, 16
        emb = HiggsFusedMultiTextEmbedding(N, V, D)
        head = HiggsFusedMultiTextHead(N, V, D)
        head.weight = emb.weight

        assert head.weight is emb.weight
        assert head.weight.data_ptr() == emb.weight.data_ptr()

    def test_tied_embedding_and_head_are_linear_duals(self):
        """Tied: head.generate(emb(one_hot_like)) recovers the expected pattern.

        Picking codes_LN = [[i, 0, ..., 0]] selects row ``i`` plus rows in
        codebooks 1..N-1 (fixed to 0). The head projection then produces
        logits whose (0, i) entry is ||embedding_row_i||^2 + cross terms.
        We just assert the output shape and that nothing crashes — this is a
        structural smoke test for weight sharing, not a correctness check of
        tied-softmax theory.
        """
        torch.manual_seed(0)
        N, V, D = 4, 8, 16
        emb = HiggsFusedMultiTextEmbedding(N, V, D)
        head = HiggsFusedMultiTextHead(N, V, D)
        torch.nn.init.normal_(emb.weight)
        head.weight = emb.weight

        codes = torch.tensor([[1, 0, 0, 0], [2, 0, 0, 0]], dtype=torch.long)
        summed = emb(codes)
        logits = head.generate(summed)
        assert logits.shape == (2, N, V)


@pytest.mark.parametrize(
    "num_codebooks,vocab_size,hidden_size", [(8, 1026, 2048), (10, 4096, 1024)]
)
def test_roundtrip_shapes(num_codebooks, vocab_size, hidden_size):
    """Parametric sanity: realistic TTS dimensions."""
    emb = HiggsFusedMultiTextEmbedding(num_codebooks, vocab_size, hidden_size)
    head = HiggsFusedMultiTextHead(num_codebooks, vocab_size, hidden_size)
    codes = torch.randint(0, vocab_size, (6, num_codebooks), dtype=torch.long)
    out = head.generate(emb(codes))
    assert out.shape == (6, num_codebooks, vocab_size)
