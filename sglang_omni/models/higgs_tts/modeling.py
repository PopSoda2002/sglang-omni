# SPDX-License-Identifier: Apache-2.0
"""Fused multi-codebook modules for HiggsMultimodalQwen3 (discrete TTS path).

Ported verbatim from boson-vllm
(`vllm/model_executor/models/higgs_multimodal_qwen3.py:464-509`). Kept as a
standalone ``nn.Module`` with no sglang/vLLM dependencies so both stacks can
reuse the same numerical definition. The full sglang-integrated model class
(Qwen3 backbone + attention + kv cache) is deferred to PR2b.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HiggsFusedMultiTextEmbedding(nn.Module):
    """Fused multi-codebook embedding: single weight ``[N*V, D]`` with offset lookup.

    Functionally equivalent to an ensemble of ``N`` per-codebook embeddings, but
    stores all codebook embeddings in one contiguous parameter. Each codebook
    occupies a slice of size ``V`` in the weight; the forward pass adds
    per-codebook offsets (``0, V, 2V, ...``) before embedding lookup, then sums
    across the codebook axis.

    Shapes:
        codes_LN: ``[..., N]`` integer codebook ids
        returns:  ``[..., D]`` summed embedding
    """

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

    def forward(self, codes_LN: torch.Tensor) -> torch.Tensor:
        N = self.num_codebooks
        V = self.vocab_size
        offsets = torch.arange(N, device=codes_LN.device, dtype=codes_LN.dtype) * V
        fused_ids = codes_LN + offsets
        emb = F.embedding(fused_ids, self.weight)  # [..., N, D]
        return emb.sum(dim=-2)  # [..., D]


class HiggsFusedMultiTextHead(nn.Module):
    """Fused multi-codebook head: single weight ``[N*V, D]``.

    Produces logits for all ``N`` codebooks via a single linear projection,
    then reshapes to ``[L, N, V]``. Typically tied with
    :class:`HiggsFusedMultiTextEmbedding` (i.e., ``head.weight is embedding.weight``)
    when ``tie_word_embeddings`` is set in the encoder config.

    Shapes:
        hidden_LD: ``[L, D]``
        returns:   ``[L, N, V]``
    """

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

    def generate(self, hidden_LD: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden_LD, self.weight)  # [L, N*V]
        return logits.reshape(
            hidden_LD.shape[0],
            self.num_codebooks,
            self.vocab_size,
        )


__all__ = ["HiggsFusedMultiTextEmbedding", "HiggsFusedMultiTextHead"]
