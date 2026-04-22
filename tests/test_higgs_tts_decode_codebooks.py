# SPDX-License-Identifier: Apache-2.0
"""Tests for the forward-embedded multi-codebook decode hooks on
:class:`HiggsTTSModel` (PR4c-i).

These tests exercise the request-slot bookkeeping + ``decode_codebooks_batch``
with a tiny random-weight model, skipping the full sglang ``forward`` path
(that's PR4c-ii once sglang's ``ForwardBatch`` is wired in).
"""

from __future__ import annotations

import os

import pytest

from sglang_omni.models.higgs_tts.delay_pattern import BOC_ID, EOC_ID
from sglang_omni.models.higgs_tts.hf_config import HiggsMultimodalQwen3Config

ARCHITECTURE = "HiggsMultimodalQwen3ForConditionalGeneration"

_TINY_TEXT_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 2048,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 128,
    "tie_word_embeddings": True,
}

_TINY_AUDIO_ENCODER_CONFIG = {
    "encoder_type": "discrete",
    "num_codebooks": 4,
    "vocab_size": 1026,
    "out_dim": 64,
    "tie_word_embeddings": True,
}


def _init_sglang_tp() -> None:
    """Single-rank TP context (mirrors test_higgs_tts_model)."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("sglang Qwen3 layers require CUDA at construction time")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    from sglang.srt.distributed import parallel_state
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    try:
        parallel_state.init_distributed_environment(
            backend="nccl", world_size=1, rank=0, local_rank=0
        )
    except RuntimeError:
        pass
    try:
        parallel_state.initialize_model_parallel()
    except (RuntimeError, AssertionError):
        pass

    import sglang.srt.layers.dp_attention as dp

    dp._ATTN_TP_SIZE = 1
    dp._ATTN_TP_RANK = 0
    dp._ATTN_DP_SIZE = 1
    dp._ATTN_DP_RANK = 0
    dp._LOCAL_ATTN_DP_SIZE = 1
    dp._LOCAL_ATTN_DP_RANK = 0


def _make_model():
    _init_sglang_tp()

    from sglang_omni.models.higgs_tts.model import HiggsTTSModel

    cfg = HiggsMultimodalQwen3Config(
        audio_encoder_config=_TINY_AUDIO_ENCODER_CONFIG,
        text_config=_TINY_TEXT_CONFIG,
        audio_token_id=-100,
        architectures=[ARCHITECTURE],
    )
    return HiggsTTSModel(cfg)


# ---------------------------------------------------------------------------
# Slot bookkeeping
# ---------------------------------------------------------------------------


def test_get_slot_creates_fresh_state():
    model = _make_model()
    slot = model.get_slot("req-a")
    assert slot.sampler.num_codebooks == model.num_codebooks
    assert slot.sampler.delay_count == 0
    assert slot.output_codes == []
    # Second access returns the same slot.
    assert model.get_slot("req-a") is slot


def test_reset_request_drops_slot():
    model = _make_model()
    slot = model.get_slot("req-a")
    slot.output_codes.append(slot.sampler.last_codes)  # noqa: E501 — any tensor
    model.reset_request("req-a")
    assert "req-a" not in model._slots
    assert model.get_slot("req-a") is not slot


def test_get_output_codes_empty_request():
    model = _make_model()
    codes = model.get_output_codes("never-touched")
    assert codes.shape == (0, model.num_codebooks)
    assert codes.dtype.is_floating_point is False


# ---------------------------------------------------------------------------
# decode_codebooks_batch
# ---------------------------------------------------------------------------


def test_decode_codebooks_batch_returns_peaked_text_logits():
    """Returned text logits peak at codebook-0's sampled value for each row."""
    import torch

    model = _make_model()
    D = model.multimodal_embedding.modality_embedding_0.weight.shape[1]
    hidden = torch.randn(2, D, device=model.modality_head.weight.device)

    from sglang_omni.models.higgs_tts.model import HiggsGenParams

    gen = [HiggsGenParams(temperature=0.0), HiggsGenParams(temperature=0.0)]
    logits_BV = model.decode_codebooks_batch(hidden, ["r1", "r2"], gen)

    V_text = model.backbone.config.vocab_size
    assert logits_BV.shape == (2, V_text)

    # Each row's argmax matches the codebook-0 of the stored slot output.
    for b, rid in enumerate(["r1", "r2"]):
        codes = model.get_output_codes(rid)
        assert codes.shape == (1, model.num_codebooks)
        cb0 = int(codes[0, 0].item())
        assert int(logits_BV[b].argmax().item()) == cb0


def test_decode_codebooks_batch_applies_delay_overrides():
    """During the delay window, later codebooks must be boc_id in the
    stored codes (covers the state-machine wiring end-to-end)."""
    import torch

    model = _make_model()
    D = model.multimodal_embedding.modality_embedding_0.weight.shape[1]
    hidden = torch.randn(1, D, device=model.modality_head.weight.device)

    from sglang_omni.models.higgs_tts.model import HiggsGenParams

    _ = model.decode_codebooks_batch(hidden, ["r1"], [HiggsGenParams(temperature=0.0)])
    codes = model.get_output_codes("r1")[0].tolist()
    # Step 0: codebooks [1:] are forced to BOC_ID.
    assert codes[1:] == [BOC_ID] * (model.num_codebooks - 1)


def test_decode_codebooks_batch_accumulates_across_steps():
    """Multiple successive calls with the same req_id extend the output
    codes list (each call = one AR step)."""
    import torch

    model = _make_model()
    D = model.multimodal_embedding.modality_embedding_0.weight.shape[1]
    from sglang_omni.models.higgs_tts.model import HiggsGenParams

    for _ in range(3):
        hidden = torch.randn(1, D, device=model.modality_head.weight.device)
        model.decode_codebooks_batch(hidden, ["r1"], [HiggsGenParams(temperature=0.0)])
    codes = model.get_output_codes("r1")
    assert codes.shape == (3, model.num_codebooks)


def test_decode_codebooks_batch_handles_concurrent_requests():
    """Two requests with disjoint ids advance their sampler states
    independently."""
    import torch

    model = _make_model()
    D = model.multimodal_embedding.modality_embedding_0.weight.shape[1]
    from sglang_omni.models.higgs_tts.model import HiggsGenParams

    # r1 gets two steps, r2 gets one.
    hidden = torch.randn(2, D, device=model.modality_head.weight.device)
    model.decode_codebooks_batch(
        hidden,
        ["r1", "r2"],
        [HiggsGenParams(temperature=0.0), HiggsGenParams(temperature=0.0)],
    )
    model.decode_codebooks_batch(hidden[:1], ["r1"], [HiggsGenParams(temperature=0.0)])
    assert model.get_slot("r1").sampler.delay_count == 2
    assert model.get_slot("r2").sampler.delay_count == 1


def test_decode_codebooks_batch_size_mismatch_raises():
    import torch

    model = _make_model()
    D = model.multimodal_embedding.modality_embedding_0.weight.shape[1]
    hidden = torch.randn(2, D, device=model.modality_head.weight.device)

    from sglang_omni.models.higgs_tts.model import HiggsGenParams

    with pytest.raises(ValueError, match="batch size"):
        model.decode_codebooks_batch(
            hidden,
            ["r1"],  # only 1 req_id for 2 batch rows
            [HiggsGenParams(temperature=0.0), HiggsGenParams(temperature=0.0)],
        )


def test_stop_code_on_generation_done_is_safe():
    """A request that's already ``generation_done`` returns ``STOP_CODE``
    (-1) from the sampler; the text logits must not crash on that."""
    import torch

    model = _make_model()
    D = model.multimodal_embedding.modality_embedding_0.weight.shape[1]
    from sglang_omni.models.higgs_tts.model import HiggsGenParams

    slot = model.get_slot("r1")
    slot.sampler.generation_done = True
    hidden = torch.randn(1, D, device=model.modality_head.weight.device)
    logits_BV = model.decode_codebooks_batch(
        hidden, ["r1"], [HiggsGenParams(temperature=0.0)]
    )
    V_text = model.backbone.config.vocab_size
    assert logits_BV.shape == (1, V_text)
    # -1 is out of range → no one-hot boost → all values stay at -1e4.
    assert logits_BV.max().item() < 0
    codes = model.get_output_codes("r1")
    assert codes[0, 0].item() == -1  # STOP_CODE sentinel stored
    # eoc_id in text vocab shouldn't be incorrectly boosted either.
    assert logits_BV[0, EOC_ID].item() < 0
