# SPDX-License-Identifier: Apache-2.0
"""sglang-native Higgs TTS model tests (PR2b).

Focuses on the wiring that PR2b adds on top of PR2a's fused modules:
- Registration with sglang's ``ModelRegistry`` and ``AutoConfig``.
- Model instantiation from a ``HiggsMultimodalQwen3Config``.
- Weight-loading name remapping from the Higgs checkpoint layout to the
  composition ``backbone.model.*`` / ``multimodal_embedding.*`` tree.

Full multi-codebook forward + sampling is covered by PR4.
"""

from __future__ import annotations

import os

import pytest

from sglang_omni.models.higgs_tts.hf_config import HiggsMultimodalQwen3Config


def _init_sglang_tp() -> None:
    """Single-rank TP=1 / PP=1 sglang context so `Qwen3ForCausalLM` can
    construct on a single GPU (mirrors tests/test_ming_omni_vision_e2e.py).

    Skips if CUDA is unavailable — sglang's attention layers allocate CUDA
    buffers at construction time.
    """
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
        pass  # Already initialised by an earlier test in this session.
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


ARCHITECTURE = "HiggsMultimodalQwen3ForConditionalGeneration"

# Minimal Qwen3 text config — values are the smallest sglang will accept that
# still exercise the qkv / gate_up stacking logic in load_weights.
_TINY_TEXT_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 1024,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 128,
    "tie_word_embeddings": True,
}

_TINY_AUDIO_ENCODER_CONFIG = {
    "encoder_type": "discrete",
    "num_codebooks": 4,
    "vocab_size": 16,
    "out_dim": 64,
    "tie_word_embeddings": True,
}


def _make_tiny_config() -> HiggsMultimodalQwen3Config:
    return HiggsMultimodalQwen3Config(
        audio_encoder_config=_TINY_AUDIO_ENCODER_CONFIG,
        text_config=_TINY_TEXT_CONFIG,
        audio_token_id=-100,
        architectures=[ARCHITECTURE],
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_omni_models_adds_higgs_entry():
    """``register_omni_models_in_sglang`` registers the Higgs class under the
    HF architecture name and registers the HF config with ``AutoConfig``."""
    from sglang.srt.models.registry import ModelRegistry
    from transformers import AutoConfig

    from sglang_omni.models.higgs_tts.model import HiggsTTSModel
    from sglang_omni.models.sglang_registry import register_omni_models_in_sglang

    register_omni_models_in_sglang()

    assert ModelRegistry.models.get(ARCHITECTURE) is HiggsTTSModel

    # AutoConfig entry — constructible from dict with ``model_type``.
    cfg = AutoConfig.for_model(
        "higgs_multimodal_qwen3",
        audio_encoder_config=_TINY_AUDIO_ENCODER_CONFIG,
        text_config=_TINY_TEXT_CONFIG,
    )
    assert isinstance(cfg, HiggsMultimodalQwen3Config)


def test_text_config_patches_null_rope_theta_to_qwen3_default():
    """Higgs checkpoints ship ``rope_theta: null`` in the text sub-config.
    ``Qwen3Config(**dict)`` then falls back to transformers' Qwen3 default of
    ``10000`` — but Qwen3 actually trains with ``1e6``. The config layer
    must patch that back to ``1_000_000`` (matching boson-vllm's
    ``set_default_rope_theta`` behaviour), otherwise positional encoding
    is silently wrong and the model diverges from boson-vllm (seed-tts
    greedy codes disagree starting row 2)."""
    null_rope_text_config = dict(_TINY_TEXT_CONFIG)
    null_rope_text_config["rope_theta"] = None

    cfg = HiggsMultimodalQwen3Config(
        audio_encoder_config=_TINY_AUDIO_ENCODER_CONFIG,
        text_config=null_rope_text_config,
    )
    assert cfg.get_text_config().rope_theta == 1_000_000


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_model_instantiates_with_tiny_config():
    """Construct the model with a tiny Qwen3-ish backbone and discrete head.

    Requires GPU (sglang's RadixAttention initialises CUDA buffers at
    construction time)."""
    _init_sglang_tp()

    from sglang_omni.models.higgs_tts.model import HiggsTTSModel

    cfg = _make_tiny_config()
    model = HiggsTTSModel(cfg)

    # Submodule layout expected by the Higgs checkpoint name mapping.
    assert hasattr(model, "backbone")
    assert hasattr(model.backbone, "model")
    assert hasattr(model.backbone, "lm_head")
    assert hasattr(model.multimodal_embedding, "modality_embedding_0")
    assert hasattr(model, "modality_head")

    # Fused embedding shape.
    emb_weight = model.multimodal_embedding.modality_embedding_0.weight
    assert emb_weight.shape == (
        _TINY_AUDIO_ENCODER_CONFIG["num_codebooks"]
        * _TINY_AUDIO_ENCODER_CONFIG["vocab_size"],
        _TINY_AUDIO_ENCODER_CONFIG["out_dim"],
    )

    # ``tie_word_embeddings=True`` must share storage between head and embedding.
    assert model.modality_head.weight is emb_weight

    assert model.num_codebooks == _TINY_AUDIO_ENCODER_CONFIG["num_codebooks"]
    assert model.codebook_vocab_size == _TINY_AUDIO_ENCODER_CONFIG["vocab_size"]


def test_untied_modality_head_is_separate_param():
    """With ``tie_word_embeddings=False``, the head has its own tensor."""
    _init_sglang_tp()

    from sglang_omni.models.higgs_tts.model import HiggsTTSModel

    enc_cfg = dict(_TINY_AUDIO_ENCODER_CONFIG, tie_word_embeddings=False)
    cfg = HiggsMultimodalQwen3Config(
        audio_encoder_config=enc_cfg,
        text_config=_TINY_TEXT_CONFIG,
        audio_token_id=-100,
        architectures=[ARCHITECTURE],
    )
    model = HiggsTTSModel(cfg)

    emb_weight = model.multimodal_embedding.modality_embedding_0.weight
    assert model.modality_head.weight is not emb_weight
    assert model.modality_head.weight.shape == emb_weight.shape


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _iter_synthetic_checkpoint(text_cfg: dict, audio_cfg: dict):
    """Build a tiny synthetic Higgs checkpoint mimicking the real one.

    Yields ``(checkpoint_name, tensor)`` pairs in the layout the real Higgs
    checkpoint uses (``tied.embedding.*``, ``body.layers.*``, etc.).
    """
    import torch

    h = text_cfg["hidden_size"]
    kv = text_cfg["num_key_value_heads"]
    head_dim = text_cfg["head_dim"]
    inter = text_cfg["intermediate_size"]
    vocab = text_cfg["vocab_size"]

    yield "tied.embedding.text_embedding.weight", torch.randn(vocab, h)
    for li in range(text_cfg["num_hidden_layers"]):
        p = f"body.layers.{li}"
        yield f"{p}.self_attn.q_proj.weight", torch.randn(
            text_cfg["num_attention_heads"] * head_dim, h
        )
        yield f"{p}.self_attn.k_proj.weight", torch.randn(kv * head_dim, h)
        yield f"{p}.self_attn.v_proj.weight", torch.randn(kv * head_dim, h)
        yield f"{p}.self_attn.o_proj.weight", torch.randn(
            h, text_cfg["num_attention_heads"] * head_dim
        )
        yield f"{p}.self_attn.q_norm.weight", torch.randn(head_dim)
        yield f"{p}.self_attn.k_norm.weight", torch.randn(head_dim)
        yield f"{p}.mlp.gate_proj.weight", torch.randn(inter, h)
        yield f"{p}.mlp.up_proj.weight", torch.randn(inter, h)
        yield f"{p}.mlp.down_proj.weight", torch.randn(h, inter)
        yield f"{p}.input_layernorm.weight", torch.randn(h)
        yield f"{p}.post_attention_layernorm.weight", torch.randn(h)
    yield "body.norm.weight", torch.randn(h)

    # Fused multi-codebook embedding weight (always present; checkpoint key
    # uses the nested ``embedding`` suffix per boson-vllm instance map).
    fused_rows = audio_cfg["num_codebooks"] * audio_cfg["vocab_size"]
    yield (
        "tied.embedding.modality_embeddings.0.embedding.weight",
        torch.randn(fused_rows, audio_cfg["out_dim"]),
    )

    # Simulated audio-tokenizer backbone weight — mapper should skip this.
    yield (
        "tied.embedding.modality_embeddings.0.model.encoder.conv.weight",
        torch.randn(8, 8, 3),
    )


def test_load_weights_routes_backbone_and_multimodal():
    _init_sglang_tp()

    import torch

    from sglang_omni.models.higgs_tts.model import HiggsTTSModel

    cfg = _make_tiny_config()
    model = HiggsTTSModel(cfg)

    # Freeze a copy of the fused embedding weight before loading so we can
    # confirm the checkpoint value overwrote it.
    emb_param = model.multimodal_embedding.modality_embedding_0.weight
    before = emb_param.detach().clone()

    weights = list(
        _iter_synthetic_checkpoint(_TINY_TEXT_CONFIG, _TINY_AUDIO_ENCODER_CONFIG)
    )
    # Pull out the fused embedding tensor for later comparison.
    fused_from_ckpt = next(
        t
        for n, t in weights
        if n == "tied.embedding.modality_embeddings.0.embedding.weight"
    )

    # Snapshot backbone embed_tokens so we can confirm the ckpt overwrote it.
    embed_before = model.backbone.model.embed_tokens.weight.detach().clone()
    embed_from_ckpt = next(
        t for n, t in weights if n == "tied.embedding.text_embedding.weight"
    )

    loaded = model.load_weights(weights)

    # Only own params appear in the returned set (multimodal embedding; head
    # is tied so its weight lives under the embedding name).
    assert loaded == {"multimodal_embedding.modality_embedding_0.weight"}

    # Multimodal embedding was overwritten by the ckpt tensor.
    torch.testing.assert_close(
        emb_param.detach().cpu().float(),
        fused_from_ckpt.to(emb_param.dtype).cpu().float(),
    )
    assert not torch.equal(emb_param.detach().cpu(), before.cpu())

    # Backbone was loaded — spot-check embed_tokens got its ckpt value (via
    # the ``body`` → ``backbone.model.embed_tokens`` remap).
    torch.testing.assert_close(
        model.backbone.model.embed_tokens.weight.detach().cpu().float(),
        embed_from_ckpt.to(model.backbone.model.embed_tokens.weight.dtype)
        .cpu()
        .float(),
    )
    assert not torch.equal(
        model.backbone.model.embed_tokens.weight.detach().cpu(), embed_before.cpu()
    )
