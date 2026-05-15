# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Higgs TTS pipeline (V1).

Pipeline shape::

    preprocessing → audio_encoder → tts_engine → vocoder

- ``create_preprocessing_executor``: text + reference audio (raw waveform OR
  pre-encoded codes) → prompt ids with ``-100`` placeholders + delayed
  ref-audio codes on the state. Returns a
  :class:`ThreadedSimpleScheduler` for CPU-heavy work.
- ``create_audio_encoder_executor``: runs the fused multi-codebook embedding
  on the delayed ref-audio codes to produce a ``[num_ref_tokens,
  hidden_size]`` tensor stashed as ``state.reference_audio_embed``.
  Returns a :class:`SimpleScheduler`.
- ``create_sglang_tts_engine_executor``: runs :class:`HiggsTTSModel` under
  sglang's worker; returns a :class:`HiggsScheduler` driving the AR loop.
- ``create_vocoder_executor``: reverses the delay pattern, decodes via
  :class:`HiggsAudioCodec` into a mono 24 kHz waveform attached to the
  payload. Returns a :class:`SimpleScheduler`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sglang_omni.models.higgs_tts.delay_pattern import (
    apply_delay_pattern,
    reverse_delay_pattern,
)
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.tokenizer import HiggsTokenizerAdapter
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


# Default open-source audio codec.
DEFAULT_AUDIO_CODEC = "bosonai/higgs-audio-v2-tokenizer"

# Shared codec instances between preprocessing and vocoder stages — saves
# ~1 GB of VRAM + one load pass at server startup.
_CODEC_CACHE: dict[tuple[str, str, str], Any] = {}


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _get_or_load_codec(path: str, device: str, dtype: str) -> Any:
    from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec

    key = (str(path), str(device), str(dtype))
    cached = _CODEC_CACHE.get(key)
    if cached is not None:
        return cached
    codec = HiggsAudioCodec.from_pretrained(
        path, device=device, dtype=getattr(torch, dtype)
    )
    _CODEC_CACHE[key] = codec
    return codec


def _to_codes_TN(raw: Any, num_codebooks: int) -> torch.Tensor | None:
    if raw is None:
        return None
    t = raw if isinstance(raw, torch.Tensor) else torch.tensor(raw)
    if t.numel() == 0:
        return None
    if t.ndim != 2 or t.shape[1] != num_codebooks:
        raise ValueError(
            f"reference_codes must have shape [T, {num_codebooks}], got {tuple(t.shape)}"
        )
    return t.to(torch.long)


def _load_audio_to_24k(reference_audio: Any) -> tuple[np.ndarray, int] | None:
    """Normalise an ``inputs["reference_audio"]`` entry to a 24 kHz mono float32
    numpy array. Accepts path, ``{audio_path|path|bytes|base64|data}`` dict, or None.
    """
    if reference_audio is None:
        return None

    from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec
    from sglang_omni.preprocessing.audio import AudioMediaIO

    io = AudioMediaIO(target_sr=HiggsAudioCodec.SAMPLE_RATE)

    if isinstance(reference_audio, (str, Path)):
        audio, sr = io.load_file(Path(reference_audio))
        return np.asarray(audio, dtype=np.float32), int(sr)

    if isinstance(reference_audio, dict):
        if "audio_path" in reference_audio or "path" in reference_audio:
            path = reference_audio.get("audio_path") or reference_audio.get("path")
            if not path:
                raise ValueError("reference_audio dict has an empty audio_path/path")
            audio, sr = io.load_file(Path(path))
            return np.asarray(audio, dtype=np.float32), int(sr)
        if "bytes" in reference_audio:
            audio, sr = io.load_bytes(reference_audio["bytes"])
            return np.asarray(audio, dtype=np.float32), int(sr)
        if "base64" in reference_audio or "data" in reference_audio:
            media_type = reference_audio.get("media_type", "audio/wav")
            data = reference_audio.get("base64") or reference_audio.get("data")
            if not data:
                raise ValueError("reference_audio dict has an empty base64/data value")
            audio, sr = io.load_base64(media_type, data)
            return np.asarray(audio, dtype=np.float32), int(sr)

    raise TypeError(
        "reference_audio must be a path, a "
        "{audio_path|path|bytes|base64|data} dict, or None"
    )


def create_preprocessing_executor(
    model_path: str,
    *,
    num_codebooks: int = 8,
    codebook_size: int = 1026,
    audio_codec_path: str | None = None,
    audio_codec_device: str = "cpu",
    audio_codec_dtype: str = "bfloat16",
    max_concurrency: int = 8,
):
    """Threaded scheduler for CPU-heavy preprocessing."""
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler

    checkpoint_dir = _resolve_checkpoint(model_path)

    # Higgs ckpts ship ``tokenizer_config.json`` with transformers v5 metadata
    # that crashes transformers<5's ``from_pretrained``. Load tokenizer.json
    # directly via the ``tokenizers`` library — schema-version-independent —
    # and wrap in a PreTrainedTokenizerFast.
    raw = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw)
    adapter = HiggsTokenizerAdapter(tokenizer)

    codec_src = audio_codec_path or DEFAULT_AUDIO_CODEC
    codec = _get_or_load_codec(codec_src, audio_codec_device, audio_codec_dtype)

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("input") or inputs.get("text") or ""
        reference_text = inputs.get("reference_text") or None
        ref_codes_TN = _to_codes_TN(inputs.get("reference_codes"), num_codebooks)

        if ref_codes_TN is None and inputs.get("reference_audio") is not None:
            loaded = _load_audio_to_24k(inputs["reference_audio"])
            assert loaded is not None
            waveform_np, sample_rate = loaded
            waveform = torch.from_numpy(waveform_np)
            ref_codes_TN = codec.encode_reference(waveform, sample_rate=sample_rate).to(
                torch.long
            )

        if ref_codes_TN is None:
            prompt_ids = adapter.build_prompt(
                text, num_ref_tokens=0, reference_text=reference_text
            )
            ref_codes_delayed: list[list[int]] | None = None
        else:
            delayed = apply_delay_pattern(ref_codes_TN)
            prompt_ids = adapter.build_prompt(
                text,
                num_ref_tokens=delayed.shape[0],
                reference_text=reference_text,
            )
            ref_codes_delayed = delayed.tolist()

        state = HiggsTtsState(
            prompt_token_ids=prompt_ids,
            reference_codes_delayed=ref_codes_delayed,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            max_new_tokens=int(params.get("max_new_tokens", 2048)),
            temperature=float(params.get("temperature", 1.0)),
            top_p=params.get("top_p"),
            top_k=params.get("top_k"),
            seed=params.get("seed"),
        )
        payload.data = state.to_dict()
        return payload

    return ThreadedSimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    num_codebooks: int = 8,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
):
    """Run the fused multi-codebook embedding on delayed ref codes.

    Stashes ``state.reference_audio_embed`` for the engine stage's prefill
    overlay to paste at ``-100`` placeholder positions.
    """
    from sglang_omni.models.higgs_tts.bootstrap import (
        load_fused_embedding_from_tts_ckpt,
    )
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    checkpoint_dir = _resolve_checkpoint(model_path)
    fused = load_fused_embedding_from_tts_ckpt(checkpoint_dir, device=device)

    def _encode(payload: StagePayload) -> StagePayload:
        state = HiggsTtsState.from_dict(payload.data)
        codes_rows = state.reference_codes_delayed
        if not codes_rows:
            return payload  # zero-shot

        codes = torch.tensor(codes_rows, dtype=torch.long)
        if codes.ndim != 2 or codes.shape[1] != num_codebooks:
            raise ValueError(
                f"reference_codes_delayed must be [T, {num_codebooks}], "
                f"got shape {tuple(codes.shape)}"
            )
        with torch.no_grad():
            embed = fused(codes.to(device))  # [T, hidden_size]
        # CPU fp32 tensor — relay_io.extract_tensors ships it on the raw
        # tensor buffer instead of pickling, and fp32 round-trips bf16
        # exactly when the engine recasts during overlay.
        state.reference_audio_embed = embed.float().cpu()
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(
        _encode,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    max_new_tokens: int = 2048,
    server_args_overrides: dict[str, Any] | None = None,
):
    """sglang-backed AR engine for Higgs TTS."""
    from sglang_omni.models.higgs_tts.bootstrap import (
        register_higgs_tts_in_sglang,
        truncate_rope_to_bf16,
    )
    from sglang_omni.models.higgs_tts.higgs_scheduler import HiggsScheduler
    from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
    from sglang_omni.models.higgs_tts.request_builders import (
        make_higgs_scheduler_adapters,
    )
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    register_higgs_tts_in_sglang()

    checkpoint_dir = _resolve_checkpoint(model_path)
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    overrides: dict[str, Any] = {
        # CUDA graph not supported — Higgs's per-request slot state lives
        # outside the captured graph, and the multi-codebook decode runs
        # in a Python loop.
        "disable_cuda_graph": True,
        "mem_fraction_static": 0.85,
        "max_running_requests": 16,
        "chunked_prefill_size": 8192,
        "dtype": "bfloat16",
        # Disable radix cache: TTS prompts share the
        # ``<|tts|> <|ref_audio|> [-100]...`` token-id prefix but the -100
        # positions are overlaid with different ref-audio embeddings per
        # request, so caching by token id alone cross-contaminates.
        "disable_radix_cache": True,
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=4096,
        **overrides,
    )
    server_args.disable_overlap_schedule = True

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        _model_config,
    ) = create_sglang_infrastructure(server_args, gpu_id)

    truncate_rope_to_bf16(model_worker.model_runner.model)

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    model_runner = HiggsTTSModelRunner(model_worker, output_proc)
    request_builder, result_adapter = make_higgs_scheduler_adapters()

    return HiggsScheduler(
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        server_args=server_args,
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        max_new_tokens=max_new_tokens,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    audio_codec_path: str | None = None,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_batch_size: int = 4,
    max_batch_wait_ms: int = 2,
):
    """Decode Higgs delayed codes to a mono 24 kHz waveform."""
    from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    del model_path  # vocoder needs only the codec, not the TTS ckpt
    codec_src = audio_codec_path or DEFAULT_AUDIO_CODEC
    codec = _get_or_load_codec(codec_src, device, dtype)
    sample_rate = HiggsAudioCodec.SAMPLE_RATE

    def _vocode(payload: StagePayload) -> StagePayload:
        state = HiggsTtsState.from_dict(payload.data)
        delayed_rows = state.output_codes_delayed

        if not delayed_rows:
            payload.data["audio_data"] = []
            payload.data["sample_rate"] = sample_rate
            payload.data["modality"] = "audio"
            return payload

        delayed_LN = torch.tensor(delayed_rows, dtype=torch.long)
        N = state.num_codebooks
        if delayed_LN.shape[0] < N:
            payload.data["audio_data"] = []
            payload.data["sample_rate"] = sample_rate
            payload.data["modality"] = "audio"
            return payload

        codes_TN = reverse_delay_pattern(delayed_LN)
        codec_vocab = state.codebook_size - 2  # 1026 - BOC - EOC
        codes_TN = torch.where(
            codes_TN >= codec_vocab, torch.zeros_like(codes_TN), codes_TN
        )
        waveform = codec.decode(codes_TN)
        audio_np = waveform.detach().to(torch.float32).cpu().numpy()

        payload.data["audio_data"] = audio_np.tolist()
        payload.data["sample_rate"] = sample_rate
        payload.data["modality"] = "audio"
        if state.prompt_tokens or state.completion_tokens or state.engine_time_s:
            usage = {
                "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
                "total_tokens": state.prompt_tokens + state.completion_tokens,
            }
            if state.engine_time_s:
                usage["engine_time_s"] = round(state.engine_time_s, 6)
            payload.data["usage"] = usage
        return payload

    return SimpleScheduler(
        _vocode,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


__all__ = [
    "create_audio_encoder_executor",
    "create_preprocessing_executor",
    "create_sglang_tts_engine_executor",
    "create_vocoder_executor",
]
