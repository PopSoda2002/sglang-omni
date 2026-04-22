# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Higgs TTS pipeline.

- ``create_preprocessing_executor``: text + reference audio (raw
  waveform OR pre-encoded codes) → prompt ids with ``-100`` placeholders,
  wrapped in a :class:`HiggsTtsState` on the payload. Server-side audio
  encoding uses :class:`HiggsAudioCodec` (PR3b); pre-encoded codes remain
  the fast path for clients that already have them.
- ``create_sglang_tts_engine_executor`` (PR4c): runs the Higgs TTS model
  under sglang's engine and returns ``[L, num_codebooks]`` codes.
- ``create_vocoder_executor`` (PR5, stub).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from sglang_omni.models.higgs_tts.delay_pattern import apply_delay_pattern
from sglang_omni.models.higgs_tts.io import HiggsTtsState
from sglang_omni.models.higgs_tts.pipeline.state_io import store_state
from sglang_omni.models.higgs_tts.tokenizer import HiggsTokenizerAdapter
from sglang_omni.proto import StagePayload


def _to_codes_TN(raw: Any, num_codebooks: int) -> torch.Tensor | None:
    """Coerce request input to an ``[T, num_codebooks]`` int64 tensor."""
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


def _load_audio_to_24k(
    reference_audio: Any,
) -> tuple[np.ndarray, int] | None:
    """Normalise an ``inputs["reference_audio"]`` entry to a 24 kHz mono
    ``float32`` numpy array.

    Accepts:
    - ``None`` → returns ``None`` (no ref audio).
    - ``str`` / :class:`Path` → treated as a filesystem path.
    - ``dict`` with ``audio_path`` or ``path`` → filesystem path.
    - ``dict`` with ``bytes`` (raw) or ``base64`` / ``data`` → in-memory audio.
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
            # ``get(...) or ...[key]`` short-circuits past the empty string
            # and would ``KeyError`` when only one key is present with an
            # empty value. Pick explicitly, then validate.
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
        "reference_audio must be a path, a dict with "
        "{audio_path|path|bytes|base64|data}, or None"
    )


def build_preprocess_fn(
    adapter: HiggsTokenizerAdapter,
    *,
    num_codebooks: int,
    codebook_size: int,
    audio_codec: Any | None = None,
):
    """Return the ``(payload) -> payload`` closure used by the preprocessing
    stage. Exposed so tests can drive it with a stub tokenizer adapter.

    When ``audio_codec`` is provided, ``inputs["reference_audio"]`` is
    accepted alongside (or instead of) pre-encoded ``reference_codes``
    and encoded server-side into ``[T, num_codebooks]`` codes. The
    pre-encoded path remains the fallback for clients that already have
    codes in hand (e.g. AReaL rollout workers).
    """

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}

        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("input") or inputs.get("text") or ""
        ref_codes_TN = _to_codes_TN(inputs.get("reference_codes"), num_codebooks)

        # Server-side encoding path: only runs when no pre-encoded codes
        # were supplied AND an audio input is present. Callers that set
        # both win with ``reference_codes`` (cheaper, deterministic).
        if ref_codes_TN is None and inputs.get("reference_audio") is not None:
            if audio_codec is None:
                raise RuntimeError(
                    "reference_audio was provided but the preprocessing stage "
                    "was built without an audio codec — pass "
                    "``audio_codec_path`` to ``create_preprocessing_executor`` "
                    "or pre-encode the reference via ``reference_codes``."
                )
            loaded = _load_audio_to_24k(inputs["reference_audio"])
            assert loaded is not None  # guarded by the ``is not None`` above
            waveform_np, sample_rate = loaded
            waveform = torch.from_numpy(waveform_np)
            ref_codes_TN = audio_codec.encode_reference(
                waveform, sample_rate=sample_rate
            ).to(torch.long)

        if ref_codes_TN is None:
            prompt_ids = adapter.build_prompt(text, num_ref_tokens=0)
            ref_codes_delayed: list[list[int]] | None = None
        else:
            delayed = apply_delay_pattern(ref_codes_TN)
            prompt_ids = adapter.build_prompt(text, num_ref_tokens=delayed.shape[0])
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
        return store_state(payload, state)

    return _preprocess


def create_preprocessing_executor(
    model_path: str,
    *,
    num_codebooks: int = 8,
    codebook_size: int = 1026,
    audio_codec_path: str | None = None,
    audio_codec_device: str = "cpu",
) -> PreprocessingExecutor:
    """Build the Higgs TTS preprocessing stage.

    ``model_path`` is a Higgs checkpoint directory or an HF repo id — passed
    straight to ``PreTrainedTokenizerFast.from_pretrained``.

    ``audio_codec_path``: optional path to a Higgs Audio V2 tokenizer
    checkpoint (e.g. ``bosonai/higgs-audio-v2-tokenizer`` or the local
    mirror at ``/ceph/models/eustlb__higgs-audio-v2-tokenizer``). When
    set, the preprocessing stage accepts raw ``reference_audio`` inputs
    and encodes them server-side. When ``None``, only pre-encoded
    ``reference_codes`` are accepted (legacy / fast path).
    """
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    adapter = HiggsTokenizerAdapter(tokenizer)

    codec: Any | None = None
    if audio_codec_path is not None:
        from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec

        codec = HiggsAudioCodec.from_pretrained(
            audio_codec_path, device=audio_codec_device
        )

    return PreprocessingExecutor(
        build_preprocess_fn(
            adapter,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            audio_codec=codec,
        )
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    max_new_tokens: int = 2048,
    mem_fraction_static: float = 0.85,
    chunked_prefill_size: int = 8192,
    max_running_requests: int = 16,
) -> EngineExecutor:
    """Build the Higgs TTS engine stage.

    Wraps :func:`create_higgs_sglang_engine` — which runs
    :class:`HiggsTTSModel` under sglang's ``ModelWorker`` + ``OmniEngine``
    — in an :class:`EngineExecutor` with request / result adapters that
    translate between :class:`HiggsTtsState` and the sglang request data.
    """
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.models.higgs_tts.factory import create_higgs_sglang_engine
    from sglang_omni.models.higgs_tts.hf_config import HiggsMultimodalQwen3Config
    from sglang_omni.models.higgs_tts.pipeline.state_io import load_state, store_state
    from sglang_omni.models.higgs_tts.runtime.higgs_sglang_ar import (
        HiggsSGLangRequestData,
    )
    from sglang_omni.models.sglang_registry import register_omni_models_in_sglang

    # Register with ``AutoConfig`` before any ``from_pretrained`` call on
    # a Higgs checkpoint — transformers resolves the ``model_type`` field
    # through ``CONFIG_MAPPING`` and bails if we didn't register first.
    register_omni_models_in_sglang()

    # Skip loading the HF tokenizer — Higgs ckpts carry ``transformers_version=5.2.0``
    # metadata with a list-shaped ``extra_special_tokens``, which pinned
    # transformers<5 refuses to load. We only need ``vocab_size`` for the
    # sglang Req, and that's in the config's text sub-config.
    _cfg = HiggsMultimodalQwen3Config.from_pretrained(model_path)
    _text_vocab_size: int = int(_cfg.get_text_config().vocab_size)

    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    server_args = ServerArgs(
        model_path=model_path,
        tp_size=1,
        dtype="bfloat16",
        mem_fraction_static=mem_fraction_static,
        chunked_prefill_size=chunked_prefill_size,
        max_running_requests=max_running_requests,
        disable_cuda_graph=True,
    )
    engine = create_higgs_sglang_engine(
        server_args,
        gpu_id=gpu_id,
        max_new_tokens=max_new_tokens,
    )

    def _request_builder(payload: StagePayload) -> HiggsSGLangRequestData:
        state = load_state(payload)
        input_ids_list = list(state.prompt_token_ids)
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)

        ref_codes = state.reference_codes_delayed
        if ref_codes is not None:
            ref_codes = torch.tensor(ref_codes, dtype=torch.long)

        sampling_params = SamplingParams(
            max_new_tokens=state.max_new_tokens,
            temperature=state.temperature,
        )
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids_list,
            sampling_params=sampling_params,
            vocab_size=_text_vocab_size,
        )
        return HiggsSGLangRequestData(
            input_ids=input_ids,
            req=req,
            reference_codes_delayed=ref_codes,
            num_codebooks=state.num_codebooks,
            codebook_size=state.codebook_size,
            max_new_tokens=state.max_new_tokens,
            temperature=state.temperature,
            top_p=state.top_p,
            top_k=state.top_k,
        )

    def _result_builder(
        payload: StagePayload, result: HiggsSGLangRequestData
    ) -> StagePayload:
        state = load_state(payload)
        if result.output_codes:
            # result.output_codes is a list of [num_codebooks] tensors —
            # stack to [L, N] and serialise.
            codes = torch.stack(result.output_codes, dim=0).to(torch.long)
            state.output_codes_delayed = codes.tolist()
            state.completion_tokens = codes.shape[0]
        else:
            state.output_codes_delayed = None
        state.prompt_tokens = (
            int(result.input_ids.shape[0]) if result.input_ids is not None else 0
        )
        payload = store_state(payload, state)
        payload.data["usage"] = {
            "prompt_tokens": state.prompt_tokens,
            "completion_tokens": state.completion_tokens,
            "total_tokens": state.prompt_tokens + state.completion_tokens,
        }
        return payload

    return EngineExecutor(
        engine=engine,
        request_builder=_request_builder,
        result_builder=_result_builder,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cpu",
) -> PreprocessingExecutor:
    del model_path, device
    raise NotImplementedError(
        "Higgs TTS vocoder stage is not implemented yet (planned for PR5)."
    )
