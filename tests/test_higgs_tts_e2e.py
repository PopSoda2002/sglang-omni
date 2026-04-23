# SPDX-License-Identifier: Apache-2.0
"""End-to-end pipeline test for Higgs TTS (PR5).

Drives the full preprocessing → tts_engine → vocoder stack with a real
reference clip from the seed-tts-eval dataset and verifies a plausible
mono WAV comes out the other side. Auto-skipped when the Higgs ckpt,
the codec ckpt, or the seed-tts dataset isn't mounted, or when CUDA is
unavailable.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import numpy as np
import pytest
import torch

_HIGGS_CKPT = "/hot-data/checkpoints/TTS/c66596d6cde44fb4ba8076cc2dc1e77c/step_35500"
_SEED_TTS_EN = "/ceph/data/audio_eval/tokenizer_eval/seed-tts-eval/seedtts_testset/en"
_SEED_TTS_META = os.path.join(_SEED_TTS_EN, "meta.lst")


def _pick_first_seed_tts_entry() -> tuple[str, str]:
    """Return ``(prompt_wav_path, synthesis_text)`` for the first
    seed-tts-eval English row. Format: ``id|ref_text|prompt-wavs/X.wav|synth_text``.
    """
    with open(_SEED_TTS_META) as f:
        first_line = f.readline().rstrip("\n")
    parts = first_line.split("|")
    assert len(parts) == 4, f"unexpected meta.lst format: {first_line!r}"
    _rid, _ref_text, rel_prompt_wav, synth_text = parts
    prompt_wav = os.path.join(_SEED_TTS_EN, rel_prompt_wav)
    return prompt_wav, synth_text


pytestmark = pytest.mark.skipif(
    not os.path.isdir(_HIGGS_CKPT)
    or not os.path.isfile(_SEED_TTS_META)
    or not torch.cuda.is_available(),
    reason="real Higgs ckpt / seed-tts dataset / CUDA required",
)


def _run_pipeline(prompt_wav: str, synth_text: str) -> dict:
    """Drive the full 3-stage pipeline manually and return the
    vocoder-stage payload.data dict (with ``audio_data`` / ``sample_rate``).
    """
    from sglang_omni.models.higgs_tts.pipeline.stages import (
        create_preprocessing_executor,
        create_sglang_tts_engine_executor,
        create_vocoder_executor,
    )
    from sglang_omni.proto import StagePayload
    from sglang_omni.proto.request import OmniRequest

    # Single-ckpt pipeline: preprocess + vocoder both take the codec
    # from the TTS ckpt's inlined ``tied.embedding.modality_embeddings``
    # weights. No separate audio-codec mount needed.
    preprocess = create_preprocessing_executor(_HIGGS_CKPT, audio_codec_device="cuda:0")
    engine = create_sglang_tts_engine_executor(
        _HIGGS_CKPT,
        device="cuda:0",
        max_new_tokens=256,
        mem_fraction_static=0.6,
        max_running_requests=2,
    )
    vocoder = create_vocoder_executor(_HIGGS_CKPT, device="cuda:0")

    payload = StagePayload(
        request_id=str(uuid.uuid4()),
        request=OmniRequest(
            inputs={
                "input": synth_text,
                "reference_audio": {"audio_path": prompt_wav},
            },
            params={"max_new_tokens": 256, "temperature": 0.8},
        ),
        data=None,
    )

    async def _run():
        # Preprocess + vocoder are PreprocessingExecutors — sync; engine
        # is an async EngineExecutor.
        await preprocess.start()
        try:
            await preprocess.add_request(payload)
            pp_out = await preprocess.get_result()
        finally:
            await preprocess.stop()

        await engine.start()
        try:
            await engine.add_request(pp_out)
            eng_out = await engine.get_result()
        finally:
            await engine.stop()

        await vocoder.start()
        try:
            await vocoder.add_request(eng_out)
            voc_out = await vocoder.get_result()
        finally:
            await vocoder.stop()

        return voc_out

    return asyncio.run(_run()).data


def test_full_pipeline_with_seed_tts_reference():
    """End-to-end: seed-tts reference wav + short text → WAV bytes."""
    prompt_wav, synth_text = _pick_first_seed_tts_entry()

    # Sanity-check the test fixture before the expensive engine spin-up.
    assert os.path.isfile(prompt_wav), f"missing {prompt_wav}"

    data = _run_pipeline(prompt_wav, synth_text)

    audio = data["audio_data"]
    sr = data["sample_rate"]
    modality = data["modality"]

    assert modality == "audio"
    assert sr == 24_000
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.ndim == 1

    # We asked for up to 256 new codebook-0 tokens; each codec frame is
    # 960 samples at 24 kHz (40 ms). Even with heavy early-EOC, we
    # should see at least a few hundred ms of audio out.
    min_samples = int(0.3 * sr)  # 0.3 s
    max_samples = 256 * 960  # generous upper bound
    assert (
        min_samples <= audio.shape[0] <= max_samples * 2
    ), f"audio length {audio.shape[0]} out of plausible range"

    # Must not be silence — real TTS should move the waveform around.
    assert float(np.abs(audio).max()) > 1e-3, "decoded audio is all silence"

    # Usage bookkeeping survived through the stages.
    assert "usage" in data
    assert data["usage"]["completion_tokens"] > 0
