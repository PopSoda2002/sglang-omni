# SPDX-License-Identifier: Apache-2.0
"""Multilingual TTS WER eval — Omnilingual ASR + Higgs multilingual WER.

This module is the transcribe-and-score half of the
``benchmark_tts_higgs_multilingual`` flow. The generation half reuses
``tasks.tts.make_tts_send_fn`` / ``save_speed_results`` / ``BenchmarkRunner``
unchanged; only the ASR + WER pieces differ from the seed-tts flow:

  1. ASR runs through ``OmnilingualASR`` (subprocess into an isolated env).
  2. WER uses ``calculate_wer_higgs_multilingual`` with per-language
     Omnilingual normalization + CJK space cleanup.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Sequence

from tqdm import tqdm

from benchmarks.benchmarker.utils import save_json_results
from benchmarks.metrics.wer import (
    calculate_asr_speed_metrics,
    calculate_wer_higgs_multilingual,
    calculate_wer_metrics,
    print_asr_speed_summary,
    print_wer_summary,
)
from benchmarks.tasks.omnilingual_asr import OmnilingualASR
from benchmarks.tasks.tts import SampleOutput, save_wer_results

logger = logging.getLogger(__name__)


@dataclass
class MultilingualTranscribeConfig:
    model: str
    output_dir: str
    lang: str
    device: str = "cuda:0"
    omni_asr_python: str | None = None
    omni_asr_model_card: str = OmnilingualASR.DEFAULT_MODEL_CARD
    omni_asr_batch_size: int = 8


def _build_sample_outputs(generated: list[dict]) -> list[SampleOutput]:
    outputs: list[SampleOutput] = []
    for entry in generated:
        outputs.append(
            SampleOutput(
                sample_id=entry.get("sample_id") or entry.get("id") or "",
                target_text=entry.get("target_text", ""),
                audio_duration_s=float(entry.get("audio_duration_s") or 0.0),
                latency_s=float(entry.get("latency_s") or 0.0),
            )
        )
    return outputs


def _wav_path_for(entry: dict) -> str | None:
    """Resolve the saved WAV path for a generation entry."""
    for key in ("wav_path", "saved_wav_path", "audio_path"):
        value = entry.get(key)
        if value:
            return value
    return None


def run_multilingual_transcribe(
    config: MultilingualTranscribeConfig,
    *,
    wer_config: dict,
    generation_mode: str | None = None,
) -> dict:
    """Transcribe saved audio with Omnilingual ASR and compute multilingual WER.

    Returns a dict with keys ``wer_summary``, ``asr_speed``, ``per_sample``.
    """
    generated_path = os.path.join(config.output_dir, "generated.json")
    with open(generated_path) as f:
        generated: list[dict] = json.load(f)
    logger.info(f"Loaded {len(generated)} entries from {generated_path}")

    outputs = _build_sample_outputs(generated)
    wav_paths: list[str | None] = [_wav_path_for(e) for e in generated]

    asr = OmnilingualASR(
        python_exe=config.omni_asr_python,
        model_card=config.omni_asr_model_card,
        batch_size=config.omni_asr_batch_size,
    )

    # Skip entries whose audio went missing (generation failure).
    transcribe_indices = [i for i, p in enumerate(wav_paths) if p]
    transcribe_paths = [wav_paths[i] for i in transcribe_indices]

    logger.info(
        f"Transcribing {len(transcribe_paths)} / {len(generated)} samples "
        f"via Omnilingual ASR ({config.omni_asr_model_card})"
    )
    asr_t0 = time.perf_counter()
    transcriptions = asr.transcribe(transcribe_paths, config.lang, config.device)
    asr_total_s = time.perf_counter() - asr_t0

    # Spread batch latency uniformly so per-sample ASR speed numbers are
    # comparable to seed-tts (sequential) without lying about throughput.
    per_sample_asr_s = (
        asr_total_s / len(transcribe_paths) if transcribe_paths else 0.0
    )

    trans_map: dict[int, str] = dict(zip(transcribe_indices, transcriptions))
    for idx, output in enumerate(tqdm(outputs, desc=f"WER ({config.lang})")):
        hyp = trans_map.get(idx, "")
        if not wav_paths[idx]:
            output.error = "No audio in response"
            continue
        if hyp is None:
            hyp = ""
        output.whisper_text = hyp
        output.asr_latency_s = per_sample_asr_s

        try:
            measures, ref_norm, hyp_norm = calculate_wer_higgs_multilingual(
                output.target_text, hyp, config.lang
            )
        except Exception as exc:
            output.error = f"WER scoring failed: {exc}"
            continue

        output.ref_norm = ref_norm
        output.hyp_norm = hyp_norm
        if not output.ref_norm:
            output.error = "Empty reference after normalization"
            continue

        output.wer = measures.wer
        output.substitutions = measures.substitutions
        output.deletions = measures.deletions
        output.insertions = measures.insertions
        output.hits = measures.hits
        output.is_success = True

    wer_metrics = calculate_wer_metrics(outputs, config.lang)
    asr_metrics = calculate_asr_speed_metrics(outputs)

    print_asr_speed_summary(asr_metrics, config.model)
    print_wer_summary(wer_metrics, config.model, generation_mode)

    save_wer_results(outputs, wer_metrics, wer_config, config.output_dir)
    save_json_results(asr_metrics, config.output_dir, "asr_speed_results.json")

    return {
        "wer_summary": wer_metrics,
        "asr_speed": asr_metrics,
        "per_sample": outputs,
    }
