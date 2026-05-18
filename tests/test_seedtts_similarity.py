# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.seedtts import SampleInput
from benchmarks.metrics.speaker_similarity import (
    SpeakerSimilarityOutput,
    calculate_speaker_similarity_metrics,
)
from benchmarks.tasks.tts import run_seedtts_similarity, save_generated_audio_metadata


def test_calculate_speaker_similarity_metrics_uses_successful_samples_only() -> None:
    outputs = [
        SpeakerSimilarityOutput(sample_id="a", similarity=60.0, is_success=True),
        SpeakerSimilarityOutput(sample_id="b", similarity=80.0, is_success=True),
        SpeakerSimilarityOutput(sample_id="c", error="missing ref"),
    ]

    metrics = calculate_speaker_similarity_metrics(outputs)

    assert metrics["total_samples"] == 3
    assert metrics["evaluated"] == 2
    assert metrics["skipped"] == 1
    assert metrics["speaker_similarity_mean"] == pytest.approx(70.0)
    assert metrics["speaker_similarity_median"] == pytest.approx(70.0)
    assert metrics["speaker_similarity_min"] == pytest.approx(60.0)
    assert metrics["speaker_similarity_max"] == pytest.approx(80.0)


def test_save_generated_audio_metadata_includes_reference_audio(tmp_path) -> None:
    output = RequestResult(
        request_id="utt1",
        is_success=True,
        latency_s=1.25,
        audio_duration_s=2.5,
        wav_path=str(tmp_path / "utt1.wav"),
    )
    sample = SampleInput(
        sample_id="utt1",
        ref_text="prompt text",
        ref_audio="prompt.wav",
        target_text="target text",
    )

    save_generated_audio_metadata([output], [sample], str(tmp_path))

    with open(tmp_path / "generated.json") as f:
        generated = json.load(f)

    assert generated[0]["ref_audio"] == "prompt.wav"
    assert generated[0]["ref_text"] == "prompt text"


@dataclass
class _SimilarityConfig:
    model: str
    meta: str
    output_dir: str
    device: str = "cpu"
    similarity_model: str = "fake-speaker-model"
    similarity_device: str | None = None


class _FakeSpeakerSimilarityScorer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def score(self, ref_audio: str, wav_path: str) -> float:
        self.calls.append((ref_audio, wav_path))
        return 61.5


def test_run_seedtts_similarity_uses_meta_reference_and_writes_results(
    tmp_path,
) -> None:
    dataset_dir = tmp_path / "seedtts" / "en"
    dataset_dir.mkdir(parents=True)
    ref_audio = dataset_dir / "prompt.wav"
    hyp_audio = tmp_path / "audio" / "utt1.wav"
    hyp_audio.parent.mkdir()
    ref_audio.write_bytes(b"ref")
    hyp_audio.write_bytes(b"hyp")

    meta_path = dataset_dir / "meta.lst"
    meta_path.write_text(
        "utt1|prompt text|prompt.wav|target text\n"
        "utt2|other prompt|other.wav|other target\n"
    )
    generated = [
        {
            "sample_id": "utt1",
            "target_text": "target text",
            "wav_path": str(hyp_audio),
            "is_success": True,
            "latency_s": 1.0,
            "audio_duration_s": 2.0,
        },
        {
            "sample_id": "utt2",
            "target_text": "other target",
            "wav_path": "",
            "is_success": False,
            "error": "generation failed",
        },
    ]
    (tmp_path / "generated.json").write_text(json.dumps(generated))
    scorer = _FakeSpeakerSimilarityScorer()

    results = run_seedtts_similarity(
        _SimilarityConfig(
            model="unit-model",
            meta=str(meta_path),
            output_dir=str(tmp_path),
        ),
        similarity_config={"model": "unit-model"},
        scorer=scorer,
    )

    assert scorer.calls == [(str(ref_audio), str(hyp_audio))]
    assert results["summary"]["evaluated"] == 1
    assert results["summary"]["skipped"] == 1
    assert results["summary"]["speaker_similarity_mean"] == pytest.approx(61.5)

    with open(tmp_path / "similarity_results.json") as f:
        saved = json.load(f)
    assert saved["per_sample"][0]["speaker_similarity"] == pytest.approx(61.5)
    assert saved["per_sample"][1]["error"] == "Generation failed: generation failed"
