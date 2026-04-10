# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Speech MMLU benchmark helpers."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import wave

import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset import speech_mmlu as speech_mmlu_dataset
from benchmarks.dataset.speech_mmlu import SpeechMmluSample, load_speech_mmlu_samples
from benchmarks.metrics.accuracy import compute_accuracy_metrics
from benchmarks.tasks.speech_mmlu import (
    build_speech_mmlu_results,
    make_speech_mmlu_send_fn,
)


def _make_wav_bytes(duration_s: float = 0.05, sample_rate: int = 16000) -> bytes:
    frame_count = int(duration_s * sample_rate)
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * frame_count)
        return buffer.getvalue()


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.last_request_json: dict | None = None

    def post(self, _url: str, *, json: dict, headers: dict | None = None) -> _FakeResponse:
        del headers
        self.last_request_json = json
        return _FakeResponse(self.payload, status=self.status)


def test_accuracy_metrics_count_unparseable_as_incorrect() -> None:
    metrics = compute_accuracy_metrics(
        [
            {
                "subject": "anatomy",
                "correct_answer": 1,
                "predicted_answer": 1,
                "is_correct": True,
                "is_parseable": True,
            },
            {
                "subject": "anatomy",
                "correct_answer": 2,
                "predicted_answer": 0,
                "is_correct": False,
                "is_parseable": True,
            },
            {
                "subject": "virology",
                "correct_answer": 3,
                "predicted_answer": None,
                "is_correct": False,
                "is_parseable": False,
            },
        ]
    )

    assert metrics["correct"] == 1
    assert metrics["incorrect"] == 2
    assert metrics["incorrect_parseable"] == 1
    assert metrics["unparseable_samples"] == 1


def test_build_results_parses_text_even_when_audio_generation_failed() -> None:
    sample = SpeechMmluSample(
        sample_id="sample-1",
        audio_path="/tmp/sample.mp3",
        question_text="Question",
        correct_answer=1,
        subject="anatomy",
    )
    request_result = RequestResult(
        request_id="sample-1",
        text="B",
        is_success=False,
        latency_s=0.5,
        error="No audio in response",
    )

    result = build_speech_mmlu_results(
        [request_result],
        [sample],
        ["text", "audio"],
    )[0]

    assert result.is_parseable is True
    assert result.is_correct is True
    assert result.has_audio is False
    assert result.error == "No audio in response"


def test_audio_mode_uses_transcript_fallback_and_flags_missing_audio() -> None:
    sample = SpeechMmluSample(
        sample_id="sample-1",
        audio_path="/tmp/sample.mp3",
        question_text="Question",
        correct_answer=1,
        subject="anatomy",
    )
    session = _FakeSession(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "audio": {"transcript": "B"},
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }
    )

    send_fn = make_speech_mmlu_send_fn(
        "qwen3-omni",
        "http://localhost:8000/v1/chat/completions",
        modalities=["text", "audio"],
    )
    result = asyncio.run(send_fn(session, sample))

    assert result.text == "B"
    assert result.is_success is False
    assert result.error == "No audio in response"
    assert result.completion_tokens == 1
    assert session.last_request_json["modalities"] == ["text", "audio"]


def test_audio_mode_marks_valid_audio_as_success() -> None:
    sample = SpeechMmluSample(
        sample_id="sample-2",
        audio_path="/tmp/sample.mp3",
        question_text="Question",
        correct_answer=2,
        subject="virology",
    )
    session = _FakeSession(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "audio": {
                            "data": base64.b64encode(_make_wav_bytes()).decode(),
                            "transcript": "C",
                        },
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }
    )

    send_fn = make_speech_mmlu_send_fn(
        "qwen3-omni",
        "http://localhost:8000/v1/chat/completions",
        modalities=["text", "audio"],
    )
    result = asyncio.run(send_fn(session, sample))

    assert result.text == "C"
    assert result.is_success is True
    assert result.audio_duration_s > 0
    assert result.rtf > 0


def test_load_samples_repairs_cached_audio_paths(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "speech_mmlu_cache"
    repo_dir = cache_dir / "repo"
    actual_audio = repo_dir / "audio" / "audio" / "anatomy" / "data" / "sample-1.mp3"
    actual_audio.parent.mkdir(parents=True, exist_ok=True)
    actual_audio.write_bytes(b"fake-mp3")

    broken_audio = repo_dir / "audio" / "anatomy" / "data" / "sample-1.mp3"
    meta_cache = cache_dir / "metadata.json"
    meta_cache.parent.mkdir(parents=True, exist_ok=True)
    meta_cache.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample-1",
                    "audio_path": str(broken_audio),
                    "question_text": "Question",
                    "correct_answer": 1,
                    "subject": "anatomy",
                }
            ]
        )
    )

    monkeypatch.setattr(
        speech_mmlu_dataset,
        "_ensure_dataset_downloaded",
        lambda _cache_dir: repo_dir,
    )

    samples = load_speech_mmlu_samples(cache_dir=str(cache_dir), max_samples=1)

    assert samples[0].audio_path == str(actual_audio.resolve())
    repaired = json.loads(meta_cache.read_text())
    assert repaired[0]["audio_path"] == str(actual_audio.resolve())
