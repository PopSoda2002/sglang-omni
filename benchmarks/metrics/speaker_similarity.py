# SPDX-License-Identifier: Apache-2.0
"""Speaker-similarity metric helpers for SeedTTS-style TTS evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from benchmarks.metrics._format import SPEED_LABEL_WIDTH, SPEED_LINE_WIDTH


DEFAULT_SPEAKER_SIMILARITY_MODEL = "microsoft/wavlm-base-plus-sv"


class SpeakerSimilarityScorer(Protocol):
    """Scores a reference/generated audio pair as cosine similarity x100."""

    def score(self, ref_audio: str, wav_path: str) -> float: ...


@dataclass
class SpeakerSimilarityOutput:
    sample_id: str
    ref_audio: str = ""
    wav_path: str = ""
    similarity: float | None = None
    audio_duration_s: float = 0.0
    scorer_latency_s: float = 0.0
    is_success: bool = False
    error: str = ""


class WavLMSpeakerSimilarityScorer:
    """WavLM x-vector speaker verifier backed by transformers.

    The returned score is cosine similarity multiplied by 100, matching the
    SeedTTS convention used in the benchmark docs.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_SPEAKER_SIMILARITY_MODEL,
        *,
        device: str = "cpu",
        sampling_rate: int = 16000,
    ) -> None:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.torch = torch
        self.device = device
        self.sampling_rate = sampling_rate
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name_or_path
        )
        self.model = WavLMForXVector.from_pretrained(model_name_or_path)
        self.model.to(device)
        self.model.eval()

    def _embedding(self, audio_path: str):
        import librosa

        audio, _ = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.no_grad():
            return self.model(**inputs).embeddings

    def score(self, ref_audio: str, wav_path: str) -> float:
        emb_ref = self._embedding(ref_audio)
        emb_hyp = self._embedding(wav_path)
        score = self.torch.nn.functional.cosine_similarity(
            emb_ref,
            emb_hyp,
            dim=-1,
        )
        return float(score.cpu().item() * 100.0)


def calculate_speaker_similarity_metrics(
    outputs: list[SpeakerSimilarityOutput],
) -> dict:
    """Compute aggregate speaker-similarity metrics."""
    successes = [
        output
        for output in outputs
        if output.is_success and output.similarity is not None
    ]
    if not successes:
        return {
            "total_samples": len(outputs),
            "evaluated": 0,
            "skipped": len(outputs),
            "speaker_similarity_mean": 0.0,
            "speaker_similarity_median": 0.0,
            "speaker_similarity_std": 0.0,
            "speaker_similarity_p05": 0.0,
            "speaker_similarity_min": 0.0,
            "speaker_similarity_max": 0.0,
            "scorer_latency_mean_s": 0.0,
            "audio_duration_mean_s": 0.0,
        }

    scores = np.array([output.similarity for output in successes], dtype=np.float64)
    latencies = [
        output.scorer_latency_s for output in successes if output.scorer_latency_s > 0
    ]
    audio_durations = [
        output.audio_duration_s for output in successes if output.audio_duration_s > 0
    ]
    return {
        "total_samples": len(outputs),
        "evaluated": len(successes),
        "skipped": len(outputs) - len(successes),
        "speaker_similarity_mean": float(np.mean(scores)),
        "speaker_similarity_median": float(np.median(scores)),
        "speaker_similarity_std": float(np.std(scores)),
        "speaker_similarity_p05": float(np.percentile(scores, 5)),
        "speaker_similarity_min": float(np.min(scores)),
        "speaker_similarity_max": float(np.max(scores)),
        "scorer_latency_mean_s": (float(np.mean(latencies)) if latencies else 0.0),
        "audio_duration_mean_s": (
            float(np.mean(audio_durations)) if audio_durations else 0.0
        ),
    }


def print_speaker_similarity_summary(
    metrics: dict,
    model_name: str,
    generation_mode: str | None = None,
) -> None:
    lw = SPEED_LABEL_WIDTH
    w = SPEED_LINE_WIDTH
    title = "SeedTTS Speaker Similarity Result"
    if generation_mode:
        title = f"{title} ({generation_mode})"
    print(f"\n{'=' * w}")
    print(f"{title:^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Model:':<{lw}} {model_name}")
    if generation_mode:
        print(f"  {'Generation mode:':<{lw}} {generation_mode}")
    print(
        f"  {'Evaluated / Total:':<{lw}} "
        f"{metrics.get('evaluated', 0)}/{metrics.get('total_samples', 0)}"
    )
    print(f"  {'Skipped:':<{lw}} {metrics.get('skipped', 0)}")
    print(f"{'-' * w}")
    print(
        f"  {'Speaker sim mean (x100):':<{lw}} "
        f"{metrics.get('speaker_similarity_mean', 0):.4f}"
    )
    print(
        f"  {'Speaker sim median:':<{lw}} "
        f"{metrics.get('speaker_similarity_median', 0):.4f}"
    )
    print(
        f"  {'Speaker sim std:':<{lw}} {metrics.get('speaker_similarity_std', 0):.4f}"
    )
    print(
        f"  {'Speaker sim p05:':<{lw}} {metrics.get('speaker_similarity_p05', 0):.4f}"
    )
    print(
        f"  {'Speaker sim min/max:':<{lw}} "
        f"{metrics.get('speaker_similarity_min', 0):.4f} / "
        f"{metrics.get('speaker_similarity_max', 0):.4f}"
    )
    print(f"{'-' * w}")
    print(
        f"  {'Scorer latency mean (s):':<{lw}} "
        f"{metrics.get('scorer_latency_mean_s', 0):.4f}"
    )
    print(
        f"  {'Audio duration mean (s):':<{lw}} "
        f"{metrics.get('audio_duration_mean_s', 0):.4f}"
    )
    print(f"{'=' * w}")
