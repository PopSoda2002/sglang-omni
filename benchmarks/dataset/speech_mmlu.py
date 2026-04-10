# SPDX-License-Identifier: Apache-2.0
"""Dataset loader for XiaomiMiMo/SpeechMMLU.

Downloads the SpeechMMLU dataset (parquet metadata + audio.tar.gz) from
HuggingFace, extracts the audio archive, and yields samples with absolute
audio file paths suitable for the sglang-omni API.

Usage::

    from benchmarks.dataset.speech_mmlu import load_speech_mmlu_samples

    samples = load_speech_mmlu_samples(max_samples=100)
    samples = load_speech_mmlu_samples(subjects=["anatomy", "virology"])
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_REPO = "XiaomiMiMo/SpeechMMLU"
ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
ALL_SUBJECTS = [
    "anatomy", "clinical_knowledge", "college_biology", "college_medicine",
    "computer_security", "econometrics", "global_facts", "high_school_biology",
    "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_microeconomics",
    "high_school_psychology", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence",
    "management", "marketing", "miscellaneous", "moral_disputes", "nutrition",
    "philosophy", "prehistory", "professional_law", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]


@dataclass
class SpeechMmluSample:
    sample_id: str
    audio_path: str
    question_text: str
    correct_answer: int  # 0-3 mapping to A-D
    subject: str


def _resolve_audio_path(data_path: Path, audio_ref: str) -> Path:
    """Resolve dataset audio paths across archive layout variants.

    The parquet metadata stores paths like ``audio/<subject>/data/<file>.mp3``.
    After extracting ``audio.tar.gz`` into ``<repo>/audio``, files end up under
    ``<repo>/audio/audio/<subject>/data/<file>.mp3``. We accept either form so
    cached metadata remains usable after loader fixes.
    """
    raw_path = Path(audio_ref)
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
        try:
            rel_path = raw_path.relative_to(data_path)
        except ValueError:
            rel_path = None
        if rel_path is not None:
            candidates.append(data_path / "audio" / rel_path)
    else:
        candidates.append(data_path / raw_path)
        candidates.append(data_path / "audio" / raw_path)

    seen: set[Path] = set()
    fallback: Path | None = None
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if fallback is None:
            fallback = resolved
        if resolved.exists():
            return resolved

    return fallback or raw_path.resolve()


def _ensure_dataset_downloaded(cache_dir: str) -> Path:
    """Download repo + extract audio.tar.gz if not already done.

    Returns the path to the dataset root directory.
    """
    from huggingface_hub import snapshot_download

    data_path = Path(cache_dir) / "repo"
    audio_dir = data_path / "audio"
    sentinel = audio_dir / ".extracted"

    if sentinel.exists():
        logger.info("SpeechMMLU already downloaded and extracted at %s", data_path)
        return data_path

    logger.info("Downloading SpeechMMLU repo (~32 GB) from %s ...", DATASET_REPO)
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=str(data_path),
    )

    audio_tar = data_path / "audio.tar.gz"
    if not audio_tar.exists():
        raise FileNotFoundError(
            f"audio.tar.gz not found at {audio_tar} after snapshot_download"
        )

    audio_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s ...", audio_tar)
    subprocess.run(
        ["tar", "-xf", str(audio_tar.resolve())],
        cwd=str(audio_dir),
        check=True,
    )
    sentinel.touch()
    logger.info("Extraction complete.")
    return data_path


def _load_metadata(
    data_path: Path, subjects: list[str] | None = None
) -> list[dict]:
    """Load sample metadata from local parquet files via the datasets library."""
    from datasets import load_dataset

    target_subjects = subjects or ALL_SUBJECTS
    metadata = []

    for subject in target_subjects:
        if subject not in ALL_SUBJECTS:
            logger.warning("Unknown subject %s, skipping", subject)
            continue
        ds = load_dataset(str(data_path), subject, split="train")
        for row in ds:
            audio_path = _resolve_audio_path(data_path, row["question_audio"])
            metadata.append(
                {
                    "sample_id": row["id"],
                    "audio_path": str(audio_path),
                    "question_text": row["question_text"],
                    "correct_answer": int(row["answer"]),
                    "subject": row["subject"],
                }
            )

    return metadata


def _repair_cached_metadata(metadata: list[dict], data_path: Path) -> tuple[bool, bool]:
    """Normalize cached audio paths and report whether any remain missing."""
    changed = False
    has_missing = False

    for item in metadata:
        resolved = str(_resolve_audio_path(data_path, item["audio_path"]))
        if item["audio_path"] != resolved:
            item["audio_path"] = resolved
            changed = True
        if not Path(item["audio_path"]).exists():
            has_missing = True

    return changed, has_missing


def load_speech_mmlu_samples(
    cache_dir: str = "benchmarks/cache/speech_mmlu",
    max_samples: int | None = None,
    subjects: list[str] | None = None,
    seed: int | None = None,
) -> list[SpeechMmluSample]:
    """Load SpeechMMLU samples, downloading + extracting on first call.

    Args:
        cache_dir: Directory for cached dataset files.
        max_samples: Maximum number of samples to return.
        subjects: Optional list of subjects to filter by.
        seed: Random seed for reproducible subsampling.

    Returns:
        List of SpeechMmluSample.
    """
    data_path = _ensure_dataset_downloaded(cache_dir)

    # Cache the metadata index for fast subsequent loads
    meta_cache = Path(cache_dir) / "metadata.json"
    if meta_cache.exists():
        with open(meta_cache) as f:
            metadata = json.load(f)
        changed, has_missing = _repair_cached_metadata(metadata, data_path)
        if has_missing:
            logger.warning(
                "Cached SpeechMMLU metadata has missing audio paths; rebuilding from local parquet files."
            )
            metadata = _load_metadata(data_path, subjects=None)
            changed = True
        if changed:
            with open(meta_cache, "w") as f:
                json.dump(metadata, f)
        if subjects:
            subject_set = set(subjects)
            metadata = [m for m in metadata if m["subject"] in subject_set]
    else:
        metadata = _load_metadata(data_path, subjects=None)
        with open(meta_cache, "w") as f:
            json.dump(metadata, f)
        if subjects:
            subject_set = set(subjects)
            metadata = [m for m in metadata if m["subject"] in subject_set]

    if seed is not None:
        random.seed(seed)
        random.shuffle(metadata)
    if max_samples is not None and len(metadata) > max_samples:
        metadata = metadata[:max_samples]

    samples = [SpeechMmluSample(**m) for m in metadata]
    logger.info(
        "Loaded %d SpeechMMLU samples (%d subjects)",
        len(samples),
        len({s.subject for s in samples}),
    )
    return samples
