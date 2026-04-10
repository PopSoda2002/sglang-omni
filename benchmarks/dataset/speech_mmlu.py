# SPDX-License-Identifier: Apache-2.0
"""Dataset loader for XiaomiMiMo/SpeechMMLU.

Loads SpeechMMLU from a local Hugging Face snapshot and yields absolute audio
file paths for the benchmark runner.

Usage::

    from benchmarks.dataset.speech_mmlu import load_speech_mmlu_samples

    samples = load_speech_mmlu_samples(max_samples=100)
    samples = load_speech_mmlu_samples(subjects=["anatomy", "virology"])
"""

from __future__ import annotations

import logging
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_REPO = "XiaomiMiMo/SpeechMMLU"
DEFAULT_CACHE_DIR = "/root/.cache/huggingface"
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
def load_speech_mmlu_samples(
    cache_dir: str = DEFAULT_CACHE_DIR,
    max_samples: int | None = None,
    subjects: list[str] | None = None,
    seed: int | None = None,
) -> list[SpeechMmluSample]:
    """Load SpeechMMLU samples from the Hugging Face datasets cache.

    Args:
        cache_dir: Hugging Face cache directory.
        max_samples: Maximum number of samples to return.
        subjects: Optional list of subjects to filter by.
        seed: Random seed for reproducible subsampling.

    Returns:
        List of SpeechMmluSample.
    """
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    repo_dir = Path(
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    )
    audio_dir = repo_dir / "audio"
    if not audio_dir.exists():
        subprocess.run(
            ["tar", "-xf", str(repo_dir / "audio.tar.gz")],
            cwd=str(repo_dir),
            check=True,
        )

    samples: list[SpeechMmluSample] = []
    target_subjects = subjects or ALL_SUBJECTS

    for subject in target_subjects:
        logger.info("Loading SpeechMMLU subject %s", subject)
        ds = load_dataset(
            str(repo_dir),
            subject,
            split="train",
        )
        for row in ds:
            samples.append(
                SpeechMmluSample(
                    sample_id=row["id"],
                    audio_path=str((repo_dir / row["question_audio"]).resolve()),
                    question_text=row["question_text"],
                    correct_answer=int(row["answer"]),
                    subject=row["subject"],
                )
            )

    if seed is not None and len(samples) > 1:
        samples = samples.copy()
        random.Random(seed).shuffle(samples)
    if max_samples is not None:
        samples = samples[:max_samples]
    logger.info(
        "Loaded %d SpeechMMLU samples (%d subjects)",
        len(samples),
        len({s.subject for s in samples}),
    )
    return samples
