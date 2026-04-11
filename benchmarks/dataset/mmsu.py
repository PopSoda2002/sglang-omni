# SPDX-License-Identifier: Apache-2.0
"""Dataset loader for MMSU."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MmsuSample:
    sample_id: str
    audio_path: str
    question: str
    choices: list[str]
    answer_text: str
    answer_index: int | None
    task_name: str
    category: str
    sub_category: str
    sub_sub_category: str
    linguistics_sub_discipline: str


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text.lower())).strip()


def _guess_audio_suffix(audio_bytes: bytes) -> str:
    if audio_bytes.startswith(b"RIFF") and audio_bytes[8:12] == b"WAVE":
        return ".wav"
    if audio_bytes.startswith(b"fLaC"):
        return ".flac"
    if audio_bytes.startswith(b"OggS"):
        return ".ogg"
    if audio_bytes.startswith(b"ID3") or audio_bytes[:2] in {
        b"\xff\xfb",
        b"\xff\xf3",
        b"\xff\xf2",
    }:
        return ".mp3"
    return ".wav"


def _materialize_audio_path(
    audio_value: Any,
    repo_dir: Path,
    sample_id: str,
) -> str:
    if isinstance(audio_value, str):
        audio_path = Path(audio_value)
        return str(
            audio_path.resolve()
            if audio_path.is_absolute()
            else (repo_dir / audio_path).resolve()
        )

    if isinstance(audio_value, dict):
        path_value = audio_value.get("path")
        if isinstance(path_value, str) and path_value:
            audio_path = Path(path_value)
            return str(
                audio_path.resolve()
                if audio_path.is_absolute()
                else (repo_dir / audio_path).resolve()
            )

        audio_bytes = audio_value.get("bytes")
        if isinstance(audio_bytes, (bytes, bytearray)):
            cache_dir = repo_dir / ".mmsu_audio"
            cache_dir.mkdir(parents=True, exist_ok=True)
            suffix = (
                Path(path_value).suffix
                if isinstance(path_value, str) and Path(path_value).suffix
                else _guess_audio_suffix(bytes(audio_bytes))
            )
            audio_path = cache_dir / f"{sample_id}{suffix}"
            if not audio_path.exists():
                with open(audio_path, "wb") as file_obj:
                    file_obj.write(audio_bytes)
            return str(audio_path.resolve())

    return str(audio_value)


def _answer_index(choices: list[str], answer_text: str) -> int | None:
    normalized_answer = _normalize_text(answer_text)
    for index, choice in enumerate(choices):
        if _normalize_text(choice) == normalized_answer:
            return index
    return None


def load_mmsu_samples(
    repo_dir: str,
    max_samples: int | None = None,
    task_names: list[str] | None = None,
    categories: list[str] | None = None,
    seed: int | None = None,
) -> list[MmsuSample]:
    from datasets import Audio, load_dataset

    repo_path = Path(repo_dir).expanduser().resolve()
    parquet_files = sorted(str(path) for path in repo_path.rglob("*.parquet"))
    dataset = load_dataset("parquet", data_files=parquet_files, split="train")
    dataset = dataset.cast_column("audio", Audio(decode=False))

    samples: list[MmsuSample] = []
    task_name_filter = {name.strip() for name in task_names or [] if name.strip()}
    category_filter = {name.strip() for name in categories or [] if name.strip()}

    for row in dataset:
        task_name = row["task_name"]
        category = row["category"]
        if task_name_filter and task_name not in task_name_filter:
            continue
        if category_filter and category not in category_filter:
            continue

        choices = [
            str(row["choice_a"]).strip(),
            str(row["choice_b"]).strip(),
            str(row["choice_c"]).strip(),
            str(row["choice_d"]).strip(),
        ]
        answer_text = str(row["answer_gt"]).strip()
        sample_id = str(row["id"])
        samples.append(
            MmsuSample(
                sample_id=sample_id,
                audio_path=_materialize_audio_path(row["audio"], repo_path, sample_id),
                question=str(row["question"]).strip(),
                choices=choices,
                answer_text=answer_text,
                answer_index=_answer_index(choices, answer_text),
                task_name=task_name,
                category=category,
                sub_category=str(row["sub-category"]).strip(),
                sub_sub_category=str(row["sub-sub-category"]).strip(),
                linguistics_sub_discipline=str(
                    row["linguistics_sub_discipline"]
                ).strip(),
            )
        )

    if seed is not None and len(samples) > 1:
        samples = samples.copy()
        random.Random(seed).shuffle(samples)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples
