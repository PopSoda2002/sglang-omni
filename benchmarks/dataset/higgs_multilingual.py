# SPDX-License-Identifier: Apache-2.0
"""Multilingual TTS dataset loader for k2-fsa/TTS_eval_datasets.

The relevant subset is ``fleurs_multilingual_102`` (102 languages, introduced
in `OmniVoice <https://arxiv.org/abs/2604.00688>`__). Layout after running::

    python -m benchmarks.dataset.prepare --dataset fleurs-multilingual

is::

    <root>/
        fleurs_multilingual_102.jsonl
        fleurs_multilingual_102/
            download/tts_eval_datasets/fleurs_multilingual_102/prompt/*.wav

Each manifest entry::

    {
      "id":            "af_fleurs_af_za_<digits>_<digits>",
      "text":          "<sentence to synthesize>",
      "ref_audio":     "download/.../prompt/<file>.wav",  # relative to <root>
      "ref_text":      "<prompt transcript>",
      "language_id":   "af",
      "language_name": "afrikaans"
    }
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Iterator

from benchmarks.dataset.seedtts import SampleInput

DEFAULT_MANIFEST = "fleurs_multilingual_102.jsonl"
DEFAULT_AUDIO_SUBDIR = "fleurs_multilingual_102"


def _iter_manifest(manifest_path: str) -> Iterator[dict]:
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_higgs_multilingual_samples(
    dataset_root: str,
    lang: str,
    max_samples: int | None = None,
    *,
    manifest: str = DEFAULT_MANIFEST,
    audio_subdir: str = DEFAULT_AUDIO_SUBDIR,
) -> list[SampleInput]:
    """Load samples for ``lang`` (ISO-639-1 code) from the k2-fsa dataset.

    ``ref_audio`` paths in the manifest are stored relative to the dataset
    root. After ``benchmarks.dataset.prepare`` extracts the audio tarball
    under ``<root>/<audio_subdir>/``, those relative paths resolve correctly.
    """
    manifest_path = os.path.join(dataset_root, manifest)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run "
            f"`python -m benchmarks.dataset.prepare --dataset fleurs-multilingual` first."
        )

    audio_root = os.path.join(dataset_root, audio_subdir)
    samples: list[SampleInput] = []
    for entry in _iter_manifest(manifest_path):
        if entry.get("language_id") != lang:
            continue
        ref_audio_rel = entry["ref_audio"]
        ref_audio_abs = os.path.join(audio_root, ref_audio_rel)
        samples.append(
            SampleInput(
                sample_id=entry["id"],
                ref_text=entry["ref_text"],
                ref_audio=ref_audio_abs,
                target_text=entry["text"],
            )
        )
        if max_samples and len(samples) >= max_samples:
            break

    if not samples:
        available = sorted(list_higgs_multilingual_langs(dataset_root))
        raise ValueError(
            f"No samples found for lang={lang!r}. Available languages: {available}"
        )
    return samples


def list_higgs_multilingual_langs(
    dataset_root: str,
    *,
    manifest: str = DEFAULT_MANIFEST,
) -> list[str]:
    """Enumerate language codes present in the manifest with sample counts."""
    manifest_path = os.path.join(dataset_root, manifest)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    counts: Counter[str] = Counter()
    for entry in _iter_manifest(manifest_path):
        counts[entry.get("language_id", "")] += 1
    counts.pop("", None)
    return sorted(counts.keys())
