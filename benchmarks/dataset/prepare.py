# SPDX-License-Identifier: Apache-2.0
"""Dataset download helpers.

Usage:
    # SeedTTS family (downloads into ./seedtts_testset by default)
    python -m benchmarks.dataset.prepare --dataset seedtts
    python -m benchmarks.dataset.prepare --dataset seedtts-mini
    python -m benchmarks.dataset.prepare --dataset seedtts-50

    # FLEURS-multilingual 102-language testset for higgs multilingual TTS WER.
    # Downloads + extracts into ./fleurs_multilingual_eval by default.
    python -m benchmarks.dataset.prepare --dataset fleurs-multilingual

    # MMMU / MMSU / Video-MME / Video-AMME (pre-warm the HuggingFace datasets cache)
    python -m benchmarks.dataset.prepare --dataset mmmu
    python -m benchmarks.dataset.prepare --dataset mmmu-ci-50
    python -m benchmarks.dataset.prepare --dataset mmsu
    python -m benchmarks.dataset.prepare --dataset videomme
    python -m benchmarks.dataset.prepare --dataset videomme-ci-50
    python -m benchmarks.dataset.prepare --dataset videomme-ci-25
    python -m benchmarks.dataset.prepare --dataset videoamme-ci-50
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import tarfile

logger = logging.getLogger(__name__)

DATASETS: dict[str, str] = {
    "seedtts": "zhaochenyang20/seed-tts-eval",
    "seedtts-mini": "zhaochenyang20/seed-tts-eval-mini",
    "seedtts-50": "zhaochenyang20/seed-tts-eval-50",
    "fleurs-multilingual": "k2-fsa/TTS_eval_datasets",
    "mmmu": "MMMU/MMMU",
    "mmmu-ci-50": "zhaochenyang20/mmmu-ci-50",
    "mmsu": "ddwang2000/MMSU",
    "mmsu-ci-2000": "zhaochenyang20/mmsu-ci-2000",
    "videomme": "zhaochenyang20/Video_MME",
    "videomme-ci-50": "zhaochenyang20/Video_MME_ci",
    "videomme-ci-25": "zhaochenyang20/Video_MME_ci_25",
    "videoamme-ci-50": "zhaochenyang20/Video_AMME_ci",
}

_CLI_LOCAL_DIRS: dict[str, str] = {
    "seedtts": "seedtts_testset",
    "seedtts-mini": "seedtts_testset",
    "seedtts-50": "seedtts_testset",
    "fleurs-multilingual": "fleurs_multilingual_eval",
}

_SEEDTTS_EXISTENCE_MARKER = "en/meta.lst"
_FLEURS_MARKER = "fleurs_multilingual_102/download/tts_eval_datasets/fleurs_multilingual_102/prompt"


def download_dataset(
    repo_id: str,
    local_dir: str | None = "seedtts_testset",
    *,
    existence_marker: str | None = _SEEDTTS_EXISTENCE_MARKER,
    quiet: bool = False,
) -> None:
    """Download a HuggingFace dataset."""
    if local_dir is not None and existence_marker:
        marker_path = os.path.join(local_dir, existence_marker)
        if os.path.exists(marker_path):
            if not quiet:
                logger.info(
                    f"Dataset already exists at {local_dir}, skipping download."
                )
            return

    if not quiet:
        where = local_dir if local_dir is not None else "HuggingFace cache"
        logger.info(f"Downloading {repo_id} to {where} ...")

    cmd = [
        "huggingface-cli",
        "download",
        repo_id,
        "--repo-type",
        "dataset",
    ]
    if local_dir is not None:
        cmd += ["--local-dir", local_dir]

    try:
        subprocess.run(cmd, check=True, capture_output=quiet, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to download dataset {repo_id}.\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    if not quiet:
        logger.info(f"Dataset {repo_id} ready.")


def _extract_tarball(tarball_path: str, dest_dir: str) -> None:
    """Extract ``tarball_path`` into ``dest_dir`` if not already extracted."""
    os.makedirs(dest_dir, exist_ok=True)
    logger.info(f"Extracting {tarball_path} -> {dest_dir} ...")
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(dest_dir)


def _prepare_fleurs_multilingual(local_dir: str) -> None:
    """Download + extract k2-fsa/TTS_eval_datasets fleurs_multilingual_102."""
    marker = os.path.join(local_dir, _FLEURS_MARKER)
    if os.path.isdir(marker):
        logger.info(f"fleurs_multilingual_102 already prepared at {local_dir}")
        return

    # The HF repo is sizeable; only pull the multilingual subset we need.
    cmd = [
        "huggingface-cli", "download",
        "k2-fsa/TTS_eval_datasets",
        "--repo-type", "dataset",
        "--local-dir", local_dir,
        "--include",
        "fleurs_multilingual_102.jsonl",
        "fleurs_multilingual_102.tar.gz",
    ]
    logger.info("Downloading k2-fsa/TTS_eval_datasets (fleurs_multilingual_102 only) ...")
    subprocess.run(cmd, check=True)

    tarball = os.path.join(local_dir, "fleurs_multilingual_102.tar.gz")
    dest = os.path.join(local_dir, "fleurs_multilingual_102")
    _extract_tarball(tarball, dest)
    logger.info(f"fleurs_multilingual_102 ready at {local_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="seedtts",
        help="Dataset to download.",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Override local directory for seedtts-family datasets. "
        "Ignored for datasets that are pulled into the HuggingFace cache.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    default_local_dir = _CLI_LOCAL_DIRS.get(args.dataset)
    local_dir = args.local_dir or default_local_dir

    if args.dataset == "fleurs-multilingual":
        _prepare_fleurs_multilingual(local_dir)
        return

    repo_id = DATASETS[args.dataset]
    existence_marker = (
        _SEEDTTS_EXISTENCE_MARKER if args.dataset in _CLI_LOCAL_DIRS else None
    )
    download_dataset(
        repo_id,
        local_dir,
        existence_marker=existence_marker,
    )


if __name__ == "__main__":
    main()
