# SPDX-License-Identifier: Apache-2.0
"""SeedTTS speaker-similarity asset bootstrapper.

Single source of truth for the two HuggingFace files that
:class:`benchmarks.metrics.speaker_similarity.WavLMSpeakerSimilarity` needs:

- ``wavlm_large_finetune.pth`` — fine-tuned WavLM SV head
  (``popsoda2002/seedtts-wavlm-sim``)
- ``wavlm_large.pt`` — WavLM base weights
  (``s3prl/converted_ckpts``)

The s3prl Python package itself is consumed as a regular PyPI dependency
(``s3prl>=0.4.18`` in ``pyproject.toml``).  Earlier versions of this PR
also git-cloned the s3prl repository to read ``s3prl.upstream.wavlm.hubconf``
off-tree; the pip-installed package ships the identical module, so the
clone has been removed (per @zhaochenyang20's PR #469 review).

The bootstrapper is consumed by exactly one runtime call site
(:func:`benchmarks.tasks.tts.run_seedtts_similarity`) so the CI workflows
and the user-facing ``--similarity-only`` entry points share one code path
for asset preparation.

Usage from CI / scripts:

    python -m benchmarks.metrics.speaker_similarity_assets --warm-cache

Usage from Python:

    assets = ensure_speaker_similarity_assets()
    scorer = WavLMSpeakerSimilarity(
        finetune_checkpoint=assets.finetune_checkpoint,
        wavlm_base=assets.wavlm_base,
        device="cuda:0",
    )
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# HuggingFace sources — kept here so anyone changing the asset provenance
# touches one constant rather than chasing strings through CI yaml.
_FINETUNE_REPO_ID = "popsoda2002/seedtts-wavlm-sim"
_FINETUNE_FILENAME = "wavlm_large_finetune.pth"
_WAVLM_BASE_REPO_ID = "s3prl/converted_ckpts"
_WAVLM_BASE_FILENAME = "wavlm_large.pt"

# Atomic completion marker — only written after both downloads succeed,
# so a partial / interrupted fetch is retried on the next call.
_MARKER_FILENAME = ".complete"

_CACHE_DIR_ENV = "SEEDTTS_SIM_CACHE_DIR"


@dataclass(frozen=True)
class SpeakerSimilarityAssets:
    """Resolved on-disk paths for the SeedTTS speaker-similarity scorer."""

    finetune_checkpoint: Path
    wavlm_base: Path


def _resolve_cache_dir(cache_dir: Path | None) -> Path:
    """Pick the cache directory in priority order: arg → env → user cache."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    env_value = os.environ.get(_CACHE_DIR_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path("~/.cache/sglang-omni/speaker_sim").expanduser().resolve()


def _hf_download(repo_id: str, filename: str, dest_dir: Path) -> Path:
    """Download ``filename`` from ``repo_id`` into ``dest_dir``.

    Uses the ``huggingface_hub`` Python API (not the ``huggingface-cli``
    subprocess) so progress, errors, and ``HF_ENDPOINT`` mirror handling are
    all in-process.
    """
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_dir),
    )
    return Path(local_path)


def ensure_speaker_similarity_assets(
    cache_dir: Path | None = None,
    finetune_checkpoint_override: Path | None = None,
) -> SpeakerSimilarityAssets:
    """Make sure the WavLM SV scorer's two asset files are on disk.

    Resolution rules:

    - ``cache_dir`` is resolved in priority order:
      explicit arg → ``SEEDTTS_SIM_CACHE_DIR`` env → ``~/.cache/sglang-omni/speaker_sim``.
    - If ``finetune_checkpoint_override`` is provided (typically the user
      passing ``--similarity-checkpoint``), it is used as-is for the fine-tune
      head and is **not** re-downloaded.  The WavLM base file is still
      ensured under ``cache_dir``.
    - Cache-hit path is keyed by the ``.complete`` marker file, so partial
      states from interrupted downloads are re-attempted on the next call.

    Idempotent: a second call after a successful first call is a no-op
    aside from a "cache HIT" log line.
    """
    cache_dir = _resolve_cache_dir(cache_dir)
    marker = cache_dir / _MARKER_FILENAME

    wavlm_base = cache_dir / _WAVLM_BASE_FILENAME

    if finetune_checkpoint_override is not None:
        finetune_checkpoint = Path(finetune_checkpoint_override).expanduser().resolve()
        if not finetune_checkpoint.is_file():
            raise FileNotFoundError(
                f"--similarity-checkpoint override not found: {finetune_checkpoint}"
            )
    else:
        finetune_checkpoint = cache_dir / _FINETUNE_FILENAME

    # Cache hit requires the marker plus all files we plan to return.  When
    # the user supplied an override, only the base file lives under the
    # cache, so the marker reflects only base-file completeness.
    cache_complete = marker.is_file() and wavlm_base.is_file()
    if finetune_checkpoint_override is None:
        cache_complete = cache_complete and finetune_checkpoint.is_file()

    if cache_complete:
        logger.info(f"[sim-assets] cache HIT at {cache_dir}")
        return SpeakerSimilarityAssets(
            finetune_checkpoint=finetune_checkpoint,
            wavlm_base=wavlm_base,
        )

    logger.info(f"[sim-assets] cache MISS at {cache_dir} — fetching")

    if marker.exists():
        marker.unlink()

    if finetune_checkpoint_override is None and not finetune_checkpoint.is_file():
        _hf_download(_FINETUNE_REPO_ID, _FINETUNE_FILENAME, cache_dir)
    if not wavlm_base.is_file():
        _hf_download(_WAVLM_BASE_REPO_ID, _WAVLM_BASE_FILENAME, cache_dir)

    marker.touch()
    logger.info(f"[sim-assets] cached to {cache_dir}")

    return SpeakerSimilarityAssets(
        finetune_checkpoint=finetune_checkpoint,
        wavlm_base=wavlm_base,
    )


def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Pre-download SeedTTS speaker-similarity assets into the cache "
            f"directory (override via {_CACHE_DIR_ENV})."
        ),
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Resolve and download all asset files into the cache directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved cache directory and intended downloads, "
        "without downloading anything.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"Override cache directory (else uses {_CACHE_DIR_ENV} or "
        "~/.cache/sglang-omni/speaker_sim).",
    )
    args = parser.parse_args()

    cache_dir = _resolve_cache_dir(args.cache_dir)
    if args.dry_run:
        logger.info(f"[sim-assets] cache dir would be: {cache_dir}")
        logger.info(
            f"[sim-assets] would fetch {_FINETUNE_REPO_ID}/{_FINETUNE_FILENAME}"
        )
        logger.info(
            f"[sim-assets] would fetch {_WAVLM_BASE_REPO_ID}/{_WAVLM_BASE_FILENAME}"
        )
        return

    if not args.warm_cache:
        parser.error("pass --warm-cache to actually download, or --dry-run")

    assets = ensure_speaker_similarity_assets(cache_dir=args.cache_dir)
    logger.info(f"[sim-assets] finetune_checkpoint = {assets.finetune_checkpoint}")
    logger.info(f"[sim-assets] wavlm_base          = {assets.wavlm_base}")


if __name__ == "__main__":
    _main()
