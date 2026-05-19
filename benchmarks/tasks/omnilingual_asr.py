# SPDX-License-Identifier: Apache-2.0
"""Omnilingual ASR scorer for multilingual TTS WER.

Wraps Meta's ``omnilingual-asr`` via subprocess into an isolated Python
environment so its ``fairseq2`` / older-torch deps don't conflict with
sglang-omni's torch 2.9 / transformers <5 pin.

The interpreter path resolves in this order:
    1. The ``python_exe`` constructor argument.
    2. The ``OMNI_ASR_PYTHON`` environment variable.

The target env must have ``omnilingual-asr`` and ``fairseq2`` installed.
See ``benchmarks/eval/benchmark_tts_higgs_multilingual.py`` for setup notes.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Sequence

import torch

logger = logging.getLogger(__name__)

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LANG_CODES_PATH = os.path.join(
    _PKG_ROOT, "resources", "omni", "omni_asr_lang_codes.json"
)
_RUNNER_SCRIPT = os.path.abspath(
    os.path.join(_PKG_ROOT, "..", "scripts", "omni_asr_runner.py")
)


def _load_omni_lang_map() -> dict[str, str]:
    with open(_LANG_CODES_PATH) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


OMNI_ASR_LANG_TO_BCP47: dict[str, str] = _load_omni_lang_map()

# Aliases for ISO-639-3 codes that appear in k2-fsa's manifest but are not
# direct keys in omni_asr_lang_codes.json (which keys most languages by their
# ISO-639-1 code). Without these, the ASR would run unconditioned for ~4
# languages and accuracy would suffer.
_HF_LANG_ALIASES: dict[str, str] = {
    "arb": "arb_Arab",   # MSA (JSON has "ar" -> arb_Arab)
    "fil": "tgl_Latn",   # Filipino ~ Tagalog (JSON has "tl" -> tgl_Latn)
    "npi": "npi_Deva",   # Nepali individual code (JSON has "ne" -> npi_Deva)
    "ory": "ory_Orya",   # Odia individual code (JSON has "or" -> ory_Orya)
}


class OmnilingualASRError(RuntimeError):
    """Raised when the Omnilingual ASR subprocess fails or is misconfigured."""


class OmnilingualASR:
    """Subprocess wrapper around ``omnilingual_asr.ASRInferencePipeline``."""

    DEFAULT_MODEL_CARD = "omniASR_LLM_7B"

    def __init__(
        self,
        python_exe: str | None = None,
        model_card: str = DEFAULT_MODEL_CARD,
        batch_size: int = 8,
        runner_script: str | None = None,
    ) -> None:
        self.python_exe = python_exe or os.environ.get("OMNI_ASR_PYTHON")
        if not self.python_exe:
            raise OmnilingualASRError(
                "Omnilingual ASR python interpreter not set. Pass "
                "--omni-asr-python or set OMNI_ASR_PYTHON to the path of a "
                "Python env that has `omnilingual-asr` installed."
            )
        if not os.path.isfile(self.python_exe):
            raise OmnilingualASRError(
                f"OMNI_ASR_PYTHON does not exist: {self.python_exe}"
            )
        self.model_card = model_card
        self.batch_size = batch_size
        self.runner_script = runner_script or _RUNNER_SCRIPT
        if not os.path.isfile(self.runner_script):
            raise OmnilingualASRError(
                f"Omnilingual ASR runner script missing: {self.runner_script}"
            )

    @staticmethod
    def to_bcp47(lang: str) -> str | None:
        """Map an ISO-639-1/3 code to Omnilingual's BCP-47 code, or None."""
        if "_" in lang:
            return lang
        bcp47 = OMNI_ASR_LANG_TO_BCP47.get(lang)
        if bcp47 is None:
            bcp47 = _HF_LANG_ALIASES.get(lang)
        return bcp47

    def transcribe(
        self,
        wav_paths: Sequence[str],
        lang: str,
        device: str = "cuda",
    ) -> list[str]:
        """Transcribe ``wav_paths``. Returns one string per input path.

        Returns ``[""] * len(wav_paths)`` if the subprocess fails so the caller
        can keep the benchmark progressing rather than aborting mid-run.
        """
        if not wav_paths:
            return []

        bcp47 = self.to_bcp47(lang)
        if bcp47 is None:
            logger.warning(
                "Omnilingual ASR: no BCP-47 mapping for %r; running unconditioned",
                lang,
            )

        langs: list[str] | None = (
            [bcp47] * len(wav_paths) if bcp47 else None
        )

        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "in.json")
            out_path = os.path.join(tmp, "out.json")
            with open(in_path, "w") as f:
                json.dump(
                    {
                        "wav_paths": list(wav_paths),
                        "langs": langs,
                        "model_card": self.model_card,
                        "batch_size": self.batch_size,
                        "device": device,
                    },
                    f,
                )

            env = os.environ.copy()
            local_rank = os.environ.get("LOCAL_RANK")
            if local_rank is not None and torch.cuda.is_available():
                visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                if visible:
                    gpus = visible.split(",")
                    idx = int(local_rank) % len(gpus)
                    env["CUDA_VISIBLE_DEVICES"] = gpus[idx]
                else:
                    idx = int(local_rank) % torch.cuda.device_count()
                    env["CUDA_VISIBLE_DEVICES"] = str(idx)

            try:
                subprocess.run(
                    [self.python_exe, self.runner_script, in_path, out_path],
                    check=True,
                    env=env,
                )
            except subprocess.CalledProcessError as exc:
                logger.error("Omnilingual ASR runner failed: %s", exc)
                return [""] * len(wav_paths)

            with open(out_path) as f:
                return json.load(f)["transcriptions"]


def get_omnilingual_asr(
    python_exe: str | None = None,
    model_card: str = OmnilingualASR.DEFAULT_MODEL_CARD,
    batch_size: int = 8,
) -> OmnilingualASR:
    """Tiny factory; kept for symmetry with the seedtts ASR loader pattern."""
    return OmnilingualASR(
        python_exe=python_exe, model_card=model_card, batch_size=batch_size
    )
