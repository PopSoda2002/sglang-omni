#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Subprocess runner for Omnilingual ASR.

Invoked by ``benchmarks.tasks.omnilingual_asr.OmnilingualASR`` via a Python
interpreter from a separate env (``OMNI_ASR_PYTHON`` / ``--omni-asr-python``)
so that ``omnilingual-asr`` / ``fairseq2`` deps stay isolated from sglang-omni.

Input JSON (path passed as ``in_path``)::

    {
      "wav_paths": ["/abs/path/a.wav", ...],
      "langs":     ["eng_Latn", ...] or null,
      "model_card": "omniASR_LLM_7B",
      "batch_size": 8,
      "device": "cuda"
    }

Output JSON (path passed as ``out_path``)::

    {"transcriptions": ["...", ...]}  # same length as wav_paths

Failures abort with a non-zero exit code; the caller handles fallback.
"""

import argparse
import json
import os

# fairseq2 caches model artifacts here; override via env if needed.
os.environ.setdefault("FAIRSEQ2_CACHE_DIR", os.path.expanduser("~/.cache/fairseq2"))

# fairseq2 requires (RANK, WORLD_SIZE) and (LOCAL_RANK, LOCAL_WORLD_SIZE)
# set together or both unset; srun may leak partial values.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Subprocess runner for Omnilingual ASR.")
    parser.add_argument("in_path", help="Input JSON config (see module docstring)")
    parser.add_argument("out_path", help="Output JSON with transcriptions")
    args = parser.parse_args()

    with open(args.in_path) as f:
        cfg = json.load(f)

    wav_paths: list[str] = cfg["wav_paths"]
    langs = cfg.get("langs")  # list[str] | None
    model_card: str = cfg.get("model_card", "omniASR_LLM_7B")
    batch_size: int = cfg.get("batch_size", 8)
    device: str = cfg.get("device", "cuda")

    pipeline = ASRInferencePipeline(model_card=model_card, device=device)
    transcriptions = pipeline.transcribe(wav_paths, lang=langs, batch_size=batch_size)
    transcriptions = [(t or "").strip() for t in transcriptions]

    with open(args.out_path, "w") as f:
        json.dump({"transcriptions": transcriptions}, f)


if __name__ == "__main__":
    main()
