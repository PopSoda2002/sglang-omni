# SPDX-License-Identifier: Apache-2.0
"""Stage factory stubs for the Higgs TTS pipeline.

PR1 scaffolding. The stage factories are registered so pipeline compilation
succeeds, but each one raises ``NotImplementedError`` when invoked. Follow-up
PRs will fill them in:

- ``create_preprocessing_executor``: PR3 (text + reference audio → prompt tokens).
- ``create_sglang_tts_engine_executor``: PR4 (sglang engine + multi-codebook
  sampling with delay pattern / EOC wind-down).
- ``create_vocoder_executor``: PR5 (higgs-audio-v2-tokenizer decode → WAV).
"""

from __future__ import annotations

from sglang_omni.executors import EngineExecutor, PreprocessingExecutor


def create_preprocessing_executor(model_path: str) -> PreprocessingExecutor:
    """TODO(PR3): build prompt tokens from text + reference audio."""
    del model_path
    raise NotImplementedError(
        "Higgs TTS preprocessing stage is not implemented yet (planned for PR3)."
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    max_new_tokens: int = 2048,
) -> EngineExecutor:
    """TODO(PR4): wrap sglang engine + multi-codebook sampler."""
    del model_path, device, max_new_tokens
    raise NotImplementedError(
        "Higgs TTS sglang engine stage is not implemented yet (planned for PR4)."
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cpu",
) -> PreprocessingExecutor:
    """TODO(PR5): decode multi-codebook tokens → waveform via higgs-audio tokenizer."""
    del model_path, device
    raise NotImplementedError(
        "Higgs TTS vocoder stage is not implemented yet (planned for PR5)."
    )
