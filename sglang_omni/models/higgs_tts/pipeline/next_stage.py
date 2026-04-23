# SPDX-License-Identifier: Apache-2.0
"""Stage routing constants and callbacks for the Higgs TTS pipeline.

Pipeline order:

    preprocessing → audio_encoder → aggregate → tts_engine → vocoder

The ``audio_encoder`` + ``aggregate`` pair mirrors the convention from
``ming_omni`` / ``qwen3_omni``: the former computes a modality-specific
feature (here, the fused multi-codebook embedding of the reference
audio codes), the latter is a pipeline barrier that hands the merged
state off to the LM engine.
"""

from __future__ import annotations

from typing import Any

PREPROCESSING_STAGE = "preprocessing"
AUDIO_ENCODER_STAGE = "audio_encoder"
AGGREGATE_STAGE = "aggregate"
TTS_ENGINE_STAGE = "tts_engine"
VOCODER_STAGE = "vocoder"


def preprocessing_next(request_id: str, output: Any) -> str | None:
    del request_id, output
    return AUDIO_ENCODER_STAGE


def audio_encoder_next(request_id: str, output: Any) -> str | None:
    del request_id, output
    return AGGREGATE_STAGE


def aggregate_next(request_id: str, output: Any) -> str | None:
    del request_id, output
    return TTS_ENGINE_STAGE


def tts_engine_next(request_id: str, output: Any) -> str | None:
    del request_id, output
    return VOCODER_STAGE


def vocoder_next(request_id: str, output: Any) -> str | None:
    del request_id, output
    return None
