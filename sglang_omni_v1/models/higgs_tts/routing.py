# SPDX-License-Identifier: Apache-2.0
"""Stage routing constants and callbacks for the Higgs TTS pipeline."""

from __future__ import annotations

from typing import Any

PREPROCESSING_STAGE = "preprocessing"
AUDIO_ENCODER_STAGE = "audio_encoder"
TTS_ENGINE_STAGE = "tts_engine"
VOCODER_STAGE = "vocoder"


def preprocessing_next(request_id: str, output: Any) -> str | None:
    del request_id, output
    return AUDIO_ENCODER_STAGE


def audio_encoder_next(request_id: str, output: Any) -> str | None:
    del request_id, output
    return TTS_ENGINE_STAGE


def tts_engine_next(request_id: str, output: Any) -> str | None:
    del request_id, output
    return VOCODER_STAGE


def vocoder_next(request_id: str, output: Any) -> str | None:
    del request_id, output
    return None


__all__ = [
    "AUDIO_ENCODER_STAGE",
    "PREPROCESSING_STAGE",
    "TTS_ENGINE_STAGE",
    "VOCODER_STAGE",
    "audio_encoder_next",
    "preprocessing_next",
    "tts_engine_next",
    "vocoder_next",
]
