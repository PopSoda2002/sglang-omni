# SPDX-License-Identifier: Apache-2.0
"""Configuration for HiggsMultimodalQwen3 model (TTS path).

Ported verbatim from boson-vllm so the same HuggingFace checkpoint works on
both serving stacks. ASR (Whisper encoder) fields are present in the config
schema but not yet exercised by the sglang-omni pipeline; the TTS path uses
the discrete audio encoder.
"""

from __future__ import annotations

from typing import Any

import transformers


class HiggsMultimodalQwen3Config(transformers.PretrainedConfig):
    """Configuration class for HiggsMultimodalQwen3 model.

    Uses a unified ``audio_encoder_config`` with an ``encoder_type``
    discriminator:

    - ``encoder_type="whisper"``: ASR model with Whisper encoder.
    - ``encoder_type="discrete"``: TTS model with codebook embeddings.

    Args:
        audio_encoder_config: Unified encoder config dict. Contains
            ``encoder_type`` plus type-specific fields (``whisper_config``,
            ``num_codebooks``, ``vocab_size``, etc.).
        text_config: Configuration for the Qwen3 text backbone.
        audio_token_id: Token id used for audio placeholders.
        mel_per_sample: Number of mel frames per audio sample (Whisper path).
    """

    model_type = "higgs_multimodal_qwen3"
    is_composition = True

    def __init__(
        self,
        audio_encoder_config: dict[str, Any] | None = None,
        text_config: dict[str, Any] | None = None,
        audio_token_id: int = -100,
        mel_per_sample: int = 8,
        **kwargs,
    ):
        self.audio_token_id = audio_token_id
        self.mel_per_sample = mel_per_sample

        self.audio_encoder_config: dict[str, Any] | None = audio_encoder_config
        self.text_config: dict[str, Any] = (
            text_config if text_config is not None else {}
        )

        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False) -> transformers.PretrainedConfig:
        text_config = self.text_config
        if isinstance(text_config, dict):
            model_type = text_config.get("model_type", "qwen3")
            return transformers.CONFIG_MAPPING[model_type](**text_config)
        return text_config
