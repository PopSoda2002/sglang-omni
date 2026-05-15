# SPDX-License-Identifier: Apache-2.0
"""Configuration for HiggsMultimodalQwen3 model (TTS path).

The config schema also carries ASR (Whisper encoder) fields; the TTS path
uses the discrete audio encoder branch and ignores the Whisper side.
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
        # Eagerly realise text_config as a concrete PretrainedConfig so consumers
        # accessing ``self.text_config.num_attention_heads`` directly (e.g.
        # sglang's hf_transformers_utils) work without going through
        # ``get_text_config()``.
        if text_config is None:
            text_config_obj: Any = {}
        elif isinstance(text_config, dict):
            model_type = text_config.get("model_type", "qwen3")
            # CONFIG_MAPPING.get silently returns None for registered keys; use [].
            try:
                cfg_cls = transformers.CONFIG_MAPPING[model_type]
            except KeyError:
                cfg_cls = None
            # Higgs ckpts ship the Qwen3 sub-config with rope_theta=null; the
            # transformers default 10000 is wrong for Qwen3 (trained at 1e6).
            # Patch before instantiation so sglang's rope picks up the right base.
            if model_type == "qwen3" and text_config.get("rope_theta") is None:
                text_config = dict(text_config)
                text_config["rope_theta"] = 1_000_000
            text_config_obj = (
                cfg_cls(**text_config) if cfg_cls is not None else text_config
            )
        else:
            text_config_obj = text_config
        self.text_config = text_config_obj

        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False) -> transformers.PretrainedConfig:
        text_config = self.text_config
        if isinstance(text_config, dict):
            model_type = text_config.get("model_type", "qwen3")
            if model_type == "qwen3" and text_config.get("rope_theta") is None:
                text_config = dict(text_config)
                text_config["rope_theta"] = 1_000_000
            return transformers.CONFIG_MAPPING[model_type](**text_config)
        return text_config
