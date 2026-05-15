# SPDX-License-Identifier: Apache-2.0
"""HiggsMultimodalQwen3 TTS model support for sglang-omni.

Registers :class:`HiggsMultimodalQwen3Config` with ``transformers.AutoConfig`` on
import so ``AutoConfig.from_pretrained()`` works before any Higgs stage factory
runs. The model class is registered lazily in
:func:`bootstrap.register_higgs_tts_in_sglang` when the tts_engine stage starts.
"""

from __future__ import annotations

from transformers import AutoConfig

from . import config
from .hf_config import HiggsMultimodalQwen3Config

AutoConfig.register("higgs_multimodal_qwen3", HiggsMultimodalQwen3Config)

__all__ = ["config", "HiggsMultimodalQwen3Config"]
