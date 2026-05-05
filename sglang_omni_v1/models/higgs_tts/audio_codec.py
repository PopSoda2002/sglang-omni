# SPDX-License-Identifier: Apache-2.0
"""Thin facade over the vendored Higgs Audio V2 tokenizer.

Gives the preprocessing stage a stable ``encode_reference`` API (raw
waveform → ``[T, num_codebooks]`` codes) and the vocoder stage a
matching ``decode`` (codes → mono waveform). Handles the upstream
shape conventions, mono-channel requirement, 24 kHz resample, and the
"pad to at least one second" quirk that the underlying model expects.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from sglang_omni_v1.models.higgs_tts._vendored.higgs_audio_v2_tokenizer_hf import (
    HiggsAudioV2TokenizerConfig,
    HiggsAudioV2TokenizerModel,
)

WaveformInput = Union[torch.Tensor, np.ndarray]

# Higgs TTS ckpts ship the audio codec weights alongside the text
# backbone, under this prefix in their ``model.safetensors`` index.
# ``from_tts_ckpt`` strips the prefix and loads the resulting keys into a
# freshly-instantiated :class:`HiggsAudioV2TokenizerModel`.
_CODEC_IN_TTS_CKPT_PREFIX = "tied.embedding.modality_embeddings.0.model."

# Bundled codec config (2.6 KB, vendored from
# ``bosonai/higgs-audio-v2-tokenizer/config.json``). The TTS ckpt's own
# ``config.json`` is the TTS model's config and doesn't describe the
# codec architecture; we keep the codec schema beside the vendored
# model class so one ckpt is enough for the full pipeline.
_BUNDLED_CODEC_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "_vendored",
    "higgs_audio_v2_tokenizer_config.json",
)


class HiggsAudioCodec:
    """Encode/decode wrapper around the Higgs Audio V2 tokenizer.

    The underlying model is frozen (``.eval()`` + ``no_grad`` everywhere
    in this class). One codec instance is safe to share across
    requests / threads as long as concurrent callers respect the
    ``no_grad`` contract — the wrapper itself is stateless.
    """

    SAMPLE_RATE: int = 24_000

    def __init__(
        self, model: HiggsAudioV2TokenizerModel, *, device: torch.device
    ) -> None:
        self.model = model
        self.device = device
        # Cache for dtype since ``model.dtype`` walks parameters.
        self._dtype = next(model.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "HiggsAudioCodec":
        """Load a standalone Higgs Audio V2 tokenizer checkpoint onto
        ``device``. Expects the path to contain ``config.json`` and
        ``model.safetensors`` matching the bundled schema.

        ``dtype`` defaults to float32 — bfloat16 works for encode but
        PR experience shows the decode ConvTranspose path is less
        stable there, so the caller should opt-in explicitly.
        """
        device = torch.device(device)
        model = HiggsAudioV2TokenizerModel.from_pretrained(
            str(model_path), local_files_only=False
        )
        model = model.to(device=device, dtype=dtype).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model, device=device)

    @classmethod
    def from_tts_ckpt(
        cls,
        tts_ckpt_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        codec_config_path: str | Path | None = None,
    ) -> "HiggsAudioCodec":
        """Load the codec directly from a Higgs TTS checkpoint.

        Higgs TTS training bundles the (frozen) audio codec weights under
        the ``tied.embedding.modality_embeddings.0.model.`` prefix inside
        the TTS ``model.safetensors``. This factory strips the prefix and
        loads those 528 tensors into a freshly-instantiated codec, so the
        full pipeline needs **only one ckpt on disk**.

        Args:
            tts_ckpt_path: Directory containing the TTS ``model.safetensors``
                (+ its index shard, if sharded).
            device / dtype: Same semantics as :meth:`from_pretrained`.
            codec_config_path: Optional override for the codec
                architecture config. Defaults to the bundled
                ``higgs_audio_v2_tokenizer_config.json`` (copied verbatim
                from ``bosonai/higgs-audio-v2-tokenizer``).
        """
        from safetensors import safe_open

        device = torch.device(device)

        config_src = (
            str(codec_config_path)
            if codec_config_path is not None
            else _BUNDLED_CODEC_CONFIG_PATH
        )
        with open(config_src) as f:
            cfg_dict = json.load(f)
        # Drop auto-injected HF metadata that ``HiggsAudioV2TokenizerConfig``
        # doesn't accept as a kwarg (``architectures``, ``torch_dtype``, …).
        cfg_dict.pop("architectures", None)
        cfg_dict.pop("torch_dtype", None)
        cfg_dict.pop("transformers_version", None)
        config = HiggsAudioV2TokenizerConfig(**cfg_dict)
        model = HiggsAudioV2TokenizerModel(config).to(dtype=dtype).eval()

        # Scan the TTS ckpt index to find which shard holds each codec
        # tensor; open each shard once and pull only the codec-prefixed
        # keys. ``model.safetensors.index.json`` is the sharded-layout
        # marker; if absent, fall back to the single-file layout.
        tts_dir = str(tts_ckpt_path)
        index_path = os.path.join(tts_dir, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            shards: dict[str, list[str]] = {}
            for full_name, shard in weight_map.items():
                if full_name.startswith(_CODEC_IN_TTS_CKPT_PREFIX):
                    shards.setdefault(shard, []).append(full_name)
        else:
            shards = {"model.safetensors": None}  # type: ignore[dict-item]

        state_dict: dict[str, torch.Tensor] = {}
        for shard, names in shards.items():
            shard_path = os.path.join(tts_dir, shard)
            with safe_open(shard_path, framework="pt") as f:
                keys = (
                    names
                    if names is not None
                    else [
                        k for k in f.keys() if k.startswith(_CODEC_IN_TTS_CKPT_PREFIX)
                    ]
                )
                for full_name in keys:
                    local_name = full_name[len(_CODEC_IN_TTS_CKPT_PREFIX) :]
                    state_dict[local_name] = f.get_tensor(full_name).to(dtype=dtype)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Codec weights from TTS ckpt had unexpected keys: "
                f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
            )
        if missing:
            raise RuntimeError(
                f"Codec weights from TTS ckpt were missing keys: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        model = model.to(device=device)
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model, device=device)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _to_tensor(waveform: WaveformInput) -> torch.Tensor:
        if isinstance(waveform, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(waveform))
        if isinstance(waveform, torch.Tensor):
            return waveform
        raise TypeError(
            f"waveform must be Tensor or ndarray, got {type(waveform).__name__}"
        )

    @staticmethod
    def _to_mono_3d(wav: torch.Tensor) -> torch.Tensor:
        """Normalise a waveform tensor to ``[1, 1, L]``."""
        if wav.ndim == 1:
            return wav.view(1, 1, -1)
        if wav.ndim == 2:
            # ``[C, L]`` — keep first channel only to enforce mono.
            return wav[:1].unsqueeze(0)
        if wav.ndim == 3:
            if wav.shape[1] != 1:
                raise ValueError(
                    f"audio must be mono (channels=1), got shape {tuple(wav.shape)}"
                )
            return wav
        raise ValueError(f"waveform must be 1-, 2- or 3-D tensor, got {wav.ndim}-D")

    # -- encode -------------------------------------------------------------
    @torch.no_grad()
    def encode_reference(
        self,
        waveform: WaveformInput,
        *,
        sample_rate: int | None = None,
    ) -> torch.Tensor:
        """Encode a reference clip into discrete codes.

        Args:
            waveform: 1-D ``[L]``, 2-D ``[C, L]`` (first channel used), or
                3-D ``[1, 1, L]`` mono waveform. ``float32`` in ``[-1, 1]``
                is expected (matches torchaudio / soundfile convention).
            sample_rate: Source sample rate. If ``None`` or equal to
                :attr:`SAMPLE_RATE`, no resampling. Otherwise resampled to
                24 kHz with ``torchaudio.functional.resample``.

        Returns:
            ``int64`` tensor of shape ``[T, num_codebooks]`` on CPU.
        """
        wav = self._to_mono_3d(self._to_tensor(waveform))
        wav = wav.to(torch.float32)

        sr = sample_rate if sample_rate is not None else self.SAMPLE_RATE
        if sr != self.SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, self.SAMPLE_RATE)

        # Upstream encoder errors on clips shorter than one second; the
        # padding is zero-valued so it becomes near-silent codes at the
        # tail and is trimmed by the delay pattern in the TTS prompt.
        L = wav.shape[-1]
        if L < self.SAMPLE_RATE:
            wav = F.pad(wav, (0, self.SAMPLE_RATE - L))

        wav = wav.to(device=self.device, dtype=self._dtype)
        codes_BNT = self.model.encode(wav).audio_codes  # [1, N, T]
        codes_TN = codes_BNT.squeeze(0).transpose(0, 1).to(torch.long).cpu()
        return codes_TN

    # -- decode -------------------------------------------------------------
    @torch.no_grad()
    def decode(self, codes_TN: torch.Tensor) -> torch.Tensor:
        """Decode ``[T, num_codebooks]`` codes into a mono waveform ``[L]``.

        Output dtype matches the codec's working dtype; the vocoder stage
        is responsible for any final ``float32``/WAV encoding step.
        """
        if codes_TN.ndim != 2:
            raise ValueError(
                f"codes must be 2-D [T, num_codebooks], got shape "
                f"{tuple(codes_TN.shape)}"
            )
        codes_BNT = (
            codes_TN.transpose(0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.long)
        )
        audio = self.model.decode(codes_BNT).audio_values  # [1, 1, L]
        return audio.squeeze(0).squeeze(0).cpu()


__all__ = ["HiggsAudioCodec"]
