"""Thin facade over the vendored Higgs Audio V2 tokenizer.

Provides ``encode_reference`` (waveform → ``[T, num_codebooks]`` codes) for the
preprocessing stage and ``decode`` (codes → mono waveform) for the vocoder.
Handles mono-channel coercion, 24 kHz resample, and the underlying model's
"≥ 1 second of audio" input requirement.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from sglang_omni.models.higgs_tts._vendored.higgs_audio_v2_tokenizer_hf import (
    HiggsAudioV2TokenizerConfig,
    HiggsAudioV2TokenizerModel,
)

WaveformInput = torch.Tensor | np.ndarray

# Higgs TTS ckpts embed the codec weights under this safetensors prefix.
_CODEC_IN_TTS_CKPT_PREFIX = "tied.embedding.modality_embeddings.0.model."

_BUNDLED_CODEC_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "_vendored",
    "higgs_audio_v2_tokenizer_config.json",
)


def _to_mono_3d(waveform: WaveformInput) -> torch.Tensor:
    """Normalise any (Tensor | ndarray) waveform to a mono ``[1, 1, L]`` tensor."""
    if isinstance(waveform, np.ndarray):
        wav = torch.from_numpy(np.ascontiguousarray(waveform))
    elif isinstance(waveform, torch.Tensor):
        wav = waveform
    else:
        raise TypeError(
            f"waveform must be Tensor or ndarray, got {type(waveform).__name__}"
        )

    if wav.ndim == 1:
        return wav.view(1, 1, -1)
    if wav.ndim == 2:
        return wav[:1].unsqueeze(0)  # [C, L] → keep first channel
    if wav.ndim == 3:
        if wav.shape[1] != 1:
            raise ValueError(f"audio must be mono, got shape {tuple(wav.shape)}")
        return wav
    raise ValueError(f"waveform must be 1-, 2- or 3-D, got {wav.ndim}-D")


class HiggsAudioCodec:
    """Frozen encode/decode wrapper around :class:`HiggsAudioV2TokenizerModel`."""

    SAMPLE_RATE: int = 24_000

    def __init__(
        self, model: HiggsAudioV2TokenizerModel, *, device: torch.device
    ) -> None:
        self.model = model
        self.device = device
        self._dtype = next(model.parameters()).dtype  # avoid model.dtype param walk

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "HiggsAudioCodec":
        """Load a standalone Higgs Audio V2 tokenizer checkpoint.

        ``dtype`` defaults to fp32. bf16 works for encode but the decode
        ConvTranspose path is less stable in bf16 — opt in explicitly.
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
        """Load the codec embedded inside a Higgs TTS checkpoint.

        Strips the ``tied.embedding.modality_embeddings.0.model.`` prefix
        from the TTS ``model.safetensors`` and reconstructs the codec —
        so the full pipeline needs only one ckpt on disk.
        """
        from safetensors import safe_open

        device = torch.device(device)
        config_src = str(codec_config_path or _BUNDLED_CODEC_CONFIG_PATH)
        with open(config_src) as f:
            cfg_dict = json.load(f)
        # HF auto-injected metadata isn't valid kwargs for the config class.
        for key in ("architectures", "torch_dtype", "transformers_version"):
            cfg_dict.pop(key, None)
        config = HiggsAudioV2TokenizerConfig(**cfg_dict)
        model = HiggsAudioV2TokenizerModel(config).to(dtype=dtype).eval()

        # Build a shard → keys mapping so each shard is opened once.
        tts_dir = str(tts_ckpt_path)
        index_path = os.path.join(tts_dir, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            shards: dict[str, list[str] | None] = {}
            for full_name, shard in weight_map.items():
                if full_name.startswith(_CODEC_IN_TTS_CKPT_PREFIX):
                    shards.setdefault(shard, []).append(full_name)  # type: ignore[union-attr]
        else:
            shards = {"model.safetensors": None}

        state_dict: dict[str, torch.Tensor] = {}
        for shard, names in shards.items():
            with safe_open(os.path.join(tts_dir, shard), framework="pt") as f:
                keys = names or [
                    k for k in f.keys() if k.startswith(_CODEC_IN_TTS_CKPT_PREFIX)
                ]
                for full_name in keys:
                    local_name = full_name[len(_CODEC_IN_TTS_CKPT_PREFIX) :]
                    state_dict[local_name] = f.get_tensor(full_name).to(dtype=dtype)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Codec weights from TTS ckpt mismatch: "
                f"missing={missing[:3]}, unexpected={unexpected[:3]}"
            )

        model = model.to(device=device)
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model, device=device)

    @torch.no_grad()
    def encode_reference(
        self,
        waveform: WaveformInput,
        *,
        sample_rate: int | None = None,
    ) -> torch.Tensor:
        """Encode a reference clip to ``[T, num_codebooks]`` int64 codes (CPU).

        Accepts 1-D ``[L]``, 2-D ``[C, L]`` (first channel), or 3-D ``[1, 1, L]``
        float32 waveform in ``[-1, 1]``. Resamples to 24 kHz if ``sample_rate``
        differs. Clips < 1 s are zero-padded (the encoder errors otherwise;
        the trailing silence is trimmed by the delay pattern downstream).
        """
        wav = _to_mono_3d(waveform).to(torch.float32)
        sr = sample_rate or self.SAMPLE_RATE
        if sr != self.SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, self.SAMPLE_RATE)
        if wav.shape[-1] < self.SAMPLE_RATE:
            wav = F.pad(wav, (0, self.SAMPLE_RATE - wav.shape[-1]))

        wav = wav.to(device=self.device, dtype=self._dtype)
        codes_BNT = self.model.encode(wav).audio_codes  # [1, N, T]
        return codes_BNT.squeeze(0).transpose(0, 1).to(torch.long).cpu()

    @torch.no_grad()
    def decode(self, codes_TN: torch.Tensor) -> torch.Tensor:
        """Decode ``[T, num_codebooks]`` codes to a mono waveform ``[L]``."""
        if codes_TN.ndim != 2:
            raise ValueError(
                f"codes must be 2-D [T, num_codebooks], got {tuple(codes_TN.shape)}"
            )
        codes_BNT = (
            codes_TN.transpose(0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.long)
        )
        return self.model.decode(codes_BNT).audio_values.squeeze(0).squeeze(0).cpu()


__all__ = ["HiggsAudioCodec"]
