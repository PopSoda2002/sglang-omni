"""Thin facade over the vendored Higgs Audio V2 tokenizer.

Provides ``encode_reference`` (waveform → ``[T, num_codebooks]`` codes) for the
preprocessing stage and ``decode`` (codes → mono waveform) for the vocoder.
Handles mono-channel coercion, 24 kHz resample, and the underlying model's
"≥ 1 second of audio" input requirement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from sglang_omni.models.higgs_tts._vendored.higgs_audio_v2_tokenizer_hf import (
    HiggsAudioV2TokenizerModel,
)

WaveformInput = torch.Tensor | np.ndarray


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
        """Load the Higgs Audio V2 tokenizer (e.g. ``bosonai/higgs-audio-v2-tokenizer``).

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
