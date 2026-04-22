# SPDX-License-Identifier: Apache-2.0
"""Thin facade over the vendored Higgs Audio V2 tokenizer.

Gives the preprocessing stage a stable ``encode_reference`` API (raw
waveform → ``[T, num_codebooks]`` codes) and the vocoder stage a
matching ``decode`` (codes → mono waveform). Handles the upstream
shape conventions, mono-channel requirement, 24 kHz resample, and the
"pad to at least one second" quirk that the underlying model expects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from sglang_omni.models.higgs_tts._vendored.higgs_audio_v2_tokenizer_hf import (
    HiggsAudioV2TokenizerModel,
)

WaveformInput = Union[torch.Tensor, np.ndarray]


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
        """Load the tokenizer checkpoint onto ``device``.

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
