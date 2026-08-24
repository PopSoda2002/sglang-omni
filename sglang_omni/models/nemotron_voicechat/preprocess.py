from __future__ import annotations

import torch
from torch import nn
from einops import rearrange, einsum

 # NeMo defaults omitted from the exported checkpoint config.
DEFAULT_PREEMPHASIS = 0.97
DEFAULT_LOG_ZERO_GUARD = 2**-24

class LogMelFeatures(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.sample_rate = int(config["sample_rate"])
        self.n_fft = int(config["n_fft"])
        self.hop_length = round(self.sample_rate * float(config["window_stride"]))
        self.win_length = round(self.sample_rate * float(config["window_size"]))
        num_mels = int(config["features"])
        num_freqs = self.n_fft // 2 + 1

        self.register_buffer("fb_MF", torch.empty(num_mels, num_freqs))
        self.register_buffer("window_W", torch.empty(self.win_length))

    # Wavform -> spectrogram -> mel -> log mel
    def forward(self, waveform_BL: torch.Tensor) -> torch.Tensor:
        waveform_BL = waveform_BL.to(dtype=self.window_W.dtype)
        preemphasized_BL = torch.cat((waveform_BL[:, :1], waveform_BL[:, 1:] - DEFAULT_PREEMPHASIS * waveform_BL[:, :-1]), dim=1)
        spectrum_BFT = torch.stft(preemphasized_BL, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, 
            return_complex=True, pad_mode="constant", center=True, window=self.window_W)
        power_BFT = spectrum_BFT.abs().square()
        mel_BMT = einsum(self.fb_MF, power_BFT,"m f, b f t -> b m t")
        log_mel_BMT = torch.log(mel_BMT + DEFAULT_LOG_ZERO_GUARD)
        return rearrange(log_mel_BMT, "b m t -> b t m")
