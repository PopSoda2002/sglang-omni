# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`HiggsAudioCodec` (PR3b).

Covers both the vendored tokenizer's happy path on a real checkpoint
and the shape/contract invariants the preprocessing / vocoder stages
depend on. The real-ckpt tests auto-skip when the checkpoint directory
isn't mounted or CUDA is unavailable.
"""

from __future__ import annotations

import math
import os

import pytest

_REAL_CODEC_CKPT = "/ceph/models/eustlb__higgs-audio-v2-tokenizer"
_REAL_TTS_CKPT = "/hot-data/checkpoints/TTS/c66596d6cde44fb4ba8076cc2dc1e77c/step_35500"

_real_codec_missing = not os.path.isdir(_REAL_CODEC_CKPT)
_real_tts_missing = not os.path.isdir(_REAL_TTS_CKPT)


@pytest.fixture(scope="module")
def codec():
    if _real_codec_missing:
        pytest.skip(f"Codec ckpt not mounted at {_REAL_CODEC_CKPT}")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the codec")

    from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec

    return HiggsAudioCodec.from_pretrained(_REAL_CODEC_CKPT, device="cuda")


def test_encode_reference_returns_TN_codes(codec):
    """1 second of 24 kHz audio encodes to ``[T, 8]`` int64 codes."""
    import torch

    wav = 0.3 * torch.sin(2 * math.pi * 440 * torch.linspace(0, 1.0, 24_000))
    codes = codec.encode_reference(wav, sample_rate=24_000)
    assert codes.ndim == 2
    assert codes.shape[1] == 8
    assert codes.shape[0] > 0
    assert codes.dtype == torch.long
    # All codes should be in the valid range [0, 1024) for a 1024-entry codebook.
    assert int(codes.min().item()) >= 0
    assert int(codes.max().item()) < 1024


def test_encode_reference_pads_short_input(codec):
    """Inputs shorter than 1 second are zero-padded to 1 second before encoding."""
    import torch

    # 0.4 second of audio — below the upstream 1-second minimum.
    wav = 0.3 * torch.sin(2 * math.pi * 440 * torch.linspace(0, 0.4, int(0.4 * 24_000)))
    codes = codec.encode_reference(wav, sample_rate=24_000)
    # 1 second of 24 kHz at the 25 Hz codec frame rate ≈ 25 frames.
    assert codes.shape[0] >= 24
    assert codes.shape[0] <= 26


def test_encode_reference_resamples_non_24k(codec):
    """A 16 kHz input is transparently resampled to 24 kHz before encoding;
    encoded length stays close to the time-domain equivalent."""
    import torch

    duration = 1.2
    wav_16k = torch.sin(
        2 * math.pi * 440 * torch.linspace(0, duration, int(duration * 16_000))
    )
    wav_24k = torch.sin(
        2 * math.pi * 440 * torch.linspace(0, duration, int(duration * 24_000))
    )
    codes_16k = codec.encode_reference(wav_16k, sample_rate=16_000)
    codes_24k = codec.encode_reference(wav_24k, sample_rate=24_000)
    # Same duration → same number of frames (±1 for rounding).
    assert abs(codes_16k.shape[0] - codes_24k.shape[0]) <= 1


def test_encode_accepts_numpy_and_2d_inputs(codec):
    """Tensor, numpy, ``[L]``, ``[C, L]``, and ``[1, 1, L]`` all work; non-mono
    multi-channel shapes are rejected."""
    import numpy as np
    import torch

    L = 24_000
    wav = 0.3 * torch.sin(2 * math.pi * 440 * torch.linspace(0, 1.0, L))

    c1 = codec.encode_reference(wav, sample_rate=24_000)
    c2 = codec.encode_reference(wav.numpy(), sample_rate=24_000)
    c3 = codec.encode_reference(wav.unsqueeze(0), sample_rate=24_000)  # [1, L]
    c4 = codec.encode_reference(wav.view(1, 1, -1), sample_rate=24_000)  # [1, 1, L]
    assert c1.shape == c2.shape == c3.shape == c4.shape

    # Quick sanity check that the numpy variant is actually numpy-backed.
    assert isinstance(wav.numpy(), np.ndarray)

    with pytest.raises(ValueError, match="mono"):
        # 3-D stereo input [B=1, C=2, L] — must be mono.
        bad = torch.zeros(1, 2, L)
        codec.encode_reference(bad, sample_rate=24_000)


def test_decode_roundtrip_shape(codec):
    """``decode(encode(wav))`` produces a waveform whose length is within
    one codec hop of the input length."""
    import torch

    L = int(1.5 * 24_000)
    wav = 0.3 * torch.sin(2 * math.pi * 440 * torch.linspace(0, 1.5, L))
    codes = codec.encode_reference(wav, sample_rate=24_000)
    out = codec.decode(codes)
    assert out.ndim == 1
    # One codec frame ≈ 960 samples; allow a couple frames of slack.
    assert abs(out.shape[0] - L) <= 2_000


def test_decode_rejects_non_2d_codes(codec):
    """``decode`` strictly expects ``[T, num_codebooks]``; 1-D / 3-D raise."""
    import torch

    with pytest.raises(ValueError, match=r"\[T, num_codebooks\]"):
        codec.decode(torch.zeros(8, dtype=torch.long))

    with pytest.raises(ValueError, match=r"\[T, num_codebooks\]"):
        codec.decode(torch.zeros(1, 10, 8, dtype=torch.long))


@pytest.mark.skipif(
    _real_tts_missing,
    reason=f"Real Higgs TTS ckpt not mounted at {_REAL_TTS_CKPT}",
)
def test_from_tts_ckpt_matches_standalone_codec():
    """Loading the codec directly from a TTS ckpt should produce a
    model with the same structure as a standalone codec load. Output
    may drift slightly because TTS training may have touched the
    (nominally frozen) codec weights — we check mel-energy overlap
    rather than bitwise equality."""
    import math

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec

    codec_tts = HiggsAudioCodec.from_tts_ckpt(_REAL_TTS_CKPT, device="cuda")
    codec_sa = HiggsAudioCodec.from_pretrained(_REAL_CODEC_CKPT, device="cuda")

    # Same parameter count / sample rate — smoke of architecture match.
    assert codec_tts.SAMPLE_RATE == codec_sa.SAMPLE_RATE
    assert sum(p.numel() for p in codec_tts.model.parameters()) == sum(
        p.numel() for p in codec_sa.model.parameters()
    )

    # Round-trip the same waveform through both; waveforms must be very
    # close (>= 0.99 cosine) even if the discrete codes diverge.
    wav = 0.3 * torch.sin(2 * math.pi * 440 * torch.linspace(0, 1.0, 24_000))
    c1 = codec_tts.encode_reference(wav, sample_rate=24_000)
    c2 = codec_sa.encode_reference(wav, sample_rate=24_000)
    w1 = codec_tts.decode(c1)
    w2 = codec_sa.decode(c2)
    cos = torch.nn.functional.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0)).item()
    assert cos >= 0.99, f"TTS-ckpt codec vs standalone cosine {cos:.4f} < 0.99"
