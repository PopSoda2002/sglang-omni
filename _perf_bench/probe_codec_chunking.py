"""Stage 0 probe for the Higgs TTS streaming plan
(docs/dev/higgs_tts/streaming_plan.md).

Answers the only question that gates the rest of the design:

    Can ``HiggsAudioCodec.decode`` be called on overlapping chunks of
    a single utterance's codes and produce audio that's perceptually
    indistinguishable from a one-shot decode of the same codes?

Algorithm
---------

For each of N reference WAVs:

  1. ``codes_TN = codec.encode_reference(wav)`` — gold codes.
  2. ``wav_full = codec.decode(codes_TN)`` — one-shot baseline.
  3. For each ``(stride, overlap, crossfade_samples)`` in a sweep grid:
       a. Walk ``codes_TN`` in windows ``[start - overlap, start + stride]``
          (clamped at 0). Decode each window with ``codec.decode``.
       b. Drop the first ``overlap * frame_length`` samples of each
          chunk; that's the overlap with the previous chunk's tail.
          The very first chunk drops nothing.
       c. Linear-crossfade ``crossfade_samples`` of each chunk's head
          into the previous chunk's tail.
       d. Concatenate.
  4. Compare ``wav_chunked`` vs ``wav_full``:
       - mse: mean (wav_chunked - wav_full[:L])**2
       - peak_diff: max |wav_chunked - wav_full[:L]|
       - n_click_edges: count of windows where local peak diff > 0.05

Output
------

A summary table per (stride, overlap, crossfade) showing mse / peak_diff /
n_click_edges averaged across the N samples. The smallest config with
``mse < 1e-4`` and ``n_click_edges == 0`` is the streaming-design pick.

Usage
-----

    docker exec sglang_omni_test bash -lc '\\
        python _perf_bench/probe_codec_chunking.py \\
            --model-path boson-sglang/higgs-audio-v3-tts-4b-base \\
            --seed-tts /ceph/data/higgs_audio_eval/zero_shot_tts/seed_tts/en \\
            --n 5'
"""

from __future__ import annotations

import argparse
import os
import statistics
from dataclasses import dataclass

import torch
import torchaudio

from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec


# Higgs codec frame stride (samples per code frame). 24 kHz / 25 Hz = 960.
# (NB: the streaming plan doc originally said 75 Hz — that was wrong; the
# codec reports ``hop_length = 960`` and ``frame_rate = 25``. So at the
# code rate 1 frame ≈ 40 ms of audio, not 13 ms.)
FRAME_LENGTH = 960

# Sweep grid. Stride is the (data frames) added per chunk; overlap is the
# number of frames of context re-fed from the previous chunk (to fix any
# receptive-field discontinuity); crossfade_samples is the linear blend
# applied across chunk boundaries.
STRIDES = [10, 20, 40]
OVERLAPS = [0, 5, 10, 20, 40]
CROSSFADES = [0, 256, 512]


@dataclass
class ChunkConfig:
    stride: int
    overlap: int
    crossfade_samples: int


def parse_meta(path: str) -> list[tuple[str, str, str, str]]:
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split("|")
            if len(parts) == 4:
                rows.append(tuple(parts))
    return rows


def load_wav(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).to(torch.float32)


def chunked_decode(
    codec: HiggsAudioCodec, codes_TN: torch.Tensor, cfg: ChunkConfig
) -> torch.Tensor:
    """Decode ``codes_TN`` chunk-by-chunk under ``cfg``, mimicking what
    the streaming vocoder will do at runtime.
    """
    T = int(codes_TN.shape[0])
    if cfg.stride <= 0:
        raise ValueError("stride must be positive")

    audio_pieces: list[torch.Tensor] = []
    emitted = 0  # data frames already emitted

    while emitted < T:
        window_start = max(0, emitted - cfg.overlap)
        window_end = min(T, emitted + cfg.stride)
        window = codes_TN[window_start:window_end]
        decoded = codec.decode(window).to(torch.float32)
        drop_samples = (emitted - window_start) * FRAME_LENGTH
        if drop_samples >= decoded.shape[-1]:
            emitted = window_end
            continue
        new_audio = decoded[drop_samples:]

        if audio_pieces and cfg.crossfade_samples > 0:
            prev_tail = audio_pieces[-1]
            n = min(
                cfg.crossfade_samples,
                int(prev_tail.shape[-1]),
                int(new_audio.shape[-1]),
            )
            if n > 0:
                fade_in = torch.linspace(0.0, 1.0, n, dtype=new_audio.dtype)
                fade_out = 1.0 - fade_in
                blended = prev_tail[-n:] * fade_out + new_audio[:n] * fade_in
                audio_pieces[-1] = torch.cat([prev_tail[:-n], blended])
                new_audio = new_audio[n:]

        audio_pieces.append(new_audio)
        emitted = window_end

    return torch.cat(audio_pieces) if audio_pieces else torch.zeros(0)


def count_click_edges(
    diff: torch.Tensor, window: int = 10, threshold: float = 0.05
) -> int:
    """Heuristic: a "click" shows up as a localised spike in
    ``|chunked - full|``. Count non-overlapping windows of ``window``
    samples where the in-window peak exceeds ``threshold``.
    """
    if diff.numel() == 0:
        return 0
    abs_diff = diff.abs()
    n_windows = abs_diff.numel() // window
    if n_windows == 0:
        return int(abs_diff.max().item() > threshold)
    trimmed = abs_diff[: n_windows * window].view(n_windows, window)
    peaks = trimmed.max(dim=-1).values
    return int((peaks > threshold).sum().item())


def evaluate(
    codec: HiggsAudioCodec, codes_TN: torch.Tensor, cfg: ChunkConfig
) -> dict:
    wav_full = codec.decode(codes_TN).to(torch.float32)
    wav_chunked = chunked_decode(codec, codes_TN, cfg)
    L = min(int(wav_full.shape[-1]), int(wav_chunked.shape[-1]))
    diff = wav_chunked[:L] - wav_full[:L]
    return {
        "mse": float((diff * diff).mean().item()),
        "peak_diff": float(diff.abs().max().item()),
        "n_click_edges": count_click_edges(diff),
        "len_full": int(wav_full.shape[-1]),
        "len_chunked": int(wav_chunked.shape[-1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--model-path",
        default="boson-sglang/higgs-audio-v3-tts-4b-base",
        help="Higgs TTS checkpoint (codec weights are bundled inside).",
    )
    ap.add_argument(
        "--seed-tts",
        default="/ceph/data/higgs_audio_eval/zero_shot_tts/seed_tts/en",
        help="Path to a seed-tts/en directory with meta.lst.",
    )
    ap.add_argument("--n", type=int, default=5, help="Number of WAVs to probe.")
    ap.add_argument(
        "--device", default="cuda:0", help="CUDA device for the codec."
    )
    args = ap.parse_args()

    codec = HiggsAudioCodec.from_pretrained(
        args.model_path, device=args.device, dtype=torch.float32
    )
    sr = codec.SAMPLE_RATE

    meta = parse_meta(os.path.join(args.seed_tts, "meta.lst"))[: args.n]
    if not meta:
        raise RuntimeError(f"no meta.lst entries under {args.seed_tts}")

    samples: list[torch.Tensor] = []
    for _utt_id, _ref_text, rel, _target in meta:
        wav = load_wav(os.path.join(args.seed_tts, rel), sr)
        codes_TN = codec.encode_reference(wav, sample_rate=sr)
        samples.append(codes_TN)
        print(f"[probe] loaded {rel}: T={codes_TN.shape[0]} frames")

    configs = [
        ChunkConfig(stride=s, overlap=o, crossfade_samples=c)
        for s in STRIDES
        for o in OVERLAPS
        for c in CROSSFADES
    ]
    print(f"\n[probe] sweeping {len(configs)} (stride, overlap, xfade) configs "
          f"x {len(samples)} samples")
    print()
    print(f"{'stride':>6} {'over':>4} {'xfade':>5}  "
          f"{'mse(mean)':>11} {'peak(max)':>10} {'clicks(sum)':>11}")
    print("-" * 64)

    best: tuple[ChunkConfig, float] | None = None
    for cfg in configs:
        per_sample = [evaluate(codec, codes, cfg) for codes in samples]
        mse_mean = statistics.fmean(r["mse"] for r in per_sample)
        peak_max = max(r["peak_diff"] for r in per_sample)
        clicks_sum = sum(r["n_click_edges"] for r in per_sample)
        print(
            f"{cfg.stride:>6d} {cfg.overlap:>4d} {cfg.crossfade_samples:>5d}  "
            f"{mse_mean:>11.4e} {peak_max:>10.4f} {clicks_sum:>11d}"
        )
        if (
            mse_mean < 1e-4
            and clicks_sum == 0
            and (
                best is None
                or (cfg.overlap + cfg.crossfade_samples / FRAME_LENGTH)
                < (best[0].overlap + best[0].crossfade_samples / FRAME_LENGTH)
            )
        ):
            best = (cfg, mse_mean)

    print()
    if best is not None:
        cfg, mse = best
        print(
            f"[probe] RECOMMENDATION: stride={cfg.stride} overlap={cfg.overlap} "
            f"crossfade_samples={cfg.crossfade_samples}  (mse={mse:.2e})"
        )
    else:
        print(
            "[probe] No config met the acceptance bar (mse<1e-4 and no clicks). "
            "Likely options: (a) raise overlap further, (b) widen the sweep, or "
            "(c) fall back to 'wait-N-then-stream' (degenerate streaming)."
        )


if __name__ == "__main__":
    main()
