#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M1 pre-flight spike: Qwen3-Omni audio encoder acausality.

Numerically compares
    (1) emb_full   = encoder(mel_extract(audio[0..T]))
    (2) emb_chunked = concat(encoder(mel_extract(audio[i..i+W])) for chunks i)

on multiple chunk widths and reports the relative error. The result
gates the M1 design:

    relative_diff < 1%   → straight KV-extend (best case)
    1% – 10%             → overlap-and-discard at chunk boundaries
    > 10%                → fallback: re-encode full prefix per append,
                           keep thinker prefix KV preserved

Usage::

    python playground/realtime_spike/encoder_acausality.py \\
        --audio /tmp/realtime_fixture.wav \\
        --chunk-ms 250 500 1000 2000

Loads only the Qwen3-Omni audio tower (not the full ~30B model).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    sys.path.insert(0, "/root/sglang-omni")
    from sglang_omni.preprocessing.audio import load_audio_path

    return load_audio_path(path, target_sr=target_sr)


def _load_processor_and_encoder(
    model_path: str, device: str
) -> tuple[Any, torch.nn.Module]:
    sys.path.insert(0, "/root/sglang-omni")
    from transformers import AutoProcessor

    from sglang_omni.models.qwen3_omni.components.audio_encoder import (
        Qwen3OmniAudioEncoder,
    )

    print(f"[spike] loading processor + audio tower from {model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    encoder = Qwen3OmniAudioEncoder(model_path, device=device, dtype="bfloat16")
    encoder.eval()
    print(
        f"[spike] encoder ready on {device}, dtype={encoder.audio_tower.dtype}",
        flush=True,
    )
    return processor, encoder


def _encode_segment(
    *,
    processor: Any,
    encoder: torch.nn.Module,
    audio: np.ndarray,
    sample_rate: int,
    device: str,
) -> torch.Tensor:
    """Run mel-extract + Qwen3-Omni audio tower on one waveform.

    Returns CPU float32 (T, D).
    """
    feat = processor.feature_extractor(
        [audio.astype(np.float32)],
        sampling_rate=sample_rate,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_features = feat["input_features"].to(device)
    attention_mask = feat.get("feature_attention_mask")
    if attention_mask is None:
        attention_mask = feat.get("attention_mask")
    attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        out = encoder(
            input_features=input_features,
            feature_attention_mask=attention_mask,
        )
    emb = out["audio_embeds"]
    if emb.dim() == 3:
        emb = emb.squeeze(0)
    return emb.float().cpu()


def _compare_full_vs_chunked(
    *,
    processor: Any,
    encoder: torch.nn.Module,
    audio: np.ndarray,
    sample_rate: int,
    chunk_ms: int,
    device: str,
) -> dict[str, Any]:
    chunk_samples = int(sample_rate * chunk_ms / 1000)

    t0 = time.monotonic()
    emb_full = _encode_segment(
        processor=processor,
        encoder=encoder,
        audio=audio,
        sample_rate=sample_rate,
        device=device,
    )
    t_full = time.monotonic() - t0

    chunks: list[torch.Tensor] = []
    t0 = time.monotonic()
    for i in range(0, len(audio), chunk_samples):
        seg = audio[i : i + chunk_samples]
        if len(seg) < int(0.05 * sample_rate):  # skip <50 ms tail
            continue
        chunks.append(
            _encode_segment(
                processor=processor,
                encoder=encoder,
                audio=seg,
                sample_rate=sample_rate,
                device=device,
            )
        )
    t_chunked = time.monotonic() - t0
    emb_chunked = torch.cat(chunks, dim=0)

    n = min(emb_full.shape[0], emb_chunked.shape[0])
    diff = (emb_full[:n] - emb_chunked[:n]).abs()
    full_mean = emb_full[:n].abs().mean().item()

    rel = diff.mean().item() / max(full_mean, 1e-8)

    per_frame = diff.mean(dim=-1)
    boundary_idxs: list[int] = []
    cumulative = 0
    for c in chunks[:-1]:
        cumulative += c.shape[0]
        boundary_idxs.append(min(cumulative - 1, n - 1))
    boundary_diffs = (
        [per_frame[i].item() for i in boundary_idxs] if boundary_idxs else []
    )

    return {
        "chunk_ms": chunk_ms,
        "n_chunks": len(chunks),
        "shape_full": list(emb_full.shape),
        "shape_chunked": list(emb_chunked.shape),
        "compared_frames": n,
        "abs_diff_mean": diff.mean().item(),
        "abs_diff_max": diff.max().item(),
        "full_abs_mean": full_mean,
        "relative_diff": rel,
        "boundary_diffs_mean": (
            float(np.mean(boundary_diffs)) if boundary_diffs else 0.0
        ),
        "per_frame_diff_p50": float(np.percentile(per_frame.numpy(), 50)),
        "per_frame_diff_p99": float(np.percentile(per_frame.numpy(), 99)),
        "t_full_s": t_full,
        "t_chunked_s": t_chunked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="HF model id or local path.",
    )
    parser.add_argument(
        "--audio",
        default="/tmp/realtime_fixture.wav",
        help="Path to test WAV (16 kHz mono recommended).",
    )
    parser.add_argument(
        "--chunk-ms",
        nargs="+",
        type=int,
        default=[250, 500, 1000, 2000],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", default="/tmp/realtime_spike_report.json")
    args = parser.parse_args()

    audio = _load_audio(Path(args.audio))
    sample_rate = 16000
    print(
        f"[spike] audio: {len(audio)} samples = {len(audio) / sample_rate:.2f}s",
        flush=True,
    )

    processor, encoder = _load_processor_and_encoder(args.model_path, args.device)

    results: list[dict[str, Any]] = []
    for chunk_ms in args.chunk_ms:
        print(f"\n[spike] chunk_ms={chunk_ms}", flush=True)
        r = _compare_full_vs_chunked(
            processor=processor,
            encoder=encoder,
            audio=audio,
            sample_rate=sample_rate,
            chunk_ms=chunk_ms,
            device=args.device,
        )
        results.append(r)
        print(json.dumps(r, indent=2), flush=True)

    print("\n=== Summary ===", flush=True)
    print(
        f"{'chunk_ms':>10} {'n_chunks':>10} {'rel_diff':>12} "
        f"{'p99_frame':>12} {'boundary':>12} {'verdict':>22}"
    )
    for r in results:
        if r["relative_diff"] < 0.01:
            verdict = "straight KV-extend"
        elif r["relative_diff"] < 0.10:
            verdict = "overlap-and-discard"
        else:
            verdict = "fallback re-encode"
        print(
            f"{r['chunk_ms']:>10} {r['n_chunks']:>10} {r['relative_diff']:>12.4f} "
            f"{r['per_frame_diff_p99']:>12.4f} {r['boundary_diffs_mean']:>12.4f} "
            f"{verdict:>22}"
        )

    Path(args.report).write_text(json.dumps(results, indent=2))
    print(f"\n[spike] report → {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
