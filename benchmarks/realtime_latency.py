#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Realtime API latency benchmark.

Measures the wall-clock latency from ``input_audio_buffer.commit`` to
the first ``conversation.item.input_audio_transcription.delta`` (i.e.
"first audible response after I stopped speaking") under both the
manual-commit path and the server-VAD path.

Usage::

    python benchmarks/realtime_latency.py \\
        --url ws://127.0.0.1:8765/v1/realtime \\
        --audio /tmp/realtime_fixture.wav \\
        --runs 10

Reports p50 / p95 / max for first-delta and full-transcript latency.
Persistent numbers live in ``benchmarks/baselines/realtime_latency.py``
(``BASELINES`` dict, keyed by hardware tag) and are used by CI to
catch regressions.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import struct
import time
import wave
from pathlib import Path

import numpy as np


def _read_wav_pcm16_16k(path: Path) -> bytes:
    with wave.open(str(path)) as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        data = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError(f"expected 16-bit PCM, got {sw * 8}-bit")
    samples = np.frombuffer(data, dtype="<i2")
    if ch > 1:
        samples = samples.reshape(-1, ch).mean(axis=1).astype(np.int16)
    if sr != 16000:
        new_len = max(int(round(samples.shape[0] * 16000 / sr)), 1)
        old_idx = np.arange(samples.shape[0], dtype=np.float64)
        new_idx = np.linspace(0.0, samples.shape[0] - 1, num=new_len)
        samples = np.interp(new_idx, old_idx, samples).astype(np.int16)
    return struct.pack(f"<{samples.size}h", *samples.tolist())


async def _one_run(
    url: str, pcm: bytes, *, mode: str, chunk_ms: int, timeout: float
) -> dict[str, float]:
    """Run a single transcription session and return per-event latencies."""
    import websockets

    chunk_bytes = int(16000 * chunk_ms / 1000) * 2
    # When using server VAD we append a 1.5 s silence trailer so VAD
    # reliably emits speech_stopped — without that the run hangs.
    pcm_with_trailer = pcm + b"\x00\x00" * (16000 * 3 // 2 if mode == "vad" else 0)

    times: dict[str, float] = {}

    async with websockets.connect(url) as ws:
        # session.update
        cfg: dict[str, object] = {
            "modalities": ["text"],
            "input_audio_format": "pcm16",
        }
        if mode == "vad":
            cfg["turn_detection"] = {
                "type": "server_vad",
                "threshold": 0.5,
                "silence_duration_ms": 600,
                "prefix_padding_ms": 200,
            }
        await ws.send(json.dumps({"type": "session.update", "session": cfg}))

        # drain session.created and session.updated
        for _ in range(2):
            await ws.recv()

        # Stream audio (real-time pacing — we want realistic numbers).
        sent = 0
        t_start_audio = time.monotonic()
        while sent < len(pcm_with_trailer):
            chunk = pcm_with_trailer[sent : sent + chunk_bytes]
            sent += len(chunk)
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            )
            target = t_start_audio + sent / 32000.0
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

        if mode == "manual":
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            times["t_commit"] = time.monotonic()

        # Receive until transcription.completed.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            raw = await asyncio.wait_for(
                ws.recv(), timeout=max(0.1, deadline - time.monotonic())
            )
            event = json.loads(raw)
            etype = event.get("type")
            now = time.monotonic()
            if etype == "input_audio_buffer.speech_stopped":
                times["t_speech_stopped"] = now
            elif etype == "input_audio_buffer.committed":
                times.setdefault("t_committed", now)
            elif etype == "conversation.item.input_audio_transcription.delta":
                times.setdefault("t_first_delta", now)
            elif etype == "conversation.item.input_audio_transcription.completed":
                times["t_completed"] = now
                break
            elif etype == "error":
                raise RuntimeError(f"server error: {event['error']}")

    return times


def _summarize(times_list: list[dict[str, float]], *, mode: str) -> dict[str, dict]:
    """Convert per-run timestamps to per-event latencies (ms)."""
    out: dict[str, list[float]] = {
        "first_delta_after_commit_ms": [],
        "completed_after_commit_ms": [],
        "first_delta_after_speech_stop_ms": [],
    }
    for t in times_list:
        ref = (
            t.get("t_commit")
            if mode == "manual"
            else t.get("t_speech_stopped") or t.get("t_committed")
        )
        if ref is None:
            continue
        if "t_first_delta" in t:
            out["first_delta_after_commit_ms"].append(
                (t["t_first_delta"] - ref) * 1000
            )
        if "t_completed" in t:
            out["completed_after_commit_ms"].append((t["t_completed"] - ref) * 1000)
        if "t_first_delta" in t and "t_speech_stopped" in t:
            out["first_delta_after_speech_stop_ms"].append(
                (t["t_first_delta"] - t["t_speech_stopped"]) * 1000
            )

    summary: dict[str, dict] = {}
    for name, vals in out.items():
        if not vals:
            continue
        summary[name] = {
            "n": len(vals),
            "p50_ms": statistics.median(vals),
            "p95_ms": (
                statistics.quantiles(vals, n=20)[-1] if len(vals) >= 4 else max(vals)
            ),
            "max_ms": max(vals),
            "min_ms": min(vals),
        }
    return summary


async def _main(args: argparse.Namespace) -> int:
    pcm = _read_wav_pcm16_16k(Path(args.audio))
    print(f"[bench] audio: {len(pcm) // 2} samples = {len(pcm) / 32000:.2f}s")

    results: dict[str, list[dict[str, float]]] = {}
    for mode in args.modes:
        per_run = []
        for i in range(args.runs):
            t = await _one_run(
                args.url,
                pcm,
                mode=mode,
                chunk_ms=args.chunk_ms,
                timeout=args.timeout,
            )
            print(f"[bench] mode={mode} run={i} → {t}")
            per_run.append(t)
        results[mode] = per_run

    print("\n=== Summary ===")
    full_report: dict[str, dict] = {}
    for mode, runs in results.items():
        s = _summarize(runs, mode=mode)
        full_report[mode] = s
        print(f"\nmode={mode}")
        for k, v in s.items():
            print(
                f"  {k:40s} n={v['n']:>3} "
                f"p50={v['p50_ms']:>7.1f}ms "
                f"p95={v['p95_ms']:>7.1f}ms "
                f"max={v['max_ms']:>7.1f}ms"
            )

    if args.report:
        Path(args.report).write_text(json.dumps(full_report, indent=2))
        print(f"\n[bench] report → {args.report}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8765/v1/realtime")
    parser.add_argument("--audio", default="/tmp/realtime_fixture.wav")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["manual", "vad"],
        default=["manual", "vad"],
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--report",
        default="/tmp/realtime_latency_report.json",
        help="Write JSON summary to this path.",
    )
    args = parser.parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
