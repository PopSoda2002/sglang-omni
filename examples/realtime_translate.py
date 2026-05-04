#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Realtime streaming translation example.

Demonstrates that the same ``/v1/realtime`` endpoint that does live
transcription also does live translation — just by setting the system
``instructions`` on the session. No code path changes.

Usage::

    python examples/realtime_translate.py path/to/audio.wav \\
        --target-language "Simplified Chinese" \\
        --url ws://127.0.0.1:8765/v1/realtime

Requires: ``websockets``, ``numpy``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import struct
import sys
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


async def _run(args: argparse.Namespace) -> int:
    import websockets

    pcm = _read_wav_pcm16_16k(Path(args.audio))
    if args.server_vad:
        pcm = pcm + b"\x00\x00" * (16000 * 3 // 2)
    chunk_bytes = int(16000 * args.chunk_ms / 1000) * 2

    instructions = (
        f"You are a realtime translator. Translate the spoken audio into "
        f"{args.target_language}. Output ONLY the translated text, no commentary."
    )

    async with websockets.connect(args.url) as ws:
        cfg: dict[str, object] = {
            "modalities": ["text"],
            "input_audio_format": "pcm16",
            "instructions": instructions,
        }
        if args.server_vad:
            cfg["turn_detection"] = {
                "type": "server_vad",
                "threshold": 0.5,
                "silence_duration_ms": 600,
                "prefix_padding_ms": 200,
            }
        await ws.send(json.dumps({"type": "session.update", "session": cfg}))

        async def receiver() -> None:
            async for raw in ws:
                evt = json.loads(raw)
                etype = evt.get("type")
                if etype == "session.created":
                    print(f"[server] session.created id={evt['session']['id']}")
                elif etype == "input_audio_buffer.speech_started":
                    print(f"\n[server] speech_started @ {evt.get('audio_start_ms')}ms")
                elif etype == "input_audio_buffer.speech_stopped":
                    print(f"[server] speech_stopped @ {evt.get('audio_end_ms')}ms")
                elif etype == "conversation.item.input_audio_transcription.delta":
                    sys.stdout.write(evt.get("delta", ""))
                    sys.stdout.flush()
                elif etype == "conversation.item.input_audio_transcription.completed":
                    sys.stdout.write("\n")
                    print(
                        f"[server] translation: "
                        f"{evt.get('transcript', '')!r}"
                    )
                elif etype == "error":
                    print(f"[server] error: {evt.get('error')}", file=sys.stderr)
                    return

        recv_task = asyncio.create_task(receiver())

        sent = 0
        t0 = time.monotonic()
        while sent < len(pcm):
            chunk = pcm[sent : sent + chunk_bytes]
            sent += len(chunk)
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            )
            target = t0 + sent / 32000.0
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

        if not args.server_vad:
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        try:
            await asyncio.wait_for(recv_task, timeout=args.timeout)
        except asyncio.TimeoutError:
            print("[client] timeout", file=sys.stderr)
            return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--url", default="ws://127.0.0.1:8765/v1/realtime")
    parser.add_argument("--target-language", default="Simplified Chinese")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--server-vad", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
