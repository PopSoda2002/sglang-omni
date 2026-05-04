#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Streaming WAV → /v1/realtime client (transcription).

Replays a 16 kHz mono WAV file at real-time speed against the
``/v1/realtime`` WebSocket endpoint, manually committing the audio buffer
on EOF, and prints the transcript deltas as they arrive.

Usage::

    python examples/realtime_file_client.py path/to/audio.wav \\
        --url ws://127.0.0.1:8000/v1/realtime --chunk-ms 200

Requirements: ``websockets``, ``numpy``. No GPU needed on the client side.
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


def _read_wav_pcm16_mono_16k(path: Path) -> bytes:
    """Read a WAV file and return raw little-endian PCM16 bytes at 16 kHz mono.

    Resampling and channel-mixing are linear, mirroring
    :func:`sglang_omni.preprocessing.audio._resample_linear`.
    """
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise ValueError(
            f"Expected 16-bit PCM WAV, got {sampwidth * 8}-bit. "
            "Convert with `ffmpeg -i in.wav -acodec pcm_s16le -ar 16000 -ac 1 out.wav`"
        )
    samples = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)

    if sr != 16000:
        # Simple linear resample to 16 kHz.
        new_len = max(int(round(samples.shape[0] * 16000 / sr)), 1)
        old_idx = np.arange(samples.shape[0], dtype=np.float64)
        new_idx = np.linspace(0.0, samples.shape[0] - 1, num=new_len)
        samples = np.interp(new_idx, old_idx, samples).astype(np.int16)

    return struct.pack(f"<{samples.size}h", *samples.tolist())


async def _replay(args: argparse.Namespace) -> int:
    import websockets

    pcm = _read_wav_pcm16_mono_16k(Path(args.audio))
    print(f"[client] loaded {len(pcm)} bytes ({len(pcm) // 2} samples @ 16 kHz)")

    chunk_samples = int(16000 * args.chunk_ms / 1000)
    chunk_bytes = chunk_samples * 2

    transcript_parts: list[str] = []
    completed = asyncio.Event()
    failed: list[str] = []

    async with websockets.connect(args.url) as ws:
        async def receiver() -> None:
            async for raw in ws:
                event = json.loads(raw)
                etype = event.get("type")
                if etype == "session.created":
                    print(f"[server] session.created id={event['session']['id']}")
                elif etype == "input_audio_buffer.committed":
                    print(f"[server] committed item={event['item_id']}")
                elif etype == "conversation.item.input_audio_transcription.delta":
                    delta = event.get("delta", "")
                    transcript_parts.append(delta)
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                elif etype == "conversation.item.input_audio_transcription.completed":
                    sys.stdout.write("\n")
                    print(
                        f"[server] transcription.completed: "
                        f"{event.get('transcript', '')!r}"
                    )
                    completed.set()
                    return
                elif etype == "conversation.item.input_audio_transcription.failed":
                    print(
                        "[server] transcription.failed: "
                        f"{event.get('error')}",
                        file=sys.stderr,
                    )
                    failed.append(json.dumps(event.get("error", {})))
                    completed.set()
                    return
                elif etype == "error":
                    print(f"[server] error: {event.get('error')}", file=sys.stderr)
                    failed.append(json.dumps(event.get("error", {})))
                    completed.set()
                    return

        async def sender() -> None:
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "modalities": ["text"],
                            "input_audio_format": "pcm16",
                        },
                    }
                )
            )

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
                if not args.no_realtime:
                    target = t0 + sent / 32000.0  # 32_000 = bytes/sec at 16k mono
                    delay = target - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)

            print(f"[client] sent {sent} bytes; committing")
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        send_task = asyncio.create_task(sender())
        recv_task = asyncio.create_task(receiver())

        try:
            await asyncio.wait_for(completed.wait(), timeout=args.timeout)
        except asyncio.TimeoutError:
            print(f"[client] timeout after {args.timeout}s", file=sys.stderr)
            send_task.cancel()
            recv_task.cancel()
            return 2
        send_task.cancel()
        recv_task.cancel()

    if failed:
        return 1
    print(f"\n[client] final transcript: {''.join(transcript_parts)!r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay a WAV file against /v1/realtime and print the transcript."
    )
    parser.add_argument("audio", help="Path to a 16-bit PCM WAV (any rate, any channels).")
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8000/v1/realtime",
        help="WebSocket URL (default: %(default)s).",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=200,
        help="Chunk size in ms per input_audio_buffer.append (default: 200).",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Send chunks back-to-back instead of pacing at real-time speed.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Overall wall-clock timeout in seconds (default: 120).",
    )
    args = parser.parse_args()
    return asyncio.run(_replay(args))


if __name__ == "__main__":
    raise SystemExit(main())
