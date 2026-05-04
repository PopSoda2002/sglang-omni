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
    if args.server_vad:
        # Append a 1.5 s silence trailer so the server VAD reliably emits
        # speech_stopped on the last utterance.
        pcm = pcm + b"\x00\x00" * (16000 * 3 // 2)
    print(f"[client] loaded {len(pcm)} bytes ({len(pcm) // 2} samples @ 16 kHz)")

    chunk_samples = int(16000 * args.chunk_ms / 1000)
    chunk_bytes = chunk_samples * 2

    transcript_parts: list[str] = []
    # Manual mode: exactly one commit ⇒ exactly one completion. Server-VAD:
    # one completion per detected utterance, and we only know we're done
    # after we've sent every byte AND every started utterance has
    # produced a stop+commit+completion.
    expected_completions = 1 if not args.server_vad else 0
    seen_completions = 0
    starts_seen = 0
    stops_seen = 0
    failed: list[str] = []
    completed = asyncio.Event()
    audio_done = asyncio.Event()

    async with websockets.connect(args.url) as ws:

        async def receiver() -> None:
            nonlocal seen_completions, expected_completions, starts_seen, stops_seen
            async for raw in ws:
                event = json.loads(raw)
                etype = event.get("type")
                if etype == "session.created":
                    print(f"[server] session.created id={event['session']['id']}")
                elif etype == "input_audio_buffer.speech_started":
                    starts_seen += 1
                    if args.server_vad:
                        expected_completions += 1
                    print(
                        f"[server] speech_started @ "
                        f"{event.get('audio_start_ms')}ms"
                    )
                elif etype == "input_audio_buffer.speech_stopped":
                    stops_seen += 1
                    print(
                        f"[server] speech_stopped @ "
                        f"{event.get('audio_end_ms')}ms"
                    )
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
                    seen_completions += 1
                    # Done when we've shipped all audio AND every started
                    # utterance has completed.
                    if (
                        audio_done.is_set()
                        and seen_completions >= max(starts_seen, expected_completions)
                    ):
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
            session_cfg: dict[str, object] = {
                "modalities": ["text"],
                "input_audio_format": "pcm16",
            }
            if args.server_vad:
                session_cfg["turn_detection"] = {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "silence_duration_ms": 600,
                    "prefix_padding_ms": 200,
                }
            await ws.send(
                json.dumps({"type": "session.update", "session": session_cfg})
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

            print(f"[client] sent {sent} bytes")
            if not args.server_vad:
                print("[client] committing")
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            audio_done.set()

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
    parser.add_argument(
        "--server-vad",
        action="store_true",
        help=(
            "Enable server-side VAD (turn_detection=server_vad). The "
            "client streams without an explicit commit; VAD auto-commits "
            "each utterance. Appends 1.5s silence to ensure the last "
            "speech_stopped fires."
        ),
    )
    args = parser.parse_args()
    return asyncio.run(_replay(args))


if __name__ == "__main__":
    raise SystemExit(main())
