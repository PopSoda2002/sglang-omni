#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microphone → /v1/realtime client (transcription).

Streams 16 kHz mono PCM16 from the default input device against
``/v1/realtime`` and prints incremental transcript deltas. Press
``ENTER`` (or send EOF on stdin) to commit the audio buffer and request
a transcription.

Usage::

    python examples/realtime_mic_client.py \\
        --url ws://127.0.0.1:8000/v1/realtime --chunk-ms 100

Requirements: ``websockets``, ``sounddevice``, ``numpy``. M0 has no
server-side VAD — you press ENTER to commit. Server VAD lands in M2 and
the commit will become automatic.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import struct
import sys


async def _read_stdin_line() -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, sys.stdin.readline)


async def _run(args: argparse.Namespace) -> int:
    try:
        import sounddevice as sd
        import websockets
    except ImportError as exc:
        print(f"missing dependency: {exc}", file=sys.stderr)
        print("install with: pip install sounddevice websockets numpy", file=sys.stderr)
        return 2

    sample_rate = 16000
    blocksize = int(sample_rate * args.chunk_ms / 1000)
    audio_q: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _audio_cb(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        # indata is float32 in [-1, 1] when dtype="float32"; convert to PCM16.
        scaled = (indata[:, 0] * 32767.0).clip(-32768, 32767).astype("int16")
        loop.call_soon_threadsafe(
            audio_q.put_nowait, struct.pack(f"<{scaled.size}h", *scaled.tolist())
        )

    print(f"[client] connecting to {args.url}")
    async with websockets.connect(args.url) as ws:
        async def receiver() -> None:
            async for raw in ws:
                event = json.loads(raw)
                etype = event.get("type")
                if etype == "session.created":
                    print(f"[server] session.created id={event['session']['id']}")
                elif etype == "conversation.item.input_audio_transcription.delta":
                    sys.stdout.write(event.get("delta", ""))
                    sys.stdout.flush()
                elif etype == "conversation.item.input_audio_transcription.completed":
                    sys.stdout.write("\n")
                    print(
                        "[server] completed: "
                        f"{event.get('transcript', '')!r}"
                    )
                elif etype == "conversation.item.input_audio_transcription.failed":
                    print(f"[server] failed: {event.get('error')}", file=sys.stderr)
                elif etype == "error":
                    print(f"[server] error: {event.get('error')}", file=sys.stderr)

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
            while True:
                chunk = await audio_q.get()
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }
                    )
                )

        async def committer() -> None:
            print("[client] recording — press ENTER to commit (Ctrl-C to quit).")
            while True:
                await _read_stdin_line()
                print("[client] committing buffer …")
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=_audio_cb,
        ):
            tasks = [
                asyncio.create_task(receiver()),
                asyncio.create_task(sender()),
                asyncio.create_task(committer()),
            ]
            try:
                await asyncio.gather(*tasks)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            for t in tasks:
                t.cancel()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream microphone audio to /v1/realtime."
    )
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8000/v1/realtime",
        help="WebSocket URL (default: %(default)s).",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=100,
        help="Per-frame size in ms for input_audio_buffer.append (default: 100).",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
