"""Smoke client for the Higgs TTS streaming endpoint.

Sends one POST /v1/audio/speech with ``stream=True``, prints per-chunk
wall-clock + cumulative audio duration, dumps the concatenated audio
to a WAV for listening.

Usage:
    docker exec sglang_omni_test bash -lc '
      python _perf_bench/smoke_streaming.py \\
        --url http://localhost:8101 \\
        --input "Hello, this is a streaming smoke test." \\
        --out /tmp/stream_smoke.wav'
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import wave

import numpy as np
import requests

DEFAULT_INPUT = "Hello, this is a streaming smoke test."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8101")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--ref-audio", default=None)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument(
        "--response-format",
        default="pcm",
        help="pcm = raw int16 PCM chunks (decodable chunk-by-chunk); "
             "wav also works but client just concatenates the headerless body.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/tmp/stream_smoke.wav")
    args = ap.parse_args()

    payload = {
        "input": args.input,
        "stream": True,
        "response_format": args.response_format,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
    }
    if args.ref_audio:
        ref: dict = {"audio_path": args.ref_audio}
        if args.ref_text:
            ref["text"] = args.ref_text
        payload["references"] = [ref]

    print(f"[smoke] POST {args.url}/v1/audio/speech  stream=true")
    t0 = time.perf_counter()
    ttfa: float | None = None

    audio_chunks: list[np.ndarray] = []
    sr_seen: int | None = None
    chunk_index = 0
    with requests.post(
        f"{args.url}/v1/audio/speech",
        json=payload,
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data = raw_line[len("data: "):].strip()
            if data == "[DONE]":
                break
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue
            audio_obj = msg.get("audio")
            if audio_obj is None:
                if msg.get("finish_reason"):
                    elapsed = time.perf_counter() - t0
                    usage = msg.get("usage")
                    print(
                        f"[smoke] finish chunk after {elapsed*1000:.0f} ms  "
                        f"finish_reason={msg['finish_reason']}  usage={usage}"
                    )
                continue
            t_now = time.perf_counter()
            if ttfa is None:
                ttfa = t_now - t0
                print(f"[smoke] TTFA = {ttfa*1000:.0f} ms")
            b64 = audio_obj.get("data")
            mime = audio_obj.get("mime_type")
            sample_rate = audio_obj.get("sample_rate")
            if sr_seen is None:
                sr_seen = sample_rate
            audio_bytes = base64.b64decode(b64)
            if mime == "audio/pcm" or args.response_format == "pcm":
                arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768
            elif mime == "audio/wav":
                # The server sends per-chunk WAV with headers; strip the 44-byte
                # PCM-WAV header before concatenating.
                arr = np.frombuffer(audio_bytes[44:], dtype=np.int16).astype(np.float32) / 32768
            else:
                print(f"[smoke] unknown mime {mime!r}; skipping")
                continue
            audio_chunks.append(arr)
            t_elapsed = t_now - t0
            total_audio_s = sum(c.shape[-1] for c in audio_chunks) / max(sr_seen, 1)
            print(
                f"[smoke] chunk {chunk_index:>2d}  +{arr.shape[-1] / sr_seen * 1000:7.0f} ms audio  "
                f"@ {t_elapsed * 1000:7.0f} ms wall  total_audio={total_audio_s:.2f} s"
            )
            chunk_index += 1

    total_wall = time.perf_counter() - t0
    if not audio_chunks:
        print("[smoke] no audio received")
        return
    audio = np.concatenate(audio_chunks)
    sr = sr_seen or 24000
    print(
        f"[smoke] DONE  total_wall={total_wall*1000:.0f} ms  "
        f"total_audio={audio.shape[-1] / sr:.2f} s  chunks={chunk_index}"
    )
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(audio)))
    print(
        f"[smoke] wrote {args.out}  duration={audio.shape[-1]/sr:.2f}s  "
        f"rms={rms:.4f}  peak={peak:.4f}"
    )


if __name__ == "__main__":
    main()
