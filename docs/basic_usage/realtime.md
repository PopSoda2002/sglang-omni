# Realtime API

`/v1/realtime` is an OpenAI-compatible WebSocket endpoint for low-latency
streaming transcription and translation. The wire format mirrors
[OpenAI's Realtime API](https://developers.openai.com/api/docs/guides/realtime),
so any OpenAI Realtime SDK client connects unmodified.

Scope today (Qwen3-Omni thinker only):

- Streaming audio in (PCM16 @ 16 kHz mono, base64 in JSON frames).
- Streaming text deltas out (`response.text.delta` /
  `conversation.item.input_audio_transcription.delta`).
- Manual commit *or* server-side VAD (`turn_detection: server_vad`)
  with auto-commit on speech end.
- Translation by setting `instructions` on the session.

Out of scope (intentionally): audio output, function calling, voice
agents, semantic VAD, Ming-Omni. Use `/v1/chat/completions` for those.

## Launch the server

The endpoint is opt-in. Add `--enable-realtime` to the launcher:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --text-only \
  --port 8765 \
  --enable-realtime \
  --thinker-tp-size 2 --thinker-gpus 0,1
```

`/v1/chat/completions` keeps working unchanged.

## Manual-commit transcription

The simplest flow: stream audio chunks, commit when you're done, read
the transcript.

```python
import asyncio, base64, json, websockets, wave

async def main():
    async with websockets.connect("ws://127.0.0.1:8765/v1/realtime") as ws:
        # 1. Configure the session.
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"modalities": ["text"], "input_audio_format": "pcm16"},
        }))

        # 2. Stream PCM16 chunks.
        with wave.open("clip.wav", "rb") as wf:
            assert wf.getframerate() == 16000 and wf.getnchannels() == 1
            while frame := wf.readframes(3200):  # 200 ms
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(frame).decode("ascii"),
                }))

        # 3. Commit and read until completion.
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        async for raw in ws:
            evt = json.loads(raw)
            if evt["type"] == "conversation.item.input_audio_transcription.delta":
                print(evt["delta"], end="", flush=True)
            elif evt["type"] == "conversation.item.input_audio_transcription.completed":
                print("\nfinal:", evt["transcript"])
                break

asyncio.run(main())
```

A complete file replay (with real-time pacing) lives at
`examples/realtime_file_client.py`; a microphone client at
`examples/realtime_mic_client.py`.

## Server-side VAD (multi-utterance)

Set `turn_detection.type` to `server_vad` and the server will emit
`speech_started` / `speech_stopped` events and auto-commit each
detected utterance:

```python
await ws.send(json.dumps({
    "type": "session.update",
    "session": {
        "modalities": ["text"],
        "input_audio_format": "pcm16",
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 200,
            "silence_duration_ms": 600,
        },
    },
}))
```

Subsequent `input_audio_buffer.append` calls don't need a manual
commit. Multiple detected utterances are queued and transcribed
serially — you get one `transcription.completed` event per utterance.

## Streaming translation

Translation is the same path with a different system prompt. Set
`instructions` on `session.update`:

```python
await ws.send(json.dumps({
    "type": "session.update",
    "session": {
        "modalities": ["text"],
        "input_audio_format": "pcm16",
        "instructions": (
            "You are a realtime translator. Translate the spoken audio "
            "into Simplified Chinese. Output ONLY the translation."
        ),
        "turn_detection": {"type": "server_vad", "silence_duration_ms": 600},
    },
}))
```

Full example: `examples/realtime_translate.py`.

## Supported events

Client → server:

| event | status |
|---|---|
| `session.update` | ✅ |
| `input_audio_buffer.append` | ✅ |
| `input_audio_buffer.commit` | ✅ |
| `input_audio_buffer.clear` | ✅ |
| `conversation.item.create` (text-only) | ✅ |
| `response.create` (modalities=text) | ✅ |
| `response.cancel` | ✅ |

Server → client:

| event | status |
|---|---|
| `session.created` / `session.updated` | ✅ |
| `input_audio_buffer.committed` / `cleared` | ✅ |
| `input_audio_buffer.speech_started` / `speech_stopped` | ✅ (server VAD) |
| `conversation.item.created` | ✅ |
| `conversation.item.input_audio_transcription.delta` / `completed` / `failed` | ✅ |
| `response.created` / `text.delta` / `text.done` / `done` | ✅ |
| `response.audio.delta` / `audio_transcript.delta` | ❌ (out of scope) |
| `response.function_call_arguments.delta` | ❌ (out of scope) |
| `error` | ✅ |

## Latency benchmark

The repo ships `benchmarks/realtime_latency.py` for measuring p50/p95
first-delta and full-transcript latency under both manual-commit and
server-VAD modes:

```bash
python benchmarks/realtime_latency.py \
  --url ws://127.0.0.1:8765/v1/realtime \
  --audio /path/to/12s_fixture.wav \
  --runs 10 \
  --modes manual vad
```

The output is checked into
`benchmarks/baselines/realtime_latency_h20.json` (when present) and is
suitable for CI regression-gating.
