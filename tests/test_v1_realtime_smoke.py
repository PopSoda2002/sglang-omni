# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the OpenAI Realtime WebSocket endpoint.

These tests run without a GPU. They use a fake ``Client`` that yields
canned ``CompletionStreamChunk`` instances so we can assert the wire
format and event sequence emitted by ``RealtimeSession`` end-to-end
through FastAPI's ``TestClient``.
"""

from __future__ import annotations

import base64
import struct
from typing import Any, AsyncIterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sglang_omni.client import (
    AbortResult,
    CompletionStreamChunk,
)
from sglang_omni.client.types import AbortLevel
from sglang_omni.preprocessing.audio import pcm16_bytes_to_float32
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime.audio_buffer import (
    AudioBufferError,
    RealtimeAudioBuffer,
)
from sglang_omni.serve.realtime.events import (
    SUPPORTED_CLIENT_EVENT_TYPES,
    parse_client_event,
    SessionUpdate,
    InputAudioBufferAppend,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeClient:
    """Replays pre-built chunks for any ``completion_stream`` call.

    Captures the last ``GenerateRequest`` so tests can assert on the
    audio payload that ``RealtimeSession`` constructed for the engine.
    """

    def __init__(self, chunks: list[CompletionStreamChunk]) -> None:
        self._chunks = chunks
        self.last_request: Any = None
        self.last_request_id: str | None = None
        self.aborted_ids: list[str] = []

    async def completion_stream(
        self, request: Any, *, request_id: str, audio_format: str = "wav"
    ) -> AsyncIterator[CompletionStreamChunk]:
        self.last_request = request
        self.last_request_id = request_id
        for chunk in self._chunks:
            yield chunk

    async def abort(
        self, request_id: str, level: AbortLevel = AbortLevel.SOFT
    ) -> AbortResult:
        self.aborted_ids.append(request_id)
        return AbortResult(success=True, level_applied=level)

    def health(self) -> dict[str, Any]:
        return {"running": True}


def _pcm16_bytes(num_samples: int, value: int = 1000) -> bytes:
    return struct.pack(f"<{num_samples}h", *([value] * num_samples))


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_pcm16_bytes_to_float32_round_trip() -> None:
    raw = _pcm16_bytes(16000, value=16384)
    audio = pcm16_bytes_to_float32(raw, source_sr=16000, target_sr=16000)
    assert audio.dtype == np.float32
    assert audio.shape == (16000,)
    # 16384 / 32768 = 0.5
    assert pytest.approx(audio.mean(), rel=1e-3) == 0.5


def test_pcm16_resample_changes_length() -> None:
    raw = _pcm16_bytes(16000, value=0)
    audio = pcm16_bytes_to_float32(raw, source_sr=16000, target_sr=24000)
    assert audio.shape == (24000,)


def test_pcm16_truncates_partial_sample() -> None:
    # 3 bytes — one full PCM16 sample (2 bytes) + dangling byte should drop.
    raw = b"\x00\x40\xff"
    audio = pcm16_bytes_to_float32(raw, source_sr=16000, target_sr=16000)
    assert audio.shape == (1,)


def test_audio_buffer_append_and_clear() -> None:
    buf = RealtimeAudioBuffer()
    assert buf.is_empty()
    appended = buf.append_b64(_b64(_pcm16_bytes(800)))  # 50 ms @ 16k
    assert appended == 1600
    assert buf.num_samples == 800
    assert buf.num_bytes == 1600

    buf.clear()
    assert buf.is_empty()
    assert buf.num_bytes == 0


def test_audio_buffer_rejects_bad_base64() -> None:
    buf = RealtimeAudioBuffer()
    with pytest.raises(AudioBufferError):
        buf.append_b64("not!!!base64!!!@@")


def test_parse_client_event_dispatch() -> None:
    evt = parse_client_event(
        {"type": "session.update", "session": {"modalities": ["text"]}}
    )
    assert isinstance(evt, SessionUpdate)
    assert evt.session.modalities == ["text"]

    evt = parse_client_event(
        {"type": "input_audio_buffer.append", "audio": "AAAA"}
    )
    assert isinstance(evt, InputAudioBufferAppend)

    assert parse_client_event({"type": "definitely.not.a.real.event"}) is None
    assert parse_client_event({"no_type": True}) is None


def test_supported_client_events_include_m0_subset() -> None:
    # Tripwire: if someone narrows the supported set, real clients break.
    expected = {
        "session.update",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "input_audio_buffer.clear",
        "conversation.item.create",
        "response.create",
        "response.cancel",
    }
    assert expected.issubset(SUPPORTED_CLIENT_EVENT_TYPES)


# ---------------------------------------------------------------------------
# WebSocket end-to-end (driven via FastAPI TestClient)
# ---------------------------------------------------------------------------


def _make_app(chunks: list[CompletionStreamChunk]) -> tuple[Any, FakeClient]:
    fake = FakeClient(chunks)
    app = create_app(fake, model_name="qwen3-omni-test", enable_realtime=True)
    return app, fake


def test_websocket_handshake_emits_session_created() -> None:
    app, _ = _make_app([])
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            evt = ws.receive_json()
            assert evt["type"] == "session.created"
            assert evt["session"]["model"] == "qwen3-omni-test"
            assert evt["session"]["modalities"] == ["text"]


def test_websocket_session_update_then_updated() -> None:
    app, _ = _make_app([])
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text"],
                        "input_audio_format": "pcm16",
                    },
                }
            )
            evt = ws.receive_json()
            assert evt["type"] == "session.updated"
            assert evt["session"]["input_audio_format"] == "pcm16"


def test_websocket_session_update_rejects_audio_modality() -> None:
    app, _ = _make_app([])
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {"modalities": ["text", "audio"]},
                }
            )
            evt = ws.receive_json()
            assert evt["type"] == "error"
            assert "audio" in evt["error"]["message"]


def test_websocket_audio_commit_emits_transcription_completed() -> None:
    chunks = [
        CompletionStreamChunk(request_id="x", text="hello", modality="text"),
        CompletionStreamChunk(request_id="x", text=" world", modality="text"),
        CompletionStreamChunk(
            request_id="x", text="", modality="text", finish_reason="stop"
        ),
    ]
    app, fake = _make_app(chunks)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "session.created"

            ws.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": _b64(_pcm16_bytes(1600)),  # 100 ms
                }
            )
            ws.send_json({"type": "input_audio_buffer.commit"})

            collected: list[dict] = []
            for _ in range(8):
                collected.append(ws.receive_json())
                if (
                    collected[-1]["type"]
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    break

    types = [e["type"] for e in collected]
    assert "input_audio_buffer.committed" in types
    assert "conversation.item.created" in types
    deltas = [
        e["delta"]
        for e in collected
        if e["type"] == "conversation.item.input_audio_transcription.delta"
    ]
    assert deltas == ["hello", " world"]
    completed = next(
        e
        for e in collected
        if e["type"] == "conversation.item.input_audio_transcription.completed"
    )
    assert completed["transcript"] == "hello world"

    # Verify the engine actually saw the audio buffer.
    assert fake.last_request is not None
    audios = fake.last_request.metadata.get("audios")
    assert audios and len(audios) == 1
    # Audio is now serialized as a `data:audio/wav;base64,...` URI so it
    # can survive the msgpack-based pipeline IPC boundary.
    assert isinstance(audios[0], str)
    assert audios[0].startswith("data:audio/wav;base64,")
    # Round-trip: WAV header + 1600 samples × 2 bytes.
    decoded = base64.b64decode(audios[0].split(",", 1)[1])
    assert decoded[:4] == b"RIFF" and decoded[8:12] == b"WAVE"


def test_websocket_audio_clear_emits_cleared() -> None:
    app, _ = _make_app([])
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            ws.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": _b64(_pcm16_bytes(800)),
                }
            )
            ws.send_json({"type": "input_audio_buffer.clear"})
            evt = ws.receive_json()
            assert evt["type"] == "input_audio_buffer.cleared"


def test_websocket_commit_with_empty_buffer_returns_error() -> None:
    app, _ = _make_app([])
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            ws.send_json({"type": "input_audio_buffer.commit"})
            evt = ws.receive_json()
            assert evt["type"] == "error"
            assert evt["error"]["code"] == "input_audio_buffer_commit_empty"


def test_websocket_unknown_event_returns_error() -> None:
    app, _ = _make_app([])
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            ws.send_json({"type": "session.delete"})  # not implemented
            evt = ws.receive_json()
            assert evt["type"] == "error"
            assert evt["error"]["code"] == "unsupported_event"


def test_websocket_invalid_json_returns_error() -> None:
    app, _ = _make_app([])
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            ws.send_text("{not valid json")
            evt = ws.receive_json()
            assert evt["type"] == "error"


def test_websocket_response_create_emits_response_done() -> None:
    chunks = [
        CompletionStreamChunk(request_id="x", text="hi there", modality="text"),
        CompletionStreamChunk(
            request_id="x", text="", modality="text", finish_reason="stop"
        ),
    ]
    app, _ = _make_app(chunks)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello"}],
                    },
                }
            )
            assert ws.receive_json()["type"] == "conversation.item.created"

            ws.send_json({"type": "response.create"})

            collected: list[dict] = []
            for _ in range(6):
                collected.append(ws.receive_json())
                if collected[-1]["type"] == "response.done":
                    break

    types = [e["type"] for e in collected]
    assert "response.created" in types
    assert "response.text.delta" in types
    assert "response.text.done" in types
    assert "response.done" in types
    done = next(e for e in collected if e["type"] == "response.done")
    assert done["response"]["status"] == "completed"


def test_websocket_response_create_rejects_audio_modality() -> None:
    app, _ = _make_app([])
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            ws.send_json(
                {
                    "type": "response.create",
                    "response": {"modalities": ["audio", "text"]},
                }
            )
            evt = ws.receive_json()
            assert evt["type"] == "error"
            assert "audio output is not yet implemented" in evt["error"]["message"]


def test_realtime_route_disabled_by_default() -> None:
    fake = FakeClient([])
    app = create_app(fake, model_name="qwen3-omni-test")  # no enable_realtime
    with TestClient(app) as client:
        with pytest.raises(Exception):
            with client.websocket_connect("/v1/realtime"):
                pass
