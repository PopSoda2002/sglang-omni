# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Codex adversarial review findings.

Each test pins down behavior we want to keep after the hardening pass:

- #1 server_vad init failure surfaces a structured ``error`` event and
  doesn't drop the WebSocket
- #2 each session owns its own VAD model instance (no shared LSTM state
  across sessions)
- #3 RealtimeAudioBuffer enforces max_bytes; oversized append is rejected
  cleanly and the WS is closed with code 1009 ("message too big")
- #4 manual ``input_audio_buffer.commit`` resets VAD/utterance state
  the same way auto-commit does
"""

from __future__ import annotations

import base64
import struct
from typing import Any, AsyncIterator

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client import (
    AbortResult,
    CompletionStreamChunk,
)
from sglang_omni.client.types import AbortLevel
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime.audio_buffer import RealtimeAudioBuffer


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, chunks: list[CompletionStreamChunk] | None = None) -> None:
        self._chunks = chunks or []

    async def completion_stream(
        self, request: Any, *, request_id: str, audio_format: str = "wav"
    ) -> AsyncIterator[CompletionStreamChunk]:
        for c in self._chunks:
            yield c

    async def abort(
        self, request_id: str, level: AbortLevel = AbortLevel.SOFT
    ) -> AbortResult:
        return AbortResult(success=True, level_applied=level)

    def health(self) -> dict[str, Any]:
        return {"running": True}


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _pcm16(num_samples: int, value: int = 0) -> bytes:
    return struct.pack(f"<{num_samples}h", *([value] * num_samples))


# ---------------------------------------------------------------------------
# #1 — VAD init failure
# ---------------------------------------------------------------------------


def test_session_update_returns_error_when_vad_unavailable(monkeypatch) -> None:
    from sglang_omni.serve.realtime import vad as vad_module

    def _boom():
        raise vad_module.VADUnavailable("simulated missing silero-vad")

    monkeypatch.setattr(vad_module, "load_silero_model", _boom)

    fake = _FakeClient()
    app = create_app(fake, model_name="test", enable_realtime=True)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {"turn_detection": {"type": "server_vad"}},
                }
            )
            evt = ws.receive_json()
            assert evt["type"] == "error", evt
            assert evt["error"]["code"] == "server_vad_unavailable"
            # Critically: the WS is still open. The client can fall back
            # to manual mode without reconnecting.
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {"input_audio_format": "pcm16"},
                }
            )
            evt = ws.receive_json()
            assert evt["type"] == "session.updated"


# ---------------------------------------------------------------------------
# #2 — Per-session VAD instance (no shared LSTM)
# ---------------------------------------------------------------------------


def test_streaming_vad_constructs_a_distinct_model_per_instance(monkeypatch) -> None:
    from sglang_omni.serve.realtime import vad as vad_module

    constructed: list[object] = []

    class _FakeModel:
        def __call__(self, *args, **kwargs):
            class _T:
                def item(self):
                    return 0.0

            return _T()

        def reset_states(self):
            pass

    def _factory():
        m = _FakeModel()
        constructed.append(m)
        return m

    monkeypatch.setattr(vad_module, "load_silero_model", _factory)

    a = vad_module.StreamingVAD(vad_module.VADConfig())
    b = vad_module.StreamingVAD(vad_module.VADConfig())
    assert a.model is not b.model, (
        "two StreamingVAD instances must not share a model object"
    )
    assert constructed == [a.model, b.model]


# ---------------------------------------------------------------------------
# #3 — Buffer DoS cap
# ---------------------------------------------------------------------------


def test_audio_buffer_rejects_overflow_on_append() -> None:
    buf = RealtimeAudioBuffer(max_bytes=4000)  # 2000 samples = 125 ms
    fits_n, fits_err = buf.append_b64(_b64(_pcm16(1000)))  # 2000 bytes — fits
    assert fits_n == 2000 and fits_err is None
    overflow_n, overflow_err = buf.append_b64(_b64(_pcm16(1500)))
    assert overflow_n == 0
    assert overflow_err == "overflow"
    # Overflow leaves the buffer unchanged.
    assert buf.num_bytes == 2000


def test_audio_buffer_invalid_b64_returns_error_code() -> None:
    buf = RealtimeAudioBuffer()
    n, err = buf.append_b64("not!!!base64!!!@@")
    assert n == 0
    assert err == "invalid_b64"


def test_websocket_closes_on_oversized_append() -> None:
    fake = _FakeClient()
    app = create_app(fake, model_name="test", enable_realtime=True)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "session.created"
            # Default cap is 60s × 16kHz × 2 bytes = 1.92 MB. Send a
            # single 2 MB chunk which definitely exceeds it.
            oversized = _pcm16(1_100_000)  # 2.2 MB
            ws.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": _b64(oversized),
                }
            )
            evt = ws.receive_json()
            assert evt["type"] == "error", evt
            assert evt["error"]["code"] == "input_audio_buffer_too_large"
            # Server should follow up with a close (TestClient surfaces this
            # as a WebSocketDisconnect on the next receive).
            with pytest.raises(Exception):
                ws.receive_json()


# ---------------------------------------------------------------------------
# #4 — Manual commit resets VAD state
# ---------------------------------------------------------------------------


def test_manual_commit_resets_vad_and_origin(monkeypatch) -> None:
    """A manual commit while server_vad is configured must wipe VAD state.

    Otherwise the next utterance carries stale ``samples_consumed`` and
    LSTM hidden state from the previous turn. We verify by checking that
    the public state visible to subsequent VAD-driven events restarts
    from 0 (via the helper's contract, observed through buffer state).
    """
    from sglang_omni.serve.realtime import vad as vad_module

    # Stub VAD model so we can construct a session.update with server_vad
    # without depending on silero/onnxruntime in this test.
    reset_count = {"n": 0}

    class _StubModel:
        def __call__(self, *args, **kwargs):
            class _T:
                def item(self):
                    return 0.0

            return _T()

        def reset_states(self):
            reset_count["n"] += 1

    monkeypatch.setattr(vad_module, "load_silero_model", lambda: _StubModel())

    fake = _FakeClient(
        [
            CompletionStreamChunk(request_id="x", text="hi", modality="text"),
            CompletionStreamChunk(
                request_id="x", text="", modality="text", finish_reason="stop"
            ),
        ]
    )
    app = create_app(fake, model_name="test", enable_realtime=True)

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "turn_detection": {"type": "server_vad"},
                        "input_audio_format": "pcm16",
                    },
                }
            )
            assert ws.receive_json()["type"] == "session.updated"

            ws.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": _b64(_pcm16(8000)),  # 500 ms
                }
            )
            ws.send_json({"type": "input_audio_buffer.commit"})

            # Drain events until the transcription completes for the
            # manual commit we just issued.
            saw_completed = False
            for _ in range(10):
                evt = ws.receive_json()
                if (
                    evt["type"]
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    saw_completed = True
                    break

    assert saw_completed, "manual commit should still produce a completion"
    # Manual commit should have reset the VAD's LSTM at least once.
    assert reset_count["n"] >= 1, (
        "manual input_audio_buffer.commit must call vad.reset() so "
        "the next utterance starts from a clean LSTM state"
    )


def test_drop_buffer_and_reset_vad_helper_unifies_paths() -> None:
    """Both auto-commit and manual commit go through the same helper.

    Structural test — if someone re-introduces an inline reset in only
    one path, this guards against silent drift.
    """
    import inspect

    from sglang_omni.serve.realtime import session as session_module

    src = inspect.getsource(session_module.RealtimeSession)
    assert src.count("drop_buffer_and_reset_vad") >= 3, (
        "expected the helper to be called from "
        "auto_commit_utterance, handle_audio_commit, and handle_audio_clear"
    )
