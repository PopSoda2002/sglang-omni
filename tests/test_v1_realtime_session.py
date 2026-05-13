# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the /v1/realtime WebSocket endpoint.

Spins up the realtime route under a FastAPI app via ``TestClient``,
injects a fake ``Client`` backend (no real engine), and drives the
session by sending real JSON over the WebSocket. Assertions are on the
events the server sends back — not on session internals — so the tests
survive any refactor that keeps the wire protocol intact.

The fake ``completion_stream`` parks on a ``threading.Event`` before
yielding any chunks, which lets a test deterministically observe the
moment the engine call has started (``program.started.wait()``) and
later release it (``program.gate.set()``). That's the only synchronization
primitive we need to exercise the slot-conflict scenario in the
single-``active_task`` model — the response engine call holds the slot
until the test releases it, so we can verify the transcription engine
call does *not* run concurrently.
"""
from __future__ import annotations

import asyncio
import base64
import threading
from dataclasses import dataclass, field

from fastapi import FastAPI
from starlette.testclient import TestClient


@dataclass
class _FakeChunk:
    text: str | None = None
    finish_reason: str | None = None
    modality: str = "text"
    usage: object | None = None


@dataclass
class _Program:
    """One scripted ``completion_stream`` invocation."""

    chunks: list[_FakeChunk]
    gate: threading.Event = field(default_factory=threading.Event)
    started: threading.Event = field(default_factory=threading.Event)
    request_id: str | None = None


class _FakeClient:
    """Stand-in for sglang_omni_v1.client.Client.

    Only implements the two methods RealtimeSession uses
    (``completion_stream`` and ``abort``). Each ``completion_stream``
    call parks on the next program's gate before yielding chunks; tests
    open the gate when they're ready to let the engine call complete.
    """

    def __init__(self) -> None:
        self.programs: list[_Program] = []
        self.aborted: list[str] = []

    def program(self, *chunks: _FakeChunk, open_gate: bool = True) -> _Program:
        p = _Program(chunks=list(chunks))
        if open_gate:
            p.gate.set()
        self.programs.append(p)
        return p

    async def completion_stream(self, _req, *, request_id):  # type: ignore[no-untyped-def]
        if not self.programs:
            yield _FakeChunk(finish_reason="stop")
            return
        p = self.programs.pop(0)
        p.request_id = request_id
        p.started.set()
        # asyncio.to_thread lets the event loop keep running while we
        # block a worker thread on the threading.Event — tests can
        # release the gate from the test thread without poking into
        # the running loop.
        await asyncio.to_thread(p.gate.wait)
        for c in p.chunks:
            yield c

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


def _make_app(client: _FakeClient) -> FastAPI:
    from sglang_omni_v1.serve.openai_api import _register_realtime

    app = FastAPI()
    app.state.client = client
    app.state.model_name = "test-model"
    _register_realtime(app)
    return app


def _silence_b64(samples: int = 1024) -> str:
    return base64.b64encode(b"\x00\x00" * samples).decode()


def _collect_until(ws, terminal_type: str, *, limit: int = 50) -> list[dict]:
    events: list[dict] = []
    for _ in range(limit):
        e = ws.receive_json()
        events.append(e)
        if e.get("type") == terminal_type:
            return events
    raise AssertionError(
        f"did not see {terminal_type} after {limit} events; "
        f"saw {[e.get('type') for e in events]}"
    )


# ---------------------------------------------------------------------------
# Handshake / control plane
# ---------------------------------------------------------------------------


def test_session_created_on_connect():
    client = _FakeClient()
    with TestClient(_make_app(client)) as tc:
        with tc.websocket_connect("/v1/realtime") as ws:
            evt = ws.receive_json()
            assert evt["type"] == "session.created"
            assert evt["session"]["id"].startswith("sess_")
            assert evt["session"]["model"] == "test-model"


def test_session_update_acked():
    client = _FakeClient()
    with TestClient(_make_app(client)) as tc:
        with tc.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {"instructions": "be terse"},
                }
            )
            evt = ws.receive_json()
            assert evt["type"] == "session.updated"
            assert evt["session"]["instructions"] == "be terse"


def test_unknown_event_emits_structured_error():
    client = _FakeClient()
    with TestClient(_make_app(client)) as tc:
        with tc.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            ws.send_json({"type": "bogus.event"})
            evt = ws.receive_json()
            assert evt["type"] == "error"
            assert evt["error"]["code"] == "unknown_event"


# ---------------------------------------------------------------------------
# Happy-path lifecycles
# ---------------------------------------------------------------------------


def test_response_create_emits_lifecycle_events():
    client = _FakeClient()
    client.program(
        _FakeChunk(text="hello", finish_reason=None),
        _FakeChunk(text=" world", finish_reason="stop"),
    )
    with TestClient(_make_app(client)) as tc:
        with tc.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            ws.send_json({"type": "response.create"})
            events = _collect_until(ws, "response.done")
            types = [e["type"] for e in events]
            assert types[0] == "response.created"
            assert "response.text.delta" in types
            assert types[-2:] == ["response.text.done", "response.done"]
            deltas = [e["delta"] for e in events if e["type"] == "response.text.delta"]
            assert "".join(deltas) == "hello world"


def test_manual_audio_commit_runs_transcription():
    client = _FakeClient()
    client.program(_FakeChunk(text="user said this", finish_reason="stop"))
    with TestClient(_make_app(client)) as tc:
        with tc.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            ws.send_json({"type": "input_audio_buffer.append", "audio": _silence_b64()})
            ws.send_json({"type": "input_audio_buffer.commit"})

            events = _collect_until(
                ws, "conversation.item.input_audio_transcription.completed"
            )
            types = [e["type"] for e in events]
            assert "input_audio_buffer.committed" in types
            assert "conversation.item.created" in types
            assert "conversation.item.input_audio_transcription.delta" in types
            completed = next(
                e
                for e in events
                if e["type"] == "conversation.item.input_audio_transcription.completed"
            )
            assert completed["transcript"] == "user said this"


# ---------------------------------------------------------------------------
# Slot-conflict regression tests (the actual reviewer bug)
# ---------------------------------------------------------------------------


def test_transcription_waits_for_in_flight_response():
    """drain_queue must not clobber the slot while a response is running.

    The fake parks the response engine call on its gate. While it's
    parked, the test commits an audio item — the transcription should
    queue up but its engine call must NOT start until the response
    completes. We verify by observing that ``response.done`` arrives
    on the wire strictly before the transcription delta.
    """
    client = _FakeClient()
    resp = client.program(_FakeChunk(text="hi", finish_reason="stop"), open_gate=False)
    trans = client.program(_FakeChunk(text="hello", finish_reason="stop"))

    with TestClient(_make_app(client)) as tc:
        with tc.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created

            ws.send_json({"type": "response.create"})
            # Wait for the response engine call to actually enter the
            # fake (i.e. acquired the slot). This is the synchronization
            # point that guarantees the next two messages contend with
            # an active response.
            assert resp.started.wait(timeout=2.0), "response engine call never started"

            ws.send_json({"type": "input_audio_buffer.append", "audio": _silence_b64()})
            ws.send_json({"type": "input_audio_buffer.commit"})

            # Transcription engine call should NOT have started — slot
            # is held by the response that's still parked on its gate.
            assert not trans.started.wait(
                timeout=0.2
            ), "transcription started before response released the slot"

            # Release the response.
            resp.gate.set()

            events = _collect_until(
                ws, "conversation.item.input_audio_transcription.completed"
            )
            types = [e["type"] for e in events]
            resp_done = types.index("response.done")
            trans_delta = types.index(
                "conversation.item.input_audio_transcription.delta"
            )
            assert (
                resp_done < trans_delta
            ), f"response.done must arrive before transcription.delta; got {types}"


def test_response_cancel_targets_only_the_response():
    """response.cancel must abort the response's request id, not the
    transcription's. The original bug was that drain_queue had already
    overwritten ``active_request_id`` with the transcription's id, so
    ``response.cancel`` aborted the wrong engine stream and silently
    killed the user's in-flight transcription.
    """
    client = _FakeClient()
    resp = client.program(_FakeChunk(text="hi", finish_reason="stop"), open_gate=False)
    trans = client.program(_FakeChunk(text="ok", finish_reason="stop"))

    with TestClient(_make_app(client)) as tc:
        with tc.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created

            ws.send_json({"type": "response.create"})
            assert resp.started.wait(timeout=2.0)
            response_request_id = resp.request_id

            ws.send_json({"type": "input_audio_buffer.append", "audio": _silence_b64()})
            ws.send_json({"type": "input_audio_buffer.commit"})

            ws.send_json({"type": "response.cancel"})

            # Release the (now-cancelled) response so the worker thread
            # blocked on gate.wait can return — keeps the to_thread pool
            # tidy.
            resp.gate.set()

            # Drain through transcription completion.
            _collect_until(ws, "conversation.item.input_audio_transcription.completed")

            # Only the response's request id should have been aborted.
            # The transcription must have run with its own id and that
            # id must NOT have been aborted.
            assert client.aborted == [
                response_request_id
            ], f"only response should be aborted; got {client.aborted}"
            assert trans.request_id is not None
            assert trans.request_id not in client.aborted


def test_response_cancel_during_transcription_is_ignored():
    """response.cancel arriving while a transcription holds the slot
    must not cancel the transcription (per OpenAI spec — response.cancel
    only affects responses)."""
    client = _FakeClient()
    trans = client.program(_FakeChunk(text="hi", finish_reason="stop"), open_gate=False)

    with TestClient(_make_app(client)) as tc:
        with tc.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created

            ws.send_json({"type": "input_audio_buffer.append", "audio": _silence_b64()})
            ws.send_json({"type": "input_audio_buffer.commit"})
            # Drain the synchronous commit acks; then transcription
            # engine call enters the fake and parks on gate.
            for _ in range(2):
                ws.receive_json()
            assert trans.started.wait(timeout=2.0)

            ws.send_json({"type": "response.cancel"})

            # Give the cancel time to NOT do its damage — there is no
            # positive signal to wait on, only the absence of an abort.
            # Release the transcription and verify it completed
            # normally with no aborts logged.
            trans.gate.set()
            _collect_until(ws, "conversation.item.input_audio_transcription.completed")

            assert (
                client.aborted == []
            ), f"response.cancel must not abort a transcription; got {client.aborted}"
