# SPDX-License-Identifier: Apache-2.0
"""Tests for RealtimeSession's single-slot serialization between the
response and transcription paths.

Both paths share ``self.active_task`` and ``self.active_request_id`` —
the bug was that ``drain_queue`` clobbered whatever the slot held, so
a ``response.create`` followed by VAD-driven transcription would leak
the response task in the background and ``response.cancel`` would
cancel the wrong engine stream.

These tests exercise the slot ownership rules directly: drain_queue
must wait for the slot to release, ``response.cancel`` must only act on
responses, and the rejection path must report which path holds the
slot.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest
from starlette.websockets import WebSocketState

from sglang_omni_v1.serve.realtime.events import ResponseCancel, ResponseCreate
from sglang_omni_v1.serve.realtime.session import RealtimeSession


@dataclass
class _FakeChunk:
    text: str | None = None
    finish_reason: str | None = None
    modality: str = "text"
    usage: object | None = None


@dataclass
class _Program:
    """One scripted completion_stream invocation.

    The fake client awaits ``gate`` before yielding any chunks so the
    test can synchronize on "engine call has started" via ``started``
    and then release with ``gate.set()``. ``request_id`` is captured so
    tests can assert abort propagates to the right id.
    """

    chunks: list[_FakeChunk]
    gate: asyncio.Event = field(default_factory=asyncio.Event)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    request_id: str | None = None


class _FakeClient:
    def __init__(self) -> None:
        self.programs: list[_Program] = []
        self.aborted: list[str] = []

    def program(self, *chunks: _FakeChunk) -> _Program:
        p = _Program(chunks=list(chunks))
        self.programs.append(p)
        return p

    async def completion_stream(self, req, *, request_id):  # type: ignore[no-untyped-def]
        if not self.programs:
            yield _FakeChunk(finish_reason="stop")
            return
        p = self.programs.pop(0)
        p.request_id = request_id
        p.started.set()
        await p.gate.wait()
        for c in p.chunks:
            yield c

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.client_state = WebSocketState.CONNECTED
        self.closed = False

    async def send_text(self, txt: str) -> None:
        self.sent.append(json.loads(txt))

    async def close(self) -> None:
        self.closed = True
        self.client_state = WebSocketState.DISCONNECTED


def _make_session() -> tuple[RealtimeSession, _FakeClient, _FakeWebSocket]:
    client = _FakeClient()
    ws = _FakeWebSocket()
    session = RealtimeSession(
        websocket=ws,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="test-model",
        session_id="sess_test",
    )
    return session, client, ws


async def _cleanup(session: RealtimeSession) -> None:
    """Cancel drain_queue + any in-flight task so the test ends cleanly."""
    session.closed = True
    if session.queue_drainer is not None and not session.queue_drainer.done():
        session.queue_drainer.cancel()
        await asyncio.gather(session.queue_drainer, return_exceptions=True)
    if session.active_task is not None and not session.active_task.done():
        session.active_task.cancel()
        await asyncio.gather(session.active_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_drain_queue_waits_for_in_flight_response():
    """Transcription must not clobber an in-flight response in the slot."""
    session, client, _ = _make_session()
    resp = client.program(_FakeChunk(text="hi", finish_reason="stop"))
    trans = client.program(_FakeChunk(text="hello", finish_reason="stop"))

    # Kick off a response — it'll park inside completion_stream awaiting resp.gate.
    await session.handle_response_create(ResponseCreate(type="response.create"))
    await asyncio.wait_for(resp.started.wait(), timeout=1.0)
    response_task = session.active_task
    assert response_task is not None
    assert session.active_response_id is not None

    # Enqueue a transcription and start the drainer.
    await session.transcription_queue.put(("item_x", "data:audio/wav;base64,YQ=="))
    session.queue_drainer = asyncio.create_task(session.drain_queue())

    # Give the drainer time to pop and discover the slot is busy. It must
    # NOT have started the transcription engine call yet.
    await asyncio.sleep(0.05)
    assert not trans.started.is_set()
    assert session.active_task is response_task

    # Release the response. Drainer should now claim the slot.
    resp.gate.set()
    await asyncio.wait_for(trans.started.wait(), timeout=1.0)
    assert session.active_response_id is None
    assert session.active_task is not response_task

    trans.gate.set()
    await _cleanup(session)


@pytest.mark.asyncio
async def test_response_cancel_ignores_in_flight_transcription():
    """response.cancel must not touch a transcription holding the slot."""
    session, client, _ = _make_session()
    trans = client.program(_FakeChunk(text="hi", finish_reason="stop"))

    await session.transcription_queue.put(("item_x", "data:audio/wav;base64,YQ=="))
    session.queue_drainer = asyncio.create_task(session.drain_queue())
    await asyncio.wait_for(trans.started.wait(), timeout=1.0)

    # Transcription holds the slot; active_response_id stays None.
    assert session.active_task is not None
    assert session.active_response_id is None

    await session.handle_response_cancel(ResponseCancel(type="response.cancel"))

    # No abort issued, no task cancelled.
    assert client.aborted == []
    assert session.active_task is not None
    assert not session.active_task.done()

    trans.gate.set()
    await _cleanup(session)


@pytest.mark.asyncio
async def test_response_cancel_aborts_in_flight_response():
    """response.cancel should abort the response's request_id and cancel the task."""
    session, client, _ = _make_session()
    resp = client.program(_FakeChunk(text="hi", finish_reason="stop"))

    await session.handle_response_create(ResponseCreate(type="response.create"))
    await asyncio.wait_for(resp.started.wait(), timeout=1.0)
    request_id = session.active_request_id
    assert request_id is not None
    assert session.active_response_id is not None
    response_task = session.active_task

    await session.handle_response_cancel(ResponseCancel(type="response.cancel"))

    assert client.aborted == [request_id]
    # Task is cancelled — gather absorbs the CancelledError.
    await asyncio.gather(response_task, return_exceptions=True)
    assert response_task.cancelled() or response_task.done()


@pytest.mark.asyncio
async def test_response_create_rejection_names_transcription_when_busy():
    """The rejection error should identify which path holds the slot."""
    session, client, ws = _make_session()
    trans = client.program(_FakeChunk(text="hi", finish_reason="stop"))

    await session.transcription_queue.put(("item_x", "data:audio/wav;base64,YQ=="))
    session.queue_drainer = asyncio.create_task(session.drain_queue())
    await asyncio.wait_for(trans.started.wait(), timeout=1.0)

    await session.handle_response_create(ResponseCreate(type="response.create"))

    errors = [e for e in ws.sent if e.get("type") == "error"]
    assert len(errors) == 1
    assert errors[0]["error"]["code"] == "response_in_progress"
    assert "transcription" in errors[0]["error"]["message"]

    trans.gate.set()
    await _cleanup(session)


@pytest.mark.asyncio
async def test_response_create_rejection_names_response_when_busy():
    """A second response.create during a live response should report 'response'."""
    session, client, ws = _make_session()
    resp = client.program(_FakeChunk(text="hi", finish_reason="stop"))

    await session.handle_response_create(ResponseCreate(type="response.create"))
    await asyncio.wait_for(resp.started.wait(), timeout=1.0)

    await session.handle_response_create(ResponseCreate(type="response.create"))

    errors = [e for e in ws.sent if e.get("type") == "error"]
    assert len(errors) == 1
    assert errors[0]["error"]["code"] == "response_in_progress"
    assert "response" in errors[0]["error"]["message"]
    assert "transcription" not in errors[0]["error"]["message"]

    resp.gate.set()
    await _cleanup(session)
