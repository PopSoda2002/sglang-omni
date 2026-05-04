# SPDX-License-Identifier: Apache-2.0
"""Per-WebSocket Realtime session.

A ``RealtimeSession`` wraps one ``WebSocket`` connection and one
``Client`` (the engine front door). It dispatches incoming OpenAI
Realtime client events, accumulates audio into a ``RealtimeAudioBuffer``,
and on each ``input_audio_buffer.commit`` / ``response.create`` it builds
a fresh ``GenerateRequest`` and streams the resulting deltas back as
Realtime server events.

This is the **M0** lifecycle: per-commit fresh ``OmniRequest``. The
M1 anchor-request lifecycle replaces ``_run_response`` with a long-lived
in-place input-extension; the WebSocket loop and event vocabulary do not
change.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from sglang_omni.client import (
    Client,
    ClientError,
    GenerateRequest,
    Message,
    SamplingParams,
)
from sglang_omni.serve.realtime.audio_buffer import (
    AudioBufferError,
    RealtimeAudioBuffer,
)
from sglang_omni.serve.realtime.events import (
    ConversationItemCreate,
    InputAudioBufferAppend,
    InputAudioBufferClear,
    InputAudioBufferCommit,
    ResponseCancel,
    ResponseCreate,
    SessionConfig,
    SessionObject,
    SessionUpdate,
    SUPPORTED_CLIENT_EVENT_TYPES,
    make_event,
    parse_client_event,
)

logger = logging.getLogger(__name__)


_DEFAULT_INSTRUCTIONS = (
    "You are a helpful realtime assistant. Reply to the user's audio."
)


@dataclass
class _ConversationItem:
    """In-memory record of a conversation item.

    Stored so that subsequent turns can include prior text context. M0
    treats user audio items as opaque (transcript-only) and assistant
    responses as text. M1 will store assistant audio output for replay.
    """

    item_id: str
    role: str
    text: str = ""
    audio_transcript: str = ""


class RealtimeSession:
    """Owns one WebSocket and one OpenAI-Realtime conversation."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        model_name: str,
        session_id: str | None = None,
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self.session_id = session_id or f"sess_{uuid.uuid4().hex}"

        self._session_object = SessionObject(
            id=self.session_id,
            model=model_name,
            modalities=["text"],
            instructions=_DEFAULT_INSTRUCTIONS,
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            turn_detection=None,
        )

        self._audio_buffer = RealtimeAudioBuffer(source_sr=16000, target_sr=16000)
        self._conversation: list[_ConversationItem] = []
        self._closed = False

        # in-flight response state
        self._active_request_id: str | None = None
        self._active_response_id: str | None = None
        self._active_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Drive the WebSocket loop until the client disconnects.

        Emits ``session.created`` immediately on entry. Subsequent
        events are dispatched in `_dispatch`. All raised exceptions are
        translated to an ``error`` server event before the socket is
        closed; we never propagate to FastAPI.
        """
        await self._send(
            make_event(
                "session.created",
                session=self._session_object.model_dump(exclude_none=True),
            )
        )

        try:
            while True:
                try:
                    raw = await self.websocket.receive_text()
                except WebSocketDisconnect:
                    break

                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    await self._send_error(
                        code="invalid_request_error",
                        message=f"Invalid JSON: {exc}",
                    )
                    continue

                if not isinstance(payload, dict):
                    await self._send_error(
                        code="invalid_request_error",
                        message="Top-level payload must be a JSON object",
                    )
                    continue

                await self._dispatch(payload)
        except Exception:
            logger.exception("Realtime session %s crashed", self.session_id)
            try:
                await self._send_error(
                    code="server_error",
                    message="internal session error",
                )
            except Exception:  # noqa: BLE001 — best-effort
                pass
        finally:
            await self._teardown()

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        if event_type not in SUPPORTED_CLIENT_EVENT_TYPES:
            await self._send_error(
                code="unsupported_event",
                message=f"Event type {event_type!r} is not supported in M0",
                event_id=payload.get("event_id"),
            )
            return

        try:
            event = parse_client_event(payload)
        except ValidationError as exc:
            await self._send_error(
                code="invalid_request_error",
                message=f"Invalid {event_type}: {exc.errors()[0]['msg']}",
                event_id=payload.get("event_id"),
            )
            return

        if event is None:
            await self._send_error(
                code="invalid_request_error",
                message=f"Could not parse event type {event_type!r}",
                event_id=payload.get("event_id"),
            )
            return

        if isinstance(event, SessionUpdate):
            await self._handle_session_update(event)
        elif isinstance(event, InputAudioBufferAppend):
            await self._handle_audio_append(event)
        elif isinstance(event, InputAudioBufferCommit):
            await self._handle_audio_commit(event)
        elif isinstance(event, InputAudioBufferClear):
            await self._handle_audio_clear(event)
        elif isinstance(event, ConversationItemCreate):
            await self._handle_item_create(event)
        elif isinstance(event, ResponseCreate):
            await self._handle_response_create(event)
        elif isinstance(event, ResponseCancel):
            await self._handle_response_cancel(event)

    # ------------------------------------------------------------------
    # session.update
    # ------------------------------------------------------------------

    async def _handle_session_update(self, event: SessionUpdate) -> None:
        cfg: SessionConfig = event.session
        # Apply only the fields that were actually set.
        update = cfg.model_dump(exclude_none=True)
        for key, value in update.items():
            if hasattr(self._session_object, key):
                setattr(self._session_object, key, value)

        # M0 supports only modalities=["text"] and pcm16 input. Reject the
        # rest with a structured error rather than silently accepting.
        unsupported = []
        if self._session_object.input_audio_format != "pcm16":
            unsupported.append(
                f"input_audio_format={self._session_object.input_audio_format!r} "
                "(M0 supports pcm16 only)"
            )
        if "audio" in (self._session_object.modalities or []):
            unsupported.append("modalities=['audio'] (M0 supports text-only output)")
        if (
            self._session_object.turn_detection
            and self._session_object.turn_detection.type
            and self._session_object.turn_detection.type != "none"
        ):
            unsupported.append(
                f"turn_detection.type="
                f"{self._session_object.turn_detection.type!r} (server VAD lands in M2)"
            )

        if unsupported:
            await self._send_error(
                code="unsupported_session_config",
                message="; ".join(unsupported),
                event_id=event.event_id,
            )
            return

        await self._send(
            make_event(
                "session.updated",
                session=self._session_object.model_dump(exclude_none=True),
            )
        )

    # ------------------------------------------------------------------
    # input_audio_buffer.*
    # ------------------------------------------------------------------

    async def _handle_audio_append(self, event: InputAudioBufferAppend) -> None:
        try:
            self._audio_buffer.append_b64(event.audio)
        except AudioBufferError as exc:
            await self._send_error(
                code="invalid_request_error",
                message=str(exc),
                event_id=event.event_id,
            )

    async def _handle_audio_clear(self, event: InputAudioBufferClear) -> None:
        self._audio_buffer.clear()
        await self._send(make_event("input_audio_buffer.cleared"))

    async def _handle_audio_commit(self, event: InputAudioBufferCommit) -> None:
        if self._audio_buffer.is_empty():
            await self._send_error(
                code="input_audio_buffer_commit_empty",
                message="No audio in buffer to commit",
                event_id=event.event_id,
            )
            return

        item_id = f"item_{uuid.uuid4().hex}"
        audio_array = self._audio_buffer.to_numpy()
        self._audio_buffer.clear()

        # Order matches OpenAI: committed → conversation.item.created →
        # transcription deltas/completed (emitted by _run_transcription).
        await self._send(
            make_event(
                "input_audio_buffer.committed",
                previous_item_id=self._previous_item_id(),
                item_id=item_id,
            )
        )
        await self._send(
            make_event(
                "conversation.item.created",
                previous_item_id=self._previous_item_id(),
                item={
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio", "transcript": None}],
                },
            )
        )

        item = _ConversationItem(item_id=item_id, role="user")
        self._conversation.append(item)

        # Run transcription as a background task so the WebSocket loop
        # stays responsive to control events (commit/cancel).
        if self._active_task is not None and not self._active_task.done():
            await self._send_error(
                code="response_in_progress",
                message="A response is already in progress",
                event_id=event.event_id,
            )
            return

        self._active_task = asyncio.create_task(
            self._run_transcription(item_id, audio_array)
        )

    # ------------------------------------------------------------------
    # conversation.item.create
    # ------------------------------------------------------------------

    async def _handle_item_create(self, event: ConversationItemCreate) -> None:
        item = event.item
        if item.type != "message":
            await self._send_error(
                code="unsupported_item_type",
                message=f"item.type={item.type!r} not supported in M0",
                event_id=event.event_id,
            )
            return

        # M0 only accepts text-only items via this path; audio attachments
        # belong on `input_audio_buffer.*`.
        text_parts: list[str] = []
        for content in item.content or []:
            if content.type in ("input_text", "text") and content.text:
                text_parts.append(content.text)

        item_id = item.id or f"item_{uuid.uuid4().hex}"
        record = _ConversationItem(
            item_id=item_id,
            role=item.role or "user",
            text="\n".join(text_parts),
        )
        self._conversation.append(record)

        await self._send(
            make_event(
                "conversation.item.created",
                previous_item_id=event.previous_item_id,
                item={
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "role": record.role,
                    "content": [
                        {"type": "input_text", "text": record.text}
                    ]
                    if record.text
                    else [],
                },
            )
        )

    # ------------------------------------------------------------------
    # response.create / response.cancel
    # ------------------------------------------------------------------

    async def _handle_response_create(self, event: ResponseCreate) -> None:
        if self._active_task is not None and not self._active_task.done():
            await self._send_error(
                code="response_in_progress",
                message="A response is already in progress",
                event_id=event.event_id,
            )
            return

        modalities = (
            event.response.modalities
            if event.response is not None and event.response.modalities is not None
            else self._session_object.modalities
        )
        if "audio" in (modalities or []):
            await self._send_error(
                code="unsupported_modality",
                message="audio output is not yet implemented (lands in M3)",
                event_id=event.event_id,
            )
            return

        self._active_task = asyncio.create_task(self._run_text_response(event))

    async def _handle_response_cancel(self, event: ResponseCancel) -> None:
        if self._active_task is None or self._active_task.done():
            return
        if self._active_request_id is not None:
            try:
                await self.client.abort(self._active_request_id)
            except Exception:  # noqa: BLE001
                logger.exception("abort failed for %s", self._active_request_id)
        self._active_task.cancel()

    # ------------------------------------------------------------------
    # Pipeline drivers (M0 — fresh OmniRequest per commit / response)
    # ------------------------------------------------------------------

    async def _run_transcription(self, item_id: str, audio_array: Any) -> None:
        """Drive the engine on a freshly-committed audio segment.

        Maps each text token delta to a
        ``conversation.item.input_audio_transcription.delta`` event;
        sends ``.completed`` on finish.
        """
        request_id = f"rt-{self.session_id}-{uuid.uuid4().hex}"
        self._active_request_id = request_id
        gen_req = self._build_transcription_request(audio_array)

        text_acc: list[str] = []
        try:
            async for chunk in self.client.completion_stream(
                gen_req, request_id=request_id
            ):
                if chunk.modality != "text":
                    continue
                if chunk.text:
                    text_acc.append(chunk.text)
                    await self._send(
                        make_event(
                            "conversation.item.input_audio_transcription.delta",
                            item_id=item_id,
                            content_index=0,
                            delta=chunk.text,
                        )
                    )
                if chunk.finish_reason is not None:
                    break

            transcript = "".join(text_acc)
            await self._send(
                make_event(
                    "conversation.item.input_audio_transcription.completed",
                    item_id=item_id,
                    content_index=0,
                    transcript=transcript,
                )
            )
            for entry in self._conversation:
                if entry.item_id == item_id:
                    entry.audio_transcript = transcript
                    break
        except (ClientError, asyncio.CancelledError) as exc:
            await self._send(
                make_event(
                    "conversation.item.input_audio_transcription.failed",
                    item_id=item_id,
                    content_index=0,
                    error={"type": "engine_error", "message": str(exc) or "cancelled"},
                )
            )
        except Exception as exc:
            logger.exception("Transcription failed for %s", request_id)
            await self._send(
                make_event(
                    "conversation.item.input_audio_transcription.failed",
                    item_id=item_id,
                    content_index=0,
                    error={"type": "engine_error", "message": str(exc)},
                )
            )
        finally:
            self._active_request_id = None

    async def _run_text_response(self, event: ResponseCreate) -> None:
        """Drive the engine on accumulated conversation context.

        Emits ``response.created`` → ``response.text.delta`` × N →
        ``response.text.done`` → ``response.done``.
        """
        response_id = f"resp_{uuid.uuid4().hex}"
        self._active_response_id = response_id
        request_id = f"rt-{self.session_id}-{uuid.uuid4().hex}"
        self._active_request_id = request_id

        await self._send(
            make_event(
                "response.created",
                response={
                    "id": response_id,
                    "object": "realtime.response",
                    "status": "in_progress",
                    "output": [],
                },
            )
        )

        gen_req = self._build_text_response_request(event)

        item_id = f"item_{uuid.uuid4().hex}"
        text_acc: list[str] = []
        finish_reason = "stop"
        usage: dict[str, Any] | None = None
        try:
            async for chunk in self.client.completion_stream(
                gen_req, request_id=request_id
            ):
                if chunk.modality == "text" and chunk.text:
                    text_acc.append(chunk.text)
                    await self._send(
                        make_event(
                            "response.text.delta",
                            response_id=response_id,
                            item_id=item_id,
                            output_index=0,
                            content_index=0,
                            delta=chunk.text,
                        )
                    )
                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason
                    if chunk.usage is not None:
                        usage = chunk.usage.to_dict()
                    break

            transcript = "".join(text_acc)
            await self._send(
                make_event(
                    "response.text.done",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    text=transcript,
                )
            )
            self._conversation.append(
                _ConversationItem(item_id=item_id, role="assistant", text=transcript)
            )

            await self._send(
                make_event(
                    "response.done",
                    response={
                        "id": response_id,
                        "object": "realtime.response",
                        "status": "completed",
                        "status_details": {"reason": finish_reason},
                        "output": [
                            {
                                "id": item_id,
                                "object": "realtime.item",
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "text", "text": transcript}],
                            }
                        ],
                        "usage": usage,
                    },
                )
            )
        except asyncio.CancelledError:
            await self._send(
                make_event(
                    "response.done",
                    response={
                        "id": response_id,
                        "object": "realtime.response",
                        "status": "cancelled",
                        "output": [],
                    },
                )
            )
            raise
        except (ClientError, Exception) as exc:
            logger.exception("Response generation failed for %s", request_id)
            await self._send_error(
                code="engine_error",
                message=str(exc),
                event_id=event.event_id,
            )
            await self._send(
                make_event(
                    "response.done",
                    response={
                        "id": response_id,
                        "object": "realtime.response",
                        "status": "failed",
                        "status_details": {"error": str(exc)},
                        "output": [],
                    },
                )
            )
        finally:
            self._active_request_id = None
            self._active_response_id = None

    # ------------------------------------------------------------------
    # GenerateRequest builders
    # ------------------------------------------------------------------

    def _base_sampling(self) -> SamplingParams:
        max_tokens = self._session_object.max_response_output_tokens
        max_new_tokens: int | None
        if isinstance(max_tokens, int):
            max_new_tokens = max_tokens
        else:
            max_new_tokens = None
        return SamplingParams(
            temperature=self._session_object.temperature,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
        )

    def _build_transcription_request(self, audio_array: Any) -> GenerateRequest:
        instructions = self._session_object.instructions or _DEFAULT_INSTRUCTIONS
        # M0: model the transcription as a chat turn whose user content is
        # an audio attachment + the standard transcription instruction.
        # Once a dedicated streaming-ASR mode lands in M2 we can swap this.
        messages = [
            Message(role="system", content=instructions),
            Message(role="user", content="Transcribe the audio."),
        ]
        return GenerateRequest(
            model=self.model_name,
            messages=messages,
            sampling=self._base_sampling(),
            stream=True,
            output_modalities=["text"],
            metadata={"audios": [audio_array]},
        )

    def _build_text_response_request(self, event: ResponseCreate) -> GenerateRequest:
        instructions = self._session_object.instructions or _DEFAULT_INSTRUCTIONS
        if event.response is not None and event.response.instructions:
            instructions = event.response.instructions

        messages: list[Message] = [Message(role="system", content=instructions)]
        for item in self._conversation:
            if item.role == "user":
                content = item.text or item.audio_transcript
                if content:
                    messages.append(Message(role="user", content=content))
            elif item.role == "assistant":
                if item.text:
                    messages.append(Message(role="assistant", content=item.text))

        return GenerateRequest(
            model=self.model_name,
            messages=messages,
            sampling=self._base_sampling(),
            stream=True,
            output_modalities=["text"],
        )

    # ------------------------------------------------------------------
    # WebSocket I/O helpers
    # ------------------------------------------------------------------

    async def _send(self, event: dict[str, Any]) -> None:
        if self._closed:
            return
        if self.websocket.client_state != WebSocketState.CONNECTED:
            return
        if "event_id" not in event:
            event["event_id"] = f"evt_{uuid.uuid4().hex}"
        try:
            await self.websocket.send_text(json.dumps(event))
        except (RuntimeError, WebSocketDisconnect):
            self._closed = True

    async def _send_error(
        self,
        *,
        code: str,
        message: str,
        event_id: str | None = None,
    ) -> None:
        await self._send(
            make_event(
                "error",
                error={
                    "type": "invalid_request_error",
                    "code": code,
                    "message": message,
                    "event_id": event_id,
                },
            )
        )

    def _previous_item_id(self) -> str | None:
        if not self._conversation:
            return None
        return self._conversation[-1].item_id

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    async def _teardown(self) -> None:
        self._closed = True
        if self._active_task is not None and not self._active_task.done():
            if self._active_request_id is not None:
                try:
                    await self.client.abort(self._active_request_id)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "abort during teardown failed for %s", self._active_request_id
                    )
            self._active_task.cancel()
            try:
                await self._active_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        if self.websocket.client_state == WebSocketState.CONNECTED:
            try:
                await self.websocket.close()
            except Exception:  # noqa: BLE001
                pass
