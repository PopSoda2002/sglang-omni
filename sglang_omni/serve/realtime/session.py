# SPDX-License-Identifier: Apache-2.0
"""Per-WebSocket Realtime session.

A ``RealtimeSession`` wraps one ``WebSocket`` connection and one
``Client`` (the engine front door). It dispatches incoming OpenAI
Realtime client events, accumulates audio into a ``RealtimeAudioBuffer``,
and on each commit (manual or server-VAD-driven) builds a fresh
``GenerateRequest`` and streams the resulting deltas back as Realtime
server events.

Errors from the framework, parser, or engine propagate freely. The only
"error handling" in this module is API design — functions return
sentinel/optional values where it makes sense (e.g. ``parse_client_event``
returns ``None`` for unrecognized event types).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketState

# Starlette's receive() returns ASGI messages directly; we handle the
# disconnect message in-band instead of catching the WebSocketDisconnect
# exception that receive_text() would otherwise raise.

from sglang_omni.client import (
    Client,
    GenerateRequest,
    Message,
    SamplingParams,
)
from sglang_omni.serve.realtime.audio_buffer import RealtimeAudioBuffer
from sglang_omni.serve.realtime.vad import (
    StreamingVAD,
    VADConfig,
    VADEvent,
    offsets_to_ms,
)
from sglang_omni.serve.realtime.events import (
    ConversationItemCreate,
    InputAudioBufferAppend,
    InputAudioBufferClear,
    InputAudioBufferCommit,
    InputAudioTranscription,
    ResponseCancel,
    ResponseCreate,
    SessionConfig,
    SessionObject,
    SessionUpdate,
    SUPPORTED_CLIENT_EVENT_TYPES,
    TurnDetection,
    make_event,
    parse_client_event,
)

logger = logging.getLogger(__name__)


DEFAULT_INSTRUCTIONS = (
    "You are a realtime speech-to-text engine. Transcribe the user's "
    "spoken audio verbatim into the same language they spoke. Output "
    "ONLY the transcript — no descriptions, no refusals, no explanations."
)


@dataclass
class ConversationItem:
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

        self.session_object = SessionObject(
            id=self.session_id,
            model=model_name,
            modalities=["text"],
            instructions=DEFAULT_INSTRUCTIONS,
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            turn_detection=None,
        )

        self.audio_buffer = RealtimeAudioBuffer(source_sr=16000, target_sr=16000)
        self.conversation: list[ConversationItem] = []
        self.closed = False

        self.active_request_id: str | None = None
        self.active_response_id: str | None = None
        self.active_task: asyncio.Task | None = None
        # Server VAD can fire multiple speech_stopped events while the
        # engine is busy on an earlier utterance — serialize them rather
        # than drop.
        self.transcription_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self.queue_drainer: asyncio.Task | None = None

        self.vad: StreamingVAD | None = None
        # Absolute (session-wall-clock) sample offset of buffer byte 0.
        # Advances each time we drop the buffer on commit so that
        # speech_started/_stopped events keep reporting wall-clock ms.
        self.buffer_origin_samples = 0
        self.utterance_start_byte: int | None = None

    async def run(self) -> None:
        """Drive the WebSocket loop.

        Uses the raw ASGI ``receive()`` instead of ``receive_text()`` so
        a disconnect arrives as an in-band message (``"websocket.disconnect"``)
        rather than a raised exception. The loop ends when the client
        disconnects, when ``self.closed`` is set (e.g. on input overflow),
        or naturally if the route is cancelled.
        """
        await self.send(
            make_event(
                "session.created",
                session=self.session_object.model_dump(exclude_none=True),
            )
        )

        while not self.closed:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message["type"] != "websocket.receive":
                continue
            raw = message.get("text")
            if raw is None:
                # Binary frames aren't part of the OpenAI Realtime wire
                # format; ignore them.
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                await self.send_error(
                    code="invalid_request_error",
                    message="Top-level payload must be a JSON object",
                )
                continue
            await self.dispatch(payload)

    async def dispatch(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        if event_type not in SUPPORTED_CLIENT_EVENT_TYPES:
            await self.send_error(
                code="unsupported_event",
                message=f"Event type {event_type!r} is not supported",
                event_id=payload.get("event_id"),
            )
            return

        event = parse_client_event(payload)
        if event is None:
            # Unrecognized type already filtered above; this shouldn't
            # happen unless the supported set drifts from the registry.
            return

        if isinstance(event, SessionUpdate):
            await self.handle_session_update(event)
        elif isinstance(event, InputAudioBufferAppend):
            await self.handle_audio_append(event)
        elif isinstance(event, InputAudioBufferCommit):
            await self.handle_audio_commit(event)
        elif isinstance(event, InputAudioBufferClear):
            await self.handle_audio_clear(event)
        elif isinstance(event, ConversationItemCreate):
            await self.handle_item_create(event)
        elif isinstance(event, ResponseCreate):
            await self.handle_response_create(event)
        elif isinstance(event, ResponseCancel):
            await self.handle_response_cancel(event)

    async def handle_session_update(self, event: SessionUpdate) -> None:
        cfg: SessionConfig = event.session
        update = cfg.model_dump(exclude_none=True, exclude_unset=True)
        for key, value in update.items():
            if not hasattr(self.session_object, key):
                continue
            # Re-validate dict values into the typed field so downstream
            # code can rely on attribute access (e.g. turn_detection.type).
            current = getattr(self.session_object, key)
            if hasattr(current, "model_validate") and isinstance(value, dict):
                value = type(current).model_validate(value)
            elif (
                key == "turn_detection"
                and isinstance(value, dict)
                and current is None
            ):
                value = TurnDetection.model_validate(value)
            elif (
                key == "input_audio_transcription"
                and isinstance(value, dict)
                and current is None
            ):
                value = InputAudioTranscription.model_validate(value)
            setattr(self.session_object, key, value)

        # Reject unsupported configs with a structured error rather than
        # silently accepting and behaving oddly downstream.
        unsupported = []
        if self.session_object.input_audio_format != "pcm16":
            unsupported.append(
                f"input_audio_format={self.session_object.input_audio_format!r} "
                "(only pcm16 is supported)"
            )
        if "audio" in (self.session_object.modalities or []):
            unsupported.append(
                "modalities=['audio'] is not yet supported (text-out only)"
            )
        td = self.session_object.turn_detection
        if td and td.type == "semantic_vad":
            unsupported.append(
                "turn_detection.type='semantic_vad' is not yet implemented"
            )

        if unsupported:
            await self.send_error(
                code="unsupported_session_config",
                message="; ".join(unsupported),
                event_id=event.event_id,
            )
            return

        if td is None or td.type in (None, "none"):
            self.vad = None
        elif td.type == "server_vad":
            cfg_kwargs: dict[str, Any] = {}
            if td.threshold is not None:
                cfg_kwargs["threshold"] = float(td.threshold)
            if td.prefix_padding_ms is not None:
                cfg_kwargs["prefix_padding_ms"] = int(td.prefix_padding_ms)
            if td.silence_duration_ms is not None:
                cfg_kwargs["silence_duration_ms"] = int(td.silence_duration_ms)
            # If silero-vad isn't installed or fails to load, the
            # exception propagates to the caller. The route's finally
            # clause still runs cleanup.
            self.vad = StreamingVAD(VADConfig(**cfg_kwargs))
            logger.info(
                "Realtime session %s: server_vad enabled (threshold=%s)",
                self.session_id,
                cfg_kwargs.get("threshold", "default"),
            )

        await self.send(
            make_event(
                "session.updated",
                session=self.session_object.model_dump(exclude_none=True),
            )
        )

    async def handle_audio_append(self, event: InputAudioBufferAppend) -> None:
        decoded_len, err = self.audio_buffer.append_b64(event.audio)
        if err == "overflow":
            # Hard close: a client growing past the cap is malicious or
            # buggy. Surface a structured error and tear down the WS.
            await self.send_error(
                code="input_audio_buffer_too_large",
                message=(
                    f"audio buffer would exceed cap of "
                    f"{self.audio_buffer.max_bytes} bytes"
                ),
                event_id=event.event_id,
            )
            self.closed = True
            await self.websocket.close(code=1009)  # 1009 = "message too big"
            return

        if self.vad is None or decoded_len == 0:
            return

        new_bytes = self.audio_buffer.tail(decoded_len)
        emits = await asyncio.to_thread(self.vad.process, new_bytes)
        for emit in emits:
            await self.handle_vad_emit(emit)

    async def handle_vad_emit(self, emit: Any) -> None:
        absolute_samples = self.buffer_origin_samples + emit.sample_offset
        timestamp_ms = offsets_to_ms(absolute_samples)
        if emit.type == VADEvent.SPEECH_STARTED:
            # Single-channel PCM16: 2 bytes/sample.
            vad_byte = max(0, emit.sample_offset * 2)
            buffer_byte = min(vad_byte, self.audio_buffer.num_bytes)
            self.utterance_start_byte = buffer_byte
            await self.send(
                make_event(
                    "input_audio_buffer.speech_started",
                    audio_start_ms=timestamp_ms,
                    item_id=f"item_{uuid.uuid4().hex}",
                )
            )
        elif emit.type == VADEvent.SPEECH_STOPPED:
            await self.send(
                make_event(
                    "input_audio_buffer.speech_stopped",
                    audio_end_ms=timestamp_ms,
                    item_id=f"item_{uuid.uuid4().hex}",
                )
            )
            await self.auto_commit_utterance(emit.sample_offset)

    def drop_buffer_and_reset_vad(self) -> None:
        """Clear the audio buffer and reset VAD/utterance bookkeeping.

        Used by both VAD-driven auto-commit and explicit manual commit so
        the next utterance always starts from a clean state.
        """
        dropped_samples = self.audio_buffer.num_samples
        self.buffer_origin_samples += dropped_samples
        self.audio_buffer.clear()
        self.utterance_start_byte = None
        if self.vad is not None:
            self.vad.reset()

    async def auto_commit_utterance(self, end_sample_offset: int) -> None:
        if self.audio_buffer.is_empty():
            return

        start_byte = self.utterance_start_byte or 0
        end_byte = min(end_sample_offset * 2, self.audio_buffer.num_bytes)
        if end_byte <= start_byte:
            return

        utterance_payload = self.audio_buffer.slice_to_wav_data_uri(
            start_byte=start_byte, end_byte=end_byte
        )
        if utterance_payload is None:
            return

        # Drop the entire buffer (committed speech + silence tail) and
        # advance the absolute origin by what we dropped — keeps future
        # speech_started/_stopped wall-clock-correct.
        self.drop_buffer_and_reset_vad()

        item_id = f"item_{uuid.uuid4().hex}"
        await self.send(
            make_event(
                "input_audio_buffer.committed",
                previous_item_id=self.previous_item_id(),
                item_id=item_id,
            )
        )
        await self.send(
            make_event(
                "conversation.item.created",
                previous_item_id=self.previous_item_id(),
                item={
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio", "transcript": None}],
                },
            )
        )
        self.conversation.append(ConversationItem(item_id=item_id, role="user"))

        await self.transcription_queue.put((item_id, utterance_payload))
        if self.queue_drainer is None or self.queue_drainer.done():
            self.queue_drainer = asyncio.create_task(self.drain_queue())

    async def handle_audio_clear(self, event: InputAudioBufferClear) -> None:
        self.drop_buffer_and_reset_vad()
        await self.send(make_event("input_audio_buffer.cleared"))

    async def handle_audio_commit(self, event: InputAudioBufferCommit) -> None:
        if self.audio_buffer.is_empty():
            await self.send_error(
                code="input_audio_buffer_commit_empty",
                message="No audio in buffer to commit",
                event_id=event.event_id,
            )
            return

        item_id = f"item_{uuid.uuid4().hex}"
        audio_payload = self.audio_buffer.to_wav_data_uri()
        # Manual commit performs the same VAD/origin reset as auto-commit
        # so a client that mixes manual commits with active server_vad
        # doesn't carry stale LSTM state into the next utterance.
        self.drop_buffer_and_reset_vad()
        if audio_payload is None:
            await self.send_error(
                code="input_audio_buffer_commit_empty",
                message="Audio buffer became empty before commit",
                event_id=event.event_id,
            )
            return

        # Order matches OpenAI: committed → conversation.item.created →
        # transcription deltas/completed (emitted by run_transcription).
        await self.send(
            make_event(
                "input_audio_buffer.committed",
                previous_item_id=self.previous_item_id(),
                item_id=item_id,
            )
        )
        await self.send(
            make_event(
                "conversation.item.created",
                previous_item_id=self.previous_item_id(),
                item={
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio", "transcript": None}],
                },
            )
        )
        self.conversation.append(ConversationItem(item_id=item_id, role="user"))

        # Manual commits go through the same FIFO queue so they serialize
        # cleanly against any in-flight VAD-driven utterance.
        await self.transcription_queue.put((item_id, audio_payload))
        if self.queue_drainer is None or self.queue_drainer.done():
            self.queue_drainer = asyncio.create_task(self.drain_queue())

    async def handle_item_create(self, event: ConversationItemCreate) -> None:
        item = event.item
        if item.type != "message":
            await self.send_error(
                code="unsupported_item_type",
                message=f"item.type={item.type!r} not supported",
                event_id=event.event_id,
            )
            return

        # Audio attachments belong on input_audio_buffer.*; this path is
        # text-only.
        text_parts: list[str] = []
        for content in item.content or []:
            if content.type in ("input_text", "text") and content.text:
                text_parts.append(content.text)

        item_id = item.id or f"item_{uuid.uuid4().hex}"
        record = ConversationItem(
            item_id=item_id,
            role=item.role or "user",
            text="\n".join(text_parts),
        )
        self.conversation.append(record)

        await self.send(
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

    async def handle_response_create(self, event: ResponseCreate) -> None:
        if self.active_task is not None and not self.active_task.done():
            await self.send_error(
                code="response_in_progress",
                message="A response is already in progress",
                event_id=event.event_id,
            )
            return

        modalities = (
            event.response.modalities
            if event.response is not None and event.response.modalities is not None
            else self.session_object.modalities
        )
        if "audio" in (modalities or []):
            await self.send_error(
                code="unsupported_modality",
                message="audio output is not yet implemented",
                event_id=event.event_id,
            )
            return

        self.active_task = asyncio.create_task(self.run_text_response(event))

    async def handle_response_cancel(self, event: ResponseCancel) -> None:
        if self.active_task is None or self.active_task.done():
            return
        if self.active_request_id is not None:
            await self.client.abort(self.active_request_id)
        self.active_task.cancel()

    async def drain_queue(self) -> None:
        """Pop utterances and run them serially through ``run_transcription``.

        ``asyncio.wait`` waits without re-raising the inner task's
        exception or cancellation, so a single failing utterance doesn't
        kill the drainer — the task's outcome is contained.
        """
        while not self.closed:
            item_id, payload = await self.transcription_queue.get()
            self.active_task = asyncio.create_task(
                self.run_transcription(item_id, payload)
            )
            await asyncio.wait({self.active_task})
            # Retrieve any exception/result so asyncio doesn't warn at GC.
            self.active_task.exception()
            self.active_task = None

    async def run_transcription(self, item_id: str, audio_payload: str) -> None:
        request_id = f"rt-{self.session_id}-{uuid.uuid4().hex}"
        self.active_request_id = request_id
        gen_req = self.build_transcription_request(audio_payload)

        text_acc: list[str] = []
        async for chunk in self.client.completion_stream(
            gen_req, request_id=request_id
        ):
            if chunk.modality != "text":
                continue
            if chunk.text:
                text_acc.append(chunk.text)
                await self.send(
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
        await self.send(
            make_event(
                "conversation.item.input_audio_transcription.completed",
                item_id=item_id,
                content_index=0,
                transcript=transcript,
            )
        )
        for entry in self.conversation:
            if entry.item_id == item_id:
                entry.audio_transcript = transcript
                break
        self.active_request_id = None

    async def run_text_response(self, event: ResponseCreate) -> None:
        """Emit response.created → response.text.delta × N → text.done → done.

        Engine errors and cancellation propagate freely; the drainer's
        ``asyncio.wait`` contains them. Partial events on the wire are
        the user's accepted tradeoff for not handling errors.
        """
        response_id = f"resp_{uuid.uuid4().hex}"
        self.active_response_id = response_id
        request_id = f"rt-{self.session_id}-{uuid.uuid4().hex}"
        self.active_request_id = request_id

        await self.send(
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

        gen_req = self.build_text_response_request(event)

        item_id = f"item_{uuid.uuid4().hex}"
        text_acc: list[str] = []
        finish_reason = "stop"
        usage: dict[str, Any] | None = None
        async for chunk in self.client.completion_stream(
            gen_req, request_id=request_id
        ):
            if chunk.modality == "text" and chunk.text:
                text_acc.append(chunk.text)
                await self.send(
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
        await self.send(
            make_event(
                "response.text.done",
                response_id=response_id,
                item_id=item_id,
                output_index=0,
                content_index=0,
                text=transcript,
            )
        )
        self.conversation.append(
            ConversationItem(item_id=item_id, role="assistant", text=transcript)
        )

        await self.send(
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
        self.active_request_id = None
        self.active_response_id = None

    def base_sampling(self) -> SamplingParams:
        max_tokens = self.session_object.max_response_output_tokens
        max_new_tokens = max_tokens if isinstance(max_tokens, int) else None
        return SamplingParams(
            temperature=self.session_object.temperature,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
        )

    def build_transcription_request(self, audio_payload: str) -> GenerateRequest:
        instructions = self.session_object.instructions or DEFAULT_INSTRUCTIONS
        # Short, concrete user message prevents drift into description /
        # refusal mode; the system prompt holds the transcription framing.
        messages = [
            Message(role="system", content=instructions),
            Message(role="user", content="Transcribe the audio verbatim."),
        ]
        return GenerateRequest(
            model=self.model_name,
            messages=messages,
            sampling=self.base_sampling(),
            stream=True,
            output_modalities=["text"],
            metadata={"audios": [audio_payload]},
        )

    def build_text_response_request(self, event: ResponseCreate) -> GenerateRequest:
        instructions = self.session_object.instructions or DEFAULT_INSTRUCTIONS
        if event.response is not None and event.response.instructions:
            instructions = event.response.instructions

        messages: list[Message] = [Message(role="system", content=instructions)]
        for item in self.conversation:
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
            sampling=self.base_sampling(),
            stream=True,
            output_modalities=["text"],
        )

    async def send(self, event: dict[str, Any]) -> None:
        if self.closed:
            return
        if self.websocket.client_state != WebSocketState.CONNECTED:
            return
        if "event_id" not in event:
            event["event_id"] = f"evt_{uuid.uuid4().hex}"
        await self.websocket.send_text(json.dumps(event))

    async def send_error(
        self,
        *,
        code: str,
        message: str,
        event_id: str | None = None,
    ) -> None:
        await self.send(
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

    def previous_item_id(self) -> str | None:
        if not self.conversation:
            return None
        return self.conversation[-1].item_id

    async def teardown(self) -> None:
        """Cancel in-flight tasks and close the WebSocket.

        Each pending task is cancelled and waited on via ``asyncio.wait``
        which contains the resulting CancelledError. Side-task results
        are explicitly retrieved so asyncio doesn't warn at GC.
        """
        self.closed = True
        if self.active_task is not None and not self.active_task.done():
            if self.active_request_id is not None:
                await self.client.abort(self.active_request_id)
            self.active_task.cancel()
            await asyncio.wait({self.active_task})
            self.active_task.exception()

        if self.queue_drainer is not None and not self.queue_drainer.done():
            self.queue_drainer.cancel()
            await asyncio.wait({self.queue_drainer})
            self.queue_drainer.exception()

        if self.websocket.client_state == WebSocketState.CONNECTED:
            await self.websocket.close()
