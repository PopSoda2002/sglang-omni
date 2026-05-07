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
    ResponseCancel,
    ResponseCreate,
    SessionObject,
    SessionUpdate,
    SUPPORTED_CLIENT_EVENT_TYPES,
    make_event,
    parse_client_event,
)

logger = logging.getLogger(__name__)


DEFAULT_INSTRUCTIONS = (
    "You are a realtime speech-to-text engine. Transcribe the user's "
    "spoken audio verbatim into the same language they spoke. Output "
    "ONLY the transcript — no descriptions, no refusals, no explanations."
)

HANDLERS: dict[type, str] = {
    SessionUpdate: "handle_session_update",
    InputAudioBufferAppend: "handle_audio_append",
    InputAudioBufferCommit: "handle_audio_commit",
    InputAudioBufferClear: "handle_audio_clear",
    ConversationItemCreate: "handle_item_create",
    ResponseCreate: "handle_response_create",
    ResponseCancel: "handle_response_cancel",
}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def user_audio_item(item_id: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "object": "realtime.item",
        "type": "message",
        "role": "user",
        "content": [{"type": "input_audio", "transcript": None}],
    }


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
        self.session_id = session_id or new_id("sess")

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
        # VAD may emit multiple speech_stopped events while the engine
        # is still busy on an earlier utterance — serialize them.
        self.transcription_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self.queue_drainer: asyncio.Task | None = None

        self.vad: StreamingVAD | None = None
        # Absolute (session-wall-clock) sample offset of buffer byte 0.
        # Advances each commit so speech_started/_stopped keep reporting
        # wall-clock ms after we drop the buffer.
        self.buffer_origin_samples = 0
        self.utterance_start_byte: int | None = None

    async def run(self) -> None:
        """Drive the WebSocket loop.

        Uses raw ASGI ``receive()`` so a disconnect arrives in-band as
        ``"websocket.disconnect"`` rather than as a raised exception.
        """
        await self.send(make_event(
            "session.created",
            session=self.session_object.model_dump(exclude_none=True),
        ))

        while not self.closed:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message["type"] != "websocket.receive":
                continue
            raw = message.get("text")
            if raw is None:
                continue
            payload = json.loads(raw)
            assert isinstance(payload, dict), "Top-level payload must be a JSON object"
            await self.dispatch(payload)

    async def dispatch(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        assert event_type in SUPPORTED_CLIENT_EVENT_TYPES, (
            f"Event type {event_type!r} is not supported"
        )
        event = parse_client_event(payload)
        if event is None:
            return
        method_name = HANDLERS.get(type(event))
        if method_name is not None:
            await getattr(self, method_name)(event)

    async def handle_session_update(self, event: SessionUpdate) -> None:
        # Pydantic merge: client-set fields overwrite, others keep value.
        update = event.session.model_dump(exclude_none=True, exclude_unset=True)
        merged = self.session_object.model_dump() | update
        self.session_object = SessionObject.model_validate(merged)

        assert self.session_object.input_audio_format == "pcm16", (
            f"input_audio_format={self.session_object.input_audio_format!r} "
            "(only pcm16 is supported)"
        )
        assert "audio" not in (self.session_object.modalities or []), (
            "modalities=['audio'] is not yet supported (text-out only)"
        )
        td = self.session_object.turn_detection
        assert not (td and td.type == "semantic_vad"), (
            "turn_detection.type='semantic_vad' is not yet implemented"
        )

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
            self.vad = StreamingVAD(VADConfig(**cfg_kwargs))
            logger.info(
                "Realtime session %s: server_vad enabled (threshold=%s)",
                self.session_id,
                cfg_kwargs.get("threshold", "default"),
            )

        await self.send(make_event(
            "session.updated",
            session=self.session_object.model_dump(exclude_none=True),
        ))

    async def handle_audio_append(self, event: InputAudioBufferAppend) -> None:
        decoded_len, err = self.audio_buffer.append_b64(event.audio)
        assert err != "overflow", (
            f"audio buffer would exceed cap of {self.audio_buffer.max_bytes} bytes"
        )

        if self.vad is None or decoded_len == 0:
            return

        new_bytes = self.audio_buffer.tail(decoded_len)
        emits = await asyncio.to_thread(self.vad.process, new_bytes)
        for emit in emits:
            await self.handle_vad_emit(emit)

    async def handle_vad_emit(self, emit: Any) -> None:
        timestamp_ms = offsets_to_ms(self.buffer_origin_samples + emit.sample_offset)
        if emit.event_type == VADEvent.SPEECH_STARTED:
            # Single-channel PCM16: 2 bytes/sample.
            vad_byte = max(0, emit.sample_offset * 2)
            self.utterance_start_byte = min(vad_byte, self.audio_buffer.num_bytes)
            await self.send(make_event(
                "input_audio_buffer.speech_started",
                audio_start_ms=timestamp_ms,
                item_id=new_id("item"),
            ))
        elif emit.event_type == VADEvent.SPEECH_STOPPED:
            await self.send(make_event(
                "input_audio_buffer.speech_stopped",
                audio_end_ms=timestamp_ms,
                item_id=new_id("item"),
            ))
            await self.auto_commit_utterance(emit.sample_offset)

    def drop_buffer_and_reset_vad(self) -> None:
        """Clear the audio buffer and reset VAD/utterance bookkeeping.

        Used by both auto-commit and manual commit so the next utterance
        always starts from a clean state.
        """
        self.buffer_origin_samples += self.audio_buffer.num_samples
        self.audio_buffer.clear()
        self.utterance_start_byte = None
        if self.vad is not None:
            self.vad.reset()

    async def commit_user_audio_item(self, payload: str) -> None:
        """Send committed → item.created → enqueue for transcription.

        Shared by manual ``input_audio_buffer.commit`` and VAD-driven
        auto-commit. Manual commits go through the same FIFO so they
        serialize cleanly against any in-flight VAD utterance.
        """
        item_id = new_id("item")
        previous = self.previous_item_id
        await self.send(make_event(
            "input_audio_buffer.committed",
            previous_item_id=previous,
            item_id=item_id,
        ))
        await self.send(make_event(
            "conversation.item.created",
            previous_item_id=previous,
            item=user_audio_item(item_id),
        ))
        self.conversation.append(ConversationItem(item_id=item_id, role="user"))

        await self.transcription_queue.put((item_id, payload))
        if self.queue_drainer is None or self.queue_drainer.done():
            self.queue_drainer = asyncio.create_task(self.drain_queue())

    async def auto_commit_utterance(self, end_sample_offset: int) -> None:
        if self.audio_buffer.is_empty():
            return
        start_byte = self.utterance_start_byte or 0
        end_byte = min(end_sample_offset * 2, self.audio_buffer.num_bytes)
        if end_byte <= start_byte:
            return
        payload = self.audio_buffer.slice_to_wav_data_uri(
            start_byte=start_byte, end_byte=end_byte
        )
        if payload is None:
            return
        # Drop the entire buffer (committed speech + silence tail) and
        # advance the absolute origin so future speech_started/_stopped
        # stay wall-clock-correct.
        self.drop_buffer_and_reset_vad()
        await self.commit_user_audio_item(payload)

    async def handle_audio_clear(self, event: InputAudioBufferClear) -> None:
        self.drop_buffer_and_reset_vad()
        await self.send(make_event("input_audio_buffer.cleared"))

    async def handle_audio_commit(self, event: InputAudioBufferCommit) -> None:
        assert not self.audio_buffer.is_empty(), "No audio in buffer to commit"
        payload = self.audio_buffer.to_wav_data_uri()
        self.drop_buffer_and_reset_vad()
        assert payload is not None, "Audio buffer became empty before commit"
        await self.commit_user_audio_item(payload)

    async def handle_item_create(self, event: ConversationItemCreate) -> None:
        item = event.item
        assert item.type == "message", f"item.type={item.type!r} not supported"

        # Audio attachments belong on input_audio_buffer.*; this path is text-only.
        text_parts = [
            c.text for c in (item.content or [])
            if c.type in ("input_text", "text") and c.text
        ]
        item_id = item.id or new_id("item")
        text = "\n".join(text_parts)
        self.conversation.append(ConversationItem(
            item_id=item_id, role=item.role or "user", text=text,
        ))

        await self.send(make_event(
            "conversation.item.created",
            previous_item_id=event.previous_item_id,
            item={
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "role": item.role or "user",
                "content": [{"type": "input_text", "text": text}] if text else [],
            },
        ))

    async def handle_response_create(self, event: ResponseCreate) -> None:
        assert self.active_task is None or self.active_task.done(), (
            "A response is already in progress"
        )

        modalities = (
            event.response.modalities if event.response and event.response.modalities
            else self.session_object.modalities
        )
        assert "audio" not in (modalities or []), "audio output is not yet implemented"

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
        kill the drainer.
        """
        while not self.closed:
            item_id, payload = await self.transcription_queue.get()
            self.active_task = asyncio.create_task(
                self.run_transcription(item_id, payload)
            )
            await asyncio.wait({self.active_task})
            # Retrieve any exception so asyncio doesn't warn at GC.
            self.active_task.exception()
            self.active_task = None

    async def run_transcription(self, item_id: str, audio_payload: str) -> None:
        request_id = f"rt-{self.session_id}-{uuid.uuid4().hex}"
        self.active_request_id = request_id

        text_acc: list[str] = []
        async for chunk in self.client.completion_stream(
            self.build_transcription_request(audio_payload), request_id=request_id,
        ):
            if chunk.modality != "text":
                continue
            if chunk.text:
                text_acc.append(chunk.text)
                await self.send(make_event(
                    "conversation.item.input_audio_transcription.delta",
                    item_id=item_id,
                    content_index=0,
                    delta=chunk.text,
                ))
            if chunk.finish_reason is not None:
                break

        transcript = "".join(text_acc)
        await self.send(make_event(
            "conversation.item.input_audio_transcription.completed",
            item_id=item_id,
            content_index=0,
            transcript=transcript,
        ))
        for entry in self.conversation:
            if entry.item_id == item_id:
                entry.audio_transcript = transcript
                break
        self.active_request_id = None

    async def run_text_response(self, event: ResponseCreate) -> None:
        """Emit response.created → response.text.delta × N → text.done → done.

        Engine errors and cancellation propagate freely; the drainer's
        ``asyncio.wait`` contains them.
        """
        response_id = new_id("resp")
        self.active_response_id = response_id
        request_id = f"rt-{self.session_id}-{uuid.uuid4().hex}"
        self.active_request_id = request_id

        await self.send(make_event(
            "response.created",
            response={
                "id": response_id,
                "object": "realtime.response",
                "status": "in_progress",
                "output": [],
            },
        ))

        item_id = new_id("item")
        text_acc: list[str] = []
        finish_reason = "stop"
        usage: dict[str, Any] | None = None
        async for chunk in self.client.completion_stream(
            self.build_text_response_request(event), request_id=request_id,
        ):
            if chunk.modality == "text" and chunk.text:
                text_acc.append(chunk.text)
                await self.send(make_event(
                    "response.text.delta",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    delta=chunk.text,
                ))
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason
                usage = chunk.usage.to_dict() if chunk.usage else None
                break

        transcript = "".join(text_acc)
        await self.send(make_event(
            "response.text.done",
            response_id=response_id,
            item_id=item_id,
            output_index=0,
            content_index=0,
            text=transcript,
        ))
        self.conversation.append(
            ConversationItem(item_id=item_id, role="assistant", text=transcript)
        )

        await self.send(make_event(
            "response.done",
            response={
                "id": response_id,
                "object": "realtime.response",
                "status": "completed",
                "status_details": {"reason": finish_reason},
                "output": [{
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": transcript}],
                }],
                "usage": usage,
            },
        ))
        self.active_request_id = None
        self.active_response_id = None

    def base_sampling(self) -> SamplingParams:
        max_tokens = self.session_object.max_response_output_tokens
        return SamplingParams(
            temperature=self.session_object.temperature,
            top_p=1.0,
            max_new_tokens=max_tokens if isinstance(max_tokens, int) else None,
        )

    def build_transcription_request(self, audio_payload: str) -> GenerateRequest:
        # Short concrete user message prevents drift into description /
        # refusal mode; the system prompt holds the framing.
        return GenerateRequest(
            model=self.model_name,
            messages=[
                Message(role="system", content=self.session_object.instructions or DEFAULT_INSTRUCTIONS),
                Message(role="user", content="Transcribe the audio verbatim."),
            ],
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
            elif item.role == "assistant" and item.text:
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
        event.setdefault("event_id", new_id("evt"))
        await self.websocket.send_text(json.dumps(event))

    @property
    def previous_item_id(self) -> str | None:
        return self.conversation[-1].item_id if self.conversation else None

    async def teardown(self) -> None:
        """Cancel in-flight tasks and close the WebSocket.

        Pending tasks are cancelled and waited on via ``asyncio.wait``,
        which contains the resulting CancelledError. ``.exception()`` is
        called explicitly so asyncio doesn't warn at GC.
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
