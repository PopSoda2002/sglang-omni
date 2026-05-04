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
    SessionConfig,
    SessionObject,
    SessionUpdate,
    SUPPORTED_CLIENT_EVENT_TYPES,
    make_event,
    parse_client_event,
)

logger = logging.getLogger(__name__)


_DEFAULT_INSTRUCTIONS = (
    "You are a realtime speech-to-text engine. Transcribe the user's "
    "spoken audio verbatim into the same language they spoke. Output "
    "ONLY the transcript — no descriptions, no refusals, no explanations."
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
        # FIFO queue of pending transcription jobs (item_id, audio_payload).
        # Server VAD can fire multiple speech_stopped events while the
        # engine is busy on an earlier utterance; we serialize them
        # rather than drop.
        self._transcription_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._queue_drainer: asyncio.Task | None = None

        # Server VAD (M2). None ⇒ manual mode (client must commit explicitly).
        self._vad: StreamingVAD | None = None
        # Sample offset of the start of the *current* audio buffer in the
        # session's wall-clock sample timeline; used to convert VAD-relative
        # offsets back into the absolute timeline reported in events.
        self._buffer_origin_samples = 0
        # Per-utterance state: when speech_started fires we snapshot the
        # buffer's current size so that on speech_stopped we slice out
        # only the speech segment (with prefix padding) for transcription.
        self._utterance_start_byte: int | None = None

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
        # Apply only the fields that were actually set. ``exclude_none``
        # guarantees we don't clobber existing values; ``model_dump``
        # would otherwise serialize ``None``s for unset fields.
        update = cfg.model_dump(exclude_none=True, exclude_unset=True)
        for key, value in update.items():
            if not hasattr(self._session_object, key):
                continue
            # Re-validate dict values into the typed field so downstream
            # code can rely on attribute access (e.g. turn_detection.type).
            current = getattr(self._session_object, key)
            if hasattr(current, "model_validate") and isinstance(value, dict):
                value = type(current).model_validate(value)
            elif (
                key == "turn_detection"
                and isinstance(value, dict)
                and current is None
            ):
                from sglang_omni.serve.realtime.events import TurnDetection

                value = TurnDetection.model_validate(value)
            elif (
                key == "input_audio_transcription"
                and isinstance(value, dict)
                and current is None
            ):
                from sglang_omni.serve.realtime.events import (
                    InputAudioTranscription,
                )

                value = InputAudioTranscription.model_validate(value)
            setattr(self._session_object, key, value)

        # Currently supported: modalities=["text"] + pcm16 input, plus
        # server_vad turn detection. Audio output (modalities=["audio"])
        # and semantic_vad land in later milestones; reject those with
        # a structured error rather than silently accepting.
        unsupported = []
        if self._session_object.input_audio_format != "pcm16":
            unsupported.append(
                f"input_audio_format={self._session_object.input_audio_format!r} "
                "(only pcm16 is supported)"
            )
        if "audio" in (self._session_object.modalities or []):
            unsupported.append(
                "modalities=['audio'] is not yet supported (text-out only)"
            )
        td = self._session_object.turn_detection
        if td and td.type == "semantic_vad":
            unsupported.append(
                "turn_detection.type='semantic_vad' is not yet implemented"
            )

        if unsupported:
            await self._send_error(
                code="unsupported_session_config",
                message="; ".join(unsupported),
                event_id=event.event_id,
            )
            return

        # (Re)configure server VAD if requested.
        if td is None or td.type in (None, "none"):
            self._vad = None
        elif td.type == "server_vad":
            cfg_kwargs: dict[str, Any] = {}
            if td.threshold is not None:
                cfg_kwargs["threshold"] = float(td.threshold)
            if td.prefix_padding_ms is not None:
                cfg_kwargs["prefix_padding_ms"] = int(td.prefix_padding_ms)
            if td.silence_duration_ms is not None:
                cfg_kwargs["silence_duration_ms"] = int(td.silence_duration_ms)
            self._vad = StreamingVAD(VADConfig(**cfg_kwargs))
            logger.info(
                "Realtime session %s: server_vad enabled (threshold=%s)",
                self.session_id,
                cfg_kwargs.get("threshold", "default"),
            )

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
            decoded_len = self._audio_buffer.append_b64(event.audio)
        except AudioBufferError as exc:
            await self._send_error(
                code="invalid_request_error",
                message=str(exc),
                event_id=event.event_id,
            )
            return

        if self._vad is None or decoded_len == 0:
            return

        # Pull the bytes we just appended back out of the buffer for VAD.
        # `append_b64` told us how many bytes were appended; the buffer
        # implementation is append-only so the new bytes are at the tail.
        new_bytes = self._audio_buffer.tail(decoded_len)
        try:
            emits = await asyncio.to_thread(self._vad.process, new_bytes)
        except Exception:  # noqa: BLE001
            logger.exception("VAD inference failed")
            return

        for emit in emits:
            await self._handle_vad_emit(emit)

    async def _handle_vad_emit(self, emit: Any) -> None:
        absolute_samples = self._buffer_origin_samples + emit.sample_offset
        timestamp_ms = offsets_to_ms(absolute_samples)
        if emit.type == VADEvent.SPEECH_STARTED:
            # Map the VAD's sample offset (relative to bytes consumed by
            # the VAD, which equals bytes appended) to a buffer byte
            # position. With single-channel PCM16 that's offset * 2.
            vad_byte = max(0, emit.sample_offset * 2)
            buffer_byte = min(vad_byte, self._audio_buffer.num_bytes)
            self._utterance_start_byte = buffer_byte
            await self._send(
                make_event(
                    "input_audio_buffer.speech_started",
                    audio_start_ms=timestamp_ms,
                    item_id=f"item_{uuid.uuid4().hex}",
                )
            )
        elif emit.type == VADEvent.SPEECH_STOPPED:
            await self._send(
                make_event(
                    "input_audio_buffer.speech_stopped",
                    audio_end_ms=timestamp_ms,
                    item_id=f"item_{uuid.uuid4().hex}",
                )
            )
            await self._auto_commit_utterance(emit.sample_offset)

    async def _auto_commit_utterance(self, end_sample_offset: int) -> None:
        """Slice [start_byte:end_byte] out of the buffer and dispatch transcription."""
        if self._audio_buffer.is_empty():
            return

        start_byte = self._utterance_start_byte or 0
        end_byte = min(end_sample_offset * 2, self._audio_buffer.num_bytes)
        if end_byte <= start_byte:
            return

        # Build a one-shot audio buffer for the utterance and discard the
        # rest — anything *after* end_byte is post-utterance silence we
        # don't want to send to the engine.
        utterance_payload = self._audio_buffer.slice_to_wav_data_uri(
            start_byte=start_byte, end_byte=end_byte
        )
        if utterance_payload is None:
            return

        # Advance the absolute-sample origin by every sample we just dropped
        # (committed speech *plus* the silence tail), so subsequent
        # speech_started/_stopped events keep reporting wall-clock offsets.
        dropped_samples = self._audio_buffer.num_samples
        self._buffer_origin_samples += dropped_samples
        self._audio_buffer.clear()
        self._utterance_start_byte = None
        if self._vad is not None:
            # The VAD's internal sample counter is keyed to bytes-fed; since
            # we cleared the buffer, also reset its counter so future
            # transitions report offsets relative to the new utterance.
            self._vad.reset()

        item_id = f"item_{uuid.uuid4().hex}"
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

        # FIFO-queue the utterance and ensure the drainer task is running.
        await self._transcription_queue.put((item_id, utterance_payload))
        if self._queue_drainer is None or self._queue_drainer.done():
            self._queue_drainer = asyncio.create_task(self._drain_queue())

    async def _handle_audio_clear(self, event: InputAudioBufferClear) -> None:
        self._audio_buffer.clear()
        if self._vad is not None:
            self._vad.reset()
        self._utterance_start_byte = None
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
        # The pipeline IPC layer (msgpack) cannot transport numpy arrays.
        # Encode the buffer as an in-memory WAV data URI; the preprocessor
        # stage's resource connector decodes it back to float32.
        audio_payload = self._audio_buffer.to_wav_data_uri()
        self._audio_buffer.clear()
        if audio_payload is None:
            await self._send_error(
                code="input_audio_buffer_commit_empty",
                message="Audio buffer became empty before commit",
                event_id=event.event_id,
            )
            return

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

        # FIFO-queue the manual commit too. The drainer serializes
        # against any in-flight utterance from server VAD.
        await self._transcription_queue.put((item_id, audio_payload))
        if self._queue_drainer is None or self._queue_drainer.done():
            self._queue_drainer = asyncio.create_task(self._drain_queue())

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

    async def _drain_queue(self) -> None:
        """Pop utterances and run them serially through ``_run_transcription``."""
        try:
            while not self._closed:
                try:
                    item_id, payload = await asyncio.wait_for(
                        self._transcription_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    if self._transcription_queue.empty():
                        return
                    continue

                self._active_task = asyncio.create_task(
                    self._run_transcription(item_id, payload)
                )
                try:
                    await self._active_task
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    logger.exception("transcription task failed")
                finally:
                    self._active_task = None
        except asyncio.CancelledError:
            return

    async def _run_transcription(self, item_id: str, audio_payload: str) -> None:
        """Drive the engine on a freshly-committed audio segment.

        ``audio_payload`` is a ``data:audio/wav;base64,...`` URI; the
        preprocessor stage's resource connector decodes it back to
        float32. We can't pass a raw numpy array because the pipeline
        IPC layer serializes via msgpack.
        """
        request_id = f"rt-{self.session_id}-{uuid.uuid4().hex}"
        self._active_request_id = request_id
        gen_req = self._build_transcription_request(audio_payload)

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

    def _build_transcription_request(self, audio_payload: str) -> GenerateRequest:
        instructions = self._session_object.instructions or _DEFAULT_INSTRUCTIONS
        # The system prompt carries the "be a transcription engine"
        # framing; the user message is short and concrete to keep the
        # model from drifting into description / refusal mode.
        messages = [
            Message(role="system", content=instructions),
            Message(role="user", content="Transcribe the audio verbatim."),
        ]
        return GenerateRequest(
            model=self.model_name,
            messages=messages,
            sampling=self._base_sampling(),
            stream=True,
            output_modalities=["text"],
            metadata={"audios": [audio_payload]},
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

        if self._queue_drainer is not None and not self._queue_drainer.done():
            self._queue_drainer.cancel()
            try:
                await self._queue_drainer
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        if self.websocket.client_state == WebSocketState.CONNECTED:
            try:
                await self.websocket.close()
            except Exception:  # noqa: BLE001
                pass
