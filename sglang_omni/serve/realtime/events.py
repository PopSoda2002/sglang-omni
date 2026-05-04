# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for OpenAI Realtime WebSocket events.

The full GA / beta protocol has ~9 client events and ~28 server events. M0
implements a usable subset for transcription:

Client (incoming):
    session.update
    input_audio_buffer.append
    input_audio_buffer.commit
    input_audio_buffer.clear
    conversation.item.create   (text-only items in M0)
    response.create            (modalities=["text"] in M0)

Server (outgoing):
    session.created / session.updated
    input_audio_buffer.committed / .cleared
    conversation.item.created
    conversation.item.input_audio_transcription.delta / .completed / .failed
    response.created / .done
    response.text.delta / .done
    response.audio_transcript.delta / .done
    error

Events not yet modeled (M1+) are accepted as opaque dicts on the wire and
rejected with a structured `error` event by the session handler.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Common / shared
# ---------------------------------------------------------------------------


class _Base(BaseModel):
    """Permissive base — Realtime events carry server-version-specific fields
    we don't always model. Allow extras through unchanged."""

    model_config = ConfigDict(extra="allow")


class TurnDetection(_Base):
    type: Literal["server_vad", "semantic_vad", "none"] | None = None
    threshold: float | None = None
    prefix_padding_ms: int | None = None
    silence_duration_ms: int | None = None
    create_response: bool | None = None
    interrupt_response: bool | None = None


class InputAudioTranscription(_Base):
    model: str | None = None
    language: str | None = None
    prompt: str | None = None


class SessionConfig(_Base):
    """Mutable session config. All fields optional to mirror OpenAI's
    `session.update` semantics — only the present fields are applied."""

    modalities: list[str] | None = None
    instructions: str | None = None
    voice: str | None = None
    input_audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"] | None = None
    output_audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"] | None = None
    input_audio_transcription: InputAudioTranscription | None = None
    turn_detection: TurnDetection | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    temperature: float | None = None
    max_response_output_tokens: int | str | None = None


class SessionObject(_Base):
    id: str
    object: Literal["realtime.session"] = "realtime.session"
    model: str
    modalities: list[str] = Field(default_factory=lambda: ["text"])
    instructions: str = ""
    voice: str | None = None
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    input_audio_transcription: InputAudioTranscription | None = None
    turn_detection: TurnDetection | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: Any = "auto"
    temperature: float = 0.8
    max_response_output_tokens: int | str = "inf"


# ---------------------------------------------------------------------------
# Client events
# ---------------------------------------------------------------------------


class ClientEvent(_Base):
    event_id: str | None = None
    type: str


class SessionUpdate(ClientEvent):
    type: Literal["session.update"]
    session: SessionConfig


class InputAudioBufferAppend(ClientEvent):
    type: Literal["input_audio_buffer.append"]
    audio: str  # base64-encoded raw PCM16 (or g711) per session.input_audio_format


class InputAudioBufferCommit(ClientEvent):
    type: Literal["input_audio_buffer.commit"]


class InputAudioBufferClear(ClientEvent):
    type: Literal["input_audio_buffer.clear"]


class ConversationItemContent(_Base):
    type: Literal["input_text", "input_audio", "text", "item_reference"]
    text: str | None = None
    audio: str | None = None  # base64
    transcript: str | None = None
    id: str | None = None


class ConversationItem(_Base):
    id: str | None = None
    type: Literal["message", "function_call", "function_call_output"] = "message"
    role: Literal["user", "assistant", "system"] | None = None
    content: list[ConversationItemContent] | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    output: str | None = None


class ConversationItemCreate(ClientEvent):
    type: Literal["conversation.item.create"]
    previous_item_id: str | None = None
    item: ConversationItem


class ResponseConfig(_Base):
    modalities: list[str] | None = None
    instructions: str | None = None
    voice: str | None = None
    output_audio_format: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    temperature: float | None = None
    max_output_tokens: int | str | None = None
    conversation: Literal["auto", "none"] | None = None
    input: list[ConversationItem] | None = None


class ResponseCreate(ClientEvent):
    type: Literal["response.create"]
    response: ResponseConfig | None = None


class ResponseCancel(ClientEvent):
    type: Literal["response.cancel"]
    response_id: str | None = None


# ---------------------------------------------------------------------------
# Server events  (constructed dict-side; not parsed from wire)
# ---------------------------------------------------------------------------


def make_event(event_type: str, **fields: Any) -> dict[str, Any]:
    """Construct a server event dict suitable for `websocket.send_json`.

    `event_id` is filled in by the session loop so handlers don't have to.
    """
    payload: dict[str, Any] = {"type": event_type}
    for k, v in fields.items():
        if v is None:
            continue
        payload[k] = v
    return payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_client_event(raw: dict[str, Any]) -> ClientEvent | None:
    """Best-effort dispatch of a raw client event dict to a typed model.

    Returns ``None`` if the type is unrecognized; the caller should emit
    a structured `error` event back to the client.
    """
    event_type = raw.get("type")
    if not isinstance(event_type, str):
        return None

    cls = _CLIENT_EVENT_TYPES.get(event_type)
    if cls is None:
        return None
    return cls.model_validate(raw)


_CLIENT_EVENT_TYPES: dict[str, type[ClientEvent]] = {
    "session.update": SessionUpdate,
    "input_audio_buffer.append": InputAudioBufferAppend,
    "input_audio_buffer.commit": InputAudioBufferCommit,
    "input_audio_buffer.clear": InputAudioBufferClear,
    "conversation.item.create": ConversationItemCreate,
    "response.create": ResponseCreate,
    "response.cancel": ResponseCancel,
}


SUPPORTED_CLIENT_EVENT_TYPES = frozenset(_CLIENT_EVENT_TYPES.keys())
