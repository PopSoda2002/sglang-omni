from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState
from transformers import AutoTokenizer

from sglang_omni.serve.realtime.events import SessionObject, make_event
from sglang_omni.serve.realtime.frame_session_ref import (
    SAMPLES_PER_FRAME,
    DuplexEngines,
    FrameSession,
)
from sglang_omni.serve.realtime.session import new_id

logger = logging.getLogger(__name__)

class DuplexRealtimeSession:
    def __init__(self, websocket: WebSocket, *, frame_session, tokenizer,
                 model_name: str, session_id: str | None = None) -> None:
        self.websocket = websocket
        self.frame = frame_session
        self.tokenizer = tokenizer
        self.session_id = session_id or new_id("sess")
        self.session_object = SessionObject(
            id=self.session_id,
            model=model_name,
            modalities=["text", "audio"],
            input_audio_format="pcm16",
            output_audio_format="pcm16",
        )
        self.response_id = new_id("resp")
        self._residual = torch.zeros(0)
        self._emitted_tokens = 0
        self.closed = False

    async def run(self) -> None:
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
            payload = json.loads(message["text"])
            assert isinstance(payload, dict)
            await self.dispatch(payload)

    async def dispatch(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        if event_type == "input_audio_buffer.append":
            await self.handle_audio_append(payload["audio"])
        elif event_type == "session.update":
            await self.send(make_event(
                "session.updated",
                session=self.session_object.model_dump(exclude_none=True),
            ))
        elif event_type == "input_audio_buffer.clear":
            self._residual = torch.zeros(0)
            await self.send(make_event("input_audio_buffer.cleared"))
        else:
            logger.debug("Duplex session ignoring event type %s", event_type)

    async def handle_audio_append(self, audio_b64: str) -> None:
        pcm = torch.frombuffer(
            bytearray(base64.b64decode(audio_b64)), dtype=torch.int16
        ).float() / 32768.0
        self._residual = torch.cat([self._residual, pcm])
        while self._residual.numel() >= SAMPLES_PER_FRAME:
            block = self._residual[:SAMPLES_PER_FRAME]
            self._residual = self._residual[SAMPLES_PER_FRAME:]
            out = await asyncio.to_thread(self.frame.push_audio, block)
            await self._emit_outputs(out)

    async def _emit_outputs(self, audio: torch.Tensor) -> None:
        if self.frame.text_ids and len(self.frame.text_ids) > self._emitted_tokens:
            fresh = self.frame.text_ids[self._emitted_tokens:]
            self._emitted_tokens = len(self.frame.text_ids)
            text = self.tokenizer.decode(
                [t for t in fresh if t not in (0, 1, self.tokenizer.eos_token_id)],
                skip_special_tokens=True,
            )
            if text:
                await self.send(make_event(
                    "response.audio_transcript.delta",
                    response_id=self.response_id,
                    delta=text,
                ))
        if audio.numel():
            pcm16 = (audio.clamp(-1, 1) * 32767).to(torch.int16)
            await self.send(make_event(
                "response.audio.delta",
                response_id=self.response_id,
                delta=base64.b64encode(pcm16.numpy().tobytes()).decode("ascii"),
            ))

    async def send(self, event: dict[str, Any]) -> None:
        if self.closed:
            return
        if self.websocket.application_state != WebSocketState.CONNECTED:
            return
        event.setdefault("event_id", new_id("evt"))
        await self.websocket.send_text(json.dumps(event))

    async def teardown(self) -> None:
        self.closed = True
        self.frame.close()


class DuplexRealtimeSessionManager:

    def __init__(self, *, engines, model_name: str) -> None:
        self.engines = engines
        self.model_name = model_name
        speech = json.loads(
            (Path(engines.model_path) / "config.json").read_text()
        )["model"]["speech_generation"]["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            speech["tts_config"]["cas_config"]["pretrained_tokenizer_name"]
        )
        self.sessions: dict[str, DuplexRealtimeSession] = {}

    def open(self, websocket: WebSocket) -> DuplexRealtimeSession:
        session_id = new_id("sess")
        session = DuplexRealtimeSession(
            websocket,
            frame_session=FrameSession(self.engines, session_id=session_id),
            tokenizer=self.tokenizer,
            model_name=self.model_name,
            session_id=session_id,
        )
        self.sessions[session.session_id] = session
        logger.info(f"Duplex realtime session opened: {session.session_id}")
        return session

    async def close(self, session_id: str) -> None:
        session = self.sessions.pop(session_id)
        await session.teardown()
        logger.info(f"Duplex realtime session closed: {session_id}")


def create_duplex_realtime_app(model_path: str, *, thinker_gpu_id: int,
                               talker_gpu_id: int, model_name: str = "nemotron-voicechat",
                               max_frames: int = 3_750):
    """Standalone FastAPI app mounting /v1/realtime over shared duplex engines."""
    engines = DuplexEngines(
        model_path, thinker_gpu_id=thinker_gpu_id, talker_gpu_id=talker_gpu_id,
        max_frames=max_frames,
    )
    manager = DuplexRealtimeSessionManager(engines=engines, model_name=model_name)
    app = FastAPI()
    app.state.realtime_manager = manager
    app.state.duplex_engines = engines

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        session = manager.open(websocket)
        try:
            await session.run()
        finally:
            await manager.close(session.session_id)

    return app
