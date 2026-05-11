# SPDX-License-Identifier: Apache-2.0
"""Lifecycle manager for active Realtime sessions.

Owns the in-process ``session_id → RealtimeSession`` map. The FastAPI
WebSocket handler creates a session via ``open()`` and removes it via
``close()`` on disconnect.
"""

from __future__ import annotations

import logging

from fastapi import WebSocket

from sglang_omni.client import Client
from sglang_omni.serve.realtime.session import RealtimeSession

logger = logging.getLogger(__name__)


class RealtimeSessionManager:
    def __init__(self, *, client: Client, model_name: str) -> None:
        self.client = client
        self.model_name = model_name
        self.sessions: dict[str, RealtimeSession] = {}

    def open(self, websocket: WebSocket) -> RealtimeSession:
        session = RealtimeSession(
            websocket,
            client=self.client,
            model_name=self.model_name,
        )
        self.sessions[session.session_id] = session
        logger.info("Realtime session opened: %s", session.session_id)
        return session

    async def close(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return
        # Always run teardown so GPU inference + asyncio tasks don't
        # leak on unexpected disconnect. Without this, every dropped
        # WebSocket left an active completion_stream + drainer task
        # running until process exit.
        await session.teardown()
        logger.info("Realtime session closed: %s", session_id)

    def active_sessions(self) -> list[str]:
        return list(self.sessions.keys())
