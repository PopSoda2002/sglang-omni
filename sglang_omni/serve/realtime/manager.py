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
        self._client = client
        self._model_name = model_name
        self._sessions: dict[str, RealtimeSession] = {}

    def open(self, websocket: WebSocket) -> RealtimeSession:
        session = RealtimeSession(
            websocket,
            client=self._client,
            model_name=self._model_name,
        )
        self._sessions[session.session_id] = session
        logger.info("Realtime session opened: %s", session.session_id)
        return session

    def close(self, session_id: str) -> None:
        if self._sessions.pop(session_id, None) is not None:
            logger.info("Realtime session closed: %s", session_id)

    def active_sessions(self) -> list[str]:
        return list(self._sessions.keys())
