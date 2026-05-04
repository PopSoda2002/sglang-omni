# SPDX-License-Identifier: Apache-2.0
"""Lifecycle manager for active Realtime sessions.

Owns the in-process map of ``session_id`` → :class:`RealtimeSession`. The
FastAPI WebSocket handler creates a session via ``open()`` and the
session removes itself via ``close()`` on disconnect. The manager exists
so future code (admin endpoints, idle-timeout reaper, multi-tenant
limits) has a single hook; M0 keeps it deliberately small.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket

from sglang_omni.client import Client
from sglang_omni.serve.realtime.session import RealtimeSession

logger = logging.getLogger(__name__)


class RealtimeSessionManager:
    """In-memory ``session_id → RealtimeSession`` map (single-process)."""

    def __init__(self, *, client: Client, model_name: str) -> None:
        self._client = client
        self._model_name = model_name
        self._sessions: dict[str, RealtimeSession] = {}
        self._lock = asyncio.Lock()

    def open(self, websocket: WebSocket) -> RealtimeSession:
        """Create and register a new session bound to ``websocket``."""
        session = RealtimeSession(
            websocket,
            client=self._client,
            model_name=self._model_name,
        )
        # `open` is sync because creating the session is sync; we acquire
        # the lock only on mutating `_sessions`. asyncio.Lock is not
        # acquired here to avoid making this an async operation; the dict
        # mutation is atomic under the GIL and a stray miss in
        # `active_sessions()` is harmless.
        self._sessions[session.session_id] = session
        logger.info("Realtime session opened: %s", session.session_id)
        return session

    def close(self, session_id: str) -> None:
        if self._sessions.pop(session_id, None) is not None:
            logger.info("Realtime session closed: %s", session_id)

    def active_sessions(self) -> list[str]:
        return list(self._sessions.keys())
