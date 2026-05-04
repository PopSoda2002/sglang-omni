# SPDX-License-Identifier: Apache-2.0
"""OpenAI Realtime API (WebSocket /v1/realtime).

This package implements the OpenAI Realtime wire format for sglang-omni v1.
M0 ships a fresh-OmniRequest-per-commit lifecycle for transcription; the
anchor-request KV-preservation lifecycle lands in M1.

References:
    https://developers.openai.com/api/docs/guides/realtime
"""

from sglang_omni.serve.realtime.manager import RealtimeSessionManager
from sglang_omni.serve.realtime.session import RealtimeSession

__all__ = ["RealtimeSession", "RealtimeSessionManager"]
