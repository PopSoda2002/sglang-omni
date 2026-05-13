# SPDX-License-Identifier: Apache-2.0
"""Real-server integration tests for the /v1/realtime WebSocket endpoint.

Launches a Qwen3-Omni thinker-only server with ``--enable-realtime`` and
drives /v1/realtime via the ``websockets`` client library. Covers:
  - ``response.create`` full lifecycle (text deltas → ``response.done``);
  - server VAD auto-commit + transcription end-to-end on a real wav;
  - clean teardown when the client disconnects mid-flight — the server
    must stay healthy and accept new connections.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import wave
from pathlib import Path

import pytest
import requests
import websockets

from sglang_omni.utils import find_available_port
from tests.utils import (
    disable_proxy,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_NAME = "qwen3-omni"
STARTUP_TIMEOUT = 600
WS_TIMEOUT = 60
AUDIO_FIXTURE = Path(__file__).parent.parent / "data" / "query_to_draw.wav"


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    port = find_available_port()
    log_file = server_log_file(tmp_path_factory, "realtime_logs")
    cmd = [
        sys.executable,
        "examples/run_qwen3_omni_server.py",
        "--version",
        "v1",
        "--model-path",
        MODEL_PATH,
        "--model-name",
        MODEL_NAME,
        "--enable-realtime",
        "--port",
        str(port),
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
    proc.port = port  # type: ignore[attr-defined]
    yield proc
    stop_server(proc)


def _ws_url(port: int) -> str:
    return f"ws://localhost:{port}/v1/realtime"


def _load_pcm16_16k_mono(path: Path) -> bytes:
    with wave.open(str(path)) as wf:
        assert wf.getnchannels() == 1, "fixture must be mono"
        assert wf.getframerate() == 16000, "fixture must be 16 kHz"
        assert wf.getsampwidth() == 2, "fixture must be PCM16"
        return wf.readframes(wf.getnframes())


async def _recv_event(ws) -> dict:
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))


async def _recv_until(ws, terminal_type: str, *, limit: int = 300) -> list[dict]:
    events: list[dict] = []
    for _ in range(limit):
        evt = await _recv_event(ws)
        events.append(evt)
        if evt.get("type") == terminal_type:
            return events
    raise AssertionError(
        f"did not see {terminal_type} after {limit} events; "
        f"saw {[e.get('type') for e in events]}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_create_streams_text_deltas(
    server_process: subprocess.Popen,
) -> None:
    """Real engine drives response.created → text.delta × N → text.done → done."""
    port: int = server_process.port  # type: ignore[attr-defined]
    with disable_proxy():
        async with websockets.connect(_ws_url(port)) as ws:
            await _recv_event(ws)  # session.created

            # Override instructions to make the response short + predictable.
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "instructions": (
                                "Reply with exactly one word: OK. " "No punctuation."
                            ),
                        },
                    }
                )
            )
            await _recv_event(ws)  # session.updated

            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Say it."}],
                        },
                    }
                )
            )
            await _recv_event(ws)  # conversation.item.created

            await ws.send(json.dumps({"type": "response.create"}))
            events = await _recv_until(ws, "response.done")

    types = [e["type"] for e in events]
    assert types[0] == "response.created", types
    assert "response.text.delta" in types, types
    assert "response.text.done" in types, types
    deltas = [e["delta"] for e in events if e["type"] == "response.text.delta"]
    assert "".join(deltas).strip(), "expected non-empty response text"


@pytest.mark.asyncio
async def test_server_vad_auto_commit_transcribes_real_audio(
    server_process: subprocess.Popen,
) -> None:
    """Server VAD detects speech, auto-commits, and yields a non-empty transcript."""
    port: int = server_process.port  # type: ignore[attr-defined]
    pcm = _load_pcm16_16k_mono(AUDIO_FIXTURE)
    # 1 s silence trailer so VAD reliably emits speech_stopped after the
    # fixture ends; without it the run can hang waiting for silence that
    # arrives only at WS close.
    pcm = pcm + b"\x00\x00" * 16000

    with disable_proxy():
        async with websockets.connect(_ws_url(port)) as ws:
            await _recv_event(ws)  # session.created

            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "instructions": (
                                "You are a transcription engine. "
                                "Transcribe the user's speech verbatim. "
                                "Output ONLY the transcript."
                            ),
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.5,
                                "silence_duration_ms": 500,
                                "prefix_padding_ms": 200,
                            },
                        },
                    }
                )
            )
            await _recv_event(ws)  # session.updated

            # 200 ms chunks; no real-time pacing — VAD processes inline.
            chunk_bytes = 16000 * 200 // 1000 * 2
            for i in range(0, len(pcm), chunk_bytes):
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(
                                pcm[i : i + chunk_bytes]
                            ).decode(),
                        }
                    )
                )

            events = await _recv_until(
                ws, "conversation.item.input_audio_transcription.completed"
            )

    types = [e["type"] for e in events]
    assert "input_audio_buffer.speech_started" in types, types
    assert "input_audio_buffer.speech_stopped" in types, types
    assert "input_audio_buffer.committed" in types, types
    assert "conversation.item.created" in types, types
    completed = next(
        e
        for e in events
        if e["type"] == "conversation.item.input_audio_transcription.completed"
    )
    assert completed.get(
        "transcript", ""
    ).strip(), f"expected non-empty transcript; got {completed!r}"


@pytest.mark.asyncio
async def test_disconnect_during_response_keeps_server_healthy(
    server_process: subprocess.Popen,
) -> None:
    """An abrupt disconnect after kicking off a response must not leak tasks
    or break the route — /health still answers and a fresh WS still works.
    """
    port: int = server_process.port  # type: ignore[attr-defined]
    with disable_proxy():
        async with websockets.connect(_ws_url(port)) as ws:
            await _recv_event(ws)  # session.created
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "instructions": "Reply with the single word: OK.",
                        },
                    }
                )
            )
            await _recv_event(ws)  # session.updated
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Now."}],
                        },
                    }
                )
            )
            await _recv_event(ws)  # conversation.item.created
            await ws.send(json.dumps({"type": "response.create"}))
            # Take a single response event to confirm the response task is
            # alive on the server, then close abruptly — the exact event
            # type doesn't matter for the disconnect-cleanup contract.
            evt = await _recv_event(ws)
            assert evt.get("type", "").startswith("response."), evt
        # Context manager exit closes the WebSocket abruptly.

        # Server's manager.close → session.teardown must clean up without
        # raising; if it leaked the engine call or re-raised CancelledError
        # from .exception(), /health would either fail or hang.
        resp = requests.get(f"http://localhost:{port}/health", timeout=10)
        assert resp.status_code == 200, resp.text

        # And a fresh WS connection should still be accepted.
        async with websockets.connect(_ws_url(port)) as ws:
            evt = await _recv_event(ws)
            assert evt["type"] == "session.created", evt
