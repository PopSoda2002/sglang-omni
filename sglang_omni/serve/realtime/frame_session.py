from __future__ import annotations

import json
import os
import queue
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from multiprocessing.connection import Client, Listener
from pathlib import Path

import torch

from sglang_omni.models.nemotron_voicechat.code2wav_stream import StreamingCodec
from sglang_omni.models.nemotron_voicechat.codec import RVQVAEDecoder
from sglang_omni.models.nemotron_voicechat.conformer import AudioPerception, StreamingPerception
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.models.nemotron_voicechat.stages import (
    _perception_config,
    create_talker_executor,
    create_thinker_executor,
)
from sglang_omni.models.weight_loader import (
    load_module,
    load_weights_by_prefix,
    resolve_model_path,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

SAMPLES_PER_FRAME = 1_280


def _engine_worker_main(kind, model_path, gpu_id, context_length, address, authkey_hex,
                        listen: bool = False):

    warnings.filterwarnings("ignore")
    connection = None
    if not listen:
        connection = Client(address, authkey=bytes.fromhex(authkey_hex))
    torch.cuda.set_device(gpu_id)

    device = f"cuda:{gpu_id}"
    if kind == "thinker":
        scheduler = create_thinker_executor(
            model_path, dtype="bfloat16", device=device, gpu_id=gpu_id,
        )
    else:
        scheduler = create_talker_executor(
            model_path, dtype="bfloat16", device=device, gpu_id=gpu_id,
            context_length=context_length,
        )

    def run_scheduler():
        torch.cuda.set_device(gpu_id)
        scheduler.start()

    threading.Thread(target=run_scheduler, daemon=True).start()
    if listen:
        # Server mode: the engine is up before the socket opens, so a
        # reachable address means an attachable engine.
        listener = Listener(address, authkey=bytes.fromhex(authkey_hex))
        connection = listener.accept()
        listener.close()
    connection.send(OutgoingMessage(request_id="", type="ready", data=kind))

    def feed():
        while True:
            message = connection.recv()
            if message is None:
                scheduler.stop()
                return
            if isinstance(message, tuple) and message[0] == "abort":
                scheduler.abort(message[1])
                continue
            scheduler.inbox.put(message)

    threading.Thread(target=feed, daemon=True).start()
    while True:
        try:
            connection.send(scheduler.outbox.get(timeout=0.5))
        except queue.Empty:
            continue


class _RemoteEngine:
    def __init__(self, kind, model_path, gpu_id, context_length=None):
        
        self.inbox: queue.Queue = queue.Queue()
        self.outbox: queue.Queue = queue.Queue()
        external = os.environ.get(f"NEMOTRON_{kind.upper()}_ENGINE_ADDRESS")
        if external:
            # An engine server launched out-of-band (sglang_omni.serve.realtime
            # engine-server mode); attach instead of spawning.
            authkey = os.environ["NEMOTRON_ENGINE_AUTHKEY"]
            self.process = None
            deadline = time.time() + 600
            while True:
                try:
                    self._connection = Client(external, authkey=bytes.fromhex(authkey))
                    break
                except (ConnectionRefusedError, FileNotFoundError):
                    assert time.time() < deadline, f"engine server at {external} never came up"
                    time.sleep(1)
        else:
            address = tempfile.mktemp(prefix=f"nemotron-{kind}-", suffix=".sock")
            authkey = secrets.token_hex(16)
            listener = Listener(address, authkey=bytes.fromhex(authkey))
            self.process = subprocess.Popen([
                sys.executable, "-u", "-c",
                "from sglang_omni.serve.realtime.frame_session import _engine_worker_main;"
                f"_engine_worker_main({kind!r}, {model_path!r}, {gpu_id}, {context_length!r}, {address!r}, {authkey!r})",
            ])
            self._connection = listener.accept()
            listener.close()

        def send_loop():
            while True:
                message = self.inbox.get()
                self._connection.send(message)
                if message is None:
                    return

        def recv_loop():
            while True:
                try:
                    self.outbox.put(self._connection.recv())
                except (EOFError, OSError):
                    return

        threading.Thread(target=send_loop, daemon=True).start()
        threading.Thread(target=recv_loop, daemon=True).start()

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        while True:
            if self.process is not None:
                assert self.process.poll() is None, "engine child process died during startup"
            try:
                message = self.outbox.get(timeout=5)
                break
            except queue.Empty:
                assert time.time() < deadline, "engine child startup timed out"
        assert message.type == "ready", message

    def abort(self, request_id: str) -> None:
        self.inbox.put(("abort", request_id))

    def close(self) -> None:
        if self.process is None:
            self.inbox.put(None)
            return
        if self.process.poll() is None:
            self.inbox.put(None)
            try:
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()


class DuplexEngines:
    def __init__(self, model_path: str, *, thinker_gpu_id: int, talker_gpu_id: int,
                 max_frames: int = 3_750, startup_timeout_s: float = 1_500.0) -> None:
        self.model_path = model_path
        self.max_frames = max_frames
        self.thinker = _RemoteEngine("thinker", model_path, thinker_gpu_id)
        self.thinker.wait_ready(startup_timeout_s)
        self.talker = _RemoteEngine("talker", model_path, talker_gpu_id, max_frames + 64)
        self.talker.wait_ready(startup_timeout_s)
        self.device = f"cuda:{thinker_gpu_id}"
        self.codec_decoder = self._build_codec_decoder(model_path, self.device)
        self.perception = self._build_perception(model_path, self.device)

    @staticmethod
    def _build_codec_decoder(model_path: str, device: str):

        config_path = Path(resolve_model_path(model_path)) / "config.json"
        codec_config = json.loads(config_path.read_text())["model"]["speech_generation"]["model"]["codec_config"]
        weights = load_weights_by_prefix(model_path, prefix=("tts_model.audio_codec.",))
        decoder_weights = {
            key: value for key, value in weights.items()
            if key.startswith("decoder.") or key.startswith("prvq.mus_list.")
        }
        decoder = RVQVAEDecoder(codec_config)
        decoder.load_state_dict(decoder_weights, strict=True)
        return decoder.to(device=device, dtype=torch.float32).eval()

    @staticmethod
    def _build_perception(model_path: str, device: str):

        module = AudioPerception(_perception_config(model_path))
        load_module(module, model_path, prefix="stt_model.perception.",
                    dtype=torch.float32, device=device)
        module.eval()
        return module

    def close(self) -> None:
        self.thinker.close()
        self.talker.close()


class FrameSession:
    """One duplex conversation on shared engines, one 80 ms frame at a time.

    Audio blocks feed the streaming perception; acoustic rows stream to the
    thinker, thinker tokens to the talker, talker codes to the incremental
    codec. ``push_audio`` returns whatever new output audio became ready.
    """

    def __init__(self, engines: DuplexEngines, *, session_id: str = "live",
                 max_frames: int | None = None) -> None:

        self.engines = engines
        self.session_id = session_id
        self.max_frames = min(max_frames or engines.max_frames, engines.max_frames)
        self._codec = StreamingCodec(engines.codec_decoder, engines.device)
        self._perception = StreamingPerception(engines.perception)

        state = NemotronVoiceChatState(num_frames=self.max_frames)
        request = OmniRequest(inputs={}, params={}, metadata={})
        for engine in (engines.talker, engines.thinker):
            engine.inbox.put(IncomingMessage(
                type="new_request", request_id=session_id,
                data=StagePayload(request_id=session_id, request=request, data=state.to_dict()),
            ))
        self.text_ids: list[int] = []
        self.frames_in = 0
        self._chunk_counter = 0
        self._closed = False

    def push_audio(self, block_S: torch.Tensor) -> torch.Tensor:
        rows = self._perception.push(block_S)
        return self.push_acoustic_rows(rows)

    def push_acoustic_rows(self, rows_ND: torch.Tensor) -> torch.Tensor:
        assert not self._closed
        for row in rows_ND:
            self._chunk_counter += 1
            self.engines.thinker.inbox.put(IncomingMessage(
                type="stream_chunk", request_id=self.session_id,
                data=StreamItem(chunk_id=self._chunk_counter,
                                data=row.detach().float().cpu(),
                                from_stage="perception"),
            ))
        self.frames_in += rows_ND.shape[0]
        return self.pump()

    def pump(self) -> torch.Tensor:
        audio_parts: list[torch.Tensor] = []
        try:
            while True:
                message = self.engines.thinker.outbox.get_nowait()
                if message.type == "stream" and message.target == "talker":
                    self.text_ids.append(int(message.data))
                    self.engines.talker.inbox.put(IncomingMessage(
                        type="stream_chunk", request_id=self.session_id,
                        data=StreamItem(chunk_id=len(self.text_ids), data=message.data,
                                        from_stage="thinker"),
                    ))
                elif message.type == "error":
                    raise RuntimeError(f"thinker error: {message.data}")
        except queue.Empty:
            pass
        try:
            while True:
                message = self.engines.talker.outbox.get_nowait()
                if message.type == "stream" and message.target == "code2wav":
                    audio_parts.append(self._codec.push(message.data))
                elif message.type == "error":
                    raise RuntimeError(f"talker error: {message.data}")
        except queue.Empty:
            pass
        if audio_parts:
            return torch.cat(audio_parts)
        return torch.zeros(0)

    def _progress(self):
        return (len(self.text_ids), len(self._codec.codes_rows), self._codec.emitted_samples)

    def drain(self, idle_s: float = 5.0, timeout_s: float = 180.0) -> torch.Tensor:
        parts = [self.pump()]
        deadline = time.time() + timeout_s
        last_progress = time.time()
        seen = self._progress()
        while time.time() < deadline and time.time() - last_progress < idle_s:
            time.sleep(0.02)
            parts.append(self.pump())
            if self._progress() != seen:
                seen = self._progress()
                last_progress = time.time()
        return torch.cat(parts)

    def flush(self) -> torch.Tensor:
        return self._codec.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.engines.thinker.abort(self.session_id)
        self.engines.talker.abort(self.session_id)
