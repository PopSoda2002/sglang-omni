# SPDX-License-Identifier: Apache-2.0
"""Streaming VAD wrapper used by the Realtime WebSocket session.

Wraps silero-vad (ONNX) into a frame-by-frame state machine matching
the OpenAI Realtime ``turn_detection`` semantics:

    speech_started  ← prob ≥ threshold for ≥ 1 frame
    speech_stopped  ← prob < threshold for silence_duration_ms continuously
"""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass
from importlib import import_module

import numpy as np

logger = logging.getLogger(__name__)

# silero-vad operates on 512-sample windows @ 16 kHz (32 ms each).
VAD_FRAME_SAMPLES = 512
VAD_SAMPLE_RATE = 16000


class VADUnavailable(RuntimeError):
    """Raised when silero-vad cannot be imported or instantiated."""


@dataclass
class VADConfig:
    """Mirrors OpenAI Realtime ``turn_detection`` (server_vad mode)."""

    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500


class VADEvent:
    SPEECH_STARTED = "speech_started"
    SPEECH_STOPPED = "speech_stopped"


@dataclass
class Emit:
    type: str
    sample_offset: int


def load_silero_model() -> object:
    """Construct a fresh silero-vad model instance.

    Each session owns its own instance — silero carries LSTM hidden
    state, so sharing across sessions would cross-contaminate VAD
    decisions under concurrency. Raises :class:`VADUnavailable` on
    import or initialization failure so the session layer can return
    a structured error instead of crashing the WebSocket.
    """
    silero = None
    with suppress(ImportError):
        silero = import_module("silero_vad")
    if silero is None:
        raise VADUnavailable(
            "silero-vad is not installed; install with "
            "`pip install silero-vad onnxruntime` or set "
            "turn_detection.type to 'none' to disable server VAD"
        )

    model: object | None = None
    with suppress(Exception):
        model = silero.load_silero_vad(onnx=True)
    if model is None:
        raise VADUnavailable("silero-vad initialization failed")
    return model


class StreamingVAD:
    """Per-session frame-by-frame VAD state machine.

    Callers feed raw PCM16 LE mono @ 16 kHz via :meth:`process`. The
    wrapper buffers up to one frame's worth of leftover bytes between
    calls so the caller doesn't have to align to 32 ms.
    """

    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        self.model = load_silero_model()
        self.leftover_pcm = bytearray()
        self.samples_consumed = 0
        self.is_speech = False
        self.silence_run_samples = 0
        self.last_speech_offset = 0

    def process(self, pcm_bytes: bytes) -> list[Emit]:
        """Feed PCM16 LE mono @ 16 kHz; return any state transitions."""
        if not pcm_bytes:
            return []
        self.leftover_pcm.extend(pcm_bytes)
        emits: list[Emit] = []

        while len(self.leftover_pcm) >= VAD_FRAME_SAMPLES * 2:
            frame_bytes = bytes(self.leftover_pcm[: VAD_FRAME_SAMPLES * 2])
            del self.leftover_pcm[: VAD_FRAME_SAMPLES * 2]
            frame = (
                np.frombuffer(frame_bytes, dtype="<i2").astype(np.float32) / 32768.0
            )

            prob = self.infer(frame)
            self.samples_consumed += VAD_FRAME_SAMPLES
            frame_end = self.samples_consumed
            speech = prob >= self.config.threshold

            if speech:
                self.silence_run_samples = 0
                self.last_speech_offset = frame_end
                if not self.is_speech:
                    self.is_speech = True
                    # OpenAI's contract: speech_started reports the start
                    # offset *minus* prefix_padding so the caller includes
                    # a leading prefix in the committed audio.
                    pad = (
                        self.config.prefix_padding_ms * VAD_SAMPLE_RATE // 1000
                    )
                    started_at = max(0, frame_end - VAD_FRAME_SAMPLES - pad)
                    emits.append(
                        Emit(VADEvent.SPEECH_STARTED, sample_offset=started_at)
                    )
            else:
                self.silence_run_samples += VAD_FRAME_SAMPLES
                if self.is_speech:
                    silence_threshold = (
                        self.config.silence_duration_ms * VAD_SAMPLE_RATE // 1000
                    )
                    if self.silence_run_samples >= silence_threshold:
                        self.is_speech = False
                        emits.append(
                            Emit(
                                VADEvent.SPEECH_STOPPED,
                                sample_offset=self.last_speech_offset,
                            )
                        )

        return emits

    def infer(self, frame: np.ndarray) -> float:
        import torch

        with torch.inference_mode():
            tensor = torch.from_numpy(frame).unsqueeze(0)
            prob = self.model(tensor, VAD_SAMPLE_RATE).item()
        return float(prob)

    def reset(self) -> None:
        self.leftover_pcm.clear()
        self.is_speech = False
        self.silence_run_samples = 0
        self.last_speech_offset = self.samples_consumed
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()  # type: ignore[union-attr]


def offsets_to_ms(samples: int) -> int:
    return samples * 1000 // VAD_SAMPLE_RATE


def emits_for_test(pcm_bytes: bytes, **cfg) -> list[tuple[str, int]]:
    """Test helper: drive the VAD on a complete byte buffer."""
    vad = StreamingVAD(VADConfig(**cfg))
    emits = vad.process(pcm_bytes)
    return [(e.type, offsets_to_ms(e.sample_offset)) for e in emits]
