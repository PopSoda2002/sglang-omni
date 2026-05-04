# SPDX-License-Identifier: Apache-2.0
"""Streaming VAD wrapper used by the Realtime WebSocket session.

Wraps silero-vad (ONNX) into a frame-by-frame state machine matching
the OpenAI Realtime ``turn_detection`` semantics:

    speech_started  ← prob ≥ threshold for ≥ 1 frame
    speech_stopped  ← prob < threshold for silence_duration_ms continuously
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# silero-vad operates on 512-sample windows @ 16 kHz (32 ms each).
VAD_FRAME_SAMPLES = 512
VAD_SAMPLE_RATE = 16000


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
    decisions under concurrency.

    If ``silero-vad`` isn't installed or the ONNX model fails to load,
    the underlying exception (``ImportError`` / runtime error)
    propagates. Callers don't catch.
    """
    from silero_vad import load_silero_vad  # type: ignore[import-not-found]

    return load_silero_vad(onnx=True)


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
