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
_VAD_FRAME_SAMPLES = 512
_VAD_SAMPLE_RATE = 16000


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
class _Emit:
    type: str
    sample_offset: int


def _load_model_instance() -> object:
    """Construct a fresh silero-vad model. Each session owns its own
    instance — the model carries LSTM hidden state, so sharing across
    sessions would cross-contaminate VAD decisions under concurrency."""
    try:
        from silero_vad import load_silero_vad  # type: ignore[import-not-found]
    except ImportError as exc:
        raise VADUnavailable(
            "silero-vad is not installed; install with "
            "`pip install silero-vad onnxruntime` or set "
            "turn_detection.type to 'none' to disable server VAD"
        ) from exc

    try:
        return load_silero_vad(onnx=True)
    except Exception as exc:  # noqa: BLE001 — onnxruntime / model-load failures
        raise VADUnavailable(f"silero-vad initialization failed: {exc}") from exc


class StreamingVAD:
    """Per-session frame-by-frame VAD state machine.

    Callers feed raw PCM16 LE mono @ 16 kHz via :meth:`process`. The
    wrapper buffers up to one frame's worth of leftover bytes between
    calls so the caller doesn't have to align to 32 ms.
    """

    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        self._model = _load_model_instance()
        self._leftover_pcm = bytearray()
        self._samples_consumed = 0
        self._is_speech = False
        self._silence_run_samples = 0
        self._last_speech_offset = 0

    def process(self, pcm_bytes: bytes) -> list[_Emit]:
        """Feed PCM16 LE mono @ 16 kHz; return any state transitions."""
        if not pcm_bytes:
            return []
        self._leftover_pcm.extend(pcm_bytes)
        emits: list[_Emit] = []

        while len(self._leftover_pcm) >= _VAD_FRAME_SAMPLES * 2:
            frame_bytes = bytes(self._leftover_pcm[: _VAD_FRAME_SAMPLES * 2])
            del self._leftover_pcm[: _VAD_FRAME_SAMPLES * 2]
            frame = (
                np.frombuffer(frame_bytes, dtype="<i2").astype(np.float32) / 32768.0
            )

            prob = self._infer(frame)
            self._samples_consumed += _VAD_FRAME_SAMPLES
            frame_end = self._samples_consumed
            speech = prob >= self.config.threshold

            if speech:
                self._silence_run_samples = 0
                self._last_speech_offset = frame_end
                if not self._is_speech:
                    self._is_speech = True
                    # OpenAI's contract: speech_started reports the start
                    # offset *minus* prefix_padding so the caller includes
                    # a leading prefix in the committed audio.
                    pad = (
                        self.config.prefix_padding_ms * _VAD_SAMPLE_RATE // 1000
                    )
                    started_at = max(0, frame_end - _VAD_FRAME_SAMPLES - pad)
                    emits.append(
                        _Emit(VADEvent.SPEECH_STARTED, sample_offset=started_at)
                    )
            else:
                self._silence_run_samples += _VAD_FRAME_SAMPLES
                if self._is_speech:
                    silence_threshold = (
                        self.config.silence_duration_ms * _VAD_SAMPLE_RATE // 1000
                    )
                    if self._silence_run_samples >= silence_threshold:
                        self._is_speech = False
                        emits.append(
                            _Emit(
                                VADEvent.SPEECH_STOPPED,
                                sample_offset=self._last_speech_offset,
                            )
                        )

        return emits

    def _infer(self, frame: np.ndarray) -> float:
        import torch

        with torch.inference_mode():
            tensor = torch.from_numpy(frame).unsqueeze(0)
            prob = self._model(tensor, _VAD_SAMPLE_RATE).item()
        return float(prob)

    def reset(self) -> None:
        self._leftover_pcm.clear()
        self._is_speech = False
        self._silence_run_samples = 0
        self._last_speech_offset = self._samples_consumed
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()  # type: ignore[union-attr]


def offsets_to_ms(samples: int) -> int:
    return samples * 1000 // _VAD_SAMPLE_RATE


def emits_for_test(pcm_bytes: bytes, **cfg) -> list[tuple[str, int]]:
    """Test helper: drive the VAD on a complete byte buffer."""
    vad = StreamingVAD(VADConfig(**cfg))
    emits = vad.process(pcm_bytes)
    return [(e.type, offsets_to_ms(e.sample_offset)) for e in emits]
