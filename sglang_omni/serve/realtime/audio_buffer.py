# SPDX-License-Identifier: Apache-2.0
"""Rolling PCM16 audio buffer for streaming WebSocket sessions.

A ``RealtimeSession`` accumulates ``input_audio_buffer.append`` payloads
(base64-decoded raw PCM16 frames) into a single buffer and slices the
buffer out on commit. M0 only supports the OpenAI default of 16 kHz mono
PCM16; M1 adds source-rate negotiation and g711 codecs.
"""

from __future__ import annotations

import base64
import binascii

import numpy as np

from sglang_omni.preprocessing.audio import pcm16_bytes_to_float32


class AudioBufferError(ValueError):
    """Raised when the client sends malformed audio data."""


class RealtimeAudioBuffer:
    """Append-only buffer of raw little-endian PCM16 bytes.

    The buffer is mutable in place — append/commit/clear correspond directly
    to the OpenAI ``input_audio_buffer.*`` event family. Slicing on commit
    returns a copy as ``float32`` mono in [-1, 1] at ``target_sr``, suitable
    to drop into ``GenerateRequest.metadata["audios"]``.
    """

    def __init__(
        self,
        *,
        source_sr: int = 16000,
        target_sr: int = 16000,
        channels: int = 1,
    ) -> None:
        self._source_sr = source_sr
        self._target_sr = target_sr
        self._channels = channels
        self._buf = bytearray()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append_b64(self, audio_b64: str) -> int:
        """Decode a base64 PCM16 chunk and extend the buffer.

        Returns the number of *bytes* appended (post-decode).
        """
        try:
            chunk = base64.b64decode(audio_b64, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise AudioBufferError(f"Invalid base64 audio: {exc}") from exc

        if not chunk:
            return 0
        self._buf.extend(chunk)
        return len(chunk)

    def clear(self) -> None:
        self._buf.clear()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def num_bytes(self) -> int:
        return len(self._buf)

    @property
    def num_samples(self) -> int:
        bytes_per_sample = 2 * self._channels
        return len(self._buf) // bytes_per_sample

    @property
    def duration_ms(self) -> int:
        if self._source_sr <= 0:
            return 0
        return int(round(self.num_samples * 1000 / self._source_sr))

    def is_empty(self) -> bool:
        return self.num_samples == 0

    def to_numpy(self) -> np.ndarray:
        """Return a float32 mono copy at ``target_sr``. Empty buffer ⇒ empty array."""
        if self.is_empty():
            return np.zeros(0, dtype=np.float32)
        return pcm16_bytes_to_float32(
            bytes(self._buf),
            source_sr=self._source_sr,
            target_sr=self._target_sr,
            channels=self._channels,
        )
