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
import io
import struct
import wave

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

    def to_wav_data_uri(self) -> str | None:
        """Serialize the current buffer as a ``data:audio/wav;base64,...`` URI.

        The pipeline IPC layer (msgpack) cannot transport ``numpy.ndarray``;
        the multimodal resource connector accepts ``data:`` URIs and
        decodes them back to float32 on the preprocessor side. Returns
        ``None`` if the buffer is empty.
        """
        return self.slice_to_wav_data_uri(start_byte=0, end_byte=len(self._buf))

    def slice_to_wav_data_uri(
        self, *, start_byte: int, end_byte: int
    ) -> str | None:
        """Slice ``[start_byte:end_byte]`` and emit a WAV data URI.

        Used by server-side VAD to commit only the speech segment.
        Returns ``None`` if the slice is empty.
        """
        start_byte = max(0, start_byte)
        end_byte = min(end_byte, len(self._buf))
        if end_byte <= start_byte:
            return None
        # Align to whole samples in case the caller passed a stray odd offset.
        bytes_per_sample = 2 * self._channels
        end_byte -= (end_byte - start_byte) % bytes_per_sample

        chunk = bytes(self._buf[start_byte:end_byte])
        if not chunk:
            return None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self._channels)
            wf.setsampwidth(2)
            wf.setframerate(self._source_sr)
            wf.writeframes(chunk)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:audio/wav;base64,{b64}"

    def tail(self, num_bytes: int) -> bytes:
        """Return the most recently appended ``num_bytes``."""
        if num_bytes <= 0 or not self._buf:
            return b""
        n = min(num_bytes, len(self._buf))
        return bytes(self._buf[-n:])

    @staticmethod
    def numpy_to_wav_data_uri(audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Encode an in-memory ``float32`` mono array as a WAV data URI."""
        if audio.size == 0:
            raise ValueError("audio is empty")
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{pcm16.size}h", *pcm16.tolist()))
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:audio/wav;base64,{b64}"
