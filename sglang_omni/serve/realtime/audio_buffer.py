# SPDX-License-Identifier: Apache-2.0
"""Rolling PCM16 audio buffer for streaming WebSocket sessions.

A ``RealtimeSession`` accumulates ``input_audio_buffer.append`` payloads
(base64-decoded raw PCM16 frames) into a single buffer and slices the
buffer out on commit. Currently 16 kHz mono PCM16 only, matching the
OpenAI Realtime default.
"""

from __future__ import annotations

import base64
import binascii
import io
import wave

class AudioBufferError(ValueError):
    """Raised when the client sends malformed audio data."""


class AudioBufferOverflow(AudioBufferError):
    """Raised when an append would exceed the configured byte cap.

    Distinct from ``AudioBufferError`` so the session layer can decide
    to commit-and-truncate or close the connection rather than just
    surfacing a generic invalid-request error.
    """


# 60 seconds @ 16 kHz mono PCM16 = 1_920_000 bytes per session by default.
# Longer-than-1-min utterances are vanishingly rare for transcription and
# almost always indicate a misbehaving / silent client; force a hard cap
# so a single WebSocket can't OOM the worker.
_DEFAULT_MAX_BUFFER_BYTES = 60 * 16000 * 2


class RealtimeAudioBuffer:
    """Append-only buffer of raw little-endian PCM16 bytes.

    Mutations (``append_b64``, ``clear``) correspond directly to the
    OpenAI ``input_audio_buffer.*`` event family. Slicing emits a
    ``data:audio/wav;base64,…`` URI so the engine's IPC layer (msgpack)
    can transport the payload.

    A configurable byte cap (``max_bytes``) prevents a single WebSocket
    from growing the buffer without bound. Exceeding the cap raises
    :class:`AudioBufferOverflow`; the session layer translates that to
    an ``error`` event and closes the connection.
    """

    def __init__(
        self,
        *,
        source_sr: int = 16000,
        target_sr: int = 16000,
        channels: int = 1,
        max_bytes: int = _DEFAULT_MAX_BUFFER_BYTES,
    ) -> None:
        self._source_sr = source_sr
        self._target_sr = target_sr
        self._channels = channels
        self._max_bytes = max_bytes
        self._buf = bytearray()

    def append_b64(self, audio_b64: str) -> int:
        """Decode a base64 PCM16 chunk and append. Returns bytes appended.

        Raises :class:`AudioBufferOverflow` if the resulting buffer would
        exceed ``max_bytes`` — the buffer is left unchanged in that case.
        """
        try:
            chunk = base64.b64decode(audio_b64, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise AudioBufferError(f"Invalid base64 audio: {exc}") from exc

        if not chunk:
            return 0
        if len(self._buf) + len(chunk) > self._max_bytes:
            raise AudioBufferOverflow(
                f"audio buffer would exceed cap of {self._max_bytes} bytes "
                f"(have {len(self._buf)}, appending {len(chunk)})"
            )
        self._buf.extend(chunk)
        return len(chunk)

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def clear(self) -> None:
        self._buf.clear()

    @property
    def num_bytes(self) -> int:
        return len(self._buf)

    @property
    def num_samples(self) -> int:
        return len(self._buf) // (2 * self._channels)

    def is_empty(self) -> bool:
        return self.num_samples == 0

    def to_wav_data_uri(self) -> str | None:
        """Serialize the full buffer as a WAV data URI; ``None`` if empty."""
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
