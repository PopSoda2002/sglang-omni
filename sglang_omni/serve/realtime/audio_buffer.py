# SPDX-License-Identifier: Apache-2.0
"""Rolling PCM16 audio buffer for streaming WebSocket sessions.

A ``RealtimeSession`` accumulates ``input_audio_buffer.append`` payloads
(base64-decoded raw PCM16 frames) into a single buffer and slices the
buffer out on commit. Currently 16 kHz mono PCM16 only, matching the
OpenAI Realtime default.
"""

from __future__ import annotations

import base64
import io
import wave
from typing import Literal

# 60 seconds @ 16 kHz mono PCM16 = 1_920_000 bytes per session by default.
# Longer-than-1-min utterances are vanishingly rare for transcription and
# almost always indicate a misbehaving / silent client; force a hard cap
# so a single WebSocket can't OOM the worker.
DEFAULT_MAX_BUFFER_BYTES = 60 * 16000 * 2


# Error codes returned by RealtimeAudioBuffer.append_b64. Plain strings
# rather than exception subclasses so the call site stays linear.
AppendError = Literal["overflow"]


class RealtimeAudioBuffer:
    """Append-only buffer of raw little-endian PCM16 bytes."""

    def __init__(
        self,
        *,
        source_sr: int = 16000,
        target_sr: int = 16000,
        channels: int = 1,
        max_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
    ) -> None:
        self.source_sr = source_sr
        self.target_sr = target_sr
        self.channels = channels
        self.max_bytes = max_bytes
        self.buf = bytearray()

    def append_b64(self, audio_b64: str) -> tuple[int, AppendError | None]:
        """Decode a base64 PCM16 chunk and append it.

        Returns ``(bytes_appended, error)``:
            - ``(N, None)`` on success
            - ``(0, "overflow")`` if appending would exceed ``max_bytes``
              (buffer left unchanged)
            - ``(0, None)`` if the chunk decoded to zero bytes

        Malformed base64 raises ``binascii.Error`` from
        :func:`base64.b64decode` — callers don't catch it.
        """
        chunk = base64.b64decode(audio_b64, validate=False)
        if not chunk:
            return 0, None
        if len(self.buf) + len(chunk) > self.max_bytes:
            return 0, "overflow"
        self.buf.extend(chunk)
        return len(chunk), None

    def clear(self) -> None:
        self.buf.clear()

    @property
    def num_bytes(self) -> int:
        return len(self.buf)

    @property
    def num_samples(self) -> int:
        return len(self.buf) // (2 * self.channels)

    def is_empty(self) -> bool:
        return self.num_samples == 0

    def to_wav_data_uri(self) -> str | None:
        """Serialize the full buffer as a WAV data URI; ``None`` if empty."""
        return self.slice_to_wav_data_uri(start_byte=0, end_byte=len(self.buf))

    def slice_to_wav_data_uri(self, *, start_byte: int, end_byte: int) -> str | None:
        """Slice ``[start_byte:end_byte]`` and emit a WAV data URI.

        Used by server-side VAD to commit only the speech segment.
        Returns ``None`` if the slice is empty.
        """
        start_byte = max(0, start_byte)
        end_byte = min(end_byte, len(self.buf))
        if end_byte <= start_byte:
            return None
        bytes_per_sample = 2 * self.channels
        end_byte -= (end_byte - start_byte) % bytes_per_sample

        chunk = bytes(self.buf[start_byte:end_byte])
        if not chunk:
            return None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.source_sr)
            wf.writeframes(chunk)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:audio/wav;base64,{b64}"

    def tail(self, num_bytes: int) -> bytes:
        """Return the most recently appended ``num_bytes``."""
        if num_bytes <= 0 or not self.buf:
            return b""
        n = min(num_bytes, len(self.buf))
        return bytes(self.buf[-n:])
