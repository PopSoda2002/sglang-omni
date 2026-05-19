# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for Higgs TTS.

Mirrors :class:`sglang_omni.models.fishaudio_s2_pro.streaming_vocoder.
S2ProVocoderScheduler` but adapts the chunking math to Higgs's
delay-pattern codec: every codebook ``c`` is delayed by ``c`` steps,
so recovering ``T`` data frames requires ``T + N - 1`` delayed rows
from the AR engine.

Message flow:

- ``type="new_request"`` from the TTS engine: register the payload.
  For streaming requests we just wait for ``stream`` messages; for
  non-streaming requests we run the existing one-shot decode path
  (full delayed codes ``→ reverse_delay_pattern → codec.decode``).
- ``type="stream"`` (per AR step): append a ``[1, N]`` delayed-code
  row, run :func:`build_stream_chunk` which decides whether to flush
  a new audio chunk (dynamic frame accumulation: small first stride,
  larger followup stride).
- ``type="stream_done"`` (when the AR finishes): flush whatever data
  frames remain plus any held-back crossfade tail.

Dynamic-frame-accumulation defaults match s2_pro (``stride=10``,
``followup_stride=90``). The codec-chunkability sweep
(:mod:`_perf_bench.probe_codec_chunking`) pins ``overlap_frames`` and
``crossfade_samples``.
"""

from __future__ import annotations

import collections
import logging
import queue as _queue_mod
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.utils import reverse_delay_pattern
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

_ABORTED_REQUEST_ID_LIMIT = 10000
_ABORTED_REQUEST_ID_RETAINED = 5000


@dataclass
class _StreamState:
    """Per-request streaming state.

    All offsets are in *delayed-frame* units (one row per AR step).
    ``code_start_frame`` is the global delayed-frame index of
    ``delayed_codes[0]`` (we trim leading rows once they're behind the
    overlap window, so this drifts upward over time).
    """

    delayed_codes: list[torch.Tensor] = field(default_factory=list)
    code_start_frame: int = 0
    total_delayed: int = 0
    last_emitted_data_frame: int = 0
    # Threshold (in *data* frames) that triggers the next emit. Stays at
    # the initial stride until the first chunk lands, then follows
    # ``last_emitted_data_frame + followup_stride``.
    next_emit_data_frame: int = 0
    pending_tail: torch.Tensor | None = None


def _build_audio_chunk_payload(
    audio_data: torch.Tensor, *, sample_rate: int
) -> dict[str, Any]:
    return {
        "audio_data": audio_data.cpu().tolist(),
        "sample_rate": sample_rate,
        "modality": "audio",
    }


def _crossfade(
    pending_tail: torch.Tensor | None,
    new_audio: torch.Tensor,
    *,
    crossfade_samples: int,
    is_final: bool,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Glue ``pending_tail`` + ``new_audio`` with a linear cross-fade and
    return ``(new_pending_tail, ready_audio)``.

    The last ``crossfade_samples`` of the result are held back as the
    next chunk's pending tail (except on the final flush, which emits
    everything).
    """
    if pending_tail is not None and pending_tail.numel() > 0:
        n = min(
            crossfade_samples,
            int(pending_tail.shape[-1]),
            int(new_audio.shape[-1]),
        )
        if n > 0:
            fade_in = torch.linspace(
                0.0, 1.0, n, dtype=new_audio.dtype, device=new_audio.device
            )
            fade_out = 1.0 - fade_in
            blended = pending_tail[-n:] * fade_out + new_audio[:n] * fade_in
            new_audio = torch.cat(
                [pending_tail[:-n], blended, new_audio[n:]]
            )
        else:
            new_audio = torch.cat([pending_tail, new_audio])

    if is_final:
        return None, new_audio

    hold = min(crossfade_samples, int(new_audio.shape[-1]))
    if hold > 0:
        new_pending = new_audio[-hold:].clone()
        new_audio = new_audio[:-hold]
        return new_pending, new_audio
    return None, new_audio


def _trim_retained_codes(
    state: _StreamState, *, keep_from_frame: int
) -> None:
    """Pop already-consumed delayed rows once they're behind the
    overlap+delay window; bounds memory in long generations.
    """
    if keep_from_frame <= state.code_start_frame:
        return
    drop = keep_from_frame - state.code_start_frame
    while drop > 0 and state.delayed_codes:
        first = state.delayed_codes[0]
        first_width = int(first.shape[0])
        if drop >= first_width:
            state.delayed_codes.pop(0)
            state.code_start_frame += first_width
            drop -= first_width
            continue
        state.delayed_codes[0] = first[drop:].contiguous()
        state.code_start_frame += drop
        drop = 0


def _try_emit_chunk(
    state: _StreamState,
    *,
    codec: Any,
    num_codebooks: int,
    frame_length: int,
    stride: int,
    followup_stride: int,
    overlap_frames: int,
    crossfade_samples: int,
    is_final: bool,
) -> dict[str, Any] | None:
    """Decide whether to flush a chunk; if so, decode it and return the
    audio payload.

    Layout (data-frame coordinates):

        last_emitted ──┐
                       ▼
        ┌──────────────┬─────────────────┐
        │   overlap    │     stride      │  ← decoded together
        └──────────────┴─────────────────┘
                       │
                       └─ drop overlap_samples from the head of decoded
                          audio; the rest goes through cross-fade with
                          any prior pending tail.

    Returns ``None`` when we don't have enough delayed frames yet (or
    nothing new to emit on a final flush).
    """
    N = num_codebooks
    target_data_end = (
        state.next_emit_data_frame or stride
    ) if not is_final else _max_recoverable_data_frames(state, N)
    if target_data_end <= state.last_emitted_data_frame:
        return None
    target_data_start = max(0, state.last_emitted_data_frame - overlap_frames)
    # Reverse delay needs target_data_end + (N - 1) delayed rows.
    delayed_needed = target_data_end + (N - 1)
    if state.total_delayed < delayed_needed:
        return None

    rel_start = target_data_start - state.code_start_frame
    rel_end = delayed_needed - state.code_start_frame
    if rel_start < 0:
        # Shouldn't happen — we only trim up to overlap behind
        # last_emitted, so the window stays in the retained slice.
        rel_start = 0
    delayed_LN = torch.cat(state.delayed_codes, dim=0)[rel_start:rel_end]
    if delayed_LN.shape[0] < N:
        return None
    # Map any sampler-vocab specials (BOC=1024 / EOC=1025) back to a
    # safe codec value; the codec only knows ids 0..codebook_size-1.
    codec_vocab = codec.model.config.codebook_size
    delayed_LN = torch.where(
        delayed_LN >= codec_vocab, torch.zeros_like(delayed_LN), delayed_LN
    )

    data_TN = reverse_delay_pattern(delayed_LN)
    audio = codec.decode(data_TN).to(torch.float32)

    drop_frames = state.last_emitted_data_frame - target_data_start
    drop_samples = drop_frames * frame_length
    if drop_samples >= audio.shape[-1]:
        return None
    new_audio = audio[drop_samples:]

    new_pending, ready_audio = _crossfade(
        state.pending_tail,
        new_audio,
        crossfade_samples=crossfade_samples,
        is_final=is_final,
    )
    state.pending_tail = new_pending
    state.last_emitted_data_frame = target_data_end
    if not is_final:
        state.next_emit_data_frame = (
            target_data_end + followup_stride
        )
    # Free delayed rows behind the overlap window.
    _trim_retained_codes(
        state, keep_from_frame=max(0, target_data_end - overlap_frames)
    )

    if ready_audio.numel() == 0:
        return None
    return _build_audio_chunk_payload(ready_audio, sample_rate=codec.SAMPLE_RATE)


def _max_recoverable_data_frames(state: _StreamState, num_codebooks: int) -> int:
    """At ``stream_done`` time, the AR has emitted ``total_delayed`` rows
    (data + N-1 wind-down), so we can recover ``total_delayed - (N-1)``
    data frames in total. Clamp to zero in case the AR aborted early.
    """
    return max(0, state.total_delayed - (num_codebooks - 1))


def build_stream_chunk(
    state: _StreamState,
    codes_row: torch.Tensor,
    *,
    codec: Any,
    num_codebooks: int,
    frame_length: int,
    stride: int,
    followup_stride: int,
    overlap_frames: int,
    crossfade_samples: int,
) -> dict[str, Any] | None:
    """Accumulate one delayed-code row; emit an audio chunk when enough
    delayed frames have arrived for ``stride`` more data frames."""
    if codes_row.ndim != 2 or codes_row.shape[1] != num_codebooks:
        raise ValueError(
            f"stream code chunk must be [L, {num_codebooks}], "
            f"got {tuple(codes_row.shape)}"
        )
    state.delayed_codes.append(codes_row.detach().to(torch.long))
    state.total_delayed += int(codes_row.shape[0])
    return _try_emit_chunk(
        state,
        codec=codec,
        num_codebooks=num_codebooks,
        frame_length=frame_length,
        stride=stride,
        followup_stride=followup_stride,
        overlap_frames=overlap_frames,
        crossfade_samples=crossfade_samples,
        is_final=False,
    )


def flush_stream_chunk(
    state: _StreamState,
    *,
    codec: Any,
    num_codebooks: int,
    frame_length: int,
    overlap_frames: int,
    crossfade_samples: int,
) -> dict[str, Any] | None:
    """``stream_done`` path: emit whatever data frames remain plus the
    held-back crossfade tail."""
    chunk = _try_emit_chunk(
        state,
        codec=codec,
        num_codebooks=num_codebooks,
        frame_length=frame_length,
        stride=0,  # unused on final path
        followup_stride=0,
        overlap_frames=overlap_frames,
        crossfade_samples=crossfade_samples,
        is_final=True,
    )
    if chunk is not None:
        return chunk
    # No new data frames but a tail might still be held back.
    tail = state.pending_tail
    if tail is not None and tail.numel() > 0:
        state.pending_tail = None
        return _build_audio_chunk_payload(tail, sample_rate=codec.SAMPLE_RATE)
    return None


class HiggsStreamingVocoderScheduler:
    """Higgs vocoder scheduler with streaming and one-shot batch paths.

    Streaming requests (``params.stream == True``) accumulate per-step
    code rows and emit audio chunks under dynamic frame accumulation.
    Non-streaming requests run the existing one-shot reverse-delay +
    decode path (parity with :func:`stages.create_vocoder_executor`).
    """

    def __init__(
        self,
        codec: Any,
        *,
        device: str,
        num_codebooks: int = 8,
        # Defaults are a Baseten-style "dynamic frame accumulation"
        # split: small first chunk for low TTFA, larger followup chunks
        # for higher concurrency throughput.
        #
        # Higgs codec runs at 25 Hz (hop_length=960, one code frame ≈ 40 ms
        # audio), so:
        #   - stride=10  -> ~400 ms first chunk  (TTFA ≈ AR(17 steps) + decode)
        #   - stream_followup_stride=40 -> ~1.6 s subsequent chunks
        #
        # ``_perf_bench/probe_codec_chunking.py`` found the codec is
        # essentially zero-context-dependence at chunk size 40+; smaller
        # strides accumulate audible chunk-boundary drift (the first
        # chunk's audio is slightly noisier than a one-shot decode of
        # the same codes). overlap > 0 makes it WORSE — decoding the
        # overlap region in a different chunk context produces a
        # subtly different sample sequence — so overlap and crossfade
        # default to zero.
        stream_stride: int = 10,
        stream_followup_stride: int = 40,
        stream_overlap_frames: int = 0,
        stream_crossfade_samples: int = 0,
        max_batch_size: int = 4,
        max_batch_wait_ms: int = 2,
    ):
        if stream_stride <= 0 or stream_followup_stride <= 0 or max_batch_size <= 0:
            raise ValueError(
                "stream_stride, stream_followup_stride, max_batch_size must be > 0"
            )
        if (
            stream_overlap_frames < 0
            or stream_crossfade_samples < 0
            or max_batch_wait_ms < 0
        ):
            raise ValueError(
                "stream_overlap_frames, stream_crossfade_samples, "
                "max_batch_wait_ms must be >= 0"
            )

        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self._codec = codec
        self._device = torch.device(device)
        self._num_codebooks = int(num_codebooks)
        # Samples per code frame: 24 kHz / 75 Hz code rate = 320.
        # ``hop_length`` is a property on HiggsAudioV2TokenizerConfig.
        self._frame_length = int(codec.model.config.hop_length)
        self._stride = int(stream_stride)
        self._followup_stride = int(stream_followup_stride)
        self._overlap_frames = int(stream_overlap_frames)
        self._crossfade_samples = int(stream_crossfade_samples)
        self._max_batch_size = int(max_batch_size)
        self._max_batch_wait_s = float(max_batch_wait_ms) / 1000.0
        self._running = False
        self._pending_messages: collections.deque[IncomingMessage] = collections.deque()
        self._payloads: dict[str, StagePayload] = {}
        self._stream_states: dict[str, _StreamState] = {}
        self._pending_done: set[str] = set()
        self._aborted_request_ids: set[str] = set()

    # ----- lifecycle -----------------------------------------------------

    def start(self) -> None:
        self._running = True
        while self._running:
            msg = self._next_message()
            if msg is None:
                continue
            if msg.request_id in self._aborted_request_ids:
                continue
            try:
                if msg.type == "new_request":
                    self._handle_new_request_batch(
                        self._collect_new_request_batch(msg)
                    )
                elif msg.type == "stream_chunk":
                    # Pipeline runtime translates upstream ``type="stream"``
                    # outbox messages into ``stream_chunk`` on the receiving
                    # scheduler's inbox; ``msg.data`` is a ``StreamItem``.
                    self._on_chunk(msg.request_id, msg.data.data)
                elif msg.type == "stream_done":
                    self._on_done(msg.request_id)
                else:
                    raise ValueError(f"Unsupported vocoder message: {msg.type}")
            except Exception as exc:
                logger.exception(
                    "HiggsStreamingVocoderScheduler failed for %s", msg.request_id
                )
                self.outbox.put(
                    OutgoingMessage(
                        request_id=msg.request_id, type="error", data=exc
                    )
                )
                self.abort(msg.request_id)

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        self._aborted_request_ids.add(request_id)
        if len(self._aborted_request_ids) > _ABORTED_REQUEST_ID_LIMIT:
            excess = len(self._aborted_request_ids) - _ABORTED_REQUEST_ID_RETAINED
            for rid in list(self._aborted_request_ids)[:excess]:
                self._aborted_request_ids.discard(rid)
        self._payloads.pop(request_id, None)
        self._stream_states.pop(request_id, None)
        self._pending_done.discard(request_id)

    # ----- message handlers ---------------------------------------------

    def _next_message(self) -> IncomingMessage | None:
        if self._pending_messages:
            return self._pending_messages.popleft()
        try:
            return self.inbox.get(timeout=0.1)
        except _queue_mod.Empty:
            return None

    def _collect_new_request_batch(
        self, first_msg: IncomingMessage
    ) -> list[IncomingMessage]:
        batch = [first_msg]
        if self._max_batch_size <= 1 or self._is_streaming_payload(first_msg.data):
            return batch
        deadline = time.monotonic() + self._max_batch_wait_s
        while len(batch) < self._max_batch_size:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    msg = self.inbox.get(timeout=remaining)
                except _queue_mod.Empty:
                    break
            if msg.request_id in self._aborted_request_ids:
                continue
            if msg.type == "new_request" and not self._is_streaming_payload(msg.data):
                batch.append(msg)
            else:
                self._pending_messages.append(msg)
                break
        return batch

    def _handle_new_request_batch(self, batch: list[IncomingMessage]) -> None:
        streaming: list[IncomingMessage] = []
        non_streaming: list[IncomingMessage] = []
        for msg in batch:
            (streaming if self._is_streaming_payload(msg.data) else non_streaming).append(
                msg
            )
        for msg in streaming:
            self._on_streaming_new_request(msg.request_id, msg.data)
        if non_streaming:
            self._vocode_non_streaming_batch(non_streaming)

    def _on_streaming_new_request(
        self, request_id: str, payload: StagePayload
    ) -> None:
        self._aborted_request_ids.discard(request_id)
        self._payloads[request_id] = payload
        self._stream_states.setdefault(request_id, _StreamState())
        if request_id in self._pending_done:
            self._pending_done.discard(request_id)
            self._on_done(request_id)

    def _on_chunk(self, request_id: str, codes: Any) -> None:
        if request_id in self._aborted_request_ids:
            return
        state = self._stream_states.setdefault(request_id, _StreamState())
        if not isinstance(codes, torch.Tensor):
            codes = torch.as_tensor(codes, dtype=torch.long)
        chunk = build_stream_chunk(
            state,
            codes,
            codec=self._codec,
            num_codebooks=self._num_codebooks,
            frame_length=self._frame_length,
            stride=self._stride,
            followup_stride=self._followup_stride,
            overlap_frames=self._overlap_frames,
            crossfade_samples=self._crossfade_samples,
        )
        if chunk is not None and request_id not in self._aborted_request_ids:
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=chunk,
                    metadata={"modality": "audio"},
                )
            )

    def _on_done(self, request_id: str) -> None:
        if request_id in self._aborted_request_ids:
            return
        if request_id not in self._stream_states:
            return
        if request_id not in self._payloads:
            # ``stream_done`` arrived before the matching ``new_request``;
            # defer the flush.
            self._pending_done.add(request_id)
            return
        state = self._stream_states[request_id]
        chunk = flush_stream_chunk(
            state,
            codec=self._codec,
            num_codebooks=self._num_codebooks,
            frame_length=self._frame_length,
            overlap_frames=self._overlap_frames,
            crossfade_samples=self._crossfade_samples,
        )
        if chunk is not None and request_id not in self._aborted_request_ids:
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=chunk,
                    metadata={"modality": "audio"},
                )
            )
        # Emit the terminal ``result`` so the OpenAI SSE handler closes
        # out the stream with a finish_reason chunk. We piggyback the
        # full state (which includes usage / engine_time_s) onto it.
        payload = self._payloads.get(request_id)
        if payload is None or request_id in self._aborted_request_ids:
            return
        result = self._finalize_streaming_payload(payload)
        self.outbox.put(
            OutgoingMessage(request_id=request_id, type="result", data=result)
        )
        self._payloads.pop(request_id, None)
        self._stream_states.pop(request_id, None)

    # ----- non-streaming (one-shot) path --------------------------------

    def _vocode_non_streaming_batch(self, batch: list[IncomingMessage]) -> None:
        for msg in batch:
            if msg.request_id in self._aborted_request_ids:
                continue
            try:
                result = self._vocode_full(msg.data)
            except Exception as exc:
                self.outbox.put(
                    OutgoingMessage(
                        request_id=msg.request_id, type="error", data=exc
                    )
                )
                continue
            self.outbox.put(
                OutgoingMessage(
                    request_id=msg.request_id, type="result", data=result
                )
            )

    def _vocode_full(self, payload: StagePayload) -> StagePayload:
        """One-shot reverse_delay + decode — parity with the original
        ``stages.create_vocoder_executor`` implementation."""
        state = HiggsTtsState.from_dict(payload.data)
        delayed_rows = state.output_codes_delayed
        sample_rate = self._codec.SAMPLE_RATE
        out_data = dict(payload.data)
        if not delayed_rows:
            out_data["audio_data"] = []
            out_data["sample_rate"] = sample_rate
            out_data["modality"] = "audio"
            return StagePayload(
                request_id=payload.request_id, request=payload.request, data=out_data
            )

        delayed_LN = torch.tensor(delayed_rows, dtype=torch.long)
        if delayed_LN.shape[0] < self._num_codebooks:
            out_data["audio_data"] = []
            out_data["sample_rate"] = sample_rate
            out_data["modality"] = "audio"
            return StagePayload(
                request_id=payload.request_id, request=payload.request, data=out_data
            )

        codec_vocab = self._codec.model.config.codebook_size
        data_TN = reverse_delay_pattern(delayed_LN)
        data_TN = torch.where(
            data_TN >= codec_vocab, torch.zeros_like(data_TN), data_TN
        )
        waveform = self._codec.decode(data_TN)
        audio_np = waveform.detach().to(torch.float32).cpu().numpy()
        out_data["audio_data"] = audio_np.tolist()
        out_data["sample_rate"] = sample_rate
        out_data["modality"] = "audio"
        if state.prompt_tokens or state.completion_tokens or state.engine_time_s:
            usage = {
                "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
                "total_tokens": state.prompt_tokens + state.completion_tokens,
            }
            if state.engine_time_s:
                usage["engine_time_s"] = round(state.engine_time_s, 6)
            out_data["usage"] = usage
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=out_data
        )

    def _finalize_streaming_payload(self, payload: StagePayload) -> StagePayload:
        """Wrap the streaming-only state into a terminal ``result`` payload
        (no audio_data — the per-chunk stream messages already carried it).
        Carries usage so the SSE handler can emit its terminal frame."""
        state = HiggsTtsState.from_dict(payload.data)
        out_data = dict(payload.data)
        out_data["sample_rate"] = self._codec.SAMPLE_RATE
        out_data["modality"] = "audio"
        out_data["audio_data"] = []
        if state.prompt_tokens or state.completion_tokens or state.engine_time_s:
            usage = {
                "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
                "total_tokens": state.prompt_tokens + state.completion_tokens,
            }
            if state.engine_time_s:
                usage["engine_time_s"] = round(state.engine_time_s, 6)
            out_data["usage"] = usage
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=out_data
        )

    @staticmethod
    def _is_streaming_payload(payload: StagePayload) -> bool:
        return bool((payload.request.params or {}).get("stream"))


__all__ = [
    "HiggsStreamingVocoderScheduler",
    "build_stream_chunk",
    "flush_stream_chunk",
]
