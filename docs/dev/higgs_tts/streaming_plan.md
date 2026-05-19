# Higgs TTS — Streaming Output (Dynamic Frame Accumulation)

Tracking doc for adding HTTP-streaming `audio/speech` output to Higgs
TTS, using the dynamic frame accumulation pattern Baseten describes
for Qwen3-TTS. **Status: plan + Stage 0 probe only.**

## Why

Today `/v1/audio/speech` with Higgs TTS is request/response: client
waits for the entire AR + vocoder pipeline before any audio comes
back. Time-to-first-audio (TTFA) at C=1 is ~1.1 s (Stage 4 with
CUDA Graph). For interactive voice agents this needs to drop into
the 300–500 ms range.

Two known wins, both compatible with the CUDA Graph path:

1. **Per-step code emission**: AR engine emits one code row each
   decode step into a stream channel instead of buffering until EOC.
2. **Dynamic frame accumulation in the vocoder**: vocoder decodes a
   small first chunk (~10 frames, ~133 ms audio) for low TTFA, then
   ramps up to large chunks (~90 frames, ~1.2 s audio) so the codec
   forward stays well-batched as concurrency rises.

sglang-omni's `fishaudio_s2_pro` model **already implements this exact
pattern** (`streaming_vocoder.py:S2ProVocoderScheduler`, 627 LOC). The
work below ports that template to Higgs TTS, with one
Higgs-specific complication: the delay-pattern codec.

## Existing pieces we don't need to touch

- **HTTP layer**: `/v1/audio/speech?stream=true` → `_speech_stream`
  SSE handler (`serve/openai_api.py:528`) already iterates
  `client.generate(...)`, encodes each chunk with `encode_audio`
  (WAV / MP3 / PCM), and emits SSE.
- **Pipeline IPC**: stage→stage `type="stream"` messages with a
  `target=` field already route to a named downstream stage; relay /
  coordinator handle it. `S2ProVocoderScheduler` is proof.
- **`reverse_delay_pattern`**: in `utils.py`, unit-tested. Recovers
  `[T, N]` data codes from `[T + N − 1, N]` delayed codes.

## What changes

```
┌───────────────┐  per-step      ┌────────────────┐  type=stream
│ HiggsTTSModel │── 1 delayed ──▶│ HiggsScheduler │── target=vocoder
│  (AR decode)  │   code row     │                │   data=codes[1, N]
└───────────────┘  [N, 1]        └────────────────┘
                                          │
                                          ▼
                                 ┌────────────────────┐ type=stream
                                 │ HiggsStreaming     │── data={audio,
                                 │ VocoderScheduler   │        sample_rate}
                                 └────────────────────┘
                                          │
                                          ▼
                                  /v1/audio/speech
                                  (SSE chunks)
```

## Stages

### Stage 0 — Codec chunkability probe ⏳

**Why first**: highest-pull/lowest-cost question. If Higgs's
codec decoder has a large receptive field or non-stationary
behaviour, chunked decode produces audible artifacts at boundaries
and the whole streaming design has to fall back to "wait-N-then-stream"
(degenerate streaming). Knowing this up front sets the rest of the
parameter sweep.

**Tool**: `_perf_bench/probe_codec_chunking.py`

Algorithm:

1. Load codec via `HiggsAudioCodec.from_pretrained`.
2. For N=5 reference WAVs (use `seed-tts/en` clips):
   - `codes_TN = codec.encode_reference(wav)` → `[T, 8]`.
   - `wav_full = codec.decode(codes_TN)` → one-shot baseline.
   - For each `(stride, overlap)` in a sweep grid:
     - Decode in chunks of `stride + overlap` data frames.
     - Drop the first `overlap * frame_length` samples of each
       chunk (those overlap the previous chunk's tail).
     - Optionally apply linear-crossfade on chunk boundaries.
     - Concatenate.
3. Compare `wav_chunked` vs `wav_full`:
   - per-sample MSE
   - peak abs diff
   - count of edges exceeding `peak_diff > 0.05` in any
     10-sample window (proxy for clicks).
4. Report a table; recommend `(stride, overlap, crossfade_samples)`.

**Accept**: there exists `(stride=10, overlap, crossfade)` with
chunked-vs-full MSE < 1e-4 and zero click edges. If not, escalate
the overlap until it does, or fall back to "wait N-frame warmup
before going streaming" mode.

### Stage 1 — Per-step code emission

**Files**: `model_runner.py`, `request_builders.py`

- `HiggsSGLangRequestData` gains `latest_stream_code_chunk: torch.Tensor | None = None`.
- `_collect_step_outputs_cg` (decode path, the captured one) writes
  `data.latest_stream_code_chunk = codes_N.unsqueeze(0)` (shape
  `[1, N]`) each step, alongside the existing
  `data.output_codes.append`. Prefill path mirrors it.

**Accept**: decode of a 50-step request leaves `latest_stream_code_chunk`
present on 50 individual scheduler ticks (snapshot test).

### Stage 2 — Scheduler emit

**Files**: `higgs_scheduler.py`

Add `_emit_stream_chunk(request)` modeled on
`fish_scheduler.py:427`:

```python
def _emit_stream_chunk(self, request):
    if not payload.request.params.get("stream"):
        return
    codes = data.latest_stream_code_chunk
    if codes is None:
        return
    self.outbox.put(OutgoingMessage(
        request_id=request.request_id, type="stream",
        data=codes, target="vocoder",
        metadata={"modality": "audio_codes"},
    ))
    data.latest_stream_code_chunk = None
```

Call from `HiggsScheduler.update()` after `iteration_controller.update_request`,
before the finish check. On finish, emit `type="stream_done"`.

**Accept**: pcap-equivalent — server log shows N stream messages
+ 1 stream_done per streaming request, before the result message.

### Stage 3 — Streaming vocoder

**Files**: new `higgs_streaming_vocoder.py` (~700 LOC), close port
of `streaming_vocoder.py`.

Key differences from s2_pro:

1. **Delay pattern**: state holds `delayed_codes_LN`, accumulated
   one row per AR step. To emit `stride` data frames, need
   `total_delayed >= last_data_emitted + stride + N - 1` rows.

2. **Decode**: inside the trigger,
   `data_codes_TN = reverse_delay_pattern(delayed_codes_LN[start:])`
   then call `codec.decode`.

3. **`stream_overlap_tokens` and `stream_crossfade_samples`** —
   pin to whatever Stage 0 recommends.

4. **trim**: drop emitted data + delay-window rows after each chunk
   to bound memory.

**Accept**: single-request streaming output, codes-domain MSE vs
batch path is 0 (same codes), audio-domain MSE matches Stage 0's
chunked-vs-full result.

### Stage 4 — Pipeline wiring

**Files**: `stages.py`, `config.py`

- Unify vocoder stage: same scheduler handles both streaming
  (`type=stream` / `type=stream_done`) and batch (`type=new_request`)
  paths. Non-stream requests route through the original batch
  decode path internally; only stream requests use the per-chunk
  accumulator.
- `stages.py:create_vocoder_executor` takes the new scheduler.
- `config.py` exposes `stream_stride`, `stream_followup_stride`,
  `stream_overlap_tokens`, `stream_crossfade_samples` as
  `factory_args`.

**Accept**: existing N=100 batch bench unchanged; new streaming
request returns multiple audio chunks via SSE.

### Stage 5 — Test + tuning

**Files**: `tests/test_higgs_tts_streaming.py`,
`_perf_bench/bench_streaming.py`.

- Streaming smoke: open `stream=True`, record per-chunk wall
  timestamps + sample counts. Verify:
  - TTFA p50 ≤ 500 ms
  - audio_s/s C=1 ≥ 3.0 (Stage 4 baseline was 3.55)
  - first chunk size ≈ `stream_stride * frame_length` samples
- Cross-check: concatenated streamed audio vs single-shot WAV from
  same prompt + seed → sample MSE on the same range as Stage 0
  predicted.
- Sweep grid: `stream_stride ∈ {4, 8, 16}`,
  `stream_followup_stride ∈ {32, 64, 128}`; pick TTFA p95 vs
  throughput pareto knee.

**Accept**: TTFA p50 < 500 ms and no audible artifacts on listening
test of 5 outputs.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Higgs codec chunked-decode has audible boundary artifacts | **Medium** | Stage 0 measures this directly; fallback is wait-N-then-stream |
| `overlap` value needed is too large (≥40 frames) for low TTFA | Low | Stage 0 sweep finds the smallest viable value |
| `stream` message firehose stalls coordinator at high C | Low | s2_pro proves this works at C=16+ |
| CUDA Graph + streaming interaction | Very low | streaming only adds scheduler / vocoder code paths; the captured forward is untouched |
| MP3 streaming doesn't work (mp3 chunks aren't independently decodable) | Low | Use PCM or OGG/Opus for streaming response_format; document the limitation |

## Out of scope

- WebSocket / Realtime API (`sglang_omni/serve/realtime/`).
- Smarter chunk sizing (e.g. adaptive based on observed concurrency).
- Stop-mid-stream / abort handling beyond what s2_pro already does.

## Timeline

| Stage | Days | Cumulative |
|---|---|---|
| 0. Codec probe | 0.5 | 0.5 |
| 1. Per-step emission | 0.5 | 1 |
| 2. Scheduler emit | 0.5 | 1.5 |
| 3. Streaming vocoder | 2 | 3.5 |
| 4. Pipeline wiring | 0.5 | 4 |
| 5. Test + tuning | 1 | 5 |
