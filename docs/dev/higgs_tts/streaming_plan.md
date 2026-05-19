# Higgs TTS — Streaming Output (Dynamic Frame Accumulation)

Tracking doc for adding HTTP-streaming `audio/speech` output to Higgs
TTS, using the dynamic frame accumulation pattern Baseten describes
for Qwen3-TTS. **Status: plan + Stage 0 probe only.**

## Why

Today `/v1/audio/speech` with Higgs TTS is request/response: client
waits for the entire AR + vocoder pipeline before any audio comes
back. For interactive voice agents this needs to drop into the
300–500 ms range.

Two wins, both independent of the CUDA Graph capture work (this PR
keeps ``disable_cuda_graph: True`` for the AR engine — the CUDA
Graph capture lives on a separate branch and can flip independently
later):

1. **Per-step code emission**: AR engine emits one code row each
   decode step into a stream channel instead of buffering until EOC.
2. **Dynamic frame accumulation in the vocoder**: vocoder decodes a
   small first chunk for low TTFA, then ramps up to larger chunks so
   the codec forward stays well-batched as concurrency rises.

**Note on code rate.** Higgs's codec runs at **25 Hz**, not 75 Hz
(``hop_length = 960``, one code frame ≈ 40 ms of audio). At the
same chunk count a Higgs chunk is therefore **3× longer** in audio
than an equivalent s2_pro chunk — the Baseten-style "~10 / ~90"
split maps to "~5 / ~40" data frames for an equivalent TTFA budget,
and ``codec.decode(40 frames) → 1.6 s`` of audio.

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

### Stage 0 — Codec chunkability probe ✅ (negative result)

**Result**: 45-config sweep (stride ∈ {10, 20, 40}, overlap ∈ {0, 5,
10, 20, 40}, crossfade_samples ∈ {0, 256, 512}) × 5 seed-tts en
samples. **No configuration met the acceptance bar** (mse<1e-4
*and* zero click edges).

Best-of-sweep (lowest mse_mean):

  | stride | overlap | xfade | mse_mean | peak_diff | clicks_sum |
  |--------|---------|-------|----------|-----------|------------|
  |   40   |    0    |   0   | 6.74e-05 |   0.43    |    414     |
  |   20   |    0    |   0   | 1.84e-04 |   0.43    |   1 425    |
  |   10   |    0    |   0   | 6.69e-04 |   0.63    |   4 059    |

Two non-obvious findings:

1. **Overlap hurts.** Every config with ``overlap > 0`` has ~10×
   more clicks and ~10× larger MSE than ``overlap=0``. Feeding the
   decoder the prior chunk's tail as "context" makes it *worse* —
   strong signal that the codec re-initialises its internal state
   on every ``decode`` call, so the "context" is just producing a
   second, misaligned re-onset.

2. **Crossfade hurts too.** Linear crossfade across boundaries also
   moves MSE up by ~5×. Same root cause: boundary samples on either
   side of the join belong to two independent decoder
   re-initialisations and have no common phase / level.

The naive "chunked-decode + crossfade" streaming design is **not
viable** on Higgs's codec. Three real options remain:

- **(A) Re-decode growing prefix.** Stream chunk *k* by decoding
  ``codes[0 : (k+1)*stride]`` from scratch every step and emitting
  only ``audio[k*stride*FRAME : (k+1)*stride*FRAME]``. Quadratic
  compute over the full utterance, but every individual decode is
  small and well-batched. TTFA = decode(``stride`` codes) ≈ tens
  of ms even at stride=40 — still well under the 500 ms goal.

- **(B) Expose decoder hidden state.** Thread the codec decoder's
  internal state (conv cache, possibly RNN h/c) across calls.
  Codec-side change; may need retraining. Out of scope for this PR.

- **(C) "Wait-N then dump".** Emit one big chunk after the first
  N codes (≈ N × 13 ms audio after N × 13 ms of AR decode →
  same as today). Degenerate — gives up the TTFA win entirely.

**Recommendation: pursue (A).** The "quadratic" compute is bounded
in practice: typical max audio ≈ 10 s = 750 code frames; at a
``followup_stride`` of 64 we re-decode the growing prefix
~``T/followup_stride`` ≈ 12 times. Each individual decode is small
enough to fit easily in GPU memory, so the wall-clock multiplier vs
one-shot decode is ~12×, but spread across the streaming window
(not in front of TTFA).

The downstream stages keep the same shape but assume strategy (A)
rather than the original chunked-decode-with-overlap.

**Tool** (preserved for reproducibility):
`_perf_bench/probe_codec_chunking.py` — re-run when codec is
updated to re-confirm chunkability conclusion.

**Original acceptance criterion** (kept for context): a
``(stride, overlap, crossfade)`` config with mse<1e-4 and zero
click edges. **Closed unmet** — see above.

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
