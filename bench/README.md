# Higgs TTS parity benchmarks

Tools for measuring the quality gap between our `sglang-omni` Higgs TTS pipeline
and boson-vllm 0.14's reference serving of the SAME checkpoint.

## Setup

Run a boson-vllm server on the Higgs branch:

```bash
# In a container with vllm 0.14 installed + PYTHONPATH pointing at a
# boson-vllm checkout on branch higgs-m4-v0.14.1:
PYTHONPATH=/path/to/boson-vllm-higgs-m4 python3 -m vllm.entrypoints.openai.api_server \
    --model /hot-data/checkpoints/TTS/c66596d6cde44fb4ba8076cc2dc1e77c/step_35500/ \
    --served-model-name tts --host 0.0.0.0 --port 8015 \
    --trust-remote-code --enable-mm-embeds --enforce-eager \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.5
```

## Scripts

- `compare_boson_vs_sglang.py` — end-to-end WER: both stacks on the same
  seed-tts-eval English samples (same ref audio, same target text,
  matched temperature / top_k / seed), decoded through the same
  `HiggsAudioCodec` and whispered through `faster-whisper large-v3`.

  ```bash
  python -m bench.compare_boson_vs_sglang \
      --boson-base http://172.17.0.1:8015 \
      --n 100 --temperature 0.8 --top-k 50
  ```

- `diff_codes.py` — greedy (temperature=0, top_k=1) per-step code diff between
  stacks. Expected post-fix behaviour: rows 0-2 byte-identical; rows 3+ may
  drift by 1-ULP bf16 noise across layers (accumulated flashinfer / GEMM
  numerics), but argmax agreement is strong.

- `diff_step1_prompt.py` — verifies our prompt token stream matches boson-vllm's
  post-expansion token stream modulo the `-100` vs `<|ref_audio|>` placeholder
  (both are overlaid identically with the fused ref-audio embedding).

- `diff_step3_fused_embed.py` — loads the fused multi-codebook embedding two
  ways (our audio_encoder stage's loader vs. a boson-style direct load) and
  runs the SAME delayed ref-audio codes through both. Asserts weight + forward
  are bitwise identical and round-tripping through `state.reference_audio_embed`
  preserves bf16 values exactly.

- `inspect_cb0_eoc.py` — monkey-patches `decode_codebooks_batch` to log the
  top-10 cb0 candidates and EOC probability at each AR step — useful when
  truncation is suspected.

## Current findings (2026-04-24)

100 seed-tts-eval English samples, temperature 0.8, top_k 50:

| | boson-vllm 0.14 | sglang-omni |
|---|---|---|
| average WER | **0.0638** | **0.0559** |
| median WER | 0.0 | 0.0 |

**Parity achieved.** The gap is within Whisper + multinomial-sampling noise —
per-sample outputs differ (different random draws; we use `torch.multinomial`
while vLLM uses the Gumbel-max trick) but the population-level WER is
indistinguishable.

### Root cause (fixed)

Higgs checkpoints ship `text_config.rope_theta = null`. When realised via
`Qwen3Config(**dict)`, transformers silently falls back to its Qwen3 default of
**10000** — but the Higgs TTS checkpoint was trained with `rope_theta = 1e6`
(the Qwen3 convention). boson-vllm patches this via
`set_default_rope_theta(config, default_theta=1000000)`; sglang's init uses
`getattr(config, "rope_theta", 1000000)`, which does *not* trigger on an
explicit `None`. Result: our backbone ran with the wrong rotary base, silently
wrecking positional encoding.

Symptom: seed-tts WER 0.586 vs boson-vllm's 0.017 at matched params; greedy
codes diverge starting row 2 despite matching row 0-1.

Fix lives in
[`sglang_omni/models/higgs_tts/hf_config.py`](../sglang_omni/models/higgs_tts/hf_config.py)
— when the text sub-config's `rope_theta` is `None`, rewrite to `1_000_000`
before constructing `Qwen3Config`. See
`tests/test_higgs_tts_model.py::test_text_config_patches_null_rope_theta_to_qwen3_default`
for the regression test.

Two additional 1-ULP bf16 drifts uncovered along the way were also fixed:

- Audio encoder stage loader now materialises the fused embedding in bf16
  *before* the weight copy ([`pipeline/stages.py`](../sglang_omni/models/higgs_tts/pipeline/stages.py)),
  so the encoder stage's forward (and `state.reference_audio_embed`) matches
  the engine's inline bf16 compute bit-for-bit.
- `HiggsTTSModel`'s `multimodal_embedding` / `modality_head` are now cast
  to the backbone's dtype in `__init__`
  ([`model.py`](../sglang_omni/models/higgs_tts/model.py)); previously the
  default `nn.Parameter(torch.empty(...))` kept them in fp32 and the
  decode-step re-embed accumulated 1 bf16 ULP of drift per step.
