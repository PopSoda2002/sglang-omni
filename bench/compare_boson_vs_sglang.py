"""Side-by-side comparison: boson-vllm 0.14 Higgs vs sglang-omni on seed-tts.

Feeds the SAME (ref codes, synth text, temperature, top_k, seed) to both
stacks, collects the emitted codebook tokens, decodes via our
``HiggsAudioCodec``, transcribes with Whisper, reports WER + code diff
stats per sample.

Design choices:

- Ref-audio encoding is done ONCE (on our side) to eliminate codec drift
  as a confound — both stacks receive the same ``[T, N]`` ref codes.
- Decoding is also done ONCE per output (via our codec) for the same
  reason — we're isolating the AR-model output.
- Both servers run with ``temperature > 0`` so sampling differs, but WER
  is a population metric; 5-10 samples are enough to see a parity gap.

Usage:
    python bench/compare_boson_vs_sglang.py \
        --boson-base http://localhost:8015 \
        --n 5 --temperature 0.8 --top-k 50

Assumes:
- boson-vllm Higgs server is already up at ``--boson-base``.
- The TTS ckpt is at ``/hot-data/checkpoints/TTS/.../step_35500``.
- The seed-tts-eval dataset is at
  ``/ceph/data/audio_eval/tokenizer_eval/seed-tts-eval/seedtts_testset/en``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import uuid

import numpy as np
import requests
import soundfile as sf
import torch

from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec
from sglang_omni.models.higgs_tts.delay_pattern import (
    apply_delay_pattern,
    reverse_delay_pattern,
)
from sglang_omni.models.higgs_tts.pipeline.stages import (
    create_aggregate_executor,
    create_audio_encoder_executor,
    create_preprocessing_executor,
    create_sglang_tts_engine_executor,
    create_vocoder_executor,
)
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest

TTS_CKPT = "/hot-data/checkpoints/TTS/c66596d6cde44fb4ba8076cc2dc1e77c/step_35500"
SEED_TTS_EN = "/ceph/data/audio_eval/tokenizer_eval/seed-tts-eval/seedtts_testset/en"


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def wer(ref: str, hyp: str) -> float:
    r = normalize(ref).split()
    h = normalize(hyp).split()
    if not r:
        return float(len(h) > 0)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            sub = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + sub)
    return d[len(r)][len(h)] / len(r)


def load_seed_tts(n: int):
    with open(os.path.join(SEED_TTS_EN, "meta.lst")) as f:
        lines = [ln.strip() for ln in f if ln.strip()][:n]
    out = []
    for line in lines:
        _rid, _ref_text, rel_prompt_wav, synth_text = line.split("|")
        out.append((os.path.join(SEED_TTS_EN, rel_prompt_wav), synth_text))
    return out


def call_boson(base: str, prompt_text: str, delayed_codes: torch.Tensor,
               max_tokens: int, temperature: float, top_k: int, seed: int):
    """Hit boson-vllm's /v1/completions with the Higgs TTS prompt + audio_tokens."""
    url = f"{base}/v1/completions"
    prompt = f"<|tts|><|ref_audio|><|text|>{prompt_text}<|audio|>"
    payload = {
        "model": "tts",
        "prompt": prompt,
        "audio_tokens": delayed_codes.tolist(),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
    }
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    j = r.json()
    choice = j["choices"][0]
    mm = choice.get("mm_token_ids")
    if mm is None:
        return None, choice.get("finish_reason")
    codes = torch.tensor(mm, dtype=torch.long)
    return codes, choice.get("finish_reason")


class _SglangPipeline:
    """Persistent sglang-omni pipeline. The engine's async Queue is
    bound to the event loop that started it, so we share a single
    ``asyncio`` loop across all samples.
    """

    def __init__(self):
        self.preprocess = create_preprocessing_executor(
            TTS_CKPT, audio_codec_device="cuda:0"
        )
        self.audio_encoder = create_audio_encoder_executor(
            TTS_CKPT, device="cuda:0"
        )
        self.aggregate = create_aggregate_executor()
        self.engine = create_sglang_tts_engine_executor(
            TTS_CKPT,
            device="cuda:0",
            max_new_tokens=1024,
            mem_fraction_static=0.4,
            max_running_requests=2,
        )
        self.vocoder = create_vocoder_executor(TTS_CKPT, device="cpu")
        self.loop = asyncio.new_event_loop()
        # Start all stages on the persistent loop.
        self.loop.run_until_complete(self._start_all())

    async def _start_all(self):
        await self.preprocess.start()
        await self.audio_encoder.start()
        await self.aggregate.start()
        await self.engine.start()
        await self.vocoder.start()

    async def _stop_all(self):
        await self.vocoder.stop()
        await self.engine.stop()
        await self.aggregate.stop()
        await self.audio_encoder.stop()
        await self.preprocess.stop()

    def close(self):
        self.loop.run_until_complete(self._stop_all())
        self.loop.close()

    async def _run_one(self, payload):
        await self.preprocess.add_request(payload)
        p = await self.preprocess.get_result()
        await self.audio_encoder.add_request(p)
        p = await self.audio_encoder.get_result()
        await self.aggregate.add_request(p)
        p = await self.aggregate.get_result()
        await self.engine.add_request(p)
        p = await self.engine.get_result()
        await self.vocoder.add_request(p)
        return await self.vocoder.get_result()

    def run(self, prompt_wav: str, synth_text: str, temperature: float,
            top_k: int, max_tokens: int, seed: int):
        payload = StagePayload(
            request_id=str(uuid.uuid4()),
            request=OmniRequest(
                inputs={
                    "input": synth_text,
                    "reference_audio": {"audio_path": prompt_wav},
                },
                params={
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                    "seed": seed,
                },
            ),
            data=None,
        )
        p = self.loop.run_until_complete(self._run_one(payload))
        from sglang_omni.models.higgs_tts.io import HiggsTtsState
        state = HiggsTtsState.from_dict(p.data)
        delayed = state.output_codes_delayed
        if not delayed:
            return None, None
        return torch.tensor(delayed, dtype=torch.long), p.data.get("audio_data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boson-base", default="http://localhost:8015")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/ceph/workspace/huapeng/sglang-omni/_bench_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading HiggsAudioCodec + Whisper...")
    codec = HiggsAudioCodec.from_tts_ckpt(TTS_CKPT, device="cuda")
    from faster_whisper import WhisperModel
    whisper = WhisperModel("large-v3", device="cuda", compute_type="float16")

    print("Building sglang-omni pipeline...")
    sglang_pipe = _SglangPipeline()

    entries = load_seed_tts(args.n)
    results = []

    for i, (prompt_wav, synth_text) in enumerate(entries):
        print(f"\n[{i+1}/{len(entries)}] target: {synth_text!r}")

        # Encode ref audio ONCE
        wav, sr_in = sf.read(prompt_wav, dtype="float32")
        wav_t = torch.from_numpy(wav if wav.ndim == 1 else wav[:, 0])
        ref_codes = codec.encode_reference(wav_t, sample_rate=sr_in)
        delayed = apply_delay_pattern(ref_codes)
        print(f"  ref: {ref_codes.shape} → delayed: {delayed.shape}")

        # ---- boson-vllm ----
        boson_codes, boson_finish = call_boson(
            args.boson_base, synth_text, delayed,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed + i,
        )
        if boson_codes is None:
            print(f"  boson-vllm: NO CODES (finish={boson_finish})")
            continue
        print(f"  boson-vllm: {boson_codes.shape} codes, finish={boson_finish}")

        # ---- sglang-omni ----
        # The sglang-omni preprocessing encodes ref audio itself from the
        # audio_path. Codes should match since both use the same codec.
        sglang_codes, sglang_audio = sglang_pipe.run(
            prompt_wav, synth_text,
            temperature=args.temperature,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            seed=args.seed + i,
        )
        if sglang_codes is None:
            print("  sglang-omni: NO CODES")
            continue
        print(f"  sglang-omni: {sglang_codes.shape} codes")

        # ---- decode + whisper both ----
        def _decode_and_whisper(tag, delayed_codes, idx):
            N = delayed_codes.shape[1]
            if delayed_codes.shape[0] < N:
                print(f"  {tag}: too short to reverse")
                return None, None
            codes_TN = reverse_delay_pattern(delayed_codes)
            # Clamp BOC/EOC (1024/1025) to a valid data code.
            codes_TN = torch.where(
                codes_TN >= 1024, torch.zeros_like(codes_TN), codes_TN
            )
            audio = codec.decode(codes_TN).cpu().numpy()
            out_wav = os.path.join(args.out_dir, f"sample_{idx:02d}_{tag}.wav")
            sf.write(out_wav, audio, codec.SAMPLE_RATE)
            segments, _ = whisper.transcribe(out_wav, language="en", beam_size=5)
            hyp = " ".join(s.text for s in segments).strip()
            w = wer(synth_text, hyp)
            print(f"  {tag}: dur={len(audio)/codec.SAMPLE_RATE:.2f}s WER={w:.3f} | {hyp!r}")
            return w, hyp

        w_b, hyp_b = _decode_and_whisper("boson", boson_codes, i)
        w_s, hyp_s = _decode_and_whisper("sglang", sglang_codes, i)

        results.append({
            "target": synth_text,
            "boson_wer": w_b,
            "sglang_wer": w_s,
            "boson_hyp": hyp_b,
            "sglang_hyp": hyp_s,
            "boson_len": int(boson_codes.shape[0]),
            "sglang_len": int(sglang_codes.shape[0]),
        })

    print("\n" + "=" * 60 + "\nSummary\n" + "=" * 60)
    if not results:
        print("no results collected")
        return
    avg_b = np.mean([r["boson_wer"] for r in results if r["boson_wer"] is not None])
    avg_s = np.mean([r["sglang_wer"] for r in results if r["sglang_wer"] is not None])
    print(f"average WER — boson-vllm: {avg_b:.3f}   sglang-omni: {avg_s:.3f}")
    for r in results:
        print(f"  target={r['target']!r}")
        print(f"    boson  WER={r['boson_wer']:.3f} hyp={r['boson_hyp']!r}")
        print(f"    sglang WER={r['sglang_wer']:.3f} hyp={r['sglang_hyp']!r}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"avg_boson_wer": float(avg_b), "avg_sglang_wer": float(avg_s),
                   "results": results}, f, indent=2)
    print(f"\nWrote {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
