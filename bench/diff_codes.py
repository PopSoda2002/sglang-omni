"""Tight diff: run boson-vllm and sglang-omni with greedy sampling on
the SAME single request, compare the code sequences step-by-step to
find the divergence point."""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid

import requests
import soundfile as sf
import torch

from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec
from sglang_omni.models.higgs_tts.delay_pattern import apply_delay_pattern
from sglang_omni.models.higgs_tts.pipeline.stages import (
    create_aggregate_executor,
    create_audio_encoder_executor,
    create_preprocessing_executor,
    create_sglang_tts_engine_executor,
)
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest

TTS_CKPT = "/hot-data/checkpoints/TTS/c66596d6cde44fb4ba8076cc2dc1e77c/step_35500"
SEED_TTS_EN = "/ceph/data/audio_eval/tokenizer_eval/seed-tts-eval/seedtts_testset/en"


def call_boson(base, text, delayed, max_tokens, temperature, top_k, seed):
    url = f"{base}/v1/completions"
    prompt = f"<|tts|><|ref_audio|><|text|>{text}<|audio|>"
    payload = {
        "model": "tts",
        "prompt": prompt,
        "audio_tokens": delayed.tolist(),
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
        return None
    return torch.tensor(mm, dtype=torch.long)


class _SglangPipeline:
    def __init__(self):
        self.preprocess = create_preprocessing_executor(
            TTS_CKPT, audio_codec_device="cuda:0"
        )
        self.audio_encoder = create_audio_encoder_executor(TTS_CKPT, device="cuda:0")
        self.aggregate = create_aggregate_executor()
        self.engine = create_sglang_tts_engine_executor(
            TTS_CKPT,
            device="cuda:0",
            max_new_tokens=1024,
            mem_fraction_static=0.35,
            max_running_requests=2,
        )
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self._start())

    async def _start(self):
        for s in (self.preprocess, self.audio_encoder, self.aggregate, self.engine):
            await s.start()

    async def _run(self, payload):
        await self.preprocess.add_request(payload)
        p = await self.preprocess.get_result()
        await self.audio_encoder.add_request(p)
        p = await self.audio_encoder.get_result()
        await self.aggregate.add_request(p)
        p = await self.aggregate.get_result()
        await self.engine.add_request(p)
        return await self.engine.get_result()

    def run(self, prompt_wav, text, temperature, top_k, max_tokens, seed):
        payload = StagePayload(
            request_id=str(uuid.uuid4()),
            request=OmniRequest(
                inputs={"input": text, "reference_audio": {"audio_path": prompt_wav}},
                params={
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                    "seed": seed,
                },
            ),
            data=None,
        )
        p = self.loop.run_until_complete(self._run(payload))
        from sglang_omni.models.higgs_tts.io import HiggsTtsState

        state = HiggsTtsState.from_dict(p.data)
        return (
            torch.tensor(state.output_codes_delayed, dtype=torch.long)
            if state.output_codes_delayed
            else None
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boson-base", default="http://172.17.0.1:8015")
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(os.path.join(SEED_TTS_EN, "meta.lst")) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    _rid, _rt, rel, synth = lines[args.sample_idx].split("|")
    prompt_wav = os.path.join(SEED_TTS_EN, rel)
    print(f"target: {synth!r}")
    print(f"ref wav: {prompt_wav}")

    codec = HiggsAudioCodec.from_tts_ckpt(TTS_CKPT, device="cuda")
    wav, sr_in = sf.read(prompt_wav, dtype="float32")
    wav_t = torch.from_numpy(wav if wav.ndim == 1 else wav[:, 0])
    ref_codes = codec.encode_reference(wav_t, sample_rate=sr_in)
    delayed = apply_delay_pattern(ref_codes)
    print(f"ref codes: {ref_codes.shape} → delayed {delayed.shape}")

    boson = call_boson(
        args.boson_base,
        synth,
        delayed,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(f"\nboson: {boson.shape if boson is not None else None}")
    if boson is not None:
        print("boson first 12 rows:")
        for i, row in enumerate(boson[:12].tolist()):
            print(f"  {i:3d}: {row}")

    pipe = _SglangPipeline()
    sglang = pipe.run(
        prompt_wav, synth, args.temperature, args.top_k, args.max_tokens, args.seed
    )
    print(f"\nsglang: {sglang.shape if sglang is not None else None}")
    if sglang is not None:
        print("sglang first 12 rows:")
        for i, row in enumerate(sglang[:12].tolist()):
            print(f"  {i:3d}: {row}")

    if boson is not None and sglang is not None:
        m = min(boson.shape[0], sglang.shape[0])
        print(f"\ndiff (first {m} rows):")
        first_diff = None
        for i in range(m):
            if not torch.equal(boson[i], sglang[i]):
                if first_diff is None:
                    first_diff = i
                if i - (first_diff or 0) < 5:
                    print(f"  row {i} DIFF")
                    print(f"    boson : {boson[i].tolist()}")
                    print(f"    sglang: {sglang[i].tolist()}")
        if first_diff is None:
            print(f"  identical for first {m} rows!")
        else:
            print(f"first divergence at row {first_diff} of min-length {m}")


if __name__ == "__main__":
    main()
