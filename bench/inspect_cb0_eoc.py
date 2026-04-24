"""Instrument sglang-omni to dump per-step cb0 logits (top-10 ids + probs).

Goal: for the failing-case samples at temp=0.8 top_k=50, show at each
AR step what the top candidates for cb0 are. If EOC (1025) rises near
top-1 early, that explains why our sampler emits it prematurely and
the generation truncates.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid

import soundfile as sf
import torch

from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec
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
EOC_ID = 1025
BOC_ID = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(os.path.join(SEED_TTS_EN, "meta.lst")) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    _rid, _rt, rel, synth = lines[args.sample_idx].split("|")
    prompt_wav = os.path.join(SEED_TTS_EN, rel)
    print(f"target: {synth!r}")
    print(f"ref wav: {prompt_wav}")

    # Monkey-patch decode_codebooks_batch to log top-10 cb0 logits per step.
    from sglang_omni.models.higgs_tts import model as model_mod
    from sglang_omni.models.higgs_tts.sampler import step as sampler_step

    orig = model_mod.HiggsTTSModel.decode_codebooks_batch
    call_count = {"n": 0}

    def wrapped(self, hidden_states_BD, req_ids, gen_params):
        # Compute logits inline so we can log them before sampler runs.
        logits_BNV = self.modality_head.generate(hidden_states_BD).to(torch.float32)
        n = call_count["n"]
        if n < 40:  # log first 40 steps
            cb0 = logits_BNV[0, 0]  # [V]
            probs = cb0.softmax(dim=-1)
            topv, topi = probs.topk(10)
            eoc_p = float(probs[EOC_ID])
            boc_p = float(probs[BOC_ID])
            top = [(int(i), round(float(v), 4)) for v, i in zip(topv.tolist(), topi.tolist())]
            print(f"  step {n:3d}: top10={top} eoc_p={eoc_p:.4f} boc_p={boc_p:.4f}")
        call_count["n"] = n + 1
        return orig(self, hidden_states_BD, req_ids, gen_params)

    model_mod.HiggsTTSModel.decode_codebooks_batch = wrapped

    preprocess = create_preprocessing_executor(TTS_CKPT, audio_codec_device="cuda:0")
    audio_encoder = create_audio_encoder_executor(TTS_CKPT, device="cuda:0")
    aggregate = create_aggregate_executor()
    engine = create_sglang_tts_engine_executor(
        TTS_CKPT, device="cuda:0", max_new_tokens=args.max_tokens,
        mem_fraction_static=0.35, max_running_requests=2,
    )

    payload = StagePayload(
        request_id=str(uuid.uuid4()),
        request=OmniRequest(
            inputs={"input": synth, "reference_audio": {"audio_path": prompt_wav}},
            params={"max_new_tokens": args.max_tokens, "temperature": args.temperature,
                    "top_k": args.top_k, "seed": args.seed},
        ),
        data=None,
    )

    async def run():
        for s in (preprocess, audio_encoder, aggregate, engine):
            await s.start()
        try:
            await preprocess.add_request(payload)
            p = await preprocess.get_result()
            await audio_encoder.add_request(p)
            p = await audio_encoder.get_result()
            await aggregate.add_request(p)
            p = await aggregate.get_result()
            await engine.add_request(p)
            return await engine.get_result()
        finally:
            for s in (engine, aggregate, audio_encoder, preprocess):
                await s.stop()

    result = asyncio.run(run())
    from sglang_omni.models.higgs_tts.io import HiggsTtsState
    state = HiggsTtsState.from_dict(result.data)
    print(f"\ntotal delayed rows: {len(state.output_codes_delayed)}")


if __name__ == "__main__":
    main()
