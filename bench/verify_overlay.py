"""Critical diagnostic: verify that our prefill overlay actually reaches the model.

We wrap the runtime's ``_inject_ref_audio_prefill`` to log what
``model_worker_batch.input_embeds`` we set, then wrap sglang's
``Qwen2Model.forward`` to log what ``input_embeds`` it actually receives.
If these differ, our overlay isn't reaching the model.
"""

from __future__ import annotations

import asyncio
import os
import uuid

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


def main():
    with open(os.path.join(SEED_TTS_EN, "meta.lst")) as f:
        line = f.readline().strip()
    _rid, _rt, rel, synth = line.split("|")
    prompt_wav = os.path.join(SEED_TTS_EN, rel)

    # Hook 1: what our runtime sets as input_embeds
    from sglang_omni.models.higgs_tts.runtime import higgs_sglang_ar

    _orig_inject = higgs_sglang_ar.HiggsSGLangModelRunner._inject_ref_audio_prefill

    def _logged_inject(self, model_worker_batch, scheduler_output):
        _orig_inject(self, model_worker_batch, scheduler_output)
        # After: log the embed we set
        embed = model_worker_batch.input_embeds
        if embed is not None:
            print(
                f"[inject] set input_embeds: shape={tuple(embed.shape)}, "
                f"dtype={embed.dtype}, device={embed.device}, "
                f"norm_per_pos={embed.norm(dim=-1)[:5].tolist()}...",
                flush=True,
            )
        else:
            print(f"[inject] input_embeds is None", flush=True)

    higgs_sglang_ar.HiggsSGLangModelRunner._inject_ref_audio_prefill = _logged_inject

    # Hook 2: what Qwen2Model.forward receives as input_embeds
    from sglang.srt.models import qwen2 as _qwen2

    _orig_forward = _qwen2.Qwen2Model.forward

    def _logged_forward(
        self, input_ids, positions, forward_batch, input_embeds=None, **kw
    ):
        if input_embeds is not None:
            norms = input_embeds.norm(dim=-1)
            print(
                f"[backbone.forward] input_embeds: shape={tuple(input_embeds.shape)}, "
                f"dtype={input_embeds.dtype}, norm_per_pos[:5]={norms[:5].tolist()}...",
                flush=True,
            )
        else:
            # embed_tokens(input_ids) path
            print(
                f"[backbone.forward] input_embeds=None, input_ids shape={tuple(input_ids.shape)}, "
                f"min={int(input_ids.min())}, max={int(input_ids.max())}",
                flush=True,
            )
        return _orig_forward(
            self, input_ids, positions, forward_batch, input_embeds, **kw
        )

    _qwen2.Qwen2Model.forward = _logged_forward

    preprocess = create_preprocessing_executor(TTS_CKPT, audio_codec_device="cuda:0")
    audio_encoder = create_audio_encoder_executor(TTS_CKPT, device="cuda:0")
    aggregate = create_aggregate_executor()
    engine = create_sglang_tts_engine_executor(
        TTS_CKPT,
        device="cuda:0",
        max_new_tokens=4,
        mem_fraction_static=0.35,
        max_running_requests=2,
    )

    payload = StagePayload(
        request_id=str(uuid.uuid4()),
        request=OmniRequest(
            inputs={"input": synth, "reference_audio": {"audio_path": prompt_wav}},
            params={"max_new_tokens": 4, "temperature": 0.8, "top_k": 50, "seed": 42},
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
            await engine.get_result()
        finally:
            for s in (engine, aggregate, audio_encoder, preprocess):
                await s.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
