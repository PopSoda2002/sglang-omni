"""Step 3: does the fused-embedding output from the TTS ckpt match
between our audio_encoder stage and what boson-vllm would compute
inline?

We load the same fused-embedding weight (same safetensors key) on both
sides and run the SAME delayed-codes tensor through. Output should be
bitwise identical if the weight + forward math is the same.
"""

from __future__ import annotations

import json
import os
import sys

import soundfile as sf
import torch
from safetensors import safe_open

TTS_CKPT = "/hot-data/checkpoints/TTS/c66596d6cde44fb4ba8076cc2dc1e77c/step_35500"
SEED_TTS_EN = "/ceph/data/audio_eval/tokenizer_eval/seed-tts-eval/seedtts_testset/en"


def load_fused_ours(dtype):
    """Use our stage's loader."""
    from sglang_omni.models.higgs_tts.pipeline.stages import (
        _load_fused_embedding_from_tts_ckpt,
    )
    m = _load_fused_embedding_from_tts_ckpt(TTS_CKPT, device="cpu")
    return m.to(dtype=getattr(torch, dtype))


def load_fused_boson_style(dtype):
    """Re-implement boson-vllm's fused-embedding load: same nn.Module
    class (which we ported verbatim), same weight tensor pulled with
    ``default_weight_loader``-like direct copy."""
    from sglang_omni.models.higgs_tts.modeling import HiggsFusedMultiTextEmbedding
    from sglang_omni.models.higgs_tts.hf_config import HiggsMultimodalQwen3Config

    cfg = HiggsMultimodalQwen3Config.from_pretrained(TTS_CKPT)
    enc = cfg.audio_encoder_config
    N = int(enc["num_codebooks"])
    V = int(enc["vocab_size"])
    D = int(enc.get("out_dim", cfg.get_text_config().hidden_size))

    m = HiggsFusedMultiTextEmbedding(num_codebooks=N, vocab_size=V, hidden_size=D)

    key = "tied.embedding.modality_embeddings.0.embedding.weight"
    idx = json.load(open(os.path.join(TTS_CKPT, "model.safetensors.index.json")))
    shard = idx["weight_map"][key]
    with safe_open(os.path.join(TTS_CKPT, shard), framework="pt") as f:
        w = f.get_tensor(key)
    with torch.no_grad():
        m.weight.copy_(w.to(dtype=getattr(torch, dtype)))
    return m.to(dtype=getattr(torch, dtype)).eval()


def main():
    dtype = "bfloat16"

    a = load_fused_ours(dtype)
    b = load_fused_boson_style(dtype)

    # Weight identical?
    wa = a.weight.detach()
    wb = b.weight.detach()
    print(f"weight shapes: {tuple(wa.shape)}, {tuple(wb.shape)}")
    print(f"weight dtype:  {wa.dtype}, {wb.dtype}")
    print(f"weight bitwise equal: {torch.equal(wa, wb)}")

    # Same forward on the same delayed codes from sample 0
    from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec
    from sglang_omni.models.higgs_tts.delay_pattern import apply_delay_pattern

    codec = HiggsAudioCodec.from_tts_ckpt(TTS_CKPT, device="cuda")
    with open(os.path.join(SEED_TTS_EN, "meta.lst")) as f:
        line = f.readline().strip()
    _rid, _rt, rel, _synth = line.split("|")
    prompt_wav = os.path.join(SEED_TTS_EN, rel)
    wav, sr_in = sf.read(prompt_wav, dtype="float32")
    wav_t = torch.from_numpy(wav if wav.ndim == 1 else wav[:, 0])
    ref_codes = codec.encode_reference(wav_t, sample_rate=sr_in)
    delayed = apply_delay_pattern(ref_codes)  # [T_ref, N]
    print(f"delayed codes: shape={tuple(delayed.shape)} sum={int(delayed.sum())}")

    with torch.no_grad():
        out_a = a(delayed)
        out_b = b(delayed)
    print(f"forward output shapes: {tuple(out_a.shape)}, {tuple(out_b.shape)}")
    print(f"forward bitwise equal: {torch.equal(out_a, out_b)}")
    print(f"forward max abs diff:  {(out_a - out_b).abs().max().item()}")

    # The SYSTEM test: run the pipeline's _inject_ref_audio_prefill (or
    # read state.reference_audio_embed after audio_encoder stage)
    # on the same input and show that matches.
    import asyncio
    import uuid
    from sglang_omni.models.higgs_tts.io import HiggsTtsState
    from sglang_omni.models.higgs_tts.pipeline.stages import (
        create_audio_encoder_executor, create_preprocessing_executor,
    )
    from sglang_omni.proto import StagePayload
    from sglang_omni.proto.request import OmniRequest

    preprocess = create_preprocessing_executor(TTS_CKPT, audio_codec_device="cuda:0")
    audio_encoder = create_audio_encoder_executor(TTS_CKPT, device="cuda:0")

    payload = StagePayload(
        request_id=str(uuid.uuid4()),
        request=OmniRequest(
            inputs={"input": "hi", "reference_audio": {"audio_path": prompt_wav}},
            params={},
        ),
        data=None,
    )

    async def run():
        await preprocess.start(); await audio_encoder.start()
        try:
            await preprocess.add_request(payload)
            p = await preprocess.get_result()
            await audio_encoder.add_request(p)
            return await audio_encoder.get_result()
        finally:
            await audio_encoder.stop(); await preprocess.stop()

    p = asyncio.run(run())
    st = HiggsTtsState.from_dict(p.data)

    # Check the CODES first — if these differ, the rest cascades.
    stage_codes = torch.tensor(st.reference_codes_delayed, dtype=torch.long)
    print(f"stage codes shape: {tuple(stage_codes.shape)} sum={int(stage_codes.sum())}")
    print(f"direct codes shape: {tuple(delayed.shape)} sum={int(delayed.sum())}")
    print(f"codes bitwise equal: {torch.equal(stage_codes, delayed)}")
    if not torch.equal(stage_codes, delayed):
        d = (stage_codes != delayed).sum().item()
        print(f"  number of different entries: {d} / {stage_codes.numel()}")
        diffs = (stage_codes != delayed).nonzero(as_tuple=False)[:5]
        for row in diffs.tolist():
            print(f"    [{row[0]},{row[1]}] stage={int(stage_codes[row[0], row[1]])} direct={int(delayed[row[0], row[1]])}")

    stage_embed = torch.tensor(st.reference_audio_embed, dtype=torch.float32)
    print(f"stage.reference_audio_embed shape: {tuple(stage_embed.shape)}")
    # Compare to the direct fused forward above (cast to fp32 for fair compare)
    out_a_f32 = out_a.float()
    print(f"stage vs direct-forward bitwise equal: {torch.equal(stage_embed, out_a_f32)}")
    print(f"stage vs direct-forward max abs diff:  {(stage_embed - out_a_f32).abs().max().item()}")


if __name__ == "__main__":
    main()
