"""Step 1: are our prompt token ids the same as boson-vllm's?

We build: [<|tts|>, <|ref_audio|>, -100*T_ref, <|text|>, ...text..., <|audio|>]
Boson sends: prompt = "<|tts|><|ref_audio|><|text|>{text}<|audio|>", and the
server-side processor expands <|ref_audio|> → (T_ref+1) copies of <|ref_audio|>
where index 0 keeps its token embed and indices 1..T_ref get multimodal embeds.

Net input_ids the BACKBONE sees should be IDENTICAL between both stacks
(we substitute -100→0 in embed_tokens, but overlay fused embeds at the
-100 positions; boson keeps <|ref_audio|> at all (T_ref+1) positions
and mask-replaces indices 1..T_ref).
"""

from __future__ import annotations

import os

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

TTS_CKPT = "/hot-data/checkpoints/TTS/c66596d6cde44fb4ba8076cc2dc1e77c/step_35500"
SEED_TTS_EN = "/ceph/data/audio_eval/tokenizer_eval/seed-tts-eval/seedtts_testset/en"


def main():
    # Load first seed-tts entry
    with open(os.path.join(SEED_TTS_EN, "meta.lst")) as f:
        line = f.readline().strip()
    _rid, _rt, _rel, synth = line.split("|")
    T_ref = 105  # delayed ref length for sample 1 (98 raw + 7)

    # --- OUR prompt build (via HiggsTokenizerAdapter) ---
    raw = Tokenizer.from_file(os.path.join(TTS_CKPT, "tokenizer.json"))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=raw)

    from sglang_omni.models.higgs_tts.tokenizer import (
        AUDIO_PLACEHOLDER_ID,
        HiggsTokenizerAdapter,
    )

    adapter = HiggsTokenizerAdapter(hf_tok)
    our_ids = adapter.build_prompt(synth, num_ref_tokens=T_ref)

    # --- BOSON equivalent: tokenize the raw prompt string, then expand ---
    prompt_str = f"<|tts|><|ref_audio|><|text|>{synth}<|audio|>"
    boson_prompt_ids = hf_tok.encode(prompt_str, add_special_tokens=False)

    # After boson's _get_prompt_updates replacement:
    # the single <|ref_audio|> token gets expanded to (T_ref+1) copies,
    # with index 0 keeping as text embed and 1..T_ref getting multimodal.
    ref_audio_id = hf_tok.get_added_vocab()["<|ref_audio|>"]
    idx = boson_prompt_ids.index(ref_audio_id)
    boson_expanded = (
        boson_prompt_ids[:idx]
        + [ref_audio_id] * (T_ref + 1)
        + boson_prompt_ids[idx + 1 :]
    )

    print(f"target text: {synth!r}")
    print(f"T_ref: {T_ref}")
    print()
    print(f"our length: {len(our_ids)}")
    print(f"boson length (after expansion): {len(boson_expanded)}")
    print()

    # Diff
    if len(our_ids) != len(boson_expanded):
        print(f"LENGTH MISMATCH: {len(our_ids)} vs {len(boson_expanded)}")

    # Show first 5 and last 5 with labels
    def _label(tok_id):
        if tok_id == AUDIO_PLACEHOLDER_ID:
            return f"  {tok_id} (OUR PLACEHOLDER -100)"
        vocab = hf_tok.get_added_vocab()
        for name, i in vocab.items():
            if i == tok_id:
                return f"  {tok_id} ({name})"
        try:
            decoded = hf_tok.decode([tok_id])
            return f"  {tok_id} ({decoded!r})"
        except Exception:
            return f"  {tok_id} (?)"

    print("--- head (first 6) ---")
    for i in range(6):
        o = our_ids[i] if i < len(our_ids) else None
        b = boson_expanded[i] if i < len(boson_expanded) else None
        marker = "   " if o == b else "!! "
        print(f"{marker}pos {i}:  ours={_label(o):60s}  boson={_label(b)}")

    print()
    print("--- tail (last 6) ---")
    for off in range(-6, 0):
        o = our_ids[off]
        b = boson_expanded[off]
        marker = "   " if o == b else "!! "
        print(f"{marker}pos {off}:  ours={_label(o):60s}  boson={_label(b)}")

    # Full element-wise diff
    min_len = min(len(our_ids), len(boson_expanded))
    diffs = [i for i in range(min_len) if our_ids[i] != boson_expanded[i]]
    print()
    print(f"total diffs in first {min_len} positions: {len(diffs)}")
    for i in diffs[:5]:
        print(f"  pos {i}: ours={_label(our_ids[i])} boson={_label(boson_expanded[i])}")


if __name__ == "__main__":
    main()
