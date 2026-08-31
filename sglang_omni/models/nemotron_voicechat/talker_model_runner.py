from __future__ import annotations

import torch
from transformers import AutoTokenizer

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)

NUM_ITER = 8


def char_vocab_from_tokenizer(tokenizer) -> dict[str, int]:
    vocab = tokenizer.get_vocab()
    chars = sorted((token for token in vocab if len(token) == 1), key=lambda t: vocab[t])
    return {char: index for index, char in enumerate(chars)}


class NemotronVoiceChatTalkerModelRunner(ModelRunner):
    def __init__(self, tp_worker, output_processor):
        super().__init__(tp_worker, output_processor)
        speech = self.model.config.nemotron_speech
        self.tokenizer = AutoTokenizer.from_pretrained(speech["tokenizer_name"])
        self.char_vocab = char_vocab_from_tokenizer(self.tokenizer)
        self.char_padding_idx = len(self.char_vocab)
        self.text_pad_id = self.tokenizer.get_vocab()["<unk>"]
        self.text_eos_id = int(self.tokenizer.eos_token_id)
        self.exponent = float(speech["tts_config"]["exponent"])
        self.top_p = float(speech["inference_top_p_or_k"])
        self.noise_scale = float(speech["inference_noise_scale"])
        self.force_silence = bool(speech["inference_force_speech_silence_on_eos"])
        self.speech_pad_id = int(speech["codec_config"]["codebook_size"])
        self._warmup_rows = None

    def _device(self):
        return self.model._fusion_buffer.device

    def _char_batch(self, token_ids: list[int]):
        device = self._device()
        sequences = [
            [self.char_vocab[c] for c in (self.tokenizer.convert_ids_to_tokens(t) or "") if c in self.char_vocab]
            or [self.char_padding_idx]
            for t in token_ids
        ]
        width = max(len(s) for s in sequences)
        char_ids = torch.full((len(sequences), width), self.char_padding_idx, dtype=torch.long, device=device)
        for row, sequence in enumerate(sequences):
            char_ids[row, : len(sequence)] = torch.tensor(sequence, device=device)
        lengths = torch.tensor([len(s) for s in sequences], device=device)
        return torch.tensor(token_ids, device=device), char_ids, lengths

    def _warmup(self):
        if self._warmup_rows is None:
            model = self.model
            talker = model.talker
            frames = model.audio_prompt_latent.shape[0]
            silence_1Q = model.codec_silence_tokens.unsqueeze(0)
            audio = torch.cat([
                model.audio_prompt_latent[:-1],
                talker.embed_codes(silence_1Q) + talker.bos_emb,
            ])
            ids, chars, lengths = self._char_batch(
                [self.text_pad_id] * (frames - 1) + [self.text_eos_id]
            )
            mask = torch.zeros(frames, dtype=torch.bool, device=self._device())
            mask[frames - 2:] = True
            text = talker.embed_subword(ids, chars, lengths, mask)
            self._warmup_rows = talker.gated_fusion_audio_text(audio, text)
        return self._warmup_rows

    def before_prefill(self, forward_batch, schedule_batch, requests) -> None:
        del schedule_batch
        rows = self._warmup()
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=torch.cat([rows] * len(requests), dim=0),
                input_embeds_are_projected=True,
            ),
        )

    def post_prefill(self, result, forward_batch, schedule_batch, requests) -> None:
        del result, forward_batch, schedule_batch
        for request in requests:
            inputs = request.data.talker_model_inputs
            inputs["prev_codes"] = torch.full(
                (1, self.model.talker.num_quantizers),
                self.speech_pad_id, dtype=torch.long, device=self._device(),
            )
            inputs["codes_rows"] = []

    def is_decode_batch_ready(self, schedule_batch) -> bool:
        return all(len(req._omni_data.pending_text_queue) > 0 for req in schedule_batch.reqs)

    def before_decode(self, forward_batch, schedule_batch, requests, *, is_lookahead=False) -> None:
        del forward_batch, schedule_batch, is_lookahead
        model = self.model
        talker = model.talker
        rows = []
        for request in requests:
            data = request.data
            inputs = data.talker_model_inputs
            token = data.pending_text_queue.popleft()
            if self.force_silence and token == self.text_eos_id:
                inputs["prev_codes"] = model.codec_silence_tokens.unsqueeze(0)
            audio_1D = talker.embed_codes(inputs["prev_codes"])
            ids, chars, lengths = self._char_batch([token])
            text_1D = talker.embed_subword(ids, chars, lengths)
            rows.append(talker.gated_fusion_audio_text(audio_1D, text_1D))
        batch = len(rows)
        model._fusion_buffer[:batch] = torch.cat(rows, dim=0)
        model._fusion_mask[:batch] = True

    def post_decode(self, result, forward_batch, schedule_batch, requests) -> None:
        del result, forward_batch, schedule_batch
        model = self.model
        for index, request in enumerate(requests):
            inputs = request.data.talker_model_inputs
            hidden = model._hidden_out[index:index + 1]
            codes = model.talker.generate_codes(
                hidden, model.mog_head,
                num_iter=NUM_ITER, exponent=self.exponent,
                top_p=self.top_p, noise_scale=self.noise_scale,
            )
            inputs["prev_codes"] = codes
            inputs["codes_rows"].append(codes[0])
            inputs["stream_chunk"] = codes.cpu()
