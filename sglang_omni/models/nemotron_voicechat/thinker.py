from __future__ import annotations

import torch
from torch import nn

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.nemotron_h import NemotronHForCausalLM
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix

from sglang_omni.models.nemotron_voicechat.fusion import AddFusion

FUNCTION_HEAD_KEY = "stt_model.function_head.weight"
BACKBONE_RENAMES_MAP = (
    ("stt_model.llm.", "backbone."),
    ("stt_model.embed_tokens.", "backbone.embeddings."),
    ("stt_model.lm_head.", "lm_head."),
)

class NemotronVoiceChatForCausalLM(nn.Module):
    def __init__(self, *, config, quant_config, prefix):
        super().__init__()
        self.llm = NemotronHForCausalLM(config=config, quant_config=quant_config, prefix=add_prefix("llm", prefix))
        self.function_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("function_head", prefix),
        )
        self.fusion = AddFusion(config.duplex)

    def get_input_embeddings(self) -> nn.Module:
        return self.llm.get_input_embeddings()

    def forward(self, input_ids, positions, forward_batch, input_embeds, **omni_kwargs):
        del omni_kwargs
        return self.llm(input_ids, positions, forward_batch, input_embeds)

    def _backbone_weights_stream(self, parameters, weights):
        # Drop RNN weights from the stream.
        for name, weight in weights:
            if name == FUNCTION_HEAD_KEY:
                parameter = parameters["function_head.weight"]
                default_weight_loader(parameter, weight)
                continue
            for source, target in BACKBONE_RENAMES_MAP:
                if name.startswith(source):
                    yield target + name[len(source) :], weight
                    break

    def load_weights(self, weights):
        parameters = dict(self.named_parameters())
        self.llm.load_weights(self._backbone_weights_stream(parameters, weights))

EntryClass = NemotronVoiceChatForCausalLM