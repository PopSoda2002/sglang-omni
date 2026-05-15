# SPDX-License-Identifier: Apache-2.0
"""sglang-native Higgs Multimodal Qwen3 TTS model.

Composes sglang's built-in :class:`sglang.srt.models.qwen3.Qwen3ForCausalLM`
as the text backbone with the fused multi-codebook embedding / head.
Registered in sglang's ``ModelRegistry`` under
``HiggsMultimodalQwen3ForConditionalGeneration`` via
:func:`sglang_omni.models.higgs_tts.bootstrap.register_higgs_tts_in_sglang`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple

import torch
from torch import nn

from sglang_omni.models.higgs_tts.hf_config import HiggsMultimodalQwen3Config
from sglang_omni.models.higgs_tts.modeling import (
    HiggsFusedMultiTextEmbedding,
    HiggsFusedMultiTextHead,
)
from sglang_omni.models.higgs_tts.sampler import STOP_CODE, HiggsSamplerState
from sglang_omni.models.higgs_tts.sampler import step as sampler_step
from sglang_omni.models.higgs_tts.weight_loader import DiscreteWeightMapper

# Prefix map for the text backbone. Since we compose :class:`Qwen3ForCausalLM`
# under ``self.backbone``, all text-backbone weights live under
# ``backbone.model.*`` (and ``backbone.lm_head.*``). :func:`load_weights` then
# strips the ``backbone.`` prefix and hands the remaining names to
# ``Qwen3ForCausalLM.load_weights``, which handles qkv / gate_up stacking and
# tied-lm-head logic.
_BACKBONE_PREFIX_MAP: dict[str, str] = {
    "tied.embedding.text_embedding.": "backbone.model.embed_tokens.",
    "body.layers.": "backbone.model.layers.",
    "body.norm.": "backbone.model.norm.",
    "tied.head.text_head.": "backbone.lm_head.",
}


@dataclass
class HiggsGenParams:
    """Per-request decoding parameters consumed by :func:`sampler.step`."""

    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None


@dataclass
class _RequestSlot:
    """Per-request runtime bookkeeping inside :class:`HiggsTTSModel`.

    Kept flat so the engine runtime can construct / introspect one slot per
    live request without reaching into private state machinery.
    """

    sampler: HiggsSamplerState
    output_codes: list[torch.Tensor] = field(default_factory=list)
    """One ``[num_codebooks]`` long tensor per AR step, accumulated in
    generation order."""


class _HiggsMultimodalEmbedding(nn.Module):
    """Container matching the Higgs checkpoint layout.

    The checkpoint stores the fused embedding under
    ``multimodal_embedding.modality_embedding_0.*``; the container name
    ``multimodal_embedding`` is preserved so weight loading is a straight
    prefix substitution (see :class:`DiscreteWeightMapper`).
    """

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.modality_embedding_0 = HiggsFusedMultiTextEmbedding(
            num_codebooks=num_codebooks,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )


class HiggsTTSModel(nn.Module):
    """Higgs Multimodal Qwen3 model (discrete TTS path) adapted for sglang.

    Composition over :class:`sglang.srt.models.qwen3.Qwen3ForCausalLM` —
    the backbone handles paged attention, KV cache, logits processing and
    standard text weight loading. This wrapper adds:

    - ``multimodal_embedding.modality_embedding_0``: the fused
      :class:`HiggsFusedMultiTextEmbedding` (shape ``[N*V, D]``).
    - ``modality_head``: the fused :class:`HiggsFusedMultiTextHead`, tied
      to the embedding weight when ``audio_encoder_config.tie_word_embeddings``.
    - :meth:`load_weights` that remaps Higgs checkpoint names and splits
      the stream between the backbone and the multimodal modules.

    Multi-codebook input embedding overlay (the ``-100`` placeholder paste
    from the reference audio) is performed by the engine model_runner; this
    model just consumes the prepared ``input_embeds`` in its forward.
    """

    def __init__(
        self,
        config: HiggsMultimodalQwen3Config,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Late import so the test suite can exercise the registry entry
        # without eagerly importing sglang's heavyweight inference stack.
        from sglang.srt.models.qwen3 import Qwen3ForCausalLM

        self.config = config

        text_config = config.get_text_config()
        self.backbone = Qwen3ForCausalLM(
            text_config,
            quant_config=quant_config,
            prefix=prefix + "backbone" if prefix else "backbone",
        )

        enc_cfg = config.audio_encoder_config or {}
        encoder_type = enc_cfg.get("encoder_type", "discrete")
        if encoder_type != "discrete":
            raise NotImplementedError(
                f"HiggsTTSModel currently supports only the discrete "
                f"TTS path; got encoder_type={encoder_type!r}. Whisper/Qwen3-AUT "
                f"(ASR) encoders are planned for a future PR."
            )

        num_codebooks: int = int(enc_cfg["num_codebooks"])
        vocab_size: int = int(enc_cfg["vocab_size"])
        hidden_size: int = int(enc_cfg.get("out_dim", text_config.hidden_size))
        self._num_codebooks = num_codebooks
        self._codebook_vocab_size = vocab_size
        self._tie_modality = bool(enc_cfg.get("tie_word_embeddings", True))

        self.multimodal_embedding = _HiggsMultimodalEmbedding(
            num_codebooks=num_codebooks,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )
        self.modality_head = HiggsFusedMultiTextHead(
            num_codebooks=num_codebooks,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )
        # Match backbone dtype (bf16) for the fused embedding / head so the
        # decode-step re-embed runs in bf16 — torch's default fp32 would
        # accumulate ~1 ULP per step and compound across the AR loop.
        backbone_dtype = self.backbone.model.embed_tokens.weight.dtype
        self.multimodal_embedding.to(dtype=backbone_dtype)
        self.modality_head.to(dtype=backbone_dtype)
        if self._tie_modality:
            self.modality_head.weight = (
                self.multimodal_embedding.modality_embedding_0.weight
            )

        # Per-request runtime state for the multi-codebook decode; routed
        # per-row inside :meth:`forward` and accessed by the engine runtime
        # via :meth:`get_slot` / :meth:`reset_request`.
        self._slots: dict[str, _RequestSlot] = {}

    def get_input_embeddings(self) -> nn.Embedding:
        return self.backbone.get_input_embeddings()

    def get_multimodal_embedding(self) -> HiggsFusedMultiTextEmbedding:
        return self.multimodal_embedding.modality_embedding_0

    def get_modality_head(self) -> HiggsFusedMultiTextHead:
        return self.modality_head

    @property
    def num_codebooks(self) -> int:
        return self._num_codebooks

    @property
    def codebook_vocab_size(self) -> int:
        return self._codebook_vocab_size

    def get_slot(self, req_id: str) -> _RequestSlot:
        """Return the slot for ``req_id``, creating a fresh
        :class:`HiggsSamplerState` on first access."""
        slot = self._slots.get(req_id)
        if slot is None:
            slot = _RequestSlot(
                sampler=HiggsSamplerState(num_codebooks=self._num_codebooks)
            )
            self._slots[req_id] = slot
        return slot

    def reset_request(self, req_id: str) -> None:
        """Discard all decode state for ``req_id``. Call when a request
        terminates or is cancelled."""
        self._slots.pop(req_id, None)

    def get_output_codes(self, req_id: str) -> torch.Tensor:
        """Return accumulated delayed codes for ``req_id``, shape
        ``[num_steps, num_codebooks]`` (or ``[0, num_codebooks]`` if the
        request has not produced any steps yet)."""
        slot = self._slots.get(req_id)
        if slot is None or not slot.output_codes:
            return torch.empty(
                (0, self._num_codebooks),
                dtype=torch.long,
                device=self.multimodal_embedding.modality_embedding_0.weight.device,
            )
        return torch.stack(slot.output_codes, dim=0).to(torch.long)

    # Forward-embedded multi-codebook decode --------------------------------
    @torch.no_grad()
    def decode_codebooks_batch(
        self,
        hidden_states_BD: torch.Tensor,
        req_ids: list[str],
        gen_params: list[HiggsGenParams],
    ) -> torch.Tensor:
        """Sample multi-codebook tokens for one forward step of a batch.

        Called at the tail of :meth:`forward`. For each row ``b``:

        - Look up / create the request's :class:`_RequestSlot`.
        - Run :func:`modality_head.generate` on ``hidden_states_BD[b]`` to
          get multi-codebook logits, then :func:`sampler.step` to advance
          the state machine and emit ``[num_codebooks]`` codes.
        - Append real (non-``STOP_CODE``) code rows to the slot's
          ``output_codes`` list.

        Args:
            hidden_states_BD: Per-request last-token hidden states, shape
                ``[B, hidden_size]``.
            req_ids: List of ``B`` request ids.
            gen_params: List of ``B`` :class:`HiggsGenParams`.

        Returns:
            ``text_logits_BV`` of shape ``[B, text_vocab_size]``, a zero
            tensor. sglang's downstream sampler still runs over this
            shape (we need to return something well-formed), but its
            ``next_token_ids`` are **discarded** — the engine-stage
            runtime in ``runtime/higgs_sglang_ar.py`` overwrites
            ``schedule_batch.output_ids`` with each request's codebook-0
            directly from :attr:`_slots`. The real multi-codebook codes
            live in ``get_output_codes(req_id)``.
        """
        batch_size = hidden_states_BD.shape[0]
        if len(req_ids) != batch_size or len(gen_params) != batch_size:
            raise ValueError(
                f"batch size mismatch: hidden={batch_size}, "
                f"req_ids={len(req_ids)}, gen_params={len(gen_params)}"
            )

        # Multi-codebook logits for the whole batch in one fused matmul.
        # Keep hidden in its native dtype (matches fused_head.weight) then
        # cast the resulting logits to float32 for the sampler's softmax
        # numerical stability.
        logits_BNV = self.modality_head.generate(hidden_states_BD).to(torch.float32)

        for b in range(batch_size):
            slot = self.get_slot(req_ids[b])
            params = gen_params[b]
            codes_N = sampler_step(
                logits_BNV[b],
                slot.sampler,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
            )
            # Skip ``STOP_CODE`` sentinel rows (returned when the sampler is
            # called past ``generation_done``). In practice the runtime
            # removes finished requests before the next step, but guard
            # defensively so a stray call can't corrupt the output stream.
            if int(codes_N[0].item()) != STOP_CODE:
                slot.output_codes.append(codes_N.detach().to(torch.long))

        # Returned logits are structurally required by sglang's sampler
        # but semantically unused — the runtime writes the real cb0 into
        # ``schedule_batch.output_ids`` directly.
        text_vocab_size = self.backbone.config.vocab_size
        return torch.zeros(
            (batch_size, text_vocab_size),
            device=hidden_states_BD.device,
            dtype=torch.float32,
        )

    # Forward ----------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        """Forward-embedded multi-codebook decode.

        The backbone text path is bypassed — we call
        ``self.backbone.model(...)`` directly to get hidden states, then
        run :meth:`decode_codebooks_batch` at the tail to produce
        multi-codebook codes stored in per-request slots. sglang's
        scheduler sees a peaked text-vocab distribution pointing at each
        request's codebook-0; KV cache and batching continue to work.

        Decode-step input embedding: when ``input_embeds`` is not
        provided (sglang's decode path), each request's previous-step
        ``state.last_codes`` is routed through
        :class:`HiggsFusedMultiTextEmbedding` to produce the token embed.

        Prefill path: callers (the engine stage) must pre-compute
        ``input_embeds`` (text embed + ref-audio fused overlay at
        ``-100`` placeholder positions) and pass it through. This model
        does NOT itself do the prefill overlay — that would require
        in-forward access to per-request reference codes, which the
        engine stage already has on hand.

        Returns ``LogitsProcessorOutput(next_token_logits, hidden_states)``
        so sglang's downstream sampler can pick the codebook-0 primary
        token.
        """
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        # Resolve per-request metadata from the forward batch.
        req_ids, gen_params = self._extract_batch_metadata(forward_batch)

        # Decode-step input overlay: if the caller didn't supply embeddings,
        # construct them from each slot's ``last_codes`` via the fused
        # multi-codebook embedding. This is the inverse of the peaked-text
        # logits trick: sglang sampled a codebook-0 id last step, but we
        # actually want to embed the FULL N-codebook row from our state.
        if input_embeds is None and self._is_decode_step(forward_batch):
            input_embeds = self._decode_step_embeds(req_ids, input_ids)

        hidden_states = self.backbone.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
        )

        # Prune to last-token positions for prefill.
        if (
            hasattr(forward_batch, "forward_mode")
            and forward_batch.forward_mode.is_extend()
            and hasattr(forward_batch, "extend_seq_lens")
        ):
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            hidden_states_last = hidden_states[last_index]
        else:
            # Decode step: hidden_states is already [batch, hidden_size].
            hidden_states_last = hidden_states
            if hidden_states_last.ndim == 3:
                hidden_states_last = hidden_states_last[:, -1, :]

        # Multi-codebook decode + slot updates.
        text_logits_BV = self.decode_codebooks_batch(
            hidden_states_last, req_ids, gen_params
        )

        return LogitsProcessorOutput(
            next_token_logits=text_logits_BV,
            hidden_states=hidden_states_last,
        )

    # -- forward helpers -----------------------------------------------------
    @staticmethod
    def _is_decode_step(forward_batch) -> bool:
        mode = getattr(forward_batch, "forward_mode", None)
        if mode is None:
            return False
        is_decode = getattr(mode, "is_decode", None)
        return bool(is_decode()) if callable(is_decode) else False

    def _extract_batch_metadata(
        self, forward_batch
    ) -> tuple[list[str], list[HiggsGenParams]]:
        """Pull request ids and per-request sampling params out of a sglang
        ``ForwardBatch``. If the batch doesn't carry the expected fields
        (e.g. we're in a unit test with a dummy batch), fall back to
        placeholder ids ``req-0``..``req-{B-1}`` and default params.
        """
        req_ids_raw = getattr(forward_batch, "req_ids", None)
        batch_size = self._infer_batch_size(forward_batch)
        if req_ids_raw is None:
            req_ids = [f"req-{i}" for i in range(batch_size)]
        else:
            req_ids = [str(r) for r in req_ids_raw]

        # sglang exposes per-request sampling params via
        # ``forward_batch.sampling_info``. When present, thread those
        # through; otherwise fall back to defaults.
        sampling_info = getattr(forward_batch, "sampling_info", None)
        gen_params: list[HiggsGenParams] = []
        for b in range(batch_size):
            gen_params.append(self._gen_params_for_row(sampling_info, b))
        return req_ids, gen_params

    @staticmethod
    def _gen_params_for_row(sampling_info, row: int) -> HiggsGenParams:
        if sampling_info is None:
            return HiggsGenParams()

        def _pick(attr: str, default):
            val = getattr(sampling_info, attr, None)
            if val is None:
                return default
            try:
                return (
                    float(val[row].item())
                    if hasattr(val[row], "item")
                    else float(val[row])
                )
            except (TypeError, IndexError):
                return default

        return HiggsGenParams(
            temperature=_pick("temperatures", 1.0),
            top_p=_pick("top_ps", None),
            top_k=int(_pick("top_ks", 0)) or None,
        )

    @staticmethod
    def _infer_batch_size(forward_batch) -> int:
        for attr in ("batch_size", "extend_seq_lens", "seq_lens", "req_pool_indices"):
            val = getattr(forward_batch, attr, None)
            if val is None:
                continue
            if isinstance(val, int):
                return val
            if hasattr(val, "shape") and len(val.shape) > 0:
                return int(val.shape[0])
            try:
                return len(val)
            except TypeError:
                continue
        return 1

    def _decode_step_embeds(
        self, req_ids: list[str], input_ids: torch.Tensor
    ) -> torch.Tensor:
        """For each request, build the per-step embedding from
        ``slot.sampler.last_codes`` via the fused multi-codebook embedding.

        Falls back to the text embedding of ``input_ids`` for any request
        whose slot has no ``last_codes`` yet (e.g. the scheduler is
        sending us a token we haven't decoded for).
        """
        device = input_ids.device
        N = self._num_codebooks
        last_codes_stack: list[torch.Tensor] = []
        mask: list[bool] = []
        for rid in req_ids:
            slot = self._slots.get(rid)
            last = None if slot is None else slot.sampler.last_codes
            if last is None:
                # Placeholder zeros; will be masked out below.
                last_codes_stack.append(torch.zeros(N, dtype=torch.long, device=device))
                mask.append(False)
            else:
                last_codes_stack.append(last.to(device=device, dtype=torch.long))
                mask.append(True)
        codes_BN = torch.stack(last_codes_stack, dim=0)
        fused_embeds = self.multimodal_embedding.modality_embedding_0(
            codes_BN
        )  # [B, D]

        text_embeds = self.backbone.model.embed_tokens(input_ids)  # [B, D] or [B, 1, D]
        if text_embeds.ndim == 3:
            text_embeds = text_embeds[:, -1, :]

        mask_t = torch.tensor(mask, device=device).unsqueeze(-1)
        combined = torch.where(mask_t, fused_embeds.to(text_embeds.dtype), text_embeds)

        # sglang flattens decode tokens to ``[total_tokens, hidden_size]``;
        # return 2-D without any per-sequence dim.
        return combined

    # Weight loading ---------------------------------------------------------
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Remap Higgs checkpoint names then route to the right submodule.

        Returns the set of **own** parameter names this wrapper loaded (i.e.,
        the multimodal embedding and optionally the untied modality head).
        Text backbone weights are delegated to
        :meth:`sglang.srt.models.qwen3.Qwen3ForCausalLM.load_weights`, which
        performs its own stacking (qkv_proj, gate_up_proj) and tied-lm-head
        logic; those loads are not reflected in the returned set.
        """
        mapper = DiscreteWeightMapper(
            text_prefix_map=_BACKBONE_PREFIX_MAP,
            tie_modality=self._tie_modality,
        )

        backbone_weights: list[Tuple[str, torch.Tensor]] = []
        self_weights: list[Tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()
        own_names = self._own_param_names()

        for name, tensor in weights:
            mapped = mapper.map(name)
            if mapped is None:
                continue  # e.g., audio-tokenizer backbone, discarded by design
            if mapped.startswith("backbone."):
                # Hand off to sglang's Qwen3 loader with the ``backbone.`` prefix
                # stripped; it expects names starting with ``model.`` or ``lm_head.``.
                backbone_weights.append((mapped[len("backbone.") :], tensor))
            elif mapped in own_names:
                self_weights.append((mapped, tensor))
            # Names that survived remapping without a destination (e.g.,
            # `tied.head.modality_heads.0.weight` under tie_modality=True) are
            # dropped silently; the checkpoint value is redundant with the
            # tied embedding weight.

        # Delegate text backbone loading. Qwen3ForCausalLM handles qkv/gate_up
        # stacking and lm_head tying internally.
        self.backbone.load_weights(iter(backbone_weights))

        # Load fused embedding (and untied head, if applicable) directly.
        own_params = dict(self.named_parameters(remove_duplicate=False))
        for name, tensor in self_weights:
            param = own_params.get(name)
            if param is None:
                continue
            if param.shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: expected {tuple(param.shape)}, "
                    f"got {tuple(tensor.shape)}"
                )
            param.data.copy_(tensor.to(param.dtype))
            loaded.add(name)

        return loaded

    def _own_param_names(self) -> set[str]:
        """Names of parameters owned directly by this wrapper (not backbone)."""
        names: set[str] = set()
        for name, _ in self.named_parameters(remove_duplicate=False):
            if not name.startswith("backbone."):
                names.add(name)
        return names


__all__ = ["HiggsGenParams", "HiggsTTSModel"]
