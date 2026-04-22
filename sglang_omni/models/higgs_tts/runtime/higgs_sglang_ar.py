# SPDX-License-Identifier: Apache-2.0
"""Higgs TTS sglang runtime: ``ResourceManager`` / ``IterationController`` /
``ModelRunner`` adapters that plug :class:`HiggsTTSModel` (with the
forward-embedded multi-codebook decode from PR4c-i) into
:class:`sglang_omni.engines.omni.engine.OmniEngine`.

Mirrors ``fishaudio_s2_pro/runtime/s2pro_sglang_ar.py`` at a high level;
simpler because Higgs has no RAS / repetition-penalty / Fish-specific
semantic-token range — the PR4a sampler + forward-embedded hooks in
:class:`HiggsTTSModel` already contain all the TTS-specific logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.mem_cache.common import release_kv_cache

from sglang_omni.engines.omni.runtime.sglang_ar import (
    SGLangARRequestData,
    SGLangBatchPlanner,
    SGLangResourceManager,
)
from sglang_omni.engines.omni.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)
from sglang_omni.models.higgs_tts.tokenizer import AUDIO_PLACEHOLDER_ID

if TYPE_CHECKING:
    from sglang_omni.engines.ar.sglang_backend.model_worker import ModelWorker
    from sglang_omni.models.higgs_tts.model import HiggsTTSModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-request data carried through the OmniEngine scheduler
# ---------------------------------------------------------------------------


@dataclass
class HiggsSGLangRequestData(SGLangARRequestData):
    """Per-request state: prompt, ref-audio codes, generation params,
    accumulated output codes."""

    reference_codes_delayed: torch.Tensor | None = None
    """Ref-audio codes AFTER delay pattern, shape
    ``[num_ref_tokens, num_codebooks]``. Consumed by the model runner
    during prefill to overlay onto ``-100`` placeholder positions."""

    num_ref_codes_consumed: int = 0
    """Count of rows from ``reference_codes_delayed`` already overlaid onto
    earlier prefill chunks. Advanced by the model runner; lets chunked
    prefill slice the codes tensor correctly across multiple extend calls."""

    num_codebooks: int = 8
    codebook_size: int = 1026

    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None

    output_codes: list[torch.Tensor] = field(default_factory=list)
    """One ``[num_codebooks]`` long tensor per decode step (captured from
    the model's ``_slots[req_id].output_codes`` during update_request)."""


@dataclass
class HiggsStepOutput:
    """Per-step output surfaced from :class:`HiggsSGLangModelRunner`."""

    codes: torch.Tensor  # shape [num_codebooks]
    generation_done: bool


# ---------------------------------------------------------------------------
# IterationController: decides when a request is finished
# ---------------------------------------------------------------------------


class HiggsSGLangIterationController:
    def __init__(self, tree_cache: Any, max_new_tokens: int = 2048) -> None:
        self.tree_cache = tree_cache
        self._max_new_tokens = max_new_tokens

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        data: HiggsSGLangRequestData = request.data
        req = data.req
        step_out: HiggsStepOutput | None = output.data

        if req.is_chunked > 0:
            output.data = None
            req.is_chunked -= 1
            return

        if step_out is None:
            return

        data.output_codes.append(step_out.codes.clone())
        # Feed codebook-0 back as the next step's input token; the model's
        # forward() will detect decode mode and replace embed_tokens(cb0)
        # with ``fused_embedding(state.last_codes)`` internally.
        cb0 = int(step_out.codes[0].item())
        if cb0 >= 0:
            req.output_ids.append(cb0)

        if not req.finished() and req.decode_batch_idx == 0:
            self.tree_cache.cache_unfinished_req(req)

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        data: HiggsSGLangRequestData = request.data
        if data.req.is_chunked > 0:
            return False

        step_out: HiggsStepOutput | None = output.data
        if step_out is not None and step_out.generation_done:
            return True

        max_tok = data.max_new_tokens or self._max_new_tokens
        return len(data.output_codes) >= max_tok


# ---------------------------------------------------------------------------
# ModelRunner: wires prefill embedding overlay + reads codes from model buffer
# ---------------------------------------------------------------------------


class HiggsSGLangModelRunner:
    """Drives one forward pass per scheduler step.

    Prefill:
        Computes ``text_embeds = embed_tokens(input_ids)``, then for each
        request with ``reference_codes_delayed`` set, overlays
        ``fused_embedding(ref_codes)`` at the ``-100`` placeholder
        positions (matches PR3a's prompt layout). Passes the result as
        ``input_embeds`` to the model's forward, bypassing the safe-id
        fallback for ``-100``.

    Decode:
        No overlay here — :class:`HiggsTTSModel` detects decode mode and
        uses each slot's ``last_codes`` via the fused embedding. We just
        set the request ids on the forward batch so the model can route
        the per-row slot lookup correctly.

    After forward:
        Reads the newly appended code row from
        ``model.get_output_codes(req_id)[-1]`` and surfaces it as a
        :class:`HiggsStepOutput` per request.
    """

    def __init__(
        self,
        model_worker: "ModelWorker",
        batch_planner: SGLangBatchPlanner,
    ) -> None:
        self.model_worker = model_worker
        self.batch_planner = batch_planner

    @property
    def _model(self) -> "HiggsTTSModel":
        return self.model_worker.model_runner.model

    # -- Prefill embedding overlay ------------------------------------------
    def _inject_ref_audio_prefill(
        self,
        model_worker_batch: Any,
        scheduler_output: SchedulerOutput,
    ) -> None:
        device = model_worker_batch.input_ids.device
        model = self._model
        embed_tokens = model.backbone.model.embed_tokens
        fused = model.multimodal_embedding.modality_embedding_0

        input_ids = model_worker_batch.input_ids
        # ``embed_tokens`` would OOB on -100; substitute 0 before embed,
        # then overwrite at placeholder positions with fused_embedding.
        placeholder_mask = input_ids == AUDIO_PLACEHOLDER_ID
        safe_ids = torch.where(placeholder_mask, torch.zeros_like(input_ids), input_ids)
        text_embeds = embed_tokens(safe_ids)

        offset = 0
        for sched_req in scheduler_output.requests:
            data: HiggsSGLangRequestData = sched_req.data
            req_len = data.req.extend_input_len
            end = offset + req_len

            if (
                data.reference_codes_delayed is not None
                and data.reference_codes_delayed.numel() > 0
            ):
                # Slice the placeholder mask to this request's window.
                full_mask = placeholder_mask[offset:end]
                n_placeholders = int(full_mask.sum().item())
                if n_placeholders > 0:
                    codes = data.reference_codes_delayed.to(
                        device=device, dtype=torch.long
                    )
                    # ``num_ref_codes_consumed`` tracks how many ref-code rows
                    # were already overlaid on previous (chunked) prefill
                    # calls. ``len(req.prefix_indices)`` is the wrong base
                    # here because it counts ALL cached tokens (text +
                    # placeholders), not just placeholder tokens.
                    consumed = data.num_ref_codes_consumed
                    if codes.shape[0] < consumed + n_placeholders:
                        logger.warning(
                            "reference_codes_delayed too short for req %s "
                            "(have %d rows, already consumed %d, need %d more); "
                            "skipping overlay",
                            sched_req.request_id,
                            codes.shape[0],
                            consumed,
                            n_placeholders,
                        )
                    else:
                        codes_slice = codes[consumed : consumed + n_placeholders]
                        fused_embeds = fused(codes_slice)  # [n, D]
                        mask_idx = full_mask.nonzero(as_tuple=True)[0] + offset
                        text_embeds[mask_idx] = fused_embeds.to(text_embeds.dtype)
                        data.num_ref_codes_consumed = consumed + n_placeholders

            offset = end

        model_worker_batch.input_embeds = text_embeds

    # -- Per-step ----------------------------------------------------------
    def _attach_req_ids(
        self, model_worker_batch: Any, scheduler_output: SchedulerOutput
    ) -> None:
        # HiggsTTSModel.forward reads ``forward_batch.req_ids`` to route the
        # per-row slot lookup. sglang's ForwardBatch doesn't populate this
        # field natively; we piggy-back on the worker batch so the
        # ``ForwardBatch.init_new`` conversion carries it through.
        model_worker_batch.req_ids = [
            sched_req.request_id for sched_req in scheduler_output.requests
        ]

    def _build_outputs(
        self, scheduler_output: SchedulerOutput
    ) -> dict[str, RequestOutput]:
        model = self._model
        outputs: dict[str, RequestOutput] = {}
        for sched_req in scheduler_output.requests:
            data: HiggsSGLangRequestData = sched_req.data
            if data.req.is_chunked > 0:
                outputs[sched_req.request_id] = RequestOutput(
                    request_id=sched_req.request_id,
                    data=None,
                    finished=False,
                )
                continue
            slot = model._slots.get(sched_req.request_id)
            if slot is None or not slot.output_codes:
                outputs[sched_req.request_id] = RequestOutput(
                    request_id=sched_req.request_id,
                    data=None,
                    finished=False,
                )
                continue
            codes_N = slot.output_codes[-1]
            outputs[sched_req.request_id] = RequestOutput(
                request_id=sched_req.request_id,
                data=HiggsStepOutput(
                    codes=codes_N,
                    generation_done=slot.sampler.generation_done,
                ),
                finished=False,
            )
        return outputs

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        schedule_batch = scheduler_output.batch_data
        model_worker_batch = schedule_batch.get_model_worker_batch()

        if schedule_batch.forward_mode.is_extend():
            self._inject_ref_audio_prefill(model_worker_batch, scheduler_output)

        self._attach_req_ids(model_worker_batch, scheduler_output)

        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.model_worker.model_runner
        )
        # Carry req_ids onto the live ForwardBatch so HiggsTTSModel.forward
        # can read them. ``ForwardBatch.init_new`` copies a fixed set of
        # fields — anything not in that set falls through as an attribute
        # assignment, which is what we want here.
        forward_batch.req_ids = model_worker_batch.req_ids

        self.model_worker.forward_batch_generation(forward_batch)
        self.batch_planner.record_last_batch(schedule_batch)

        # Advance sglang's scheduler: populate ``schedule_batch.output_ids``
        # with codebook-0 per request so the next step's worker batch sees
        # non-None ``input_ids``. Mirrors s2_pro's
        # ``schedule_batch.output_ids = text_model._output_semantic_ids[:bs]``.
        model = self._model
        cb0_ids: list[int] = []
        for sched_req in scheduler_output.requests:
            slot = model._slots.get(sched_req.request_id)
            if slot is None or not slot.output_codes:
                cb0_ids.append(0)
            else:
                cb0_ids.append(int(slot.output_codes[-1][0].item()))
        if cb0_ids:
            schedule_batch.output_ids = torch.tensor(
                cb0_ids,
                dtype=torch.long,
                device=model_worker_batch.input_ids.device,
            )

        outputs = self._build_outputs(scheduler_output)
        req_ids = [req.request_id for req in scheduler_output.requests]
        req_id_to_index = {rid: idx for idx, rid in enumerate(req_ids)}
        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )


# ---------------------------------------------------------------------------
# ResourceManager: KV-cache + per-request slot cleanup
# ---------------------------------------------------------------------------


class HiggsSGLangResourceManager(SGLangResourceManager):
    def __init__(
        self,
        token_to_kv_pool_allocator,
        req_to_token_pool,
        tree_cache,
        model: "HiggsTTSModel",
    ) -> None:
        super().__init__(token_to_kv_pool_allocator, req_to_token_pool, tree_cache)
        self._model = model

    def free(self, request: SchedulerRequest) -> None:
        data: HiggsSGLangRequestData = request.data
        release_kv_cache(data.req, self.tree_cache)
        self._model.reset_request(request.request_id)


__all__ = [
    "HiggsSGLangIterationController",
    "HiggsSGLangModelRunner",
    "HiggsSGLangRequestData",
    "HiggsSGLangResourceManager",
    "HiggsStepOutput",
]
