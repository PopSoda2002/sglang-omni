# SPDX-License-Identifier: Apache-2.0
"""Factory for a Higgs TTS sglang engine.

Analogous to ``fishaudio_s2_pro/factory.py::create_s2pro_sglang_engine``:
wires an :class:`sglang_omni.engines.omni.engine.OmniEngine` around sglang's
``ModelWorker`` (which loads :class:`HiggsTTSModel` via the sglang model
registry that PR2b set up) + Higgs-specific runtime adapters from
:mod:`sglang_omni.models.higgs_tts.runtime.higgs_sglang_ar`.

No CUDA graph handling — the model's forward-embedded decode runs eager.
"""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.engines.omni.engine import OmniEngine
from sglang_omni.engines.omni.scheduler import Scheduler
from sglang_omni.models.higgs_tts.runtime.higgs_sglang_ar import (
    HiggsSGLangIterationController,
    HiggsSGLangModelRunner,
    HiggsSGLangResourceManager,
)
from sglang_omni.models.sglang_registry import register_omni_models_in_sglang

logger = logging.getLogger(__name__)


def create_higgs_sglang_engine(
    server_args: Any,
    *,
    gpu_id: int = 0,
    max_new_tokens: int = 2048,
) -> OmniEngine:
    """Build an :class:`OmniEngine` serving a Higgs TTS checkpoint.

    Args:
        server_args: sglang :class:`sglang.srt.server_args.ServerArgs` with
            ``model_path`` pointing at a Higgs TTS checkpoint directory.
            ``disable_cuda_graph`` is forced on — the forward-embedded
            multi-codebook decode runs eager (user requested, 2026-04-22).
        gpu_id: CUDA device index for the model worker.
        max_new_tokens: Cap consumed by the iteration controller when a
            per-request value is not set.
    """
    # Ensure HiggsTTSModel is discoverable by sglang's model loader before
    # ModelWorker tries to resolve the checkpoint's ``architectures`` field.
    register_omni_models_in_sglang()

    # Lazy imports — keep the top-level import lightweight for the
    # PIPELINE_CONFIG_REGISTRY discovery pass.
    from sglang_omni.engines.ar.sglang_backend.model_worker import (
        ModelWorker,
        ModelWorkerConfig,
    )
    from sglang_omni.engines.ar.sglang_backend.scheduler.cache import create_tree_cache
    from sglang_omni.engines.ar.sglang_backend.scheduler.decode import DecodeManager
    from sglang_omni.engines.ar.sglang_backend.scheduler.prefill import PrefillManager
    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangBatchPlanner

    # Eager-only. Keep the caller's attention backend choice intact; sglang
    # defaults work for Qwen3.
    server_args.disable_cuda_graph = True

    model_worker = ModelWorker(
        config=ModelWorkerConfig(),
        server_args=server_args,
        gpu_id=gpu_id,
    )
    model = model_worker.model_runner.model

    req_to_token_pool, token_to_kv_pool_allocator = model_worker.get_memory_pool()

    tree_cache = create_tree_cache(
        server_args,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        server_args.page_size,
    )
    prefill_mgr = PrefillManager(
        page_size=server_args.page_size,
        chunked_prefill_size=server_args.chunked_prefill_size,
        max_prefill_tokens=server_args.max_prefill_tokens,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_worker.model_config,
        enable_overlap=False,
    )
    decode_mgr = DecodeManager(
        server_args=server_args,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        on_retract=lambda req: prefill_mgr.add_one_request(req),
    )
    batch_planner = SGLangBatchPlanner(prefill_mgr, decode_mgr, server_args)

    resource_mgr = HiggsSGLangResourceManager(
        token_to_kv_pool_allocator,
        req_to_token_pool,
        tree_cache,
        model=model,
    )
    iteration_ctrl = HiggsSGLangIterationController(
        tree_cache=tree_cache,
        max_new_tokens=max_new_tokens,
    )

    def _stream_adapter(request, output):
        step = output.data
        if step is None:
            return None
        return step.codes

    scheduler = Scheduler(
        batch_planner=batch_planner,
        resource_manager=resource_mgr,
        iteration_controller=iteration_ctrl,
        stream_adapter=_stream_adapter,
    )
    runner = HiggsSGLangModelRunner(model_worker, batch_planner)

    return OmniEngine(scheduler=scheduler, model_runner=runner)


__all__ = ["create_higgs_sglang_engine"]
