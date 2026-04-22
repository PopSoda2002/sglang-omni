# SPDX-License-Identifier: Apache-2.0
"""End-to-end test: Higgs TTS running under sglang's engine (PR4c).

Loads the real 1.7 B Higgs checkpoint via
:func:`sglang_omni.models.higgs_tts.factory.create_higgs_sglang_engine`,
submits a short zero-shot TTS request through
:func:`create_sglang_tts_engine_executor`, and verifies we get
multi-codebook codes back. Auto-skipped when the checkpoint isn't mounted
or CUDA is unavailable.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
import torch

_REAL_CKPT = "/hot-data/checkpoints/TTS/c66596d6cde44fb4ba8076cc2dc1e77c/step_35500"


pytestmark = pytest.mark.skipif(
    not os.path.isdir(_REAL_CKPT) or not torch.cuda.is_available(),
    reason="real Higgs checkpoint not mounted or no CUDA",
)


def test_sglang_engine_zero_shot_smoke():
    """Spin the sglang engine, submit one zero-shot TTS request, verify
    multi-codebook codes are produced (shape, non-trivial length)."""
    from sglang_omni.models.higgs_tts.io import HiggsTtsState
    from sglang_omni.models.higgs_tts.pipeline.stages import (
        create_sglang_tts_engine_executor,
    )
    from sglang_omni.proto import StagePayload
    from sglang_omni.proto.request import OmniRequest

    # Zero-shot prompt: <|text|> some-ids <|audio|>  (see tokenizer.py tests
    # for the real ckpt's special-token id mapping).
    text_id = 151672  # <|text|>
    audio_id = 151670  # <|audio|>
    prompt_ids = [text_id, 100, 200, 300, audio_id]

    state = HiggsTtsState(
        prompt_token_ids=prompt_ids,
        reference_codes_delayed=None,
        num_codebooks=8,
        codebook_size=1026,
        max_new_tokens=64,
        temperature=1.0,
    )
    payload = StagePayload(
        request_id=str(uuid.uuid4()),
        request=OmniRequest(inputs={"input": "test"}, params={}),
        data=state.to_dict(),
    )

    executor = create_sglang_tts_engine_executor(
        _REAL_CKPT,
        device="cuda:0",
        max_new_tokens=64,
        mem_fraction_static=0.7,
        max_running_requests=4,
    )

    async def _run() -> StagePayload:
        await executor.start()
        try:
            await executor.add_request(payload)
            return await executor.get_result()
        finally:
            await executor.stop()

    out_payload = asyncio.run(_run())

    out_state = HiggsTtsState.from_dict(out_payload.data)
    assert out_state.output_codes_delayed is not None, "engine returned no codes"
    codes = out_state.output_codes_delayed
    assert len(codes) > 0
    # Each row is a full [num_codebooks] list.
    for row in codes:
        assert len(row) == out_state.num_codebooks
    # Should have advanced past the delay window (N=8 steps of BOC overrides).
    assert len(codes) >= out_state.num_codebooks
    # Usage bookkeeping is populated.
    assert out_payload.data["usage"]["completion_tokens"] == len(codes)
