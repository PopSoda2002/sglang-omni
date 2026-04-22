# SPDX-License-Identifier: Apache-2.0
"""Autoregressive loop driver around the PR4a sampler state machine.

Steps :func:`sglang_omni.models.higgs_tts.sampler.step` across time until
the sampler signals termination (``generation_done``) or ``max_new_tokens``
is hit. Logits for each step come from a caller-supplied
:class:`LogitsProducer`, so this module is sglang-agnostic: unit tests
drive it with synthetic logits; PR4c will either wrap it over a real
sglang forward call (manual AR path) or bypass it entirely if we pick the
forward-embedded approach (s2_pro pattern). Either way this stays useful
as the sampler's integration-level test harness.

Output ``codes`` shape is ``[L, num_codebooks]`` — the raw (delayed)
codebook IDs emitted per step, including BOC/EOC fills introduced by the
state machine. PR5's vocoder applies :func:`reverse_delay_pattern` before
decoding.

**Invariant:** ``codes`` never contains a ``STOP_CODE`` row. The driver
checks ``state.generation_done`` before each call to ``sampler_step``, so
the sampler's terminal ``[-1, ..., -1]`` sentinel is never accumulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from sglang_omni.models.higgs_tts.sampler import HiggsSamplerState
from sglang_omni.models.higgs_tts.sampler import step as sampler_step


class LogitsProducer(Protocol):
    """Per-step hook supplied by the engine integration (PR4c).

    An implementation typically closes over the live
    :class:`HiggsSamplerState` and a loaded :class:`HiggsTTSModel`, and
    for each step:

    1. Reads ``state.last_codes`` (written by the previous ``sampler_step``
       call) to feed into the model's ``embed_input_ids`` at the next
       position (audio-placeholder overlay).
    2. Runs the model's backbone forward to obtain hidden states.
    3. Projects through ``model.modality_head.generate`` to get
       ``[1, N, V_codebook]`` and squeezes the leading dim.

    **Scope**: audio-codebook logits only. This driver does not inspect
    text logits — if the checkpoint's text channel emits a text-side stop
    token (e.g. ``<|im_end|>``) before codebook-0 emits ``<|eoc|>``, the
    loop will run until ``max_new_tokens``. PR4c should decide whether to
    fold a text-EOS signal into ``state.generation_done`` manually.
    """

    def __call__(self, step_idx: int) -> torch.Tensor:
        """Return ``[num_codebooks, codebook_vocab]`` float logits for the
        step at index ``step_idx`` (0-based)."""
        ...


@dataclass
class ArLoopResult:
    codes: torch.Tensor
    """Accumulated codebook ids, shape ``[L, num_codebooks]``. Device and
    dtype follow whatever the :class:`LogitsProducer` returned (long
    casting for dtype is done inside the sampler); the caller is
    responsible for moving to CPU / the vocoder's device as needed."""

    num_steps: int
    """How many sampler steps ran. Equals ``codes.shape[0]``."""

    hit_max_new_tokens: bool
    """True iff the loop exited because the cap was reached *without* the
    sampler flipping ``generation_done``. False for clean EOC exit and
    for the case where the caller passed an already-done state."""

    state: HiggsSamplerState
    """The final sampler state. Useful for debugging and for resuming a
    streaming request across multiple :func:`run_ar_loop` calls."""


def run_ar_loop(
    logits_producer: LogitsProducer,
    *,
    num_codebooks: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    top_k: int | None = None,
    state: HiggsSamplerState | None = None,
) -> ArLoopResult:
    """Run the multi-codebook AR loop for a single request.

    See :class:`LogitsProducer` for the producer contract. ``state`` may
    be passed in to resume a partial request or seed non-default initial
    conditions; otherwise a fresh :class:`HiggsSamplerState` is created.
    """
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")

    if state is None:
        state = HiggsSamplerState(num_codebooks=num_codebooks)
    elif state.num_codebooks != num_codebooks:
        raise ValueError(
            f"state.num_codebooks={state.num_codebooks} disagrees with "
            f"num_codebooks={num_codebooks}"
        )

    codes_per_step: list[torch.Tensor] = []
    for step_idx in range(max_new_tokens):
        if state.generation_done:
            break
        logits_NV = logits_producer(step_idx)
        codes_N = sampler_step(
            logits_NV,
            state,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        codes_per_step.append(codes_N)

    if codes_per_step:
        codes = torch.stack(codes_per_step, dim=0).to(torch.long)
    else:
        codes = torch.empty((0, num_codebooks), dtype=torch.long)

    # Self-computing terminal flag: hit_max iff the cap blocked us *and*
    # the sampler never terminated. Done-on-entry and EOC both give False.
    hit_max_new_tokens = (
        not state.generation_done and len(codes_per_step) >= max_new_tokens
    )

    return ArLoopResult(
        codes=codes,
        num_steps=codes.shape[0],
        hit_max_new_tokens=hit_max_new_tokens,
        state=state,
    )


__all__ = ["ArLoopResult", "LogitsProducer", "run_ar_loop"]
