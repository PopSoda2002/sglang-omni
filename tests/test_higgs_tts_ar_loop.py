# SPDX-License-Identifier: Apache-2.0
"""Tests for the Higgs TTS AR-loop driver (PR4b)."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.ar_loop import run_ar_loop
from sglang_omni.models.higgs_tts.delay_pattern import BOC_ID, EOC_ID
from sglang_omni.models.higgs_tts.sampler import HiggsSamplerState


def _deterministic_logits(picks_N: list[int], vocab_size: int = 1026) -> torch.Tensor:
    """Build logits where argmax per codebook is ``picks_N[c]``."""
    logits = torch.full((len(picks_N), vocab_size), -1e9)
    for c, pick in enumerate(picks_N):
        logits[c, pick] = 0.0
    return logits


def _make_producer(schedule):
    """Wrap a list of ``picks_N`` per step in a :class:`LogitsProducer` closure."""

    def producer(step_idx: int) -> torch.Tensor:
        if step_idx >= len(schedule):
            # Default: pick token 7 across all codebooks forever.
            N = len(schedule[0])
            return _deterministic_logits([7] * N)
        return _deterministic_logits(schedule[step_idx])

    return producer


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_full_sequence_delay_then_eoc_wind_down():
    """Drive: N steps of delay → codebook-0 emits EOC → wind-down → done."""
    N = 4
    # Step N (one past delay) emits EOC on codebook-0.
    schedule = [[7] * N] * N + [[EOC_ID] + [7] * (N - 1)]
    producer = _make_producer(schedule)

    result = run_ar_loop(producer, num_codebooks=N, max_new_tokens=100, temperature=0.0)

    # Exactly N delay steps + 1 EOC-trigger step + (N-2) wind-down steps = 2N-1.
    assert result.num_steps == 2 * N - 1
    assert not result.hit_max_new_tokens
    assert result.codes.shape == (2 * N - 1, N)

    # Delay window layout: step i has BOC at codebooks (i+1, N) for i < N-1.
    for i in range(N - 1):
        row = result.codes[i].tolist()
        assert row[0] == 7
        assert row[i + 1 :] == [BOC_ID] * (N - (i + 1))


def test_hits_max_new_tokens_when_no_eoc():
    """Never emit EOC → loop runs until the cap; caller can resume from
    the returned state."""
    N = 3
    producer = _make_producer([[5, 6, 7]])  # same row forever
    result = run_ar_loop(producer, num_codebooks=N, max_new_tokens=10, temperature=0.0)
    assert result.num_steps == 10
    assert result.hit_max_new_tokens
    # Resume contract: state is still live, caller could feed it back in.
    assert not result.state.generation_done
    assert result.state.delay_count == N  # past delay window


def test_small_N_terminates_right_after_eoc():
    """N=2: EOC on codebook-0 immediately finalises the sampler."""
    N = 2
    schedule = [[7, 7], [7, 7], [EOC_ID, 7]]
    result = run_ar_loop(
        _make_producer(schedule),
        num_codebooks=N,
        max_new_tokens=20,
        temperature=0.0,
    )
    # 2 delay steps (delay_count caps at N==2) + 1 EOC step = 3.
    assert result.num_steps == 3
    assert not result.hit_max_new_tokens
    # Last step's codebook-0 is EOC.
    assert result.codes[-1, 0].item() == EOC_ID


# ---------------------------------------------------------------------------
# Resuming with a prebuilt state
# ---------------------------------------------------------------------------


def test_resume_from_prebuilt_state_past_delay():
    """If delay is already done, step 0 should free-sample codebook 0."""
    N = 4
    producer = _make_producer([[7] * N])
    state = HiggsSamplerState(num_codebooks=N, delay_count=N)

    result = run_ar_loop(
        producer, num_codebooks=N, max_new_tokens=1, temperature=0.0, state=state
    )
    assert result.num_steps == 1
    assert result.codes[0].tolist() == [7, 7, 7, 7]  # no BOC overrides


def test_done_state_produces_zero_steps():
    """An already-done state exits the loop with an empty result."""
    N = 4
    producer = _make_producer([[7] * N])
    state = HiggsSamplerState(num_codebooks=N, generation_done=True)
    result = run_ar_loop(
        producer, num_codebooks=N, max_new_tokens=100, temperature=0.0, state=state
    )
    assert result.num_steps == 0
    assert result.codes.shape == (0, N)
    assert result.codes.dtype == torch.long
    assert not result.hit_max_new_tokens
    assert result.state is state  # caller's state passed through


# ---------------------------------------------------------------------------
# Invariant: codes column c matches state.last_codes across steps
# ---------------------------------------------------------------------------


def test_last_codes_tracks_most_recent_row():
    N = 3
    schedule = [[i, i, i] for i in range(5)]
    producer = _make_producer(schedule)
    state = HiggsSamplerState(num_codebooks=N)
    result = run_ar_loop(
        producer, num_codebooks=N, max_new_tokens=5, temperature=0.0, state=state
    )
    # After the final step, last_codes mirrors codes[-1] (and the returned
    # state is the same object the caller passed in).
    assert result.state is state
    assert state.last_codes.tolist() == result.codes[-1].tolist()


# ---------------------------------------------------------------------------
# Misuse
# ---------------------------------------------------------------------------


def test_zero_max_new_tokens_rejected():
    with pytest.raises(ValueError, match="max_new_tokens"):
        run_ar_loop(_make_producer([[7, 7]]), num_codebooks=2, max_new_tokens=0)


def test_state_num_codebooks_mismatch_rejected():
    state = HiggsSamplerState(num_codebooks=3)
    with pytest.raises(ValueError, match="num_codebooks"):
        run_ar_loop(
            _make_producer([[7] * 4]),
            num_codebooks=4,
            max_new_tokens=5,
            state=state,
        )
