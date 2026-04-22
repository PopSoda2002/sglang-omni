# SPDX-License-Identifier: Apache-2.0
"""Tests for the Higgs TTS multi-codebook sampler state machine (PR4a)."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.delay_pattern import BOC_ID, EOC_ID
from sglang_omni.models.higgs_tts.sampler import STOP_CODE, HiggsSamplerState, step


def _deterministic_logits(picks_N: list[int], vocab_size: int = 1026) -> torch.Tensor:
    """Build logits where argmax per codebook is ``picks_N[c]``.

    Default ``vocab_size=1026`` matches the Higgs codebook vocab so
    ``_deterministic_logits([EOC_ID, ...])`` works without overriding it.
    """
    logits = torch.full((len(picks_N), vocab_size), -1e9)
    for c, pick in enumerate(picks_N):
        logits[c, pick] = 0.0
    return logits


# ---------------------------------------------------------------------------
# Delay window
# ---------------------------------------------------------------------------


class TestDelayWindow:
    def test_first_step_forces_later_codebooks_to_boc(self):
        N = 4
        logits = _deterministic_logits([7, 7, 7, 7])
        state = HiggsSamplerState(num_codebooks=N)
        codes = step(logits, state, temperature=0.0)
        # Codebook 0 samples freely (picks 7); codebooks 1..3 forced to BOC.
        assert codes.tolist() == [7, BOC_ID, BOC_ID, BOC_ID]
        assert state.delay_count == 1

    def test_delay_narrows_by_one_each_step(self):
        N = 4
        logits = _deterministic_logits([7, 7, 7, 7])
        state = HiggsSamplerState(num_codebooks=N)

        # Step 0: free [0], BOC [1:4]
        step(logits, state, temperature=0.0)
        # Step 1: free [0:2], BOC [2:4]
        codes = step(logits, state, temperature=0.0)
        assert codes.tolist() == [7, 7, BOC_ID, BOC_ID]
        # Step 2: free [0:3], BOC [3:4]
        codes = step(logits, state, temperature=0.0)
        assert codes.tolist() == [7, 7, 7, BOC_ID]

    def test_delay_exits_after_N_steps(self):
        N = 4
        logits = _deterministic_logits([7, 7, 7, 7])
        state = HiggsSamplerState(num_codebooks=N)
        # Run N steps; after that, all codebooks sample freely.
        for _ in range(N):
            step(logits, state, temperature=0.0)
        assert state.delay_count == N
        codes = step(logits, state, temperature=0.0)
        assert codes.tolist() == [7, 7, 7, 7]  # free sampling

    def test_delay_last_step_does_not_override(self):
        """When delay_count == N-1, next_cb == N, so no codebook is overridden.
        (Boundary: the ``if next_cb < N`` guard in the impl.)"""
        N = 3
        logits = _deterministic_logits([5, 5, 5])
        state = HiggsSamplerState(num_codebooks=N, delay_count=N - 1)
        codes = step(logits, state, temperature=0.0)
        assert codes.tolist() == [5, 5, 5]
        assert state.delay_count == N


# ---------------------------------------------------------------------------
# EOC detection & wind-down
# ---------------------------------------------------------------------------


class TestEocWindDown:
    @pytest.mark.parametrize("N", [3, 4, 5])
    def test_codebook0_eoc_after_delay_triggers_wind_down(self, N: int):
        """Wind-down length is exactly N - 2 steps for N > 2."""
        eoc_logits = _deterministic_logits([EOC_ID] + [7] * (N - 1))
        normal_logits = _deterministic_logits([7] * N)
        state = HiggsSamplerState(num_codebooks=N, delay_count=N)

        codes = step(eoc_logits, state, temperature=0.0)
        assert codes[0].item() == EOC_ID
        assert state.eoc_countdown == N - 2
        assert not state.generation_done

        for i in range(N - 2):
            assert not state.generation_done, f"done prematurely at step {i}"
            step(normal_logits, state, temperature=0.0)
        assert state.generation_done

    def test_wind_down_allows_free_sampling(self):
        N = 4
        state = HiggsSamplerState(num_codebooks=N, delay_count=N, eoc_countdown=2)
        logits = _deterministic_logits([5, 6, 7, 8])
        codes = step(logits, state, temperature=0.0)
        # No overrides during wind-down.
        assert codes.tolist() == [5, 6, 7, 8]
        assert state.eoc_countdown == 1
        # last_codes tracks wind-down samples (PR 4b's embed_input_ids needs it).
        assert state.last_codes.tolist() == [5, 6, 7, 8]

    def test_wind_down_ignores_retriggered_eoc(self):
        """Once in wind-down, a new EOC on codebook-0 must NOT reset the
        countdown — the elif chain's ordering keeps us in the wind-down
        branch."""
        N = 4
        state = HiggsSamplerState(num_codebooks=N, delay_count=N, eoc_countdown=2)
        logits = _deterministic_logits([EOC_ID, 7, 7, 7])
        step(logits, state, temperature=0.0)
        assert state.eoc_countdown == 1  # just decremented, not reset to N-2

    def test_small_N_skips_wind_down(self):
        """N=2: EOC on codebook-0 immediately finalises."""
        N = 2
        state = HiggsSamplerState(num_codebooks=N, delay_count=N)
        logits = _deterministic_logits([EOC_ID, 7])
        step(logits, state, temperature=0.0)
        assert state.generation_done
        assert state.eoc_countdown is None  # never set

    def test_eoc_ignored_during_delay_window(self):
        """EOC on codebook-0 during the delay window does NOT trigger wind-down.
        (The impl's elif chain: delay check runs before EOC check.)"""
        N = 4
        state = HiggsSamplerState(num_codebooks=N)
        assert state.delay_count == 0
        logits = _deterministic_logits([EOC_ID, 7, 7, 7])
        step(logits, state, temperature=0.0)
        assert state.eoc_countdown is None
        assert not state.generation_done


# ---------------------------------------------------------------------------
# Stop signal
# ---------------------------------------------------------------------------


class TestStopSignal:
    def test_returns_minus_one_after_done(self):
        N = 3
        state = HiggsSamplerState(num_codebooks=N, generation_done=True)
        logits = _deterministic_logits([7, 7, 7])
        codes = step(logits, state, temperature=0.0)
        assert codes.tolist() == [STOP_CODE, STOP_CODE, STOP_CODE]


# ---------------------------------------------------------------------------
# last_codes tracking
# ---------------------------------------------------------------------------


class TestLastCodes:
    def test_last_codes_updated_each_step(self):
        N = 3
        state = HiggsSamplerState(num_codebooks=N)
        logits = _deterministic_logits([4, 5, 6])
        codes = step(logits, state, temperature=0.0)
        assert state.last_codes is not None
        assert state.last_codes.tolist() == codes.tolist()

    def test_last_codes_captures_boc_overrides(self):
        """Delay-window overrides are reflected in last_codes."""
        N = 4
        state = HiggsSamplerState(num_codebooks=N)
        logits = _deterministic_logits([3, 3, 3, 3])
        step(logits, state, temperature=0.0)
        assert state.last_codes.tolist() == [3, BOC_ID, BOC_ID, BOC_ID]

    def test_last_codes_not_updated_once_done(self):
        N = 3
        state = HiggsSamplerState(num_codebooks=N, generation_done=True)
        logits = _deterministic_logits([4, 5, 6])
        step(logits, state, temperature=0.0)
        assert state.last_codes is None  # was never set


# ---------------------------------------------------------------------------
# Independence between concurrent states
# ---------------------------------------------------------------------------


def test_concurrent_states_do_not_bleed():
    """Two states exercised in parallel reach different delay_counts."""
    N = 4
    logits = _deterministic_logits([7, 7, 7, 7])
    s1 = HiggsSamplerState(num_codebooks=N)
    s2 = HiggsSamplerState(num_codebooks=N)

    step(logits, s1, temperature=0.0)
    step(logits, s1, temperature=0.0)
    step(logits, s2, temperature=0.0)

    assert s1.delay_count == 2
    assert s2.delay_count == 1


# ---------------------------------------------------------------------------
# Sampling params — stochastic smoke checks
# ---------------------------------------------------------------------------


class TestSamplingParams:
    def test_greedy_matches_argmax(self):
        torch.manual_seed(0)
        N, V = 4, 20
        state = HiggsSamplerState(num_codebooks=N, delay_count=N)
        logits = torch.randn(N, V)
        codes = step(logits, state, temperature=0.0)
        assert codes.tolist() == logits.argmax(-1).tolist()

    def test_top_k_restricts_samples(self):
        torch.manual_seed(0)
        N, V = 4, 20
        logits = torch.randn(N, V)
        # Pick the top-2 indices per codebook a priori.
        top2 = logits.topk(2, dim=-1).indices.tolist()

        # Each trial gets a fresh state (delay_count past the window) and a
        # fresh seed so sampling is both stochastic and reproducible.
        for trial in range(20):
            state = HiggsSamplerState(num_codebooks=N, delay_count=N)
            torch.manual_seed(trial)
            codes = step(logits, state, temperature=1.0, top_k=2)
            for c, code in enumerate(codes.tolist()):
                assert code in top2[c], f"codebook {c} sampled {code}, not in {top2[c]}"

    def test_top_p_restricts_samples(self):
        """top_p=0.01 ≈ argmax."""
        torch.manual_seed(0)
        N, V = 4, 20
        state = HiggsSamplerState(num_codebooks=N, delay_count=N)
        # Heavily peaked logits so the top-1 token dominates cumulative mass.
        logits = torch.full((N, V), -1e4)
        for c in range(N):
            logits[c, c] = 10.0
        codes = step(logits, state, temperature=1.0, top_p=0.01)
        assert codes.tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Misuse
# ---------------------------------------------------------------------------


def test_logits_shape_mismatch_raises():
    state = HiggsSamplerState(num_codebooks=4)
    bad = torch.zeros(3, 10)
    with pytest.raises(ValueError, match="num_codebooks=4"):
        step(bad, state, temperature=0.0)
