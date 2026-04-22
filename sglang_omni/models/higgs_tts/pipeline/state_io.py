# SPDX-License-Identifier: Apache-2.0
"""Helpers to convert between StagePayload.data and HiggsTtsState."""

from __future__ import annotations

from sglang_omni.models.higgs_tts.io import HiggsTtsState
from sglang_omni.proto import StagePayload


def load_state(payload: StagePayload) -> HiggsTtsState:
    return HiggsTtsState.from_dict(payload.data)


def store_state(payload: StagePayload, state: HiggsTtsState) -> StagePayload:
    payload.data = state.to_dict()
    return payload
