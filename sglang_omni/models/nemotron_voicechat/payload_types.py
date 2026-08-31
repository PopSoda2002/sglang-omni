from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire

@dataclass
class NemotronVoiceChatState(DeclarativeStateBase):
    # 16 kHZ mono audio
    waveform: Any | None = wire(None, codec="typed_tensor")
    acoustic_frames: Any | None = wire(None, codec="typed_tensor")
    num_frames: int = wire(0, codec="int")
    text_ids: list = wire(default_factory=list, codec="list")
    function_ids: list = wire(default_factory=list, codec="list")
    codes: Any | None = wire(None, codec="typed_tensor")
    # 22.05 kHz mono audio
    output_waveform: Any | None = wire(None, codec="typed_tensor")