from pydantic import Field
from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

# The ckpt is all saved in float32.
MODEL_DTYPE = "float32"

MODEL_STAGES_PREFIX = "sglang_omni.models.nemotron_voicechat.stages"


def nemotron_voicechat_stages_factory() -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="pippeline",
            factory=f"{MODEL_STAGES_PREFIX}.create_preprocessing_executor",
            next="perception",
        ),
        StageConfig(
            name="perception",
            process="pippeline",
            factory=f"{MODEL_STAGES_PREFIX}.create_perception_executor",
            factory_args={"dtype": MODEL_DTYPE},
            gpu=0,
            next="thinker",
        ),
        StageConfig(
            name="thinker",
            process="pippeline",
            factory=f"{MODEL_STAGES_PREFIX}.create_thinker_executor",
            factory_args={"dtype": MODEL_DTYPE},
            gpu=0,
            terminal=True,
        )
    ]

class NemotronVoiceChatPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "NemotronVoiceChatForCausalLM"
    model_path: str
    stages: list[StageConfig] = Field(default_factory=nemotron_voicechat_stages_factory)

EntryClass = NemotronVoiceChatPipelineConfig