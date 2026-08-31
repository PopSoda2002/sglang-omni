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
            process="pipeline",
            factory=f"{MODEL_STAGES_PREFIX}.create_preprocessing_executor",
            next="perception",
        ),
        StageConfig(
            name="perception",
            process="pipeline",
            factory=f"{MODEL_STAGES_PREFIX}.create_perception_executor",
            factory_args={"dtype": MODEL_DTYPE},
            gpu=0,
            next="thinker",
        ),
        StageConfig(
            name="thinker",
            process="pipeline",
            factory=f"{MODEL_STAGES_PREFIX}.create_thinker_executor",
            factory_args={"dtype": MODEL_DTYPE},
            gpu=0,
            next="decode",
            stream_to=["talker"],
        ),
        StageConfig(
            name="decode",
            process="pipeline",
            factory=f"{MODEL_STAGES_PREFIX}.create_decode_executor",
            terminal=True,
        ),
        StageConfig(
            name="talker",
            process="pipeline",
            factory=f"{MODEL_STAGES_PREFIX}.create_talker_executor",
            factory_args={"dtype": "bfloat16"},
            gpu=1,
            wait_for=["perception"],
            merge_fn="sglang_omni.models.nemotron_voicechat.request_builders.merge_for_talker",
            can_accept_stream_before_payload=True,
            next="code2wav",
            stream_to=["code2wav"],
        ),
        StageConfig(
            name="code2wav",
            process="pipeline",
            factory=f"{MODEL_STAGES_PREFIX}.create_code2wav_executor",
            factory_args={"dtype": MODEL_DTYPE},
            gpu=1,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

class NemotronVoiceChatPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "NemotronVoiceChatForCausalLM"
    model_path: str
    stages: list[StageConfig] = Field(default_factory=nemotron_voicechat_stages_factory)

EntryClass = NemotronVoiceChatPipelineConfig