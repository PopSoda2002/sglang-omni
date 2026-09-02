from pydantic import Field
from typing import ClassVar

from sglang_omni.config import (
    PipelineConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
)

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
            process="perception",
            factory=f"{MODEL_STAGES_PREFIX}.create_perception_executor",
            factory_args={"dtype": MODEL_DTYPE},
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.12)
            ),
            next="thinker",
        ),
        StageConfig(
            name="thinker",
            process="thinker",
            factory=f"{MODEL_STAGES_PREFIX}.create_thinker_executor",
            # NemotronH runs under SGLang, whose rmsnorm kernel has no float32
            # path; the rest of the chain keeps the checkpoint's precision.
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.52)
            ),
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
            process="talker",
            factory=f"{MODEL_STAGES_PREFIX}.create_talker_executor",
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.22)
            ),
            wait_for=["perception"],
            merge_fn="sglang_omni.models.nemotron_voicechat.request_builders.merge_for_talker",
            can_accept_stream_before_payload=True,
            next="code2wav",
            stream_to=["code2wav"],
        ),
        StageConfig(
            name="code2wav",
            process="code2wav",
            factory=f"{MODEL_STAGES_PREFIX}.create_code2wav_executor",
            factory_args={"dtype": MODEL_DTYPE},
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.08)
            ),
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

class NemotronVoiceChatPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "NemotronVoiceChatForCausalLM"
    model_path: str
    stages: list[StageConfig] = Field(default_factory=nemotron_voicechat_stages_factory)

EntryClass = NemotronVoiceChatPipelineConfig