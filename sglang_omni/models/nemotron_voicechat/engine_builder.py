from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sglang_omni.models.nemotron_voicechat.hf_config import (
    VOICECHAT_MODEL_ARCH_OVERRIDE,
    NemotronVoiceChatConfig,
    register_voicechat_hf_config,
)
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.models.nemotron_voicechat.model_runner import NemotronVoiceChatModelRunner
from sglang_omni.models.nemotron_voicechat.request_builders import (
    apply_thinker_result,
    build_thinker_request,
)

SHIM_DIR = Path.home() / ".cache" / "sglang-omni" / "voicechat"

def _config_shim(model_path):
    SHIM_DIR.mkdir(parents=True, exist_ok=True)
    source = Path(model_path)
    for entry in source.iterdir():
        link = SHIM_DIR / entry.name
        if entry.name != "config.json" and not link.exists():
            link.symlink_to(entry)
    config = NemotronVoiceChatConfig.from_dict(json.loads((source / "config.json").read_text()))
    (SHIM_DIR / "config.json").write_text(config.to_json_string())
    return str(SHIM_DIR)

class NemotronVoiceChatEngineBuilder(TtsEngineBuilder):
    model_name = "nemotron-voicechat"
    context_length = 8192

    def __init__(self, *, max_running_requests = 1):
        self.model_arch_override = VOICECHAT_MODEL_ARCH_OVERRIDE
        self.max_running_requests = max_running_requests

    def resolve_checkpoint(self, model_path):
        return _config_shim(resolve_model_path(model_path))

    def pre_infra_setup(self, checkpoint_dir):
        del checkpoint_dir
        register_voicechat_hf_config()

    def generation_defaults(self, *, dtype):
        return {
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "max_running_requests": self.max_running_requests,
            "chunked_prefill_size": -1,
            "dtype": dtype,
            "trust_remote_code": False,
        }

    def make_model_runner(self, model_worker, output_proc):
        return NemotronVoiceChatModelRunner(model_worker, output_proc)

    def setup_model(self, *, model_worker, checkpoint_dir, device, gpu_id, server_args):
        del model_worker, checkpoint_dir, device, gpu_id, server_args

    def make_adapters(self, model):
        del model
        return build_thinker_request, apply_thinker_result