import asyncio
import logging
import multiprocessing as mp

import uvicorn

from sglang_omni.client import Client
from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.serve.openai_api import create_app


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    config = Qwen3OmniSpeechPipelineConfig(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        relay_backend="shm",
        gpu_placement={
            "thinker": 0,
            "talker_ar": 0,
            "code_predictor": 0,
            "code2wav": 0,
        },
    )

    for stage in config.stages:
        if stage.name == "thinker":
            stage.executor.args["thinker_max_seq_len"] = 2048
            stage.executor.args.setdefault("server_args_overrides", {})
            stage.executor.args["server_args_overrides"]["cpu_offload_gb"] = 40
            stage.executor.args["server_args_overrides"]["mem_fraction_static"] = 0.95
            stage.executor.args["server_args_overrides"]["max_running_requests"] = 1
            stage.executor.args["server_args_overrides"]["max_prefill_tokens"] = 2048
        elif stage.name == "talker_ar":
            stage.executor.args["talker_max_seq_len"] = 1024
            stage.executor.args.setdefault("server_args_overrides", {})
            stage.executor.args["server_args_overrides"]["mem_fraction_static"] = 0.2
            stage.executor.args["server_args_overrides"]["max_running_requests"] = 1
            stage.executor.args["server_args_overrides"]["max_prefill_tokens"] = 1024
        elif stage.name == "code_predictor":
            stage.executor.args["code_predictor_max_seq_len"] = 128

    runner = MultiProcessPipelineRunner(config)
    logger.info("Starting single-GPU speech pipeline...")
    await runner.start(timeout=600)
    logger.info("Pipeline ready.")

    try:
        client = Client(runner.coordinator)
        app = create_app(client, model_name="qwen3-omni")
        server_config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(server_config)
        await server.serve()
    finally:
        await runner.stop()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    asyncio.run(main())
