"""Quick launcher with reduced mem_fraction_static."""
import asyncio
import logging
import multiprocessing as mp
import os

import uvicorn

logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def main():
    from sglang_omni.client import Client
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.serve.openai_api import create_app

    gpu_placement = {
        "thinker": 5,
        "talker_ar": 6,
        "code_predictor": 0,
        "code2wav": 0,
    }

    config = Qwen3OmniSpeechPipelineConfig(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        relay_backend="shm",
        gpu_placement=gpu_placement,
    )

    # Patch thinker to offload weights to CPU so it fits on 1 GPU
    for stage in config.stages:
        if stage.name == "thinker":
            if "server_args_overrides" not in stage.executor.args:
                stage.executor.args["server_args_overrides"] = {}
            stage.executor.args["server_args_overrides"]["cpu_offload_gb"] = 40
            stage.executor.args["server_args_overrides"]["mem_fraction_static"] = 0.5

    runner = MultiProcessPipelineRunner(config)
    logger.info("Starting pipeline...")
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
