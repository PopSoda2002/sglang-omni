# SPDX-License-Identifier: Apache-2.0
"""MMSU benchmark."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.mmsu import load_mmsu_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.mmsu import (
    compute_mmsu_metrics,
    build_mmsu_results,
    make_mmsu_send_fn,
    print_mmsu_summary,
    save_mmsu_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


@dataclass
class MmsuBenchmarkConfig:
    model: str = "qwen3-omni"
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    modalities: list[str] = field(default_factory=lambda: ["text"])
    repo_dir: str = ""
    output_dir: str = "results/mmsu"
    max_samples: int | None = None
    task_names: list[str] | None = None
    categories: list[str] | None = None
    prompt: str | None = None
    max_tokens: int = 32
    temperature: float = 0.0
    warmup: int = 1
    max_concurrency: int = 1
    request_rate: float = float("inf")
    save_audio: bool = False
    disable_tqdm: bool = False
    seed: int | None = None


def _build_base_url(config: MmsuBenchmarkConfig) -> str:
    return config.base_url or f"http://{config.host}:{config.port}"


def _parse_modalities(value: str) -> list[str]:
    if value == "text":
        return ["text"]
    if value == "text+audio":
        return ["text", "audio"]
    raise argparse.ArgumentTypeError(
        f"Invalid modalities: {value}. Use 'text' or 'text+audio'."
    )


def _parse_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _config_from_args(args: argparse.Namespace) -> MmsuBenchmarkConfig:
    return MmsuBenchmarkConfig(
        model=args.model,
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        modalities=_parse_modalities(args.modalities),
        repo_dir=args.repo_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        task_names=_parse_list(args.task_names),
        categories=_parse_list(args.categories),
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        warmup=args.warmup,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        save_audio=args.save_audio,
        disable_tqdm=args.disable_tqdm,
        seed=args.seed,
    )


async def run_mmsu_benchmark(config: MmsuBenchmarkConfig) -> dict:
    base_url = _build_base_url(config)
    api_url = f"{base_url}/v1/chat/completions"

    samples = load_mmsu_samples(
        repo_dir=config.repo_dir,
        max_samples=config.max_samples,
        task_names=config.task_names,
        categories=config.categories,
        seed=config.seed,
    )

    save_audio_dir = None
    if config.save_audio and config.output_dir:
        save_audio_dir = os.path.join(config.output_dir, "audio")
        os.makedirs(save_audio_dir, exist_ok=True)

    send_fn_kwargs = {
        "modalities": config.modalities,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "save_audio_dir": save_audio_dir,
    }
    if config.prompt is not None:
        send_fn_kwargs["prompt"] = config.prompt

    send_fn = make_mmsu_send_fn(config.model, api_url, **send_fn_kwargs)
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.max_concurrency,
            request_rate=config.request_rate,
            warmup=config.warmup,
            disable_tqdm=config.disable_tqdm,
        )
    )
    request_results = await runner.run(samples, send_fn)

    results = build_mmsu_results(request_results, samples, config.modalities)
    metrics = compute_mmsu_metrics(results)
    speed_metrics = compute_speed_metrics(
        request_results,
        wall_clock_s=runner.wall_clock_s,
    )

    print_mmsu_summary(
        metrics,
        config.model,
        speed_metrics=speed_metrics if "audio" in config.modalities else None,
    )

    if config.output_dir:
        save_mmsu_results(
            results,
            metrics,
            {
                "model": config.model,
                "base_url": base_url,
                "modalities": config.modalities,
                "max_samples": config.max_samples,
                "task_names": config.task_names,
                "categories": config.categories,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "warmup": config.warmup,
                "max_concurrency": config.max_concurrency,
                "seed": config.seed,
            },
            config.output_dir,
            speed_metrics=speed_metrics if "audio" in config.modalities else None,
        )

    return {
        "accuracy": metrics,
        "speed": speed_metrics,
    }


async def benchmark(args: argparse.Namespace) -> dict:
    return await run_mmsu_benchmark(_config_from_args(args))


def main() -> None:
    parser = argparse.ArgumentParser(description="MMSU benchmark.")

    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="qwen3-omni")
    parser.add_argument(
        "--modalities",
        type=str,
        choices=["text", "text+audio"],
        default="text",
    )
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/mmsu")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--task-names", type=str, default=None)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--save-audio", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    wait_for_service(args.base_url or f"http://{args.host}:{args.port}")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
