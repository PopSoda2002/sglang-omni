# SPDX-License-Identifier: Apache-2.0
"""Multilingual TTS benchmark — 102 languages, scored via Omnilingual ASR.

Uses the ``fleurs_multilingual_102`` subset of
`k2-fsa/TTS_eval_datasets <https://huggingface.co/datasets/k2-fsa/TTS_eval_datasets>`__
(introduced in OmniVoice). Pipeline:

  1. Generate TTS audio for each (lang, sample) of the multilingual testset.
  2. Transcribe the saved audio with Meta's Omnilingual ASR 7B running in an
     isolated Python env (subprocess) so its ``fairseq2`` deps don't conflict
     with sglang-omni's torch 2.9 / transformers <5 pin.
  3. Compute per-language WER (CER for CJK) using the Omnilingual-style text
     normalizer ported from `k2-fsa/OmniVoice fleurs.py <https://github.com/k2-fsa/OmniVoice/blob/master/omnivoice/eval/wer/fleurs.py>`__.

Prepare the Omnilingual ASR env separately::

    conda create -n higgs-omni-asr python=3.12 -y
    conda activate higgs-omni-asr
    pip install omnilingual-asr fairseq2

    # Then point this benchmark at it:
    export OMNI_ASR_PYTHON=/path/to/envs/higgs-omni-asr/bin/python
    # or pass --omni-asr-python on the CLI.

Usage::

    # 1. Download + extract the dataset (~115 MB):
    python -m benchmarks.dataset.prepare --dataset fleurs-multilingual

    # 2. Launch the server (S2-Pro / Voxtral / etc.):
    python -m sglang_omni.cli serve --model-path fishaudio/s2-pro --port 8000

    # 3. Run a single-language eval (English, 50 samples):
    python -m benchmarks.eval.benchmark_tts_higgs_multilingual \\
        --dataset-root fleurs_multilingual_eval \\
        --lang en \\
        --max-samples 50 \\
        --max-concurrency 16 \\
        --omni-asr-python $OMNI_ASR_PYTHON \\
        --model fishaudio/s2-pro --port 8000

    # Generate-only (split phases for CI):
    python -m benchmarks.eval.benchmark_tts_higgs_multilingual \\
        --generate-only \\
        --dataset-root fleurs_multilingual_eval --lang en \\
        --output-dir results/higgs_ml_en \\
        --model fishaudio/s2-pro --port 8000

    # Transcribe-only on saved audio:
    python -m benchmarks.eval.benchmark_tts_higgs_multilingual \\
        --transcribe-only \\
        --dataset-root fleurs_multilingual_eval --lang en \\
        --output-dir results/higgs_ml_en \\
        --model fishaudio/s2-pro --device cuda:0 \\
        --omni-asr-python $OMNI_ASR_PYTHON
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import dataclass

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.higgs_multilingual import (
    list_higgs_multilingual_langs,
    load_higgs_multilingual_samples,
)
from benchmarks.metrics.performance import (
    build_speed_results,
    compute_speed_metrics,
    print_speed_summary,
)
from benchmarks.tasks.multilingual_tts import (
    MultilingualTranscribeConfig,
    run_multilingual_transcribe,
)
from benchmarks.tasks.tts import (
    build_base_url,
    make_tts_send_fn,
    save_generated_audio_metadata,
    save_speed_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class HiggsMultilingualBenchmarkConfig:
    model: str
    dataset_root: str
    lang: str
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    voice: str | None = None
    voice_clone: bool = True
    output_dir: str = "results/tts_higgs_multilingual"
    max_samples: int | None = None
    max_new_tokens: int | None = 2048
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    warmup: int = 1
    concurrency: int = 1
    request_rate: float = float("inf")
    stream: bool = False
    disable_tqdm: bool = False
    # Transcribe phase
    device: str = "cuda:0"
    omni_asr_python: str | None = None
    omni_asr_model_card: str = "omniASR_LLM_7B"
    omni_asr_batch_size: int = 8


def _build_generation_kwargs(config: HiggsMultilingualBenchmarkConfig) -> dict:
    kw: dict = {}
    if config.max_new_tokens is not None:
        kw["max_new_tokens"] = config.max_new_tokens
    if config.temperature is not None:
        kw["temperature"] = config.temperature
    if config.top_p is not None:
        kw["top_p"] = config.top_p
    if config.top_k is not None:
        kw["top_k"] = config.top_k
    if config.repetition_penalty is not None:
        kw["repetition_penalty"] = config.repetition_penalty
    return kw


def _build_results_config(
    config: HiggsMultilingualBenchmarkConfig, *, base_url: str
) -> dict:
    return {
        "model": config.model,
        "base_url": base_url,
        "dataset_root": config.dataset_root,
        "lang": config.lang,
        "voice_clone": config.voice_clone,
        "voice": config.voice,
        "stream": config.stream,
        "max_samples": config.max_samples,
        "max_new_tokens": config.max_new_tokens,
        "warmup": config.warmup,
        "concurrency": config.concurrency,
        "request_rate": config.request_rate,
    }


async def run_generation(config: HiggsMultilingualBenchmarkConfig) -> dict:
    base_url = build_base_url(config)
    api_url = f"{base_url}/v1/audio/speech"

    samples = load_higgs_multilingual_samples(
        config.dataset_root, config.lang, config.max_samples
    )
    logger.info(f"Loaded {len(samples)} samples for lang={config.lang}")

    save_audio_dir = os.path.abspath(os.path.join(config.output_dir, "audio"))
    os.makedirs(save_audio_dir, exist_ok=True)

    generation_kwargs = _build_generation_kwargs(config)
    send_fn = make_tts_send_fn(
        config.model,
        api_url,
        stream=config.stream,
        no_ref_audio=not config.voice_clone,
        voice=config.voice,
        save_audio_dir=save_audio_dir,
        **generation_kwargs,
    )

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.concurrency,
            request_rate=config.request_rate,
            warmup=config.warmup,
            disable_tqdm=config.disable_tqdm,
        )
    )
    outputs = await runner.run(samples, send_fn)

    metrics = compute_speed_metrics(outputs, wall_clock_s=runner.wall_clock_s)
    results_config = _build_results_config(config, base_url=base_url)
    benchmark_results = build_speed_results(outputs, metrics, results_config)
    save_speed_results(outputs, metrics, results_config, config.output_dir)
    save_generated_audio_metadata(outputs, samples, config.output_dir)
    return benchmark_results


def run_transcribe(config: HiggsMultilingualBenchmarkConfig) -> dict:
    generation_mode = "streaming" if config.stream else "non-streaming"
    wer_config = {
        "model": config.model,
        "dataset_root": config.dataset_root,
        "lang": config.lang,
        "voice_clone": config.voice_clone,
        "voice": config.voice,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "max_samples": config.max_samples,
        "stream": config.stream,
        "concurrency": config.concurrency,
        "omni_asr_model_card": config.omni_asr_model_card,
        "omni_asr_batch_size": config.omni_asr_batch_size,
    }
    transcribe_config = MultilingualTranscribeConfig(
        model=config.model,
        output_dir=config.output_dir,
        lang=config.lang,
        device=config.device,
        omni_asr_python=config.omni_asr_python,
        omni_asr_model_card=config.omni_asr_model_card,
        omni_asr_batch_size=config.omni_asr_batch_size,
    )
    return run_multilingual_transcribe(
        transcribe_config,
        wer_config=wer_config,
        generation_mode=generation_mode,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Higgs multilingual TTS benchmark scored via Omnilingual ASR.",
    )
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        type=str,
        default="fishaudio/s2-pro",
        help="Model name for the API request.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help=(
            "Root directory containing {lang}.jsonl and {lang}/ subdirs. "
            "See module docstring for layout."
        ),
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help=(
            "ISO-639-1 (or 3) language code. Use --list-langs to enumerate "
            "available languages under --dataset-root."
        ),
    )
    parser.add_argument(
        "--list-langs",
        action="store_true",
        help="Print available language codes and exit.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Built-in speaker preset for plain TTS models.",
    )
    parser.add_argument(
        "--no-ref-audio",
        dest="no_ref_audio",
        action="store_true",
        help="Skip ref audio from dataset (plain TTS without voice cloning).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/tts_higgs_multilingual"
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--concurrency",
        "--max-concurrency",
        dest="concurrency",
        type=int,
        default=1,
    )
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device passed to the Omnilingual ASR subprocess.",
    )
    parser.add_argument(
        "--omni-asr-python",
        type=str,
        default=os.environ.get("OMNI_ASR_PYTHON"),
        help=(
            "Path to the Python interpreter that has omnilingual-asr "
            "installed. Defaults to $OMNI_ASR_PYTHON."
        ),
    )
    parser.add_argument(
        "--omni-asr-model-card",
        type=str,
        default="omniASR_LLM_7B",
    )
    parser.add_argument(
        "--omni-asr-batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=1200,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true")
    mode.add_argument("--transcribe-only", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> HiggsMultilingualBenchmarkConfig:
    return HiggsMultilingualBenchmarkConfig(
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        model=args.model,
        dataset_root=args.dataset_root,
        lang=args.lang,
        voice=args.voice,
        voice_clone=not args.no_ref_audio,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        warmup=args.warmup,
        concurrency=args.concurrency,
        request_rate=args.request_rate,
        stream=args.stream,
        disable_tqdm=args.disable_tqdm,
        device=args.device,
        omni_asr_python=args.omni_asr_python,
        omni_asr_model_card=args.omni_asr_model_card,
        omni_asr_batch_size=args.omni_asr_batch_size,
    )


async def _generate(config: HiggsMultilingualBenchmarkConfig) -> dict:
    results = await run_generation(config)
    print_speed_summary(
        results["summary"], config.model, concurrency=config.concurrency
    )
    return results


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.list_langs:
        for lang in list_higgs_multilingual_langs(args.dataset_root):
            print(lang)
        return

    config = _config_from_args(args)

    if args.transcribe_only:
        run_transcribe(config)
        return

    wait_for_service(build_base_url(config), timeout=args.server_timeout)
    asyncio.run(_generate(config))

    if args.generate_only:
        return

    run_transcribe(config)


if __name__ == "__main__":
    main()
