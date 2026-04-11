# SPDX-License-Identifier: Apache-2.0
"""MMSU task runner and evaluation helpers."""

from __future__ import annotations

import base64
import csv
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import SendFn
from benchmarks.benchmarker.utils import get_wav_duration
from benchmarks.dataset.mmsu import MmsuSample
from benchmarks.metrics.accuracy import INDEX_TO_LETTER, extract_answer_letter

DEFAULT_PROMPT = (
    "Listen to the audio and answer the multiple-choice question. "
    "Reply with only A, B, C, or D."
)


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text_value = item.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
        return "".join(text_parts).strip()
    return ""


def _extract_message_text(message: dict[str, Any]) -> str:
    text = _extract_text_content(message.get("content"))
    if text:
        return text
    audio_obj = message.get("audio")
    if isinstance(audio_obj, dict):
        transcript = audio_obj.get("transcript")
        if isinstance(transcript, str):
            return transcript.strip()
    return ""


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text.lower())).strip()


def _build_question_text(sample: MmsuSample, prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        f"Question: {sample.question}\n"
        f"A. {sample.choices[0]}\n"
        f"B. {sample.choices[1]}\n"
        f"C. {sample.choices[2]}\n"
        f"D. {sample.choices[3]}"
    )


def _extract_prediction(
    raw_response: str,
    choices: list[str],
) -> tuple[int | None, str]:
    predicted_index = extract_answer_letter(raw_response)
    if predicted_index is not None and predicted_index < len(choices):
        return predicted_index, choices[predicted_index]

    normalized_response = _normalize_text(raw_response)
    for index, choice in enumerate(choices):
        normalized_choice = _normalize_text(choice)
        if not normalized_choice:
            continue
        if (
            normalized_response == normalized_choice
            or normalized_choice in normalized_response
        ):
            return index, choice

    return None, ""


def _build_group_metrics(
    results: list["MmsuResult"],
    key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "parseable": 0}
    )
    for result in results:
        value = getattr(result, key)
        grouped[value]["total"] += 1
        if result.is_parseable:
            grouped[value]["parseable"] += 1
        if result.is_correct:
            grouped[value]["correct"] += 1

    metrics: dict[str, dict[str, Any]] = {}
    for name, counts in sorted(grouped.items()):
        metrics[name] = {
            "total": counts["total"],
            "correct": counts["correct"],
            "parseable": counts["parseable"],
            "accuracy": round(counts["correct"] / counts["total"], 4),
        }
    return metrics


@dataclass
class MmsuResult:
    sample_id: str = ""
    task_name: str = ""
    category: str = ""
    sub_category: str = ""
    sub_sub_category: str = ""
    linguistics_sub_discipline: str = ""
    correct_choice: str = ""
    correct_answer: str = ""
    predicted_choice: str = ""
    predicted_answer: str = ""
    raw_response: str = ""
    is_correct: bool = False
    is_parseable: bool = False
    latency_s: float = 0.0
    has_audio: bool = False
    audio_duration_s: float = 0.0
    error: str = ""


def make_mmsu_send_fn(
    model_name: str,
    api_url: str,
    *,
    modalities: list[str] | None = None,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 32,
    temperature: float = 0.0,
    save_audio_dir: str | None = None,
) -> SendFn:
    if modalities is None:
        modalities = ["text"]

    audio_mode = "audio" in modalities

    async def send_fn(
        session: aiohttp.ClientSession,
        sample: MmsuSample,
    ) -> RequestResult:
        result = RequestResult(request_id=sample.sample_id)
        start_time = time.perf_counter()
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": _build_question_text(sample, prompt),
                }
            ],
            "audios": [sample.audio_path],
            "modalities": modalities,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if audio_mode:
            payload["audio"] = {"format": "wav"}

        try:
            async with session.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                response_json = await response.json()

            message = response_json.get("choices", [{}])[0].get("message", {})
            result.text = _extract_message_text(message)

            usage = response_json.get("usage", {})
            result.prompt_tokens = usage.get("prompt_tokens", 0)
            result.completion_tokens = usage.get("completion_tokens", 0)

            if audio_mode:
                audio_obj = message.get("audio")
                if isinstance(audio_obj, dict) and audio_obj.get("data"):
                    wav_bytes = base64.b64decode(audio_obj["data"])
                    result.audio_duration_s = get_wav_duration(wav_bytes)
                    if save_audio_dir and len(wav_bytes) > 44:
                        wav_path = os.path.join(save_audio_dir, f"{sample.sample_id}.wav")
                        with open(wav_path, "wb") as file_obj:
                            file_obj.write(wav_bytes)
                        result.wav_path = wav_path
                    result.is_success = bool(result.text or result.audio_duration_s > 0)
                else:
                    result.error = "No audio in response"
            elif result.text:
                result.is_success = True
            else:
                result.error = "Empty response"
        except (aiohttp.ClientError, Exception) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time
            result.engine_time_s = result.latency_s
            if result.audio_duration_s > 0:
                result.rtf = result.latency_s / result.audio_duration_s
            if result.completion_tokens > 0 and result.engine_time_s > 0:
                result.tok_per_s = result.completion_tokens / result.engine_time_s

        return result

    return send_fn


def build_mmsu_results(
    request_results: list[RequestResult],
    samples: list[MmsuSample],
    modalities: list[str] | None = None,
) -> list[MmsuResult]:
    if modalities is None:
        modalities = ["text"]

    audio_mode = "audio" in modalities
    sample_map = {sample.sample_id: sample for sample in samples}
    results: list[MmsuResult] = []

    for request_result in request_results:
        sample = sample_map.get(request_result.request_id)
        if sample is None:
            continue

        predicted_index, predicted_answer = _extract_prediction(
            request_result.text,
            sample.choices,
        )
        correct_index = sample.answer_index

        result = MmsuResult(
            sample_id=sample.sample_id,
            task_name=sample.task_name,
            category=sample.category,
            sub_category=sample.sub_category,
            sub_sub_category=sample.sub_sub_category,
            linguistics_sub_discipline=sample.linguistics_sub_discipline,
            correct_choice=INDEX_TO_LETTER.get(correct_index, ""),
            correct_answer=sample.answer_text,
            predicted_choice=INDEX_TO_LETTER.get(predicted_index, ""),
            predicted_answer=predicted_answer,
            raw_response=request_result.text,
            is_correct=(
                predicted_index is not None and correct_index == predicted_index
            )
            or (
                predicted_answer
                and _normalize_text(predicted_answer)
                == _normalize_text(sample.answer_text)
            ),
            is_parseable=predicted_index is not None or bool(predicted_answer),
            latency_s=request_result.latency_s,
            error=request_result.error,
        )

        if audio_mode:
            result.has_audio = request_result.audio_duration_s > 0
            result.audio_duration_s = request_result.audio_duration_s

        results.append(result)

    return results


def compute_mmsu_metrics(results: list[MmsuResult]) -> dict[str, Any]:
    total = len(results)
    parseable = sum(1 for result in results if result.is_parseable)
    correct = sum(1 for result in results if result.is_correct)

    return {
        "total_samples": total,
        "parseable_samples": parseable,
        "unparseable_samples": total - parseable,
        "correct": correct,
        "incorrect": total - correct,
        "overall_accuracy": round(correct / total, 4) if total else 0.0,
        "per_task": _build_group_metrics(results, "task_name"),
        "per_category": _build_group_metrics(results, "category"),
        "per_sub_category": _build_group_metrics(results, "sub_category"),
        "per_sub_sub_category": _build_group_metrics(results, "sub_sub_category"),
        "per_linguistics_sub_discipline": _build_group_metrics(
            results,
            "linguistics_sub_discipline",
        ),
    }


def print_mmsu_summary(
    metrics: dict[str, Any],
    model_name: str,
    *,
    speed_metrics: dict[str, Any] | None = None,
) -> None:
    print("\n" + "=" * 60)
    print(f"  MMSU Results - {model_name}")
    print("=" * 60)
    print(f"  Total samples:    {metrics['total_samples']}")
    print(f"  Parseable:        {metrics['parseable_samples']}")
    print(f"  Correct:          {metrics['correct']}")
    print(f"  Overall accuracy: {metrics['overall_accuracy']:.2%}")
    print("-" * 60)
    print(f"  {'Category':<18} {'Acc':>8} {'N':>6}")
    print("-" * 60)
    for name, info in metrics["per_category"].items():
        print(f"  {name:<18} {info['accuracy']:>8.2%} {info['total']:>6}")
    if speed_metrics:
        print("-" * 60)
        print(f"  Latency mean:     {speed_metrics.get('latency_mean_s', 0):.3f}s")
        print(f"  Latency p95:      {speed_metrics.get('latency_p95_s', 0):.3f}s")
        if speed_metrics.get("audio_duration_mean_s", 0) > 0:
            print(
                f"  Audio mean:       {speed_metrics.get('audio_duration_mean_s', 0):.3f}s"
            )
        if speed_metrics.get("rtf_mean") is not None:
            print(f"  RTF mean:         {speed_metrics.get('rtf_mean', 0):.4f}")
        print(f"  Throughput:       {speed_metrics.get('throughput_qps', 0):.2f} req/s")
    print("=" * 60)


def save_mmsu_results(
    results: list[MmsuResult],
    metrics: dict[str, Any],
    config: dict[str, Any],
    output_dir: str,
    *,
    speed_metrics: dict[str, Any] | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    summary_output = {
        "summary": metrics,
        "config": config,
        "per_sample": [asdict(result) for result in results],
    }
    if speed_metrics:
        summary_output["speed_metrics"] = speed_metrics

    json_path = os.path.join(output_dir, "mmsu_results.json")
    with open(json_path, "w") as file_obj:
        json.dump(summary_output, file_obj, indent=2)

    jsonl_path = os.path.join(output_dir, "mmsu_predictions.jsonl")
    with open(jsonl_path, "w") as file_obj:
        for result in results:
            file_obj.write(json.dumps(asdict(result)) + "\n")

    csv_path = os.path.join(output_dir, "mmsu_results.csv")
    if results:
        fieldnames = list(asdict(results[0]).keys())
        with open(csv_path, "w", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))
