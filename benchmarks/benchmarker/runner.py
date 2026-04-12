# SPDX-License-Identifier: Apache-2.0
"""BenchmarkRunner: warmup + concurrent dispatch with semaphore and rate limiting."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from benchmarks.benchmarker.data import RequestResult

logger = logging.getLogger(__name__)

SendFn = Callable[[aiohttp.ClientSession, Any], Coroutine[Any, Any, RequestResult]]


@dataclass
class RunConfig:
    max_concurrency: int = 1
    request_rate: float = float("inf")
    warmup: int = 1
    disable_tqdm: bool = False
    timeout_s: int = 300


class BenchmarkRunner:
    """Support concurrent requests sending in a single benchmark run.

    Note (chenyang):
    max_concurrency is default to 1, thus all the requests are runs sequentially.

    TODO (chenyang):
    Current concurrency implementation of models are not fully supported.
    https://github.com/sgl-project/sglang-omni/issues/229
    https://github.com/sgl-project/sglang-omni/issues/228
    """

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.wall_clock_s: float = 0.0

    @staticmethod
    def _sample_request_id(sample: Any) -> str:
        getter = sample.get if isinstance(sample, dict) else lambda key: getattr(sample, key, None)
        for key in ("sample_id", "request_id", "id"):
            value = getter(key)
            if value is not None:
                return str(value)
        return ""

    def _error_result(self, sample: Any, error: Any) -> RequestResult:
        return RequestResult(
            request_id=self._sample_request_id(sample),
            error=str(error),
        )

    @staticmethod
    def _finalize_result(result: RequestResult, elapsed_s: float) -> RequestResult:
        if result.latency_s <= 0:
            result.latency_s = elapsed_s
        if result.engine_time_s <= 0:
            result.engine_time_s = result.latency_s
        if result.audio_duration_s > 0 and result.rtf <= 0:
            result.rtf = result.latency_s / result.audio_duration_s
        if result.completion_tokens > 0 and result.engine_time_s > 0 and result.tok_per_s <= 0:
            result.tok_per_s = result.completion_tokens / result.engine_time_s
        return result

    async def _invoke_send_fn(
        self,
        session: aiohttp.ClientSession,
        sample: Any,
        send_fn: SendFn,
    ) -> RequestResult:
        started_at = time.perf_counter()
        outcome = (
            await asyncio.gather(send_fn(session, sample), return_exceptions=True)
        )[0]

        if isinstance(outcome, RequestResult):
            result = outcome
        elif isinstance(outcome, BaseException):
            result = self._error_result(sample, outcome)
        else:
            result = self._error_result(
                sample,
                f"Unexpected send_fn result: {type(outcome).__name__}",
            )

        elapsed_s = time.perf_counter() - started_at
        return self._finalize_result(result, elapsed_s)

    async def run(self, samples: list, send_fn: SendFn) -> list[RequestResult]:
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if self.config.warmup > 0:
                await self._warmup(session, samples, send_fn)

            logger.info(
                "Benchmarking %d requests (max_concurrency=%s)...",
                len(samples),
                self.config.max_concurrency,
            )
            t0 = time.perf_counter()
            results = await self._dispatch(session, samples, send_fn)
            self.wall_clock_s = time.perf_counter() - t0
        return results

    async def _warmup(
        self,
        session: aiohttp.ClientSession,
        samples: list,
        send_fn: SendFn,
    ) -> None:
        count = min(self.config.warmup, len(samples))
        logger.info("Warmup (%d requests)...", count)
        for i in range(count):
            result = await self._invoke_send_fn(session, samples[i], send_fn)
            status = "ok" if result.is_success else result.error
            logger.info("  warmup %d/%d: %s", i + 1, count, status)

    async def _dispatch(
        self,
        session: aiohttp.ClientSession,
        samples: list,
        send_fn: SendFn,
    ) -> list[RequestResult]:
        semaphore = (
            asyncio.Semaphore(self.config.max_concurrency)
            if self.config.max_concurrency
            else None
        )
        with tqdm(total=len(samples), disable=self.config.disable_tqdm) as pbar:

            async def _limited(sample: Any) -> RequestResult:
                if semaphore:
                    async with semaphore:
                        result = await self._invoke_send_fn(session, sample, send_fn)
                else:
                    result = await self._invoke_send_fn(session, sample, send_fn)
                pbar.update(1)
                return result

            tasks: list[asyncio.Task] = []
            for sample in samples:
                if self.config.request_rate != float("inf"):
                    interval = np.random.exponential(1.0 / self.config.request_rate)
                    await asyncio.sleep(interval)
                tasks.append(asyncio.create_task(_limited(sample)))

            results: list[RequestResult] = list(await asyncio.gather(*tasks))
        return results
