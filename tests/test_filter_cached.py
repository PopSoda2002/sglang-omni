# SPDX-License-Identifier: Apache-2.0
"""Regression tests for OmniEngine._filter_cached (fixes #299)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from sglang_omni.engines.omni.engine import OmniEngine, _PendingResult
from sglang_omni.engines.omni.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
    SchedulerStatus,
)

# -----------------------------------------------------------------------------
# Minimal fakes for Scheduler, BatchPlanner, CacheManager, ModelRunner
# -----------------------------------------------------------------------------


@dataclass
class _FakeBatchPlanner:
    build_count: int = 0

    def build_batch(self, requests: list[SchedulerRequest]) -> dict[str, Any]:
        self.build_count += 1
        return {"request_ids": [r.request_id for r in requests]}


class _FakeScheduler:
    """Minimal scheduler that records update() calls and resolves futures."""

    def __init__(self) -> None:
        self.batch_planner = _FakeBatchPlanner()
        self.update_calls: list[SchedulerOutput] = []
        self.failed: list[tuple[str, Exception]] = []
        self.iteration_controller = object()  # no needs_feedback attr

    def update(
        self, scheduler_output: SchedulerOutput, model_output: ModelRunnerOutput
    ) -> list[SchedulerRequest]:
        self.update_calls.append(scheduler_output)
        finished: list[SchedulerRequest] = []
        for req in scheduler_output.requests:
            if req.status != SchedulerStatus.RUNNING:
                continue
            req.status = SchedulerStatus.FINISHED
            finished.append(req)
        return finished

    def fail_request(self, request_id: str, error: Exception) -> None:
        self.failed.append((request_id, error))


class _FakeCacheManager:
    def __init__(self, cached: dict[str, RequestOutput] | None = None) -> None:
        self._cache = cached or {}
        self.put_calls: list[tuple[str, RequestOutput]] = []

    def get(self, request: SchedulerRequest) -> RequestOutput | None:
        return self._cache.get(request.request_id)

    def put(self, request: SchedulerRequest, output: RequestOutput) -> None:
        self.put_calls.append((request.request_id, output))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_request(
    req_id: str, status: SchedulerStatus = SchedulerStatus.RUNNING
) -> SchedulerRequest:
    return SchedulerRequest(request_id=req_id, status=status)


def _make_engine(cache_manager: _FakeCacheManager) -> OmniEngine:
    engine = OmniEngine.__new__(OmniEngine)
    engine.scheduler = _FakeScheduler()  # type: ignore[assignment]
    engine.model_runner = None  # not used in these tests
    engine.cache_manager = cache_manager
    engine.enable_overlap = False
    engine._feedback_mailbox = None
    engine._running = False
    engine._loop_task = None
    from collections import deque

    engine._result_queue = deque()
    engine._last_scheduler_output = None
    return engine


def _make_output(request_id: str, data: Any = None) -> RequestOutput:
    return RequestOutput(request_id=request_id, data=data or {"tok": 1})


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_filter_cached_all_cached_returns_none_and_resolves_pending() -> None:
    cached = {"r1": _make_output("r1"), "r2": _make_output("r2")}
    cm = _FakeCacheManager(cached)
    engine = _make_engine(cm)
    scheduler_output = SchedulerOutput(
        requests=[_make_request("r1"), _make_request("r2")],
        batch_data=None,
        step_id=1,
    )

    uncached, cached_pending = engine._filter_cached(scheduler_output)

    assert uncached is None
    assert cached_pending is not None
    assert cached_pending.update_cache is False
    assert [r.request_id for r in cached_pending.scheduler_output.requests] == [
        "r1",
        "r2",
    ]
    assert set(cached_pending.model_output.outputs.keys()) == {"r1", "r2"}


def test_filter_cached_all_uncached_returns_output_and_no_pending() -> None:
    cm = _FakeCacheManager({})
    engine = _make_engine(cm)
    scheduler_output = SchedulerOutput(
        requests=[_make_request("r1"), _make_request("r2")],
        batch_data={"orig": True},
        step_id=7,
    )

    uncached, cached_pending = engine._filter_cached(scheduler_output)

    assert cached_pending is None
    assert uncached is not None
    assert [r.request_id for r in uncached.requests] == ["r1", "r2"]
    assert uncached.step_id == 7
    # batch_planner was invoked for the uncached subset
    assert engine.scheduler.batch_planner.build_count == 1


def test_filter_cached_empty_batch_returns_none_none() -> None:
    cm = _FakeCacheManager({})
    engine = _make_engine(cm)
    scheduler_output = SchedulerOutput(requests=[], batch_data=None, step_id=3)

    uncached, cached_pending = engine._filter_cached(scheduler_output)

    assert uncached is None
    assert cached_pending is None


def test_filter_cached_mixed_batch_resolves_cached_and_returns_uncached() -> None:
    """Regression test for #299: mixed batches must not drop cache-hit requests."""
    cached = {"r_hit": _make_output("r_hit", data={"from": "cache"})}
    cm = _FakeCacheManager(cached)
    engine = _make_engine(cm)
    scheduler_output = SchedulerOutput(
        requests=[
            _make_request("r_miss_a"),
            _make_request("r_hit"),
            _make_request("r_miss_b"),
        ],
        batch_data={"orig": True},
        step_id=42,
    )

    uncached, cached_pending = engine._filter_cached(scheduler_output)

    assert cached_pending is not None
    assert [r.request_id for r in cached_pending.scheduler_output.requests] == ["r_hit"]
    assert cached_pending.model_output.outputs["r_hit"].data == {"from": "cache"}
    assert cached_pending.update_cache is False

    assert uncached is not None
    assert [r.request_id for r in uncached.requests] == ["r_miss_a", "r_miss_b"]
    assert uncached.step_id == 42


def test_filter_cached_mixed_batch_then_apply_resolves_futures() -> None:
    """End-to-end: mixed batch → apply cached_pending → cached future resolved."""
    cached = {"r_hit": _make_output("r_hit")}
    cm = _FakeCacheManager(cached)
    engine = _make_engine(cm)
    hit_req = _make_request("r_hit")
    miss_req = _make_request("r_miss")
    scheduler_output = SchedulerOutput(
        requests=[miss_req, hit_req], batch_data={"orig": True}, step_id=1
    )

    uncached, cached_pending = engine._filter_cached(scheduler_output)
    assert cached_pending is not None
    engine._apply_pending_result(cached_pending)

    # Cache-hit request is now FINISHED — exactly the behavior missing in #299.
    assert hit_req.status == SchedulerStatus.FINISHED
    # Cache-miss request remains RUNNING until the model runner executes.
    assert miss_req.status == SchedulerStatus.RUNNING
    # scheduler.update was called exactly once with only the cached subset.
    assert len(engine.scheduler.update_calls) == 1
    assert [r.request_id for r in engine.scheduler.update_calls[0].requests] == [
        "r_hit"
    ]


def test_cached_pending_not_put_back_into_cache() -> None:
    """Applying cached_pending must NOT re-put already-cached outputs (update_cache=False)."""
    cached = {"r_hit": _make_output("r_hit")}
    cm = _FakeCacheManager(cached)
    engine = _make_engine(cm)
    scheduler_output = SchedulerOutput(
        requests=[_make_request("r_hit")], batch_data=None, step_id=1
    )

    _, cached_pending = engine._filter_cached(scheduler_output)
    assert cached_pending is not None
    engine._apply_pending_result(cached_pending)

    assert cm.put_calls == []


def test_non_running_cached_request_is_failed_not_silently_skipped() -> None:
    """If a cached request is not RUNNING, fail_request must resolve its future."""
    cached = {"stale": _make_output("stale")}
    cm = _FakeCacheManager(cached)
    engine = _make_engine(cm)
    stale_req = _make_request("stale", status=SchedulerStatus.ABORTED)
    scheduler_output = SchedulerOutput(requests=[stale_req], batch_data=None, step_id=1)

    _, cached_pending = engine._filter_cached(scheduler_output)

    assert cached_pending is not None
    assert len(engine.scheduler.failed) == 1
    failed_id, err = engine.scheduler.failed[0]
    assert failed_id == "stale"
    assert isinstance(err, RuntimeError)


def test_mixed_batch_in_overlap_mode_enqueues_cached_pending() -> None:
    """Simulate _step_overlap's cache filter step: cached_pending joins _result_queue."""
    cached = {"r_hit": _make_output("r_hit")}
    cm = _FakeCacheManager(cached)
    engine = _make_engine(cm)
    engine.enable_overlap = True
    scheduler_output = SchedulerOutput(
        requests=[_make_request("r_miss"), _make_request("r_hit")],
        batch_data={"orig": True},
        step_id=5,
    )

    uncached, cached_pending = engine._filter_cached(scheduler_output)

    assert cached_pending is not None
    engine._result_queue.append(cached_pending)
    # Simulate the overlap path's subsequent buffer of uncached result.
    uncached_model_output = ModelRunnerOutput(
        outputs={"r_miss": _make_output("r_miss")},
        req_ids=["r_miss"],
        req_id_to_index={"r_miss": 0},
    )
    assert uncached is not None
    engine._result_queue.append(
        _PendingResult(
            scheduler_output=uncached,
            model_output=uncached_model_output,
        )
    )

    # Drain: cached first (FIFO), then uncached.
    assert len(engine._result_queue) == 2
    engine._process_pending_result()
    engine._process_pending_result()

    # Both requests resolved; scheduler.update called exactly twice, cached first.
    assert [
        [r.request_id for r in call.requests] for call in engine.scheduler.update_calls
    ] == [["r_hit"], ["r_miss"]]


def test_pending_with_update_cache_true_puts_into_cache() -> None:
    """Sanity: regular (uncached) pending results DO hit cache_manager.put."""
    cm = _FakeCacheManager({})
    engine = _make_engine(cm)
    output = _make_output("r1")
    pending = _PendingResult(
        scheduler_output=SchedulerOutput(
            requests=[_make_request("r1")], batch_data=None, step_id=1
        ),
        model_output=ModelRunnerOutput(
            outputs={"r1": output}, req_ids=["r1"], req_id_to_index={"r1": 0}
        ),
        update_cache=True,
    )
    engine._apply_pending_result(pending)

    assert cm.put_calls == [("r1", output)]


# -----------------------------------------------------------------------------
# Integration test: drive _step_overlap end-to-end
# -----------------------------------------------------------------------------


class _FakeModelRunner:
    """Returns a predetermined ModelRunnerOutput per call."""

    def __init__(self, outputs_per_call: list[dict[str, RequestOutput]]) -> None:
        self._outputs_per_call = outputs_per_call
        self._call = 0
        # Force inline execution (no threadpool) so the test stays deterministic.
        self.execute_in_thread = False

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        outputs = self._outputs_per_call[self._call]
        self._call += 1
        req_ids = [r.request_id for r in scheduler_output.requests]
        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        )


def test_step_overlap_mixed_batch_preserves_update_order() -> None:
    """Drive _step_overlap with a mixed batch and verify FIFO scheduler.update ordering.

    This is the strongest coverage for the overlap ordering invariant — it
    actually runs the run_in_executor + await path rather than manually
    driving _result_queue.
    """
    hit_req = _make_request("r_hit")
    miss_req = _make_request("r_miss")
    scheduler_output = SchedulerOutput(
        requests=[miss_req, hit_req], batch_data={"orig": True}, step_id=1
    )

    # CacheManager hit for r_hit only.
    cm = _FakeCacheManager({"r_hit": _make_output("r_hit", data={"src": "cache"})})

    # ModelRunner will be called once, for the uncached subset [r_miss].
    mr = _FakeModelRunner(
        outputs_per_call=[
            {"r_miss": _make_output("r_miss", data={"src": "model"})},
        ]
    )

    engine = _make_engine(cm)
    engine.model_runner = mr
    engine.enable_overlap = True

    # Pre-load scheduler: simulate schedule() having returned this output already.
    # We monkeypatch scheduler.schedule to return the prepared output once then None.
    scheduler_outputs = [scheduler_output, None]

    def _fake_schedule() -> SchedulerOutput | None:
        return scheduler_outputs.pop(0) if scheduler_outputs else None

    engine.scheduler.schedule = _fake_schedule  # type: ignore[method-assign]

    # Run one overlap step, then drain the pending queue (cached + uncached).
    asyncio.new_event_loop().run_until_complete(engine._step_overlap())
    engine._drain_pending_results()

    # Both updates happened, cached first, then uncached, same step_id.
    call_reqs = [
        [r.request_id for r in call.requests] for call in engine.scheduler.update_calls
    ]
    assert call_reqs == [["r_hit"], ["r_miss"]]
    assert all(call.step_id == 1 for call in engine.scheduler.update_calls)

    # ModelRunner executed exactly once (on uncached subset only).
    assert mr._call == 1

    # Cached output was NOT re-put; uncached output WAS put.
    assert cm.put_calls == [("r_miss", _make_output("r_miss", data={"src": "model"}))]
