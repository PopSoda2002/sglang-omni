# SPDX-License-Identifier: Apache-2.0
"""Realtime API latency baselines, keyed by hardware tag.

Baselines live in Python (not JSON) so CI regression checks can ``from
benchmarks.baselines.realtime_latency import BASELINES`` without
parsing a sibling file. Add a new entry per host class as benchmarks
are run; the keys follow the GPU SKU we benchmarked on.

Each ``mode`` entry is the dict produced by
``benchmarks/realtime_latency.py`` ``_summarize`` — first-delta and
completed latency in milliseconds, with sample size, p50, p95, max,
and min.
"""

from __future__ import annotations

from typing import Any

BASELINES: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
    "h200": {
        "manual": {
            "first_delta_after_commit_ms": {
                "n": 5,
                "p50_ms": 190.513315028511,
                "p95_ms": 303.35272665834054,
                "max_ms": 285.5347649892792,
                "min_ms": 177.36815800890326,
            },
            "completed_after_commit_ms": {
                "n": 5,
                "p50_ms": 190.5804219422862,
                "p95_ms": 303.43148878309876,
                "max_ms": 285.61520704533905,
                "min_ms": 177.4345439625904,
            },
        },
    },
}
