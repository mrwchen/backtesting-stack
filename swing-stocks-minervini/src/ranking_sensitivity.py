"""Frozen neutral-ranking salts and distribution summaries.

The salts are deliberately prescribed here.  They are not sampled at runtime
and the runner never selects a winning salt.  Each salt therefore represents
one reproducible, full path-dependent portfolio simulation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


NEUTRAL_RANK_SALTS = (
    "v7-neutral-00",
    "v7-neutral-01",
    "v7-neutral-02",
    "v7-neutral-03",
    "v7-neutral-04",
    "v7-neutral-05",
    "v7-neutral-06",
    "v7-neutral-07",
    "v7-neutral-08",
    "v7-neutral-09",
    "v7-neutral-10",
    "v7-neutral-11",
    "v7-neutral-12",
    "v7-neutral-13",
    "v7-neutral-14",
    "v7-neutral-15",
    "v7-neutral-16",
    "v7-neutral-17",
    "v7-neutral-18",
    "v7-neutral-19",
    "v7-neutral-20",
    "v7-neutral-21",
    "v7-neutral-22",
    "v7-neutral-23",
    "v7-neutral-24",
    "v7-neutral-25",
    "v7-neutral-26",
    "v7-neutral-27",
    "v7-neutral-28",
    "v7-neutral-29",
    "v7-neutral-30",
    "v7-neutral-31",
)

CORE_METRICS = (
    "total_return",
    "cagr",
    "max_drawdown",
    "profit_factor",
    "avg_r_multiple",
)


def summarize(
    results: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float | int]]:
    """Return median, adverse decile and worst for finite core metrics.

    Lower is adverse for returns, CAGR, profit factor and average R, so their
    adverse decile is p10. Higher drawdown is adverse, so its decile is p90.
    """
    summary: dict[str, dict[str, float | int]] = {}
    for metric in CORE_METRICS:
        values = np.asarray(
            [
                float(item[metric])
                for item in results
                if item.get(metric) is not None
                and np.isfinite(float(item[metric]))
            ],
            dtype=float,
        )
        if values.size == 0:
            continue
        higher_is_worse = metric == "max_drawdown"
        worst = values.max() if higher_is_worse else values.min()
        summary[metric] = {
            "count": int(values.size),
            "missing_count": len(results) - int(values.size),
            "median": float(np.median(values)),
            "adverse_quantile": float(
                np.quantile(values, 0.90 if higher_is_worse else 0.10)
            ),
            "worst": float(worst),
        }
    return summary
