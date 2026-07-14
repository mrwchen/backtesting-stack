"""Frozen v9 forward-shadow protocol.

The forward boundary and every control salt are prescribed before the first
eligible session.  The relative-quality shadow is one deterministic path.  The
neutral controls vary only the causal tie-break lottery; they do not resample,
reweight or otherwise alter calibration labels.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np


HISTORY_START_DATE = date(2020, 1, 2)
FORWARD_START_DATE = date(2026, 7, 13)
RANKING_MODES = ("neutral", "relative_quality")
PORTFOLIO_SETUP_TYPES = ("flat_base",)

RELATIVE_QUALITY_SHADOW_SALT = "v9-relative-quality-shadow"
NEUTRAL_CONTROL_SALTS = tuple(
    f"v9-neutral-control-{index:02d}" for index in range(32)
)


@dataclass(frozen=True)
class ForwardPortfolioCase:
    """One predeclared Flat-Base portfolio path in the v9 protocol."""

    name: str
    role: str
    ranking_mode: str
    setup_types: tuple[str, ...]
    neutral_rank_salt: str


RELATIVE_QUALITY_SHADOW_CASE = ForwardPortfolioCase(
    name="relative_quality_flat_base_shadow",
    role="shadow",
    ranking_mode="relative_quality",
    setup_types=PORTFOLIO_SETUP_TYPES,
    neutral_rank_salt=RELATIVE_QUALITY_SHADOW_SALT,
)

NEUTRAL_CONTROL_CASES = tuple(
    ForwardPortfolioCase(
        name=f"neutral_flat_base_control_{index:02d}",
        role="control",
        ranking_mode="neutral",
        setup_types=PORTFOLIO_SETUP_TYPES,
        neutral_rank_salt=salt,
    )
    for index, salt in enumerate(NEUTRAL_CONTROL_SALTS)
)

PORTFOLIO_CASES = (RELATIVE_QUALITY_SHADOW_CASE, *NEUTRAL_CONTROL_CASES)

CORE_METRICS = (
    "total_return",
    "cagr",
    "max_drawdown",
    "profit_factor",
    "avg_r_multiple",
)


def summarize_controls(
    results: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float | int]]:
    """Summarize the prescribed neutral controls without selecting a path."""
    if len(results) != len(NEUTRAL_CONTROL_CASES):
        raise ValueError(
            f"expected {len(NEUTRAL_CONTROL_CASES)} neutral-control results, "
            f"got {len(results)}"
        )

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
        summary[metric] = {
            "count": int(values.size),
            "missing_count": len(results) - int(values.size),
            "median": float(np.median(values)),
            "adverse_quantile": float(
                np.quantile(values, 0.90 if higher_is_worse else 0.10)
            ),
            "worst": float(values.max() if higher_is_worse else values.min()),
        }
    return summary
