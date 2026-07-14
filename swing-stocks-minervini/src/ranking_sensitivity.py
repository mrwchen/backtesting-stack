"""Frozen v8 ranking experiment cases, bootstrap salts and summaries.

The salts are deliberately prescribed here.  They are not sampled at runtime
and the runner never selects a winning salt.  Each salt therefore represents
one reproducible completion-day-cluster Bayesian-bootstrap draw and one full
path-dependent portfolio simulation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import blake2b
from math import log

import numpy as np

from .candidate_ranking import QualityCalibrationLabel


RANKING_MODES = ("neutral", "quality_only", "validated")

_SLEEVES = (
    ("flat_base", ("flat_base",)),
    ("vcp", ("vcp",)),
    ("combined", ("flat_base", "vcp")),
)


@dataclass(frozen=True)
class RankingExperimentCase:
    """One predeclared ranking-mode and portfolio-sleeve combination."""

    name: str
    ranking_mode: str
    sleeve: str
    setup_types: tuple[str, ...]


EXPERIMENT_CASES = tuple(
    RankingExperimentCase(
        name=f"{ranking_mode}_{sleeve}",
        ranking_mode=ranking_mode,
        sleeve=sleeve,
        setup_types=setup_types,
    )
    for ranking_mode in RANKING_MODES
    for sleeve, setup_types in _SLEEVES
)


NEUTRAL_RANK_SALTS = (
    "v8-bootstrap-00",
    "v8-bootstrap-01",
    "v8-bootstrap-02",
    "v8-bootstrap-03",
    "v8-bootstrap-04",
    "v8-bootstrap-05",
    "v8-bootstrap-06",
    "v8-bootstrap-07",
    "v8-bootstrap-08",
    "v8-bootstrap-09",
    "v8-bootstrap-10",
    "v8-bootstrap-11",
    "v8-bootstrap-12",
    "v8-bootstrap-13",
    "v8-bootstrap-14",
    "v8-bootstrap-15",
    "v8-bootstrap-16",
    "v8-bootstrap-17",
    "v8-bootstrap-18",
    "v8-bootstrap-19",
    "v8-bootstrap-20",
    "v8-bootstrap-21",
    "v8-bootstrap-22",
    "v8-bootstrap-23",
    "v8-bootstrap-24",
    "v8-bootstrap-25",
    "v8-bootstrap-26",
    "v8-bootstrap-27",
    "v8-bootstrap-28",
    "v8-bootstrap-29",
    "v8-bootstrap-30",
    "v8-bootstrap-31",
)

CORE_METRICS = (
    "total_return",
    "cagr",
    "max_drawdown",
    "profit_factor",
    "avg_r_multiple",
)


def _cluster_bootstrap_draw(
    salt: str,
    setup_type: str,
    available_date: object,
) -> float:
    """Return one deterministic positive Exp(1) draw for a completion cluster."""
    identity = "\x1f".join(
        (
            salt,
            setup_type,
            np.datetime_as_string(np.datetime64(available_date, "D"), unit="D"),
        )
    ).encode("utf-8")
    digest = blake2b(
        identity,
        digest_size=8,
        person=b"minervini-v8-bb",
    ).digest()
    # Retain the top 52 bits so both half-step endpoints are exactly
    # representable on every IEEE-754 binary64 platform.  The resulting
    # uniform draw is therefore strictly inside (0, 1).
    integer = int.from_bytes(digest, byteorder="big", signed=False) >> 12
    uniform = (integer + 0.5) / float(1 << 52)
    return -log(uniform)


def bootstrap_quality_labels(
    labels: Sequence[QualityCalibrationLabel],
    salt: str,
) -> tuple[QualityCalibrationLabel, ...]:
    """Reweight quality labels with a deterministic clustered Bayesian bootstrap.

    Labels from the same setup class and completion day receive one common
    multiplier, so correlated same-day exits never masquerade as independent
    bootstrap observations.  Each Exp(1) draw depends only on its salt and
    cluster identity.  It is deliberately not normalized against the complete
    label collection: doing so would let future completion clusters change an
    earlier label's weight. Fill labels are outside this function and unchanged.
    """
    if not isinstance(salt, str):
        raise TypeError("bootstrap salt must be a string")
    if not salt:
        raise ValueError("bootstrap salt must not be empty")
    if not labels:
        return ()

    cluster_keys = {
        (label.setup_type, label.available_date.date().isoformat())
        for label in labels
    }
    multipliers = {
        key: _cluster_bootstrap_draw(salt, key[0], key[1])
        for key in sorted(cluster_keys)
    }
    return tuple(
        replace(
            label,
            weight=label.weight
            * multipliers[
                (label.setup_type, label.available_date.date().isoformat())
            ],
        )
        for label in labels
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
