"""Parameter grid loading and deterministic candidate generation."""

from __future__ import annotations

import configparser
import hashlib
import itertools
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import RunConfig


DEFAULT_SAMPLING_SEED = 12345

PARAMETER_NAMES = [
    "LOOKBACK_BARS",
    "LONG_CROSS_QUANTILE",
    "SHORT_CROSS_QUANTILE",
    "ALL_STOP_MODES_TAKE_PROFIT_POINTS",
    "BAND_STOP_MIN_PROFILE_RANGE_POINTS",
    "BAND_STOP_PROFILE_LOWER_QUANTILE",
    "BAND_STOP_PROFILE_UPPER_QUANTILE",
    "BAND_STOP_PROFILE_BUFFER_POINTS",
    "BAND_STOP_MIN_DISTANCE_POINTS",
    "BAND_STOP_MAX_DISTANCE_POINTS",
]

INTEGER_PARAMETERS = {"LOOKBACK_BARS"}
QUANTILE_PARAMETERS = {
    "LONG_CROSS_QUANTILE",
    "SHORT_CROSS_QUANTILE",
    "BAND_STOP_PROFILE_LOWER_QUANTILE",
    "BAND_STOP_PROFILE_UPPER_QUANTILE",
}


def load_grid(path: str) -> dict[str, list[int | float]]:
    parser = configparser.ConfigParser()
    loaded = parser.read(path)
    if not loaded:
        raise RuntimeError(f"Parameter grid file not found: {Path(path).resolve()}")
    if "coarse" not in parser:
        raise RuntimeError("Parameter grid needs a [coarse] section")

    grid: dict[str, list[int | float]] = {}
    section = parser["coarse"]
    for name in PARAMETER_NAMES:
        if name not in section:
            raise RuntimeError(f"Parameter grid misses {name}")
        grid[name] = _parse_values(name, section[name])
    return grid


def build_stage1_candidates(
    grid: dict[str, list[int | float]],
    max_sets: int = 0,
    seed: int = DEFAULT_SAMPLING_SEED,
) -> list[dict[str, int | float]]:
    axis_values = [grid[name] for name in PARAMETER_NAMES]
    full_size = 1
    for values in axis_values:
        full_size *= len(values)
    if max_sets <= 0 or full_size <= max_sets:
        candidates = [
            dict(zip(PARAMETER_NAMES, values))
            for values in itertools.product(*axis_values)
        ]
        candidates = [candidate for candidate in candidates if is_valid(candidate)]
        return _dedupe_candidates(candidates)
    return _sample_candidates(axis_values, max_sets, seed)


def build_stage2_candidates(
    seed_candidates: Iterable[dict[str, int | float]],
    coarse_grid: dict[str, list[int | float]],
    previous_hashes: set[str],
    max_sets: int = 0,
    seed: int = DEFAULT_SAMPLING_SEED,
) -> list[dict[str, int | float]]:
    out: list[dict[str, int | float]] = []
    seen = set(previous_hashes)
    for candidate_seed in seed_candidates:
        local_values = [_local_values(name, candidate_seed[name], coarse_grid[name]) for name in PARAMETER_NAMES]
        for values in itertools.product(*local_values):
            candidate = dict(zip(PARAMETER_NAMES, values))
            if not is_valid(candidate):
                continue
            digest = parameter_hash(candidate)
            if digest in seen:
                continue
            seen.add(digest)
            out.append(candidate)
    return _limit_candidates(out, max_sets, seed)


def _sample_candidates(
    axis_values: list[list[int | float]],
    max_sets: int,
    seed: int,
) -> list[dict[str, int | float]]:
    """Seeded Latin-Hypercube sample over the grid axes (no full-product blow-up).

    Each parameter value is drawn with near-equal frequency (good marginal
    coverage), combined randomly across dimensions, then filtered and de-duped.
    Far better space-filling than the previous even-stride subsample, and the
    seed makes the candidate set fully reproducible.
    """
    rng = np.random.default_rng(seed)
    seen: set[str] = set()
    out: list[dict[str, int | float]] = []
    max_attempts = max(max_sets * 8, max_sets + 1000)
    attempts = 0
    while len(out) < max_sets and attempts < max_attempts:
        for values in _lhs_draw(axis_values, max_sets, rng):
            attempts += 1
            candidate = dict(zip(PARAMETER_NAMES, values))
            if not is_valid(candidate):
                continue
            digest = parameter_hash(candidate)
            if digest in seen:
                continue
            seen.add(digest)
            out.append(candidate)
            if len(out) >= max_sets:
                break
    out.sort(key=parameter_signature)
    return out


def _lhs_draw(
    axis_values: list[list[int | float]],
    n: int,
    rng: np.random.Generator,
) -> list[tuple[int | float, ...]]:
    columns: list[np.ndarray] = []
    for values in axis_values:
        k = len(values)
        reps = -(-n // k)  # ceil division
        pool = np.tile(np.arange(k), reps)[:n]
        rng.shuffle(pool)
        columns.append(pool)
    return [
        tuple(axis_values[d][int(columns[d][row])] for d in range(len(axis_values)))
        for row in range(n)
    ]


def is_valid(values: dict[str, int | float]) -> bool:
    try:
        if int(values["LOOKBACK_BARS"]) < 1:
            return False
        if not 0.0 <= float(values["LONG_CROSS_QUANTILE"]) <= 1.0:
            return False
        if not 0.0 <= float(values["SHORT_CROSS_QUANTILE"]) <= 1.0:
            return False
        if float(values["ALL_STOP_MODES_TAKE_PROFIT_POINTS"]) <= 0:
            return False
        if float(values["BAND_STOP_MIN_PROFILE_RANGE_POINTS"]) < 0:
            return False
        if not 0.0 <= float(values["BAND_STOP_PROFILE_LOWER_QUANTILE"]) < float(values["BAND_STOP_PROFILE_UPPER_QUANTILE"]) <= 1.0:
            return False
        if float(values["BAND_STOP_PROFILE_BUFFER_POINTS"]) < 0:
            return False
        if float(values["BAND_STOP_MIN_DISTANCE_POINTS"]) <= 0:
            return False
        if float(values["BAND_STOP_MAX_DISTANCE_POINTS"]) <= float(values["BAND_STOP_MIN_DISTANCE_POINTS"]):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def parameter_hash(values: dict[str, int | float]) -> str:
    text = parameter_signature(values)
    return hashlib.sha1(text.encode("ascii")).hexdigest()


def parameter_signature(values: dict[str, int | float]) -> str:
    parts = []
    for name in PARAMETER_NAMES:
        parts.append(f"{name}:{_format_value(values[name])}")
    return "|".join(parts)


def parameter_label(values: dict[str, int | float]) -> str:
    return (
        f"lb{int(values['LOOKBACK_BARS'])}_"
        f"lq{_format_value(values['LONG_CROSS_QUANTILE'])}_"
        f"sq{_format_value(values['SHORT_CROSS_QUANTILE'])}_"
        f"alltp{_format_value(values['ALL_STOP_MODES_TAKE_PROFIT_POINTS'])}_"
        f"bandrange{_format_value(values['BAND_STOP_MIN_PROFILE_RANGE_POINTS'])}_"
        f"bandq{_format_value(values['BAND_STOP_PROFILE_LOWER_QUANTILE'])}-{_format_value(values['BAND_STOP_PROFILE_UPPER_QUANTILE'])}_"
        f"bandbuf{_format_value(values['BAND_STOP_PROFILE_BUFFER_POINTS'])}_"
        f"bandstop{_format_value(values['BAND_STOP_MIN_DISTANCE_POINTS'])}-{_format_value(values['BAND_STOP_MAX_DISTANCE_POINTS'])}"
    )


def values_from_config(cfg: RunConfig) -> dict[str, int | float]:
    return {
        "LOOKBACK_BARS": cfg.lookback_bars,
        "LONG_CROSS_QUANTILE": cfg.long_cross_quantile,
        "SHORT_CROSS_QUANTILE": cfg.short_cross_quantile,
        "ALL_STOP_MODES_TAKE_PROFIT_POINTS": cfg.take_profit_points,
        "BAND_STOP_MIN_PROFILE_RANGE_POINTS": cfg.min_profile_range_points,
        "BAND_STOP_PROFILE_LOWER_QUANTILE": cfg.stop_profile_lower_quantile,
        "BAND_STOP_PROFILE_UPPER_QUANTILE": cfg.stop_profile_upper_quantile,
        "BAND_STOP_PROFILE_BUFFER_POINTS": cfg.stop_profile_buffer_points,
        "BAND_STOP_MIN_DISTANCE_POINTS": cfg.min_stop_distance_points,
        "BAND_STOP_MAX_DISTANCE_POINTS": cfg.max_stop_distance_points,
    }


def _parse_values(name: str, raw: str) -> list[int | float]:
    values: list[int | float] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        if name in INTEGER_PARAMETERS:
            value: int | float = int(float(text))
        else:
            value = round(float(text), 6)
        values.append(value)
    if not values:
        raise RuntimeError(f"Parameter grid {name} has no values")
    return sorted(set(values))


def _local_values(name: str, value: int | float, coarse_values: list[int | float]) -> list[int | float]:
    values = sorted(set(coarse_values))
    current = int(value) if name in INTEGER_PARAMETERS else float(value)
    out = {current}
    lowers = [item for item in values if item < current]
    uppers = [item for item in values if item > current]
    if lowers:
        out.add(_midpoint(name, lowers[-1], current))
    if uppers:
        out.add(_midpoint(name, current, uppers[0]))
    if name in QUANTILE_PARAMETERS:
        return sorted(_clamp_quantile(float(item)) for item in out)
    return sorted(out)


def _midpoint(name: str, left: int | float, right: int | float) -> int | float:
    value = (float(left) + float(right)) / 2.0
    if name in INTEGER_PARAMETERS:
        return max(1, int(round(value)))
    return round(value, 6)


def _clamp_quantile(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 6)


def _dedupe_candidates(candidates: Iterable[dict[str, int | float]]) -> list[dict[str, int | float]]:
    seen: set[str] = set()
    out: list[dict[str, int | float]] = []
    for candidate in candidates:
        digest = parameter_hash(candidate)
        if digest in seen:
            continue
        seen.add(digest)
        out.append(candidate)
    return out


def _limit_candidates(
    candidates: list[dict[str, int | float]],
    max_sets: int,
    seed: int = DEFAULT_SAMPLING_SEED,
) -> list[dict[str, int | float]]:
    if max_sets <= 0 or len(candidates) <= max_sets:
        return candidates
    rng = np.random.default_rng(seed)
    indexes = sorted(rng.choice(len(candidates), size=max_sets, replace=False).tolist())
    return [candidates[index] for index in indexes]


def _format_value(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"
