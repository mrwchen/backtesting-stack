"""Parameter grid loading and deterministic candidate generation."""

from __future__ import annotations

import configparser
import hashlib
import itertools
import math
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple

import numpy as np

from .sessions import SESSION_TYPES


DEFAULT_SAMPLING_SEED = 12345

PARAMETER_NAMES = [
    "LOOKBACK_BARS",
    "LONG_CROSS_QUANTILE",
    "SHORT_CROSS_QUANTILE",
    "ENTRY_PRICE_RANGE_POSITION_MAX_DEVIATION_PCT",
    "ALL_STOP_MODES_TAKE_PROFIT_POINTS",
    "BAND_STOP_MIN_PROFILE_RANGE_POINTS",
    "BAND_STOP_PROFILE_LOWER_QUANTILE",
    "BAND_STOP_PROFILE_UPPER_QUANTILE",
    "BAND_STOP_PROFILE_BUFFER_POINTS",
    "BAND_STOP_MIN_DISTANCE_POINTS",
    "BAND_STOP_MAX_DISTANCE_POINTS",
]

PROFILE_PARAMETER_NAMES = [
    "LOOKBACK_BARS",
    "LONG_CROSS_QUANTILE",
    "SHORT_CROSS_QUANTILE",
    "BAND_STOP_PROFILE_LOWER_QUANTILE",
    "BAND_STOP_PROFILE_UPPER_QUANTILE",
]
NON_PROFILE_PARAMETER_NAMES = [name for name in PARAMETER_NAMES if name not in PROFILE_PARAMETER_NAMES]

INTEGER_PARAMETERS = {"LOOKBACK_BARS"}
QUANTILE_PARAMETERS = {
    "LONG_CROSS_QUANTILE",
    "SHORT_CROSS_QUANTILE",
    "BAND_STOP_PROFILE_LOWER_QUANTILE",
    "BAND_STOP_PROFILE_UPPER_QUANTILE",
}


class CandidateBatch(NamedTuple):
    start_index: int
    candidates: list[dict[str, int | float]]


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


def load_single_session_parameters(path: str) -> dict[str, dict[str, int | float]]:
    parser = configparser.ConfigParser()
    loaded = parser.read(path)
    if not loaded:
        raise RuntimeError(f"Single parameter file not found: {Path(path).resolve()}")

    expected_sections = {session_type for session_type, _label, _sort_order in SESSION_TYPES}
    actual_sections = set(parser.sections())
    unknown_sections = sorted(actual_sections - expected_sections)
    if unknown_sections:
        raise RuntimeError(f"Single parameter file has unknown session sections: {', '.join(unknown_sections)}")

    missing_sections = [session_type for session_type, _label, _sort_order in SESSION_TYPES if session_type not in parser]
    if missing_sections:
        raise RuntimeError(f"Single parameter file misses session sections: {', '.join(missing_sections)}")

    out: dict[str, dict[str, int | float]] = {}
    for session_type, _label, _sort_order in SESSION_TYPES:
        section = parser[session_type]
        values: dict[str, int | float] = {}
        for name in PARAMETER_NAMES:
            if name not in section:
                raise RuntimeError(f"Single parameter file section [{session_type}] misses {name}")
            values[name] = _parse_single_value(name, section[name], session_type)
        if not is_valid(values):
            raise RuntimeError(f"Single parameter file section [{session_type}] has invalid parameter values")
        out[session_type] = values
    return out


def stage1_candidate_count(grid: dict[str, list[int | float]], max_sets: int = 0) -> int:
    full_size = _valid_stage1_grid_size(grid)
    if max_sets <= 0:
        return full_size
    return min(full_size, int(max_sets))


def iter_stage1_candidate_batches(
    grid: dict[str, list[int | float]],
    max_sets: int = 0,
    seed: int = DEFAULT_SAMPLING_SEED,
    batch_size: int = 65_536,
) -> Iterator[CandidateBatch]:
    batch_size = max(1, int(batch_size))
    target_count = stage1_candidate_count(grid, max_sets)
    if target_count <= 0:
        return

    profile_combos = _profile_combinations(grid)
    non_profile_combos = _non_profile_combinations(grid)
    if not profile_combos or not non_profile_combos:
        return

    full_size = len(profile_combos) * len(non_profile_combos)
    sample = target_count < full_size
    base_take, remainder = divmod(target_count, len(profile_combos))

    yielded = 0
    batch: list[dict[str, int | float]] = []
    batch_start = 0
    for profile_index, profile_values in enumerate(profile_combos):
        take = base_take + (1 if profile_index < remainder else 0)
        if take <= 0:
            continue
        if sample:
            non_profile_indexes = _lcg_indexes(len(non_profile_combos), take, seed + profile_index * 104_729)
        else:
            non_profile_indexes = range(len(non_profile_combos))

        profile_dict = dict(zip(PROFILE_PARAMETER_NAMES, profile_values))
        for non_profile_index in non_profile_indexes:
            non_profile_dict = dict(zip(NON_PROFILE_PARAMETER_NAMES, non_profile_combos[non_profile_index]))
            candidate = {**profile_dict, **non_profile_dict}
            if not is_valid(candidate):
                continue
            if not batch:
                batch_start = yielded
            batch.append(candidate)
            yielded += 1
            if len(batch) >= batch_size:
                yield CandidateBatch(batch_start, batch)
                batch = []
    if batch:
        yield CandidateBatch(batch_start, batch)


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


def _valid_stage1_grid_size(grid: dict[str, list[int | float]]) -> int:
    profile_count = len(_profile_combinations(grid))
    non_profile_count = len(_non_profile_combinations(grid))
    return profile_count * non_profile_count


def _profile_combinations(grid: dict[str, list[int | float]]) -> list[tuple[int | float, ...]]:
    combos = []
    for values in itertools.product(*(grid[name] for name in PROFILE_PARAMETER_NAMES)):
        candidate = dict(zip(PROFILE_PARAMETER_NAMES, values))
        lower = float(candidate["BAND_STOP_PROFILE_LOWER_QUANTILE"])
        upper = float(candidate["BAND_STOP_PROFILE_UPPER_QUANTILE"])
        if 0.0 <= lower < upper <= 1.0:
            combos.append(values)
    return combos


def _non_profile_combinations(grid: dict[str, list[int | float]]) -> list[tuple[int | float, ...]]:
    combos = []
    for values in itertools.product(*(grid[name] for name in NON_PROFILE_PARAMETER_NAMES)):
        candidate = dict(zip(NON_PROFILE_PARAMETER_NAMES, values))
        min_stop = float(candidate["BAND_STOP_MIN_DISTANCE_POINTS"])
        max_stop = float(candidate["BAND_STOP_MAX_DISTANCE_POINTS"])
        if min_stop > 0.0 and max_stop > min_stop:
            combos.append(values)
    return combos


def _lcg_indexes(size: int, take: int, seed: int) -> Iterator[int]:
    if take >= size:
        yield from range(size)
        return

    rng = np.random.default_rng(seed)
    offset = int(rng.integers(0, size))
    step = int(rng.integers(1, size))
    while math.gcd(step, size) != 1:
        step += 1
        if step >= size:
            step = 1
    for index in range(take):
        yield (offset + index * step) % size


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
        if float(values["ENTRY_PRICE_RANGE_POSITION_MAX_DEVIATION_PCT"]) < 0:
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
        f"rangepos{_format_value(values['ENTRY_PRICE_RANGE_POSITION_MAX_DEVIATION_PCT'])}_"
        f"alltp{_format_value(values['ALL_STOP_MODES_TAKE_PROFIT_POINTS'])}_"
        f"bandrange{_format_value(values['BAND_STOP_MIN_PROFILE_RANGE_POINTS'])}_"
        f"bandq{_format_value(values['BAND_STOP_PROFILE_LOWER_QUANTILE'])}-{_format_value(values['BAND_STOP_PROFILE_UPPER_QUANTILE'])}_"
        f"bandbuf{_format_value(values['BAND_STOP_PROFILE_BUFFER_POINTS'])}_"
        f"bandstop{_format_value(values['BAND_STOP_MIN_DISTANCE_POINTS'])}-{_format_value(values['BAND_STOP_MAX_DISTANCE_POINTS'])}"
    )


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


def _parse_single_value(name: str, raw: str, session_type: str) -> int | float:
    values = _parse_values(name, raw)
    if len(values) != 1:
        raise RuntimeError(f"Single parameter file section [{session_type}] {name} must contain exactly one value")
    return values[0]


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
