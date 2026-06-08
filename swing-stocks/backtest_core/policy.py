"""Common portfolio policy helpers for the backtester."""

from .config import MAX_OPEN_POSITIONS


def direction_risk_multiplier(exposure: dict, direction: str) -> float:
    return float(exposure[f"{direction.lower()}_risk_multiplier"])


def direction_max_positions(exposure: dict, direction: str) -> int:
    return int(exposure[f"max_{direction.lower()}_positions"])


def exposure_max_total_positions(exposure: dict) -> int:
    return int(exposure.get("max_total_positions", MAX_OPEN_POSITIONS))
