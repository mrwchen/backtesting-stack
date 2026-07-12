"""Fixed, bounded strictness sensitivity design for the Minervini model."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta

from .config import Config


SENSITIVITY_SPLIT_DATE = date(2024, 1, 1)


@dataclass(frozen=True)
class SensitivityVariant:
    name: str
    vcp_score_min: float
    dryup_ratio_min: float
    dryup_ratio_max: float

    @property
    def detection_key(self) -> tuple[float, float, float]:
        return (
            self.vcp_score_min,
            self.dryup_ratio_min,
            self.dryup_ratio_max,
        )

    def apply(self, cfg: Config, phase: str, start: date, end: date) -> Config:
        return replace(
            cfg,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            run_label=f"{cfg.run_label}_{phase}_{self.name}",
            vcp_score_min=self.vcp_score_min,
            dryup_ratio_min=self.dryup_ratio_min,
            dryup_ratio_max=self.dryup_ratio_max,
            market_filter_enable=False,
        )


VARIANTS = (
    SensitivityVariant("baseline", 65.0, 0.50, 0.70),
    SensitivityVariant("score60", 60.0, 0.50, 0.70),
    SensitivityVariant("score55", 55.0, 0.50, 0.70),
    SensitivityVariant("dryup_low20", 65.0, 0.20, 0.70),
    SensitivityVariant("dryup_high85", 65.0, 0.50, 0.85),
    SensitivityVariant("moderate", 60.0, 0.20, 0.85),
)


def phases(start: date, end: date) -> tuple[tuple[str, date, date], ...]:
    """Return fixed development and held-out reporting periods."""
    if not start < SENSITIVITY_SPLIT_DATE <= end:
        raise ValueError(
            "sensitivity data must span both sides of 2024-01-01"
        )
    return (
        ("dev", start, SENSITIVITY_SPLIT_DATE - timedelta(days=1)),
        ("oos", SENSITIVITY_SPLIT_DATE, end),
    )
