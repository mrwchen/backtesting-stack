"""Fixed development-only market-filter ablation for the Minervini model."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from .config import Config


DEVELOPMENT_START_DATE = date(2020, 1, 2)
DEVELOPMENT_END_DATE = date(2023, 12, 31)


@dataclass(frozen=True)
class SensitivityVariant:
    name: str
    market_filter_enable: bool

    @property
    def detection_key(self) -> str:
        return "shared_model"

    def apply(self, cfg: Config, phase: str, start: date, end: date) -> Config:
        return replace(
            cfg,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            run_label=f"{cfg.run_label}_{phase}_{self.name}",
            market_filter_enable=self.market_filter_enable,
        )


VARIANTS = (
    SensitivityVariant("market_off", False),
    SensitivityVariant("market_on", True),
)


def validate_configured_window(start: date, end: date) -> None:
    """Require the frozen development window before any source data is loaded."""
    if start != DEVELOPMENT_START_DATE or end != DEVELOPMENT_END_DATE:
        raise ValueError(
            "market-filter ablation requires 2020-01-02 through 2023-12-31"
        )


def phases(start: date, end: date) -> tuple[tuple[str, date, date], ...]:
    """Return the development period and reject any already-inspected OOS data."""
    if start > end:
        raise ValueError("sensitivity start date must not be after end date")
    if end > DEVELOPMENT_END_DATE:
        raise ValueError(
            "market-filter ablation is development-only through 2023-12-31"
        )
    return (("dev", start, end),)
