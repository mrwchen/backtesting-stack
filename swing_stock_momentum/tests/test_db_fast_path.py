from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from swing_stock_momentum.config import Config
from swing_stock_momentum.contracts import ANALYSER_CRITERION_COLUMNS
from swing_stock_momentum.db import (
    SnapshotMetadata,
    _candidate_query,
    _market_query,
    iter_market_days,
)


class _Cursor:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows
        self.offset = 0
        self.itersize = 0
        self.executed_params: tuple[Any, ...] | None = None

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, _: object, params: tuple[Any, ...]) -> None:
        self.executed_params = params

    def __iter__(self):
        return iter(self.rows)

    def fetchmany(self, size: int) -> list[Any]:
        batch = self.rows[self.offset : self.offset + size]
        self.offset += len(batch)
        return batch


class _Connection:
    def __init__(self, cursors: list[_Cursor]) -> None:
        self.cursors = cursors

    def cursor(self, *_: object, **__: object) -> _Cursor:
        return self.cursors.pop(0)


def test_market_query_is_compact_and_candidate_query_filters_all_criteria() -> None:
    cfg = Config.from_env()

    market_repr = repr(_market_query(cfg))
    candidate_repr = repr(_candidate_query(cfg))

    assert "LEFT JOIN" not in market_repr
    assert "analyser__" not in market_repr
    assert "trend_template_pass" in candidate_repr
    assert "daily_price_change_pct" in candidate_repr
    assert "adjusted_volume_vs_sma21_prior_ratio" in candidate_repr
    for column in ANALYSER_CRITERION_COLUMNS:
        assert column in candidate_repr


def test_compact_market_rows_receive_only_matching_analyser_candidate() -> None:
    cfg = Config.from_env()
    valuation_date = cfg.strategy.requested_start_date
    candidate = {
        "period_end_date": valuation_date,
        "symbol": "A",
        "exchange": "NASDAQ",
        "cik": 1,
        "price_continuity_segment": 1,
        "marker": "candidate",
    }
    market_rows = [
        (
            valuation_date,
            "A",
            "NASDAQ",
            1,
            1,
            Decimal("99"),
            Decimal("102"),
            Decimal("98"),
            Decimal("100"),
            Decimal("3"),
            cfg.strategy.prior_high_lookback_sessions,
            Decimal("105"),
        ),
        (
            valuation_date,
            "B",
            "NYSE",
            2,
            1,
            Decimal("49"),
            Decimal("52"),
            Decimal("48"),
            Decimal("50"),
            Decimal("4"),
            cfg.strategy.prior_high_lookback_sessions,
            Decimal("55"),
        ),
    ]
    candidate_cursor = _Cursor([candidate])
    market_cursor = _Cursor(market_rows)
    connection = _Connection([candidate_cursor, market_cursor])
    metadata = SnapshotMetadata(
        source_watermark_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        analyser_watermark_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_end_date=valuation_date,
        analyser_end_date=valuation_date,
        lookback_start_date=valuation_date,
        source_row_count=2,
        analyser_row_count=2,
    )

    days = list(iter_market_days(connection, cfg, metadata))  # type: ignore[arg-type]

    assert len(days) == 1
    assert [bar.symbol for bar in days[0][1]] == ["A", "B"]
    assert days[0][1][0].analyser == candidate
    assert days[0][1][1].analyser is None
    assert candidate_cursor.executed_params == (
        valuation_date,
        valuation_date,
        cfg.strategy.currency,
        cfg.strategy.min_daily_price_change_pct,
        cfg.strategy.max_daily_price_change_pct_exclusive,
        cfg.strategy.min_volume_vs_sma21_ratio_exclusive,
    )

