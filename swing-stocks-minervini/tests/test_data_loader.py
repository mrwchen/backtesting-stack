import pandas as pd
import pytest

from src import data_loader
from src.data_loader import (
    _normalize_quarterly_fundamental_events,
    _require_quarterly_fundamental_events,
    _validate_prices,
)

from .util import make_cfg


def test_quarterly_fundamental_events_keep_latest_economic_period_point_in_time():
    events = pd.DataFrame(
        [
            ("AAA", "2024-05-01", "2024-04-30T20:00:00Z", "q1", "2024-03-31", 1.20),
            ("AAA", "2024-08-01", "2024-07-31T20:00:00Z", "q2", "2024-06-30", 1.50),
            # A late Q1 amendment must not roll the current result back from Q2.
            ("AAA", "2024-09-01", "2024-08-31T20:00:00Z", "q1a", "2024-03-31", 1.30),
            # Same-period amendments remain eligible; latest acceptance wins
            # when several accessions become usable on the same day.
            ("AAA", "2024-10-01", "2024-09-30T20:00:00Z", "q2a", "2024-06-30", 1.60),
            ("AAA", "2024-10-01", "2024-09-30T21:00:00Z", "q2b", "2024-06-30", 1.70),
        ],
        columns=[
            "symbol",
            "available_date",
            "accepted_at",
            "accession_number",
            "fiscal_period_end_date",
            "diluted_eps",
        ],
    )
    events["prior_year_diluted_eps"] = 1.0

    normalized = _normalize_quarterly_fundamental_events(events)

    assert normalized["accession_number"].tolist() == ["q1", "q2", "q2b"]
    assert normalized["diluted_eps"].tolist() == [1.20, 1.50, 1.70]


def test_unknown_acceptance_cannot_override_known_same_day_filing():
    events = pd.DataFrame(
        [
            ("AAA", "2024-05-02", None, "zzz-unknown", "2024-03-31", 9.0),
            (
                "AAA",
                "2024-05-02",
                "2024-05-01T20:00:00Z",
                "aaa-known",
                "2024-03-31",
                1.5,
            ),
        ],
        columns=[
            "symbol",
            "available_date",
            "accepted_at",
            "accession_number",
            "fiscal_period_end_date",
            "diluted_eps",
        ],
    )
    events["prior_year_diluted_eps"] = 1.0

    normalized = _normalize_quarterly_fundamental_events(events)

    assert normalized["accession_number"].tolist() == ["aaa-known"]
    assert normalized["diluted_eps"].tolist() == [1.5]


def test_empty_quarterly_fundamental_source_is_an_integration_error():
    with pytest.raises(RuntimeError, match="schema init and startup historical backfill"):
        _require_quarterly_fundamental_events(pd.DataFrame())


def test_all_stock_inputs_use_the_same_full_security_identity():
    for query in (
        data_loader.PRICES_SQL,
        data_loader.UNIVERSE_SQL,
        data_loader.QUARTERLY_FUNDAMENTALS_SQL,
        data_loader.SPONSORSHIP_SQL,
    ):
        assert "canonical_identity" in query
        assert ".symbol" in query
        assert ".exchange" in query
        assert ".cik" in query


def test_prices_expose_raw_close_without_replacing_adjusted_technical_fields():
    assert "p.adjusted_close::float8                 AS close" in data_loader.PRICES_SQL
    assert "p.raw_close::float8                      AS raw_close" in data_loader.PRICES_SQL


def test_price_validation_rejects_ambiguous_duplicate_identity_dates():
    prices = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "open": [10.0, 10.0],
            "high": [11.0, 11.0],
            "low": [9.0, 9.0],
            "close": [10.5, 10.5],
            "raw_close": [10.5, 10.5],
            "volume": [1000.0, 1000.0],
        }
    )

    with pytest.raises(RuntimeError, match="duplicate symbol/date"):
        _validate_prices(prices)


def test_price_validation_discards_non_causal_or_missing_nominal_bars():
    prices = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02"]),
            "open": [10.0, 10.0, 20.0],
            "high": [11.0, 9.0, 21.0],
            "low": [9.0, 8.0, 19.0],
            "close": [10.5, 10.5, 20.5],
            "raw_close": [10.5, 10.5, None],
            "volume": [1000.0, 1000.0, 1000.0],
        }
    )

    clean = _validate_prices(prices)

    assert clean[["symbol", "date"]].to_dict("records") == [
        {"symbol": "AAA", "date": pd.Timestamp("2024-01-02")}
    ]


def test_cache_path_contains_explicit_contract_version(tmp_path):
    cfg = make_cfg(cache_dir=str(tmp_path))
    path = data_loader._cache_path(cfg, "prices_2020_2023")

    assert data_loader.CACHE_SCHEMA_VERSION in path
    assert path.endswith(".parquet")


def test_sponsorship_loader_reads_compact_identity_events_and_refreshes_cache_version():
    assert "stock_core_13f_sponsorship_identity_events" in data_loader.SPONSORSHIP_SQL
    assert "stock_core_13f_sponsorship_events e" not in data_loader.SPONSORSHIP_SQL


def test_sponsorship_loader_uses_v3_source_validated_cache(tmp_path, monkeypatch):
    observed = {}

    def read_df(_conn, query, _params):
        observed["query"] = query
        return pd.DataFrame(
            {
                "symbol": ["AAA"],
                "available_date": ["2023-12-29"],
                "manager_count_delta": [1.0],
                "net_activity_delta": [1.0],
            }
        )

    monkeypatch.setattr(data_loader.db, "read_df", read_df)
    def cached(_cfg, name, loader, **_kwargs):
        observed["cache_name"] = name
        observed["cache_kwargs"] = _kwargs
        return loader()

    monkeypatch.setattr(data_loader, "_cached", cached)
    cfg = make_cfg(
        cache_dir=str(tmp_path), force_refresh=True, end_date="2023-12-31"
    )

    result = data_loader.load_sponsorship_events(object(), cfg)

    assert len(result) == 1
    assert "stock_core_13f_sponsorship_identity_events" in observed["query"]
    assert observed["cache_name"] == "sponsorship_identity_v3_2023-12-31"
    assert observed["cache_kwargs"]["validate_source"] is True


def test_sponsorship_live_empty_source_cannot_be_hidden_by_cache(tmp_path, monkeypatch):
    calls = 0

    def source_changes(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return pd.DataFrame(
                {
                    "symbol": ["AAA"],
                    "available_date": ["2023-12-29"],
                    "manager_count_delta": [1.0],
                    "net_activity_delta": [1.0],
                }
            )
        return pd.DataFrame()

    monkeypatch.setattr(data_loader.db, "read_df", source_changes)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda *_args, **_kwargs: None)
    cfg = make_cfg(
        cache_dir=str(tmp_path),
        end_date="2023-12-31",
    )

    assert len(data_loader.load_sponsorship_events(object(), cfg)) == 1
    assert data_loader.load_sponsorship_events(object(), cfg).empty
    assert calls == 2


def test_market_index_loader_fails_without_every_configured_proxy(tmp_path, monkeypatch):
    def read_only_qqq(*_args, **_kwargs):
        return pd.DataFrame(
            [
                {
                    "symbol": "QQQ",
                    "date": "2024-01-02",
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1_000_000.0,
                }
            ]
        )

    monkeypatch.setattr(data_loader.db, "read_df", read_only_qqq)
    cfg = make_cfg(
        cache_dir=str(tmp_path),
        force_refresh=True,
        market_filter_enable=True,
        market_index_symbols=("QQQ", "VOO"),
        market_primary_index="QQQ",
        end_date="2024-01-31",
    )

    with pytest.raises(RuntimeError, match="required market indexes: VOO"):
        data_loader.load_market_indexes(object(), cfg)

    assert list(tmp_path.glob("market_indexes*.parquet")) == []


class _FakeConnection:
    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


def test_regime_read_failure_is_not_cached_when_filter_is_disabled(tmp_path, monkeypatch):
    def fail_read(*_args, **_kwargs):
        raise OSError("temporary DB failure")

    monkeypatch.setattr(data_loader.db, "read_df", fail_read)
    conn = _FakeConnection()
    cfg = make_cfg(
        cache_dir=str(tmp_path),
        regime_entry_filter_enable=False,
        end_date="2024-01-31",
    )

    result = data_loader.load_regime_scores(conn, cfg)

    assert result.empty
    assert conn.rollbacks == 1
    assert list(tmp_path.glob("regime*.parquet")) == []


def test_regime_read_failure_aborts_when_entry_filter_is_enabled(tmp_path, monkeypatch):
    def fail_read(*_args, **_kwargs):
        raise OSError("temporary DB failure")

    monkeypatch.setattr(data_loader.db, "read_df", fail_read)
    conn = _FakeConnection()
    cfg = make_cfg(
        cache_dir=str(tmp_path),
        regime_entry_filter_enable=True,
        end_date="2024-01-31",
    )

    with pytest.raises(RuntimeError, match="not readable"):
        data_loader.load_regime_scores(conn, cfg)

    assert conn.rollbacks == 1
    assert list(tmp_path.glob("regime*.parquet")) == []
