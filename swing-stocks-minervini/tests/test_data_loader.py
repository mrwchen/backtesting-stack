import pandas as pd
import pytest

from src import data_loader
from src.data_loader import _normalize_quarterly_eps_events, _require_quarterly_eps_events

from .util import make_cfg


def test_quarterly_eps_events_keep_latest_economic_period_point_in_time():
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

    normalized = _normalize_quarterly_eps_events(events)

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

    normalized = _normalize_quarterly_eps_events(events)

    assert normalized["accession_number"].tolist() == ["aaa-known"]
    assert normalized["diluted_eps"].tolist() == [1.5]


def test_empty_quarterly_eps_source_is_an_integration_error():
    with pytest.raises(RuntimeError, match="schema init and startup historical backfill"):
        _require_quarterly_eps_events(pd.DataFrame())


def test_all_stock_inputs_use_the_same_full_security_identity():
    for query in (
        data_loader.PRICES_SQL,
        data_loader.UNIVERSE_SQL,
        data_loader.FUNDAMENTALS_SQL,
        data_loader.QUARTERLY_EPS_SQL,
    ):
        assert "canonical_identity" in query
        assert ".symbol" in query
        assert ".exchange" in query
        assert ".cik" in query


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
