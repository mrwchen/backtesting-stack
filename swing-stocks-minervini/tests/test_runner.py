import numpy as np
import pandas as pd
import pytest

from src import runner
from src.candidate_ranking import FillCalibrationLabel, QualityCalibrationLabel
from src.runner import _attach_regime_attribution, _regime_entry_allowed

from .util import make_cfg


class _TransactionConnection:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _matrices(dates: pd.DatetimeIndex) -> dict:
    close = pd.DataFrame({"AAA": [100.0] * len(dates)}, index=dates)
    continuity = pd.DataFrame(1, index=dates, columns=close.columns, dtype="int64")
    boundary = pd.DataFrame(False, index=dates, columns=close.columns)
    boundary.iloc[0] = True
    return {
        "dates": dates,
        "symbols": close.columns,
        "open": close.copy(),
        "high": close.copy(),
        "low": close.copy(),
        "close": close.copy(),
        "volume": close.copy(),
        "price_continuity_segment": continuity,
        "price_continuity_break": boundary,
    }


def _screen_passes(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period_end_date": dates.date,
            "symbol": ["AAA"] * len(dates),
            "trend_template_pass": [True] * len(dates),
            "rs_rating": np.arange(90, 90 + len(dates)),
            "fundamental_score": [75.0] * len(dates),
            "fundamental_coverage": [1.0] * len(dates),
            "eps_yoy": [0.30] * len(dates),
            "revenue_yoy": [0.20] * len(dates),
        }
    )


def test_regime_gate_uses_score_only_from_next_calendar_day() -> None:
    dates = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"])
    regime = pd.DataFrame(
        {
            "day": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-07", "2024-01-08"]),
            "regime_label": ["RISK-ON", "RISK-OFF", "RISK-ON", "RISK-OFF"],
        }
    )
    cfg = make_cfg(regime_allowed_labels=("RISK-ON",))

    allowed = _regime_entry_allowed(dates, regime, cfg)

    # Friday sees Thursday, Monday sees Sunday's row, Tuesday sees Monday.
    assert allowed.tolist() == [True, True, False]


def test_regime_attribution_uses_same_causal_availability_mapping() -> None:
    trades = pd.DataFrame(
        {
            "entry_date": pd.to_datetime(["2024-01-05", "2024-01-08"]),
            "symbol": ["AAA", "BBB"],
        }
    )
    regime = pd.DataFrame(
        {
            "day": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-07"]),
            "regime_composite": [25.0, 85.0, 35.0],
            "regime_label": ["RISK-ON", "RISK-OFF", "CONSTRUCTIVE"],
        }
    )

    attributed = _attach_regime_attribution(trades, regime)

    assert attributed["regime_composite"].tolist() == [25.0, 35.0]
    assert attributed["regime_label"].tolist() == ["RISK-ON", "CONSTRUCTIVE"]


def test_regime_attribution_normalizes_mixed_datetime_resolutions() -> None:
    trades = pd.DataFrame(
        {
            "entry_date": np.array(["2024-01-05", "2024-01-08"], dtype="datetime64[s]"),
            "symbol": ["AAA", "BBB"],
        }
    )
    regime = pd.DataFrame(
        {
            "day": np.array(
                ["2024-01-04", "2024-01-07"], dtype="datetime64[us]"
            ),
            "regime_composite": [25.0, 35.0],
            "regime_label": ["RISK-ON", "CONSTRUCTIVE"],
        }
    )

    attributed = _attach_regime_attribution(trades, regime)

    assert attributed["regime_composite"].tolist() == [25.0, 35.0]
    assert attributed["regime_label"].tolist() == ["RISK-ON", "CONSTRUCTIVE"]


def test_candidate_context_is_exactly_aligned_without_forward_fill() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    passes = _screen_passes(dates).iloc[[0, 2]].copy()

    actual = runner._candidate_context_matrices(
        passes, dates, pd.Index(["AAA", "BBB"])
    )

    assert set(actual) == set(runner.CANDIDATE_CONTEXT_COLUMNS)
    for matrix in actual.values():
        assert matrix.index.equals(dates)
        assert matrix.columns.equals(pd.Index(["AAA", "BBB"]))
        assert pd.isna(matrix.loc[dates[1], "AAA"])
        assert matrix["BBB"].isna().all()
    assert actual["rs_rating"].loc[dates[0], "AAA"] == 90.0
    assert actual["rs_rating"].loc[dates[2], "AAA"] == 92.0


def test_nonpassing_daily_screen_row_still_supplies_ranking_context() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    screen_daily = _screen_passes(dates)
    screen_daily["screen_pass"] = [True, False]

    setup_days = runner._screen_pass_rows(screen_daily)
    context = runner._candidate_context_matrices(
        screen_daily, dates, pd.Index(["AAA"])
    )

    assert setup_days["period_end_date"].tolist() == [dates[0].date()]
    assert context["rs_rating"].loc[dates[1], "AAA"] == 91.0
    assert context["fundamental_coverage"].loc[dates[1], "AAA"] == 1.0


def test_candidate_context_rejects_duplicate_symbol_sessions() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    passes = pd.concat(
        [_screen_passes(dates), _screen_passes(dates).iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate"):
        runner._candidate_context_matrices(passes, dates, pd.Index(["AAA"]))


def test_simulation_fingerprint_tracks_volume_and_daily_candidate_context() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    matrices = _matrices(dates)
    context = runner._candidate_context_matrices(
        _screen_passes(dates), dates, matrices["symbols"]
    )
    setups = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "detect_date": [dates[0].date()],
            "valid_until": [dates[-1].date()],
        }
    )
    cfg = make_cfg()
    universe = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "ibkr_industry": ["Software"],
            "ibkr_category": ["Application"],
        }
    )

    first = runner._simulation_input_fingerprint(
        cfg, setups, matrices, universe, None, None, pd.DataFrame(), context
    )
    volume_changed = {**matrices, "volume": matrices["volume"].copy()}
    volume_changed["volume"].iat[0, 0] = 101.0
    second = runner._simulation_input_fingerprint(
        cfg, setups, volume_changed, universe, None, None, pd.DataFrame(), context
    )
    context_changed = {**context, "rs_rating": context["rs_rating"].copy()}
    context_changed["rs_rating"].iat[0, 0] = 99.0
    third = runner._simulation_input_fingerprint(
        cfg, setups, matrices, universe, None, None, pd.DataFrame(), context_changed
    )
    taxonomy_changed = universe.copy()
    taxonomy_changed.loc[0, "ibkr_industry"] = "Semiconductors"
    fourth = runner._simulation_input_fingerprint(
        cfg,
        setups,
        matrices,
        taxonomy_changed,
        None,
        None,
        pd.DataFrame(),
        context,
    )

    assert first != second
    assert first != third
    assert first != fourth


def test_both_mode_runs_ungated_first_touch_then_original_portfolio(
    monkeypatch,
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    matrices = _matrices(dates)
    calls = []
    conn = _TransactionConnection()
    quality_labels = (
        QualityCalibrationLabel(
            setup_type="vcp",
            information_date=dates[0],
            available_date=dates[1],
            raw_quality_score=75.0,
            realized_r_multiple=1.5,
            walk_forward_quality_score=0.25,
        ),
    )
    fill_labels = (
        FillCalibrationLabel(
            setup_type="vcp",
            information_date=dates[0],
            available_date=dates[1],
            readiness_signal=80.0,
            filled=True,
        ),
    )

    def fake_run(*args, **kwargs):
        phase_cfg = args[1]
        calls.append(
            (
                phase_cfg.simulation_mode,
                phase_cfg.market_filter_enable,
                phase_cfg.regime_entry_filter_enable,
                phase_cfg.run_label,
                kwargs["candidate_context"],
                kwargs["commit"],
                kwargs.get("quality_labels", ()),
                kwargs.get("fill_labels", ()),
                kwargs.get("label_sink"),
            )
        )
        if kwargs.get("label_sink") is not None:
            kwargs["label_sink"]["quality_labels"] = quality_labels
            kwargs["label_sink"]["fill_labels"] = fill_labels
        return len(calls), {"num_positions": len(calls)}

    monkeypatch.setattr(runner, "_run_simulation", fake_run)
    monkeypatch.setattr(
        runner.data_loader, "load_regime_scores", lambda *_: pd.DataFrame()
    )
    cfg = make_cfg(
        simulation_mode="both", market_filter_enable=True, run_label="daily"
    )

    actual = runner.run_sim(
        conn,
        cfg,
        matrices,
        pd.DataFrame(),
        pd.DataFrame(index=dates),
        _screen_passes(dates),
        dates[0].date(),
        dates[-1].date(),
        setups=pd.DataFrame(
            {
                "symbol": ["AAA"],
                "detect_date": [dates[0].date()],
                "valid_until": [dates[-1].date()],
            }
        ),
    )

    assert actual == ((1, {"num_positions": 1}), (2, {"num_positions": 2}))
    assert [call[:4] for call in calls] == [
        ("independent", False, False, "daily_first_touch"),
        ("portfolio", True, False, "daily_portfolio"),
    ]
    assert calls[0][4] is calls[1][4]
    assert calls[0][5] is calls[1][5] is False
    assert calls[0][6:8] == ((), ())
    assert calls[1][6] is quality_labels
    assert calls[1][7] is fill_labels
    assert calls[0][8] is not None
    assert calls[1][8] is None
    assert conn.commits == 1
    assert conn.rollbacks == 0


def test_both_mode_rolls_back_both_arms_when_second_arm_fails(monkeypatch) -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    matrices = _matrices(dates)
    conn = _TransactionConnection()
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if _kwargs.get("label_sink") is not None:
            _kwargs["label_sink"]["quality_labels"] = ()
            _kwargs["label_sink"]["fill_labels"] = ()
        if calls == 2:
            raise RuntimeError("portfolio persistence failed")
        return 1, {}

    monkeypatch.setattr(runner, "_run_simulation", fake_run)
    monkeypatch.setattr(
        runner.data_loader, "load_regime_scores", lambda *_: pd.DataFrame()
    )

    with pytest.raises(RuntimeError, match="portfolio persistence failed"):
        runner.run_sim(
            conn,
            make_cfg(simulation_mode="both"),
            matrices,
            pd.DataFrame(),
            pd.DataFrame(index=dates),
            _screen_passes(dates),
            dates[0].date(),
            dates[-1].date(),
            setups=pd.DataFrame(
                {
                    "symbol": ["AAA"],
                    "detect_date": [dates[0].date()],
                    "valid_until": [dates[-1].date()],
                }
            ),
        )

    assert conn.commits == 0
    assert conn.rollbacks == 1


def test_single_independent_mode_is_canonical_unfiltered_first_touch(
    monkeypatch,
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    matrices = _matrices(dates)
    observed = {}

    def fake_run(*args, **kwargs):
        observed["cfg"] = args[1]
        return 1, {}

    monkeypatch.setattr(runner, "_run_simulation", fake_run)
    monkeypatch.setattr(
        runner.data_loader, "load_regime_scores", lambda *_: pd.DataFrame()
    )

    runner.run_sim(
        _TransactionConnection(),
        make_cfg(
            simulation_mode="independent",
            market_filter_enable=True,
            regime_entry_filter_enable=True,
        ),
        matrices,
        pd.DataFrame(),
        pd.DataFrame(index=dates),
        _screen_passes(dates),
        dates[0].date(),
        dates[-1].date(),
        setups=pd.DataFrame(
            {
                "symbol": ["AAA"],
                "detect_date": [dates[0].date()],
                "valid_until": [dates[-1].date()],
            }
        ),
    )

    assert observed["cfg"].simulation_mode == "independent"
    assert not observed["cfg"].market_filter_enable
    assert not observed["cfg"].regime_entry_filter_enable
