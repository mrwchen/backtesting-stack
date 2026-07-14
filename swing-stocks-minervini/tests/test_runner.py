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


def _market(dates: pd.DatetimeIndex, cap: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {"entry_exposure_cap": [cap] * len(dates)},
        index=dates,
    )


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


def _simulation_fingerprint_inputs() -> tuple[
    pd.DatetimeIndex,
    dict,
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
]:
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
    universe = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "ibkr_industry": ["Software"],
            "ibkr_category": ["Application"],
        }
    )
    return dates, matrices, context, setups, universe


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
    _, matrices, context, setups, universe = _simulation_fingerprint_inputs()
    cfg = make_cfg()
    label_fingerprints = {
        "quality_labels_fingerprint": (
            runner.reproducibility.quality_calibration_labels_fingerprint(())
        ),
        "fill_labels_fingerprint": (
            runner.reproducibility.fill_calibration_labels_fingerprint(())
        ),
        "online_calibration": True,
    }

    first = runner._simulation_input_fingerprint(
        cfg,
        setups,
        matrices,
        universe,
        None,
        None,
        pd.DataFrame(),
        context,
        **label_fingerprints,
    )
    volume_changed = {**matrices, "volume": matrices["volume"].copy()}
    volume_changed["volume"].iat[0, 0] = 101.0
    second = runner._simulation_input_fingerprint(
        cfg,
        setups,
        volume_changed,
        universe,
        None,
        None,
        pd.DataFrame(),
        context,
        **label_fingerprints,
    )
    context_changed = {**context, "rs_rating": context["rs_rating"].copy()}
    context_changed["rs_rating"].iat[0, 0] = 99.0
    third = runner._simulation_input_fingerprint(
        cfg,
        setups,
        matrices,
        universe,
        None,
        None,
        pd.DataFrame(),
        context_changed,
        **label_fingerprints,
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
        **label_fingerprints,
    )

    assert first != second
    assert first != third
    assert first != fourth


def test_simulation_fingerprint_tracks_actual_calibration_mode() -> None:
    _, matrices, context, setups, universe = _simulation_fingerprint_inputs()
    cfg = make_cfg()
    quality_fingerprint = (
        runner.reproducibility.quality_calibration_labels_fingerprint(())
    )
    fill_fingerprint = (
        runner.reproducibility.fill_calibration_labels_fingerprint(())
    )

    online = runner._simulation_input_fingerprint(
        cfg,
        setups,
        matrices,
        universe,
        None,
        None,
        pd.DataFrame(),
        context,
        quality_labels_fingerprint=quality_fingerprint,
        fill_labels_fingerprint=fill_fingerprint,
        online_calibration=True,
    )
    preloaded = runner._simulation_input_fingerprint(
        cfg,
        setups,
        matrices,
        universe,
        None,
        None,
        pd.DataFrame(),
        context,
        quality_labels_fingerprint=quality_fingerprint,
        fill_labels_fingerprint=fill_fingerprint,
        online_calibration=False,
    )

    assert online != preloaded


def _forward_labels(dates: pd.DatetimeIndex):
    quality = tuple(
        QualityCalibrationLabel(
            setup_type=setup_type,
            information_date=dates[0],
            available_date=dates[1],
            raw_quality_score=75.0,
            realized_r_multiple=realized_r,
            walk_forward_quality_score=0.25,
        )
        for setup_type, realized_r in (("flat_base", 1.5), ("vcp", -0.5))
    )
    fills = tuple(
        FillCalibrationLabel(
            setup_type=setup_type,
            information_date=dates[0],
            available_date=dates[1],
            readiness_signal=80.0,
            filled=True,
        )
        for setup_type in ("flat_base", "vcp")
    )
    return quality, fills


def _forward_setups(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "setup_type": ["flat_base", "vcp", "power_play"],
            "detect_date": [dates[0].date()] * 3,
            "valid_until": [dates[-1].date()] * 3,
        }
    )


def test_forward_shadow_runs_one_research_one_shadow_and_32_controls(
    monkeypatch, caplog
) -> None:
    dates = pd.DatetimeIndex(["2026-07-10", "2026-07-13", "2026-07-14"])
    matrices = _matrices(dates)
    setups = _forward_setups(dates)
    quality_labels, fill_labels = _forward_labels(dates)
    calibration_calls = []
    run_calls = []

    def fake_calibration(*args, **kwargs):
        calibration_calls.append((args, kwargs))
        return quality_labels, fill_labels

    def fake_run(*args, **kwargs):
        phase_cfg = args[1]
        run_calls.append((args, kwargs))
        path_index = len(run_calls) - 1
        return len(run_calls), {
            "total_return": path_index / 100.0,
            "cagr": path_index / 200.0,
            "max_drawdown": path_index / 300.0,
            "profit_factor": 1.0 + path_index / 100.0,
            "avg_r_multiple": path_index / 1000.0,
        }

    monkeypatch.setattr(runner, "_first_touch_calibration_labels", fake_calibration)
    monkeypatch.setattr(runner, "_run_simulation", fake_run)
    monkeypatch.setattr(
        runner.data_loader, "load_regime_scores", lambda *_: pd.DataFrame()
    )

    with caplog.at_level("INFO", logger="runner"):
        first_touch, shadow, controls = runner.run_sim(
            _TransactionConnection(),
            make_cfg(run_label="v9"),
            matrices,
            pd.DataFrame(),
            _market(dates),
            _screen_passes(dates),
            dates[0].date(),
            dates[-1].date(),
            setups=setups,
        )

    assert first_touch[0] == 1
    assert shadow[0] == 2
    assert len(controls) == 32
    assert len(run_calls) == 34
    assert len(calibration_calls) == 1
    calibration_cfg = calibration_calls[0][0][0]
    assert calibration_cfg.simulation_mode == "independent"
    assert calibration_cfg.portfolio_ranking_mode == "relative_quality"
    assert calibration_cfg.force_close_at_end is False
    assert calibration_calls[0][0][2] is setups
    assert calibration_calls[0][1]["state_start_idx"] == 0
    assert calibration_calls[0][1]["market_exposure_cap"].tolist() == [1.0] * 3

    first_cfg = run_calls[0][0][1]
    assert first_cfg.simulation_mode == "independent"
    assert not first_cfg.market_filter_enable
    assert not first_cfg.regime_entry_filter_enable
    assert first_cfg.run_label == "v9_first_touch_research"
    assert first_cfg.force_close_at_end is False
    assert run_calls[0][0][5] is setups
    assert run_calls[0][0][7] == dates[1].date()
    assert run_calls[0][1]["state_start"] == dates[0].date()
    assert run_calls[0][1]["quality_labels"] is quality_labels
    assert run_calls[0][1]["fill_labels"] is fill_labels

    portfolio_calls = run_calls[1:]
    expected_cases = runner.forward_shadow.PORTFOLIO_CASES
    assert [call[0][1].run_label for call in portfolio_calls] == [
        f"v9_{case.name}" for case in expected_cases
    ]
    assert [call[0][1].portfolio_ranking_mode for call in portfolio_calls] == [
        case.ranking_mode for case in expected_cases
    ]
    assert [call[0][1].neutral_rank_salt for call in portfolio_calls] == [
        case.neutral_rank_salt for case in expected_cases
    ]
    flat_quality = portfolio_calls[0][1]["quality_labels"]
    flat_fills = portfolio_calls[0][1]["fill_labels"]
    assert [label.setup_type for label in flat_quality] == ["flat_base"]
    assert [label.setup_type for label in flat_fills] == ["flat_base"]
    for args, kwargs in portfolio_calls:
        phase_cfg = args[1]
        phase_setups = args[5]
        assert phase_cfg.simulation_mode == "portfolio"
        assert phase_cfg.portfolio_setup_types == ("flat_base",)
        assert phase_cfg.force_close_at_end is False
        assert phase_setups["setup_type"].tolist() == ["flat_base"]
        assert kwargs["quality_labels"] is flat_quality
        assert kwargs["fill_labels"] is flat_fills
        assert kwargs["calibration_labels_supplied"] is True
        assert kwargs["state_start"] == dates[0].date()
        assert kwargs.get("commit", True) is True
    assert all(
        call[1]["calibration_label_fingerprints"]
        is portfolio_calls[0][1]["calibration_label_fingerprints"]
        for call in portfolio_calls
    )
    assert sum(
        "forward shadow neutral-control summary" in record.message
        for record in caplog.records
    ) == 5
    assert any(
        "forward shadow done portfolio-paths 33 neutral-controls 32 persisted-runs 34"
        in record.message
        for record in caplog.records
    )
    assert not any("best" in record.message.lower() for record in caplog.records)


def test_forward_shadow_has_no_completion_summary_after_partial_failure(
    monkeypatch, caplog
) -> None:
    dates = pd.DatetimeIndex(["2026-07-10", "2026-07-13", "2026-07-14"])
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 7:
            raise RuntimeError("control failed")
        return calls, {
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 1.0,
            "avg_r_multiple": 0.0,
        }

    monkeypatch.setattr(
        runner, "_first_touch_calibration_labels", lambda *args, **kwargs: ((), ())
    )
    monkeypatch.setattr(runner, "_run_simulation", fake_run)
    monkeypatch.setattr(
        runner.data_loader, "load_regime_scores", lambda *_: pd.DataFrame()
    )

    with caplog.at_level("INFO", logger="runner"):
        with pytest.raises(RuntimeError, match="control failed"):
            runner.run_sim(
                _TransactionConnection(),
                make_cfg(),
                _matrices(dates),
                pd.DataFrame(),
                _market(dates),
                _screen_passes(dates),
                dates[0].date(),
                dates[-1].date(),
                setups=_forward_setups(dates),
            )

    assert calls == 7
    assert not any(" neutral-control summary " in r.message for r in caplog.records)
    assert not any(" forward shadow done " in r.message for r in caplog.records)


def test_forward_shadow_rejects_run_without_forward_session(monkeypatch) -> None:
    dates = pd.DatetimeIndex(["2026-07-09", "2026-07-10"])
    monkeypatch.setattr(
        runner.data_loader, "load_regime_scores", lambda *_: pd.DataFrame()
    )

    with pytest.raises(ValueError, match="no price session"):
        runner.run_sim(
            _TransactionConnection(),
            make_cfg(),
            _matrices(dates),
            pd.DataFrame(),
            _market(dates),
            _screen_passes(dates),
            dates[0].date(),
            dates[-1].date(),
            setups=_forward_setups(dates),
        )


def test_forward_shadow_rejects_later_session_when_exact_start_is_missing(
    monkeypatch,
) -> None:
    dates = pd.DatetimeIndex(["2026-07-10", "2026-07-14", "2026-07-15"])
    monkeypatch.setattr(
        runner.data_loader, "load_regime_scores", lambda *_: pd.DataFrame()
    )

    with pytest.raises(
        ValueError,
        match=(
            "requires the exact price session 2026-07-13; "
            "first available session is 2026-07-14"
        ),
    ):
        runner.run_sim(
            _TransactionConnection(),
            make_cfg(),
            _matrices(dates),
            pd.DataFrame(),
            _market(dates),
            _screen_passes(dates),
            dates[0].date(),
            dates[-1].date(),
            setups=_forward_setups(dates),
        )


def test_portfolio_concentration_excludes_partial_only_open_positions() -> None:
    trades = pd.DataFrame(
        [
            {
                "position_id": 1, "symbol": "AAA", "leg": "final",
                "entry_price": 100.0, "stop_price": 90.0,
                "shares": 10, "pnl": 200.0,
            },
            {
                "position_id": 2, "symbol": "BBB", "leg": "partial",
                "entry_price": 100.0, "stop_price": 90.0,
                "shares": 4, "pnl": 40.0,
            },
            {
                "position_id": 2, "symbol": "BBB", "leg": "final",
                "entry_price": 100.0, "stop_price": 90.0,
                "shares": 6, "pnl": 60.0,
            },
            {
                "position_id": 3, "symbol": "AAA", "leg": "final",
                "entry_price": 100.0, "stop_price": 90.0,
                "shares": 10, "pnl": -50.0,
            },
            {
                "position_id": 4, "symbol": "OPEN", "leg": "partial",
                "entry_price": 100.0, "stop_price": 90.0,
                "shares": 5, "pnl": 100.0,
            },
        ]
    )

    actual = runner._portfolio_concentration(trades)

    assert actual["closed_positions"] == 3
    assert actual["winner_symbols"] == 2
    assert actual["gross_positive_r"] == pytest.approx(3.0)
    assert actual["net_r"] == pytest.approx(2.5)
    assert actual["top"][1]["positive_r_share"] == pytest.approx(2 / 3)
    assert actual["top"][1]["leave_out_net_r"] == pytest.approx(0.5)
    assert actual["top"][1]["leave_out_profit_factor"] == pytest.approx(2.0)
    assert actual["top"][3]["leave_out_net_r"] == pytest.approx(-0.5)
    assert actual["top"][3]["leave_out_profit_factor"] == pytest.approx(0.0)
