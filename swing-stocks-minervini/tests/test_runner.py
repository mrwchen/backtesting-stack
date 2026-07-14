from dataclasses import replace

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


def test_single_run_and_bootstrapped_labels_cannot_share_fingerprint() -> None:
    dates, matrices, context, setups, universe = _simulation_fingerprint_inputs()
    cfg = make_cfg(
        simulation_mode="portfolio",
        portfolio_ranking_mode="quality_only",
        portfolio_setup_types=("vcp",),
        neutral_rank_salt="v8-bootstrap-00",
    )
    quality_labels = (
        QualityCalibrationLabel(
            setup_type="vcp",
            information_date=dates[0],
            available_date=dates[1],
            raw_quality_score=75.0,
            realized_r_multiple=1.5,
            walk_forward_quality_score=0.25,
            competition_size=2,
        ),
    )
    bootstrapped = runner.ranking_sensitivity.bootstrap_quality_labels(
        quality_labels, cfg.neutral_rank_salt
    )
    fill_fingerprint = (
        runner.reproducibility.fill_calibration_labels_fingerprint(())
    )

    def fingerprint(labels) -> str:
        return runner._simulation_input_fingerprint(
            cfg,
            setups,
            matrices,
            universe,
            None,
            None,
            pd.DataFrame(),
            context,
            quality_labels_fingerprint=(
                runner.reproducibility.quality_calibration_labels_fingerprint(
                    labels
                )
            ),
            fill_labels_fingerprint=fill_fingerprint,
            online_calibration=False,
        )

    assert fingerprint(quality_labels) != fingerprint(bootstrapped)


def test_all_288_experiment_paths_have_unique_input_fingerprints() -> None:
    dates, matrices, context, setups, universe = _simulation_fingerprint_inputs()
    base_cfg = make_cfg(simulation_mode="portfolio")
    quality_labels = (
        QualityCalibrationLabel(
            setup_type="vcp",
            information_date=dates[0],
            available_date=dates[1],
            raw_quality_score=75.0,
            realized_r_multiple=1.5,
            walk_forward_quality_score=0.25,
            competition_size=2,
        ),
    )
    fill_fingerprint = (
        runner.reproducibility.fill_calibration_labels_fingerprint(())
    )
    fingerprints = set()
    for salt in runner.ranking_sensitivity.NEUTRAL_RANK_SALTS:
        weighted_labels = runner.ranking_sensitivity.bootstrap_quality_labels(
            quality_labels, salt
        )
        quality_fingerprint = (
            runner.reproducibility.quality_calibration_labels_fingerprint(
                weighted_labels
            )
        )
        for case in runner.ranking_sensitivity.EXPERIMENT_CASES:
            cfg = replace(
                base_cfg,
                portfolio_ranking_mode=case.ranking_mode,
                portfolio_setup_types=case.setup_types,
                neutral_rank_salt=salt,
            )
            fingerprints.add(
                runner._simulation_input_fingerprint(
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
            )

    assert len(fingerprints) == 288


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

    calibration_calls = []

    def fake_calibration(*args, **kwargs):
        calibration_calls.append((args, kwargs))
        return quality_labels, fill_labels

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
                kwargs["calibration_labels_supplied"],
            )
        )
        return len(calls), {"num_positions": len(calls)}

    monkeypatch.setattr(runner, "_first_touch_calibration_labels", fake_calibration)
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
        _market(dates),
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
    assert calls[0][6] is calls[1][6] is quality_labels
    assert calls[0][7] is calls[1][7] is fill_labels
    assert calls[0][8] is calls[1][8] is True
    assert len(calibration_calls) == 1
    assert calibration_calls[0][1]["market_exposure_cap"].tolist() == [1.0, 1.0]
    assert conn.commits == 1
    assert conn.rollbacks == 0


def test_ranking_experiment_runs_complete_case_salt_cartesian_product(
    monkeypatch, caplog
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    matrices = _matrices(dates)
    conn = _TransactionConnection()
    calls = []
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

    calibration_calls = []
    fingerprint_calls = {"quality": 0, "fill": 0}
    quality_fingerprint = (
        runner.reproducibility.quality_calibration_labels_fingerprint
    )
    fill_fingerprint = runner.reproducibility.fill_calibration_labels_fingerprint

    def counted_quality_fingerprint(labels):
        fingerprint_calls["quality"] += 1
        return quality_fingerprint(labels)

    def counted_fill_fingerprint(labels):
        fingerprint_calls["fill"] += 1
        return fill_fingerprint(labels)

    def fake_calibration(*args, **kwargs):
        calibration_calls.append((args[0], kwargs))
        return quality_labels, fill_labels

    def fake_run(*args, **kwargs):
        phase_cfg = args[1]
        calls.append((phase_cfg, kwargs))
        salt_number = max(0, len(calls) - 1)
        return len(calls), {
            "total_return": salt_number / 100,
            "cagr": salt_number / 200,
            "max_drawdown": salt_number / 300,
            "profit_factor": 1.0 + salt_number / 100,
            "avg_r_multiple": salt_number / 1000,
        }

    monkeypatch.setattr(runner, "_first_touch_calibration_labels", fake_calibration)
    monkeypatch.setattr(runner, "_run_simulation", fake_run)
    monkeypatch.setattr(
        runner.reproducibility,
        "quality_calibration_labels_fingerprint",
        counted_quality_fingerprint,
    )
    monkeypatch.setattr(
        runner.reproducibility,
        "fill_calibration_labels_fingerprint",
        counted_fill_fingerprint,
    )
    monkeypatch.setattr(
        runner.data_loader, "load_regime_scores", lambda *_: pd.DataFrame()
    )
    cfg = make_cfg(
        simulation_mode="both",
        portfolio_ranking_experiment_enable=True,
        neutral_rank_salt="outer-salt-must-not-affect-calibration",
        run_label="ensemble",
    )

    with caplog.at_level("INFO", logger="runner"):
        first_touch, portfolio = runner.run_sim(
            conn,
            cfg,
            matrices,
            pd.DataFrame(),
            _market(dates),
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

    assert first_touch[0] == 1
    assert len(portfolio) == 288
    assert len(calls) == 289
    assert len(calibration_calls) == 1
    assert calibration_calls[0][0].neutral_rank_salt == "v8-bootstrap-00"
    assert calibration_calls[0][0].portfolio_ranking_mode == "quality_only"
    assert calls[0][0].run_label == "ensemble_first_touch"
    assert calls[0][0].simulation_mode == "independent"
    assert calls[0][0].neutral_rank_salt == "v8-bootstrap-00"
    assert calls[0][1]["quality_labels"] is quality_labels
    assert calls[0][1]["fill_labels"] is fill_labels
    assert calls[0][1]["calibration_labels_supplied"] is True
    base_label_fingerprints = calls[0][1]["calibration_label_fingerprints"]
    assert calls[0][1].get("commit", True) is True
    expected_configs = [
        (salt, case)
        for salt in runner.ranking_sensitivity.NEUTRAL_RANK_SALTS
        for case in runner.ranking_sensitivity.EXPERIMENT_CASES
    ]
    assert [
        (call[0].neutral_rank_salt, call[0].portfolio_ranking_mode,
         call[0].portfolio_setup_types)
        for call in calls[1:]
    ] == [
        (salt, case.ranking_mode, case.setup_types)
        for salt, case in expected_configs
    ]
    assert [call[0].run_label for call in calls[1:]] == [
        f"ensemble_{case.name}_salt_{salt_index:02d}"
        for salt_index, _salt in enumerate(
            runner.ranking_sensitivity.NEUTRAL_RANK_SALTS
        )
        for case in runner.ranking_sensitivity.EXPERIMENT_CASES
    ]
    for salt_index in range(32):
        salt_calls = calls[1 + 9 * salt_index : 1 + 9 * (salt_index + 1)]
        weighted_tuple = salt_calls[0][1]["quality_labels"]
        assert all(call[1]["quality_labels"] is weighted_tuple for call in salt_calls)
        fingerprint_tuple = salt_calls[0][1]["calibration_label_fingerprints"]
        assert all(
            call[1]["calibration_label_fingerprints"] is fingerprint_tuple
            for call in salt_calls
        )
        assert fingerprint_tuple[1] == base_label_fingerprints[1]
    for _, kwargs in calls[1:]:
        assert kwargs["fill_labels"] is fill_labels
        assert kwargs["calibration_labels_supplied"] is True
        assert kwargs.get("commit", True) is True
    assert fingerprint_calls == {"quality": 33, "fill": 1}
    assert len(
        {
            kwargs["calibration_label_fingerprints"][0]
            for _, kwargs in calls[1:]
        }
    ) == 32
    assert sum(
        "portfolio ranking experiment summary" in record.message
        for record in caplog.records
    ) == 45
    assert any(
        "portfolio ranking experiment done cases 9 salts 32 portfolio-paths 288 persisted-runs 289"
        in record.message
        for record in caplog.records
    )
    assert not any("best" in record.message.lower() for record in caplog.records)


def test_portfolio_only_ranking_experiment_builds_calibration_once(
    monkeypatch,
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    matrices = _matrices(dates)
    calibration_calls = 0
    calibration_salts = []
    portfolio_calls = []
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

    def fake_calibration(*_args, **_kwargs):
        nonlocal calibration_calls
        calibration_calls += 1
        calibration_salts.append(_args[0].neutral_rank_salt)
        return quality_labels, fill_labels

    def fake_run(*args, **kwargs):
        portfolio_calls.append((args[1], kwargs))
        return len(portfolio_calls), {
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 1.0,
            "avg_r_multiple": 0.0,
        }

    monkeypatch.setattr(runner, "_first_touch_calibration_labels", fake_calibration)
    monkeypatch.setattr(runner, "_run_simulation", fake_run)
    monkeypatch.setattr(
        runner.data_loader, "load_regime_scores", lambda *_: pd.DataFrame()
    )

    first_touch, portfolio = runner.run_sim(
        _TransactionConnection(),
        make_cfg(
            simulation_mode="portfolio",
            portfolio_ranking_experiment_enable=True,
            neutral_rank_salt="outer-salt-must-not-affect-calibration",
        ),
        matrices,
        pd.DataFrame(),
        _market(dates),
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

    assert first_touch is None
    assert len(portfolio) == 288
    assert calibration_calls == 1
    assert calibration_salts == ["v8-bootstrap-00"]
    assert len(portfolio_calls) == 288
    for _, kwargs in portfolio_calls:
        assert kwargs["fill_labels"] is fill_labels
        assert kwargs["calibration_labels_supplied"] is True
    for salt_index in range(32):
        salt_calls = portfolio_calls[9 * salt_index : 9 * (salt_index + 1)]
        weighted_tuple = salt_calls[0][1]["quality_labels"]
        assert weighted_tuple is not quality_labels
        assert all(call[1]["quality_labels"] is weighted_tuple for call in salt_calls)


def test_ranking_experiment_has_no_completion_summary_after_partial_failure(
    monkeypatch, caplog
) -> None:
    calls = 0

    def fake_run(*_args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 7:
            raise RuntimeError("salt arm failed")
        return calls, {
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 1.0,
            "avg_r_multiple": 0.0,
        }

    dates = pd.bdate_range("2024-01-02", periods=2)
    matrices = _matrices(dates)
    monkeypatch.setattr(
        runner, "_first_touch_calibration_labels", lambda *args, **kwargs: ((), ())
    )
    monkeypatch.setattr(runner, "_run_simulation", fake_run)
    monkeypatch.setattr(
        runner.data_loader, "load_regime_scores", lambda *_: pd.DataFrame()
    )

    with caplog.at_level("INFO", logger="runner"):
        with pytest.raises(RuntimeError, match="salt arm failed"):
            runner.run_sim(
                _TransactionConnection(),
                make_cfg(
                    simulation_mode="both",
                    portfolio_ranking_experiment_enable=True,
                ),
                matrices,
                pd.DataFrame(),
                _market(dates),
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

    assert calls == 7
    assert not any(" summary " in record.message for record in caplog.records)
    assert not any(" experiment done " in record.message for record in caplog.records)


def test_both_mode_rolls_back_both_arms_when_second_arm_fails(monkeypatch) -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    matrices = _matrices(dates)
    conn = _TransactionConnection()
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("portfolio persistence failed")
        return 1, {}

    monkeypatch.setattr(
        runner, "_first_touch_calibration_labels", lambda *args, **kwargs: ((), ())
    )
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
            _market(dates),
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
        observed["kwargs"] = kwargs
        return 1, {}

    calibration_calls = []

    def fake_calibration(*args, **kwargs):
        calibration_calls.append((args, kwargs))
        return (), ()

    monkeypatch.setattr(runner, "_first_touch_calibration_labels", fake_calibration)
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
        _market(dates),
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
    assert observed["kwargs"]["calibration_labels_supplied"] is True
    assert len(calibration_calls) == 1
    assert calibration_calls[0][1]["market_exposure_cap"].tolist() == [1.0, 1.0]
