from datetime import date

import pandas as pd

from src import runner, sensitivity
from src.config import Config
from src.simulator import SimResult
from tests.util import make_cfg


def _matrices(dates: pd.DatetimeIndex) -> dict:
    frame = pd.DataFrame({"AAA": [100.0] * len(dates)}, index=dates)
    return {
        "dates": dates,
        "symbols": frame.columns,
        "open": frame.copy(),
        "high": frame.copy(),
        "low": frame.copy(),
        "close": frame.copy(),
        "volume": frame.copy(),
    }


def test_sensitivity_stage_is_valid_configuration(monkeypatch) -> None:
    monkeypatch.setenv("STAGE", "sensitivity")

    cfg = Config.from_env()

    assert cfg.stage == "sensitivity"


def test_fixed_matrix_and_periods_are_deterministic() -> None:
    assert len(sensitivity.VARIANTS) == 8
    assert len({variant.name for variant in sensitivity.VARIANTS}) == 8
    assert len({variant.detection_key for variant in sensitivity.VARIANTS}) == 6
    assert sensitivity.phases(date(2020, 1, 2), date(2026, 7, 10)) == (
        ("dev", date(2020, 1, 2), date(2023, 12, 31)),
        ("oos", date(2024, 1, 1), date(2026, 7, 10)),
    )


def test_variant_replaces_only_strictness_and_run_identity() -> None:
    cfg = make_cfg(run_label="matrix", simulation_mode="independent")
    variant = next(v for v in sensitivity.VARIANTS if v.name == "moderate")

    actual = variant.apply(
        cfg, "oos", date(2024, 1, 1), date(2026, 7, 10)
    )

    assert actual.run_label == "matrix_oos_moderate"
    assert actual.start_date == "2024-01-01"
    assert actual.end_date == "2026-07-10"
    assert actual.vcp_score_min == 60.0
    assert actual.dryup_ratio_min == 0.20
    assert actual.dryup_ratio_max == 0.85
    assert actual.breakout_volume_min_ratio == 1.20
    assert actual.market_filter_enable == cfg.market_filter_enable
    assert actual.breakout_require_close_above_pivot


def test_in_memory_setups_use_non_database_ids_and_asof_screen_context() -> None:
    setups = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "detect_date": [date(2024, 1, 3)],
            "valid_until": [date(2024, 1, 10)],
        }
    )
    screen = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "period_end_date": [date(2024, 1, 3)],
            "rs_rating": [92],
            "ibkr_industry_rs_rating": [88],
            "ibkr_category_rs_rating": [85],
            "stock_industry_rs_rating": [90],
            "stock_category_rs_rating": [89],
            "eps_yoy": [0.35],
            "revenue_yoy": [0.22],
        }
    )

    actual = runner._prepare_simulation_setups(setups, screen)

    assert actual.loc[0, "setup_id"] == -1
    assert actual.loc[0, "rs_rating"] == 92
    assert actual.loc[0, "eps_yoy"] == 0.35


def test_sensitivity_reuses_vcp_detection_and_runs_every_phase(monkeypatch) -> None:
    dates = pd.bdate_range("2023-12-20", "2024-01-10")
    matrices = _matrices(dates)
    screen = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "period_end_date": [date(2023, 12, 29)],
            "rs_rating": [90],
            "ibkr_industry_rs_rating": [90],
            "ibkr_category_rs_rating": [90],
            "stock_industry_rs_rating": [90],
            "stock_category_rs_rating": [90],
            "eps_yoy": [0.3],
            "revenue_yoy": [0.2],
        }
    )
    setups = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "detect_date": [date(2023, 12, 29)],
            "valid_until": [date(2024, 1, 5)],
        }
    )
    detection_calls = []
    simulation_calls = []

    def fake_detect(cfg, *args, **kwargs):
        detection_calls.append(
            (cfg.vcp_score_min, cfg.dryup_ratio_min, cfg.dryup_ratio_max)
        )
        return setups.copy()

    def fake_simulation(conn, cfg, *args, **kwargs):
        simulation_calls.append((cfg.run_label, cfg.start_date, cfg.end_date))
        return len(simulation_calls), {}

    monkeypatch.setattr(runner, "detect_setups", fake_detect)
    monkeypatch.setattr(runner, "_run_simulation", fake_simulation)
    monkeypatch.setattr(
        runner.data_loader, "load_regime_scores", lambda conn, cfg: pd.DataFrame()
    )

    runner.run_sensitivity(
        object(),
        make_cfg(run_label="matrix", simulation_mode="independent"),
        pd.DataFrame({"symbol": ["AAA"]}),
        screen,
        matrices,
        pd.DataFrame(
            {"symbol": ["AAA"], "ibkr_industry": ["I"], "ibkr_category": ["C"]}
        ),
        pd.DataFrame(index=dates),
        date(2023, 12, 20),
        date(2024, 1, 10),
    )

    assert len(detection_calls) == 6
    assert len(set(detection_calls)) == 6
    assert len(simulation_calls) == 16
    assert {call[1] for call in simulation_calls} == {"2023-12-20", "2024-01-01"}


def test_zero_setup_simulation_still_persists_a_run(monkeypatch) -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    matrices = _matrices(dates)
    empty_setups = pd.DataFrame(
        columns=["setup_id", "symbol", "detect_date", "valid_until"]
    )
    created = []

    monkeypatch.setattr(
        runner.breakout_confirmation,
        "confirm_daily_breakouts",
        lambda *args, **kwargs: (empty_setups.copy(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        runner,
        "simulate",
        lambda *args, **kwargs: SimResult(
            metrics={
                "initial_equity": 100000.0,
                "final_equity": 100000.0,
                "total_return": 0.0,
                "num_positions": 0,
            }
        ),
    )
    monkeypatch.setattr(
        runner.persistence,
        "create_run",
        lambda conn, cfg, metrics, start, end: created.append((start, end)) or 7,
    )
    monkeypatch.setattr(runner.persistence, "write_trades", lambda *args: None)
    monkeypatch.setattr(
        runner.persistence, "write_breakout_events", lambda *args: None
    )
    monkeypatch.setattr(runner.persistence, "write_equity", lambda *args: None)

    run_id, metrics = runner._run_simulation(
        object(),
        make_cfg(market_filter_enable=False, regime_entry_filter_enable=False),
        matrices,
        pd.DataFrame(
            columns=["symbol", "ibkr_industry", "ibkr_category"]
        ),
        pd.DataFrame(index=dates),
        empty_setups,
        pd.DataFrame(),
        dates[0].date(),
        dates[-1].date(),
    )

    assert run_id == 7
    assert metrics["num_positions"] == 0
    assert created == [(dates[0].date(), dates[-1].date())]
