from datetime import date

import pandas as pd
import pytest

from src import runner, sensitivity
from src.config import Config
from src.simulator import SimResult
from tests.util import make_cfg


class _TransactionConnection:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


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


def _candidate_context(dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    values = {
        "trend_template_pass": 1.0,
        "rs_rating": 90.0,
        "fundamental_score": 75.0,
        "fundamental_coverage": 1.0,
        "eps_yoy": 0.30,
        "revenue_yoy": 0.20,
    }
    return {
        field: pd.DataFrame({"AAA": [value] * len(dates)}, index=dates)
        for field, value in values.items()
    }


def test_sensitivity_stage_is_valid_configuration(monkeypatch) -> None:
    monkeypatch.setenv("STAGE", "sensitivity")

    cfg = Config.from_env()

    assert cfg.stage == "sensitivity"


def test_fixed_matrix_and_periods_are_deterministic() -> None:
    assert [variant.name for variant in sensitivity.VARIANTS] == [
        "market_off",
        "market_on",
    ]
    assert len({variant.detection_key for variant in sensitivity.VARIANTS}) == 1
    assert {variant.market_filter_enable for variant in sensitivity.VARIANTS} == {
        False,
        True,
    }
    assert sensitivity.phases(date(2020, 1, 2), date(2023, 12, 29)) == (
        ("dev", date(2020, 1, 2), date(2023, 12, 29)),
    )
    sensitivity.validate_configured_window(
        date(2020, 1, 2), date(2023, 12, 31)
    )
    with pytest.raises(ValueError, match="development-only"):
        sensitivity.phases(date(2020, 1, 2), date(2024, 1, 1))
    with pytest.raises(ValueError, match="requires 2020-01-02"):
        sensitivity.validate_configured_window(
            date(2020, 1, 2), date(2026, 7, 10)
        )


def test_variants_differ_only_by_market_gate_and_run_identity() -> None:
    cfg = make_cfg(
        run_label="matrix",
        simulation_mode="independent",
        market_filter_enable=False,
    )
    actual = [
        variant.apply(
            cfg, "dev", date(2020, 1, 2), date(2023, 12, 31)
        )
        for variant in sensitivity.VARIANTS
    ]

    assert [item.run_label for item in actual] == [
        "matrix_dev_market_off",
        "matrix_dev_market_on",
    ]
    assert {item.start_date for item in actual} == {"2020-01-02"}
    assert {item.end_date for item in actual} == {"2023-12-31"}
    assert [item.market_filter_enable for item in actual] == [False, True]
    ignored = {"run_label", "market_filter_enable"}
    assert {
        key: value for key, value in vars(actual[0]).items() if key not in ignored
    } == {
        key: value for key, value in vars(actual[1]).items() if key not in ignored
    }


def test_sensitivity_requires_complete_market_index_inputs() -> None:
    cfg = make_cfg(stage="sensitivity", market_filter_enable=False)

    actual = runner._market_data_config(cfg)

    assert actual.market_filter_enable
    assert not cfg.market_filter_enable


def test_sensitivity_rejects_combined_simulation_mode_before_work() -> None:
    with pytest.raises(ValueError, match="independent or portfolio"):
        runner.run_sensitivity(
            object(),
            make_cfg(simulation_mode="both"),
            pd.DataFrame(),
            pd.DataFrame(),
            {},
            pd.DataFrame(),
            pd.DataFrame(),
            date(2020, 1, 2),
            date(2023, 12, 31),
        )


def test_in_memory_setups_use_non_database_ids_without_detect_date_context() -> None:
    setups = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "detect_date": [date(2024, 1, 3)],
            "valid_until": [date(2024, 1, 10)],
        }
    )
    actual = runner._prepare_simulation_setups(setups)

    assert actual.loc[0, "setup_id"] == -1
    assert "rs_rating" not in actual
    assert "eps_yoy" not in actual


def test_sensitivity_reuses_detection_and_runs_only_dev_gate_arms(monkeypatch) -> None:
    dates = pd.bdate_range("2023-12-20", "2023-12-29")
    matrices = _matrices(dates)
    screen = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "period_end_date": [date(2023, 12, 21)],
            "rs_rating": [90],
            "ibkr_industry_rs_rating": [90],
            "ibkr_category_rs_rating": [90],
            "stock_industry_rs_rating": [90],
            "stock_category_rs_rating": [90],
            "trend_template_pass": [True],
            "screen_pass": [True],
            "fundamental_score": [75],
            "fundamental_coverage": [1.0],
            "eps_yoy": [0.3],
            "revenue_yoy": [0.2],
        }
    )
    setups = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "detect_date": [date(2023, 12, 21)],
            "valid_until": [date(2023, 12, 29)],
        }
    )
    detection_calls = []
    simulation_calls = []

    def fake_detect(cfg, *args, **kwargs):
        detection_calls.append(
            cfg.dryup_score_zero_ratio
        )
        return setups.copy()

    def fake_simulation(conn, cfg, *args, **kwargs):
        simulation_calls.append(
            (
                cfg.run_label,
                cfg.start_date,
                cfg.end_date,
                kwargs.get("state_start"),
                cfg.market_filter_enable,
                kwargs["candidate_context"],
            )
        )
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
        date(2023, 12, 29),
    )

    assert detection_calls == [1.25]
    assert len(simulation_calls) == 2
    assert {call[0] for call in simulation_calls} == {
        "matrix_dev_market_off",
        "matrix_dev_market_on",
    }
    assert {call[1] for call in simulation_calls} == {"2023-12-20"}
    assert {call[2] for call in simulation_calls} == {"2023-12-29"}
    assert {call[3] for call in simulation_calls} == {date(2023, 12, 20)}
    assert {call[4] for call in simulation_calls} == {False, True}
    for call in simulation_calls:
        assert call[5]["rs_rating"].index.equals(dates)
        assert call[5]["rs_rating"].columns.equals(pd.Index(["AAA"]))


def test_zero_setup_simulation_still_persists_a_run(monkeypatch) -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    matrices = _matrices(dates)
    empty_setups = pd.DataFrame(
        columns=["setup_id", "symbol", "detect_date", "valid_until"]
    )
    created = []
    simulated = {}

    def fake_simulate(*args, **kwargs):
        simulated["setups"] = args[6]
        simulated["market_exposure_cap"] = kwargs["market_exposure_cap"]
        simulated["volume_m"] = kwargs["volume_m"]
        simulated["candidate_context"] = kwargs["candidate_context"]
        return SimResult(
            metrics={
                "initial_equity": 100000.0,
                "final_equity": 100000.0,
                "total_return": 0.0,
                "num_positions": 0,
            }
        )

    monkeypatch.setattr(runner, "simulate", fake_simulate)
    monkeypatch.setattr(
        runner.persistence,
        "create_run",
        lambda conn, cfg, metrics, start, end, **kwargs: created.append((start, end)) or 7,
    )
    monkeypatch.setattr(runner.persistence, "write_trades", lambda *args: None)
    monkeypatch.setattr(
        runner.persistence, "write_breakout_events", lambda *args: None
    )
    monkeypatch.setattr(runner.persistence, "write_equity", lambda *args: None)

    conn = _TransactionConnection()
    run_id, metrics = runner._run_simulation(
        conn,
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
        candidate_context=_candidate_context(dates),
    )

    assert run_id == 7
    assert metrics["num_positions"] == 0
    assert created == [(dates[0].date(), dates[-1].date())]
    assert simulated["setups"] is empty_setups
    assert simulated["market_exposure_cap"] is None
    assert simulated["volume_m"] is matrices["volume"]
    assert simulated["candidate_context"]["rs_rating"].equals(
        _candidate_context(dates)["rs_rating"]
    )
    assert conn.commits == 1
    assert conn.rollbacks == 0


def test_single_run_rolls_back_when_a_result_write_fails(monkeypatch) -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    matrices = _matrices(dates)
    conn = _TransactionConnection()

    monkeypatch.setattr(
        runner,
        "simulate",
        lambda *args, **kwargs: SimResult(
            metrics={"initial_equity": 100000.0, "final_equity": 100000.0}
        ),
    )
    monkeypatch.setattr(runner.persistence, "create_run", lambda *args, **kwargs: 7)
    monkeypatch.setattr(
        runner.persistence,
        "write_trades",
        lambda *args: (_ for _ in ()).throw(RuntimeError("COPY failed")),
    )

    with pytest.raises(RuntimeError, match="COPY failed"):
        runner._run_simulation(
            conn,
            make_cfg(market_filter_enable=False, regime_entry_filter_enable=False),
            matrices,
            pd.DataFrame(
                columns=["symbol", "ibkr_industry", "ibkr_category"]
            ),
            pd.DataFrame(index=dates),
            pd.DataFrame(
                columns=["setup_id", "symbol", "detect_date", "valid_until"]
            ),
            pd.DataFrame(),
            dates[0].date(),
            dates[-1].date(),
            candidate_context=_candidate_context(dates),
        )

    assert conn.commits == 0
    assert conn.rollbacks == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_run_simulation_passes_market_caps_only_to_enabled_arm(
    monkeypatch, enabled: bool
) -> None:
    dates = pd.bdate_range("2023-12-27", periods=3)
    matrices = _matrices(dates)
    observed = {}
    market = pd.DataFrame(
        {"entry_exposure_cap": [0.0, 0.25, 1.0]}, index=dates
    )

    def fake_simulate(*args, **kwargs):
        observed["market_exposure_cap"] = kwargs["market_exposure_cap"]
        return SimResult(
            metrics={"initial_equity": 100000.0, "final_equity": 100000.0}
        )

    monkeypatch.setattr(runner, "simulate", fake_simulate)
    monkeypatch.setattr(runner.persistence, "create_run", lambda *args, **kwargs: 11)
    monkeypatch.setattr(runner.persistence, "write_trades", lambda *args: None)
    monkeypatch.setattr(
        runner.persistence, "write_breakout_events", lambda *args: None
    )
    monkeypatch.setattr(runner.persistence, "write_equity", lambda *args: None)

    runner._run_simulation(
        _TransactionConnection(),
        make_cfg(
            market_filter_enable=enabled, regime_entry_filter_enable=False
        ),
        matrices,
        pd.DataFrame(columns=["symbol", "ibkr_industry", "ibkr_category"]),
        market,
        pd.DataFrame(columns=["setup_id", "symbol", "detect_date", "valid_until"]),
        pd.DataFrame(),
        dates[0].date(),
        dates[-1].date(),
        candidate_context=_candidate_context(dates),
    )

    actual = observed["market_exposure_cap"]
    if enabled:
        assert actual.tolist() == [0.0, 0.25, 1.0]
    else:
        assert actual is None


def test_runner_persists_same_day_breakout_event_without_confirmation(monkeypatch) -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    matrices = _matrices(dates)
    setups = pd.DataFrame(
        {
            "setup_id": [1],
            "symbol": ["AAA"],
            "detect_date": [dates[0].date()],
            "valid_until": [dates[-1].date()],
        }
    )
    events = pd.DataFrame(
        {
            "setup_id": [1],
            "symbol": ["AAA"],
            "setup_detect_date": [dates[0].date()],
            "breakout_date": [dates[1].date()],
            "pivot": [100.0],
            "trigger_price": [100.1],
            "entry_filled": [True],
            "entry_date": [dates[1].date()],
            "entry_price": [100.1],
            "decision": ["filled"],
        }
    )
    written = []

    monkeypatch.setattr(
        runner,
        "simulate",
        lambda *args, **kwargs: SimResult(
            breakout_events=events.copy(),
            metrics={"initial_equity": 100000.0, "final_equity": 100000.0},
        ),
    )
    monkeypatch.setattr(runner.persistence, "create_run", lambda *args, **kwargs: 9)
    monkeypatch.setattr(runner.persistence, "write_trades", lambda *args: None)
    monkeypatch.setattr(
        runner.persistence,
        "write_breakout_events",
        lambda conn, run_id, frame: written.append((run_id, frame.copy())),
    )
    monkeypatch.setattr(runner.persistence, "write_equity", lambda *args: None)

    runner._run_simulation(
        _TransactionConnection(),
        make_cfg(market_filter_enable=False, regime_entry_filter_enable=False),
        matrices,
        pd.DataFrame(columns=["symbol", "ibkr_industry", "ibkr_category"]),
        pd.DataFrame(index=dates),
        setups,
        pd.DataFrame(),
        dates[0].date(),
        dates[-1].date(),
        candidate_context=_candidate_context(dates),
    )

    assert written[0][0] == 9
    assert written[0][1].to_dict("records") == events.to_dict("records")
