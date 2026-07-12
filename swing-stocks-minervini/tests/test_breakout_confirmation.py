import numpy as np
import pandas as pd

from src.breakout_confirmation import attach_fills, confirm_daily_breakouts
from src.simulator import simulate
from tests.util import make_cfg


def _inputs(*, breakout_volume=150.0, breakout_close=100.5, next_high=101.0):
    dates = pd.bdate_range("2024-01-02", periods=24)
    bars = np.tile(np.array([99.0, 99.5, 98.0, 99.0]), (len(dates), 1))
    bars[20] = [99.0, 101.0, 98.0, breakout_close]
    bars[21] = [99.0, next_high, 98.0, 100.0]
    volume = np.full(len(dates), 100.0)
    volume[20] = breakout_volume
    frames = [
        pd.DataFrame({"AAA": bars[:, field]}, index=dates) for field in range(4)
    ]
    volume_m = pd.DataFrame({"AAA": volume}, index=dates)
    setups = pd.DataFrame(
        [
            {
                "setup_id": 7,
                "symbol": "AAA",
                "detect_date": dates[18].date(),
                "pivot": 100.0,
                "last_low": 95.0,
                "stop_level": 95.0,
                "valid_until": dates[23].date(),
                "vcp_score": 80.0,
            }
        ]
    )
    cfg = make_cfg(
        pivot_buffer_pct=0.0,
        breakout_volume_lookback_sessions=50,
        breakout_volume_min_history_sessions=20,
        breakout_volume_min_ratio=1.4,
        breakout_require_close_above_pivot=True,
    )
    return dates, frames, volume_m, setups, cfg


def _confirm(**kwargs):
    dates, frames, volume_m, setups, cfg = _inputs(**kwargs)
    confirmed, events = confirm_daily_breakouts(
        dates,
        pd.Index(["AAA"]),
        *frames,
        volume_m,
        setups,
        cfg,
    )
    return dates, confirmed, events


def _sim_cfg():
    return make_cfg(
        slippage_pct=0.0,
        commission_pct=0.0,
        partial_fraction=0.0,
        pivot_buffer_pct=0.0,
        max_buy_zone_pct=0.02,
        time_stop_sessions=999,
        profit_protection_trigger_r=999.0,
        exposure_levels=(1.0,),
    )


def test_breakout_volume_excludes_breakout_day_and_creates_d_plus_one_candidate():
    dates, confirmed, events = _confirm()

    assert len(events) == 1
    event = events.iloc[0]
    assert event["breakout_date"] == dates[20].date()
    assert event["average_volume_prior"] == 100.0
    assert event["breakout_volume_ratio"] == 1.5
    assert bool(event["confirmation_pass"])
    assert event["decision"] == "confirmed"
    assert confirmed.iloc[0]["detect_date"] == dates[20].date()
    assert confirmed.iloc[0]["valid_until"] == dates[21].date()


def test_low_volume_first_breakout_is_rejected_and_consumes_setup():
    dates, frames, volume_m, setups, cfg = _inputs(breakout_volume=130.0)
    frames[1].iloc[21, 0] = 102.0
    volume_m.iloc[21, 0] = 300.0

    confirmed, events = confirm_daily_breakouts(
        dates, pd.Index(["AAA"]), *frames, volume_m, setups, cfg
    )

    assert confirmed.empty
    assert len(events) == 1
    assert events.iloc[0]["breakout_date"] == dates[20].date()
    assert events.iloc[0]["decision"] == "volume_below_threshold"


def test_breakout_must_close_above_pivot():
    _, confirmed, events = _confirm(breakout_close=99.5)

    assert confirmed.empty
    assert events.iloc[0]["decision"] == "close_below_pivot"
    assert not bool(events.iloc[0]["confirmation_pass"])


def test_confirmed_signal_without_next_day_retrigger_is_not_filled():
    dates, frames, volume_m, setups, cfg = _inputs(next_high=99.5)
    confirmed, events = confirm_daily_breakouts(
        dates, pd.Index(["AAA"]), *frames, volume_m, setups, cfg
    )
    simulation = simulate(
        dates, pd.Index(["AAA"]), *frames, confirmed, _sim_cfg()
    )

    final_events = attach_fills(
        events, simulation.trades, simulation.entry_decisions
    )

    assert len(confirmed) == 1
    assert final_events.iloc[0]["planned_entry_date"] == dates[21].date()
    assert final_events.iloc[0]["decision"] == "no_retrigger"
    assert not bool(final_events.iloc[0]["entry_filled"])


def test_attach_fills_records_actual_d_plus_one_entry():
    dates, _, events = _confirm()
    trades = pd.DataFrame(
        [
            {
                "setup_id": 7,
                "entry_date": dates[21].date(),
                "entry_price": 100.0,
                "position_id": 1,
            }
        ]
    )

    final_events = attach_fills(events, trades)

    assert bool(final_events.iloc[0]["entry_filled"])
    assert final_events.iloc[0]["entry_date"] == dates[21].date()
    assert final_events.iloc[0]["decision"] == "filled"


def test_entry_decision_preserves_fill_when_no_exit_leg_exists():
    dates, _, events = _confirm()
    decisions = pd.DataFrame(
        [
            {
                "setup_id": 7,
                "entry_decision": "filled",
                "entry_date": dates[21].date(),
                "entry_price": 100.0,
            }
        ]
    )
    trades = pd.DataFrame(columns=["setup_id", "entry_date", "entry_price", "position_id"])

    final_events = attach_fills(events, trades, decisions)

    assert bool(final_events.iloc[0]["entry_filled"])
    assert final_events.iloc[0]["entry_date"] == dates[21].date()
    assert final_events.iloc[0]["decision"] == "filled"


def test_confirmed_breakout_can_only_enter_on_d_plus_one():
    dates, frames, volume_m, setups, cfg = _inputs()
    confirmed, _ = confirm_daily_breakouts(
        dates, pd.Index(["AAA"]), *frames, volume_m, setups, cfg
    )
    result = simulate(
        dates,
        pd.Index(["AAA"]),
        *frames,
        confirmed,
        _sim_cfg(),
    )

    assert result.trades.iloc[0]["entry_date"] == dates[21].date()
    assert result.trades.iloc[0]["entry_date"] != dates[20].date()


def test_market_gate_rejection_is_persistable_as_exact_decision():
    dates, frames, volume_m, setups, cfg = _inputs()
    confirmed, events = confirm_daily_breakouts(
        dates, pd.Index(["AAA"]), *frames, volume_m, setups, cfg
    )
    simulation = simulate(
        dates,
        pd.Index(["AAA"]),
        *frames,
        confirmed,
        _sim_cfg(),
        market_exposure_cap=np.zeros(len(dates)),
    )

    final_events = attach_fills(
        events, simulation.trades, simulation.entry_decisions
    )

    assert simulation.trades.empty
    assert final_events.iloc[0]["decision"] == "market_gate_blocked"
