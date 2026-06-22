import unittest
from dataclasses import replace

from hfmed_core import config, parameters
from hfmed_core.optimizer import Evaluation, score_session_stats, select_session_parameters


def _values(**overrides):
    values = {
        "LOOKBACK_BARS": 60,
        "LONG_CROSS_QUANTILE": 0.5,
        "SHORT_CROSS_QUANTILE": 0.5,
        "ALL_STOP_MODES_TAKE_PROFIT_POINTS": 20.0,
        "BAND_STOP_MIN_PROFILE_RANGE_POINTS": 40.0,
        "BAND_STOP_PROFILE_LOWER_QUANTILE": 0.05,
        "BAND_STOP_PROFILE_UPPER_QUANTILE": 0.95,
        "BAND_STOP_PROFILE_BUFFER_POINTS": 1.0,
        "BAND_STOP_MIN_DISTANCE_POINTS": 12.0,
        "BAND_STOP_MAX_DISTANCE_POINTS": 30.0,
    }
    values.update(overrides)
    return values


def _stats(trades, wins, gross_profit, gross_loss, std_trade_pnl):
    losses = max(0, trades - wins)
    net = gross_profit - gross_loss
    return {
        "total_trades": trades,
        "winning_trades": wins,
        "losing_trades": losses,
        "breakeven_trades": 0,
        "win_rate_pct": wins / trades * 100.0 if trades else 0.0,
        "gross_profit_eur": gross_profit,
        "gross_loss_eur": gross_loss,
        "net_profit_eur": net,
        "avg_trade_pnl_eur": net / trades if trades else 0.0,
        "std_trade_pnl_eur": std_trade_pnl,
    }


def _evaluation(values, session_type, stats):
    digest = parameters.parameter_hash(values)
    return Evaluation(
        stage="stage1",
        fold_index=1,
        window_role="train",
        values=values,
        parameter_hash=digest,
        parameter_label=parameters.parameter_label(values),
        window_start=None,
        window_end=None,
        ticks_simulated=0,
        bars_total=0,
        signals_total=0,
        long_signals=0,
        short_signals=0,
        rejected_missing_band=0,
        rejected_band_too_narrow=0,
        rejected_stop_too_small=0,
        rejected_stop_too_large=0,
        skipped_no_size=0,
        ruined=False,
        summary={},
        score=0.0,
        session_stats={session_type: stats},
        trades=[],
    )


class SessionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.base_cfg = replace(config.active_run_config(), initial_equity=5000.0)
        self.opt_cfg = replace(
            config.active_optimizer_config(),
            session_selector_min_trades=20,
            session_selector_lcb_z=1.0,
            session_selector_top_n=5,
            session_selector_plateau_weight=0.0,
            session_selector_previous_keep_score_tolerance=0.0,
            min_oos_profit_factor=1.1,
        )

    def test_lower_confidence_score_prefers_stable_candidate(self):
        session = "ny_morning"
        spiky = _evaluation(
            _values(LOOKBACK_BARS=30),
            session,
            _stats(trades=20, wins=12, gross_profit=2000.0, gross_loss=1000.0, std_trade_pnl=250.0),
        )
        stable = _evaluation(
            _values(LOOKBACK_BARS=90),
            session,
            _stats(trades=30, wins=18, gross_profit=1200.0, gross_loss=400.0, std_trade_pnl=25.0),
        )

        self.assertGreater(spiky.session_stats[session]["net_profit_eur"], stable.session_stats[session]["net_profit_eur"])
        self.assertGreater(
            score_session_stats(stable.session_stats[session], self.base_cfg, self.opt_cfg),
            score_session_stats(spiky.session_stats[session], self.base_cfg, self.opt_cfg),
        )

        selected = select_session_parameters([spiky, stable], self.base_cfg, self.opt_cfg)

        self.assertEqual(selected[session].parameter_hash, stable.parameter_hash)

    def test_previous_selection_is_kept_when_close_enough(self):
        session = "ny_morning"
        previous = _evaluation(
            _values(LOOKBACK_BARS=60),
            session,
            _stats(trades=35, wins=20, gross_profit=1100.0, gross_loss=400.0, std_trade_pnl=30.0),
        )
        challenger = _evaluation(
            _values(LOOKBACK_BARS=120),
            session,
            _stats(trades=36, wins=22, gross_profit=1160.0, gross_loss=390.0, std_trade_pnl=30.0),
        )
        opt_cfg = replace(self.opt_cfg, session_selector_previous_keep_score_tolerance=100.0)

        selected = select_session_parameters(
            [previous, challenger],
            self.base_cfg,
            opt_cfg,
            previous_selected_by_session={session: previous},
        )

        self.assertEqual(selected[session].parameter_hash, previous.parameter_hash)


if __name__ == "__main__":
    unittest.main()
