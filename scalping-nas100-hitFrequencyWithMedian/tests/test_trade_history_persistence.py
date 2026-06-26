import unittest
from datetime import datetime, timedelta, timezone

from hfmed_core import config, persistence
from hfmed_core.entities import ClosedTrade


def _closed_trade(entry_ts: datetime, exit_ts: datetime, direction: str = "LONG") -> ClosedTrade:
    return ClosedTrade(
        signal_ts=entry_ts,
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        direction=direction,
        entry_session="ny_morning",
        cross_quantile=0.5,
        cross_level=100.0,
        profile_low=90.0,
        profile_high=110.0,
        profile_range=20.0,
        cross_price_range_position_pct=50.0,
        entry_price_range_position_pct=51.0,
        range_position_deviation_pct=1.0,
        median_level=100.0,
        signal_mid=100.5,
        previous_mid=99.5,
        entry_bid=100.0,
        entry_ask=101.0,
        entry_price=101.0,
        exit_bid=105.0,
        exit_ask=106.0,
        exit_price=105.0,
        stop_price=99.0,
        take_profit_price=105.0,
        units=2.5,
        notional_eur=252.5,
        margin_used_eur=12.63,
        gross_pnl_eur=10.0,
        extra_costs_eur=1.234,
        pnl_eur=8.766,
        equity_before=5000.0,
        equity_after=5008.766,
        return_pct=0.17532,
        price_pnl_points=4.0,
        outcome_status="HIT_TP",
        ticks_held=42,
        seconds_held=(exit_ts - entry_ts).total_seconds(),
        realized_risk_eur=5.0,
        realized_risk_pct=0.1,
        margin_capped=False,
    )


class TradeHistoryPersistenceTests(unittest.TestCase):
    def test_trade_history_rows_use_configured_defaults_and_trade_values(self):
        cfg = config.active_run_config()
        entry_ts = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
        exit_ts = entry_ts + timedelta(minutes=3)

        rows = persistence._trade_history_rows(cfg, [_closed_trade(entry_ts, exit_ts)])
        row = dict(zip(persistence.TRADE_HISTORY_COLUMNS, rows[0]))

        self.assertEqual("000001", row["account_number"])
        self.assertEqual("backtesting", row["account_type"])
        self.assertEqual(-1, row["closing_deal_id"])
        self.assertEqual(-1, row["position_id"])
        self.assertEqual(cfg.symbol, row["symbol_name"])
        self.assertEqual("Buy", row["trade_type"])
        self.assertEqual(entry_ts, row["entry_time_utc"])
        self.assertEqual(exit_ts, row["closing_time_utc"])
        self.assertEqual(timedelta(minutes=3), row["holding_duration"])
        self.assertEqual(10.0, row["gross_profit"])
        self.assertEqual(8.77, row["net_profit"])
        self.assertEqual(-1.23, row["commissions"])
        self.assertEqual(5008.77, row["balance"])
        self.assertEqual(cfg.account_currency, row["deposit_asset"])

    def test_trade_history_rows_are_sorted_by_closing_time(self):
        cfg = config.active_run_config()
        base = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
        later = _closed_trade(base, base + timedelta(minutes=5), direction="SHORT")
        earlier = _closed_trade(base, base + timedelta(minutes=1), direction="LONG")

        rows = persistence._trade_history_rows(cfg, [later, earlier])
        mapped = [dict(zip(persistence.TRADE_HISTORY_COLUMNS, row)) for row in rows]

        self.assertEqual(base + timedelta(minutes=1), mapped[0]["closing_time_utc"])
        self.assertEqual("Buy", mapped[0]["trade_type"])
        self.assertEqual(base + timedelta(minutes=5), mapped[1]["closing_time_utc"])
        self.assertEqual("Sell", mapped[1]["trade_type"])


if __name__ == "__main__":
    unittest.main()
