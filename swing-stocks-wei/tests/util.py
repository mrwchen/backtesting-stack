from __future__ import annotations

from src.config import Config


def make_config(**overrides) -> Config:
    values = dict(
        start_date="2024-01-01",
        end_date="2024-12-31",
        warmup_calendar_days=550,
        run_label="test",
        cache_dir="/tmp",
        force_refresh=True,
        log_level="INFO",
        min_price=0.01,
        min_market_cap_usd=2_000_000_000.0,
        revenue_yoy_min=0.20,
        revenue_stale_trading_days=280,
        high_lookback_days=30,
        high_recent_days=10,
        ema_fast_days=3,
        ema_slow_days=5,
        volume_sma_days=5,
        volume_filter_enable=True,
        ibkr_category_breadth_filter_enable=False,
        ibkr_category_breadth_on_threshold=0.55,
        ibkr_category_breadth_off_threshold=0.45,
        ibkr_category_breadth_min_symbols=5,
        initial_equity=100000.0,
        position_size_usd=1000.0,
        stop_loss_pct=0.05,
        trailing_activate_pct=0.10,
        trailing_loss_pct=0.05,
        allow_fractional_shares=True,
    )
    values.update(overrides)
    cfg = Config(**values)
    cfg.validate()
    return cfg
