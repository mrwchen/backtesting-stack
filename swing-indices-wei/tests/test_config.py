import pytest

from src.config import Config


def _base_env(monkeypatch, **overrides):
    env = {
        "START_DATE": "2022-01-03",
        "END_DATE": "2026-07-06",
        **overrides,
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def test_default_table_prefix_and_names(monkeypatch):
    _base_env(monkeypatch)
    cfg = Config.from_env()
    assert cfg.table_prefix == "backtest_wei_"
    assert cfg.runs_table == "backtest_wei_runs"
    assert cfg.trades_table == "backtest_wei_trades"
    assert cfg.equity_table == "backtest_wei_equity_daily"


def test_custom_table_prefix(monkeypatch):
    _base_env(monkeypatch, TABLE_PREFIX="my_experiment_")
    cfg = Config.from_env()
    assert cfg.runs_table == "my_experiment_runs"


@pytest.mark.parametrize("bad", ["1abc_", "Upper_", "has space", "semi;colon", ""])
def test_invalid_table_prefix_rejected(monkeypatch, bad):
    _base_env(monkeypatch, TABLE_PREFIX=bad)
    with pytest.raises(ValueError, match="TABLE_PREFIX"):
        Config.from_env()


def test_default_source_tables(monkeypatch):
    _base_env(monkeypatch)
    cfg = Config.from_env()
    assert cfg.prices_table == "alpaca_market_data_1day"
    assert cfg.scores_table == "world_regime_daily_scores_mv"


def test_custom_source_tables(monkeypatch):
    _base_env(monkeypatch, PRICES_TABLE="ibkr_market_data", SCORES_TABLE="my_scores")
    cfg = Config.from_env()
    assert cfg.prices_table == "ibkr_market_data"
    assert cfg.scores_table == "my_scores"


@pytest.mark.parametrize("env_name", ["PRICES_TABLE", "SCORES_TABLE"])
def test_invalid_source_table_rejected(monkeypatch, env_name):
    _base_env(monkeypatch, **{env_name: "bad;table"})
    with pytest.raises(ValueError, match=env_name):
        Config.from_env()


def test_ema_order_enforced(monkeypatch):
    _base_env(monkeypatch, EMA_FAST="21", EMA_SLOW="9")
    with pytest.raises(ValueError, match="EMA_FAST"):
        Config.from_env()


def test_hysteresis_order_enforced(monkeypatch):
    _base_env(monkeypatch, STRESS_ENTER="52", STRESS_EXIT="57")
    with pytest.raises(ValueError, match="STRESS_EXIT"):
        Config.from_env()
