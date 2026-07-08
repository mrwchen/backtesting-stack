import pytest

from src.config import Config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("START_DATE", "END_DATE", "CATEGORIES", "EMA_FAST", "EMA_SLOW",
                "STRESS_ENTER", "STRESS_EXIT", "CAT_MOM_DEEP_THRESHOLD",
                "TABLE_PREFIX", "METRICS_TABLE", "MAX_POSITIONS",
                "MAX_PER_CATEGORY", "WEIGHT_DEEP_PCT", "MIN_COVERAGE_PCT",
                "SIMULATION_MODE", "TOP_N_PER_CATEGORY"):
        monkeypatch.delenv(var, raising=False)


def test_defaults_are_valid():
    cfg = Config.from_env()
    assert cfg.ema_fast == 9
    assert cfg.ema_slow == 21
    assert cfg.stress_enter == 57
    assert cfg.stress_exit == 52
    assert cfg.cat_mom_window == 63
    assert cfg.cat_mom_deep_threshold == -0.10
    assert cfg.simulation_mode == "portfolio"
    assert cfg.table_prefix == "backtest_wei_stocks_"
    assert cfg.runs_table == "backtest_wei_stocks_runs"
    assert cfg.trades_table == "backtest_wei_stocks_trades"
    assert cfg.equity_table == "backtest_wei_stocks_equity_daily"


def test_categories_parsing(monkeypatch):
    monkeypatch.setenv("CATEGORIES", "Banks, Oil&Gas ,Semiconductors")
    cfg = Config.from_env()
    assert cfg.categories == ("Banks", "Oil&Gas", "Semiconductors")


def test_empty_categories_means_all(monkeypatch):
    monkeypatch.setenv("CATEGORIES", "")
    assert Config.from_env().categories == ()


def test_rejects_inverted_emas(monkeypatch):
    monkeypatch.setenv("EMA_FAST", "21")
    monkeypatch.setenv("EMA_SLOW", "9")
    with pytest.raises(ValueError, match="EMA_FAST"):
        Config.from_env()


def test_rejects_inverted_hysteresis(monkeypatch):
    monkeypatch.setenv("STRESS_ENTER", "50")
    monkeypatch.setenv("STRESS_EXIT", "55")
    with pytest.raises(ValueError, match="STRESS_EXIT"):
        Config.from_env()


def test_rejects_positive_deep_threshold(monkeypatch):
    monkeypatch.setenv("CAT_MOM_DEEP_THRESHOLD", "0.10")
    with pytest.raises(ValueError, match="CAT_MOM_DEEP_THRESHOLD"):
        Config.from_env()


def test_rejects_bad_table_prefix(monkeypatch):
    monkeypatch.setenv("TABLE_PREFIX", "Bad-Prefix!")
    with pytest.raises(ValueError, match="TABLE_PREFIX"):
        Config.from_env()


def test_rejects_bad_source_table(monkeypatch):
    monkeypatch.setenv("METRICS_TABLE", "metrics; DROP TABLE x")
    with pytest.raises(ValueError, match="METRICS_TABLE"):
        Config.from_env()


def test_accepts_independent_mode(monkeypatch):
    monkeypatch.setenv("SIMULATION_MODE", "independent")
    assert Config.from_env().simulation_mode == "independent"


def test_rejects_unknown_simulation_mode(monkeypatch):
    monkeypatch.setenv("SIMULATION_MODE", "cash_machine")
    with pytest.raises(ValueError, match="SIMULATION_MODE"):
        Config.from_env()


def test_top_n_zero_means_uncapped_universe(monkeypatch):
    monkeypatch.setenv("TOP_N_PER_CATEGORY", "0")
    assert Config.from_env().top_n_per_category == 0


def test_rejects_negative_top_n(monkeypatch):
    monkeypatch.setenv("TOP_N_PER_CATEGORY", "-1")
    with pytest.raises(ValueError, match="TOP_N_PER_CATEGORY"):
        Config.from_env()
