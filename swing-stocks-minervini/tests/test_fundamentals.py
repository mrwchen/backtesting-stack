import numpy as np
import pandas as pd

from src.fundamentals import combine, eps_flags, revenue_margin_flags
from tests.util import make_cfg


def _grid():
    dates = pd.bdate_range("2022-01-03", "2024-12-31")
    symbols = pd.Index(["AAA"])
    return dates, symbols


def _fundamental_filings(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": "AAA",
            "available_date": pd.date_range("2022-02-01", periods=n, freq="91D"),
            "revenue_ttm": 1000.0 * 1.05 ** np.arange(n),
            "net_margin_ttm": 0.10 + 0.005 * np.arange(n),
        }
    )


def _eps_events(rows: list[tuple[str, float | None, float | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": "AAA",
            "available_date": pd.to_datetime([row[0] for row in rows]),
            "diluted_eps": [row[1] for row in rows],
            "prior_year_diluted_eps": [row[2] for row in rows],
        }
    )


def test_eps_yoy_uses_reported_current_and_prior_quarter():
    dates, symbols = _grid()
    events = _eps_events([("2022-02-01", 1.25, 1.00)])
    cfg = make_cfg(eps_yoy_min=0.20)

    eps_pass, eps_yoy = eps_flags(events, dates, symbols, cfg)

    assert not eps_pass.loc[: "2022-01-31", "AAA"].any()
    assert bool(eps_pass.loc["2022-02-01", "AAA"]) is True
    assert abs(eps_yoy.loc["2022-02-01", "AAA"] - 0.25) < 1e-12


def test_eps_growth_below_threshold_fails():
    dates, symbols = _grid()
    events = _eps_events([("2022-02-01", 1.19, 1.00)])

    eps_pass, eps_yoy = eps_flags(events, dates, symbols, make_cfg(eps_yoy_min=0.20))

    assert bool(eps_pass.loc["2022-02-01", "AAA"]) is False
    assert abs(eps_yoy.loc["2022-02-01", "AAA"] - 0.19) < 1e-12


def test_positive_eps_turnaround_passes_without_ratio():
    dates, symbols = _grid()
    events = _eps_events([("2022-02-01", 0.10, -0.25)])

    eps_pass, eps_yoy = eps_flags(events, dates, symbols, make_cfg())

    assert bool(eps_pass.loc["2022-02-01", "AAA"]) is True
    assert pd.isna(eps_yoy.loc["2022-02-01", "AAA"])


def test_missing_eps_event_clears_previous_state():
    dates, symbols = _grid()
    events = _eps_events(
        [
            ("2022-02-01", 1.25, 1.00),
            ("2022-05-02", None, 1.00),
        ]
    )

    eps_pass, eps_yoy = eps_flags(events, dates, symbols, make_cfg())

    assert bool(eps_pass.loc["2022-04-29", "AAA"]) is True
    assert bool(eps_pass.loc["2022-05-02", "AAA"]) is False
    assert pd.isna(eps_yoy.loc["2022-05-02", "AAA"])
    assert pd.isna(eps_yoy.loc["2022-06-01", "AAA"])


def test_eps_flag_goes_stale():
    dates, symbols = _grid()
    events = _eps_events([("2022-02-01", 1.25, 1.00)])
    cfg = make_cfg(eps_stale_trading_days=130)

    eps_pass, _ = eps_flags(events, dates, symbols, cfg)

    event_idx = dates.get_loc(pd.Timestamp("2022-02-01"))
    assert bool(eps_pass.iloc[event_idx + 130]["AAA"]) is True
    assert bool(eps_pass.iloc[event_idx + 131]["AAA"]) is False


def test_revenue_and_margin_flags():
    dates, symbols = _grid()
    filings = _fundamental_filings(9)
    cfg = make_cfg(revenue_yoy_min=0.10)

    revenue_pass, revenue_yoy, margin_pass = revenue_margin_flags(filings, dates, symbols, cfg)

    late = filings["available_date"].iloc[-1] + pd.Timedelta(days=5)
    late_row = revenue_pass.loc[late:].iloc[0]
    # 4 filings/year at 5% each ~ 21% YoY -> pass; margin strictly rising -> pass
    assert bool(late_row["AAA"]) is True
    assert revenue_yoy.loc[late:, "AAA"].iloc[0] > 0.15
    assert bool(margin_pass.loc[late:, "AAA"].iloc[0]) is True


def test_revenue_yoy_zero_prior_revenue_is_not_infinite():
    dates, symbols = _grid()
    filings = pd.DataFrame(
        {
            "symbol": "AAA",
            "available_date": pd.to_datetime(["2022-02-01", "2023-02-01"]),
            "revenue_ttm": [0.0, 1000.0],
            "net_margin_ttm": [0.10, 0.12],
        }
    )

    revenue_pass, revenue_yoy, _ = revenue_margin_flags(filings, dates, symbols, make_cfg())

    check_date = pd.Timestamp("2023-02-06")
    assert not np.isinf(revenue_yoy.to_numpy(dtype=float)).any()
    assert pd.isna(revenue_yoy.loc[check_date, "AAA"])
    assert bool(revenue_pass.loc[check_date, "AAA"]) is False


def test_combine_requires_min_pass():
    dates, symbols = _grid()
    yes = pd.DataFrame(True, index=dates, columns=symbols)
    no = pd.DataFrame(False, index=dates, columns=symbols)

    assert combine(yes, yes, no, make_cfg(fundamentals_min_pass=2)).all().all()
    assert not combine(yes, no, no, make_cfg(fundamentals_min_pass=2)).any().any()
