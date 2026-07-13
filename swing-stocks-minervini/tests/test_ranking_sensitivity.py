import pytest

from src import ranking_sensitivity


def test_neutral_ranking_salts_are_frozen_unique_and_complete() -> None:
    salts = ranking_sensitivity.NEUTRAL_RANK_SALTS

    assert len(salts) == 32
    assert len(set(salts)) == 32
    assert salts[0] == "v7-neutral-00"
    assert salts[-1] == "v7-neutral-31"


def test_summary_reports_median_adverse_decile_and_worst() -> None:
    results = [
        {
            "total_return": float(value),
            "cagr": float(value) / 10,
            "max_drawdown": float(value) / 100,
            "profit_factor": None if value == 0 else float(value),
            "avg_r_multiple": -float(value),
        }
        for value in range(32)
    ]

    summary = ranking_sensitivity.summarize(results)

    assert summary["total_return"] == {
        "count": 32,
        "missing_count": 0,
        "median": 15.5,
        "adverse_quantile": pytest.approx(3.1),
        "worst": 0.0,
    }
    assert summary["max_drawdown"]["adverse_quantile"] == pytest.approx(0.279)
    assert summary["max_drawdown"]["worst"] == pytest.approx(0.31)
    assert summary["avg_r_multiple"]["worst"] == -31.0
    assert summary["profit_factor"]["count"] == 31
    assert summary["profit_factor"]["missing_count"] == 1
