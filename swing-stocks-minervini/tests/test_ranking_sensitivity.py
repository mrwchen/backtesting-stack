import pytest
import pandas as pd

from src import ranking_sensitivity
from src.candidate_ranking import QualityCalibrationLabel


def test_bootstrap_salts_are_frozen_unique_and_complete() -> None:
    salts = ranking_sensitivity.NEUTRAL_RANK_SALTS

    assert len(salts) == 32
    assert len(set(salts)) == 32
    assert salts[0] == "v8-bootstrap-00"
    assert salts[-1] == "v8-bootstrap-31"


def test_experiment_cases_are_the_fixed_nine_mode_sleeve_combinations() -> None:
    cases = ranking_sensitivity.EXPERIMENT_CASES

    assert [case.name for case in cases] == [
        "neutral_flat_base",
        "neutral_vcp",
        "neutral_combined",
        "quality_only_flat_base",
        "quality_only_vcp",
        "quality_only_combined",
        "validated_flat_base",
        "validated_vcp",
        "validated_combined",
    ]
    assert len({case.name for case in cases}) == 9
    assert {case.ranking_mode for case in cases} == {
        "neutral",
        "quality_only",
        "validated",
    }
    assert {
        (case.sleeve, case.setup_types) for case in cases
    } == {
        ("flat_base", ("flat_base",)),
        ("vcp", ("vcp",)),
        ("combined", ("flat_base", "vcp")),
    }


def _quality_label(
    setup_type: str,
    available_date: str,
    *,
    information_date: str = "2024-01-02",
    weight: float = 1.0,
) -> QualityCalibrationLabel:
    return QualityCalibrationLabel(
        setup_type=setup_type,
        information_date=pd.Timestamp(information_date),
        available_date=pd.Timestamp(available_date),
        raw_quality_score=75.0,
        realized_r_multiple=1.0,
        walk_forward_quality_score=0.25,
        weight=weight,
    )


def test_clustered_bootstrap_is_deterministic_positive_and_salt_sensitive() -> None:
    labels = (
        _quality_label("flat_base", "2024-01-03"),
        _quality_label("flat_base", "2024-01-03", weight=2.0),
        _quality_label("flat_base", "2024-01-04"),
        _quality_label("vcp", "2024-01-03"),
        _quality_label("vcp", "2024-01-05"),
    )

    first = ranking_sensitivity.bootstrap_quality_labels(
        labels, "v8-bootstrap-00"
    )
    repeated = ranking_sensitivity.bootstrap_quality_labels(
        labels, "v8-bootstrap-00"
    )
    changed = ranking_sensitivity.bootstrap_quality_labels(
        labels, "v8-bootstrap-01"
    )

    assert first == repeated
    assert all(label.weight > 0 for label in first)
    assert first != changed
    # The two same-class/same-completion-day labels share one multiplier.
    assert first[0].weight / labels[0].weight == pytest.approx(
        first[1].weight / labels[1].weight
    )
    assert labels[0].weight == 1.0


def test_clustered_bootstrap_is_input_order_independent() -> None:
    labels = (
        _quality_label("flat_base", "2024-01-03"),
        _quality_label("flat_base", "2024-01-04"),
        _quality_label("vcp", "2024-01-05"),
    )

    forward = ranking_sensitivity.bootstrap_quality_labels(
        labels, "v8-bootstrap-07"
    )
    reversed_result = ranking_sensitivity.bootstrap_quality_labels(
        tuple(reversed(labels)), "v8-bootstrap-07"
    )

    forward_weights = {
        (label.setup_type, label.available_date): label.weight for label in forward
    }
    reversed_weights = {
        (label.setup_type, label.available_date): label.weight
        for label in reversed_result
    }
    assert forward_weights == reversed_weights


def test_future_completion_cluster_cannot_reweight_an_earlier_label() -> None:
    early = _quality_label("flat_base", "2024-01-03")
    future = _quality_label("flat_base", "2024-06-03")

    early_only = ranking_sensitivity.bootstrap_quality_labels(
        (early,), "v8-bootstrap-12"
    )
    with_future = ranking_sensitivity.bootstrap_quality_labels(
        (early, future), "v8-bootstrap-12"
    )

    assert early_only[0].weight == with_future[0].weight


def test_clustered_bootstrap_validates_salt_and_accepts_empty_labels() -> None:
    assert ranking_sensitivity.bootstrap_quality_labels((), "salt") == ()
    with pytest.raises(ValueError, match="salt"):
        ranking_sensitivity.bootstrap_quality_labels((), "")
    with pytest.raises(TypeError, match="salt"):
        ranking_sensitivity.bootstrap_quality_labels((), 7)  # type: ignore[arg-type]


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
