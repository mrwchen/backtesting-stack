from dataclasses import replace
from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from src import forward_shadow, persistence
from src.candidate_ranking import FillCalibrationLabel, QualityCalibrationLabel
from src.reproducibility import (
    SCREEN_CONFIG_FIELDS,
    SIM_CONFIG_FIELDS,
    config_fingerprint,
    fill_calibration_labels_fingerprint,
    frame_fingerprint,
    matrix_fingerprint,
    quality_calibration_labels_fingerprint,
)
from tests.util import make_cfg


def test_config_fingerprint_ignores_run_identity_but_tracks_screen_inputs() -> None:
    first = make_cfg(run_label="one", stage="screen", rs_min=70)
    same_inputs = make_cfg(run_label="two", stage="setup", rs_min=70)
    changed = make_cfg(run_label="two", stage="screen", rs_min=80)

    assert config_fingerprint(
        first, SCREEN_CONFIG_FIELDS, model_version="m1"
    ) == config_fingerprint(same_inputs, SCREEN_CONFIG_FIELDS, model_version="m1")
    assert config_fingerprint(
        first, SCREEN_CONFIG_FIELDS, model_version="m1"
    ) != config_fingerprint(changed, SCREEN_CONFIG_FIELDS, model_version="m1")


def test_sim_config_fingerprint_tracks_slate_risk_floor() -> None:
    first = make_cfg(min_slate_risk_utilization=0.50)
    changed = make_cfg(min_slate_risk_utilization=0.75)

    assert config_fingerprint(
        first, SIM_CONFIG_FIELDS, model_version="m1"
    ) != config_fingerprint(changed, SIM_CONFIG_FIELDS, model_version="m1")


def test_sim_config_fingerprint_tracks_daily_order_limit() -> None:
    first = make_cfg(portfolio_max_daily_orders=3)
    changed = make_cfg(portfolio_max_daily_orders=2)

    assert config_fingerprint(
        first, SIM_CONFIG_FIELDS, model_version="m1"
    ) != config_fingerprint(changed, SIM_CONFIG_FIELDS, model_version="m1")


def test_sim_config_fingerprint_tracks_neutral_rank_salt() -> None:
    first = make_cfg(neutral_rank_salt="v9-neutral-control-00")
    changed = make_cfg(neutral_rank_salt="v9-neutral-control-01")

    assert config_fingerprint(
        first, SIM_CONFIG_FIELDS, model_version="m1"
    ) != config_fingerprint(changed, SIM_CONFIG_FIELDS, model_version="m1")


def test_sim_config_fingerprint_tracks_ranking_mode_and_setup_types() -> None:
    base = make_cfg(
        portfolio_ranking_mode="neutral",
        portfolio_setup_types=("flat_base",),
    )
    changed_mode = make_cfg(portfolio_ranking_mode="relative_quality")
    changed_types = make_cfg(portfolio_setup_types=("vcp",))

    base_fingerprint = config_fingerprint(
        base, SIM_CONFIG_FIELDS, model_version="m1"
    )
    assert base_fingerprint != config_fingerprint(
        changed_mode, SIM_CONFIG_FIELDS, model_version="m1"
    )
    assert base_fingerprint != config_fingerprint(
        changed_types, SIM_CONFIG_FIELDS, model_version="m1"
    )


def test_all_frozen_control_salts_have_unique_simulation_fingerprints() -> None:
    fingerprints = {
        config_fingerprint(
            make_cfg(neutral_rank_salt=salt),
            SIM_CONFIG_FIELDS,
            model_version="m1",
        )
        for salt in forward_shadow.NEUTRAL_CONTROL_SALTS
    }

    assert len(fingerprints) == 32


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("forward_start_date", "2026-07-14"),
        ("force_close_at_end", True),
    ],
)
def test_sim_config_fingerprint_tracks_forward_contract(
    field: str, changed_value: object
) -> None:
    baseline = make_cfg()
    changed = make_cfg(**{field: changed_value})

    assert config_fingerprint(
        baseline, SIM_CONFIG_FIELDS, model_version="m1"
    ) != config_fingerprint(changed, SIM_CONFIG_FIELDS, model_version="m1")


def test_frame_fingerprint_is_order_independent_and_content_sensitive() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["BBB", "AAA"],
            "period_end_date": [pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-02")],
        }
    )
    reordered = frame.iloc[::-1].reset_index(drop=True)
    changed = frame.copy()
    changed.loc[0, "symbol"] = "CCC"

    columns = ("symbol", "period_end_date")
    assert frame_fingerprint(frame, columns) == frame_fingerprint(reordered, columns)
    assert frame_fingerprint(frame, columns) != frame_fingerprint(changed, columns)


def test_frame_fingerprint_is_stable_across_database_numeric_types() -> None:
    memory = pd.DataFrame(
        {"symbol": ["AAA"], "pivot": [100.125], "depths": [(0.2, 0.1)]}
    )
    database = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "pivot": [Decimal("100.12500000")],
            "depths": [[Decimal("0.200000"), Decimal("0.100000")]],
        }
    )

    columns = ("symbol", "pivot", "depths")
    assert frame_fingerprint(memory, columns) == frame_fingerprint(database, columns)


def test_quality_label_fingerprint_is_order_independent() -> None:
    first = QualityCalibrationLabel(
        setup_type="vcp",
        information_date="2024-01-02",
        available_date="2024-01-05",
        raw_quality_score=75.0,
        realized_r_multiple=1.5,
        walk_forward_quality_score=0.25,
        weight=1.25,
    )
    second = replace(
        first,
        setup_type="flat_base",
        information_date="2024-01-03",
        available_date="2024-01-08",
    )

    assert quality_calibration_labels_fingerprint(
        (first, second)
    ) == quality_calibration_labels_fingerprint((second, first))


@pytest.mark.parametrize(
    "changes",
    (
        {"setup_type": "flat_base"},
        {"information_date": "2024-01-03"},
        {"available_date": "2024-01-08"},
        {"raw_quality_score": 76.0},
        {"realized_r_multiple": -0.5},
        {"walk_forward_quality_score": 0.50},
        {"weight": 2.0},
    ),
)
def test_quality_label_fingerprint_tracks_every_semantic_field(changes) -> None:
    label = QualityCalibrationLabel(
        setup_type="vcp",
        information_date="2024-01-02",
        available_date="2024-01-05",
        raw_quality_score=75.0,
        realized_r_multiple=1.5,
        walk_forward_quality_score=0.25,
        weight=1.25,
    )

    assert quality_calibration_labels_fingerprint(
        (label,)
    ) != quality_calibration_labels_fingerprint((replace(label, **changes),))


def test_fill_label_fingerprint_is_order_independent_and_content_sensitive() -> None:
    first = FillCalibrationLabel(
        setup_type="vcp",
        information_date="2024-01-02",
        available_date="2024-01-03",
        readiness_signal=75.0,
        filled=True,
        weight=1.25,
    )
    second = replace(
        first,
        setup_type="flat_base",
        information_date="2024-01-03",
        available_date="2024-01-04",
        filled=False,
    )

    assert fill_calibration_labels_fingerprint(
        (first, second)
    ) == fill_calibration_labels_fingerprint((second, first))
    for changes in (
        {"readiness_signal": 76.0},
        {"filled": False},
        {"weight": 2.0},
    ):
        assert fill_calibration_labels_fingerprint(
            (first,)
        ) != fill_calibration_labels_fingerprint((replace(first, **changes),))


def test_matrix_fingerprint_tracks_price_content_not_dataframe_dtype() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    symbols = pd.Index(["AAA"])
    first = {"close": pd.DataFrame({"AAA": [10.0, 11.0]}, index=dates)}
    same = {"close": first["close"].astype("float32")}
    changed = {"close": pd.DataFrame({"AAA": [10.0, 12.0]}, index=dates)}

    assert matrix_fingerprint(dates, symbols, first) == matrix_fingerprint(
        dates, symbols, same
    )
    assert matrix_fingerprint(dates, symbols, first) != matrix_fingerprint(
        dates, symbols, changed
    )


def test_stage_state_requires_the_exact_persisted_date_range(monkeypatch) -> None:
    state = pd.DataFrame(
        {
            "model_version": ["m1"],
            "config_fingerprint": ["cfg"],
            "input_fingerprint": ["input"],
            "output_fingerprint": ["output"],
            "start_date": [date(2020, 1, 2)],
            "end_date": [date(2023, 12, 31)],
        }
    )
    monkeypatch.setattr(persistence.db, "read_df", lambda *_args, **_kwargs: state)

    with pytest.raises(RuntimeError, match="does not match"):
        persistence.require_stage_state(
            object(),
            stage="screen",
            model_version="m1",
            config_fingerprint="cfg",
            input_fingerprint="input",
            start=date(2022, 1, 3),
            end=date(2023, 12, 31),
        )
