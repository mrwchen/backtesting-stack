from decimal import Decimal
from datetime import date

import pandas as pd
import pytest

from src import persistence
from src.reproducibility import (
    SCREEN_CONFIG_FIELDS,
    SIM_CONFIG_FIELDS,
    config_fingerprint,
    frame_fingerprint,
    matrix_fingerprint,
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
