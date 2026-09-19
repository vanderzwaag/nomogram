"""The reproduction checker's own sensitivity.

A checker that never fails is worthless and a checker that always fails gets
switched off, so both directions are asserted here: floating-point rounding
must pass, and a result actually moving must not.
"""

import numpy as np
import pandas as pd
import pytest

import check_frozen as cf


def _frame():
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "model": ["a", "b", "c", "d"],
        "residual_iu": [21643.2, 19477.9, 33000.0, 12800.5],
        # A column that is numerical zero by construction: an objective drives
        # it to zero, so its whole range is a rounding artefact.
        "bias_recalibrated_iu": [0.0104475966354366, -0.0451654194297807,
                                 0.0292842663750197, -0.0011],
        "p_value": [5e-77, 0.43, 1e-12, 0.99],
    })


def _write(tmp_path, name, frame):
    d = tmp_path / name
    d.mkdir()
    frame.to_csv(d / "01_results.csv", index=False)
    (d / "manifest.json").write_text('{"decisions": {"seed": 1}}')
    return d


def test_rounding_in_a_near_zero_column_passes(tmp_path, capsys):
    """One ulp on a near-cancellation is not a result moving.

    This is the case that failed CI: a bias of 0.0104 IU differing in its last
    bits looks like a relative difference of 1e-10, but the absolute difference
    is 1e-12 IU and the whole column spans 0.045 IU.
    """
    frozen = _frame()
    fresh = frozen.copy()
    fresh.loc[0, "bias_recalibrated_iu"] = 0.0104475966365353
    fresh.loc[0, "p_value"] = 5.0000000000003e-77

    a = _write(tmp_path, "frozen", frozen)
    b = _write(tmp_path, "fresh", fresh)
    drift = cf.compare_csv(a / "01_results.csv", b / "01_results.csv",
                           1e-9, {"value": 0.0, "where": None})
    assert drift == [], drift


def test_a_result_moving_fails(tmp_path):
    """The copula defect, in miniature: a reported value shifts by 1-2%."""
    frozen = _frame()
    fresh = frozen.copy()
    fresh.loc[1, "residual_iu"] = 19477.9 * 1.015

    a = _write(tmp_path, "frozen", frozen)
    b = _write(tmp_path, "fresh", fresh)
    drift = cf.compare_csv(a / "01_results.csv", b / "01_results.csv",
                           1e-9, {"value": 0.0, "where": None})
    assert drift and "residual_iu" in drift[0]


def test_a_small_absolute_move_in_a_large_column_still_fails(tmp_path):
    """Scaling by the column must not hide a change that matters.

    1 IU on a 21,643 IU residual is 5e-5 of the column: far below anything
    clinically meaningful, but far above rounding, so it is reported.
    """
    frozen = _frame()
    fresh = frozen.copy()
    fresh.loc[0, "residual_iu"] = 21644.2

    a = _write(tmp_path, "frozen", frozen)
    b = _write(tmp_path, "fresh", fresh)
    drift = cf.compare_csv(a / "01_results.csv", b / "01_results.csv",
                           1e-9, {"value": 0.0, "where": None})
    assert drift


def test_missing_and_extra_rows_are_reported(tmp_path):
    a = _write(tmp_path, "frozen", _frame())
    b = _write(tmp_path, "fresh", _frame().iloc[:2])
    drift = cf.compare_csv(a / "01_results.csv", b / "01_results.csv",
                           1e-9, {"value": 0.0, "where": None})
    assert drift and "rows" in drift[0]

    drift = cf.compare_csv(a / "01_results.csv", tmp_path / "nope.csv",
                           1e-9, {"value": 0.0, "where": None})
    assert drift and "missing" in drift[0]
