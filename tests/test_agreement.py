"""Tests for the agreement statistics added for EB-3."""

import numpy as np
import pandas as pd
import pytest

import agreement as A


def _cohort(n, rng):
    return pd.DataFrame({
        "elapsed_time": rng.uniform(40, 130, n),
        "time_on_cpb": rng.uniform(30, 115, n),
        "ibw": rng.normal(70, 10, n),
    })


def test_sign_convention_is_reference_minus_nomogram():
    """The whole point of fixing a convention: a nomogram reading LOW must give
    a POSITIVE difference, everywhere, with no exceptions."""
    assert A.difference(10000.0, 9000.0) == pytest.approx(1000.0)
    assert A.SIGN_CONVENTION == "reference_minus_nomogram"


def test_bland_altman_recovers_a_known_bias_and_spread():
    rng = np.random.default_rng(0)
    ref = rng.uniform(5000, 30000, 20000)
    nom = ref + 250.0 + rng.normal(0, 400, 20000)      # nomogram reads 250 IU high
    r = A.bland_altman(ref, nom)
    assert r.bias == pytest.approx(-250.0, abs=10)
    assert r.sd_diff == pytest.approx(400.0, rel=0.05)
    assert r.loa_low == pytest.approx(-250 - 1.96 * 400, abs=20)
    assert r.loa_high == pytest.approx(-250 + 1.96 * 400, abs=20)
    assert r.bias_ci_low < r.bias < r.bias_ci_high


def test_bland_altman_stats_tuple_is_unchanged_for_callers():
    rng = np.random.default_rng(1)
    ref, nom = rng.uniform(1e4, 3e4, 500), rng.uniform(1e4, 3e4, 500)
    bias, lo, hi = A.bland_altman_stats(ref, nom)
    full = A.bland_altman(ref, nom)
    assert (bias, lo, hi) == (full.bias, full.loa_low, full.loa_high)


def test_bland_altman_needs_enough_pairs():
    with pytest.raises(ValueError):
        A.bland_altman([1.0, 2.0], [1.0, 2.0])


def test_proportional_bias_detects_a_planted_slope():
    rng = np.random.default_rng(2)
    ref = rng.uniform(5000, 30000, 5000)
    nom = 0.95 * ref + rng.normal(0, 200, 5000)     # 5% multiplicative error
    r = A.proportional_bias(ref, nom)
    assert r.significant
    # difference = ref - 0.95*ref = 0.05*ref, and mean = 0.975*ref, so the
    # expected slope is 0.05 / 0.975.
    assert r.slope == pytest.approx(0.05 / 0.975, rel=0.05)
    assert r.slope_ci_low < r.slope < r.slope_ci_high
    assert r.predicted_diff_at_p95 > r.predicted_diff_at_p5


def test_proportional_bias_absent_when_error_is_additive_only():
    rng = np.random.default_rng(3)
    ref = rng.uniform(5000, 30000, 5000)
    nom = ref - 300.0 + rng.normal(0, 200, 5000)
    r = A.proportional_bias(ref, nom)
    assert not r.significant
    assert abs(r.slope) < 0.01


def test_mean_bias_can_hide_a_large_proportional_bias():
    """The reviewers' actual concern (EB-3): a bias near zero is not evidence of
    agreement when the error rotates about the centre of the range."""
    ref = np.linspace(5000, 30000, 2000)
    centre = ref.mean()
    nom = ref + 0.3 * (ref - centre)                 # pure rotation, zero mean bias
    ba = A.bland_altman(ref, nom)
    pb = A.proportional_bias(ref, nom)
    assert abs(ba.bias) < 1.0                        # looks perfect ...
    assert pb.significant and abs(pb.slope) > 0.1    # ... but is not


def test_relative_errors_are_signed_consistently_with_the_convention():
    ref = np.full(100, 10000.0)
    nom = np.full(100, 9000.0)                       # nomogram reads 10% low
    r = A.relative_errors(ref, nom)
    assert r.mean_pct_error == pytest.approx(10.0)
    assert r.mean_abs_pct_error == pytest.approx(10.0)
    assert r.rmse_iu == pytest.approx(1000.0)
    assert r.mae_iu == pytest.approx(1000.0)


def test_threshold_coverage_counts_correctly():
    ref = np.full(100, 10000.0)
    nom = np.concatenate([np.full(60, 9500.0), np.full(40, 8000.0)])
    cov = A.threshold_coverage(ref, nom, threshold_iu=1000.0, threshold_pct=10.0)
    assert cov["pct_within_threshold_iu"] == pytest.approx(60.0)
    assert cov["pct_within_threshold_pct"] == pytest.approx(60.0)
    assert cov["protamine_mg_equivalent_of_threshold"] == pytest.approx(10.0)


def test_stratify_partitions_every_observation():
    rng = np.random.default_rng(4)
    ref = rng.uniform(5000, 30000, 1000)
    nom = ref * 0.97
    by = rng.uniform(0, 100, 1000)
    df = A.stratify(ref, nom, by, "dummy", n_bins=5)
    assert df["n"].sum() == 1000
    assert len(df) == 5


def test_stratify_rejects_a_constant_stratifier():
    with pytest.raises(ValueError):
        A.stratify(np.arange(10.0), np.arange(10.0), np.ones(10), "constant")


def test_stratification_exposes_a_gradient_the_pooled_bias_hides():
    ref = np.linspace(5000, 30000, 4000)
    centre = ref.mean()
    nom = ref + 0.3 * (ref - centre)
    cohort = pd.DataFrame({"elapsed_time": np.linspace(40, 130, 4000),
                           "time_on_cpb": np.linspace(30, 115, 4000)})
    summary, strata = A.full_agreement_report(cohort, ref, nom)
    assert abs(summary["bias"]) < 1.0
    load = strata[strata["stratified_by"] == "residual_load_iu"]
    assert load["bias_iu"].iloc[0] * load["bias_iu"].iloc[-1] < 0, (
        "the stratified table must reveal the sign reversal across the range"
    )


def test_full_report_contains_every_quantity_the_revision_needs():
    rng = np.random.default_rng(5)
    ref = rng.uniform(5000, 30000, 800)
    nom = ref * 0.98 + rng.normal(0, 150, 800)
    summary, strata = A.full_agreement_report(_cohort(800, rng), ref, nom)
    for key in ("bias", "bias_ci_low", "loa_low", "loa_high",
                "prop_bias_slope", "prop_bias_slope_p", "prop_bias_significant",
                "mean_pct_error", "mean_abs_pct_error", "max_abs_pct_error",
                "rmse_iu", "mae_iu", "pct_within_threshold_iu",
                "descriptive_r_squared", "sign_convention"):
        assert key in summary, f"{key} missing from the agreement summary"
    assert set(strata["stratified_by"]) >= {"elapsed_time_min", "residual_load_iu"}
