"""Tests for the calibration machinery -- above all, that it is reproducible.

The reproducibility tests here are the ones that answer EB-3 / R1's Figure 2
comment at its root: the submitted code drew unseeded samples, so no two runs
produced the same statistics, and the sign of a near-zero bias was effectively
random from run to run.
"""

import numpy as np
import pytest

import agreement as A
import calibration as cal
import nomogram_core as core

SPEC = cal.CohortSpec("test", 400.0, 5000.0, 70.0, 10.0, 15.0, 3.75, 60.0, 15.0)
MODELS = list(core.MODEL_NAMES)


# ------------------------------------------------------------ reproducibility

@pytest.mark.parametrize("model", MODELS)
def test_same_seed_gives_identical_k(model):
    a = cal.find_best_k(SPEC, model, seed=12345, n_sim=400)
    b = cal.find_best_k(SPEC, model, seed=12345, n_sim=400)
    assert a.k == b.k


@pytest.mark.parametrize("model", MODELS)
def test_different_seed_gives_different_cohort(model):
    """Guards against the opposite failure: a seed that is silently ignored."""
    a = cal.find_best_k(SPEC, model, seed=1, n_sim=400)
    b = cal.find_best_k(SPEC, model, seed=2, n_sim=400)
    assert a.k != b.k


def test_cohort_sampling_is_seeded():
    c1 = cal.sample_cohort(SPEC, 100, np.random.default_rng(7))
    c2 = cal.sample_cohort(SPEC, 100, np.random.default_rng(7))
    c3 = cal.sample_cohort(SPEC, 100, np.random.default_rng(8))
    assert c1.equals(c2)
    assert not c1.equals(c3)


def test_monte_carlo_precision_is_reproducible():
    a = cal.monte_carlo_precision(SPEC, "lanoiselee", seed=99, n_replicates=5, n_sim=200)
    b = cal.monte_carlo_precision(SPEC, "lanoiselee", seed=99, n_replicates=5, n_sim=200)
    assert np.array_equal(a["k"].to_numpy(), b["k"].to_numpy())


def test_input_sensitivity_is_reproducible():
    a = cal.input_sensitivity(SPEC, "lanoiselee", seed=4, n_sim=200)
    b = cal.input_sensitivity(SPEC, "lanoiselee", seed=4, n_sim=200)
    assert np.array_equal(a["k"].to_numpy(), b["k"].to_numpy())
    assert set(a["parameter"]) == {"ibw_mean", "ibw_sd", "t_to_mean",
                                   "t_to_sd", "t_on_mean", "t_on_sd"}
    assert set(a["variation"]) == {0.2}


def test_bias_sign_is_unstable_without_a_fixed_seed():
    """Reproduces the reviewers' EB-3 observation from first principles.

    The primary objective drives the mean bias to nearly zero, so when the
    agreement statistics are computed on a cohort other than the one k was
    calibrated on -- which is what the submitted pipeline did -- the residual
    bias is a few IU with a sign that changes from run to run. That is the
    mechanism behind -5.5 IU in the text and +5.62 IU in the Figure 2 legend.
    """
    signs = set()
    for seed in range(12):
        k = cal.find_best_k(SPEC, "lanoiselee", seed=1000 + seed, n_sim=1000).k
        cohort = cal.sample_cohort(SPEC, 1000, np.random.default_rng(2000 + seed))
        times = cal.evaluation_times(cohort, "reversal_endpoint")
        ref = cal.reference_values(cohort, "lanoiselee", times).ravel()
        nom = cal.nomogram_values(cohort, times, k).ravel()
        bias = A.bland_altman(ref, nom).bias
        assert abs(bias) < 50.0          # small in magnitude ...
        signs.add(bias > 0)              # ... but not stable in sign
    assert signs == {True, False}


# ------------------------------------------------------------------ mechanics

@pytest.mark.parametrize("model", MODELS)
def test_k_stays_inside_the_plausible_interval(model):
    r = cal.find_best_k(SPEC, model, seed=11, n_sim=400)
    assert cal.K_BOUNDS[0] <= r.k <= cal.K_BOUNDS[1]
    assert not r.at_lower_bound and not r.at_upper_bound, (
        f"{model} optimum sits on the search boundary; the interval "
        f"{cal.K_BOUNDS} is too narrow for it"
    )


@pytest.mark.parametrize("objective", sorted(cal.OBJECTIVES))
def test_every_objective_calibrates(objective):
    r = cal.find_best_k(SPEC, "lanoiselee", seed=3, n_sim=400, objective=objective)
    assert cal.K_BOUNDS[0] < r.k < cal.K_BOUNDS[1]


def test_objective_sensitivity_reports_all_objectives():
    df = cal.objective_sensitivity(SPEC, "lanoiselee", seed=3, n_sim=400)
    assert set(df["objective"]) == set(cal.OBJECTIVES)
    assert (df.loc[df["objective"] == "bland_altman", "pct_change_vs_primary"] == 0).all()


def test_asymmetric_objective_shifts_k_in_the_expected_direction():
    """Penalising under-estimation of residual heparin must make the nomogram
    read higher, i.e. must select a SLOWER decay constant."""
    base = cal.find_best_k(SPEC, "lanoiselee", seed=5, n_sim=800, objective="rmse").k
    asym = cal.find_best_k(SPEC, "lanoiselee", seed=5, n_sim=800, objective="asymmetric",
                           objective_kwargs={"penalty": 4.0}).k
    assert asym < base


@pytest.mark.parametrize("mode", cal.EVALUATION_MODES)
def test_evaluation_modes_produce_the_expected_number_of_points(mode):
    cohort = cal.sample_cohort(SPEC, 50, np.random.default_rng(1))
    times = cal.evaluation_times(cohort, mode, n_points=9)
    expected = 1 if mode == "reversal_endpoint" else 9
    assert times.shape == (50, expected)


def test_trajectory_calibration_differs_from_endpoint_calibration():
    """If these agreed there would be nothing to decide for EB-4."""
    endpoint = cal.find_best_k(SPEC, "delavenne", seed=6, n_sim=600,
                               evaluation_mode="reversal_endpoint").k
    traj = cal.find_best_k(SPEC, "delavenne", seed=6, n_sim=600,
                           evaluation_mode="trajectory_full").k
    assert abs(traj - endpoint) / endpoint > 0.01


def test_evaluation_times_span_the_requested_window():
    cohort = cal.sample_cohort(SPEC, 20, np.random.default_rng(2))
    full = cal.evaluation_times(cohort, "trajectory_full", 5)
    assert np.allclose(full[:, 0], 0.0)
    assert np.allclose(full[:, -1], cohort["elapsed_time"].to_numpy())

    cpb = cal.evaluation_times(cohort, "trajectory_cpb", 5)
    assert np.allclose(cpb[:, 0], cohort["time_to_cpb"].to_numpy())
    assert np.allclose(cpb[:, -1], cohort["elapsed_time"].to_numpy())


def test_prime_timing_sensitivity_reports_both_conventions():
    df = cal.prime_timing_sensitivity(SPEC, "lanoiselee", seed=8, n_sim=400)
    assert set(df["prime_timing"]) == set(core.PRIME_TIMING_CHOICES)
    assert df["k"].nunique() == 2


def test_grid_cohort_is_deterministic_and_complete():
    df = cal.grid_cohort([400], [40, 70], [0, 5000], [5, 15], [30, 60])
    assert len(df) == 1 * 2 * 2 * 2 * 2
    assert (df["elapsed_time"] == df["time_to_cpb"] + df["time_on_cpb"]).all()


def test_correlated_cohort_induces_correlation():
    plain = cal.sample_correlated_cohort(SPEC, 4000, np.random.default_rng(1), correlation=0.0)
    corr = cal.sample_correlated_cohort(SPEC, 4000, np.random.default_rng(1), correlation=0.6)
    assert abs(np.corrcoef(plain["ibw"], plain["time_on_cpb"])[0, 1]) < 0.1
    assert np.corrcoef(corr["ibw"], corr["time_on_cpb"])[0, 1] > 0.4


# --------------------------------------------------- parameter uncertainty

def test_parameter_uncertainty_uses_published_numbers_for_every_population_model():
    """EB-2 is now answered from published uncertainty, with no proxy needed."""
    for model in ("lanoiselee", "delavenne", "jia"):
        rep = cal.parameter_uncertainty(SPEC, model, seed=1, n_draws=6, n_sim=150)
        assert rep.available, rep.reason
        assert rep.source == "estimate"
        assert rep.provenance and "not reported" not in rep.provenance.values()


def test_jia_propagation_uses_its_bootstrap_interval():
    rep = cal.parameter_uncertainty(SPEC, "jia", seed=1, n_draws=4, n_sim=150)
    assert rep.provenance["Vp_L"] == "published bootstrap 95% CI"
    assert "bootstrap" in rep.reason


def test_delavenne_propagation_includes_the_weight_exponent():
    """The covariate exponent carries a 29% RSE and must be drawn, not frozen."""
    rep = cal.parameter_uncertainty(SPEC, "delavenne", seed=1, n_draws=8, n_sim=150)
    assert "param_wt_exponent_Cl" in rep.draws
    assert rep.draws["param_wt_exponent_Cl"].nunique() > 1
    # The fixed exponent on Vc has no uncertainty and must not move.
    assert rep.draws["param_wt_exponent_Vc"].nunique() == 1


def test_iiv_proxy_is_still_available_but_no_longer_needed():
    rep = cal.parameter_uncertainty(SPEC, "lanoiselee", seed=1, n_draws=6, n_sim=150,
                                    source="iiv")
    assert rep.available and rep.source == "iiv"
    assert "overstates" in rep.reason


def test_iiv_proxy_is_wider_than_published_estimation_uncertainty():
    """Interindividual variability is spread between patients, not uncertainty in
    the estimate, so substituting it must overstate the interval."""
    est = cal.parameter_uncertainty(SPEC, "lanoiselee", seed=3, n_draws=40, n_sim=300)
    iiv = cal.parameter_uncertainty(SPEC, "lanoiselee", seed=3, n_draws=40, n_sim=300,
                                    source="iiv")
    width = lambda d: cal.summarise_k(d["k"])["k_p97_5"] - cal.summarise_k(d["k"])["k_p2_5"]
    assert width(iiv.draws) > width(est.draws)


def test_closed_form_models_have_no_parameter_uncertainty_to_propagate():
    rep = cal.parameter_uncertainty(SPEC, "prodose", seed=1, n_draws=2, n_sim=100,
                                    allow_iiv_proxy=True)
    assert not rep.available
    assert "closed-form" in rep.reason


def test_parameter_uncertainty_is_wider_than_sampling_precision():
    mc = cal.monte_carlo_precision(SPEC, "lanoiselee", seed=21, n_replicates=40, n_sim=400)
    pu = cal.parameter_uncertainty(SPEC, "lanoiselee", seed=21, n_draws=40, n_sim=400,
                                   allow_iiv_proxy=True)
    mc_width = cal.summarise_k(mc["k"])["k_p97_5"] - cal.summarise_k(mc["k"])["k_p2_5"]
    pu_width = cal.summarise_k(pu.draws["k"])["k_p97_5"] - cal.summarise_k(pu.draws["k"])["k_p2_5"]
    assert pu_width > mc_width, (
        "Propagating parameter uncertainty must widen the interval; if it does "
        "not, the two intervals are measuring the same thing and the relabelling "
        "in EB-2 is not enough."
    )
