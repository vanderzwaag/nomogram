"""Unit tests for the pharmacokinetic core (EB-6: 'unit tests would materially
strengthen confidence')."""

import itertools
import math

import numpy as np
import pytest

import nomogram_core as core

ALL_MODELS = list(core.MODEL_NAMES)

# A spread of inputs covering the whole adult calibration grid plus its corners.
CASES = list(itertools.product(
    [250, 400, 600],        # IU/kg
    [40, 70, 115],          # kg
    [5, 15, 35],            # time to CPB
    [30, 60, 120],          # time on CPB
    [0, 5000, 10000],       # prime
))


@pytest.mark.parametrize("model", ALL_MODELS)
def test_endpoint_matches_trajectory_at_reversal(model):
    """The single-endpoint and trajectory implementations must not drift apart.

    This is the test that makes trajectory-based calibration (EB-4) legitimate:
    without it, the two code paths could silently diverge and the endpoint and
    trajectory results would not be comparable.
    """
    for dose_kg, ibw, t_to, t_on, prime in CASES:
        bolus = dose_kg * ibw
        endpoint = core.get_reference_dose(model, bolus, prime, ibw, t_to, t_on)
        trajectory = core.reference_amount(model, t_to + t_on, bolus, prime, ibw, t_to)
        assert endpoint == pytest.approx(trajectory, rel=1e-12)


@pytest.mark.parametrize("model", ALL_MODELS)
def test_vectorised_matches_scalar(model):
    times = np.array([0.0, 1.0, 5.0, 14.9, 15.0, 15.1, 45.0, 75.0, 120.0, 240.0])
    for dose_kg, ibw, t_to, t_on, prime in CASES[::7]:
        bolus = dose_kg * ibw
        vec = core.reference_amount_array(model, times, bolus, prime, ibw, t_to)
        sca = np.array([core.reference_amount(model, t, bolus, prime, ibw, t_to) for t in times])
        assert np.allclose(vec, sca, rtol=1e-12, atol=1e-8)


@pytest.mark.parametrize("model", ALL_MODELS)
def test_monotone_decay_after_prime(model):
    """Once the prime is in, the central amount must fall monotonically."""
    bolus, prime, ibw, t_to = 28000.0, 5000.0, 70.0, 15.0
    times = np.linspace(t_to, 240.0, 200)
    vals = core.reference_amount_array(model, times, bolus, prime, ibw, t_to)
    assert np.all(np.diff(vals) < 1e-9)


@pytest.mark.parametrize("model", core.DOSE_LINEAR_MODELS)
def test_linear_in_dose(model):
    """The population-PK models and Meesters are linear in dose, which is what
    makes the superposition used for supplemental boluses (EB-5) exact."""
    ibw, t_to, t_on, prime = 70.0, 15.0, 60.0, 0.0
    a = core.get_reference_dose(model, 10000.0, prime, ibw, t_to, t_on)
    b = core.get_reference_dose(model, 30000.0, prime, ibw, t_to, t_on)
    assert b == pytest.approx(3.0 * a, rel=1e-12)


@pytest.mark.parametrize("model", core.DOSE_DEPENDENT_MODELS)
def test_dose_dependent_models_are_not_linear(model):
    """PRODOSE and PRODOSE-2 make the slow-pool half-life a function of IU/kg,
    so they are NOT linear in dose and plain superposition of a supplemental
    bolus is invalid for them.

    The submitted top-up analysis superposed a 5,000 IU top-up by calling the
    PRODOSE trajectory with 5,000 IU as the bolus, which gives that top-up the
    half-life of a 71 IU/kg induction dose instead of the patient's own.
    """
    ibw, t_to, t_on, prime = 70.0, 15.0, 60.0, 0.0
    a = core.get_reference_dose(model, 10000.0, prime, ibw, t_to, t_on)
    b = core.get_reference_dose(model, 30000.0, prime, ibw, t_to, t_on)
    assert b != pytest.approx(3.0 * a, rel=1e-3)


@pytest.mark.parametrize("model", ALL_MODELS)
def test_linear_in_dose_once_covariate_is_fixed(model):
    """Holding the dose-dependent covariate at the index bolus restores
    linearity for every model, which is what makes superposition of a
    supplemental bolus well defined."""
    ibw, t_to, t_on, prime = 70.0, 15.0, 60.0, 0.0
    index = 28000.0
    a = core.reference_amount(model, t_to + t_on, 10000.0, prime, ibw, t_to,
                              covariate_bolus=index)
    b = core.reference_amount(model, t_to + t_on, 30000.0, prime, ibw, t_to,
                              covariate_bolus=index)
    assert b == pytest.approx(3.0 * a, rel=1e-12)


@pytest.mark.parametrize("model", ALL_MODELS)
def test_prime_enters_at_cpb_onset(model):
    """Documented prime-timing convention (EB-5): the prime appears at t_to."""
    bolus, prime, ibw, t_to = 28000.0, 5000.0, 70.0, 15.0
    before = core.reference_amount(model, t_to - 1e-9, bolus, prime, ibw, t_to)
    after = core.reference_amount(model, t_to, bolus, prime, ibw, t_to)
    assert after - before == pytest.approx(prime, rel=1e-6)


@pytest.mark.parametrize("model", ALL_MODELS)
def test_prime_absent_before_cpb(model):
    """Before CPB onset the result must not depend on the prime dose at all."""
    bolus, ibw, t_to = 28000.0, 70.0, 15.0
    a = core.reference_amount(model, 10.0, bolus, 0.0, ibw, t_to)
    b = core.reference_amount(model, 10.0, bolus, 10000.0, ibw, t_to)
    assert a == pytest.approx(b, rel=1e-12)


@pytest.mark.parametrize("model", ALL_MODELS)
def test_topup_uses_the_patients_own_elimination_constant(model):
    """A top-up must decay at the patient's half-life, not at the half-life its
    own size would imply."""
    bolus, prime, ibw, t_to = 28000.0, 5000.0, 70.0, 15.0
    t, t_topup, topup = 150.0, 60.0, 5000.0
    combined = core.reference_amount_with_topups(
        model, t, bolus, prime, ibw, t_to, ((t_topup, topup),))
    separate = (core.reference_amount(model, t, bolus, prime, ibw, t_to)
                + core.reference_amount(model, t - t_topup, topup, 0.0, ibw, 0.0,
                                        covariate_bolus=bolus))
    assert combined == pytest.approx(separate, rel=1e-12)


@pytest.mark.parametrize("model", core.DOSE_DEPENDENT_MODELS)
def test_naive_topup_superposition_differs_materially(model):
    """Quantifies the defect above: for the dose-dependent models, superposing a
    top-up without fixing the covariate changes the answer by a clinically
    relevant amount."""
    bolus, prime, ibw, t_to = 28000.0, 5000.0, 70.0, 15.0
    t, t_topup, topup = 150.0, 60.0, 5000.0
    correct = core.reference_amount_with_topups(
        model, t, bolus, prime, ibw, t_to, ((t_topup, topup),))
    naive = (core.reference_amount(model, t, bolus, prime, ibw, t_to)
             + core.reference_amount(model, t - t_topup, topup, 0.0, ibw, 0.0))
    assert abs(correct - naive) > 100.0


# ---------------------------------------------------------------- simplified

def test_simplified_lumped_prime_is_the_submitted_behaviour():
    bolus, prime, k, t_to, t_on = 28000.0, 5000.0, 0.007, 15.0, 60.0
    expected = (bolus + prime) * math.exp(-k * (t_to + t_on))
    assert core.simplified_model_dose(bolus, prime, 70.0, t_to, t_on, k) == pytest.approx(expected)


def test_simplified_cpb_onset_prime_differs_from_lumped():
    """The two prime-timing conventions must give different answers, otherwise
    the EB-5 sensitivity analysis would be vacuous."""
    bolus, prime, k, t_to, t_on = 28000.0, 5000.0, 0.007, 15.0, 60.0
    lumped = core.simplified_model_dose(bolus, prime, 70.0, t_to, t_on, k, "lumped_t0")
    onset = core.simplified_model_dose(bolus, prime, 70.0, t_to, t_on, k, "cpb_onset")
    assert onset > lumped
    assert abs(onset - lumped) > 100.0


def test_axis_reset_equals_superposition_in_simplified_model():
    """The bedside 'reset the time axis' rule is exact within the simplified
    model, so any divergence measured in the top-up analysis belongs to the
    reference model comparison, not to the reset rule."""
    bolus, prime, k, t_to = 28000.0, 5000.0, 0.007, 15.0
    t_topup, topup, t = 60.0, 5000.0, 180.0

    residual_at_topup = core.simplified_amount(t_topup, bolus, prime, k, t_to)
    reset = (residual_at_topup + topup) * math.exp(-k * (t - t_topup))
    superposed = core.simplified_amount_with_topups(
        t, bolus, prime, k, t_to, ((t_topup, topup),))
    assert reset == pytest.approx(superposed, rel=1e-12)


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        core.get_reference_dose("not_a_model", 28000, 5000, 70, 15, 60)


def test_unknown_prime_timing_rejected():
    with pytest.raises(ValueError):
        core.simplified_model_dose(28000, 5000, 70, 15, 60, 0.007, "whenever")


def test_jia_has_no_weight_covariate_by_default():
    """Documents a real limitation flagged for EB-4: the Jia implementation
    carries no weight covariate, so a 5 kg neonate and a 70 kg adult are given
    identical kinetics. A paediatric parameter space is only meaningful once
    this is either accepted explicitly or replaced by allometric scaling."""
    a = core.jia_response(1000.0, 60.0, 5.0)
    b = core.jia_response(1000.0, 60.0, 70.0)
    assert a == pytest.approx(b, rel=1e-12)

    scaled_small = core.jia_response(1000.0, 60.0, 5.0, allometric=True)
    scaled_large = core.jia_response(1000.0, 60.0, 70.0, allometric=True)
    assert scaled_small != pytest.approx(scaled_large, rel=1e-6)


def test_parameter_provenance_is_recorded():
    for key in core.MODEL_NAMES:
        pset = core.MODEL_PARAMETERS[key]
        assert pset.source, f"{key} has no recorded source"
        assert pset.values, f"{key} has no recorded parameter values"
