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


def test_jia_has_no_weight_covariate():
    """The Jia model carries no weight covariate, so its kinetics are identical
    at every body weight.

    In the submitted code this was hidden behind the manuscript's claim that Jia
    was a paediatric model, which made it look like a defect. It is not: Jia is
    an adult model and simply has no weight term. Within the pipeline IBW
    therefore scales the administered bolus but never the elimination, which is
    a stated limitation rather than a bug.
    """
    for w in (45.0, 70.0, 115.0):
        assert core.jia_response(1000.0, 60.0, w) == pytest.approx(
            core.jia_response(1000.0, 60.0, 70.0), rel=1e-12)


def test_no_model_is_marked_paediatric():
    """All six reference models are adult, Jia included (EB-4, R1 p11 L46)."""
    assert core.PAEDIATRIC_MODELS == ()


def test_jia_parameters_are_adult_scale():
    """The evidence that settled the paediatric question: Jia's central volume is
    an adult one, within a few percent of Delavenne's."""
    jia_vc = core.MODEL_PARAMETERS["jia"].values["Vc_L"]
    delavenne_vc = core.MODEL_PARAMETERS["delavenne"].values["Vc_L"]
    assert abs(jia_vc - delavenne_vc) / delavenne_vc < 0.05
    # A 10 kg child's central volume would be roughly an order of magnitude less.
    assert jia_vc > 5 * (10 * 40 / 1000)


def test_delavenne_weight_exponents_are_parameters_not_literals():
    """The clearance exponent carries a 29% RSE, so it has to be overridable for
    that uncertainty to propagate (EB-2)."""
    vals = core.MODEL_PARAMETERS["delavenne"].values
    assert vals["wt_exponent_Vc"] == 1.0
    assert vals["wt_exponent_Cl"] == 0.767
    assert core.MODEL_PARAMETERS["delavenne"].rse["wt_exponent_Cl"] == 0.29

    base = core.get_reference_dose("delavenne", 28000, 5000, 115, 15, 60)
    shifted = core.reference_amount("delavenne", 75, 28000, 5000, 115, 15,
                                    overrides={**vals, "wt_exponent_Cl": 1.0})
    assert shifted != pytest.approx(base, rel=1e-6)


def test_covariate_uncertainty_vanishes_at_the_reference_weight():
    """The covariate is centred on 70 kg, so changing its exponent can have no
    effect there and grows towards the ends of the weight grid. That is why it
    moves the boundary results and not the headline interval."""
    vals = core.MODEL_PARAMETERS["delavenne"].values
    alt = {**vals, "wt_exponent_Cl": 1.2}

    at_70 = core.reference_amount("delavenne", 75, 28000, 5000, 70, 15)
    at_70_alt = core.reference_amount("delavenne", 75, 28000, 5000, 70, 15, overrides=alt)
    assert at_70 == pytest.approx(at_70_alt, rel=1e-12)

    for w in (40.0, 115.0):
        a = core.reference_amount("delavenne", 75, 28000, 5000, w, 15)
        b = core.reference_amount("delavenne", 75, 28000, 5000, w, 15, overrides=alt)
        assert abs(a - b) / a > 0.01


def test_published_uncertainty_is_recorded_for_every_population_model():
    """EB-2 can only be answered from published numbers, so every structural
    parameter of the three population-PK models must carry one."""
    for key in ("lanoiselee", "delavenne", "jia"):
        pset = core.MODEL_PARAMETERS[key]
        assert pset.missing_rse() == (), (
            f"{key} still lacks published uncertainty for {pset.missing_rse()}")
        for name in pset.values:
            assert pset.uncertainty_basis(name) != "not reported"


def test_jia_prefers_its_published_bootstrap_interval():
    """Jia publishes bootstrap CIs; for the poorly identified parameters they are
    materially wider than the asymptotic RSE, so the CI must win."""
    pset = core.MODEL_PARAMETERS["jia"]
    assert pset.uncertainty_basis("Vp_L") == "published bootstrap 95% CI"
    assert pset.uncertainty_sigma("Vp_L") > 1.5 * pset.rse["Vp_L"]
    # Well-identified parameters: the two bases agree closely.
    assert pset.uncertainty_sigma("Vc_L") == pytest.approx(pset.rse["Vc_L"], rel=0.1)


def test_jia_iiv_is_the_square_root_of_the_published_omega_squared():
    """The source tabulates omega-squared; MODEL_PARAMETERS stores omega.

    The submitted code instead held the population-mean %RSE column, with Vp and
    Q transposed, and then square-rooted it -- so Vp received 38% variability
    that the source fixes at zero.
    """
    published_omega_sq = {"Cl_L_h": 0.122, "Vc_L": 0.105, "Q_L_h": 0.0978, "Vp_L": 0.0}
    iiv = core.MODEL_PARAMETERS["jia"].iiv
    for name, w2 in published_omega_sq.items():
        assert iiv[name] == pytest.approx(math.sqrt(w2), abs=1e-4)
    assert iiv["Vp_L"] == 0.0, "the source fixes the peripheral volume's IIV at zero"

    superseded = {"Cl_L_h": 0.073, "Vc_L": 0.081, "Vp_L": 0.144, "Q_L_h": 0.318}
    for name, bad in superseded.items():
        assert iiv[name] != pytest.approx(bad, abs=1e-6)


def test_lanoiselee_iiv_is_not_the_rse_of_the_iiv():
    """The submitted values were that column's parenthetical %RSE (9.83, 11.1,
    39.5, 21.0) divided by 100, which understated the variability about 2.1x."""
    iiv = core.MODEL_PARAMETERS["lanoiselee"].iiv
    assert iiv == {"Cl": 0.27, "Vc": 0.22, "Vp": 0.74, "Q": 0.41}
    for name, superseded in (("Cl", 0.0983), ("Vc", 0.111), ("Vp", 0.395), ("Q", 0.21)):
        assert iiv[name] != pytest.approx(superseded, abs=1e-6)


def test_delavenne_iiv_was_already_correct():
    """Delavenne is the control: its transcription was right, which is what
    established that the other two were wrong rather than following a different
    convention."""
    iiv = core.MODEL_PARAMETERS["delavenne"].iiv
    assert iiv["Cl_L_h"] == 0.221 and iiv["Vc_L"] == 0.119
    # The source shows "--" for Vp and Q: no random effect.
    assert iiv["Vp_L"] == 0.0 and iiv["Q_L_h"] == 0.0
    # But both do carry an RSE, so they still enter the EB-2 propagation.
    assert core.MODEL_PARAMETERS["delavenne"].rse["Vp_L"] > 0


def test_parameter_provenance_is_recorded():
    for key in core.MODEL_NAMES:
        pset = core.MODEL_PARAMETERS[key]
        assert pset.source, f"{key} has no recorded source"
        assert pset.values, f"{key} has no recorded parameter values"
