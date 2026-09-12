"""
Pharmacokinetic core for the heparin decay nomogram pipeline.

This module is deliberately free of Streamlit, PyNomo, ReportLab and matplotlib
imports so that every quantity reported in the manuscript can be regenerated
head-lessly, from a single command, inside a unit test or in CI.

Two forms are provided for each reference model:

  ``<model>_amount(t, ...)``  -- amount of heparin in the central compartment at
                                 absolute time ``t`` minutes after the induction
                                 bolus (the decay *trajectory*).
  ``<model>_dose(...)``       -- amount remaining at the reversal timepoint,
                                 i.e. at ``t = time_to_cpb + time_on_cpb``
                                 (the single *endpoint*).

``test_core.py`` asserts that ``<model>_amount(t_to + t_on) == <model>_dose()``
for every model, so the trajectory and endpoint implementations cannot drift
apart.  That equality is what makes trajectory-based calibration (EB-4) a
supported option rather than an assumption.

Parameter provenance is recorded in ``MODEL_PARAMETERS`` below; anything marked
``UNVERIFIED`` must be checked against the source publication before the
manuscript is resubmitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

CORE_VERSION = "2.0.0"

# All six reference models, in the order used throughout the manuscript.
MODEL_NAMES = ("delavenne", "jia", "lanoiselee", "meesters", "prodose", "prodose-2")

# Display names, so the app and the analysis runner cannot disagree on spelling.
DISPLAY_NAMES = {
    "delavenne": "Delavenne",
    "jia": "Jia",
    "lanoiselee": "Lanoiselee",
    "meesters": "Meesters",
    "prodose": "PRODOSE",
    "prodose-2": "PRODOSE-2",
}

# Models that are paediatric in origin and must not be pooled with the adult
# models or run in an adult parameter space (EB-4, R1 p11 L46).
PAEDIATRIC_MODELS = ("jia",)


# ==========================================================================
# PARAMETER PROVENANCE
# ==========================================================================

@dataclass(frozen=True)
class ParameterSet:
    """Published population parameters and their reported uncertainty.

    ``iiv`` is between-subject (interindividual) variability, expressed as the
    SD of the log-normal random effect.  ``rse`` is the relative standard error
    of the *population estimate* -- the quantity EB-2 asks us to propagate.
    ``None`` means the source publication did not report it (or it has not yet
    been transcribed), and the analysis runner will say so rather than silently
    substituting a number.
    """

    name: str
    source: str
    values: Dict[str, float]
    iiv: Dict[str, float] = field(default_factory=dict)
    rse: Dict[str, float] = field(default_factory=dict)
    weight_covariate: bool = False
    notes: str = ""

    def missing_rse(self) -> Sequence[str]:
        return tuple(k for k in self.values if self.rse.get(k) is None)


MODEL_PARAMETERS: Dict[str, ParameterSet] = {
    "lanoiselee": ParameterSet(
        name="Lanoiselee",
        source="Lanoiselee et al. Br J Anaesth 2026;136(3):847-855 (doi 10.1016/j.bja.2025.11.057)",
        # mL and mL/min, as used by the original R script.
        values={"Cl": 1500.18 / 60.0, "Vc": 4011.12, "Vp": 1458.59, "Q": 287.57 / 60.0},
        iiv={"Cl": 0.0983, "Vc": 0.111, "Vp": 0.395, "Q": 0.21},
        rse={"Cl": None, "Vc": None, "Vp": None, "Q": None},
        weight_covariate=False,
        notes=(
            "Fixed population parameters; the published model carries no weight "
            "covariate, so IBW affects the bolus size only, not the kinetics. "
            "RSE must be transcribed from the source publication before the "
            "parameter-uncertainty interval can be reported (EB-2). "
            "NOTE: the submitted code held two spellings of Q -- 4.7928 mL/min in "
            "the endpoint model and 287.57/60 = 4.79283 mL/min in the credible-"
            "interval simulation. The unrounded value is used throughout here; "
            "the difference is about 0.02 IU at the reversal timepoint."
        ),
    ),
    "delavenne": ParameterSet(
        name="Delavenne",
        source="Delavenne et al.",
        # Litres and L/h as published; converted to mL/min at point of use.
        values={"Cl_L_h": 0.841, "Vc_L": 3.1, "Vp_L": 2.23, "Q_L_h": 4.67},
        iiv={"Cl_L_h": 0.221, "Vc_L": 0.119, "Vp_L": 0.0, "Q_L_h": 0.0},
        rse={"Cl_L_h": None, "Vc_L": None, "Vp_L": None, "Q_L_h": None},
        weight_covariate=True,
        notes=(
            "Vc scales with weight (exponent 1.0) and Cl with weight (exponent "
            "0.767). The published model is driven by ACTUAL body weight; the "
            "pipeline passes IBW, which is a documented approximation (EB-1). "
            "Vp and Q have no reported IIV and are held fixed."
        ),
    ),
    "jia": ParameterSet(
        name="Jia",
        source="Jia et al. (paediatric cardiac surgery)",
        values={"Cl_L_h": 1.18, "Vc_L": 3.04, "Vp_L": 8.01, "Q_L_h": 0.171},
        # UNVERIFIED: the dashboard docstring and the dashboard code disagreed.
        # The docstring said CL 0.176 / Vc 0.114 / Q 0.0573 / Vp 0.111 while the
        # code used [0.073, 0.081, 0.144, 0.318] and additionally took a square
        # root of them. Both cannot be right. The values below are the
        # docstring values, treated as omega SDs; see notes.
        iiv={"Cl_L_h": 0.176, "Vc_L": 0.114, "Vp_L": 0.111, "Q_L_h": 0.0573},
        rse={"Cl_L_h": None, "Vc_L": None, "Vp_L": None, "Q_L_h": None},
        weight_covariate=False,
        notes=(
            "UNVERIFIED IIV -- resolve against the source publication before "
            "resubmission. The implementation carries NO weight covariate, so a "
            "5 kg neonate and a 70 kg adult are given identical kinetics; a "
            "paediatric parameter space (EB-4) is only meaningful once this is "
            "settled. See allometric_jia_params()."
        ),
    ),
    "prodose": ParameterSet(
        name="PRODOSE",
        source="PRODOSE nomogram (published closed-form expression)",
        values={"fast_fraction": 0.1, "k_fast": 0.0693, "t_half_intercept": 26.0, "t_half_slope": 0.323},
        weight_covariate=True,
        notes="Closed-form biexponential; the half-life of the slow pool depends on IU/kg.",
    ),
    "prodose-2": ParameterSet(
        name="PRODOSE-2",
        source="PRODOSE-2 (revised closed-form expression with circuit dilution)",
        values={
            "fast_fraction": 0.1,
            "k_fast": 0.2847,
            "t_half_intercept": 52.44,
            "t_half_slope": 0.2968,
            "v_prime_mL": 1500.0,
            "reinfusion": 1.0,
        },
        weight_covariate=True,
        notes="Slow-pool elimination is scaled by the circuit dilution factor EBV/(EBV+prime volume).",
    ),
    "meesters": ParameterSet(
        name="Meesters",
        source="Meesters et al.",
        values={"fast_fraction": 0.1, "k_fast": 0.0693, "t_half_slow": 250.0},
        weight_covariate=False,
        notes="Fixed 250 min slow half-life; no weight or dose covariate.",
    ),
}


def canonical_model_name(model_name: str) -> str:
    """Map any accepted spelling of a model onto its canonical key."""
    key = model_name.strip().lower()
    if key not in MODEL_PARAMETERS:
        raise ValueError(f"Unknown model: {model_name!r}. Known models: {MODEL_NAMES}")
    return key


# ==========================================================================
# TWO-COMPARTMENT ANALYTICAL SOLUTION
# ==========================================================================

def _biexponential_constants(Cl: float, Vc: float, Vp: float, Q: float):
    """Return (alpha, beta, k21, Vc) for a linear two-compartment model.

    Units must be consistent (mL and mL/min throughout this module).
    """
    k10 = Cl / Vc
    k12 = Q / Vc
    k21 = Q / Vp

    sum_k = k10 + k12 + k21
    root = math.sqrt(sum_k ** 2 - 4.0 * k10 * k21)
    alpha = (sum_k + root) / 2.0
    beta = (sum_k - root) / 2.0
    return alpha, beta, k21, Vc


def _biexponential_amount(dose: float, t: float, alpha: float, beta: float, k21: float) -> float:
    """Central-compartment amount at time ``t`` after a bolus ``dose``."""
    if t < 0:
        return 0.0
    a_coeff = (alpha - k21) / (alpha - beta)
    b_coeff = (k21 - beta) / (alpha - beta)
    return dose * (a_coeff * math.exp(-alpha * t) + b_coeff * math.exp(-beta * t))


# ---------------------------------------------------------------- Lanoiselee

def get_lanoiselee_params(overrides: Dict[str, float] | None = None):
    p = dict(MODEL_PARAMETERS["lanoiselee"].values)
    if overrides:
        p.update(overrides)
    return _biexponential_constants(p["Cl"], p["Vc"], p["Vp"], p["Q"])


def lanoiselee_response(dose, t, overrides=None):
    alpha, beta, k21, _ = get_lanoiselee_params(overrides)
    return _biexponential_amount(dose, t, alpha, beta, k21)


# ---------------------------------------------------------------- Delavenne

def get_delavenne_params(weight_kg: float, overrides: Dict[str, float] | None = None):
    p = dict(MODEL_PARAMETERS["delavenne"].values)
    if overrides:
        p.update(overrides)

    Vc = p["Vc_L"] * (weight_kg / 70.0) ** 1.0 * 1000.0
    Cl = (p["Cl_L_h"] * (weight_kg / 70.0) ** 0.767) * 1000.0 / 60.0
    Vp = p["Vp_L"] * 1000.0
    Q = p["Q_L_h"] * 1000.0 / 60.0
    return _biexponential_constants(Cl, Vc, Vp, Q)


def delavenne_response(dose, t, weight_kg, overrides=None):
    alpha, beta, k21, _ = get_delavenne_params(weight_kg, overrides)
    return _biexponential_amount(dose, t, alpha, beta, k21)


# ---------------------------------------------------------------------- Jia

# Conventional allometric exponents. These are NOT taken from the Jia source
# publication; they are the standard values used when a paediatric model has to
# be extrapolated across a wide weight range. Enabling them is an explicit
# author decision (EB-4) and is recorded in the run manifest.
JIA_ALLOMETRY = {"reference_weight_kg": 70.0, "volume_exponent": 1.0, "clearance_exponent": 0.75}


def get_jia_params(weight_kg: float, overrides=None, allometric: bool = False):
    p = dict(MODEL_PARAMETERS["jia"].values)
    if overrides:
        p.update(overrides)

    Vc_L, Cl_L_h, Vp_L, Q_L_h = p["Vc_L"], p["Cl_L_h"], p["Vp_L"], p["Q_L_h"]

    if allometric:
        ratio = weight_kg / JIA_ALLOMETRY["reference_weight_kg"]
        Vc_L *= ratio ** JIA_ALLOMETRY["volume_exponent"]
        Vp_L *= ratio ** JIA_ALLOMETRY["volume_exponent"]
        Cl_L_h *= ratio ** JIA_ALLOMETRY["clearance_exponent"]
        Q_L_h *= ratio ** JIA_ALLOMETRY["clearance_exponent"]

    return _biexponential_constants(
        Cl_L_h * 1000.0 / 60.0, Vc_L * 1000.0, Vp_L * 1000.0, Q_L_h * 1000.0 / 60.0
    )


def jia_response(dose, t, weight_kg, overrides=None, allometric: bool = False):
    alpha, beta, k21, _ = get_jia_params(weight_kg, overrides, allometric)
    return _biexponential_amount(dose, t, alpha, beta, k21)


# ==========================================================================
# REFERENCE MODEL TRAJECTORIES
# ==========================================================================
#
# Prime-heparin timing convention, stated once here and used everywhere
# (EB-5): the systemic bolus enters at t = 0; the pump prime enters the
# circulation at CPB onset, t = time_to_cpb, and therefore decays only for the
# duration of bypass. The simplified nomogram model can be run under the same
# convention or under the lumped convention -- see simplified_amount().

def reference_amount(
    model_name: str,
    t: float,
    heparin_bolus: float,
    heparin_prime: float,
    weight_kg: float,
    time_to_cpb: float,
    *,
    overrides: Dict[str, float] | None = None,
    jia_allometric: bool = False,
    covariate_bolus: float | None = None,
) -> float:
    """Central-compartment heparin amount (IU) at absolute time ``t`` (min).

    ``covariate_bolus``
        PRODOSE and PRODOSE-2 make the slow-pool half-life a function of the
        administered dose per kilogram, so unlike the three population-PK
        models and Meesters they are NOT linear in dose and superposition of a
        supplemental bolus is not exact. Passing ``covariate_bolus`` fixes the
        dose used for that covariate -- normally the index bolus, which is a
        property of the patient -- so that a 5,000 IU top-up is not given the
        elimination half-life of a 5,000 IU induction dose. See
        ``reference_amount_with_topups``.
    """
    key = canonical_model_name(model_name)
    if t < 0:
        return 0.0

    covariate_dose = heparin_bolus if covariate_bolus is None else covariate_bolus
    prime_elapsed = t - time_to_cpb
    prime_active = prime_elapsed >= 0

    if key == "prodose":
        p = dict(MODEL_PARAMETERS["prodose"].values)
        if overrides:
            p.update(overrides)
        k2 = 0.693 / (p["t_half_intercept"] + p["t_half_slope"] * (covariate_dose / weight_kg))
        amount = (
            heparin_bolus * p["fast_fraction"] * math.exp(-p["k_fast"] * t)
            + heparin_bolus * (1.0 - p["fast_fraction"]) * math.exp(-k2 * t)
        )
        if prime_active:
            amount += heparin_prime * math.exp(-k2 * prime_elapsed)
        return amount

    if key == "meesters":
        p = dict(MODEL_PARAMETERS["meesters"].values)
        if overrides:
            p.update(overrides)
        k2 = 0.693 / p["t_half_slow"]
        amount = (
            heparin_bolus * p["fast_fraction"] * math.exp(-p["k_fast"] * t)
            + heparin_bolus * (1.0 - p["fast_fraction"]) * math.exp(-k2 * t)
        )
        if prime_active:
            amount += heparin_prime * math.exp(-k2 * prime_elapsed)
        return amount

    if key == "prodose-2":
        p = dict(MODEL_PARAMETERS["prodose-2"].values)
        if overrides:
            p.update(overrides)
        k2_base = 0.693 / (p["t_half_intercept"] + p["t_half_slope"] * (covariate_dose / weight_kg))
        ebv = weight_kg * 70.0
        v_factor = ebv / (ebv + p["v_prime_mL"])
        k2_cpb = k2_base * v_factor

        pool_fast = heparin_bolus * p["fast_fraction"] * math.exp(-p["k_fast"] * t)
        if not prime_active:
            pool_slow = heparin_bolus * (1.0 - p["fast_fraction"]) * math.exp(-k2_cpb * t)
        else:
            slow_at_cpb = heparin_bolus * (1.0 - p["fast_fraction"]) * math.exp(-k2_cpb * time_to_cpb)
            pool_slow = (slow_at_cpb + heparin_prime) * math.exp(-k2_cpb * prime_elapsed)
        # With reinfusion == 1.0 the patient and circuit fractions recombine, so
        # the returned amount is the whole slow pool; the split is retained for
        # clarity and for future partial-reinfusion scenarios.
        reinfusion = p["reinfusion"]
        patient_mass = pool_slow * v_factor
        circuit_mass = pool_slow * (1.0 - v_factor) * reinfusion
        return pool_fast + patient_mass + circuit_mass

    if key == "lanoiselee":
        amount = lanoiselee_response(heparin_bolus, t, overrides)
        if prime_active:
            amount += lanoiselee_response(heparin_prime, prime_elapsed, overrides)
        return amount

    if key == "delavenne":
        amount = delavenne_response(heparin_bolus, t, weight_kg, overrides)
        if prime_active:
            amount += delavenne_response(heparin_prime, prime_elapsed, weight_kg, overrides)
        return amount

    if key == "jia":
        amount = jia_response(heparin_bolus, t, weight_kg, overrides, jia_allometric)
        if prime_active:
            amount += jia_response(heparin_prime, prime_elapsed, weight_kg, overrides, jia_allometric)
        return amount

    raise ValueError(f"Unhandled model: {model_name!r}")


def get_reference_dose(
    model_name: str,
    heparin_bolus: float,
    heparin_prime: float,
    weight_kg: float,
    time_to_cpb: float,
    time_on_cpb: float,
    *,
    overrides: Dict[str, float] | None = None,
    jia_allometric: bool = False,
) -> float:
    """Reference-model residual heparin at the reversal timepoint."""
    return reference_amount(
        model_name,
        time_to_cpb + time_on_cpb,
        heparin_bolus,
        heparin_prime,
        weight_kg,
        time_to_cpb,
        overrides=overrides,
        jia_allometric=jia_allometric,
    )


def reference_amount_with_topups(
    model_name: str,
    t: float,
    heparin_bolus: float,
    heparin_prime: float,
    weight_kg: float,
    time_to_cpb: float,
    topups: Sequence[tuple] = (),
    *,
    overrides=None,
    jia_allometric: bool = False,
) -> float:
    """Reference amount at ``t`` including supplemental boluses by superposition.

    ``topups`` is a sequence of ``(time_min, dose_IU)`` pairs. Superposition is
    exact for the linear two-compartment models and for the closed-form
    biexponential expressions, all of which are linear in dose.
    """
    amount = reference_amount(
        model_name, t, heparin_bolus, heparin_prime, weight_kg, time_to_cpb,
        overrides=overrides, jia_allometric=jia_allometric,
    )
    for t_bolus, dose in topups:
        if t >= t_bolus and dose:
            # A supplemental bolus is a systemic bolus with no prime attached.
            # The dose-dependent models keep the index bolus as the covariate,
            # so the top-up is eliminated at the patient's own half-life rather
            # than at the half-life a small induction dose would have had.
            amount += reference_amount(
                model_name, t - t_bolus, dose, 0.0, weight_kg, 0.0,
                overrides=overrides, jia_allometric=jia_allometric,
                covariate_bolus=heparin_bolus,
            )
    return amount


# Superposition of a supplemental bolus is exact only for the models that are
# linear in dose. For PRODOSE and PRODOSE-2 it is exact only once the
# dose-dependent covariate is held fixed at the index bolus, which is an
# explicit modelling assumption and is reported as such (EB-5).
DOSE_LINEAR_MODELS = ("lanoiselee", "delavenne", "jia", "meesters")
DOSE_DEPENDENT_MODELS = ("prodose", "prodose-2")


# ==========================================================================
# SIMPLIFIED (NOMOGRAM) MODEL
# ==========================================================================

PRIME_TIMING_CHOICES = ("lumped_t0", "cpb_onset")


def simplified_amount(
    t: float,
    heparin_bolus: float,
    heparin_prime: float,
    k: float,
    time_to_cpb: float = 0.0,
    prime_timing: str = "lumped_t0",
) -> float:
    """Mono-exponential nomogram estimate of residual heparin at time ``t``.

    ``prime_timing``
        ``"lumped_t0"``   -- bolus and prime are treated as one load decaying
                             from t = 0. This is what the printed nomogram
                             does, and what the submitted analysis used.
        ``"cpb_onset"``   -- the prime enters at CPB onset, matching the
                             reference models. Running both quantifies how much
                             of the prime-timing mismatch the calibrated k is
                             absorbing (EB-5).
    """
    if t < 0:
        return 0.0
    if prime_timing == "lumped_t0":
        return (heparin_bolus + heparin_prime) * math.exp(-k * t)
    if prime_timing == "cpb_onset":
        amount = heparin_bolus * math.exp(-k * t)
        if t >= time_to_cpb:
            amount += heparin_prime * math.exp(-k * (t - time_to_cpb))
        return amount
    raise ValueError(f"prime_timing must be one of {PRIME_TIMING_CHOICES}, got {prime_timing!r}")


def simplified_model_dose(
    heparin_bolus: float,
    heparin_prime: float,
    ibw: float,
    time_to_cpb: float,
    time_on_cpb: float,
    k: float,
    prime_timing: str = "lumped_t0",
) -> float:
    """Nomogram estimate at the reversal timepoint.

    ``ibw`` is accepted for signature compatibility with the reference-model
    functions; the simplified model has no weight term by construction.
    """
    return simplified_amount(
        time_to_cpb + time_on_cpb, heparin_bolus, heparin_prime, k,
        time_to_cpb=time_to_cpb, prime_timing=prime_timing,
    )


def simplified_amount_with_topups(
    t: float,
    heparin_bolus: float,
    heparin_prime: float,
    k: float,
    time_to_cpb: float = 0.0,
    topups: Sequence[tuple] = (),
    prime_timing: str = "lumped_t0",
) -> float:
    """Nomogram estimate under the bedside 'axis reset' rule.

    At each supplemental bolus the user reads off the current residual, adds the
    new bolus to it, and restarts the time axis at zero. Because the simplified
    model is a single exponential, that rule is algebraically identical to
    superposing each dose with its own elapsed time -- which is what is
    computed here.
    """
    amount = simplified_amount(t, heparin_bolus, heparin_prime, k, time_to_cpb, prime_timing)
    for t_bolus, dose in topups:
        if t >= t_bolus and dose:
            amount += dose * math.exp(-k * (t - t_bolus))
    return amount


# ==========================================================================
# BACKWARDS-COMPATIBLE ENDPOINT WRAPPERS
# ==========================================================================
# The original module exposed one function per model. They are kept so that the
# Streamlit app and any existing notebooks continue to work unchanged.

def prodose_dose(heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb):
    return get_reference_dose("prodose", heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb)


def prodose2_dose(heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb):
    return get_reference_dose("prodose-2", heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb)


def meesters_dose(heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb):
    return get_reference_dose("meesters", heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb)


def lanoiselee_dose(heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb):
    return get_reference_dose("lanoiselee", heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb)


def delavenne_dose(heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb):
    return get_reference_dose("delavenne", heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb)


def jia_dose(heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb):
    return get_reference_dose("jia", heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb)


MODEL_DISPATCH: Dict[str, Callable] = {
    "prodose": prodose_dose,
    "prodose-2": prodose2_dose,
    "meesters": meesters_dose,
    "lanoiselee": lanoiselee_dose,
    "delavenne": delavenne_dose,
    "jia": jia_dose,
}


# ==========================================================================
# VECTORISED EVALUATION
# ==========================================================================
# The calibration, uncertainty and coverage analyses evaluate the reference
# models millions of times. These array versions use exactly the same
# expressions as the scalar functions above; ``test_core.py`` asserts they agree
# to floating-point tolerance, so there is no second implementation to maintain.

def reference_amount_array(
    model_name: str,
    t,
    heparin_bolus,
    heparin_prime,
    weight_kg,
    time_to_cpb,
    *,
    overrides: Dict[str, float] | None = None,
    jia_allometric: bool = False,
    covariate_bolus=None,
):
    """Vectorised ``reference_amount``. All arguments broadcast."""
    import numpy as np

    key = canonical_model_name(model_name)
    t = np.asarray(t, dtype=float)
    bolus = np.asarray(heparin_bolus, dtype=float)
    prime = np.asarray(heparin_prime, dtype=float)
    weight = np.asarray(weight_kg, dtype=float)
    t_to = np.asarray(time_to_cpb, dtype=float)

    covariate_dose = bolus if covariate_bolus is None else np.asarray(covariate_bolus, dtype=float)
    prime_elapsed = t - t_to
    prime_on = prime_elapsed >= 0
    prime_elapsed_safe = np.where(prime_on, prime_elapsed, 0.0)

    def _biexp(dose, tt, alpha, beta, k21):
        a = (alpha - k21) / (alpha - beta)
        b = (k21 - beta) / (alpha - beta)
        return dose * (a * np.exp(-alpha * tt) + b * np.exp(-beta * tt))

    def _constants(Cl, Vc, Vp, Q):
        k10, k12, k21 = Cl / Vc, Q / Vc, Q / Vp
        sum_k = k10 + k12 + k21
        root = np.sqrt(sum_k ** 2 - 4.0 * k10 * k21)
        return (sum_k + root) / 2.0, (sum_k - root) / 2.0, k21

    if key in ("prodose", "meesters"):
        p = dict(MODEL_PARAMETERS[key].values)
        if overrides:
            p.update(overrides)
        if key == "prodose":
            k2 = 0.693 / (p["t_half_intercept"] + p["t_half_slope"] * (covariate_dose / weight))
        else:
            k2 = np.full_like(t, 0.693 / p["t_half_slow"])
        out = (
            bolus * p["fast_fraction"] * np.exp(-p["k_fast"] * t)
            + bolus * (1.0 - p["fast_fraction"]) * np.exp(-k2 * t)
        )
        return out + np.where(prime_on, prime * np.exp(-k2 * prime_elapsed_safe), 0.0)

    if key == "prodose-2":
        p = dict(MODEL_PARAMETERS[key].values)
        if overrides:
            p.update(overrides)
        k2_base = 0.693 / (p["t_half_intercept"] + p["t_half_slope"] * (covariate_dose / weight))
        ebv = weight * 70.0
        v_factor = ebv / (ebv + p["v_prime_mL"])
        k2_cpb = k2_base * v_factor

        pool_fast = bolus * p["fast_fraction"] * np.exp(-p["k_fast"] * t)
        slow_before = bolus * (1.0 - p["fast_fraction"]) * np.exp(-k2_cpb * t)
        slow_at_cpb = bolus * (1.0 - p["fast_fraction"]) * np.exp(-k2_cpb * t_to)
        slow_after = (slow_at_cpb + prime) * np.exp(-k2_cpb * prime_elapsed_safe)
        pool_slow = np.where(prime_on, slow_after, slow_before)
        recovered = pool_slow * v_factor + pool_slow * (1.0 - v_factor) * p["reinfusion"]
        return pool_fast + recovered

    if key == "lanoiselee":
        p = dict(MODEL_PARAMETERS[key].values)
        if overrides:
            p.update(overrides)
        alpha, beta, k21 = _constants(p["Cl"], p["Vc"], p["Vp"], p["Q"])
    elif key == "delavenne":
        p = dict(MODEL_PARAMETERS[key].values)
        if overrides:
            p.update(overrides)
        Vc = p["Vc_L"] * (weight / 70.0) * 1000.0
        Cl = p["Cl_L_h"] * (weight / 70.0) ** 0.767 * 1000.0 / 60.0
        alpha, beta, k21 = _constants(Cl, Vc, p["Vp_L"] * 1000.0, p["Q_L_h"] * 1000.0 / 60.0)
    elif key == "jia":
        p = dict(MODEL_PARAMETERS[key].values)
        if overrides:
            p.update(overrides)
        Vc_L, Cl_L_h, Vp_L, Q_L_h = p["Vc_L"], p["Cl_L_h"], p["Vp_L"], p["Q_L_h"]
        if jia_allometric:
            ratio = weight / JIA_ALLOMETRY["reference_weight_kg"]
            Vc_L = Vc_L * ratio ** JIA_ALLOMETRY["volume_exponent"]
            Vp_L = Vp_L * ratio ** JIA_ALLOMETRY["volume_exponent"]
            Cl_L_h = Cl_L_h * ratio ** JIA_ALLOMETRY["clearance_exponent"]
            Q_L_h = Q_L_h * ratio ** JIA_ALLOMETRY["clearance_exponent"]
        alpha, beta, k21 = _constants(
            Cl_L_h * 1000.0 / 60.0, Vc_L * 1000.0, Vp_L * 1000.0, Q_L_h * 1000.0 / 60.0
        )
    else:
        raise ValueError(f"Unhandled model: {model_name!r}")

    out = _biexp(bolus, np.maximum(t, 0.0), alpha, beta, k21)
    out = out + np.where(prime_on, _biexp(prime, prime_elapsed_safe, alpha, beta, k21), 0.0)
    return out


def simplified_amount_array(t, heparin_bolus, heparin_prime, k, time_to_cpb=0.0,
                            prime_timing: str = "lumped_t0"):
    """Vectorised ``simplified_amount``."""
    import numpy as np

    t = np.asarray(t, dtype=float)
    bolus = np.asarray(heparin_bolus, dtype=float)
    prime = np.asarray(heparin_prime, dtype=float)

    if prime_timing == "lumped_t0":
        return (bolus + prime) * np.exp(-k * t)
    if prime_timing == "cpb_onset":
        t_to = np.asarray(time_to_cpb, dtype=float)
        elapsed = t - t_to
        return bolus * np.exp(-k * t) + np.where(elapsed >= 0, prime * np.exp(-k * np.maximum(elapsed, 0.0)), 0.0)
    raise ValueError(f"prime_timing must be one of {PRIME_TIMING_CHOICES}, got {prime_timing!r}")
