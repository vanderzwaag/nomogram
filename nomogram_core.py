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
    "lanoiselee": "Lanoiselée",
    "meesters": "Meesters",
    "prodose": "PRODOSE",
    "prodose-2": "PRODOSE-2",
}

# All six reference models are adult. The manuscript described Jia as paediatric
# and EB-4 / R1 p11 L46 both follow from that description, but its published
# parameters are adult-scale (Vc 3.04 L, within 2% of Delavenne's adult 3.1 L)
# and the source is an adult cardiac-surgical study in a relatively small-bodied
# population. That comment is answered by correcting the text, not by a
# paediatric re-analysis; the transportability evaluation instead includes a
# small-bodied adult institution matching Jia's derivation population.
PAEDIATRIC_MODELS: tuple = ()


# ==========================================================================
# PARAMETER PROVENANCE
# ==========================================================================

@dataclass(frozen=True)
class ParameterSet:
    """Published population parameters and their reported uncertainty.

    ``iiv``
        Between-subject (interindividual) variability as the SD of the
        log-normal random effect, i.e. omega and NOT omega-squared. Sources
        differ in which they tabulate, so each entry below records what the
        source printed and any conversion applied.
    ``rse``
        Relative standard error of the *population estimate*, as a fraction --
        the quantity EB-2 asks us to propagate. ``None`` means the source did
        not report it, and the runner says so rather than substituting a number.
    ``bootstrap_ci``
        Published non-parametric 95% confidence interval as ``(low, high)``.
        Where a source reports one it is preferred over ``rse``: the asymptotic
        standard error assumes a symmetric, quadratic likelihood, which breaks
        down for poorly identified parameters -- for Jia's Vp the bootstrap
        interval is roughly twice the width the RSE implies.
    """

    name: str
    source: str
    values: Dict[str, float]
    iiv: Dict[str, float] = field(default_factory=dict)
    rse: Dict[str, float] = field(default_factory=dict)
    bootstrap_ci: Dict[str, tuple] = field(default_factory=dict)
    weight_covariate: bool = False
    notes: str = ""

    def missing_rse(self) -> Sequence[str]:
        return tuple(k for k in self.values
                     if self.rse.get(k) is None and k not in self.bootstrap_ci)

    def uncertainty_sigma(self, name: str) -> float | None:
        """Log-scale SD describing uncertainty in the estimate of ``name``.

        Derived from the published bootstrap interval when there is one --
        sigma = (ln(high) - ln(low)) / (2 x 1.96) -- and from the RSE otherwise.
        """
        import math as _math

        ci = self.bootstrap_ci.get(name)
        if ci and ci[0] > 0 and ci[1] > 0:
            return (_math.log(ci[1]) - _math.log(ci[0])) / (2 * 1.96)
        return self.rse.get(name)

    def uncertainty_basis(self, name: str) -> str:
        if self.bootstrap_ci.get(name):
            return "published bootstrap 95% CI"
        if self.rse.get(name) is not None:
            return "published %RSE of the estimate"
        return "not reported"


MODEL_PARAMETERS: Dict[str, ParameterSet] = {
    "lanoiselee": ParameterSet(
        name="Lanoiselee",
        source=("Lanoiselee et al. Br J Anaesth 2026;136(3):847-855 "
                "(doi 10.1016/j.bja.2025.11.057), Supplementary Table 1"),
        # mL and mL/min. The source tabulates Cl and Q in mL/h.
        values={"Cl": 1500.18 / 60.0, "Vc": 4011.12, "Vp": 1458.59, "Q": 287.57 / 60.0},
        # Inter-individual variability column of Supplementary Table 1.
        #
        # CORRECTED. The submitted code used 0.0983 / 0.111 / 0.395 / 0.21, which
        # are that column's PARENTHETICAL %RSE values (9.83, 11.1, 39.5, 21.0)
        # divided by 100 -- the precision of the variability estimates, not the
        # variability itself. All four matched to the digit, so this was a
        # read-across error rather than a coincidence. The true values are about
        # 2.1x larger, so every credible band drawn for this model was roughly
        # half the width it should have been.
        iiv={"Cl": 0.27, "Vc": 0.22, "Vp": 0.74, "Q": 0.41},
        # %RSE of the population means, as fractions (4.18, 3.58, 36.5, 11.0).
        rse={"Cl": 0.0418, "Vc": 0.0358, "Vp": 0.365, "Q": 0.110},
        weight_covariate=False,
        notes=(
            "Fixed population parameters; the published model carries no weight "
            "covariate, so IBW affects the bolus size only, not the kinetics. "
            "Vp is the least well identified parameter in any of the three "
            "population models (36.5% RSE), and Vp and Q are precisely what make "
            "the model biexponential -- the structural feature a mono-exponential "
            "nomogram is least able to reproduce is also the one the reference "
            "model itself pins down worst. "
            "The source reports a full PK/PD model (effect compartment plus "
            "sigmoidal Emax to ACT: Ke 12.77/h, ACT0 115.87 s, Emax 403.97, "
            "gamma 2.35, C50 2.62 anti-Xa IU/mL) which this pipeline does not "
            "implement; see PD_PARAMETERS and the EB-1 note below. "
            "The variability column is labelled only 'Inter-Individual "
            "Variability' and is READ HERE AS omega (log-scale SD); the source "
            "does not state the convention. "
            "NOTE: the submitted code held two spellings of Q -- 4.7928 mL/min in "
            "the endpoint model and 287.57/60 = 4.79283 mL/min in the credible-"
            "interval simulation. The unrounded value is used throughout here; "
            "the difference is about 0.02 IU at the reversal timepoint."
        ),
    ),
    "delavenne": ParameterSet(
        name="Delavenne",
        source="Delavenne et al., Table 2",
        # Litres and L/h as published; converted to mL/min at point of use.
        # The two weight exponents are parameters, not literals, so that the
        # 29% RSE on the clearance exponent can be propagated (EB-2).
        values={"Cl_L_h": 0.841, "Vc_L": 3.1, "Vp_L": 2.23, "Q_L_h": 4.67,
                "wt_exponent_Vc": 1.0, "wt_exponent_Cl": 0.767},
        # Interpatient variability column. VERIFIED CORRECT against Table 2 --
        # this model's values were transcribed properly, which is what confirmed
        # the Lanoiselee and Jia entries were not. Vp and Q show "--" in the
        # source and carry no random effect.
        iiv={"Cl_L_h": 0.221, "Vc_L": 0.119, "Vp_L": 0.0, "Q_L_h": 0.0,
             "wt_exponent_Vc": 0.0, "wt_exponent_Cl": 0.0},
        # %RSE of the population means, as fractions (10, 4, 8, 15), plus the
        # covariate row: the Vc exponent is fixed at 1 and carries no
        # uncertainty; the Cl exponent is 0.767 with 29% RSE.
        rse={"Cl_L_h": 0.10, "Vc_L": 0.04, "Vp_L": 0.08, "Q_L_h": 0.15,
             "wt_exponent_Vc": 0.0, "wt_exponent_Cl": 0.29},
        weight_covariate=True,
        notes=(
            "Vc scales with weight (exponent 1.0, fixed) and Cl with weight "
            "(exponent 0.767, 29% RSE), both as (WT/70)^exponent. The published "
            "model is driven by ACTUAL body weight; the pipeline passes IBW, "
            "which is a documented approximation (EB-1). "
            "Because the covariate is centred on 70 kg, uncertainty in the "
            "clearance exponent contributes EXACTLY NOTHING at 70 kg -- the IBW "
            "of the canonical cohort -- and up to about 25% on clearance at the "
            "40 and 115 kg ends of the calibration grid. It therefore affects the "
            "boundary and transportability analyses (EB-4) and not the headline "
            "interval. "
            "Vp and Q have no reported IIV and are held fixed between patients, "
            "but both have a reported RSE and so do contribute to the "
            "parameter-uncertainty interval. "
            "The source reports a PD layer (ACT0 116 s, C50 3.49 IU/mL, Emax 720 s, "
            "no Hill coefficient) not implemented here; see PD_PARAMETERS."
        ),
    ),
    "jia": ParameterSet(
        name="Jia",
        source=("Jia et al., Pharmacokinetic model of unfractionated heparin during "
                "and after cardiopulmonary bypass in cardiac surgery"),
        values={"Cl_L_h": 1.18, "Vc_L": 3.04, "Vp_L": 8.01, "Q_L_h": 0.171},
        # CORRECTED, and the previous values were wrong three times over.
        #
        # The submitted code used [0.073, 0.081, 0.144, 0.318] under a comment
        # declaring the order [Cl, Vc, Vp, Q]. Those numbers are the
        # POPULATION-MEAN %RSE column (7.25, 8.09, 14.40, 31.8) divided by 100,
        # in the source's printed ROW order Cl, Vc, Q, Vp -- so Vp and Q also
        # received each other's values. The code then took a square root of them,
        # which would have been right had they been the omega-squared column but
        # was not right for these. Net effect: Vp was given 38% variability that
        # the source FIXES AT ZERO, and Q was overstated about 1.8-fold.
        #
        # The docstring in that same function gave yet another set
        # (0.176 / 0.114 / 0.0573 / 0.111) which matches no column of the source
        # under any transformation; its origin is unknown and it is discarded.
        #
        # The source tabulates omega-SQUARED. Converted to omega here:
        #   Cl sqrt(0.122) = 0.3493, Vc sqrt(0.105) = 0.3240,
        #   Q  sqrt(0.0978) = 0.3127, Vp fixed at 0.
        iiv={"Cl_L_h": 0.34928, "Vc_L": 0.32404, "Vp_L": 0.0, "Q_L_h": 0.31273},
        # %RSE of the population means, as fractions (7.25, 8.09, 31.8, 14.40).
        rse={"Cl_L_h": 0.0725, "Vc_L": 0.0809, "Vp_L": 0.318, "Q_L_h": 0.1440},
        # The source also publishes non-parametric bootstrap 95% CIs, which are
        # preferred over the asymptotic RSE. For the well-identified parameters
        # the two agree; for Q and Vp the bootstrap interval is about twice as
        # wide, which is the usual behaviour when the likelihood is not locally
        # quadratic.
        bootstrap_ci={"Cl_L_h": (0.99, 1.35), "Vc_L": (2.57, 3.51),
                      "Vp_L": (2.63, 27.6), "Q_L_h": (0.0977, 0.311)},
        weight_covariate=False,
        notes=(
            "AN ADULT MODEL. The submitted manuscript described Jia as paediatric "
            "and EB-4 and R1 p11 L46 both proceed from that description, but the "
            "source is an adult cardiac-surgical study in a relatively small-bodied "
            "(Chinese) population. Its Vc of 3.04 L is within 2% of Delavenne's "
            "adult 3.1 L and implies a body size of roughly 45-75 kg; a 10 kg child "
            "would have a Vc near 0.4 L. The correct response to EB-4 and R1 p11 "
            "L46 is therefore a correction to the manuscript text, not a paediatric "
            "re-analysis. "
            "The model carries no weight covariate at all, so within the pipeline "
            "IBW scales the administered bolus but never the kinetics; the model "
            "cannot be individualised by weight. Applying it across the upper end "
            "of the adult calibration grid extrapolates well beyond the body size "
            "of its derivation population, which is a stated limitation. "
            "Its peripheral compartment is barely identified (Vp bootstrap CI 2.63 "
            "to 27.6 L, a tenfold range), so there is little second exponential for "
            "a mono-exponential approximation to miss -- consistent with Jia being "
            "the easiest of the six models to approximate."
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


# ==========================================================================
# THE PHARMACODYNAMIC LAYER THIS PIPELINE DOES NOT IMPLEMENT (EB-1)
# ==========================================================================
#
# Two of the three population models are PK/PD models: they carry an explicit
# link from anti-Xa activity to activated clotting time. The nomogram pipeline
# stops at the pharmacokinetic central-compartment amount and applies a fixed
# institutional protamine ratio to it. That is a deliberate, stated choice, and
# the parameters below are recorded so the decision can be defended rather than
# merely conceded.
#
# The argument for stopping at the PK layer is that the two reference models
# disagree about the PD relationship by more than the error being characterised:
# at the same anti-Xa concentration they predict activated clotting times 90 to
# 125 s apart across the whole clinical range, with ceilings of 520 s and 836 s.
# A PD layer would import that unresolved disagreement rather than remove it.
# For scale, a 10% error in residual heparin moves the Lanoiselee-predicted ACT
# by about 17 s, well inside that model's own 11% proportional residual error.
#
# These are reference values only -- nothing in the pipeline reads them.
PD_PARAMETERS = {
    "lanoiselee": {
        "model": "effect compartment (Ke 12.77 /h) plus sigmoidal Emax",
        "ACT0_s": 115.87, "Emax_s": 403.97, "gamma": 2.35,
        "C50_antiXa_IU_mL": 2.62,
        "proportional_residual_error": {"anti_Xa": 0.29, "ACT": 0.11},
    },
    "delavenne": {
        "model": "Emax, no Hill coefficient reported",
        "ACT0_s": 116.0, "Emax_s": 720.0, "gamma": 1.0,
        "C50_antiXa_IU_mL": 3.49,
    },
}


def canonical_model_name(model_name: str) -> str:
    """Map any accepted spelling of a model onto its canonical key.

    Accents are stripped, so both "Lanoiselee" and the correct "Lanoiselée"
    resolve to the ASCII key the lookup tables and filenames use. R1 asked for
    the acute accent to be used throughout; the display name carries it while
    the key stays ASCII, so no file path or pickle key changes.
    """
    import unicodedata

    key = unicodedata.normalize("NFKD", model_name.strip().lower())
    key = "".join(c for c in key if not unicodedata.combining(c))
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

    Vc = p["Vc_L"] * (weight_kg / 70.0) ** p["wt_exponent_Vc"] * 1000.0
    Cl = (p["Cl_L_h"] * (weight_kg / 70.0) ** p["wt_exponent_Cl"]) * 1000.0 / 60.0
    Vp = p["Vp_L"] * 1000.0
    Q = p["Q_L_h"] * 1000.0 / 60.0
    return _biexponential_constants(Cl, Vc, Vp, Q)


def delavenne_response(dose, t, weight_kg, overrides=None):
    alpha, beta, k21, _ = get_delavenne_params(weight_kg, overrides)
    return _biexponential_amount(dose, t, alpha, beta, k21)


# ---------------------------------------------------------------------- Jia

def get_jia_params(weight_kg: float, overrides=None):
    """Jia parameters.

    ``weight_kg`` is accepted for signature symmetry with the other models and
    is deliberately unused: the published model carries no weight covariate, so
    its kinetics are identical at every body weight. Within the pipeline, IBW
    therefore scales the administered bolus but never the elimination.
    """
    p = dict(MODEL_PARAMETERS["jia"].values)
    if overrides:
        p.update(overrides)
    return _biexponential_constants(
        p["Cl_L_h"] * 1000.0 / 60.0, p["Vc_L"] * 1000.0,
        p["Vp_L"] * 1000.0, p["Q_L_h"] * 1000.0 / 60.0,
    )


def jia_response(dose, t, weight_kg, overrides=None):
    alpha, beta, k21, _ = get_jia_params(weight_kg, overrides)
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
        amount = jia_response(heparin_bolus, t, weight_kg, overrides)
        if prime_active:
            amount += jia_response(heparin_prime, prime_elapsed, weight_kg, overrides)
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
) -> float:
    """Reference amount at ``t`` including supplemental boluses by superposition.

    ``topups`` is a sequence of ``(time_min, dose_IU)`` pairs. Superposition is
    exact for the linear two-compartment models and for the closed-form
    biexponential expressions, all of which are linear in dose.
    """
    amount = reference_amount(
        model_name, t, heparin_bolus, heparin_prime, weight_kg, time_to_cpb,
        overrides=overrides,
    )
    for t_bolus, dose in topups:
        if t >= t_bolus and dose:
            # A supplemental bolus is a systemic bolus with no prime attached.
            # The dose-dependent models keep the index bolus as the covariate,
            # so the top-up is eliminated at the patient's own half-life rather
            # than at the half-life a small induction dose would have had.
            amount += reference_amount(
                model_name, t - t_bolus, dose, 0.0, weight_kg, 0.0,
                overrides=overrides,
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
        Vc = p["Vc_L"] * (weight / 70.0) ** p["wt_exponent_Vc"] * 1000.0
        Cl = p["Cl_L_h"] * (weight / 70.0) ** p["wt_exponent_Cl"] * 1000.0 / 60.0
        alpha, beta, k21 = _constants(Cl, Vc, p["Vp_L"] * 1000.0, p["Q_L_h"] * 1000.0 / 60.0)
    elif key == "jia":
        p = dict(MODEL_PARAMETERS[key].values)
        if overrides:
            p.update(overrides)
        alpha, beta, k21 = _constants(
            p["Cl_L_h"] * 1000.0 / 60.0, p["Vc_L"] * 1000.0,
            p["Vp_L"] * 1000.0, p["Q_L_h"] * 1000.0 / 60.0,
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
