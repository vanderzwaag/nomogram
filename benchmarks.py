"""
Verification of each model implementation (EB-6).

The reviewer asked us to "demonstrate that each implementation reproduces
benchmarks from its source publication". That splits into two things, and only
one of them can be done from inside the repository:

1. **Internal verification** -- does the closed-form solution we coded actually
   solve the differential equations of the published model, and does it satisfy
   the identities those equations imply (initial condition, area under the
   curve, terminal slope)? This is fully automatic and runs in the test suite.
   It is what catches a mis-transcribed micro-constant or a swapped Vp and Vc.

2. **External verification** -- does it reproduce a specific number printed in
   the source paper (a figure value, a table entry, a worked example)? That
   requires the numbers themselves. ``SOURCE_BENCHMARKS`` is the table to fill
   in; each entry stays ``PENDING`` and is reported as such until an expected
   value is supplied, so the manuscript cannot claim a verification that has
   not happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.integrate import quad, solve_ivp

import nomogram_core as core

TWO_COMPARTMENT_MODELS = ("lanoiselee", "delavenne", "jia")


# ==========================================================================
# 1. INTERNAL VERIFICATION
# ==========================================================================

def _micro_constants(model_key: str, weight_kg: float):
    """(Cl, Vc, Vp, Q) in mL and mL/min for a two-compartment model."""
    p = core.MODEL_PARAMETERS[model_key].values
    if model_key == "lanoiselee":
        return p["Cl"], p["Vc"], p["Vp"], p["Q"]
    if model_key == "delavenne":
        Vc = p["Vc_L"] * (weight_kg / 70.0) * 1000.0
        Cl = p["Cl_L_h"] * (weight_kg / 70.0) ** 0.767 * 1000.0 / 60.0
        return Cl, Vc, p["Vp_L"] * 1000.0, p["Q_L_h"] * 1000.0 / 60.0
    if model_key == "jia":
        return (p["Cl_L_h"] * 1000.0 / 60.0, p["Vc_L"] * 1000.0,
                p["Vp_L"] * 1000.0, p["Q_L_h"] * 1000.0 / 60.0)
    raise ValueError(model_key)


def verify_against_ode(model_key: str, dose: float = 28000.0,
                       weight_kg: float = 70.0, t_end: float = 240.0,
                       rtol: float = 1e-9) -> Dict[str, float]:
    """Compare the analytical solution with a numerical solution of the ODEs.

    The dashboard solves the two-compartment system numerically for its
    credible intervals while the calibration uses the closed form. If the two
    disagree, one of them is wrong; this check ties them together.
    """
    Cl, Vc, Vp, Q = _micro_constants(model_key, weight_kg)

    def rhs(_t, y):
        Ac, Ap = y
        return [-(Cl / Vc) * Ac + Q * (Ap / Vp - Ac / Vc),
                Q * (Ac / Vc - Ap / Vp)]

    times = np.linspace(0.0, t_end, 241)
    sol = solve_ivp(rhs, (0.0, t_end), [dose, 0.0], t_eval=times,
                    rtol=rtol, atol=1e-9, method="LSODA")
    numeric = sol.y[0]
    analytic = np.array([
        core.reference_amount(model_key, t, dose, 0.0, weight_kg, 0.0) for t in times
    ])
    rel = np.abs(numeric - analytic) / np.maximum(np.abs(numeric), 1.0)
    return {
        "model": model_key,
        "check": "analytical solution vs numerical ODE solution",
        "max_abs_diff_iu": float(np.max(np.abs(numeric - analytic))),
        "max_rel_diff": float(np.max(rel)),
    }


def verify_pk_identities(model_key: str, dose: float = 28000.0,
                         weight_kg: float = 70.0) -> List[Dict]:
    """Check the identities a linear two-compartment model must satisfy.

    * A(0) = dose -- the whole bolus starts in the central compartment.
    * integral of A_c dt = dose * Vc / Cl -- equivalently AUC of concentration
      is dose / Cl, the defining property of clearance.
    * the terminal log-linear slope equals -beta.
    * the initial central concentration equals dose / Vc.
    """
    Cl, Vc, Vp, Q = _micro_constants(model_key, weight_kg)
    alpha, beta, k21, _ = {
        "lanoiselee": core.get_lanoiselee_params(),
        "delavenne": core.get_delavenne_params(weight_kg),
        "jia": core.get_jia_params(weight_kg),
    }[model_key]

    amount = lambda t: core.reference_amount(model_key, t, dose, 0.0, weight_kg, 0.0)

    auc_amount, _ = quad(amount, 0.0, np.inf, limit=400)

    # The terminal slope only equals -beta once the distribution (alpha) term has
    # died away. For Jia the beta term carries less than 1% of the dose, so the
    # alpha term still dominates at 10 hours; the sampling times are therefore
    # derived from alpha and beta rather than fixed.
    t1 = 30.0 / alpha
    t2 = t1 + 1.0 / beta
    terminal_slope = float(np.log(amount(t2) / amount(t1)) / (t2 - t1))

    return [
        {"model": model_key, "check": "A(0) == dose",
         "expected": dose, "observed": amount(0.0), "tolerance_rel": 1e-9},
        {"model": model_key, "check": "integral A_c dt == dose * Vc / Cl",
         "expected": dose * Vc / Cl, "observed": auc_amount, "tolerance_rel": 1e-6},
        {"model": model_key, "check": "terminal slope == -beta",
         "expected": -beta, "observed": terminal_slope, "tolerance_rel": 1e-4},
        {"model": model_key, "check": "C(0) == dose / Vc",
         "expected": dose / Vc, "observed": amount(0.0) / Vc, "tolerance_rel": 1e-9},
    ]


def verify_closed_form_models(dose_per_kg: float = 400.0, weight_kg: float = 70.0) -> List[Dict]:
    """Checks for the three closed-form (non-ODE) models.

    These are published as algebraic expressions rather than as a compartmental
    system, so the verifiable properties are the stated half-lives and the
    fast/slow pool split.
    """
    bolus = dose_per_kg * weight_kg
    out = []

    # Note on the expected values below: all three published expressions write
    # the decay constant as 0.693 / t_half rather than ln(2) / t_half, so the
    # slow pool falls to exp(-0.693) = 0.50003 of its starting value at t_half,
    # not to exactly one half. The expected values use the published constant,
    # which makes the check exact and records the rounding.
    half_published = float(np.exp(-0.693))

    # PRODOSE: slow pool half-life = 26 + 0.323 * (IU/kg).
    p = core.MODEL_PARAMETERS["prodose"].values
    expected_t_half = p["t_half_intercept"] + p["t_half_slope"] * dose_per_kg
    slow_only = bolus * (1.0 - p["fast_fraction"])
    a1 = core.reference_amount("prodose", 0.0, bolus, 0.0, weight_kg, 0.0)
    a2 = core.reference_amount("prodose", expected_t_half, bolus, 0.0, weight_kg, 0.0)
    fast_at_t = bolus * p["fast_fraction"] * np.exp(-p["k_fast"] * expected_t_half)
    out.append({"model": "prodose",
                "check": f"slow pool decays to exp(-0.693) by t = {expected_t_half:.2f} min",
                "expected": slow_only * half_published, "observed": a2 - fast_at_t,
                "tolerance_rel": 1e-9})
    out.append({"model": "prodose", "check": "A(0) == bolus",
                "expected": bolus, "observed": a1, "tolerance_rel": 1e-9})

    # Meesters: fixed 250 min slow half-life.
    m = core.MODEL_PARAMETERS["meesters"].values
    t_half = m["t_half_slow"]
    slow_only = bolus * (1.0 - m["fast_fraction"])
    a2 = core.reference_amount("meesters", t_half, bolus, 0.0, weight_kg, 0.0)
    fast_at_t = bolus * m["fast_fraction"] * np.exp(-m["k_fast"] * t_half)
    out.append({"model": "meesters", "check": "slow pool decays to exp(-0.693) by t = 250 min",
                "expected": slow_only * half_published, "observed": a2 - fast_at_t,
                "tolerance_rel": 1e-9})

    # PRODOSE-2: with full reinfusion the recovered slow pool is undiluted, so
    # the amount at t = 0 must still be the whole bolus.
    a1 = core.reference_amount("prodose-2", 0.0, bolus, 0.0, weight_kg, 0.0)
    out.append({"model": "prodose-2", "check": "A(0) == bolus with reinfusion = 1.0",
                "expected": bolus, "observed": a1, "tolerance_rel": 1e-9})

    # Prime timing: every model must place the prime at CPB onset, so the
    # amount must jump by exactly the prime dose across t = time_to_cpb.
    for model in core.MODEL_NAMES:
        t_to, prime = 15.0, 5000.0
        just_before = core.reference_amount(model, t_to - 1e-6, bolus, prime, weight_kg, t_to)
        just_after = core.reference_amount(model, t_to, bolus, prime, weight_kg, t_to)
        out.append({"model": model, "check": "prime enters at CPB onset (jump == prime dose)",
                    "expected": prime, "observed": just_after - just_before,
                    "tolerance_rel": 1e-6})
    return out


def internal_verification_report(weight_kg: float = 70.0) -> pd.DataFrame:
    """All internal checks, with a pass/fail column."""
    rows: List[Dict] = []
    for m in TWO_COMPARTMENT_MODELS:
        rows.extend(verify_pk_identities(m, weight_kg=weight_kg))
    rows.extend(verify_closed_form_models(weight_kg=weight_kg))

    df = pd.DataFrame(rows)
    df["abs_error"] = (df["observed"] - df["expected"]).abs()
    df["rel_error"] = df["abs_error"] / df["expected"].abs().clip(lower=1e-12)
    df["pass"] = df["rel_error"] <= df["tolerance_rel"]

    ode = pd.DataFrame([verify_against_ode(m, weight_kg=weight_kg)
                        for m in TWO_COMPARTMENT_MODELS])
    ode["pass"] = ode["max_rel_diff"] < 1e-6
    return pd.concat([df, ode], ignore_index=True)


# ==========================================================================
# 2. EXTERNAL VERIFICATION AGAINST THE SOURCE PUBLICATIONS
# ==========================================================================

@dataclass
class SourceBenchmark:
    """One number taken from a source publication, to be reproduced."""

    model: str
    description: str
    inputs: Dict[str, float]
    expected: Optional[float]       # None until transcribed from the paper
    units: str
    tolerance_pct: float = 10.0
    citation: str = ""

    def run(self) -> Dict:
        observed = core.reference_amount(
            self.model,
            self.inputs["t"],
            self.inputs["heparin_bolus"],
            self.inputs.get("heparin_prime", 0.0),
            self.inputs.get("weight_kg", 70.0),
            self.inputs.get("time_to_cpb", 0.0),
        )
        row = {
            "model": self.model,
            "description": self.description,
            "citation": self.citation,
            "units": self.units,
            "expected": self.expected,
            "observed": observed,
            "tolerance_pct": self.tolerance_pct,
        }
        if self.expected is None:
            row["status"] = "PENDING - expected value not yet transcribed from source"
            row["pct_error"] = None
        else:
            pct = 100.0 * abs(observed - self.expected) / abs(self.expected)
            row["pct_error"] = pct
            row["status"] = "PASS" if pct <= self.tolerance_pct else "FAIL"
        return row


# AUTHOR ACTION REQUIRED (EB-6): fill in ``expected`` for each entry from the
# source publication -- a concentration-time figure value, a table entry or a
# worked example. Until then the runner reports PENDING and the manuscript must
# not state that per-source reproduction has been demonstrated.
SOURCE_BENCHMARKS: List[SourceBenchmark] = [
    SourceBenchmark(
        model="lanoiselee",
        description="Central-compartment amount 60 min after a 300 IU/kg bolus in a 70 kg adult",
        inputs={"t": 60.0, "heparin_bolus": 21000.0, "weight_kg": 70.0},
        expected=None, units="IU",
        citation="Lanoiselee et al. Br J Anaesth 2026;136(3):847-855",
    ),
    SourceBenchmark(
        model="delavenne",
        description="Central-compartment amount 60 min after a 300 IU/kg bolus in a 70 kg adult",
        inputs={"t": 60.0, "heparin_bolus": 21000.0, "weight_kg": 70.0},
        expected=None, units="IU", citation="Delavenne et al.",
    ),
    SourceBenchmark(
        model="jia",
        description="Central-compartment amount 60 min after a 400 IU/kg bolus in a 10 kg child",
        inputs={"t": 60.0, "heparin_bolus": 4000.0, "weight_kg": 10.0},
        expected=None, units="IU", citation="Jia et al.",
    ),
    SourceBenchmark(
        model="prodose",
        description="Residual heparin for the worked example in the PRODOSE publication",
        inputs={"t": 75.0, "heparin_bolus": 28000.0, "heparin_prime": 5000.0,
                "weight_kg": 70.0, "time_to_cpb": 15.0},
        expected=None, units="IU", citation="PRODOSE",
    ),
    SourceBenchmark(
        model="meesters",
        description="Residual heparin for the worked example in the Meesters publication",
        inputs={"t": 75.0, "heparin_bolus": 28000.0, "heparin_prime": 5000.0,
                "weight_kg": 70.0, "time_to_cpb": 15.0},
        expected=None, units="IU", citation="Meesters et al.",
    ),
]


def source_verification_report() -> pd.DataFrame:
    return pd.DataFrame([b.run() for b in SOURCE_BENCHMARKS])
