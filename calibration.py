"""
Seeded, reproducible calibration of the simplified decay constant k.

Three things changed relative to the submitted version, each answering a
specific reviewer comment:

EB-3 / R1 Fig 2 -- **every** draw of randomness now comes from an explicit
    ``numpy.random.Generator`` seeded from a single integer that is recorded in
    the run manifest. The submitted code called the global ``np.random`` with no
    seed anywhere, so ``find_best_k``, the replicate interval and the
    sensitivity analysis each produced a different cohort on every execution.
    That, and not rounding, is why the manuscript body and the Figure 2 legend
    could disagree on both the value and the sign of the bias.

EB-3 -- the objective function is a parameter, not a hard-coded expression, so
    the robustness of k to the choice of loss (Bland-Altman composite, RMSE,
    MAPE, asymmetric clinical loss) can be reported.

EB-4 -- calibration can target the residual amount at the single reversal
    timepoint (what the submitted analysis did) or the decay trajectory across
    a grid of timepoints per patient. The manuscript's claim must match
    whichever is used.

EB-2 -- two distinct intervals are produced and must never be conflated:
    a Monte Carlo sampling-precision interval (repeat cohorts, fixed published
    parameters -- what the submitted "bootstrap" measured) and a
    parameter-uncertainty interval (published PK parameters resampled from
    their reported uncertainty).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import nomogram_core as core

CALIBRATION_VERSION = "2.0.0"

# Physiologically plausible search interval for k, in min^-1 (R1 p9 L29).
# 0.001 corresponds to a ~693 min half-life, 0.03 to a ~23 min half-life; the
# interval brackets every published heparin elimination half-life used here.
K_BOUNDS = (0.001, 0.03)
K_BOUNDS_NOTE = "k restricted to 0.001-0.03 /min, i.e. an apparent half-life of 23 to 693 min"


# ==========================================================================
# COHORT DEFINITION AND SAMPLING
# ==========================================================================

@dataclass(frozen=True)
class CohortSpec:
    """A simulated institution: the input distributions calibration draws from."""

    name: str
    dose_per_kg: float
    prime_heparin: float
    ibw_mean: float
    ibw_sd: float
    t_to_mean: float
    t_to_sd: float
    t_on_mean: float
    t_on_sd: float
    population: str = "adult"          # "adult" or "paediatric"
    ibw_min: float = 1.0               # truncation guards; normal draws can go absurd
    ibw_max: float = 250.0

    def describe(self) -> str:
        return (
            f"{self.name}: {self.dose_per_kg:g} IU/kg, prime {self.prime_heparin:g} IU, "
            f"IBW {self.ibw_mean:g}+/-{self.ibw_sd:g} kg, "
            f"time to CPB {self.t_to_mean:g}+/-{self.t_to_sd:g} min, "
            f"time on CPB {self.t_on_mean:g}+/-{self.t_on_sd:g} min [{self.population}]"
        )

    def as_dict(self) -> Dict:
        return asdict(self)


def sample_cohort(spec: CohortSpec, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw one synthetic cohort.

    Weight, time to CPB and time on CPB are drawn independently, as in the
    submitted analysis. That independence assumption is a stated limitation
    (R2): in real practice the three covary. ``sample_correlated_cohort``
    provides the correlated counterpart for the sensitivity analysis.
    """
    ibw = np.clip(rng.normal(spec.ibw_mean, spec.ibw_sd, n), spec.ibw_min, spec.ibw_max)
    t_to = np.clip(rng.normal(spec.t_to_mean, spec.t_to_sd, n), 0.0, None)
    t_on = np.clip(rng.normal(spec.t_on_mean, spec.t_on_sd, n), 0.0, None)
    return pd.DataFrame({
        "ibw": ibw,
        "heparin_bolus": spec.dose_per_kg * ibw,
        "heparin_prime": np.full(n, float(spec.prime_heparin)),
        "time_to_cpb": t_to,
        "time_on_cpb": t_on,
        "elapsed_time": t_to + t_on,
    })


def sample_correlated_cohort(spec: CohortSpec, n: int, rng: np.random.Generator,
                             correlation: float = 0.3) -> pd.DataFrame:
    """Cohort with a Gaussian-copula correlation among IBW, time to and time on CPB.

    Addresses R2's covariance point as a demonstrative sensitivity run: larger
    patients tend to have longer, more complex operations. ``correlation`` is
    the pairwise Pearson correlation imposed on the latent normals.
    """
    corr = np.full((3, 3), float(correlation))
    np.fill_diagonal(corr, 1.0)
    z = rng.multivariate_normal(np.zeros(3), corr, size=n)

    ibw = np.clip(spec.ibw_mean + spec.ibw_sd * z[:, 0], spec.ibw_min, spec.ibw_max)
    t_to = np.clip(spec.t_to_mean + spec.t_to_sd * z[:, 1], 0.0, None)
    t_on = np.clip(spec.t_on_mean + spec.t_on_sd * z[:, 2], 0.0, None)
    return pd.DataFrame({
        "ibw": ibw,
        "heparin_bolus": spec.dose_per_kg * ibw,
        "heparin_prime": np.full(n, float(spec.prime_heparin)),
        "time_to_cpb": t_to,
        "time_on_cpb": t_on,
        "elapsed_time": t_to + t_on,
    })


def grid_cohort(dose_per_kg: Sequence[float], ibw: Sequence[float],
                prime: Sequence[float], t_to: Sequence[float],
                t_on: Sequence[float]) -> pd.DataFrame:
    """Every combination of the listed values -- the deterministic coverage grid.

    Used for the full-range and boundary evaluation EB-4 asks for; it contains
    no randomness at all, so it is reproducible by construction.
    """
    import itertools

    rows = list(itertools.product(dose_per_kg, ibw, prime, t_to, t_on))
    arr = np.array(rows, dtype=float)
    return pd.DataFrame({
        "ibw": arr[:, 1],
        "heparin_bolus": arr[:, 0] * arr[:, 1],
        "heparin_prime": arr[:, 2],
        "time_to_cpb": arr[:, 3],
        "time_on_cpb": arr[:, 4],
        "elapsed_time": arr[:, 3] + arr[:, 4],
        "dose_per_kg": arr[:, 0],
    })


# ==========================================================================
# EVALUATION POINTS -- ENDPOINT VERSUS TRAJECTORY (EB-4)
# ==========================================================================

EVALUATION_MODES = ("reversal_endpoint", "trajectory_cpb", "trajectory_full")

EVALUATION_MODE_DESCRIPTIONS = {
    "reversal_endpoint":
        "One point per patient: residual heparin at the reversal timepoint "
        "t = time to CPB + time on CPB. This is what the submitted analysis "
        "calibrated against; with it, the manuscript may claim agreement in "
        "residual amount at reversal, NOT reproduction of entire trajectories.",
    "trajectory_cpb":
        "A grid of timepoints per patient spanning CPB onset to reversal, the "
        "window in which the nomogram is actually read. Supports a trajectory "
        "claim over that window.",
    "trajectory_full":
        "A grid of timepoints per patient spanning induction (t = 0) to "
        "reversal, including the pre-bypass distribution phase. This is the "
        "only mode that supports an unqualified 'reproduces the trajectory' "
        "claim, and it is where a mono-exponential approximation of a "
        "biexponential model is worst (R2).",
}


def evaluation_times(cohort: pd.DataFrame, mode: str = "reversal_endpoint",
                     n_points: int = 13) -> np.ndarray:
    """Times (patients x points) at which reference and nomogram are compared."""
    t_to = cohort["time_to_cpb"].to_numpy()
    t_end = cohort["elapsed_time"].to_numpy()

    if mode == "reversal_endpoint":
        return t_end[:, None]
    if mode == "trajectory_cpb":
        frac = np.linspace(0.0, 1.0, n_points)[None, :]
        return t_to[:, None] + frac * (t_end - t_to)[:, None]
    if mode == "trajectory_full":
        frac = np.linspace(0.0, 1.0, n_points)[None, :]
        return frac * t_end[:, None]
    raise ValueError(f"mode must be one of {EVALUATION_MODES}, got {mode!r}")


def reference_values(cohort: pd.DataFrame, model_name: str, times: np.ndarray,
                     overrides=None) -> np.ndarray:
    return core.reference_amount_array(
        model_name,
        times,
        cohort["heparin_bolus"].to_numpy()[:, None],
        cohort["heparin_prime"].to_numpy()[:, None],
        cohort["ibw"].to_numpy()[:, None],
        cohort["time_to_cpb"].to_numpy()[:, None],
        overrides=overrides,
    )


def nomogram_values(cohort: pd.DataFrame, times: np.ndarray, k: float,
                    prime_timing: str = "lumped_t0") -> np.ndarray:
    return core.simplified_amount_array(
        times,
        cohort["heparin_bolus"].to_numpy()[:, None],
        cohort["heparin_prime"].to_numpy()[:, None],
        k,
        cohort["time_to_cpb"].to_numpy()[:, None],
        prime_timing=prime_timing,
    )


# ==========================================================================
# OBJECTIVE FUNCTIONS (EB-3)
# ==========================================================================
#
# Every objective takes (reference, nomogram) flattened arrays and returns a
# scalar to minimise. The convention is the same as in agreement.py:
# difference = reference - nomogram, so a positive residual means the nomogram
# reads low.

def objective_bland_altman(ref: np.ndarray, nom: np.ndarray, loa_weight: float = 0.1) -> float:
    """|bias| + w * (width of the limits of agreement).

    This is the objective used in the submitted analysis, with w = 0.1. It is a
    composite: the first term centres the errors, the second penalises their
    spread. The weight is arbitrary, which is exactly the reviewer's objection
    (EB-3, R1 p9 L44) -- hence it is exposed here and the sensitivity of k to it
    is reported rather than defended.
    """
    diff = ref - nom
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    return abs(bias) + loa_weight * (2 * 1.96 * sd)


def objective_rmse(ref: np.ndarray, nom: np.ndarray) -> float:
    return float(np.sqrt(np.mean((ref - nom) ** 2)))


def objective_mape(ref: np.ndarray, nom: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(ref != 0, np.abs(ref - nom) / np.abs(ref), np.nan)
    return float(100.0 * np.nanmean(pct))


def objective_mae(ref: np.ndarray, nom: np.ndarray) -> float:
    return float(np.mean(np.abs(ref - nom)))


def objective_asymmetric(ref: np.ndarray, nom: np.ndarray, penalty: float = 2.0) -> float:
    """Asymmetric clinical loss.

    Under-estimating residual heparin (nomogram below reference, difference > 0)
    would lead to under-dosing of protamine and residual anticoagulation;
    over-estimating leads to excess protamine. These are not clinically
    equivalent, so the under-estimation limb is weighted by ``penalty``.
    The direction and the size of the penalty are stated assumptions, not
    derived quantities, and the resulting k is reported as a sensitivity
    analysis rather than as the primary calibration.
    """
    diff = ref - nom
    loss = np.where(diff > 0, penalty * diff ** 2, diff ** 2)
    return float(np.sqrt(np.mean(loss)))


OBJECTIVES: Dict[str, Callable[..., float]] = {
    "bland_altman": objective_bland_altman,
    "rmse": objective_rmse,
    "mape": objective_mape,
    "mae": objective_mae,
    "asymmetric": objective_asymmetric,
}

OBJECTIVE_LABELS = {
    "bland_altman": "|bias| + 0.1 x LoA width (primary)",
    "rmse": "Root mean squared error (IU)",
    "mape": "Mean absolute percentage error (%)",
    "mae": "Mean absolute error (IU)",
    "asymmetric": "Asymmetric loss, under-estimation penalised 2:1",
}


# ==========================================================================
# CALIBRATION
# ==========================================================================

@dataclass
class CalibrationResult:
    k: float
    objective: str
    objective_value: float
    model: str
    evaluation_mode: str
    prime_timing: str
    n_patients: int
    n_timepoints: int
    seed: Optional[int]
    at_lower_bound: bool
    at_upper_bound: bool
    half_life_min: float = field(init=False)

    def __post_init__(self):
        self.half_life_min = float(np.log(2.0) / self.k) if self.k > 0 else float("inf")

    def as_dict(self) -> Dict:
        return asdict(self)


def calibrate_k(cohort: pd.DataFrame, model_name: str, *,
                objective: str = "bland_altman",
                evaluation_mode: str = "reversal_endpoint",
                n_timepoints: int = 13,
                prime_timing: str = "lumped_t0",
                k_bounds: tuple = K_BOUNDS,
                overrides=None,
                seed: Optional[int] = None,
                objective_kwargs: Optional[Dict] = None) -> CalibrationResult:
    """Find the decay constant minimising ``objective`` on this cohort."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}, got {objective!r}")

    times = evaluation_times(cohort, evaluation_mode, n_timepoints)
    ref = reference_values(cohort, model_name, times, overrides).ravel()
    loss = OBJECTIVES[objective]
    kwargs = objective_kwargs or {}

    def f(k: float) -> float:
        nom = nomogram_values(cohort, times, k, prime_timing).ravel()
        return loss(ref, nom, **kwargs)

    res = minimize_scalar(f, bounds=k_bounds, method="bounded",
                          options={"xatol": 1e-7})
    k = float(res.x)

    return CalibrationResult(
        k=k,
        objective=objective,
        objective_value=float(res.fun),
        model=core.canonical_model_name(model_name),
        evaluation_mode=evaluation_mode,
        prime_timing=prime_timing,
        n_patients=int(len(cohort)),
        n_timepoints=int(times.shape[1]),
        seed=seed,
        at_lower_bound=bool(abs(k - k_bounds[0]) < 1e-6),
        at_upper_bound=bool(abs(k - k_bounds[1]) < 1e-6),
    )


def find_best_k(spec: CohortSpec, model_name: str, *, seed: int,
                n_sim: int = 1000, **kwargs) -> CalibrationResult:
    """Sample one seeded cohort and calibrate against it."""
    rng = np.random.default_rng(seed)
    cohort = sample_cohort(spec, n_sim, rng)
    return calibrate_k(cohort, model_name, seed=seed, **kwargs)


# ==========================================================================
# INTERVAL 1: MONTE CARLO SAMPLING PRECISION (EB-2)
# ==========================================================================

def monte_carlo_precision(spec: CohortSpec, model_name: str, *, seed: int,
                          n_replicates: int = 500, n_sim: int = 1000,
                          **kwargs) -> pd.DataFrame:
    """Repeat calibration on fresh cohorts drawn from the SAME fixed distributions.

    This is what the submitted code called a "bootstrap". It is not one: no
    dataset is being resampled. It quantifies only how precisely k is pinned
    down by a cohort of ``n_sim`` simulated patients, with the published PK
    parameters held at their point estimates and the input distributions held
    exactly as specified. It says nothing about uncertainty in those parameters
    or in those distributions, and must be labelled a Monte Carlo
    sampling-precision interval wherever it appears.
    """
    root = np.random.default_rng(seed)
    seeds = root.integers(0, 2 ** 32 - 1, size=n_replicates)

    rows = []
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        cohort = sample_cohort(spec, n_sim, rng)
        r = calibrate_k(cohort, model_name, seed=int(s), **kwargs)
        rows.append({"replicate": i, "replicate_seed": int(s), "k": r.k,
                     "objective_value": r.objective_value,
                     "at_bound": r.at_lower_bound or r.at_upper_bound})
    return pd.DataFrame(rows)


# ==========================================================================
# INTERVAL 2: PARAMETER UNCERTAINTY PROPAGATION (EB-2)
# ==========================================================================

def _draw_parameter_overrides(model_key: str, rng: np.random.Generator,
                              source: str = "estimate") -> Optional[Dict[str, float]]:
    """One draw of the published PK parameters from their reported uncertainty.

    ``source``
        ``"estimate"`` -- uncertainty in the population estimate, which is what
            EB-2 asks to be propagated: how well the published study pinned the
            parameter down. Taken from the published non-parametric bootstrap
            interval where the source reports one and from the asymptotic %RSE
            otherwise; ``ParameterSet.uncertainty_basis`` records which was used
            for each parameter. The two agree for well-identified parameters and
            diverge where they should: for Jia's peripheral volume the bootstrap
            interval is roughly twice the width the RSE implies.
        ``"iiv"`` -- interindividual variability. This describes spread between
            patients, not uncertainty in the estimate, and substituting it
            overstates parameter uncertainty. It is only used on request, and the
            substitution is recorded in the manifest.

    Returns ``None`` when the requested uncertainty is not available for this
    model, so the caller can report "not propagated" rather than silently
    reporting an interval built from invented numbers.
    """
    pset = core.MODEL_PARAMETERS[model_key]

    if source == "iiv":
        spread = {k: pset.iiv.get(k) for k in pset.values}
    else:
        spread = {k: pset.uncertainty_sigma(k) for k in pset.values}

    if not spread or all(not spread.get(k) for k in pset.values):
        return None

    out = {}
    for name, value in pset.values.items():
        sigma = spread.get(name)
        if sigma:
            # Log-normal perturbation: keeps volumes, clearances and the weight
            # exponents positive, and reproduces the asymmetry of a bootstrap
            # interval on a ratio-scale parameter.
            out[name] = float(value * np.exp(rng.normal(0.0, sigma)))
        else:
            out[name] = float(value)
    return out


def uncertainty_provenance(model_key: str) -> Dict[str, str]:
    """Which published quantity each parameter's uncertainty came from."""
    pset = core.MODEL_PARAMETERS[core.canonical_model_name(model_key)]
    return {name: pset.uncertainty_basis(name) for name in pset.values}


@dataclass
class UncertaintyReport:
    model: str
    source: str
    available: bool
    reason: str
    draws: Optional[pd.DataFrame] = None
    provenance: Optional[Dict[str, str]] = None


def parameter_uncertainty(spec: CohortSpec, model_name: str, *, seed: int,
                          n_draws: int = 200, n_sim: int = 500,
                          source: str = "estimate",
                          allow_iiv_proxy: bool = False,
                          **kwargs) -> UncertaintyReport:
    """Re-calibrate k with the published PK parameters resampled each draw.

    This is the analysis EB-2 describes as the single most substantial addition:
    it answers "how much does k move if the published model parameters are only
    known to their reported precision?", which the Monte Carlo sampling-precision
    interval does not address at all.

    For Delavenne the draw includes the weight exponent on clearance (0.767,
    29% RSE). Because that covariate is centred on 70 kg, its contribution is
    exactly zero at the canonical cohort's IBW and grows towards the ends of the
    weight grid -- so it moves the boundary and transportability results (EB-4)
    rather than the headline interval.
    """
    key = core.canonical_model_name(model_name)
    pset = core.MODEL_PARAMETERS[key]

    effective_source = source
    probe = _draw_parameter_overrides(key, np.random.default_rng(0), source)
    if probe is None and source == "estimate" and allow_iiv_proxy:
        effective_source = "iiv"
        probe = _draw_parameter_overrides(key, np.random.default_rng(0), "iiv")

    if probe is None:
        return UncertaintyReport(
            model=key, source=effective_source, available=False,
            reason=(
                f"{pset.name} is a closed-form expression published without "
                "parameter uncertainty; no interval can be propagated for it."
            ),
        )

    root = np.random.default_rng(seed)
    rows = []
    for i in range(n_draws):
        rng = np.random.default_rng(int(root.integers(0, 2 ** 32 - 1)))
        overrides = _draw_parameter_overrides(key, rng, effective_source)
        cohort = sample_cohort(spec, n_sim, rng)
        r = calibrate_k(cohort, model_name, overrides=overrides, **kwargs)
        row = {"draw": i, "k": r.k, "at_bound": r.at_lower_bound or r.at_upper_bound}
        row.update({f"param_{k}": v for k, v in overrides.items()})
        rows.append(row)

    bases = uncertainty_provenance(key)
    if effective_source == "iiv":
        reason = ("interindividual variability used as a proxy for parameter "
                  "uncertainty; this overstates it and must be described as such")
    else:
        used = sorted(set(bases.values()) - {"not reported"})
        reason = "; ".join(used)

    return UncertaintyReport(
        model=key, source=effective_source, available=True,
        reason=reason, draws=pd.DataFrame(rows), provenance=bases,
    )


def summarise_k(k_values) -> Dict[str, float]:
    k = np.asarray(k_values, dtype=float)
    return {
        "k_mean": float(np.mean(k)),
        "k_median": float(np.median(k)),
        "k_sd": float(np.std(k, ddof=1)) if k.size > 1 else 0.0,
        "k_p2_5": float(np.percentile(k, 2.5)),
        "k_p97_5": float(np.percentile(k, 97.5)),
        "half_life_min": float(np.log(2.0) / np.mean(k)),
        "n": int(k.size),
    }


# ==========================================================================
# SENSITIVITY ANALYSES
# ==========================================================================

def objective_sensitivity(spec: CohortSpec, model_name: str, *, seed: int,
                          n_sim: int = 1000, **kwargs) -> pd.DataFrame:
    """Re-derive k under each objective function on the SAME cohort (EB-3).

    Using one cohort for all objectives is deliberate: any movement in k is then
    attributable to the loss function alone and not to a different random draw.
    """
    rng = np.random.default_rng(seed)
    cohort = sample_cohort(spec, n_sim, rng)

    rows = []
    for name in OBJECTIVES:
        r = calibrate_k(cohort, model_name, objective=name, seed=seed, **kwargs)
        rows.append({
            "model": r.model,
            "objective": name,
            "objective_label": OBJECTIVE_LABELS[name],
            "k": r.k,
            "half_life_min": r.half_life_min,
            "at_bound": r.at_lower_bound or r.at_upper_bound,
        })
    df = pd.DataFrame(rows)
    k_primary = float(df.loc[df["objective"] == "bland_altman", "k"].iloc[0])
    df["pct_change_vs_primary"] = 100.0 * (df["k"] - k_primary) / k_primary
    return df


def loa_weight_sensitivity(spec: CohortSpec, model_name: str, *, seed: int,
                           weights: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.5),
                           n_sim: int = 1000, **kwargs) -> pd.DataFrame:
    """How much does the arbitrary one-tenth LoA weight move k? (R1 p9 L44)"""
    rng = np.random.default_rng(seed)
    cohort = sample_cohort(spec, n_sim, rng)

    rows = []
    for w in weights:
        r = calibrate_k(cohort, model_name, objective="bland_altman",
                        objective_kwargs={"loa_weight": float(w)}, seed=seed, **kwargs)
        rows.append({"model": r.model, "loa_weight": float(w), "k": r.k,
                     "half_life_min": r.half_life_min})
    df = pd.DataFrame(rows)
    base = float(df.loc[np.isclose(df["loa_weight"], 0.1), "k"].iloc[0])
    df["pct_change_vs_0_1"] = 100.0 * (df["k"] - base) / base
    return df


def input_sensitivity(spec: CohortSpec, model_name: str, *, seed: int,
                      variation: float = 0.2, n_sim: int = 1000,
                      **kwargs) -> pd.DataFrame:
    """One-factor-at-a-time +/- ``variation`` on each input moment.

    Each perturbed configuration is calibrated on its own seeded cohort, and the
    seed is derived deterministically from the run seed and the parameter name,
    so the table is identical on every execution.
    """
    fields = ("ibw_mean", "ibw_sd", "t_to_mean", "t_to_sd", "t_on_mean", "t_on_sd")
    root = np.random.default_rng(seed)
    base_r = find_best_k(spec, model_name, seed=int(root.integers(0, 2 ** 32 - 1)),
                         n_sim=n_sim, **kwargs)

    rows = []
    for name in fields:
        for direction, factor in (("low", 1.0 - variation), ("high", 1.0 + variation)):
            modified = CohortSpec(**{**spec.as_dict(), name: getattr(spec, name) * factor})
            s = int(root.integers(0, 2 ** 32 - 1))
            r = find_best_k(modified, model_name, seed=s, n_sim=n_sim, **kwargs)
            rows.append({
                "model": r.model,
                "parameter": name,
                "direction": direction,
                "variation": variation,
                "modified_value": getattr(modified, name),
                "k": r.k,
                "k_base": base_r.k,
                "pct_change": 100.0 * (r.k - base_r.k) / base_r.k,
                "seed": s,
            })
    return pd.DataFrame(rows)


def prime_timing_sensitivity(spec: CohortSpec, model_name: str, *, seed: int,
                             n_sim: int = 1000, **kwargs) -> pd.DataFrame:
    """How much of the prime-timing mismatch is k absorbing? (EB-5)

    The reference models introduce prime heparin at CPB onset; the printed
    nomogram lumps bolus and prime into one load decaying from induction. If the
    two calibrated constants differ materially, the calibration is compensating
    for a structural mismatch, and departures from the assumed workflow will
    degrade the estimate in a way the agreement statistics do not reveal.
    """
    kwargs.pop("prime_timing", None)
    rows = []
    for timing in core.PRIME_TIMING_CHOICES:
        r = find_best_k(spec, model_name, seed=seed, n_sim=n_sim,
                        prime_timing=timing, **kwargs)
        rows.append({"model": r.model, "prime_timing": timing, "k": r.k,
                     "half_life_min": r.half_life_min})
    df = pd.DataFrame(rows)
    base = float(df.loc[df["prime_timing"] == "lumped_t0", "k"].iloc[0])
    df["pct_change_vs_lumped"] = 100.0 * (df["k"] - base) / base
    return df


def evaluation_mode_comparison(spec: CohortSpec, model_name: str, *, seed: int,
                               n_sim: int = 1000, n_timepoints: int = 13,
                               **kwargs) -> pd.DataFrame:
    """k calibrated to the reversal endpoint versus to the trajectory (EB-4)."""
    kwargs.pop("evaluation_mode", None)
    rows = []
    for mode in EVALUATION_MODES:
        r = find_best_k(spec, model_name, seed=seed, n_sim=n_sim,
                        evaluation_mode=mode, n_timepoints=n_timepoints, **kwargs)
        rows.append({
            "model": r.model,
            "evaluation_mode": mode,
            "description": EVALUATION_MODE_DESCRIPTIONS[mode],
            "n_timepoints_per_patient": r.n_timepoints,
            "k": r.k,
            "half_life_min": r.half_life_min,
        })
    df = pd.DataFrame(rows)
    base = float(df.loc[df["evaluation_mode"] == "reversal_endpoint", "k"].iloc[0])
    df["pct_change_vs_endpoint"] = 100.0 * (df["k"] - base) / base
    return df
