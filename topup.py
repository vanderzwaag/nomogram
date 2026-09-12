"""
Supplemental-bolus ("top-up") evaluation over a grid of dosing histories (EB-5).

The submitted analysis tested one scenario: a single 5,000 IU bolus at 60 min.
The reviewer asked for a grid of clinically plausible dosing histories, with
worst-case as well as average error reported. That is what this module produces.

What is being compared
----------------------
The bedside rule the printed nomogram asks the user to apply after a top-up is
an *axis reset*: read off the current residual, add the new bolus to it, and
start the time axis again from zero. Because the simplified model is a single
exponential, an axis reset is algebraically identical to superposing each dose
with its own elapsed time -- so the rule is exact *within* the simplified model,
and any divergence measured here is the divergence of the simplified model from
the reference model, not an error in the reset rule itself.

The reference side uses exact superposition, which is valid because every
reference model here is linear in dose.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

import nomogram_core as core
from agreement import DEFAULT_THRESHOLD_IU, difference

# Clinically plausible supplemental dosing histories.
TOPUP_SIZES_IU = (2500, 5000, 10000)
TOPUP_TIMES_MIN = (30, 60, 90, 120)

# Repeated-bolus histories, as (time, dose) pairs; sizes are substituted in.
REPEAT_PATTERNS: Dict[str, Sequence[float]] = {
    "single": (60.0,),
    "two_boluses": (60.0, 120.0),
    "three_boluses": (45.0, 90.0, 135.0),
}


def topup_scenarios(sizes: Iterable[float] = TOPUP_SIZES_IU,
                    times: Iterable[float] = TOPUP_TIMES_MIN,
                    patterns: Dict[str, Sequence[float]] | None = None) -> List[Dict]:
    """Enumerate the dosing histories to be tested."""
    scenarios = []
    for size in sizes:
        for t in times:
            scenarios.append({
                "pattern": "single",
                "label": f"{size:.0f} IU at {t:.0f} min",
                "size_iu": float(size),
                "topups": ((float(t), float(size)),),
            })
    for name, ts in (patterns or REPEAT_PATTERNS).items():
        if name == "single":
            continue
        for size in sizes:
            scenarios.append({
                "pattern": name,
                "label": f"{len(ts)} x {size:.0f} IU at {'/'.join(f'{t:.0f}' for t in ts)} min",
                "size_iu": float(size),
                "topups": tuple((float(t), float(size)) for t in ts),
            })
    return scenarios


def evaluate_scenario(model_name: str, k: float, *, heparin_bolus: float,
                      heparin_prime: float, weight_kg: float, time_to_cpb: float,
                      topups: Sequence[tuple], horizon_min: float = 240.0,
                      n_points: int = 49, prime_timing: str = "lumped_t0",
                      jia_allometric: bool = False,
                      threshold_iu: float = DEFAULT_THRESHOLD_IU) -> Dict[str, float]:
    """Divergence between nomogram and reference over the whole post-bolus window.

    Evaluation starts at the first supplemental bolus and runs to ``horizon_min``,
    so the reported worst case is the worst case a clinician could actually read,
    not just the value at one arbitrary follow-up time.
    """
    t_first = min(t for t, _ in topups)
    times = np.linspace(t_first, horizon_min, n_points)

    ref = np.array([
        core.reference_amount_with_topups(
            model_name, t, heparin_bolus, heparin_prime, weight_kg, time_to_cpb,
            topups, jia_allometric=jia_allometric)
        for t in times
    ])
    nom = np.array([
        core.simplified_amount_with_topups(
            t, heparin_bolus, heparin_prime, k, time_to_cpb, topups, prime_timing)
        for t in times
    ])

    diff = difference(ref, nom)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(ref > 0, 100.0 * diff / ref, np.nan)

    worst_idx = int(np.nanargmax(np.abs(diff)))
    return {
        "mean_diff_iu": float(np.mean(diff)),
        "mean_abs_diff_iu": float(np.mean(np.abs(diff))),
        "worst_abs_diff_iu": float(np.abs(diff[worst_idx])),
        "worst_diff_iu": float(diff[worst_idx]),
        "worst_diff_time_min": float(times[worst_idx]),
        "mean_pct_error": float(np.nanmean(pct)),
        "mean_abs_pct_error": float(np.nanmean(np.abs(pct))),
        "worst_abs_pct_error": float(np.nanmax(np.abs(pct))),
        "pct_of_window_within_threshold": float(100.0 * np.mean(np.abs(diff) <= threshold_iu)),
        "n_evaluation_points": int(times.size),
        "horizon_min": float(horizon_min),
    }


def topup_grid(model_ks: Dict[str, float], *, heparin_bolus: float,
               heparin_prime: float, weight_kg: float, time_to_cpb: float,
               sizes: Iterable[float] = TOPUP_SIZES_IU,
               times: Iterable[float] = TOPUP_TIMES_MIN,
               horizon_min: float = 240.0,
               prime_timing: str = "lumped_t0",
               jia_allometric: bool = False,
               threshold_iu: float = DEFAULT_THRESHOLD_IU) -> pd.DataFrame:
    """Full size x timing x repeat grid, for every model, with its own calibrated k."""
    scenarios = topup_scenarios(sizes, times)
    rows = []
    for model, k in model_ks.items():
        for sc in scenarios:
            stats = evaluate_scenario(
                model, k, heparin_bolus=heparin_bolus, heparin_prime=heparin_prime,
                weight_kg=weight_kg, time_to_cpb=time_to_cpb, topups=sc["topups"],
                horizon_min=horizon_min, prime_timing=prime_timing,
                jia_allometric=jia_allometric, threshold_iu=threshold_iu,
            )
            rows.append({
                "model": core.canonical_model_name(model),
                "k": k,
                "pattern": sc["pattern"],
                "scenario": sc["label"],
                "size_iu": sc["size_iu"],
                "first_topup_min": min(t for t, _ in sc["topups"]),
                "n_topups": len(sc["topups"]),
                "total_topup_iu": sum(d for _, d in sc["topups"]),
                **stats,
            })
    return pd.DataFrame(rows)


def worst_case_summary(grid: pd.DataFrame) -> pd.DataFrame:
    """Per-model mean and worst case across the whole grid -- the Table 2 replacement."""
    g = grid.groupby("model")
    out = pd.DataFrame({
        "k": g["k"].first(),
        "n_scenarios": g.size(),
        "mean_abs_diff_iu": g["mean_abs_diff_iu"].mean(),
        "mean_abs_pct_error": g["mean_abs_pct_error"].mean(),
        "worst_abs_diff_iu": g["worst_abs_diff_iu"].max(),
        "worst_abs_pct_error": g["worst_abs_pct_error"].max(),
    })
    worst_rows = grid.loc[grid.groupby("model")["worst_abs_diff_iu"].idxmax()]
    out["worst_case_scenario"] = worst_rows.set_index("model")["scenario"]
    out["worst_case_time_min"] = worst_rows.set_index("model")["worst_diff_time_min"]
    return out.reset_index().sort_values("worst_abs_pct_error", ascending=False)
