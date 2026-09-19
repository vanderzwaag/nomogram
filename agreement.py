"""
Agreement statistics for the heparin decay nomogram (EB-3).

A single enforced sign convention, a formal test for proportional bias,
relative as well as absolute error, performance stratified by elapsed time and
by residual load, and coverage against a stated clinically relevant threshold.

Sign convention
---------------
One convention is defined here and used by every figure, table and sentence:

    difference = reference model - nomogram

A POSITIVE difference therefore means the nomogram UNDER-estimates residual
heparin (and so would lead to under-dosing of protamine). Nothing in this
repository is permitted to compute the difference the other way round: two
orderings in use at once is how one analysis came to be reported with two
different signs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats

SIGN_CONVENTION = "reference_minus_nomogram"
SIGN_CONVENTION_LABEL = "Difference = reference model - nomogram (positive = nomogram under-estimates)"

# Clinically relevant error threshold (EB-3).
#
# Protamine is given in whole syringe increments. At a 1 mg : 100 IU reversal
# ratio, the smallest increment an anaesthetist realistically alters -- 10 mg of
# protamine -- corresponds to 1000 IU of heparin. An error smaller than that
# cannot change the dose actually drawn up, so it is adopted as the threshold
# below which a difference is described as not dose-relevant under the
# simulated assumptions. It is a modelling threshold, not a clinical claim, and
# is configurable throughout the pipeline.
DEFAULT_THRESHOLD_IU = 1000.0
DEFAULT_THRESHOLD_PCT = 10.0
DEFAULT_PROTAMINE_RATIO_MG_PER_100IU = 1.0


def difference(reference, nomogram):
    """The one permitted difference ordering."""
    return np.asarray(reference, dtype=float) - np.asarray(nomogram, dtype=float)


# ==========================================================================
# BLAND-ALTMAN
# ==========================================================================

@dataclass
class BlandAltmanResult:
    n: int
    bias: float
    bias_ci_low: float
    bias_ci_high: float
    sd_diff: float
    loa_low: float
    loa_high: float
    loa_low_ci_low: float
    loa_low_ci_high: float
    loa_high_ci_low: float
    loa_high_ci_high: float
    sign_convention: str = SIGN_CONVENTION

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def bland_altman(reference, nomogram, confidence: float = 0.95) -> BlandAltmanResult:
    """Bias and 95% limits of agreement, each with its own confidence interval.

    The confidence intervals on the limits use Bland & Altman's standard error,
    SE(LoA) = sqrt(3) * SD / sqrt(n); they are what tells a reader whether two
    reported limits actually differ or merely reflect sampling noise.
    """
    diff = difference(reference, nomogram)
    n = diff.size
    if n < 3:
        raise ValueError("At least three paired observations are required")

    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    tcrit = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))

    se_bias = sd / np.sqrt(n)
    se_loa = np.sqrt(3.0) * sd / np.sqrt(n)
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    return BlandAltmanResult(
        n=n,
        bias=bias,
        bias_ci_low=bias - tcrit * se_bias,
        bias_ci_high=bias + tcrit * se_bias,
        sd_diff=sd,
        loa_low=loa_low,
        loa_high=loa_high,
        loa_low_ci_low=loa_low - tcrit * se_loa,
        loa_low_ci_high=loa_low + tcrit * se_loa,
        loa_high_ci_low=loa_high - tcrit * se_loa,
        loa_high_ci_high=loa_high + tcrit * se_loa,
    )


def bland_altman_stats(reference, nomogram):
    """Backwards-compatible three-tuple (bias, loa_low, loa_high)."""
    r = bland_altman(reference, nomogram)
    return r.bias, r.loa_low, r.loa_high


# ==========================================================================
# PROPORTIONAL BIAS
# ==========================================================================

@dataclass
class ProportionalBiasResult:
    slope: float
    slope_ci_low: float
    slope_ci_high: float
    slope_p: float
    intercept: float
    r_squared: float
    significant: bool
    # Difference predicted by the regression at the low and high ends of the
    # measured range -- the practical size of the proportional bias.
    mean_at_p5: float
    mean_at_p95: float
    predicted_diff_at_p5: float
    predicted_diff_at_p95: float
    sign_convention: str = SIGN_CONVENTION

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def proportional_bias(reference, nomogram, alpha: float = 0.05) -> ProportionalBiasResult:
    """Regress the difference on the mean (the formal test EB-3 asks for).

    A mean bias near zero is uninformative when the difference varies
    systematically with the magnitude being measured: positive and negative
    errors simply cancel. The slope below is that systematic component, and its
    p-value is the test of whether it is distinguishable from zero.
    """
    ref = np.asarray(reference, dtype=float)
    nom = np.asarray(nomogram, dtype=float)
    diff = difference(ref, nom)
    mean_vals = (ref + nom) / 2.0

    fit = stats.linregress(mean_vals, diff)
    n = diff.size
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 2))

    p5, p95 = np.percentile(mean_vals, [5, 95])

    return ProportionalBiasResult(
        slope=float(fit.slope),
        slope_ci_low=float(fit.slope - tcrit * fit.stderr),
        slope_ci_high=float(fit.slope + tcrit * fit.stderr),
        slope_p=float(fit.pvalue),
        intercept=float(fit.intercept),
        r_squared=float(fit.rvalue ** 2),
        significant=bool(fit.pvalue < alpha),
        mean_at_p5=float(p5),
        mean_at_p95=float(p95),
        predicted_diff_at_p5=float(fit.intercept + fit.slope * p5),
        predicted_diff_at_p95=float(fit.intercept + fit.slope * p95),
    )


# ==========================================================================
# RELATIVE ERROR
# ==========================================================================

@dataclass
class RelativeErrorResult:
    n: int
    mean_pct_error: float          # MPE, signed -- the relative counterpart of bias
    mean_abs_pct_error: float      # MAPE
    median_abs_pct_error: float
    pct_error_p2_5: float
    pct_error_p97_5: float
    max_abs_pct_error: float
    rmse_iu: float
    mae_iu: float
    max_abs_error_iu: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def relative_errors(reference, nomogram) -> RelativeErrorResult:
    """Relative (%) errors alongside absolute (IU) errors.

    Percentage error is expressed relative to the reference-model value, and
    carries the same sign convention as everything else: positive means the
    nomogram read low.
    """
    ref = np.asarray(reference, dtype=float)
    nom = np.asarray(nomogram, dtype=float)
    diff = difference(ref, nom)

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(ref != 0, 100.0 * diff / ref, np.nan)
    pct = pct[np.isfinite(pct)]

    return RelativeErrorResult(
        n=int(diff.size),
        mean_pct_error=float(np.mean(pct)),
        mean_abs_pct_error=float(np.mean(np.abs(pct))),
        median_abs_pct_error=float(np.median(np.abs(pct))),
        pct_error_p2_5=float(np.percentile(pct, 2.5)),
        pct_error_p97_5=float(np.percentile(pct, 97.5)),
        max_abs_pct_error=float(np.max(np.abs(pct))),
        rmse_iu=float(np.sqrt(np.mean(diff ** 2))),
        mae_iu=float(np.mean(np.abs(diff))),
        max_abs_error_iu=float(np.max(np.abs(diff))),
    )


def threshold_coverage(reference, nomogram,
                       threshold_iu: float = DEFAULT_THRESHOLD_IU,
                       threshold_pct: float = DEFAULT_THRESHOLD_PCT) -> Dict[str, float]:
    """Proportion of patients inside the stated clinically relevant threshold."""
    ref = np.asarray(reference, dtype=float)
    diff = np.abs(difference(ref, nomogram))
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(ref != 0, 100.0 * diff / ref, np.inf)
    return {
        "threshold_iu": float(threshold_iu),
        "threshold_pct": float(threshold_pct),
        "pct_within_threshold_iu": float(100.0 * np.mean(diff <= threshold_iu)),
        "pct_within_threshold_pct": float(100.0 * np.mean(pct <= threshold_pct)),
        "protamine_mg_equivalent_of_threshold": float(
            threshold_iu / 100.0 * DEFAULT_PROTAMINE_RATIO_MG_PER_100IU
        ),
    }


# ==========================================================================
# STRATIFIED PERFORMANCE
# ==========================================================================

def stratify(reference, nomogram, by, by_name: str, n_bins: int = 4,
             edges: Sequence[float] | None = None,
             threshold_iu: float = DEFAULT_THRESHOLD_IU) -> pd.DataFrame:
    """Agreement within strata of ``by`` (elapsed time, residual load, ...).

    Strata are quantile-based by default, so each contains a comparable number
    of patients; pass explicit ``edges`` for clinically chosen cut-points.
    """
    ref = np.asarray(reference, dtype=float)
    nom = np.asarray(nomogram, dtype=float)
    by = np.asarray(by, dtype=float)
    diff = difference(ref, nom)

    if edges is None:
        qs = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.percentile(by, qs))
    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        raise ValueError(f"Cannot stratify by {by_name}: it takes a single value")

    idx = np.clip(np.digitize(by, edges[1:-1], right=False), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() < 3:
            continue
        d = diff[sel]
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(ref[sel] != 0, 100.0 * d / ref[sel], np.nan)
        rows.append({
            "stratified_by": by_name,
            "stratum": f"{edges[b]:.4g} to {edges[b + 1]:.4g}",
            "stratum_low": edges[b],
            "stratum_high": edges[b + 1],
            "n": int(sel.sum()),
            "bias_iu": float(np.mean(d)),
            "sd_iu": float(np.std(d, ddof=1)),
            "loa_low_iu": float(np.mean(d) - 1.96 * np.std(d, ddof=1)),
            "loa_high_iu": float(np.mean(d) + 1.96 * np.std(d, ddof=1)),
            "mean_pct_error": float(np.nanmean(pct)),
            "mean_abs_pct_error": float(np.nanmean(np.abs(pct))),
            "max_abs_error_iu": float(np.max(np.abs(d))),
            "pct_within_threshold_iu": float(100.0 * np.mean(np.abs(d) <= threshold_iu)),
        })
    return pd.DataFrame(rows)


def stratified_report(cohort: pd.DataFrame, reference, nomogram,
                      threshold_iu: float = DEFAULT_THRESHOLD_IU) -> pd.DataFrame:
    """Stratified performance by elapsed time and by residual load (EB-3).

    ``cohort`` must carry ``elapsed_time`` (min); residual load is taken from
    the reference-model value, which is the quantity a clinician would be
    reading off the nomogram.
    """
    frames = [
        stratify(reference, nomogram, cohort["elapsed_time"].to_numpy(),
                 "elapsed_time_min", threshold_iu=threshold_iu),
        stratify(reference, nomogram, np.asarray(reference, dtype=float),
                 "residual_load_iu", threshold_iu=threshold_iu),
    ]
    if "time_on_cpb" in cohort:
        frames.append(stratify(reference, nomogram, cohort["time_on_cpb"].to_numpy(),
                               "time_on_cpb_min", threshold_iu=threshold_iu))
    if "ibw" in cohort and cohort["ibw"].nunique() > 1:
        frames.append(stratify(reference, nomogram, cohort["ibw"].to_numpy(),
                               "ibw_kg", threshold_iu=threshold_iu))
    return pd.concat(frames, ignore_index=True)


# ==========================================================================
# ONE-CALL SUMMARY
# ==========================================================================

def full_agreement_report(cohort: pd.DataFrame, reference, nomogram,
                          threshold_iu: float = DEFAULT_THRESHOLD_IU,
                          threshold_pct: float = DEFAULT_THRESHOLD_PCT):
    """Every agreement quantity the revision needs, for one model and cohort."""
    ba = bland_altman(reference, nomogram)
    pb = proportional_bias(reference, nomogram)
    re_ = relative_errors(reference, nomogram)
    cov = threshold_coverage(reference, nomogram, threshold_iu, threshold_pct)

    summary = {**ba.as_dict(), **re_.as_dict(), **cov}
    summary.update({f"prop_bias_{k}": v for k, v in pb.as_dict().items()
                    if k != "sign_convention"})
    # R-squared is retained only as a descriptor of the scatter, never as an
    # agreement metric (EB-3).
    ref_arr = np.asarray(reference, dtype=float)
    nom_arr = np.asarray(nomogram, dtype=float)
    summary["descriptive_r_squared"] = float(np.corrcoef(ref_arr, nom_arr)[0, 1] ** 2)

    return summary, stratified_report(cohort, reference, nomogram, threshold_iu)
