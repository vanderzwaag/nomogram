import hashlib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import itertools
import pandas as pd

APP_VERSION = "v2.0.0"

# Every simulation in this module takes an explicit seed; without one, no two
# runs produce the same cohort, k or agreement statistics (EB-3, R1 Figure 2).
DEFAULT_SEED = 20260912

# ==========================================
# MODEL DEFINITIONS
# ==========================================
#
# The pharmacokinetics now live in nomogram_core, which imports neither
# Streamlit nor PyNomo, so that every published number can be regenerated
# head-lessly by run_analysis.py and checked by the test suite (EB-6). The
# names below are re-exported unchanged, so existing callers are unaffected.

from nomogram_core import (                                     # noqa: E402
    MODEL_NAMES,
    MODEL_PARAMETERS,
    DISPLAY_NAMES,
    PAEDIATRIC_MODELS,
    PRIME_TIMING_CHOICES,
    canonical_model_name,
    delavenne_dose,
    delavenne_response,
    get_delavenne_params,
    get_jia_params,
    get_lanoiselee_params,
    get_reference_dose,
    jia_dose,
    jia_response,
    lanoiselee_dose,
    lanoiselee_response,
    meesters_dose,
    prodose2_dose,
    prodose_dose,
    reference_amount,
    reference_amount_array,
    simplified_amount,
    simplified_amount_array,
    simplified_model_dose,
)

# Agreement statistics, including the proportional-bias regression, relative
# errors and stratified performance the revision requires (EB-3).
from agreement import (                                         # noqa: E402
    SIGN_CONVENTION,
    SIGN_CONVENTION_LABEL,
    DEFAULT_THRESHOLD_IU,
    bland_altman,
    bland_altman_stats,
    difference,
    full_agreement_report,
    proportional_bias,
    relative_errors,
    threshold_coverage,
)

# ==========================================
# STATS & OPTIMIZATION
# ==========================================
#
# The calibration machinery lives in calibration.py, which is seeded, has a
# parameterised objective function and can target either the reversal endpoint
# or the decay trajectory. The wrappers below keep this module's original call
# signatures so the dashboard is unchanged, while adding an explicit `seed`.

from calibration import (                                       # noqa: E402
    K_BOUNDS,
    K_BOUNDS_NOTE,
    OBJECTIVES,
    OBJECTIVE_LABELS,
    EVALUATION_MODES,
    CohortSpec,
    calibrate_k,
    evaluation_times,
    input_sensitivity,
    monte_carlo_precision,
    nomogram_values,
    objective_sensitivity,
    parameter_uncertainty,
    reference_values,
    sample_cohort,
    summarise_k,
)
from calibration import find_best_k as _calibrate_from_spec     # noqa: E402


def bland_altman_score(ref, test, loa_weight=0.1):
    """The primary objective: |bias| + 0.1 x LoA width.

    Kept here for backwards compatibility; the weight is now exposed because
    the one-tenth value is arbitrary and its influence on k is reported as a
    sensitivity analysis rather than defended (EB-3, R1 p9 L44).
    """
    bias, loa_low, loa_high = bland_altman_stats(ref, test)
    return abs(bias) + loa_weight * (loa_high - loa_low)


def _spec(initial_dose_per_kg, prime_heparin, ibw_mean, ibw_sd,
          t_to_mean, t_to_sd, t_on_mean, t_on_sd, population="adult"):
    return CohortSpec(
        name=f"{initial_dose_per_kg:g}IUkg_{ibw_mean:g}kg_{t_to_mean:g}_{t_on_mean:g}",
        dose_per_kg=initial_dose_per_kg, prime_heparin=prime_heparin,
        ibw_mean=ibw_mean, ibw_sd=ibw_sd,
        t_to_mean=t_to_mean, t_to_sd=t_to_sd,
        t_on_mean=t_on_mean, t_on_sd=t_on_sd, population=population,
    )


@st.cache_data(show_spinner=False)
def find_best_k(initial_dose_per_kg, prime_heparin, ibw_mean, ibw_sd,
                t_to_mean, t_to_sd, t_on_mean, t_on_sd,
                model_name, n_sim=1000, seed=DEFAULT_SEED,
                objective="bland_altman", evaluation_mode="reversal_endpoint",
                prime_timing="lumped_t0"):
    """Calibrate k against one seeded synthetic cohort.

    `seed` is part of the Streamlit cache key, so a given configuration always
    returns the same constant within and across sessions.
    """
    spec = _spec(initial_dose_per_kg, prime_heparin, ibw_mean, ibw_sd,
                 t_to_mean, t_to_sd, t_on_mean, t_on_sd)
    return _calibrate_from_spec(spec, model_name, seed=seed, n_sim=n_sim,
                                objective=objective,
                                evaluation_mode=evaluation_mode,
                                prime_timing=prime_timing).k


@st.cache_data(show_spinner=False)
def monte_carlo_k_distribution(initial_dose_per_kg, prime_heparin, ibw_mean, ibw_sd,
                               t_to_mean, t_to_sd, t_on_mean, t_on_sd,
                               model_name, n_boot=1000, n_sim=1000,
                               seed=DEFAULT_SEED, **kwargs):
    """Repeat the calibration on fresh cohorts from the SAME fixed distributions.

    Not a bootstrap: no dataset is resampled, a new synthetic cohort is
    simulated each time. What it measures is Monte Carlo sampling precision,
    and that is how it must be labelled wherever it appears (EB-2). It does not
    quantify uncertainty in the published PK parameters, in the institutional
    input estimates, in model selection, or in an individual patient's
    prediction -- for the first of those, see `parameter_uncertainty`.
    """
    spec = _spec(initial_dose_per_kg, prime_heparin, ibw_mean, ibw_sd,
                 t_to_mean, t_to_sd, t_on_mean, t_on_sd)
    df = monte_carlo_precision(spec, model_name, seed=seed,
                               n_replicates=n_boot, n_sim=n_sim, **kwargs)
    return df["k"].to_numpy()


# Retained under the old name so existing callers keep working; the label is
# wrong and should not be used in the manuscript.
bootstrap_k_distribution = monte_carlo_k_distribution


def summarize_k_distribution(k_values):
    """Mean and 2.5th/97.5th percentiles of the replicate distribution.

    The interval is a Monte Carlo sampling-precision interval, not a predicted
    margin of error (EB-2).
    """
    s = summarise_k(k_values)
    return s["k_mean"], s["k_p2_5"], s["k_p97_5"]
# ==========================================
# NOMOGRAM RENDERING
# ==========================================
#
# The chart geometry lives in nomogram_render and is shared by the printed PDF
# and the interactive chart, so the two cannot disagree. The previous
# implementation drew the PDF with PyNomo (which pulls in PyX, LaTeX and
# Ghostscript) from one decay constant, while the dashboard drew its own
# matplotlib chart from a different one -- the same "several constants in
# circulation" defect that EB-3 was about, surviving in the drawing layer.

from k_table_io import table_path, write_k_table               # noqa: E402
from nomogram_render import (                                   # noqa: E402,F401
    NomogramGeometry,   # re-exported for callers of this module
    build_geometry,
    draw_nomogram,      # re-exported
    footer_for,
    render_pdf,
)


def nomogram_geometry_for(k, initial_dose_per_kg, prime_heparin, ibw_mean,
                          t_max=120.0, span=0.25):
    """Size a chart around one institution's typical total load.

    The dose axis spans ``span`` either side of the baseline load. Narrower is
    better for legibility as well as coverage: the residual axis must cover the
    dose range times the decay over the whole time range, so the elapsed-time
    axis gets the fraction (dr - dd) / 2dr of the sheet, where dd and dr are the
    decades each outer axis spans. Widening the dose range shrinks the time
    axis. +/-25% of the baseline load covers the patients an institution
    actually sees while leaving the time axis readable.
    """
    baseline = initial_dose_per_kg * ibw_mean + prime_heparin
    return build_geometry(k, d0_min=baseline * (1.0 - span),
                          d0_max=baseline * (1.0 + span), t_max=t_max)


def build_nomogram(k_value, output_filename="heparin_dose_decay_nomogram.pdf",
                   d0_min=15000.0, d0_max=50000.0, t_max=120.0,
                   footer_lines=(), title=None, patient=None):
    """Render a nomogram PDF for one decay constant.

    Kept under its original name so existing callers are unaffected; it no
    longer requires PyNomo, PyX, LaTeX or Ghostscript.
    """
    geom = build_geometry(k_value, d0_min=d0_min, d0_max=d0_max, t_max=t_max)
    return render_pdf(geom, output_filename, footer_lines=footer_lines,
                      title=title, patient=patient)


# ==========================================
# PLOTTING & UTILS
# ==========================================

def bland_altman_plot(ref, test, model_label="Reference", threshold_iu=DEFAULT_THRESHOLD_IU):
    """Bland-Altman plot with the proportional-bias regression drawn on it (EB-3).

    The axis label states the sign convention explicitly, so the figure cannot
    be read against the direction the text quotes.
    """
    ref = np.asarray(ref, dtype=float)
    test = np.asarray(test, dtype=float)
    diff = difference(ref, test)
    mean_vals = (ref + test) / 2.0

    ba = bland_altman(ref, test)
    pb = proportional_bias(ref, test)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(mean_vals, diff, alpha=0.35, s=14, color="steelblue",
               edgecolors="none")
    ax.axhline(ba.bias, color="red", linestyle="--", lw=1.4,
               label=f"Bias = {ba.bias:.2f} IU")
    ax.axhline(ba.loa_low, color="green", linestyle="--", lw=1.2,
               label=f"95% LoA = {ba.loa_low:.1f} to {ba.loa_high:.1f} IU")
    ax.axhline(ba.loa_high, color="green", linestyle="--", lw=1.2)
    ax.axhline(0.0, color="grey", lw=0.6)

    xs = np.linspace(mean_vals.min(), mean_vals.max(), 100)
    ax.plot(xs, pb.intercept + pb.slope * xs, color="darkorange", lw=1.6,
            label=(f"Proportional bias: slope {pb.slope:+.4f} "
                   f"(p = {pb.slope_p:.2g})"))

    if threshold_iu:
        ax.axhspan(-threshold_iu, threshold_iu, color="grey", alpha=0.08,
                   label=f"+/-{threshold_iu:,.0f} IU threshold")

    ax.set_xlabel(f"Mean of {model_label} and nomogram (IU)")
    ax.set_ylabel(f"Difference, {model_label} minus nomogram (IU)")
    ax.set_title(f"{model_label} versus simplified nomogram\n"
                 "positive = nomogram under-estimates residual heparin",
                 fontsize=10)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def plot_k_posterior(k_values, label="Monte Carlo sampling precision"):
    """Distribution of k across replicate cohorts.

    Deliberately NOT called a posterior: no prior is specified and no Bayesian
    inference is performed. It is the spread of the point estimate across
    repeated simulated cohorts (EB-2).
    """
    from scipy.stats import gaussian_kde

    k_values = np.asarray(k_values, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        kde = gaussian_kde(k_values)
        xs = np.linspace(k_values.min() * 0.98, k_values.max() * 1.02, 250)
        ax.plot(xs, kde(xs), color="darkblue", lw=2)
    except Exception:
        pass
    ax.hist(k_values, bins=25, density=True, alpha=0.3, color="steelblue")
    lo, hi = np.percentile(k_values, [2.5, 97.5])
    ax.axvline(k_values.mean(), color="red", ls="--", lw=1.2,
               label=f"mean {k_values.mean():.5f}")
    ax.axvline(lo, color="green", ls=":", lw=1.0)
    ax.axvline(hi, color="green", ls=":", lw=1.0,
               label=f"2.5-97.5% {lo:.5f} to {hi:.5f}")
    ax.set_title(f"Distribution of k across replicate cohorts\n({label})", fontsize=10)
    ax.set_xlabel("k (/min)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
@st.cache_resource(show_spinner=False)
def plot_tornado(df):
    params = []
    low_vals = []
    high_vals = []
    spreads = []

    # 1. Extract values
    for param in df["Parameter"].unique():
        subset = df[df["Parameter"] == param]
        k_low = subset[subset["Direction"] == "low"]["k_best"].values[0]
        k_high = subset[subset["Direction"] == "high"]["k_best"].values[0]
        
        params.append(param)
        low_vals.append(k_low)
        high_vals.append(k_high)
        spreads.append(abs(k_high - k_low))

    # 2. Bind into a DataFrame for guaranteed aligned sorting
    plot_df = pd.DataFrame({
        "Parameter": params,
        "k_low": low_vals,
        "k_high": high_vals,
        "spread": spreads
    })

    # 3. Sort ascending (so the largest spread is plotted last, at the top)
    plot_df = plot_df.sort_values(by="spread", ascending=True).reset_index(drop=True)

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(plot_df))
    
    widths = np.abs(plot_df["k_high"] - plot_df["k_low"])
    lefts = np.minimum(plot_df["k_low"], plot_df["k_high"])
    
    ax.barh(y_pos, widths, left=lefts, height=0.6, color="steelblue", alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["Parameter"])
    ax.set_xlabel("k value")
    ax.set_title("Tornado Plot")
    
    return fig


def clinical_summary(k_best, agreement, ibw_mean, model_name,
                     mc_interval=None, threshold_iu=DEFAULT_THRESHOLD_IU):
    """Modelling summary for the dashboard.

    Deliberately free of clinical claims: the pipeline reports the
    pharmacokinetic central-compartment amount only, and does not model the
    pharmacodynamic (anti-Xa or ACT) layer, peripheral-compartment heparin,
    antithrombin, protamine pharmacology or rebound (EB-1). Errors are
    described relative to one protamine dosing increment under the simulated
    assumptions, not as clinically negligible (R1 p13 L9).
    """
    interval = ""
    if mc_interval is not None:
        interval = (f"\n    - Monte Carlo sampling-precision interval: "
                    f"**{mc_interval[0]:.5f} to {mc_interval[1]:.5f}** "
                    f"(not a predictive or parameter-uncertainty interval)")

    prop = ("a statistically detectable proportional bias"
            if agreement["prop_bias_significant"] else
            "no statistically detectable proportional bias")

    return f"""
    **Approximation summary, {model_name} as the reference model**

    - Calibrated decay constant: **k = {k_best:.5f} /min**
      (apparent half-life {np.log(2.0) / k_best:.0f} min).{interval}

    - Difference convention: {SIGN_CONVENTION_LABEL}.

    - Mean bias **{agreement['bias']:+.1f} IU**
      (95% CI {agreement['bias_ci_low']:+.1f} to {agreement['bias_ci_high']:+.1f}),
      95% limits of agreement **{agreement['loa_low']:+.1f} to {agreement['loa_high']:+.1f} IU**.

    - Relative error: mean **{agreement['mean_pct_error']:+.2f}%**, mean absolute
      **{agreement['mean_abs_pct_error']:.2f}%**, largest absolute
      **{agreement['max_abs_pct_error']:.1f}%**.

    - Bland-Altman regression of difference on mean: slope
      **{agreement['prop_bias_slope']:+.4f}** (p = {agreement['prop_bias_slope_p']:.2g}),
      i.e. {prop}.

    - **{agreement['pct_within_threshold_iu']:.1f}%** of simulated patients fall within
      {threshold_iu:,.0f} IU, one {threshold_iu / 100:.0f} mg protamine increment at a
      1 mg : 100 IU ratio.

    - Population IBW mean: **{ibw_mean:.0f} kg**. The comparison cohort is a fresh
      draw from the same assumed distributions, i.e. an internal resample, and is
      not an external validation cohort.
    """

# ==========================================
# SENSITIVITY ANALYSIS
# ==========================================

@st.cache_data(show_spinner=False)
def sensitivity_analysis(initial_dose_per_kg, prime_heparin,
                         ibw_mean, ibw_sd, t_to_mean, t_to_sd,
                         t_on_mean, t_on_sd, model_name="PRODOSE",
                         variation=0.2, n_sim=500, seed=DEFAULT_SEED,
                         **kwargs):
    """One-factor-at-a-time sensitivity of k to each input moment.

    Each perturbed configuration draws its own cohort from a seed derived
    deterministically from `seed`, so the table is identical on every run. The
    default variation is +/-20%, which the manuscript now reports in the main
    text rather than the supplement (EB-8).
    """
    spec = _spec(initial_dose_per_kg, prime_heparin, ibw_mean, ibw_sd,
                 t_to_mean, t_to_sd, t_on_mean, t_on_sd)
    df = input_sensitivity(spec, model_name, seed=seed, variation=variation,
                           n_sim=n_sim, **kwargs)
    # Column names the existing dashboard expects.
    return df.rename(columns={"parameter": "Parameter", "direction": "Direction",
                              "modified_value": "Modified value", "k": "k_best"})


@st.cache_resource(show_spinner=False)
def plot_sensitivity(df):
    """How k moves when each input is varied by +/- the stated fraction."""
    fig, ax = plt.subplots(figsize=(7, 4))
    variation = df["variation"].iloc[0] if "variation" in df else 0.2
    for param in df["Parameter"].unique():
        subset = df[df["Parameter"] == param].sort_values("Direction")
        ax.plot([f"Low (-{variation:.0%})", f"High (+{variation:.0%})"],
                subset["k_best"], marker="o", label=param)
    ax.set_title("Sensitivity of k to the simulation inputs")
    ax.set_ylabel("Calibrated k (/min)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


# ==========================================
# MAIN RUNNER
# ==========================================

def run_nomogram(initial_dose_per_kg, t_to_mean, prime_heparin,
                 ibw_mean, ibw_sd, t_to_sd, t_on_mean, t_on_sd,
                 model_name, seed=DEFAULT_SEED, n_sim=1000,
                 n_replicates=500, variation=0.2,
                 evaluation_mode="reversal_endpoint",
                 prime_timing="lumped_t0",
                 threshold_iu=DEFAULT_THRESHOLD_IU):
    """Full diagnostic run for one reference model.

    Two rules hold the numbers together (EB-3):

    1. **One k, used everywhere.** The reported constant is the mean over the
       replicate cohorts, and that single value is used for the statistics, the
       plots, the PDF and its footer alike. Three constants in circulation for
       one configuration is what made the reported numbers irreconcilable.

    2. **One test cohort, drawn from a declared seed.** The objective drives the
       residual bias to nearly zero, so an undeclared cohort leaves a bias of a
       few IU whose sign changes from run to run.
    """
    spec = _spec(initial_dose_per_kg, prime_heparin, ibw_mean, ibw_sd,
                 t_to_mean, t_to_sd, t_on_mean, t_on_sd)
    calib = dict(evaluation_mode=evaluation_mode, prime_timing=prime_timing)

    # 1. Calibrate, and take the replicate mean as THE decay constant.
    k_draws = monte_carlo_precision(spec, model_name, seed=seed,
                                    n_replicates=n_replicates, n_sim=n_sim,
                                    **calib)["k"].to_numpy()
    k_summary = summarise_k(k_draws)
    k_best = k_summary["k_mean"]
    ci_low, ci_high = k_summary["k_p2_5"], k_summary["k_p97_5"]

    # 2. Sensitivity analyses.
    sens_df = sensitivity_analysis(initial_dose_per_kg, prime_heparin,
                                   ibw_mean, ibw_sd, t_to_mean, t_to_sd,
                                   t_on_mean, t_on_sd, model_name=model_name,
                                   variation=variation, n_sim=max(500, n_sim // 2),
                                   seed=seed, **calib)
    sens_fig = plot_sensitivity(sens_df)
    tornado_fig = plot_tornado(sens_df)
    obj_df = objective_sensitivity(spec, model_name, seed=seed, n_sim=n_sim, **calib)

    # 3. Evaluate on one declared internal test cohort.
    test_rng = np.random.default_rng(seed + 1)
    cohort = sample_cohort(spec, n_sim, test_rng)
    times = evaluation_times(cohort, "reversal_endpoint")
    ref_doses = reference_values(cohort, model_name, times).ravel()
    test_doses = nomogram_values(cohort, times, k_best, prime_timing).ravel()

    agreement, strata_df = full_agreement_report(cohort, ref_doses, test_doses,
                                                 threshold_iu=threshold_iu)
    bias, loa_low, loa_high = (agreement["bias"], agreement["loa_low"],
                               agreement["loa_high"])

    sim_df = pd.DataFrame({
        "IBW": cohort["ibw"], "TimeTo": cohort["time_to_cpb"],
        "TimeOn": cohort["time_on_cpb"], "Elapsed": cohort["elapsed_time"],
        "Ref_Dose": ref_doses, "Simp_Dose": test_doses,
        "Diff_Ref_minus_Simp": difference(ref_doses, test_doses),
    })

    diagnostics = {
        "k": k_best,
        "k sampling interval": f"{ci_low:.5f} to {ci_high:.5f}",
        "RMSE (IU)": agreement["rmse_iu"],
        "MAE (IU)": agreement["mae_iu"],
        "Max abs error (IU)": agreement["max_abs_error_iu"],
        "Mean % error": agreement["mean_pct_error"],
        "MAPE (%)": agreement["mean_abs_pct_error"],
        "Max abs % error": agreement["max_abs_pct_error"],
        "Proportional bias slope": agreement["prop_bias_slope"],
        "Proportional bias p": agreement["prop_bias_slope_p"],
        f"% within {threshold_iu:,.0f} IU": agreement["pct_within_threshold_iu"],
        "% within 10%": agreement["pct_within_threshold_pct"],
        # Retained as a descriptor of the scatter only; R-squared is not a
        # measure of agreement (EB-3).
        "R² (descriptive only)": agreement["descriptive_r_squared"],
    }

    # 4. Figures and the printed nomogram -- all using the same k_best, and now
    #    the same geometry as the interactive chart.
    spec_text = (f"Initial heparin {initial_dose_per_kg} IU/kg, prime {prime_heparin} IU, "
                 f"time to CPB {t_to_mean}+/-{t_to_sd} min, "
                 f"time on CPB {t_on_mean}+/-{t_on_sd} min, "
                 f"IBW {ibw_mean}+/-{ibw_sd} kg.  "
                 f"Calibration endpoint {evaluation_mode}; prime timing {prime_timing}.")
    geom = nomogram_geometry_for(k_best, initial_dose_per_kg, prime_heparin, ibw_mean)
    pdf_path = "nomogram_final.pdf"
    render_pdf(geom, pdf_path,
               title=f"Heparin decay nomogram - {model_name}",
               footer_lines=footer_for(model_name, k_best, spec_text, seed=seed,
                                       interval=(ci_low, ci_high),
                                       version=APP_VERSION))

    fig_ba = bland_altman_plot(ref_doses, test_doses, model_label=model_name,
                               threshold_iu=threshold_iu)
    k_post_fig = plot_k_posterior(k_draws)

    scatter_fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(ref_doses, test_doses, c=cohort["elapsed_time"],
                    cmap="viridis", alpha=0.8, s=14)
    lims = [float(min(ref_doses.min(), test_doses.min())),
            float(max(ref_doses.max(), test_doses.max()))]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel(f"{model_name} residual heparin (IU)")
    ax.set_ylabel("Nomogram residual heparin (IU)")
    scatter_fig.colorbar(sc, ax=ax, label="Elapsed time (min)")
    scatter_fig.tight_layout()

    summary_text = clinical_summary(k_best, agreement, ibw_mean, model_name,
                                    mc_interval=(ci_low, ci_high),
                                    threshold_iu=threshold_iu)

    metadata = {
        "seed": seed, "n_sim": n_sim, "n_replicates": n_replicates,
        "evaluation_mode": evaluation_mode, "prime_timing": prime_timing,
        "sign_convention": SIGN_CONVENTION, "k_search_bounds": K_BOUNDS,
        "threshold_iu": threshold_iu, "app_version": APP_VERSION,
        "variation": variation, "n_test": n_sim,
        "agreement": agreement, "strata": strata_df, "objectives": obj_df,
        # The geometry object the PDF was rendered from, so the dashboard can
        # draw the same chart on screen without rebuilding it. Rebuilding is
        # how the screen and the print came to disagree in the first place.
        "geometry": geom,
        "worked_example": {
            "total": initial_dose_per_kg * ibw_mean + prime_heparin,
            "elapsed": t_to_mean + t_on_mean,
            "label": f"{ibw_mean:g} kg, {initial_dose_per_kg:g} IU/kg "
                     f"+ {prime_heparin:,.0f} IU prime",
        },
    }

    return (pdf_path, k_best, bias, loa_low, loa_high, fig_ba, k_best, ci_low, ci_high,
            k_post_fig, scatter_fig, sim_df, summary_text, sens_df, sens_fig,
            tornado_fig, metadata, diagnostics)


# ==========================================
# GRID GENERATION
# ==========================================

from parameter_spaces import (                                  # noqa: E402,F401
    ADULT_GRID,
    CANONICAL_COHORTS,   # re-exported
    grid_for,
)

# Kept as module-level names because the dashboard imports them.
heparin_grid = ADULT_GRID["dose_per_kg"]
ibw_grid = ADULT_GRID["ibw"]
time_to_grid = ADULT_GRID["time_to_cpb"]
time_on_grid = ADULT_GRID["time_on_cpb"]
prime_grid = ADULT_GRID["prime"]


def _node_seed(seed, *parts):
    """A reproducible per-node seed derived from the run seed and the node."""
    digest = hashlib.blake2b(repr((seed,) + parts).encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def generate_v2_table_deterministic(seed=DEFAULT_SEED, n_sim=200,
                                    evaluation_mode="reversal_endpoint",
                                    prime_timing="lumped_t0"):
    """Rebuild the k lookup table for every active model.

    Deterministic: each grid node derives its own seed from `seed` and from the
    node itself, so the same command always produces the same tables (EB-6).

    Every reference model is adult, so there is one grid. The `lo`/`hi` entries
    are the k values obtained at time-on-CPB plus and minus two standard
    deviations -- a scenario range, not an uncertainty interval, and the
    dashboard labels them as such.

    Withheld models (nomogram_core.PENDING_MODELS) are skipped, so no table is
    shipped for a model the pipeline does not offer.
    """
    main_prog = st.progress(0.0)
    status_text = st.empty()

    for m_i, model in enumerate(MODEL_NAMES):
        population = "adult"
        g = grid_for(population)
        combos = list(itertools.product(g["dose_per_kg"], g["ibw"],
                                        g["time_to_cpb"], g["time_on_cpb"],
                                        g["prime"]))
        k_table = {"__metadata__": {
            "model": model, "population": population, "seed": seed,
            "n_sim": n_sim, "grid": g, "evaluation_mode": evaluation_mode,
            "prime_timing": prime_timing, "app_version": APP_VERSION,
            "k_search_bounds": list(K_BOUNDS),
            "lo_hi_meaning": "k recalibrated at time on CPB -/+ 2 SD; a scenario "
                             "range, not an uncertainty interval",
        }}
        status_text.text(f"Generating {model} ({population}, {len(combos)} nodes)...")

        for i, (hpkg, ibw_m, tto_m, ton_m, p_hep) in enumerate(combos):
            ibw_sd_val = 10.0
            tto_sd_val = 0.25 * tto_m
            ton_sd_val = 15.0
            # A node-specific but fully determined seed. Python's built-in
            # hash() is salted per process, so a stable digest is used instead --
            # otherwise the "deterministic" table would differ between runs.
            node_seed = _node_seed(seed, model, hpkg, ibw_m, tto_m, ton_m, p_hep)

            def get_k(t_on_val):
                return find_best_k(hpkg, p_hep, ibw_m, ibw_sd_val, tto_m, tto_sd_val,
                                   max(t_on_val, 1.0), ton_sd_val, model_name=model,
                                   n_sim=n_sim, seed=node_seed,
                                   evaluation_mode=evaluation_mode,
                                   prime_timing=prime_timing)

            vals = [get_k(ton_m), get_k(ton_m - 2 * ton_sd_val), get_k(ton_m + 2 * ton_sd_val)]
            k_table[(hpkg, ibw_m, tto_m, ton_m, p_hep)] = {
                "mu": vals[0], "lo": min(vals), "hi": max(vals)}

            if i % 100 == 0:
                main_prog.progress((m_i + (i + 1) / len(combos)) / len(MODEL_NAMES))

        filename = write_k_table(k_table, table_path(model))
        st.success(f"Saved {filename} ({len(combos)} nodes, {population} grid)")

    main_prog.empty()
    status_text.empty()
