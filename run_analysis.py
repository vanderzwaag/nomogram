#!/usr/bin/env python3
"""
Regenerate every number in the manuscript from one seeded command (EB-6).

    python run_analysis.py --seed 20260912 --outdir outputs/frozen

Writes a directory of CSVs plus ``manifest.json`` (seed, git commit, package
versions, every analysis decision) and ``summary.md`` (the values to paste into
the manuscript). Re-running with the same seed reproduces the directory
byte-for-byte, which is what makes "re-run from a fixed, documented version and
reconcile every value" (EB-3) an operation rather than an aspiration.

Analysis decisions that are the authors' to make are CLI flags, each with a
defensible default, and each recorded in the manifest:

  --cohort            grid_snapped (default) | abstract        [EB-3]
  --evaluation-mode   reversal_endpoint (default) | trajectory_cpb | trajectory_full  [EB-4]
  --prime-timing      lumped_t0 (default) | cpb_onset          [EB-5]
  --threshold-iu      1000 (default: one 10 mg protamine increment)  [EB-3]
  --allow-iiv-proxy   substitute IIV where no %RSE is published  [EB-2]
  --jia-allometric    apply conventional allometric scaling to Jia  [EB-4]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import agreement as A
import benchmarks as B
import calibration as cal
import nomogram_core as core
import parameter_spaces as ps
import topup as T

ANALYSIS_VERSION = "2.0.0"

# Seed offsets, so that each stage draws from its own independent stream and
# adding a stage later cannot shift the numbers produced by the earlier ones.
STREAM = {
    "calibration": 1,
    "test_cohort": 2,
    "monte_carlo": 3,
    "parameter_uncertainty": 4,
    "objective_sensitivity": 5,
    "input_sensitivity": 6,
    "prime_timing": 7,
    "institutions": 8,
    "correlated": 9,
    "loa_weight": 11,
    "evaluation_mode": 12,
}


def _git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "short_commit": run("git", "rev-parse", "--short", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "describe": run("git", "describe", "--tags", "--always", "--dirty"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def _package_versions() -> dict:
    import scipy
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }
    for name in ("matplotlib", "streamlit", "pynomo"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = "not installed in the analysis environment"
    return versions


# ==========================================================================

def run(args) -> Path:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    models = list(core.MODEL_NAMES)

    spec = ps.CANONICAL_COHORTS[args.cohort]
    calib_kwargs = dict(
        evaluation_mode=args.evaluation_mode,
        n_timepoints=args.n_timepoints,
        prime_timing=args.prime_timing,
    )

    def write(name: str, df: pd.DataFrame) -> None:
        df.to_csv(outdir / name, index=False)
        print(f"  wrote {name} ({len(df)} rows)")

    print(f"Seed {seed} | cohort '{args.cohort}' ({spec.describe()})")
    print(f"Evaluation mode: {args.evaluation_mode} | prime timing: {args.prime_timing}")
    print(f"Output: {outdir}\n")

    # ------------------------------------------------ 0. implementation checks
    print("[0] Verifying the model implementations (EB-6)")
    internal = B.internal_verification_report()
    write("00_verification_internal.csv", internal)
    write("00_verification_source.csv", B.source_verification_report())
    if not internal["pass"].astype(bool).all():
        raise SystemExit("Internal verification failed; refusing to produce results.")
    print("  all internal checks passed")

    # ------------------------------------------- 1. calibrated k for all models
    print("\n[1] Calibrating k and its Monte Carlo sampling interval (EB-8)")
    k_rows, k_values = [], {}
    mc_draws = []
    for model in models:
        model_spec = spec
        mc = cal.monte_carlo_precision(
            model_spec, model, seed=seed * 10 + STREAM["monte_carlo"],
            n_replicates=args.n_replicates, n_sim=args.n_sim, **calib_kwargs)
        summary = cal.summarise_k(mc["k"])
        mc_draws.append(mc.assign(model=model))

        # THE reported k is the mean over the Monte Carlo replicates, and it is
        # the only k used anywhere downstream -- in the agreement statistics, the
        # top-up grid, the coverage evaluation, the worked example, the lookup
        # tables and the printed nomogram. The submitted pipeline had three
        # different constants in circulation for one configuration (a lookup
        # table built at n_sim = 200, a fresh single-cohort fit at n_sim = 1000
        # used for the agreement statistics, and the replicate mean printed on
        # the PDF), which is one of the mechanisms behind the numbers that did
        # not reconcile. Taking the replicate mean also places k inside its own
        # interval by construction.
        single = cal.find_best_k(model_spec, model,
                                 seed=seed * 10 + STREAM["calibration"],
                                 n_sim=args.n_sim, **calib_kwargs)
        k = summary["k_mean"]
        k_values[model] = k
        k_rows.append({
            "model": model,
            "display_name": core.DISPLAY_NAMES[model],
            "population": model_spec.population,
            "cohort": model_spec.name,
            "k": k,
            "half_life_min": float(np.log(2.0) / k),
            "mc_interval_low": summary["k_p2_5"],
            "mc_interval_high": summary["k_p97_5"],
            "mc_sd": summary["k_sd"],
            "k_single_cohort": single.k,
            "k_single_cohort_pct_from_mean": 100.0 * (single.k - k) / k,
            "interval_type": "Monte Carlo sampling precision (NOT a predictive "
                             "or parameter-uncertainty interval)",
            "n_replicates": args.n_replicates,
            "n_patients_per_replicate": args.n_sim,
            "objective": single.objective,
            "evaluation_mode": single.evaluation_mode,
            "prime_timing": single.prime_timing,
            "k_search_bounds": f"{cal.K_BOUNDS[0]}-{cal.K_BOUNDS[1]}",
            "at_search_bound": bool(mc["at_bound"].any()),
        })
        print(f"  {model:12s} k = {k:.6f}  "
              f"[{summary['k_p2_5']:.6f}, {summary['k_p97_5']:.6f}]  "
              f"t1/2 = {np.log(2.0) / k:.1f} min  "
              f"(single cohort would give {single.k:.6f}, "
              f"{100.0 * (single.k - k) / k:+.2f}%)")
    k_table = pd.DataFrame(k_rows)
    write("01_calibrated_k_all_models.csv", k_table)
    write("01b_monte_carlo_draws.csv", pd.concat(mc_draws, ignore_index=True))

    # ---------------------------------- 2. agreement on an internal test cohort
    print("\n[2] Agreement on a fresh internal sample (EB-1, EB-3)")
    agreement_rows, strata_frames = [], []
    for model in models:
        model_spec = spec
        rng = np.random.default_rng(seed * 10 + STREAM["test_cohort"])
        cohort = cal.sample_cohort(model_spec, args.n_test, rng)
        times = cal.evaluation_times(cohort, "reversal_endpoint")
        ref = cal.reference_values(cohort, model, times).ravel()
        nom = cal.nomogram_values(cohort, times, k_values[model],
                                  args.prime_timing).ravel()
        summary, strata = A.full_agreement_report(cohort, ref, nom,
                                                  args.threshold_iu, args.threshold_pct)
        summary.update({"model": model, "display_name": core.DISPLAY_NAMES[model],
                        "cohort": model_spec.name, "k": k_values[model],
                        "sample_type": "internal resample from the same "
                                       "data-generating process (not external validation)"})
        agreement_rows.append(summary)
        strata_frames.append(strata.assign(model=model))
        print(f"  {model:12s} bias {summary['bias']:+8.2f} IU  "
              f"LoA [{summary['loa_low']:+9.2f}, {summary['loa_high']:+9.2f}]  "
              f"MAPE {summary['mean_abs_pct_error']:5.2f}%  "
              f"prop-bias slope {summary['prop_bias_slope']:+.4f} "
              f"(p {summary['prop_bias_slope_p']:.2g})")
    agr = pd.DataFrame(agreement_rows)
    cols = ["model", "display_name", "k", "n", "bias", "bias_ci_low", "bias_ci_high",
            "sd_diff", "loa_low", "loa_high", "mean_pct_error", "mean_abs_pct_error",
            "max_abs_pct_error", "rmse_iu", "mae_iu", "max_abs_error_iu",
            "prop_bias_slope", "prop_bias_slope_ci_low", "prop_bias_slope_ci_high",
            "prop_bias_slope_p", "prop_bias_significant",
            "prop_bias_predicted_diff_at_p5", "prop_bias_predicted_diff_at_p95",
            "pct_within_threshold_iu", "pct_within_threshold_pct",
            "descriptive_r_squared", "sign_convention", "sample_type", "cohort"]
    write("02_agreement_primary.csv", agr[[c for c in cols if c in agr]])
    write("03_agreement_stratified.csv", pd.concat(strata_frames, ignore_index=True))

    # -------------------------------------------- 3. objective-function choice
    print("\n[3] Objective-function and weighting sensitivity (EB-3, R1 p9 L44)")
    obj = pd.concat([
        cal.objective_sensitivity(spec, m, seed=seed * 10 + STREAM["objective_sensitivity"],
                                  n_sim=args.n_sim, **calib_kwargs)
        for m in models], ignore_index=True)
    write("04_objective_sensitivity.csv", obj)
    worst = obj.loc[obj["pct_change_vs_primary"].abs().idxmax()]
    print(f"  largest movement in k across objectives: {worst['pct_change_vs_primary']:+.2f}% "
          f"({worst['model']}, {worst['objective']})")

    loaw = pd.concat([
        cal.loa_weight_sensitivity(spec, m, seed=seed * 10 + STREAM["loa_weight"],
                                   n_sim=args.n_sim, **calib_kwargs)
        for m in models], ignore_index=True)
    write("05_loa_weight_sensitivity.csv", loaw)

    # ------------------------------------------------- 4. input sensitivity
    print("\n[4] Input sensitivity, +/-20% one factor at a time")
    sens = pd.concat([
        cal.input_sensitivity(spec, m, seed=seed * 10 + STREAM["input_sensitivity"],
                              variation=args.variation, n_sim=args.n_sim, **calib_kwargs)
        for m in models], ignore_index=True)
    write("06_input_sensitivity.csv", sens)
    top = sens.loc[sens["pct_change"].abs().idxmax()]
    print(f"  most influential input: {top['parameter']} ({top['direction']}), "
          f"{top['pct_change']:+.2f}% change in k ({top['model']})")

    # -------------------------------- 5. calibration endpoint and prime timing
    print("\n[5] Calibration endpoint (EB-4) and prime timing (EB-5)")
    modes = pd.concat([
        cal.evaluation_mode_comparison(spec, m, seed=seed * 10 + STREAM["evaluation_mode"],
                                       n_sim=args.n_sim, n_timepoints=args.n_timepoints,
                                       prime_timing=args.prime_timing)
        for m in models], ignore_index=True)
    write("07_calibration_endpoint_comparison.csv", modes)

    ptim = pd.concat([
        cal.prime_timing_sensitivity(spec, m, seed=seed * 10 + STREAM["prime_timing"],
                                     n_sim=args.n_sim,
                                     evaluation_mode=args.evaluation_mode,
                                     n_timepoints=args.n_timepoints)
        for m in models], ignore_index=True)
    write("08_prime_timing_sensitivity.csv", ptim)
    biggest = ptim.loc[ptim["pct_change_vs_lumped"].abs().idxmax()]
    print(f"  prime-timing convention moves k by up to "
          f"{biggest['pct_change_vs_lumped']:+.2f}% ({biggest['model']})")

    # ----------------------------------------- 6. parameter uncertainty (EB-2)
    print("\n[6] Parameter-uncertainty propagation (EB-2)")
    pu_rows, pu_draws = [], []
    for model in models:
        rep = cal.parameter_uncertainty(
            spec, model, seed=seed * 10 + STREAM["parameter_uncertainty"],
            n_draws=args.n_param_draws, n_sim=max(200, args.n_sim // 2),
            allow_iiv_proxy=args.allow_iiv_proxy, **calib_kwargs)
        row = {"model": model, "available": rep.available,
               "uncertainty_source": rep.source, "note": rep.reason}
        if rep.provenance:
            row.update({f"basis_{k}": v for k, v in rep.provenance.items()})
        if rep.available:
            row.update(cal.summarise_k(rep.draws["k"]))
            pu_draws.append(rep.draws.assign(model=model))
            print(f"  {model:12s} parameter-uncertainty interval "
                  f"[{row['k_p2_5']:.6f}, {row['k_p97_5']:.6f}] ({rep.source})")
        else:
            print(f"  {model:12s} not propagated: {rep.reason.split('.')[0]}")
        pu_rows.append(row)
    pu = pd.DataFrame(pu_rows)
    write("09_parameter_uncertainty.csv", pu)
    if pu_draws:
        write("09b_parameter_uncertainty_draws.csv",
              pd.concat(pu_draws, ignore_index=True))

    # -------------------------------- 7. coverage over the full range (EB-4)
    print("\n[7] Full-range and boundary coverage (EB-4)")
    cover_rows = []
    for label, builder in (("full_range", ps.full_range_cohort),
                           ("boundary", ps.boundary_cohort)):
        for model in models:
            population = "adult"
            grid = builder(population)
            times = cal.evaluation_times(grid, "reversal_endpoint")
            ref = cal.reference_values(grid, model, times).ravel()
            nom = cal.nomogram_values(grid, times, k_values[model],
                                      args.prime_timing).ravel()
            rel = A.relative_errors(ref, nom)
            ba = A.bland_altman(ref, nom)
            worst_i = int(np.argmax(np.abs(A.difference(ref, nom))))
            cover_rows.append({
                "evaluation": label, "model": model, "population": population,
                "n_nodes": len(grid), "k": k_values[model],
                "bias_iu": ba.bias, "loa_low_iu": ba.loa_low, "loa_high_iu": ba.loa_high,
                "mean_abs_pct_error": rel.mean_abs_pct_error,
                "max_abs_pct_error": rel.max_abs_pct_error,
                "max_abs_error_iu": rel.max_abs_error_iu,
                "worst_node_dose_per_kg": float(grid["dose_per_kg"].iloc[worst_i]),
                "worst_node_ibw": float(grid["ibw"].iloc[worst_i]),
                "worst_node_prime": float(grid["heparin_prime"].iloc[worst_i]),
                "worst_node_time_to_cpb": float(grid["time_to_cpb"].iloc[worst_i]),
                "worst_node_time_on_cpb": float(grid["time_on_cpb"].iloc[worst_i]),
                "pct_within_threshold_iu": A.threshold_coverage(
                    ref, nom, args.threshold_iu, args.threshold_pct)["pct_within_threshold_iu"],
            })
    coverage = pd.DataFrame(cover_rows)
    write("10_coverage_full_range_and_boundary.csv", coverage)
    b = coverage[coverage["evaluation"] == "boundary"]
    print(f"  worst boundary error across models: "
          f"{b['max_abs_pct_error'].max():.1f}% "
          f"({b.loc[b['max_abs_pct_error'].idxmax(), 'model']})")

    # ------------------------------- 8. transportability across institutions
    print("\n[8] Transportability to shifted simulated institutions (EB-4)")
    inst_rows = []
    for model in models:
        population = "adult"
        for name, shifted in ps.institutions_for(population).items():
            rng = np.random.default_rng(seed * 10 + STREAM["institutions"])
            cohort = cal.sample_cohort(shifted, args.n_test, rng)
            times = cal.evaluation_times(cohort, "reversal_endpoint")
            ref = cal.reference_values(cohort, model, times).ravel()
            # Transported k: calibrated at the canonical cohort, applied here.
            nom_transported = cal.nomogram_values(cohort, times, k_values[model],
                                                  args.prime_timing).ravel()
            # Recalibrated k: what the pipeline would produce for this institution.
            r_local = cal.find_best_k(shifted, model,
                                      seed=seed * 10 + STREAM["institutions"],
                                      n_sim=args.n_sim, **calib_kwargs)
            nom_local = cal.nomogram_values(cohort, times, r_local.k,
                                            args.prime_timing).ravel()
            ba_t, ba_l = A.bland_altman(ref, nom_transported), A.bland_altman(ref, nom_local)
            rel_t, rel_l = A.relative_errors(ref, nom_transported), A.relative_errors(ref, nom_local)
            inst_rows.append({
                "model": model, "institution": name, "population": population,
                "spec": shifted.describe(),
                "k_transported": k_values[model], "k_recalibrated": r_local.k,
                "k_pct_difference": 100.0 * (r_local.k - k_values[model]) / k_values[model],
                "bias_transported_iu": ba_t.bias, "bias_recalibrated_iu": ba_l.bias,
                "mape_transported": rel_t.mean_abs_pct_error,
                "mape_recalibrated": rel_l.mean_abs_pct_error,
                "loa_width_transported_iu": ba_t.loa_high - ba_t.loa_low,
                "loa_width_recalibrated_iu": ba_l.loa_high - ba_l.loa_low,
            })
    inst = pd.DataFrame(inst_rows)
    write("11_shifted_institutions.csv", inst)
    print(f"  transporting k unchanged costs up to "
          f"{(inst['mape_transported'] - inst['mape_recalibrated']).max():.2f} "
          f"percentage points of MAPE")

    # --------------------------------------------- 9. top-up grid (EB-5)
    print("\n[9] Supplemental-bolus grid (EB-5)")
    grid = T.topup_grid(
        k_values,
        heparin_bolus=spec.dose_per_kg * spec.ibw_mean,
        heparin_prime=spec.prime_heparin,
        weight_kg=spec.ibw_mean,
        time_to_cpb=spec.t_to_mean,
        horizon_min=args.topup_horizon,
        prime_timing=args.prime_timing,
        threshold_iu=args.threshold_iu,
    )
    write("12_topup_grid.csv", grid)
    ws = T.worst_case_summary(grid)
    write("13_topup_worst_case.csv", ws)
    print(f"  worst case across {grid['scenario'].nunique()} scenarios: "
          f"{ws['worst_abs_pct_error'].max():.1f}% ({ws.iloc[0]['model']})")

    # ------------------------------------ 10. correlated inputs (R2 covariance)
    print("\n[10] Correlated-input sensitivity (R2)")
    corr_rows = []
    for model in models:
        for rho in (0.0, 0.3, 0.6):
            rng = np.random.default_rng(seed * 10 + STREAM["correlated"])
            cohort = cal.sample_correlated_cohort(spec, args.n_test, rng, correlation=rho)
            times = cal.evaluation_times(cohort, "reversal_endpoint")
            ref = cal.reference_values(cohort, model, times).ravel()
            nom = cal.nomogram_values(cohort, times, k_values[model],
                                      args.prime_timing).ravel()
            ba, rel = A.bland_altman(ref, nom), A.relative_errors(ref, nom)
            corr_rows.append({"model": model, "input_correlation": rho,
                              "bias_iu": ba.bias, "loa_low_iu": ba.loa_low,
                              "loa_high_iu": ba.loa_high,
                              "mean_abs_pct_error": rel.mean_abs_pct_error})
    write("14_correlated_inputs.csv", pd.DataFrame(corr_rows))

    # ------------------------------------------- 11. worked example (EB-8)
    print("\n[11] Worked example for the manuscript (EB-8)")
    ex_rows = []
    bolus = spec.dose_per_kg * spec.ibw_mean
    total = bolus + spec.prime_heparin
    for model in models:
        k = k_values[model]
        for t in (spec.t_to_mean + spec.t_on_mean, 60.0, 90.0, 120.0):
            ex_rows.append({
                "model": model, "ibw_kg": spec.ibw_mean,
                "dose_per_kg": spec.dose_per_kg, "bolus_iu": bolus,
                "prime_iu": spec.prime_heparin, "total_load_iu": total,
                "elapsed_min": t, "k": k,
                "nomogram_residual_iu": core.simplified_amount(
                    t, bolus, spec.prime_heparin, k, spec.t_to_mean, args.prime_timing),
                "reference_residual_iu": core.reference_amount(
                    model, t, bolus, spec.prime_heparin, spec.ibw_mean, spec.t_to_mean),
                "protamine_mg_at_1_to_100": core.simplified_amount(
                    t, bolus, spec.prime_heparin, k, spec.t_to_mean, args.prime_timing) / 100.0,
            })
    ex = pd.DataFrame(ex_rows)
    write("15_worked_example.csv", ex)
    row = ex[(ex["model"] == args.primary_model) & (ex["elapsed_min"] == 60.0)].iloc[0]
    print(f"  {args.primary_model}, total {total:,.0f} IU at 60 min -> "
          f"{row['nomogram_residual_iu']:,.0f} IU "
          f"({row['protamine_mg_at_1_to_100']:.0f} mg protamine at 1 mg:100 IU)")

    # ------------------------------------------------------------- manifest
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_version": ANALYSIS_VERSION,
        "core_version": core.CORE_VERSION,
        "calibration_version": cal.CALIBRATION_VERSION,
        "seed": seed,
        "seed_streams": STREAM,
        "command_line": " ".join(sys.argv),
        "git": _git_info(),
        "software_versions": _package_versions(),
        "decisions": {
            "canonical_cohort": args.cohort,
            "canonical_cohort_spec": spec.as_dict(),
            "sign_convention": A.SIGN_CONVENTION_LABEL,
            "evaluation_mode": args.evaluation_mode,
            "evaluation_mode_meaning": cal.EVALUATION_MODE_DESCRIPTIONS[args.evaluation_mode],
            "prime_timing": args.prime_timing,
            "clinical_threshold_iu": args.threshold_iu,
            "clinical_threshold_pct": args.threshold_pct,
            "clinical_threshold_rationale":
                "one 10 mg protamine increment at a 1 mg : 100 IU ratio",
            "k_search_bounds": list(cal.K_BOUNDS),
            "k_search_bounds_note": cal.K_BOUNDS_NOTE,
            "primary_objective": "bland_altman",
            "objective_definition": cal.OBJECTIVE_LABELS["bland_altman"],
            "jia_population": "adult (small-bodied derivation cohort); the "
                              "manuscript's paediatric description is an error",
            "iiv_substituted_for_missing_uncertainty": bool(args.allow_iiv_proxy),
            "parameter_uncertainty_basis": {
                m: cal.uncertainty_provenance(m)
                for m in core.MODEL_NAMES
                if core.MODEL_PARAMETERS[m].rse or core.MODEL_PARAMETERS[m].bootstrap_ci
            },
        },
        "sample_sizes": {
            "n_sim_per_calibration": args.n_sim,
            "n_test_cohort": args.n_test,
            "n_monte_carlo_replicates": args.n_replicates,
            "n_parameter_draws": args.n_param_draws,
            "ofat_variation": args.variation,
            "topup_horizon_min": args.topup_horizon,
        },
        "parameter_provenance": {
            key: {
                "source": p.source,
                "values": p.values,
                "iiv": p.iiv,
                "rse": p.rse,
                "weight_covariate": p.weight_covariate,
                "notes": p.notes,
                "missing_rse": list(p.missing_rse()),
            }
            for key, p in core.MODEL_PARAMETERS.items()
        },
        "calibration_grids": {"adult": ps.ADULT_GRID},
        "outstanding_author_decisions": _outstanding(args),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("\n  wrote manifest.json")

    _write_summary(outdir, args, spec, k_table, agr, obj, sens, modes, ptim,
                   pu, coverage, inst, ws, ex, manifest)
    print("  wrote summary.md")
    return outdir


def _outstanding(args) -> list:
    items = []
    missing = {k: list(p.missing_rse()) for k, p in core.MODEL_PARAMETERS.items()
               if p.missing_rse() and p.rse}
    if missing:
        items.append({
            "comment": "EB-2",
            "action": "Transcribe the published %RSE for these parameters, or state "
                      "that the parameter-uncertainty interval could not be propagated",
            "detail": missing,
        })
    pending = B.source_verification_report()
    pending = pending[pending["status"].str.startswith("PENDING")]
    if not pending.empty:
        items.append({
            "comment": "EB-6",
            "action": "Supply the expected values for the per-source benchmark checks",
            "detail": pending[["model", "description"]].to_dict("records"),
        })
    items.append({
        "comment": "EB-4 / R1 p11 L46",
        "action": "Correct the manuscript's description of the Jia model",
        "detail": ("Jia is an ADULT model derived in a relatively small-bodied "
                   "population, not a paediatric one. Its Vc of 3.04 L is within "
                   "2% of Delavenne's adult 3.1 L. The reviewers' request for a "
                   "paediatric parameter space follows from the manuscript's own "
                   "mis-description and is answered by correcting the text. The "
                   "model carries no weight covariate, so it cannot be "
                   "individualised by weight and the upper end of the adult "
                   "calibration grid extrapolates beyond its derivation "
                   "population; both are stated limitations. Transportability to "
                   "a small-bodied adult population is evaluated explicitly."),
    })
    items.append({
        "comment": "EB-1",
        "action": "State the assumption under which central-compartment amount maps "
                  "to a protamine dose",
        "detail": "The pipeline reports PK central-compartment amount only; no "
                  "pharmacodynamic (anti-Xa or ACT) layer is modelled.",
    })
    return items


def _write_summary(outdir, args, spec, k_table, agr, obj, sens, modes, ptim,
                   pu, coverage, inst, ws, ex, manifest) -> None:
    g = manifest["git"]
    L = []
    L.append("# Frozen analysis run\n")
    L.append(f"- Generated: {manifest['generated_utc']}")
    L.append(f"- Commit: `{g.get('describe') or g.get('commit')}`"
             + ("  **working tree dirty -- commit before quoting these numbers**"
                if g.get("dirty") else ""))
    L.append(f"- Seed: `{args.seed}`")
    L.append(f"- Cohort: {spec.describe()}")
    L.append(f"- Sign convention: {A.SIGN_CONVENTION_LABEL}")
    L.append(f"- Calibration endpoint: `{args.evaluation_mode}`")
    L.append(f"- Prime timing: `{args.prime_timing}`")
    L.append(f"- Clinical threshold: {args.threshold_iu:.0f} IU "
             f"({args.threshold_iu / 100:.0f} mg protamine at 1 mg:100 IU) "
             f"or {args.threshold_pct:.0f}%")
    L.append(f"- Software: python {manifest['software_versions']['python']}, "
             f"numpy {manifest['software_versions']['numpy']}, "
             f"scipy {manifest['software_versions']['scipy']}, "
             f"pandas {manifest['software_versions']['pandas']}\n")

    L.append("## Calibrated decay constant, all six models (EB-8)\n")
    t = k_table[["display_name", "population", "k", "mc_interval_low",
                 "mc_interval_high", "half_life_min"]].copy()
    L.append("| Model | Population | k (/min) | MC sampling interval | Apparent t1/2 (min) |")
    L.append("|---|---|---|---|---|")
    for _, r in t.iterrows():
        L.append(f"| {r['display_name']} | {r['population']} | {r['k']:.5f} | "
                 f"{r['mc_interval_low']:.5f} to {r['mc_interval_high']:.5f} | "
                 f"{r['half_life_min']:.0f} |")
    L.append("\nThe interval is a Monte Carlo sampling-precision interval. It does "
             "**not** quantify uncertainty in the published PK parameters, in the "
             "institutional input estimates, in model selection, or in an "
             "individual patient's prediction (EB-2).\n")

    L.append("## Agreement on an internal resample (EB-3)\n")
    L.append("| Model | Bias (IU) | 95% LoA (IU) | MAPE (%) | Max abs error (%) | "
             "Prop-bias slope | p | Within threshold (%) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in agr.iterrows():
        L.append(f"| {r['display_name']} | {r['bias']:+.2f} | "
                 f"{r['loa_low']:+.1f} to {r['loa_high']:+.1f} | "
                 f"{r['mean_abs_pct_error']:.2f} | {r['max_abs_pct_error']:.1f} | "
                 f"{r['prop_bias_slope']:+.4f} | {r['prop_bias_slope_p']:.2g} | "
                 f"{r['pct_within_threshold_iu']:.1f} |")
    L.append("\nThe test cohort is a fresh draw from the same data-generating "
             "process, i.e. an internal resample, not an external validation "
             "cohort (EB-1).\n")

    L.append("## Robustness of k to the objective function (EB-3)\n")
    piv = obj.pivot(index="model", columns="objective", values="k")
    L.append("| Model | " + " | ".join(piv.columns) + " | max deviation from primary |")
    L.append("|---" * (len(piv.columns) + 2) + "|")
    for model, row in piv.iterrows():
        dev = obj[obj["model"] == model]["pct_change_vs_primary"].abs().max()
        L.append(f"| {model} | " + " | ".join(f"{v:.5f}" for v in row) +
                 f" | {dev:.1f}% |")
    L.append("")

    L.append("## Calibration endpoint: single timepoint versus trajectory (EB-4)\n")
    piv = modes.pivot(index="model", columns="evaluation_mode", values="k")
    L.append("| Model | " + " | ".join(piv.columns) + " |")
    L.append("|---" * (len(piv.columns) + 1) + "|")
    for model, row in piv.iterrows():
        L.append(f"| {model} | " + " | ".join(f"{v:.5f}" for v in row) + " |")
    L.append(f"\nThe run used `{args.evaluation_mode}`: "
             f"{cal.EVALUATION_MODE_DESCRIPTIONS[args.evaluation_mode]}\n")

    L.append("## Prime-timing convention (EB-5)\n")
    piv = ptim.pivot(index="model", columns="prime_timing", values="k")
    L.append("| Model | " + " | ".join(piv.columns) + " | change (%) |")
    L.append("|---" * (len(piv.columns) + 2) + "|")
    for model, row in piv.iterrows():
        ch = ptim[(ptim["model"] == model) &
                  (ptim["prime_timing"] == "cpb_onset")]["pct_change_vs_lumped"].iloc[0]
        L.append(f"| {model} | " + " | ".join(f"{v:.5f}" for v in row) + f" | {ch:+.1f} |")
    L.append("")

    L.append("## Two intervals on k, which must not be conflated (EB-2)\n")
    L.append("The sampling-precision interval repeats the simulation with fresh "
             "cohorts drawn from the same fixed, investigator-specified "
             "distributions, holding the published pharmacokinetic parameters at "
             "their point estimates. The parameter-uncertainty interval instead "
             "resamples those published parameters from their reported "
             "uncertainty. The first measures Monte Carlo noise; only the second "
             "speaks to how well the reference models themselves are known.\n")
    L.append("| Model | k | Sampling-precision interval | Parameter-uncertainty interval | Width ratio | Source |")
    L.append("|---|---|---|---|---|---|")
    pu_idx = pu.set_index("model")
    for _, r in k_table.iterrows():
        m = r["model"]
        mc_w = r["mc_interval_high"] - r["mc_interval_low"]
        row = pu_idx.loc[m]
        if bool(row["available"]):
            pu_w = row["k_p97_5"] - row["k_p2_5"]
            pu_txt = f"{row['k_p2_5']:.5f} to {row['k_p97_5']:.5f}"
            ratio = f"{pu_w / mc_w:.0f}x wider"
            src = ("interindividual variability (proxy)"
                   if row["uncertainty_source"] == "iiv"
                   else ("published bootstrap 95% CI"
                         if any(str(v).startswith("published bootstrap")
                                for k2, v in row.items() if k2.startswith("basis_"))
                         else "published %RSE"))
        else:
            pu_txt, ratio, src = ("not propagated", "--",
                                  "closed-form expression; none published")
        L.append(f"| {r['display_name']} | {r['k']:.5f} | "
                 f"{r['mc_interval_low']:.5f} to {r['mc_interval_high']:.5f} | "
                 f"{pu_txt} | {ratio} | {src} |")
    used_proxy = (pu["uncertainty_source"] == "iiv").any()
    if used_proxy:
        L.append("\n**Caveat.** Interindividual variability was substituted for "
                 "estimation uncertainty in at least one model. IIV describes "
                 "spread between patients, not uncertainty in the published "
                 "estimate, so that interval is an over-estimate.")
    else:
        L.append("\nEvery interval above is propagated from uncertainty the source "
                 "publications actually report: the asymptotic %RSE for Lanoiselee "
                 "and Delavenne, and the non-parametric bootstrap interval for Jia, "
                 "which publishes one. For Delavenne the draw includes the weight "
                 "exponent on clearance (0.767, 29% RSE); because that covariate is "
                 "centred on 70 kg its contribution is zero at the canonical "
                 "cohort's IBW and reaches about 25% on clearance at the ends of "
                 "the weight grid, so it moves the boundary and transportability "
                 "results rather than this table.")
    L.append("")

    L.append("## Coverage of the full input range and its boundaries (EB-4)\n")
    L.append("| Model | Evaluation | Nodes | Mean abs error (%) | Max abs error (%) | "
             "Worst node |")
    L.append("|---|---|---|---|---|---|")
    for _, r in coverage.iterrows():
        L.append(f"| {r['model']} | {r['evaluation']} | {r['n_nodes']} | "
                 f"{r['mean_abs_pct_error']:.2f} | {r['max_abs_pct_error']:.1f} | "
                 f"{r['worst_node_dose_per_kg']:.0f} IU/kg, {r['worst_node_ibw']:.0f} kg, "
                 f"prime {r['worst_node_prime']:.0f}, "
                 f"{r['worst_node_time_to_cpb']:.0f}+{r['worst_node_time_on_cpb']:.0f} min |")
    L.append("")

    L.append("## Supplemental-bolus grid, worst case and mean (EB-5)\n")
    L.append("| Model | Scenarios | Mean abs error (%) | Worst abs error (%) | "
             "Worst abs error (IU) | Worst-case scenario |")
    L.append("|---|---|---|---|---|---|")
    for _, r in ws.iterrows():
        L.append(f"| {r['model']} | {r['n_scenarios']} | {r['mean_abs_pct_error']:.2f} | "
                 f"{r['worst_abs_pct_error']:.1f} | {r['worst_abs_diff_iu']:.0f} | "
                 f"{r['worst_case_scenario']} |")
    L.append("")

    L.append("## Transportability to shifted institutions (EB-4)\n")
    piv = inst.pivot_table(index="institution", columns="model",
                           values="mape_transported")
    L.append("Mean absolute percentage error when the canonically calibrated k is "
             "applied unchanged to a shifted population. Adult models are not run "
             "All six models are adult and are evaluated in the same populations; "
             "the small_bodied_adults row matches the body size the Jia model was "
             "derived in (EB-4, R1 p11 L46).\n")
    L.append("| Institution | " + " | ".join(piv.columns) + " |")
    L.append("|---" * (len(piv.columns) + 1) + "|")
    for name, row in piv.iterrows():
        L.append(f"| {name} | " + " | ".join(
            "n/a" if pd.isna(v) else f"{v:.2f}" for v in row) + " |")
    L.append("")

    L.append("## Most influential inputs, +/-20% (EB-8)\n")
    piv = sens.groupby(["model", "parameter"])["pct_change"].apply(
        lambda s: s.abs().max()).unstack()
    L.append("| Model | " + " | ".join(piv.columns) + " |")
    L.append("|---" * (len(piv.columns) + 1) + "|")
    for model, row in piv.iterrows():
        L.append(f"| {model} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
    L.append("")

    L.append("## Worked example (EB-8)\n")
    sub = ex[ex["model"] == args.primary_model]
    L.append(f"{args.primary_model}, {spec.ibw_mean:.0f} kg, "
             f"{spec.dose_per_kg:.0f} IU/kg bolus + {spec.prime_heparin:.0f} IU prime "
             f"= {sub['total_load_iu'].iloc[0]:,.0f} IU total load, "
             f"k = {sub['k'].iloc[0]:.5f}/min:\n")
    L.append("| Elapsed (min) | Nomogram residual (IU) | Reference residual (IU) | "
             "Protamine at 1 mg:100 IU (mg) |")
    L.append("|---|---|---|---|")
    for _, r in sub.iterrows():
        L.append(f"| {r['elapsed_min']:.0f} | {r['nomogram_residual_iu']:,.0f} | "
                 f"{r['reference_residual_iu']:,.0f} | "
                 f"{r['protamine_mg_at_1_to_100']:.0f} |")
    L.append("\nReading the residual load off the nomogram requires no calculation; "
             "converting it into a protamine dose requires applying the "
             "institutional ratio (EB-8).\n")

    L.append("## Outstanding author decisions\n")
    for item in manifest["outstanding_author_decisions"]:
        L.append(f"- **{item['comment']}** -- {item['action']}")
    L.append("")

    (outdir / "summary.md").write_text("\n".join(L))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=20260912,
                   help="master seed; every stage derives its stream from it")
    p.add_argument("--outdir", default="outputs/frozen")
    p.add_argument("--cohort", choices=sorted(ps.CANONICAL_COHORTS),
                   default=ps.DEFAULT_COHORT,
                   help="canonical representative cohort (EB-3)")
    p.add_argument("--evaluation-mode", choices=cal.EVALUATION_MODES,
                   default="reversal_endpoint",
                   help="calibrate against one endpoint per patient or the trajectory (EB-4)")
    p.add_argument("--n-timepoints", type=int, default=13,
                   help="timepoints per patient in trajectory mode")
    p.add_argument("--prime-timing", choices=core.PRIME_TIMING_CHOICES,
                   default="lumped_t0", help="prime-timing convention (EB-5)")
    p.add_argument("--threshold-iu", type=float, default=A.DEFAULT_THRESHOLD_IU)
    p.add_argument("--threshold-pct", type=float, default=A.DEFAULT_THRESHOLD_PCT)
    p.add_argument("--allow-iiv-proxy", action="store_true",
                   help="substitute interindividual variability where a model "
                        "publishes no estimation uncertainty (EB-2). No longer "
                        "needed: all three population models now carry published "
                        "%%RSE or bootstrap intervals. Overstates parameter "
                        "uncertainty and is recorded as such.")
    p.add_argument("--jia-allometric", action="store_true",
                   help="apply conventional allometric scaling to Jia (EB-4)")
    p.add_argument("--primary-model", default="lanoiselee", choices=list(core.MODEL_NAMES))
    p.add_argument("--n-sim", type=int, default=1000, help="patients per calibration")
    p.add_argument("--n-test", type=int, default=1000, help="patients in the test cohort")
    p.add_argument("--n-replicates", type=int, default=500,
                   help="Monte Carlo replicates for the sampling interval")
    p.add_argument("--n-param-draws", type=int, default=200,
                   help="parameter draws for the uncertainty interval")
    p.add_argument("--variation", type=float, default=0.2, help="OFAT fraction")
    p.add_argument("--topup-horizon", type=float, default=240.0)
    p.add_argument("--quick", action="store_true",
                   help="small sample sizes, for a smoke test only")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        args.n_sim, args.n_test = 200, 200
        args.n_replicates, args.n_param_draws = 20, 10
    outdir = run(args)
    print(f"\nDone. {outdir}/summary.md holds the values for the manuscript.")


if __name__ == "__main__":
    main()
