#!/usr/bin/env python3
"""
Generate the supplementary figures from the frozen analysis run.

    python make_supplementary_figures.py --indir outputs/frozen

Every curve is computed from the same seeded pipeline and the same calibrated
constants as the tables, so the figures cannot drift from the text. Output is
written as both PDF (vector, for typesetting) and PNG at 600 dpi.

Figure S1  Reference-model decay trajectories with each model's own calibrated
           approximation, as small multiples.
Figure S2  The two intervals on the approximation, drawn to scale (EB-2).
Figure S3  Axis-reset accuracy after a supplemental bolus, best and worst model.
Figure S4  One-factor-at-a-time sensitivity of the calibrated constant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nomogram_core as core
import parameter_spaces as ps

# Categorical slots, in the fixed validated order. Assigned to models once and
# never re-ordered, so a model keeps its colour across every figure.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MODEL_ORDER = ["lanoiselee", "delavenne", "jia", "meesters", "prodose", "prodose-2"]
COLOUR = dict(zip(MODEL_ORDER, SLOTS))

INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"
GRID, SURFACE = "#e8e8e4", "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "xtick.color": INK_3, "ytick.color": INK_3,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 7.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def tidy(ax, grid_axis="y"):
    """Recessive chrome: hairline grid, no top or right spine."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def save(fig, outdir: Path, stem: str):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(outdir / f"{stem}.{ext}", **kw)
    plt.close(fig)
    print(f"  wrote {stem}.pdf and {stem}.png")


# ==========================================================================

def figure_s1(k, spec, outdir):
    """Decay trajectories, one panel per reference model.

    Small multiples rather than seven curves on one axis, because each model has
    its OWN calibrated constant -- there is no single approximation to overlay.
    """
    bolus, prime = spec.dose_per_kg * spec.ibw_mean, spec.prime_heparin
    t_to, t_end = spec.t_to_mean, spec.t_to_mean + spec.t_on_mean
    t = np.linspace(0, 120, 601)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey=True)
    for ax, model in zip(axes.ravel(), MODEL_ORDER):
        ref = core.reference_amount_array(model, t, bolus, prime, spec.ibw_mean, t_to)
        nom = core.simplified_amount_array(t, bolus, prime, k[model], t_to)

        ax.axvline(t_end, color=INK_3, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.plot(t, ref / 1000, color=COLOUR[model], lw=2.0, zorder=3,
                label="Reference model")
        ax.plot(t, nom / 1000, color=INK, lw=1.4, ls=(0, (5, 2)), zorder=4,
                label="Nomogram approximation")

        ax.set_title(core.DISPLAY_NAMES[model], pad=4)
        ax.text(0.96, 0.93, f"$k$ = {k[model]:.5f} min$^{{-1}}$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=INK_2)
        tidy(ax)
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 35)

    axes[0, 0].text(t_end + 2, 33.5, "reversal", fontsize=6.5, color=INK_3)
    # The offset before CPB onset is the prime-timing convention, not an error:
    # the reference models introduce prime heparin at bypass onset while the
    # printed nomogram necessarily lumps it into the load from induction (EB-5).
    axes[0, 0].annotate("prime enters at\nCPB onset", xy=(t_to, 27.6),
                        xytext=(t_to + 12, 15.5), fontsize=6.5, color=INK_3,
                        linespacing=1.4,
                        arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.7))
    for ax in axes[1, :]:
        ax.set_xlabel("Time from induction (min)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Heparin in central\ncompartment (×1000 IU)")
    axes[0, 0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.02))
    fig.tight_layout(w_pad=1.4, h_pad=1.2)
    save(fig, outdir, "figure_S1_decay_trajectories")


def figure_s2(k, k_mc, k_pu, spec, outdir, model="lanoiselee"):
    """The two intervals, drawn to the same scale (EB-2).

    The narrow band is Monte Carlo sampling precision, which is what the
    submitted manuscript reported as a predicted margin of error. The wide band
    propagates the published parameter uncertainty. Drawing both on one axis is
    the point of the figure: they are not the same quantity, and the difference
    is not subtle.
    """
    bolus, prime = spec.dose_per_kg * spec.ibw_mean, spec.prime_heparin
    t_to, t_end = spec.t_to_mean, spec.t_to_mean + spec.t_on_mean
    t = np.linspace(0, 120, 601)
    total = bolus + prime
    band = lambda lo, hi: (total * np.exp(-hi * t) / 1000,
                           total * np.exp(-lo * t) / 1000)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    pu_lo, pu_hi = band(*k_pu[model])
    mc_lo, mc_hi = band(*k_mc[model])

    ax.fill_between(t, pu_lo, pu_hi, color=COLOUR[model], alpha=0.22, lw=0,
                    zorder=2,
                    label="Parameter uncertainty (published %RSE)")
    ax.fill_between(t, mc_lo, mc_hi, color="#0d3f7a", alpha=1.0, lw=0, zorder=6,
                    label="Monte Carlo sampling precision")
    ax.plot(t, total * np.exp(-k[model] * t) / 1000, color=COLOUR[model],
            lw=2.0, zorder=5, label="Nomogram approximation")
    ax.plot(t, core.reference_amount_array(model, t, bolus, prime,
                                           spec.ibw_mean, t_to) / 1000,
            color=INK, lw=1.4, ls=(0, (5, 2)), zorder=4,
            label=f"{core.DISPLAY_NAMES[model]} reference model")
    ax.axvline(t_end, color=INK_3, lw=0.7, ls=(0, (4, 3)), zorder=1)

    i = np.argmin(np.abs(t - t_end))
    w_pu = (pu_hi[i] - pu_lo[i]) * 1000
    w_mc = (mc_hi[i] - mc_lo[i]) * 1000
    ax.annotate(f"parameter uncertainty\n{w_pu:,.0f} IU at reversal",
                xy=(t_end, pu_hi[i]), xytext=(t_end + 9, pu_hi[i] + 4.6),
                fontsize=7, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.7))
    # The sampling band is thinner than the line that draws it, which is the
    # finding rather than a rendering problem -- so it is stated in words.
    ax.annotate(f"sampling precision {w_mc:,.0f} IU\n"
                f"({w_pu / w_mc:.0f}× narrower — thinner\nthan the plotted line)",
                xy=(t_end, mc_lo[i]), xytext=(t_end - 56, mc_lo[i] - 7.2),
                fontsize=7, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.7))

    tidy(ax)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 36)
    ax.set_xlabel("Time from induction (min)")
    ax.set_ylabel("Heparin in central compartment (×1000 IU)")
    ax.set_title(f"{core.DISPLAY_NAMES[model]}: the two intervals on the "
                 "approximation,\ndrawn to the same scale", pad=6)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0))
    fig.tight_layout()
    save(fig, outdir, "figure_S2_uncertainty_bands")


def figure_s3(k, spec, outdir, best="lanoiselee", worst="delavenne"):
    """Axis-reset accuracy after a supplemental bolus, best and worst model."""
    bolus, prime = spec.dose_per_kg * spec.ibw_mean, spec.prime_heparin
    t_to = spec.t_to_mean
    t_bolus, dose = 60.0, 5000.0
    t = np.linspace(0, 240, 961)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1]})
    for col, model in enumerate((best, worst)):
        ref = np.array([core.reference_amount_with_topups(
            model, tt, bolus, prime, spec.ibw_mean, t_to, ((t_bolus, dose),))
            for tt in t])
        nom = np.array([core.simplified_amount_with_topups(
            tt, bolus, prime, k[model], t_to, ((t_bolus, dose),)) for tt in t])

        top, bot = axes[0, col], axes[1, col]
        top.axvline(t_bolus, color=INK_3, lw=0.7, ls=(0, (4, 3)), zorder=1)
        top.plot(t, ref / 1000, color=COLOUR[model], lw=2.0, zorder=3,
                 label="Reference model (superposition)")
        top.plot(t, nom / 1000, color=INK, lw=1.4, ls=(0, (5, 2)), zorder=4,
                 label="Nomogram (axis reset)")
        top.set_title(f"{core.DISPLAY_NAMES[model]}", pad=4)
        tidy(top)
        top.set_xlim(0, 240)
        top.set_ylim(0, 36)

        # Divergence is evaluated only from the supplemental bolus onward, which
        # is the window the top-up analysis reports. Before the bolus the two
        # curves differ by the prime-timing convention (see Figure S1), which is
        # a separate matter and would otherwise dominate the "worst case" here.
        win = t >= t_bolus
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(ref[win] > 0, 100.0 * (ref[win] - nom[win]) / ref[win], np.nan)
        bot.axhline(0, color=INK_3, lw=0.7, zorder=1)
        bot.axvline(t_bolus, color=INK_3, lw=0.7, ls=(0, (4, 3)), zorder=1)
        bot.fill_between(t[win], 0, pct, color=COLOUR[model], alpha=0.22, lw=0, zorder=2)
        bot.plot(t[win], pct, color=COLOUR[model], lw=1.8, zorder=3)
        tidy(bot)
        bot.set_xlim(0, 240)
        bot.set_ylim(-12, 78)
        bot.set_xlabel("Time from induction (min)")

        j = int(np.nanargmax(np.abs(pct)))
        bot.annotate(f"worst {pct[j]:+.1f}%", xy=(t[win][j], pct[j]),
                     xytext=(t[win][j] - 88, pct[j] + 13),
                     fontsize=7.5, color=INK_2,
                     arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.7))
        bot.axvspan(0, t_bolus, color=GRID, alpha=0.45, lw=0, zorder=0)

    for col in (0, 1):
        axes[0, col].text(t_bolus + 4, 31.5, f"{dose:,.0f} IU top-up",
                          fontsize=6.5, color=INK_3)
    axes[1, 0].text(t_bolus / 2, 68, "excluded:\nprime-timing\noffset", fontsize=6.5,
                    color=INK_3, ha="center", va="top", linespacing=1.4)
    axes[0, 0].set_ylabel("Heparin in central\ncompartment (×1000 IU)")
    axes[1, 0].set_ylabel("Difference\n(reference − nomogram, %)")
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0.0, 0.02))
    fig.tight_layout(w_pad=1.6, h_pad=0.9)
    save(fig, outdir, "figure_S3_axis_reset")


def figure_s4(sens, outdir):
    """One-factor-at-a-time sensitivity, on a shared scale.

    A shared x-axis is deliberate. The panels are near-flat for four of the six
    models and that is the finding: the calibrated constant is insensitive to
    the institutional inputs except for Delavenne, where it is not. Per-panel
    axes would hide exactly the thing the figure exists to show.
    """
    labels = {"ibw_mean": "Mean IBW", "ibw_sd": "SD of IBW",
              "t_to_mean": "Mean time to CPB", "t_to_sd": "SD of time to CPB",
              "t_on_mean": "Mean time on CPB", "t_on_sd": "SD of time on CPB"}
    order = ["ibw_mean", "t_on_mean", "t_to_mean", "ibw_sd", "t_on_sd", "t_to_sd"]
    limit = np.ceil(sens["pct_change"].abs().max() / 5) * 5

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True, sharey=True)
    for ax, model in zip(axes.ravel(), MODEL_ORDER):
        sub = sens[sens["model"] == model]
        y = np.arange(len(order))
        for direction, hatch in (("low", None), ("high", "////")):
            vals = [float(sub[(sub["parameter"] == p) &
                              (sub["direction"] == direction)]["pct_change"].iloc[0])
                    for p in order]
            ax.barh(y + (0.19 if direction == "high" else -0.19), vals, height=0.34,
                    color=COLOUR[model], alpha=1.0 if direction == "low" else 0.45,
                    hatch=hatch, edgecolor=SURFACE, linewidth=0.8, zorder=3,
                    label=f"input {'−' if direction == 'low' else '+'}20%")
        ax.axvline(0, color=INK_3, lw=0.8, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([labels[p] for p in order])
        ax.set_title(core.DISPLAY_NAMES[model], pad=4)
        tidy(ax, grid_axis="x")
        ax.set_xlim(-limit, limit)
        ax.invert_yaxis()

        # Top row is an SD parameter, whose bars are negligible in every panel,
        # so this corner is free of marks for all six models.
        worst = sub.loc[sub["pct_change"].abs().idxmax()]
        ax.text(0.03, 0.95, f"largest {worst['pct_change']:+.1f}%",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, color=INK_2)

    for ax in axes[1, :]:
        ax.set_xlabel("Change in calibrated $k$ (%)")
    # Neutral swatches: the legend encodes direction (solid vs hatched) only.
    # Borrowing a panel's hue would imply the colour carried meaning here too.
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=INK_2, label="input −20%"),
                        Patch(facecolor=INK_2, alpha=0.45, hatch="////",
                              edgecolor=SURFACE, label="input +20%")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.035),
               handlelength=1.6)
    fig.tight_layout(w_pad=1.3, h_pad=1.1)
    save(fig, outdir, "figure_S4_input_sensitivity")


# ==========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default="outputs/frozen")
    ap.add_argument("--outdir", default="outputs/frozen/figures")
    ap.add_argument("--cohort", default=ps.DEFAULT_COHORT,
                    choices=sorted(ps.CANONICAL_COHORTS))
    args = ap.parse_args()

    indir, outdir = Path(args.indir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    spec = ps.CANONICAL_COHORTS[args.cohort]

    kt = pd.read_csv(indir / "01_calibrated_k_all_models.csv").set_index("model")
    k = kt["k"].to_dict()
    k_mc = {m: (r["mc_interval_low"], r["mc_interval_high"]) for m, r in kt.iterrows()}

    pu = pd.read_csv(indir / "09_parameter_uncertainty.csv").set_index("model")
    k_pu = {m: (r["k_p2_5"], r["k_p97_5"])
            for m, r in pu.iterrows() if bool(r["available"])}

    sens = pd.read_csv(indir / "06_input_sensitivity.csv")

    print(f"Cohort: {spec.describe()}")
    print(f"Reading {indir}, writing {outdir}\n")
    figure_s1(k, spec, outdir)
    figure_s2(k, k_mc, k_pu, spec, outdir)
    figure_s3(k, spec, outdir)
    figure_s4(sens, outdir)
    print("\nDone.")


if __name__ == "__main__":
    main()
