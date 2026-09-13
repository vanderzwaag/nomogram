"""
Nomogram geometry and rendering, with no PyNomo, PyX, LaTeX or Ghostscript.

The printed nomogram and the interactive one are drawn from this one module, so
they cannot disagree. Previously there were two independent implementations fed
by two different decay constants -- PyNomo's ``build_nomogram(k_best)`` for the
PDF and a hand-rolled matplotlib chart reading ``k_mu`` from the lookup table for
the screen -- which is the same "several constants in circulation" defect that
produced the irreconcilable numbers in the submitted manuscript, surviving in
the drawing layer.

The geometry
------------
The nomogram solves a single relation::

    R = D0 * exp(-k * t)        equivalently    log10 R = log10 D0 - k*t/ln 10

which is an alignment chart with three parallel axes: the total load on the
left, elapsed time in the middle, residual on the right. Writing ``m`` for the
scale modulus (length per decade) and ``H`` for the axis height::

    y_dose(D0) = m * (log10 D0 - log10 D0_min)
    y_res(R)   = H - m * (log10 R - log10 R_min)          [descending]
    y_time(t)  = (y_dose(D0) + y_res(R)) / 2

The third line is what makes the chart work: substituting the relation above,
the D0 terms cancel, so the midpoint of any valid (dose, residual) pair depends
on elapsed time alone. That cancellation requires the residual axis to run
downwards while the dose axis runs upwards -- reverse either one and the time
axis stops being a function of time. ``test_nomogram_render.py`` asserts the
collinearity directly rather than trusting the derivation.

Because ``k`` is baked into the spacing of the time axis, a printed nomogram
cannot carry an uncertainty band: a different ``k`` is a differently-spaced time
axis, not a wider line. That is why the band is a feature of the interactive
tool only, and the print footer says so (EB-2, EB-8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import matplotlib
import numpy as np

RENDER_VERSION = "2.0.0"

# Print palette, consistent with the supplementary figures.
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"
RULE, ACCENT, BAND = "#c9c8c2", "#2a78d6", "#e34948"

LN10 = math.log(10.0)


@dataclass(frozen=True)
class NomogramGeometry:
    """Axis placement for one calibrated decay constant."""

    k: float
    d0_min: float
    d0_max: float
    t_min: float
    t_max: float
    height: float
    modulus: float
    r_min: float
    r_max: float
    # The residual axis always spans more decades than the dose axis, by
    # k*(t_max - t_min)/ln 10, so the dose axis is shorter. Bottom-anchoring it
    # leaves the top third of the sheet empty; this offset centres it. Adding a
    # constant to the dose axis shifts the time axis by half as much and leaves
    # the alignment property intact, which test_nomogram_render asserts.
    dose_offset: float = 0.0
    x_dose: float = 0.0
    x_time: float = 1.0
    x_res: float = 2.0

    # ----------------------------------------------------------- mappings
    def y_dose(self, d0):
        return self.dose_offset + self.modulus * (np.log10(d0) - math.log10(self.d0_min))

    def y_res(self, r):
        return self.height - self.modulus * (np.log10(r) - math.log10(self.r_min))

    def y_time(self, t):
        """Position of elapsed time on the middle axis.

        Derived by substituting the decay relation into the midpoint of the two
        outer axes; the dose terms cancel exactly.
        """
        const = (self.dose_offset + self.height
                 - self.modulus * math.log10(self.d0_min)
                 + self.modulus * math.log10(self.r_min))
        return (const + self.modulus * self.k * np.asarray(t, dtype=float) / LN10) / 2.0

    def residual(self, d0, t):
        return np.asarray(d0, dtype=float) * np.exp(-self.k * np.asarray(t, dtype=float))


def build_geometry(k: float, *, d0_min: float, d0_max: float,
                   t_min: float = 0.0, t_max: float = 120.0,
                   height: float = 10.0) -> NomogramGeometry:
    """Size the chart so both outer axes fit exactly within ``height``.

    The residual axis always spans more decades than the dose axis -- by
    k*(t_max - t_min)/ln 10 -- so the modulus is set by the residual axis and
    the dose axis occupies the corresponding fraction of the height.
    """
    if not d0_max > d0_min > 0:
        raise ValueError("require 0 < d0_min < d0_max")
    if not t_max > t_min >= 0:
        raise ValueError("require 0 <= t_min < t_max")
    if k <= 0:
        raise ValueError("require k > 0")

    r_min = d0_min * math.exp(-k * t_max)
    r_max = d0_max * math.exp(-k * t_min)
    decades_res = math.log10(r_max / r_min)
    decades_dose = math.log10(d0_max / d0_min)
    modulus = height / decades_res
    return NomogramGeometry(
        k=k, d0_min=d0_min, d0_max=d0_max, t_min=t_min, t_max=t_max,
        height=height, modulus=modulus, r_min=r_min, r_max=r_max,
        dose_offset=(height - modulus * decades_dose) / 2.0,
    )


# ==========================================================================
# TICKS
# ==========================================================================

def _nice_ticks(lo: float, hi: float, targets: Sequence[int]) -> list:
    """Round tick values inside [lo, hi] at the coarsest step that still gives
    at least ``targets[0]`` marks, plus the finer steps for minor ticks."""
    out = []
    for step in targets:
        first = math.ceil(lo / step) * step
        out.append([v for v in np.arange(first, hi + step * 1e-9, step) if lo <= v <= hi])
    return out


def _axis_ticks(lo, hi):
    """(major, minor) tick values for a heparin axis, in IU."""
    span = hi - lo
    major_step = 5000 if span > 20000 else (2000 if span > 8000 else 1000)
    minor_step = major_step // 5
    major, minor = _nice_ticks(lo, hi, [major_step, minor_step])
    minor = [v for v in minor if v not in set(major)]
    return major, minor


# ==========================================================================
# DRAWING
# ==========================================================================

def draw_nomogram(geom: NomogramGeometry, ax=None, *,
                  patient: dict | None = None,
                  residual_band: tuple | None = None,
                  title: str | None = None,
                  time_major: float = 15.0, time_minor: float = 5.0):
    """Draw the alignment chart onto ``ax`` (created if not supplied).

    ``patient``
        ``{"total": IU, "elapsed": min, "label": str}`` draws the straightedge.
    ``residual_band``
        ``(low_IU, high_IU)`` shades a range on the residual axis. Interactive
        use only: pass ``None`` for anything printed, since the printed chart's
        time axis is spaced for one specific k.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 8.2))
    H = geom.height

    def scale(x, values_major, values_minor, ypos, side, fmt, extent=None):
        lo, hi = extent if extent else (0.0, H)
        ax.vlines(x, lo, hi, color=INK, lw=1.1, zorder=3)
        for v in values_minor:
            y = ypos(v)
            if -1e-9 <= y <= H + 1e-9:
                ax.hlines(y, x - 0.022, x + 0.022, color=INK_3, lw=0.6, zorder=4)
        for v in values_major:
            y = ypos(v)
            if -1e-9 <= y <= H + 1e-9:
                ax.hlines(y, x - 0.05, x + 0.05, color=INK, lw=1.0, zorder=4)
                ax.text(x + (0.1 if side == "r" else -0.1), y, fmt(v),
                        ha="left" if side == "r" else "right", va="center",
                        fontsize=7.5, color=INK_2, zorder=5)

    iu = lambda v: f"{v:,.0f}"
    d_major, d_minor = _axis_ticks(geom.d0_min, geom.d0_max)
    r_major, r_minor = _axis_ticks(geom.r_min, geom.r_max)
    t_major = list(np.arange(geom.t_min, geom.t_max + 1e-9, time_major))
    t_minor = [v for v in np.arange(geom.t_min, geom.t_max + 1e-9, time_minor)
               if v not in set(t_major)]

    scale(geom.x_dose, d_major, d_minor, geom.y_dose, "l", iu,
          extent=(float(geom.y_dose(geom.d0_min)), float(geom.y_dose(geom.d0_max))))
    scale(geom.x_res, r_major, r_minor, geom.y_res, "r", iu)
    # The time axis occupies only the part of the height its own range spans.
    ax.vlines(geom.x_time, float(geom.y_time(geom.t_min)),
              float(geom.y_time(geom.t_max)), color=INK, lw=1.1, zorder=3)
    for v in t_minor:
        y = float(geom.y_time(v))
        ax.hlines(y, geom.x_time - 0.022, geom.x_time + 0.022,
                  color=INK_3, lw=0.6, zorder=4)
    for v in t_major:
        y = float(geom.y_time(v))
        ax.hlines(y, geom.x_time - 0.05, geom.x_time + 0.05, color=INK, lw=1.0, zorder=4)
        ax.text(geom.x_time + 0.09, y, f"{v:.0f}", ha="left", va="center",
                fontsize=7.5, color=INK_2, zorder=5)

    # The residual scale runs downwards. That is not a stylistic choice: the
    # dose terms only cancel out of the time axis if one outer scale is
    # reversed, so the chart would not work otherwise. It is called out because
    # a reader who assumes "higher is more" would misread the output.
    ax.annotate("", xy=(geom.x_res - 0.30, 0.08 * H), xytext=(geom.x_res - 0.30, 0.30 * H),
                arrowprops=dict(arrowstyle="-|>", color=INK_3, lw=0.9))
    ax.text(geom.x_res - 0.36, 0.19 * H, "increasing\nresidual", fontsize=6.5,
            color=INK_3, va="center", ha="right", linespacing=1.4)

    for x, label, sub in ((geom.x_dose, "Total heparin", "bolus + prime (IU)"),
                          (geom.x_time, "Elapsed time", "minutes"),
                          (geom.x_res, "Residual heparin", "IU, reading downwards")):
        ax.text(x, H + 0.42, label, ha="center", fontsize=9,
                fontweight="bold", color=INK)
        ax.text(x, H + 0.14, sub, ha="center", fontsize=7, color=INK_3)

    if residual_band:
        lo, hi = sorted(residual_band)
        y_lo, y_hi = float(geom.y_res(hi)), float(geom.y_res(lo))
        ax.fill_betweenx([np.clip(y_lo, 0, H), np.clip(y_hi, 0, H)],
                         geom.x_res + 0.055, geom.x_res + 0.16,
                         color=BAND, alpha=0.3, lw=0, zorder=2)

    if patient:
        d0, t = float(patient["total"]), float(patient["elapsed"])
        r = float(geom.residual(d0, t))
        y_d, y_r = float(geom.y_dose(d0)), float(geom.y_res(r))
        ax.plot([geom.x_dose, geom.x_res], [y_d, y_r], color=ACCENT, lw=1.3,
                zorder=6, solid_capstyle="round")
        for x, y in ((geom.x_dose, y_d), (geom.x_time, float(geom.y_time(t))),
                     (geom.x_res, y_r)):
            ax.plot([x], [y], "o", ms=4.5, color=ACCENT,
                    markeredgecolor="white", markeredgewidth=0.9, zorder=7)
        # Offset clear of the axis tick labels, which sit at x_res + 0.1.
        ax.text(geom.x_res + 0.40, y_r, f"{r:,.0f} IU", va="center",
                fontsize=8.5, color=ACCENT, fontweight="bold", zorder=7)
        ax.annotate("", xy=(geom.x_res + 0.37, y_r), xytext=(geom.x_res + 0.06, y_r),
                    arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.7, alpha=0.5),
                    zorder=6)

    if title:
        ax.text(geom.x_time, H + 1.05, title, ha="center", fontsize=10,
                fontweight="bold", color=INK)

    ax.set_xlim(-0.68, 3.00)
    ax.set_ylim(-0.5, H + 1.5)
    ax.axis("off")
    return ax.figure


def render_pdf(geom: NomogramGeometry, path: str, *, footer_lines: Iterable[str],
               title: str | None = None, patient: dict | None = None) -> str:
    """Write a print-ready A4 nomogram. No band is drawn -- see the module note."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.27, 11.69))          # A4 portrait
    ax = fig.add_axes([0.10, 0.17, 0.80, 0.74])
    draw_nomogram(geom, ax=ax, title=title, patient=patient, residual_band=None)

    import textwrap

    y = 0.125
    for i, line in enumerate(footer_lines):
        for part in textwrap.wrap(line, width=115) or [""]:
            fig.text(0.10, y, part, fontsize=6.6,
                     color=INK_2 if i == 0 else INK_3, ha="left")
            y -= 0.0155
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def footer_for(model_name: str, k: float, spec_text: str, *, seed,
               interval: tuple | None = None, version: str = "",
               commit: str | None = None) -> list:
    """The provenance block printed under every nomogram (EB-6, EB-8)."""
    interval_txt = (f"  Monte Carlo sampling interval {interval[0]:.5f}-{interval[1]:.5f}."
                    if interval else "")
    return [
        f"Reference model: {model_name}.  k = {k:.5f} /min "
        f"(apparent half-life {math.log(2) / k:.0f} min).{interval_txt}",
        spec_text,
        f"Seed {seed}." + (f"  Commit {commit}." if commit else "") +
        (f"  {version}" if version else ""),
        "Research and educational instrument. Reading the residual load requires no "
        "calculation; converting it to a protamine dose requires the institutional ratio.",
        "This printed chart carries no uncertainty band: the elapsed-time axis is "
        "spaced for this one decay constant, so a different constant is a different "
        "chart rather than a wider line. The interval above is Monte Carlo sampling "
        "precision only.",
    ]
