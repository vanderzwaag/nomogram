"""
Parameter spaces, calibration grids and simulated institutions.

Single source of truth for every input range quoted in the manuscript, so the
Methods section, Table 1, the lookup-table generator and the analysis runner
cannot describe different things (EB-3, EB-8).

Four questions the reviewers raised are settled here rather than in prose:

EB-3  Which cohort is THE representative cohort? Two definitions were in
      circulation. See CANONICAL_COHORTS and the note below.
EB-4  Are the full input ranges and their boundary combinations covered?
      See ADULT_GRID / PAEDIATRIC_GRID and boundary_cohort().
EB-4  Is transportability tested? See SHIFTED_INSTITUTIONS.
EB-4  Is the paediatric Jia model run in a paediatric space?
      See PAEDIATRIC_GRID and CANONICAL_COHORTS["paediatric"].
"""

from __future__ import annotations

import itertools
from typing import Dict, Sequence

import pandas as pd

from calibration import CohortSpec, grid_cohort

# ==========================================================================
# THE TWO COHORT DEFINITIONS THAT WERE IN CIRCULATION
# ==========================================================================
#
# The submitted manuscript described the representative cohort twice, with
# different numbers, and attached the same agreement statistics to both.
# Re-running the analysis identifies unambiguously which one produced the
# published statistics:
#
#   IBW 70, t_to 15, t_on 60  -> limits of agreement about -188 to +188 IU
#   IBW 75, t_to 20, t_on 65  -> limits of agreement about -240 to +240 IU
#
# The submitted values (-243 to +232 in the text, -232.91 to +244.16 in the
# Figure 2 legend) are the 75 / 20 / 65 cohort. Table 1's 70 / 15 / 60 is the
# description that does not match the statistics -- it describes the lookup-table
# NODE the tool snaps to, not the cohort the simulation drew.
#
# "grid_snapped" is the default because it is the only choice under which the
# calibration cohort, Table 1 and the k that the printed nomogram actually uses
# are the same thing. Choosing "abstract" keeps the published numbers but leaves
# the tool reading a k calibrated at a neighbouring grid node, which then has to
# be stated explicitly.

CANONICAL_COHORTS: Dict[str, CohortSpec] = {
    "grid_snapped": CohortSpec(
        name="grid_snapped_70_15_60",
        dose_per_kg=400.0, prime_heparin=5000.0,
        ibw_mean=70.0, ibw_sd=10.0,
        t_to_mean=15.0, t_to_sd=3.75,
        t_on_mean=60.0, t_on_sd=15.0,
        population="adult",
    ),
    "abstract": CohortSpec(
        name="abstract_75_20_65",
        dose_per_kg=400.0, prime_heparin=5000.0,
        ibw_mean=75.0, ibw_sd=10.0,
        t_to_mean=20.0, t_to_sd=5.0,
        t_on_mean=65.0, t_on_sd=18.0,
        population="adult",
    ),
    # Paediatric counterpart for the Jia model (EB-4, R1 p11 L46).
    # AUTHOR DECISION REQUIRED: these are defensible defaults for a mixed
    # paediatric cardiac-surgical population, not values taken from the Jia
    # publication. Confirm or replace them before resubmission.
    "paediatric": CohortSpec(
        name="paediatric_10kg",
        dose_per_kg=400.0, prime_heparin=1500.0,
        ibw_mean=10.0, ibw_sd=4.0,
        t_to_mean=10.0, t_to_sd=3.0,
        t_on_mean=90.0, t_on_sd=30.0,
        population="paediatric",
        ibw_min=2.5, ibw_max=60.0,
    ),
}

DEFAULT_COHORT = "grid_snapped"


# ==========================================================================
# CALIBRATION GRIDS
# ==========================================================================
# These are the nodes the lookup tables are generated on, and the ranges the
# Methods section must state (EB-8: "define the calibration grid mentioned only
# in Table 1").

ADULT_GRID = {
    "dose_per_kg": [250, 300, 350, 400, 450, 500, 550, 600],   # IU/kg
    "ibw": [40, 55, 70, 85, 100, 115],                          # kg
    "prime": [0, 5000, 10000],                                  # IU
    "time_to_cpb": [5, 15, 25, 35],                             # min
    "time_on_cpb": [30, 60, 90, 120],                           # min
}

# Paediatric space for the Jia model. AUTHOR DECISION REQUIRED -- see above.
PAEDIATRIC_GRID = {
    "dose_per_kg": [300, 350, 400],
    "ibw": [3, 6, 10, 20, 40],
    "prime": [0, 1000, 2500],
    "time_to_cpb": [5, 15, 25],
    "time_on_cpb": [30, 60, 90, 120, 180],
}


def grid_for(population: str) -> Dict[str, Sequence[float]]:
    if population == "paediatric":
        return PAEDIATRIC_GRID
    if population == "adult":
        return ADULT_GRID
    raise ValueError(f"population must be 'adult' or 'paediatric', got {population!r}")


def full_range_cohort(population: str = "adult") -> pd.DataFrame:
    """Every node of the calibration grid -- the full-range coverage evaluation."""
    g = grid_for(population)
    return grid_cohort(g["dose_per_kg"], g["ibw"], g["prime"],
                       g["time_to_cpb"], g["time_on_cpb"])


def boundary_cohort(population: str = "adult") -> pd.DataFrame:
    """Only the corners of the input hyper-rectangle (EB-4).

    Boundary combinations are where a population-average decay constant is
    least likely to hold -- the smallest patient given the largest dose with the
    longest bypass, and so on -- and they are exactly the combinations a
    quantile-based cohort almost never samples.
    """
    g = grid_for(population)
    extremes = {k: [min(v), max(v)] for k, v in g.items()}
    df = grid_cohort(extremes["dose_per_kg"], extremes["ibw"], extremes["prime"],
                     extremes["time_to_cpb"], extremes["time_on_cpb"])
    df["corner"] = [
        f"dose={d:g},ibw={w:g},prime={p:g},t_to={a:g},t_on={b:g}"
        for d, w, p, a, b in zip(df["dose_per_kg"], df["ibw"], df["heparin_prime"],
                                 df["time_to_cpb"], df["time_on_cpb"])
    ]
    return df


# ==========================================================================
# SHIFTED SIMULATED INSTITUTIONS (EB-4, transportability)
# ==========================================================================
# Each is a deliberate departure from the calibration cohort in one clinically
# meaningful direction. Calibrating at the canonical cohort and then evaluating
# on these tests whether an institution-specific k transports -- which is the
# pipeline's central claim.

SHIFTED_INSTITUTIONS: Dict[str, CohortSpec] = {
    "heavier_population": CohortSpec(
        "heavier_population", 400.0, 5000.0, 95.0, 15.0, 15.0, 3.75, 60.0, 15.0),
    "lighter_population": CohortSpec(
        "lighter_population", 400.0, 5000.0, 55.0, 8.0, 15.0, 3.75, 60.0, 15.0),
    "long_bypass": CohortSpec(
        "long_bypass", 400.0, 5000.0, 70.0, 10.0, 15.0, 3.75, 110.0, 25.0),
    "short_bypass": CohortSpec(
        "short_bypass", 400.0, 5000.0, 70.0, 10.0, 15.0, 3.75, 35.0, 10.0),
    "high_dose": CohortSpec(
        "high_dose", 550.0, 10000.0, 70.0, 10.0, 15.0, 3.75, 60.0, 15.0),
    "low_dose_no_prime": CohortSpec(
        "low_dose_no_prime", 300.0, 0.0, 70.0, 10.0, 15.0, 3.75, 60.0, 15.0),
    "slow_start": CohortSpec(
        "slow_start", 400.0, 5000.0, 70.0, 10.0, 35.0, 8.0, 60.0, 15.0),
    "wide_case_mix": CohortSpec(
        "wide_case_mix", 400.0, 5000.0, 75.0, 20.0, 20.0, 10.0, 75.0, 35.0),
}

PAEDIATRIC_INSTITUTIONS: Dict[str, CohortSpec] = {
    "neonatal": CohortSpec(
        "neonatal", 400.0, 1000.0, 3.5, 0.8, 8.0, 2.0, 120.0, 40.0,
        population="paediatric", ibw_min=2.0, ibw_max=10.0),
    "infant": CohortSpec(
        "infant", 400.0, 1500.0, 7.0, 2.0, 10.0, 3.0, 100.0, 35.0,
        population="paediatric", ibw_min=3.0, ibw_max=20.0),
    "older_child": CohortSpec(
        "older_child", 350.0, 2500.0, 25.0, 8.0, 12.0, 4.0, 80.0, 30.0,
        population="paediatric", ibw_min=10.0, ibw_max=60.0),
}


def institutions_for(population: str = "adult") -> Dict[str, CohortSpec]:
    return PAEDIATRIC_INSTITUTIONS if population == "paediatric" else SHIFTED_INSTITUTIONS


def describe_grids() -> str:
    """Methods-ready description of the calibration grid (EB-8)."""
    lines = []
    for label, g in (("Adult", ADULT_GRID), ("Paediatric (Jia)", PAEDIATRIC_GRID)):
        n = 1
        for v in g.values():
            n *= len(v)
        lines.append(f"{label} calibration grid ({n} nodes):")
        lines.append(f"  heparin dose   : {g['dose_per_kg']} IU/kg")
        lines.append(f"  ideal body wt  : {g['ibw']} kg")
        lines.append(f"  prime heparin  : {g['prime']} IU")
        lines.append(f"  time to CPB    : {g['time_to_cpb']} min")
        lines.append(f"  time on CPB    : {g['time_on_cpb']} min")
    return "\n".join(lines)
