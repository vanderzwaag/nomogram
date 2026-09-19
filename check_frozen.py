#!/usr/bin/env python3
"""
Check that a fresh run reproduces the frozen one.

    python run_analysis.py --seed 20260912 --outdir /tmp/check
    python check_frozen.py /tmp/check                 # numbers agree
    python check_frozen.py /tmp/check --exact         # files are identical

Two claims, and they are not the same claim.

``--exact`` asserts every CSV is byte-identical. That is a claim about one
machine, not about the analysis. Re-running on the same machine in the pinned
environment reproduces the archive exactly, and that is worth being able to
check. Across machines it does not hold and cannot be made to: the last bit of
a transcendental function is not specified by IEEE 754, so a different libm,
or a different build of one, moves a result by one unit in the last place, and
everything downstream of it follows.

The default asserts instead that every number agrees to a relative tolerance,
which is the claim that holds anywhere and the one that fails if a result has
actually moved. Both modes print the largest relative difference they found,
so a pass says how close it came and a failure says whether it is rounding or
a result moving.

A difference is judged against the larger of the value itself and the biggest
value in its column. Some columns are numerical zero by construction -- a
recalibrated bias is a mean of a thousand differences of a few hundred IU that
almost cancel, so the whole column spans 0.045 IU -- and dividing by the value
alone turns one unit in the last place into an apparent 1e-10 disagreement.

The manifest is compared by its decisions rather than its bytes, since it
records the timestamp and commit of its own run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FROZEN = Path(__file__).resolve().parent / "outputs" / "frozen"

MANIFEST_KEYS = ("decisions", "sample_sizes", "parameter_provenance", "csv_outputs")


def compare_csv(frozen: Path, fresh: Path, tolerance: float | None,
                worst: dict) -> list:
    """Differences between one pair of CSVs, as human-readable lines.

    ``worst`` accumulates the largest relative difference seen anywhere, so a
    run that passes still says how close it came and a run that fails says how
    far off it is. Reporting only pass or fail leaves the one question that
    matters -- is this rounding or a result moving? -- unanswered.
    """
    if not fresh.exists():
        return [f"{frozen.name}: missing from the fresh run"]
    identical = frozen.read_bytes() == fresh.read_bytes()
    if identical and tolerance is not None:
        return []

    a, b = pd.read_csv(frozen), pd.read_csv(fresh)
    if list(a.columns) != list(b.columns):
        return [f"{frozen.name}: columns differ"]
    if len(a) != len(b):
        return [f"{frozen.name}: {len(a)} rows frozen, {len(b)} fresh"]

    out = []
    for column in a.columns:
        x, y = a[column], b[column]
        if not (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)):
            if not x.equals(y):
                out.append(f"{frozen.name}: text column '{column}' differs")
            continue
        xv, yv = x.to_numpy(float), y.to_numpy(float)
        if not np.array_equal(np.isnan(xv), np.isnan(yv)):
            out.append(f"{frozen.name}: '{column}' differs in which values are missing")
            continue
        finite = ~np.isnan(xv)
        # Judge each difference against the larger of the value itself and the
        # biggest value in its column. Dividing by the value alone explodes on
        # a near-cancellation: the recalibrated bias is a mean of a thousand
        # differences of a few hundred IU that almost cancel, so the whole
        # column spans 0.045 IU and one ulp of wobble reads as 1e-10. Measured
        # against the column it is 2e-11, which is what it is -- rounding.
        scale = float(np.max(np.abs(xv[finite]))) if finite.any() else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            absdiff = np.abs(xv[finite] - yv[finite])
            rel = absdiff / np.maximum(np.abs(xv[finite]), max(scale, 1e-300))
        if rel.size and rel.max() > worst["value"]:
            row = int(np.argmax(rel))
            worst.update(value=float(rel.max()), where=f"{frozen.name} '{column}' row {row}",
                         frozen=float(xv[finite][row]), fresh=float(yv[finite][row]))
        if tolerance is not None and rel.size and rel.max() > tolerance:
            row = int(np.argmax(rel))
            out.append(f"{frozen.name}: '{column}' differs by {rel.max():.3e} "
                       f"(row {row}: {xv[finite][row]!r} vs {yv[finite][row]!r})")
    if tolerance is None and not identical:
        out.append(f"{frozen.name}: differs byte-for-byte")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fresh", help="directory holding the run to check")
    p.add_argument("--frozen", default=str(FROZEN), help="the archived run")
    p.add_argument("--exact", action="store_true",
                   help="require byte-identical files, not just equal numbers")
    p.add_argument("--tolerance", type=float, default=1e-9,
                   help="relative tolerance when not --exact (default 1e-9)")
    args = p.parse_args()

    frozen, fresh = Path(args.frozen), Path(args.fresh)
    tolerance = None if args.exact else args.tolerance

    drift = []
    worst = {"value": 0.0, "where": None, "frozen": None, "fresh": None}
    frozen_csvs = sorted(frozen.glob("*.csv"))
    if not frozen_csvs:
        sys.exit(f"no frozen CSVs found in {frozen}")
    for path in frozen_csvs:
        drift += compare_csv(path, fresh / path.name, tolerance, worst)

    extra = {q.name for q in fresh.glob("*.csv")} - {q.name for q in frozen_csvs}
    for name in sorted(extra):
        drift.append(f"{name}: produced by the fresh run but not in the archive")

    a = json.loads((frozen / "manifest.json").read_text())
    b = json.loads((fresh / "manifest.json").read_text())
    for key in MANIFEST_KEYS:
        if a.get(key) != b.get(key):
            drift.append(f"manifest['{key}'] differs")

    if worst["where"] is None:
        magnitude = "every number is bit-identical"
    else:
        magnitude = (f"largest relative difference {worst['value']:.3e} at "
                     f"{worst['where']} ({worst['frozen']!r} vs {worst['fresh']!r})")

    if drift:
        print("the fresh run does not reproduce the archive:", file=sys.stderr)
        for line in drift:
            print(f"  {line}", file=sys.stderr)
        print(f"  {magnitude}", file=sys.stderr)
        return 1

    how = "byte-for-byte" if args.exact else f"to a relative tolerance of {args.tolerance:g}"
    print(f"all {len(frozen_csvs)} frozen CSVs and the manifest decisions reproduce {how}")
    print(f"  {magnitude}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
