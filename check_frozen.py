#!/usr/bin/env python3
"""
Check that a fresh run reproduces the frozen one.

    python run_analysis.py --seed 20260912 --outdir /tmp/check
    python check_frozen.py /tmp/check                 # numbers agree
    python check_frozen.py /tmp/check --exact         # files are identical

Two claims, and they are not the same claim.

``--exact`` asserts every CSV is byte-identical. That holds only in the
environment the archive was produced in, which ``requirements-frozen.txt``
pins: a different scipy converges its optimiser a fraction differently, and a
different pandas can format a float differently, so bytes drift even though
nothing about the analysis has changed.

The default asserts instead that every number agrees to a relative tolerance.
That is the claim worth making across environments, and it is the one that
fails if a result has actually moved. Measured drift between the archive's
environment and one a major release behind (numpy 2.2.6, scipy 1.15.3, pandas
2.3.3) is 5e-14 at worst, in a p-value -- fourteen orders of magnitude below
anything reported.

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


def compare_csv(frozen: Path, fresh: Path, tolerance: float | None) -> list:
    """Differences between one pair of CSVs, as human-readable lines."""
    if not fresh.exists():
        return [f"{frozen.name}: missing from the fresh run"]
    if frozen.read_bytes() == fresh.read_bytes():
        return []
    if tolerance is None:
        return [f"{frozen.name}: differs byte-for-byte"]

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
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(xv[finite] - yv[finite]) / np.maximum(np.abs(xv[finite]), 1e-300)
        if rel.size and rel.max() > tolerance:
            worst = int(np.argmax(rel))
            out.append(f"{frozen.name}: '{column}' differs by {rel.max():.3e} "
                       f"(row {worst}: {xv[finite][worst]!r} vs {yv[finite][worst]!r})")
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
    frozen_csvs = sorted(frozen.glob("*.csv"))
    if not frozen_csvs:
        sys.exit(f"no frozen CSVs found in {frozen}")
    for path in frozen_csvs:
        drift += compare_csv(path, fresh / path.name, tolerance)

    extra = {q.name for q in fresh.glob("*.csv")} - {q.name for q in frozen_csvs}
    for name in sorted(extra):
        drift.append(f"{name}: produced by the fresh run but not in the archive")

    a = json.loads((frozen / "manifest.json").read_text())
    b = json.loads((fresh / "manifest.json").read_text())
    for key in MANIFEST_KEYS:
        if a.get(key) != b.get(key):
            drift.append(f"manifest['{key}'] differs")

    if drift:
        print("the fresh run does not reproduce the archive:", file=sys.stderr)
        for line in drift:
            print(f"  {line}", file=sys.stderr)
        return 1

    how = "byte-for-byte" if args.exact else f"to a relative tolerance of {args.tolerance:g}"
    print(f"all {len(frozen_csvs)} frozen CSVs and the manifest decisions reproduce {how}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
