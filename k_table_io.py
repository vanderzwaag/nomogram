"""
Reading and writing the per-model decay-constant lookup tables.

The tables are plain CSV with a commented JSON header, rather than pickles.
Unpickling executes arbitrary code, so a pickle shipped in a repository is a
file every user of a clone or fork must trust; a lookup table has no need of
that. CSV is also diffable, so a change to a table shows up in review, and it
opens in any spreadsheet for inspection.

File layout::

    # Decay-constant lookup table. Do not edit by hand.
    # metadata: {"model": "lanoiselee", "seed": 20260912, ...}
    dose_per_kg,ibw,time_to_cpb,time_on_cpb,prime,mu,lo,hi
    250,40,5,30,0,0.007346249647363108,0.007285274735344699,0.007392267661986352

Floats are written with ``repr``, which round-trips exactly in Python, so a
table converted from the pickles it replaces holds bit-identical values.

In memory a table keeps the shape the dashboard has always used: a dict mapping
``(dose_per_kg, ibw, time_to_cpb, time_on_cpb, prime)`` to ``{"mu", "lo", "hi"}``,
plus a ``"__metadata__"`` entry recording the seed, grid and conventions it was
built under. A table without that entry cannot be reproduced and must not be
used for anything quoted (EB-6).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

METADATA_KEY = "__metadata__"
KEY_FIELDS = ("dose_per_kg", "ibw", "time_to_cpb", "time_on_cpb", "prime")
VALUE_FIELDS = ("mu", "lo", "hi")
_HEADER = "# Decay-constant lookup table. Do not edit by hand."
_META_PREFIX = "# metadata: "


def table_path(model: str) -> str:
    """Filename for one model's table. ASCII, so accented names still resolve."""
    return f"k_table_v2_{model}.csv"


def _number(text: str):
    """Grid coordinates stay integers; everything else is a float.

    The keys are tuples used for exact dict lookup, so 40 and 40.0 are not
    interchangeable -- they hash alike but print differently and would make the
    written table drift from the grid it was built on.
    """
    return int(text) if text.lstrip("-").isdigit() else float(text)


def write_k_table(table: Dict, path: str) -> str:
    """Write one lookup table, sorted by grid coordinate so diffs are stable."""
    meta = table.get(METADATA_KEY, {})
    rows = sorted(k for k in table if k != METADATA_KEY)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(_HEADER + "\n")
        fh.write(_META_PREFIX + json.dumps(meta, sort_keys=True, default=str) + "\n")
        writer = csv.writer(fh)
        writer.writerow(KEY_FIELDS + VALUE_FIELDS)
        for key in rows:
            entry = table[key]
            writer.writerow([repr(v) if isinstance(v, float) else v for v in key]
                            + [repr(float(entry[f])) for f in VALUE_FIELDS])
    return path


def read_k_table(path: str) -> Dict | None:
    """Read one lookup table, or return None if it is not there."""
    p = Path(path)
    if not p.exists():
        return None

    table: Dict = {}
    with p.open(newline="", encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith(_META_PREFIX):
            table[METADATA_KEY] = json.loads(line[len(_META_PREFIX):])
        if not line.startswith("#"):
            body_start = i
            break

    reader = csv.DictReader(lines[body_start:])
    for row in reader:
        key = tuple(_number(row[f]) for f in KEY_FIELDS)
        table[key] = {f: float(row[f]) for f in VALUE_FIELDS}
    return table
