"""Round-trip and provenance checks for the lookup-table format."""

import json
from pathlib import Path

import pytest

import k_table_io as io


def _sample_table():
    return {
        io.METADATA_KEY: {"model": "lanoiselee", "seed": 20260912,
                          "grid": {"ibw": [40, 70]}, "k_search_bounds": [0.001, 0.03]},
        (250, 40, 5, 30, 0): {"mu": 0.007346249647363108,
                              "lo": 0.007285274735344699,
                              "hi": 0.007392267661986352},
        (600, 115, 35, 120, 10000): {"mu": 0.0042, "lo": 0.004, "hi": 0.0044},
    }


def test_round_trip_is_exact(tmp_path):
    """Floats are written with repr, so a table survives a save and reload
    bit-for-bit. Anything less and the shipped k would drift from the
    calibrated one every time the tables were rewritten."""
    original = _sample_table()
    path = tmp_path / "k_table_v2_lanoiselee.csv"
    io.write_k_table(original, str(path))
    back = io.read_k_table(str(path))

    assert set(back) == set(original)
    assert back[io.METADATA_KEY] == original[io.METADATA_KEY]
    for key, entry in original.items():
        if key == io.METADATA_KEY:
            continue
        for field in io.VALUE_FIELDS:
            assert back[key][field] == entry[field], f"{key} {field} drifted"


def test_grid_coordinates_stay_integers(tmp_path):
    """The keys are looked up exactly, so 40 must not come back as 40.0."""
    path = tmp_path / "k_table_v2_m.csv"
    io.write_k_table(_sample_table(), str(path))
    for key in io.read_k_table(str(path)):
        if key != io.METADATA_KEY:
            assert all(type(v) is int for v in key), key


def test_missing_table_reads_as_none(tmp_path):
    assert io.read_k_table(str(tmp_path / "absent.csv")) is None


def test_tables_resolve_from_any_working_directory(tmp_path, monkeypatch):
    """The dashboard is launched from wherever the user happens to be; a
    relative path meant it silently found no table and fell back."""
    monkeypatch.chdir(tmp_path)
    assert io.read_k_table(io.table_path("lanoiselee")) is not None


def test_shipped_tables_are_csv_with_metadata():
    """Every shipped table is readable, carries its provenance, and covers the
    whole grid its metadata declares."""
    tables = io.shipped_tables()
    assert tables, "no lookup tables are shipped"
    assert not list(io.TABLE_DIR.parent.rglob("*.pkl")), (
        "a pickled lookup table is back; unpickling executes arbitrary code"
    )

    for path in tables:
        table = io.read_k_table(str(path))
        meta = table[io.METADATA_KEY]
        assert meta["seed"] and meta["grid"], f"{path} has no provenance"
        expected = 1
        for values in meta["grid"].values():
            expected *= len(values)
        assert len(table) - 1 == expected, f"{path} is not the full grid"


def test_no_module_imports_pickle():
    """Checked on the syntax tree, so the docstring explaining why the format
    is not a pickle does not count as a use of one."""
    import ast

    for module in sorted(Path(".").glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert "pickle" not in names, (
                f"{module.name} imports pickle; unpickling a file shipped in "
                "the repository executes whatever is in it"
            )


def test_shipped_tables_are_never_written_by_the_suite():
    """A guard against the mistake that prompted it.

    ``table_path`` returns an absolute path, and ``tmp_path / "/abs/path"``
    discards tmp_path and yields the absolute one -- so a test that wrote to
    ``tmp_path / io.table_path(model)`` silently overwrote the shipped table
    with its two-row fixture. The suite is not allowed to touch these files, so
    their digests must be the same before and after a full run.
    """
    import hashlib
    import subprocess
    import sys

    def digests():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in io.shipped_tables()}

    before = digests()
    assert before, "no lookup tables are shipped"
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_k_table_io.py",
                    "-q", "-p", "no:cacheprovider",
                    "--deselect",
                    "tests/test_k_table_io.py::test_shipped_tables_are_never_written_by_the_suite"],
                   capture_output=True, check=False)
    assert digests() == before, (
        "running the suite modified a shipped lookup table"
    )
