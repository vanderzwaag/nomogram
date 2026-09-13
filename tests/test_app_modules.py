"""Import-level tests for the two Streamlit-facing modules.

Streamlit, PyNomo and ReportLab are not needed to check that the analysis code
inside these modules is wired up correctly, and requiring them would keep the
test suite out of CI. They are stubbed here instead, so a broken import, a lost
re-export or a renamed helper is caught.
"""

import sys
import types

import numpy as np
import pytest


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return mod


@pytest.fixture(scope="module")
def nomogram_models():
    """Import Nomogram_Models with the presentation-layer dependencies stubbed."""

    def passthrough_decorator(*dargs, **dkwargs):
        def wrap(fn):
            return fn
        if dargs and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        return wrap

    st = _stub("streamlit",
               cache_data=passthrough_decorator,
               cache_resource=passthrough_decorator,
               progress=lambda *a, **k: types.SimpleNamespace(
                   progress=lambda *a, **k: None, empty=lambda: None),
               empty=lambda: types.SimpleNamespace(text=lambda *a: None),
               success=lambda *a, **k: None,
               error=lambda *a, **k: None)
    # PyNomo, PyX, LaTeX, Ghostscript, ReportLab and PyPDF2 are gone: the
    # nomogram is plain matplotlib now, so nothing here needs stubbing for it.

    import matplotlib
    matplotlib.use("Agg")

    import Nomogram_Models
    assert st is sys.modules["streamlit"]
    return Nomogram_Models


def test_module_imports_and_reexports_the_public_api(nomogram_models):
    for name in ("prodose_dose", "prodose2_dose", "meesters_dose", "lanoiselee_dose",
                 "delavenne_dose", "jia_dose", "get_reference_dose",
                 "simplified_model_dose", "bland_altman_stats", "bland_altman_score",
                 "find_best_k", "summarize_k_distribution", "sensitivity_analysis",
                 "run_nomogram", "generate_v2_table_deterministic",
                 "heparin_grid", "ibw_grid", "time_to_grid", "time_on_grid",
                 "prime_grid"):
        assert hasattr(nomogram_models, name), f"{name} is no longer exported"


def test_bootstrap_name_is_retained_but_aliased(nomogram_models):
    """The old name still resolves, so nothing breaks, but it points at the
    correctly named function (EB-2)."""
    assert (nomogram_models.bootstrap_k_distribution
            is nomogram_models.monte_carlo_k_distribution)


def test_find_best_k_is_seeded(nomogram_models):
    kwargs = dict(model_name="lanoiselee", n_sim=200)
    a = nomogram_models.find_best_k(400, 5000, 70, 10, 15, 3.75, 60, 15, seed=3, **kwargs)
    b = nomogram_models.find_best_k(400, 5000, 70, 10, 15, 3.75, 60, 15, seed=3, **kwargs)
    c = nomogram_models.find_best_k(400, 5000, 70, 10, 15, 3.75, 60, 15, seed=4, **kwargs)
    assert a == b and a != c


def test_bland_altman_score_matches_the_submitted_objective(nomogram_models):
    """The primary objective must be numerically unchanged, so that the only
    reason k moves is the seeding -- not a redefinition of the loss."""
    rng = np.random.default_rng(0)
    ref, test = rng.uniform(1e4, 3e4, 500), rng.uniform(1e4, 3e4, 500)
    diff = np.asarray(ref) - np.asarray(test)
    expected = abs(diff.mean()) + 0.1 * (2 * 1.96 * diff.std(ddof=1))
    assert nomogram_models.bland_altman_score(ref, test) == pytest.approx(expected)


def test_node_seed_is_stable_across_processes(nomogram_models):
    """Built-in hash() is salted per process; the lookup-table seeds must not be.

    The same node is seeded in two subprocesses started with different
    PYTHONHASHSEED values. If generate_v2_table_deterministic went back to
    hash(), these would disagree and the "deterministic" tables would differ
    between runs.
    """
    import os
    import subprocess

    code = (
        "import sys; sys.path.insert(0, %r); "
        "import hashlib; "
        "d = hashlib.blake2b(repr((1, 'jia', 400, 70)).encode(), digest_size=4).digest(); "
        "print(int.from_bytes(d, 'big'))"
    ) % os.getcwd()

    outs = []
    for hashseed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())

    assert len(set(outs)) == 1, f"seed is not stable across hash seeds: {outs}"
    assert str(nomogram_models._node_seed(1, "jia", 400, 70)) == outs[0]
    # And distinct nodes must get distinct seeds.
    assert (nomogram_models._node_seed(1, "jia", 400, 70)
            != nomogram_models._node_seed(1, "jia", 400, 85))


def test_summarize_k_distribution_returns_mean_and_percentiles(nomogram_models):
    k = np.linspace(0.006, 0.008, 1000)
    mean, lo, hi = nomogram_models.summarize_k_distribution(k)
    assert mean == pytest.approx(0.007)
    assert lo < mean < hi


def test_run_nomogram_end_to_end(nomogram_models, tmp_path, monkeypatch):
    """Exercise the dashboard's diagnostic path and check the two defects the
    revision fixes: one k used everywhere, and a declared seed."""
    monkeypatch.chdir(tmp_path)

    # The PDF stage runs for real now.
    res = nomogram_models.run_nomogram(
        400, 15, 5000, 70, 10, 3.75, 60, 15, "lanoiselee",
        seed=777, n_sim=300, n_replicates=20)
    (pdf_path, k_best, bias, loa_low, loa_high, fig_ba, k_reported, ci_low, ci_high,
     k_post_fig, scatter_fig, sim_df, summary_text, sens_df, sens_fig,
     tornado_fig, metadata, diagnostics) = res

    # One k: the value used for the statistics is the value reported and printed.
    assert k_best == k_reported
    assert ci_low <= k_best <= ci_high, (
        "the reported k must lie inside its own sampling interval"
    )
    # The printed nomogram is produced, and it is a real PDF.
    pdf = tmp_path / pdf_path
    assert pdf.exists() and pdf.read_bytes().startswith(b"%PDF")
    assert pdf.stat().st_size > 5000

    # The agreement statistics carry the new quantities and the sign convention.
    assert metadata["sign_convention"] == "reference_minus_nomogram"
    assert "prop_bias_slope" in metadata["agreement"]
    assert not metadata["strata"].empty
    assert set(metadata["objectives"]["objective"]) == set(nomogram_models.OBJECTIVES)
    assert bias == pytest.approx(metadata["agreement"]["bias"])
    assert (loa_low, loa_high) == pytest.approx(
        (metadata["agreement"]["loa_low"], metadata["agreement"]["loa_high"]))

    # The exported cohort carries the difference under the stated convention.
    assert "Diff_Ref_minus_Simp" in sim_df
    assert np.allclose(sim_df["Diff_Ref_minus_Simp"],
                       sim_df["Ref_Dose"] - sim_df["Simp_Dose"])

    # R-squared survives only as a descriptor.
    assert "R² (descriptive only)" in diagnostics
    assert "R²" not in diagnostics


def test_run_nomogram_is_reproducible(nomogram_models, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def k_of(seed):
        return nomogram_models.run_nomogram(
            400, 15, 5000, 70, 10, 3.75, 60, 15, "lanoiselee",
            seed=seed, n_sim=200, n_replicates=10)[1]

    assert k_of(5) == k_of(5)
    assert k_of(5) != k_of(6)
