"""Implementation verification (EB-6)."""

import pandas as pd
import pytest

import benchmarks as B
import nomogram_core as core


def test_all_internal_checks_pass():
    df = B.internal_verification_report()
    failed = df[~df["pass"].astype(bool)]
    assert failed.empty, f"internal verification failures:\n{failed.to_string()}"


@pytest.mark.parametrize("model", B.TWO_COMPARTMENT_MODELS)
@pytest.mark.parametrize("weight", [10.0, 70.0, 115.0])
def test_analytical_solution_solves_the_ode(model, weight):
    """The dashboard integrates the two-compartment ODEs numerically for its
    credible intervals while the calibration uses the closed form. They must be
    the same model."""
    r = B.verify_against_ode(model, weight_kg=weight)
    assert r["max_rel_diff"] < 1e-6, r


@pytest.mark.parametrize("model", B.TWO_COMPARTMENT_MODELS)
def test_pk_identities_hold(model):
    for row in B.verify_pk_identities(model):
        rel = abs(row["observed"] - row["expected"]) / max(abs(row["expected"]), 1e-12)
        assert rel < 1e-4, row


def test_derived_constants_are_reported_for_every_model():
    """EB-6: the supplement must state what each source's parameters imply, so a
    reader can check them against the publication without participant data."""
    df = B.derived_quantities_report()
    assert len(df) == len(core.MODEL_NAMES)
    assert df["source"].str.len().gt(0).all()

    two_comp = df[df["structure"].str.startswith("two-compartment")]
    for col in ("Vc_L", "Vss_L", "Cl_L_per_h", "terminal_half_life_min",
                "mean_residence_time_min"):
        assert two_comp[col].notna().all() and (two_comp[col] > 0).all()


def test_derived_constants_recover_the_published_inputs():
    """A guard against the table drifting from MODEL_PARAMETERS."""
    d = B.derived_quantities("delavenne", weight_kg=70.0)
    vals = core.MODEL_PARAMETERS["delavenne"].values
    assert d["Vc_L"] == pytest.approx(vals["Vc_L"], rel=1e-9)
    assert d["Vp_L"] == pytest.approx(vals["Vp_L"], rel=1e-9)
    assert d["Cl_L_per_h"] == pytest.approx(vals["Cl_L_h"], rel=1e-9)
    assert d["Vss_L"] == pytest.approx(vals["Vc_L"] + vals["Vp_L"], rel=1e-9)
    # Mean residence time is Vss / Cl by definition.
    assert d["mean_residence_time_min"] == pytest.approx(
        d["Vss_L"] / d["Cl_L_per_h"] * 60.0, rel=1e-9)


def test_delavenne_derived_constants_scale_with_weight():
    small = B.derived_quantities("delavenne", weight_kg=50.0)
    large = B.derived_quantities("delavenne", weight_kg=100.0)
    assert large["Vc_L"] > small["Vc_L"]
    assert large["Cl_L_per_h"] > small["Cl_L_per_h"]


def test_jia_derived_constants_do_not_scale_with_weight():
    """No weight covariate, so the constants are identical at every body size."""
    for field in ("Vc_L", "Vss_L", "Cl_L_per_h", "terminal_half_life_min"):
        assert B.derived_quantities("jia", 45.0)[field] == pytest.approx(
            B.derived_quantities("jia", 115.0)[field], rel=1e-12)


def test_source_checks_are_optional_and_pass_when_supplied():
    """An empty SOURCE_CHECKS list is a fact about the publications, not a gap:
    no participant-level data is held for any source study, so nothing is
    re-fitted and no predictive check is possible or claimed."""
    report = B.source_check_report()
    if report.empty:
        assert list(report.columns)  # the schema exists for when one is added
    else:
        assert (report["status"] == "PASS").all(), report[report["status"] != "PASS"]


def test_source_check_machinery_works_if_a_value_is_added():
    """Exercises the path so that adding a real printed value cannot silently
    fail on a plumbing error."""
    expected = B.derived_quantities("lanoiselee")["terminal_half_life_min"]
    check = B.SourceCheck(
        model="lanoiselee", description="self-consistency probe",
        expected=expected, units="min", kind="terminal_half_life_min",
        citation="synthetic, for the test suite only")
    result = check.run()
    assert result["status"] == "PASS" and result["pct_error"] == pytest.approx(0.0)


def test_verification_statement_does_not_overclaim():
    """The manuscript must not describe this as external or predictive
    validation (EB-1, EB-6)."""
    text = B.VERIFICATION_STATEMENT.lower()
    assert "no participant-level data" in text
    assert "not re-fitted" in text or "not re-fitted" in text
    assert "not that the published model is correct" in text
