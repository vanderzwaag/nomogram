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


def test_source_benchmarks_are_declared_for_every_model_with_a_publication():
    covered = {b.model for b in B.SOURCE_BENCHMARKS}
    assert covered >= {"lanoiselee", "delavenne", "jia"}


def test_pending_source_benchmarks_are_reported_as_pending():
    """The manuscript may not claim per-source reproduction until these carry an
    expected value transcribed from the publication."""
    df = B.source_verification_report()
    pending = df[df["status"].str.startswith("PENDING")]
    assert not pending.empty, (
        "every source benchmark now has an expected value -- update the "
        "manuscript to state that per-source reproduction has been demonstrated"
    )
    assert df["observed"].notna().all()
