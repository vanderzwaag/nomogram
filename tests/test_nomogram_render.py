"""Tests for the nomogram geometry.

The printed nomogram was previously the only output the test suite could not
cover, because rendering it required PyNomo, PyX, LaTeX and Ghostscript. It is
now plain matplotlib, so the chart a clinician would actually read is checked
here like everything else.
"""

import math

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

import nomogram_render as nr

GEOM = nr.build_geometry(0.00703, d0_min=15000.0, d0_max=50000.0, t_max=120.0)


def test_the_straightedge_passes_through_the_time_axis():
    """The defining property of an alignment chart: for any dose and elapsed
    time, the straight line joining them on the outer axes crosses the middle
    axis at exactly that elapsed time.

    This is what makes the chart readable at all, and it holds only because the
    residual axis descends while the dose axis ascends.
    """
    rng = np.random.default_rng(0)
    for _ in range(5000):
        d0 = rng.uniform(GEOM.d0_min, GEOM.d0_max)
        t = rng.uniform(GEOM.t_min, GEOM.t_max)
        r = float(GEOM.residual(d0, t))
        midpoint_of_line = (GEOM.y_dose(d0) + GEOM.y_res(r)) / 2.0
        assert midpoint_of_line == pytest.approx(float(GEOM.y_time(t)), abs=1e-9)


def test_reading_the_chart_recovers_the_decay_relation():
    """Read dose and time off the chart, follow the straightedge, and the
    residual axis must give D0 * exp(-k t)."""
    for d0, t in ((20000, 30), (33000, 75), (48000, 120)):
        y_d, y_t = float(GEOM.y_dose(d0)), float(GEOM.y_time(t))
        y_r = 2.0 * y_t - y_d                       # the line, extrapolated
        # Invert the residual axis mapping.
        r = 10 ** (math.log10(GEOM.r_min) + (GEOM.height - y_r) / GEOM.modulus)
        assert r == pytest.approx(d0 * math.exp(-GEOM.k * t), rel=1e-9)


def test_both_outer_axes_fit_inside_the_chart():
    assert 0.0 <= float(GEOM.y_dose(GEOM.d0_min)) <= GEOM.height
    assert float(GEOM.y_dose(GEOM.d0_max)) <= GEOM.height + 1e-9
    assert float(GEOM.y_res(GEOM.r_max)) >= -1e-9
    assert float(GEOM.y_res(GEOM.r_min)) <= GEOM.height + 1e-9
    for t in (GEOM.t_min, GEOM.t_max):
        assert 0.0 <= float(GEOM.y_time(t)) <= GEOM.height


def test_residual_axis_descends_and_dose_axis_ascends():
    """Reverse either and the time axis stops being a function of time."""
    assert GEOM.y_dose(GEOM.d0_max) > GEOM.y_dose(GEOM.d0_min)
    assert GEOM.y_res(GEOM.r_max) < GEOM.y_res(GEOM.r_min)


def test_time_axis_spacing_is_set_by_k():
    """A different constant is a differently-spaced chart, not a wider line --
    which is why the printed nomogram cannot carry an uncertainty band."""
    slow = nr.build_geometry(0.004, d0_min=15000.0, d0_max=50000.0, t_max=120.0)
    fast = nr.build_geometry(0.011, d0_min=15000.0, d0_max=50000.0, t_max=120.0)
    span = lambda g: float(g.y_time(g.t_max)) - float(g.y_time(g.t_min))
    assert span(fast) != pytest.approx(span(slow), rel=0.05)


def test_geometry_rejects_impossible_ranges():
    for kwargs in ({"d0_min": 50000.0, "d0_max": 15000.0},
                   {"d0_min": 0.0, "d0_max": 50000.0}):
        with pytest.raises(ValueError):
            nr.build_geometry(0.007, t_max=120.0, **kwargs)
    with pytest.raises(ValueError):
        nr.build_geometry(0.0, d0_min=15000.0, d0_max=50000.0)
    with pytest.raises(ValueError):
        nr.build_geometry(0.007, d0_min=15000.0, d0_max=50000.0, t_min=60.0, t_max=30.0)


def test_pdf_renders_without_latex_or_ghostscript(tmp_path):
    """The whole point of replacing PyNomo: this runs in CI."""
    out = tmp_path / "nomogram.pdf"
    nr.render_pdf(GEOM, str(out), title="Test",
                  footer_lines=nr.footer_for("Lanoiselée", GEOM.k,
                                             "spec", seed=1, interval=(0.007, 0.0071)),
                  patient={"total": 33000.0, "elapsed": 75.0})
    assert out.exists() and out.stat().st_size > 5000
    assert out.read_bytes().startswith(b"%PDF")


def test_footer_states_why_the_print_carries_no_band():
    text = " ".join(nr.footer_for("Lanoiselée", 0.00703, "spec", seed=1))
    assert "no uncertainty band" in text
    assert "sampling precision" in text
    assert "institutional ratio" in text


def test_drawing_accepts_a_band_only_for_interactive_use(tmp_path):
    """The band is drawable, but render_pdf never passes one."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    nr.draw_nomogram(GEOM, ax=ax, residual_band=(18000.0, 21000.0),
                     patient={"total": 33000.0, "elapsed": 75.0})
    assert ax.patches or ax.collections
    plt.close(fig)
