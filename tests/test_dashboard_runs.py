"""Execute streamlit_app.py end to end against a stubbed Streamlit.

Nothing previously exercised the dashboard as a whole: the module is a script,
so a NameError in a branch only surfaces when a user clicks that tab. This
stub records every call, runs the script top to bottom, and then asserts on what
it drew -- which catches a removed variable, a renamed helper, or a widget that
no longer matches the model registry.
"""

import sys
import types
from contextlib import contextmanager

import pytest

import nomogram_core as core


class _SessionState(dict):
    """Streamlit's session state allows both dict and attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _Recorder:
    """Minimal Streamlit stand-in that records what the script asked for."""

    def __init__(self):
        self.calls = {}
        self.session_state = _SessionState()
        self.stopped = False
        self.errors = []
        self.figures = []

    # -- widgets return their default so the script takes the default path --
    def _record(self, name, *a, **k):
        self.calls.setdefault(name, []).append((a, k))

    def selectbox(self, label, options, index=0, format_func=str, **k):
        self._record("selectbox", label, options)
        self.calls.setdefault("selectbox_labels", []).extend(
            format_func(o) for o in options)
        return options[index]

    def slider(self, label, lo=None, hi=None, value=None, **k):
        self._record("slider", label)
        return value if value is not None else lo

    def number_input(self, label, value=0.0, **k):
        self._record("number_input", label)
        return value

    def checkbox(self, label, value=False, **k):
        self._record("checkbox", label)
        return value

    def button(self, *a, **k):
        self._record("button", *a)
        return False

    def pyplot(self, fig=None, **k):
        self.figures.append(fig)

    def error(self, msg, *a, **k):
        self.errors.append(str(msg))

    def stop(self):
        self.stopped = True
        raise _Stopped()

    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [self for _ in range(n)]

    def tabs(self, names):
        self._record("tabs", names)
        return [self for _ in names]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @contextmanager
    def _ctx(self, *a, **k):
        yield self

    expander = spinner = form = container = _ctx

    def cache_data(self, *dargs, **dkw):
        """Passthrough decorator, in both bare and called forms."""
        if dargs and callable(dargs[0]) and not dkw:
            return dargs[0]
        return lambda fn: fn

    cache_resource = cache_data

    def progress(self, *a, **k):
        return types.SimpleNamespace(progress=lambda *a, **k: None, empty=lambda: None)

    def empty(self):
        return types.SimpleNamespace(text=lambda *a: None)

    @property
    def sidebar(self):
        return self

    def __getattr__(self, name):
        # markdown, caption, write, metric, info, warning, success, subheader,
        # header, title, divider, dataframe, image, download_button, set_page_config
        def _f(*a, **k):
            self._record(name, *a[:1])
            return None
        return _f


class _Stopped(Exception):
    pass


@pytest.fixture(scope="module")
def dashboard():
    st = _Recorder()
    sys.modules["streamlit"] = st

    import matplotlib
    matplotlib.use("Agg")

    # Password gate: the script calls check_password() and stops if it fails.
    # Supply the secret so the body runs.
    st.secrets = {"password": "x"}
    st.session_state["password_correct"] = True

    import importlib
    try:
        importlib.import_module("streamlit_app")
    except _Stopped:
        pass
    return st


def test_dashboard_executes(dashboard):
    """It ran far enough to lay out its tabs and draw something."""
    assert dashboard.calls.get("tabs"), "the script never reached st.tabs"
    assert dashboard.figures, "the script drew no figures"
    assert not dashboard.errors, dashboard.errors


def test_every_model_offered_uses_the_registry_spelling(dashboard):
    """R1 asked for the acute accent. The sidebar previously carried its own
    hardcoded list and spelled it without one."""
    labels = dashboard.calls.get("selectbox_labels", [])
    assert labels, "no model selector was rendered"
    assert set(labels) == set(core.DISPLAY_NAMES.values())
    assert "Lanoiselée" in labels
    assert "Lanoiselee" not in labels


def test_model_checkboxes_cover_all_six_models(dashboard):
    labelled = {a[0] for a, _ in dashboard.calls.get("checkbox", [])}
    for name in core.DISPLAY_NAMES.values():
        assert any(name in lab for lab in labelled), f"no checkbox mentions {name}"


def test_no_stale_model_spelling_anywhere_user_facing():
    """A grep is cruder than the execution test above, but it also covers the
    branches that only run on a button press."""
    src = open("streamlit_app.py", encoding="utf-8").read()
    body = "\n".join(line for line in src.splitlines()
                     if not line.strip().startswith("#"))
    assert '"Lanoiselee"' not in body, "a hardcoded unaccented spelling remains"


def test_dashboard_reads_the_shared_renderer_not_its_own():
    src = open("streamlit_app.py", encoding="utf-8").read()
    assert "from nomogram_render import draw_nomogram" in src
    assert "def draw_nomo(" not in src, (
        "the dashboard still defines its own nomogram; the printed and "
        "on-screen charts could diverge again"
    )


def test_dashboard_surfaces_parameter_uncertainty():
    """EB-2's headline analysis has to be visible in the reviewer-facing tool."""
    src = open("streamlit_app.py", encoding="utf-8").read()
    assert "parameter_uncertainty(" in src
    assert "Parameter uncertainty" in src
