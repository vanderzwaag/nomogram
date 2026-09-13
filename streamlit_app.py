import numpy as np
from scipy.integrate import odeint
import streamlit as st
import matplotlib.pyplot as plt
import math
import pickle
import itertools
import pandas as pd

from calibration import parameter_uncertainty, summarise_k
from Nomogram_Models import (
    DEFAULT_SEED,
    _spec,
    nomogram_geometry_for,
    run_nomogram,
    prodose_dose,
    lanoiselee_dose,
    delavenne_dose,
    delavenne_response,
    jia_response,
    generate_v2_table_deterministic,
)
from nomogram_core import (
    DISPLAY_NAMES,
    MODEL_NAMES,
    MODEL_PARAMETERS,
    canonical_model_name,
    reference_amount,
    reference_amount_with_topups,
)
from nomogram_render import draw_nomogram
from agreement import SIGN_CONVENTION_LABEL, difference
from parameter_spaces import ADULT_GRID, describe_grids

# ==========================================
# LEGAL DISCLAIMER
# ==========================================
st.warning("""
**⚠️ STRICTLY FOR RESEARCH AND EDUCATIONAL USE ONLY**

This application is an experimental informatics pipeline and technical proof-of-concept. It has **not** been cleared, approved, or evaluated by the U.S. Food and Drug Administration (FDA), the European Medicines Agency (EMA), or any other regulatory authority, and we make no claim about how it would be classified: using any output to derive a protamine dose may bring the tool within medical-device oversight, and that determination is jurisdiction-specific and is not one we attempt to make here. 

The predictive models, nomograms, and calculations provided by this software are strictly for educational and research purposes. They must **never** be used for clinical decision-making, patient care, or to dictate drug dosages. The user assumes all liability and risk associated with the use of this software. By continuing to use this application, you acknowledge and agree to these terms.
""")

# ==========================================
# PEER REVIEW LOGIN GATE
# ==========================================
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Use st.secrets to securely check the password
        if st.session_state["password"] == st.secrets["peer_review_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password in session state
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "🔒 Peer Review Access: Please enter the password provided in the manuscript.", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error.
        st.text_input(
            "🔒 Peer Review Access: Please enter the password provided in the manuscript.", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("Password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()

# ==========================================
# 1. SETUP & UTILS
# ==========================================
st.set_page_config(layout="wide", page_title="Heparin Decay Dashboard")
APP_VERSION = "v1.2.0"

# Model identity comes from one registry. The dashboard previously carried six
# separate hardcoded lists, which is how the sidebar ended up spelling
# "Lanoiselee" without the acute accent while the reports and figures spelled it
# correctly (R1).
MODEL_KEYS = list(MODEL_NAMES)
# The three population-PK models, the only ones with published variability.
CRI_MODELS_ALL = ("lanoiselee", "delavenne", "jia")


def display_name(key_or_label):
    return DISPLAY_NAMES[canonical_model_name(key_or_label)]


# Define your "hardcoded" defaults
DEFAULTS = {
    "ibw_base": 70,
    "h_base": 400,
    "p_base": 5000,
    "t_to_base": 15,
    "t_on_base": 60,
    "model_choice": "lanoiselee"
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# The grids are defined once, in parameter_spaces, and imported here. They were
# previously duplicated in both files under a comment reading "use these exact
# lists in both files", which is a drift risk rather than a guarantee; the
# Methods section now quotes parameter_spaces.describe_grids() (EB-8).
heparin_grid = ADULT_GRID["dose_per_kg"]
ibw_grid = ADULT_GRID["ibw"]
time_to_grid = ADULT_GRID["time_to_cpb"]
time_on_grid = ADULT_GRID["time_on_cpb"]
prime_grid = ADULT_GRID["prime"]

@st.cache_data
def load_k_table(model_name):
    """Load the pre-computed k lookup table for one reference model.

    Tables written by the current generator carry a `__metadata__` entry
    recording the seed, the grid, the sample size and the calibration
    conventions used to build them. A table without that entry was produced by
    the unseeded pipeline and cannot be reproduced, so it must not be used for
    anything quoted in the manuscript (EB-6).
    """
    filename = f"k_table_v2_{canonical_model_name(model_name)}.pkl"
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def table_metadata(table):
    return (table or {}).get("__metadata__")


def nearest(value, grid):
    return min(grid, key=lambda x: abs(x - value))


def get_k_stats(hpkg, ibw_val, tto, ton, prime_val, table, model_name=None):
    """Look up k for the nearest grid node.

    Three problems in the submitted version are fixed here.

    1. A missing table returned a hard-coded k of 0.007 with no warning at all.
       No table was ever shipped for PRODOSE-2, so selecting that model silently
       produced every figure at k = 0.007 instead of its calibrated value of
       about 0.0042 -- a 67% error in the decay constant, invisible to the user.
       There is now no silent fallback.

    2. The grid was hard-coded to the adult one, so the paediatric Jia model was
       snapped to adult weight nodes (EB-4). The grid is now read from the
       table's own metadata.

    3. Nodes whose calibration ran into the k search bound were returned without
       comment; they are now flagged, because a value sitting on a bound is an
       artefact of the bound rather than an optimum (R1 p9 L29).
    """
    if table is None:
        st.error(
            f"No lookup table found for {model_name or 'this model'} "
            f"(expected k_table_v2_{canonical_model_name(model_name or 'lanoiselee')}.pkl). "
            "Use 'Regenerate Lookup Tables' in the sidebar. No default decay "
            "constant is substituted: a wrong k would propagate silently into "
            "every figure on this page."
        )
        st.stop()

    meta = table_metadata(table)
    grid = meta["grid"] if meta else {
        "dose_per_kg": heparin_grid, "ibw": ibw_grid,
        "time_to_cpb": time_to_grid, "time_on_cpb": time_on_grid,
        "prime": prime_grid,
    }

    lookup_key = (
        nearest(hpkg, grid["dose_per_kg"]),
        nearest(ibw_val, grid["ibw"]),
        nearest(tto, grid["time_to_cpb"]),
        nearest(ton, grid["time_on_cpb"]),
        nearest(prime_val, grid["prime"]),
    )
    stats = table.get(lookup_key)
    if stats is None:
        st.error(
            f"The lookup table for {model_name or 'this model'} has no entry for "
            f"{lookup_key}. Regenerate the tables before using these values."
        )
        st.stop()

    k_lo_bound, k_hi_bound = (meta["k_search_bounds"] if meta else (0.001, 0.03))
    if (max(stats["mu"], stats["hi"]) >= k_hi_bound - 1e-6
            or min(stats["mu"], stats["lo"]) <= k_lo_bound + 1e-6):
        st.warning(
            f"At this grid node the calibration for {model_name or 'this model'} "
            f"reached the edge of the k search interval "
            f"({k_lo_bound}-{k_hi_bound} /min). The value shown is the boundary, "
            "not an optimum, and should not be quoted."
        )

    return stats


def describe_k_table(table, model_name):
    """Provenance line for the lookup table currently in use (EB-6)."""
    meta = table_metadata(table)
    if not meta:
        return (f":warning: The {model_name} lookup table carries no provenance "
                "record, so it was produced by the unseeded pipeline and cannot "
                "be reproduced. Regenerate it before quoting any value from it.")
    return (
        f"{model_name} table: seed {meta['seed']}, {meta['n_sim']} simulated "
        f"patients per node, {meta['population']} grid, calibration endpoint "
        f"`{meta['evaluation_mode']}`, prime timing `{meta['prime_timing']}`, "
        f"built with {meta['app_version']}. The lo/hi entries are "
        f"{meta['lo_hi_meaning']}."
    )


# Wrapper to get the remaining amount from the REFERENCE model for plotting.
#
# The submitted version reimplemented every model's trajectory here, alongside a
# separate endpoint implementation in Nomogram_Models. Two copies of the same
# equations can drift apart silently, and the manuscript then has no single
# definition to quote. Both now come from nomogram_core, and the test suite
# asserts that the trajectory evaluated at t = t_to + t_on equals the endpoint
# (EB-6).
def get_reference_remaining(model_name, t, bolus, prime, ibw, t_to, t_on=None):
    """Reference-model central-compartment amount (IU) at absolute time t (min).

    Prime heparin enters the circulation at CPB onset, t = t_to, and therefore
    decays only for the duration of bypass; the systemic bolus enters at t = 0.
    That convention is stated once, in nomogram_core, and applies to every
    model (EB-5). `t_on` is accepted for call compatibility and is not used --
    the trajectory depends on the absolute time, not on the planned duration.
    """
    return reference_amount(model_name, t, bolus, prime, ibw, t_to)


def lanoiselee_model(y, t, Cl, Vc, Vp, Q):
    AcH, ApH = y
    dAcH = -(Cl / Vc) * AcH + Q * (ApH / Vp - AcH / Vc)
    dApH = Q * (AcH / Vc - ApH / Vp)
    return [dAcH, dApH]
  
@st.cache_data(show_spinner=False)
def get_lanoiselee_cri(initial_bolus, additional_boluses, n_pat=250, seed=DEFAULT_SEED):
    """Interindividual-variability band for the Lanoiselee model.

    additional_boluses: list of tuples [(time, dose), ...]

    `seed` makes the band reproducible: the submitted version drew from the
    unseeded global np.random, so the band moved on every rerun and no reported
    interval could be reproduced (EB-6).

    The band describes spread BETWEEN simulated patients under the published
    interindividual variability. It is not uncertainty in the published
    parameter estimates themselves -- see parameter_uncertainty() in
    calibration.py for that (EB-2).
    """
    rng = np.random.default_rng(seed)
    # Population Means (from your R script)
    pop_params = {
        'Cl': 1500.18 / 60,
        'Vc': 4011.12,
        'Vp': 1458.59,
        'Q': 287.57 / 60
    }
    # Variability (omega SDs), read from the single parameter record.
    _iiv = MODEL_PARAMETERS["lanoiselee"].iiv
    omega = np.array([_iiv["Cl"], _iiv["Vc"], _iiv["Vp"], _iiv["Q"]])
    
    times = np.linspace(0, 120, 121)
    all_sims = np.zeros((n_pat, len(times)))

    # Sort additional boluses by time and filter ones within our time range
    boluses = sorted([b for b in additional_boluses if 0 < b[0] < 120], key=lambda x: x[0])
    # Create a list of 'event' times: [0, bolus_t1, bolus_t2, ..., 120]
    event_times = [0] + [b[0] for b in boluses] + [120]

    for i in range(n_pat):
        # Apply Log-Normal variability
        indiv_p = [val * np.exp(rng.normal(0, omega[idx]))
                   for idx, val in enumerate(pop_params.values())]
        
        # Initial state: [Central (AcH), Peripheral (ApH)]
        # Add the very first bolus at t=0
        curr_state = [initial_bolus, 0.0] 
        full_traj_x = []
        full_traj_y = []

        # Iterate through time segments between boluses
        for j in range(len(event_times) - 1):
            t_start = event_times[j]
            t_end = event_times[j+1]
            
            # Define time points for this specific segment
            segment_t = times[(times >= t_start) & (times <= t_end)]
            
            if len(segment_t) > 0:
                # Solve for this segment
                sol = odeint(lanoiselee_model, curr_state, segment_t, args=tuple(indiv_p))
                
                # Store results (avoid duplicating the first point of segments)
                if not full_traj_y:
                    full_traj_y.extend(sol[:, 0])
                else:
                    full_traj_y.extend(sol[1:, 0]) # Skip first point as it's the 'start'
                
                # Update current state for the NEXT segment
                # Get the last solved state
                last_state = sol[-1, :]
                
                # If there is another bolus coming up at t_end, add it now
                if j < len(boluses):
                    dose = boluses[j][1]
                    curr_state = [last_state[0] + dose, last_state[1]]
                else:
                    curr_state = last_state

        # Map back to the original 'times' grid (interpolation handles any slight mismatches)
        all_sims[i, :] = np.interp(times, np.linspace(0, 120, len(full_traj_y)), full_traj_y)

    # Calculate Percentiles
    cri_lo = np.percentile(all_sims, 2.5, axis=0)
    cri_hi = np.percentile(all_sims, 97.5, axis=0)
    median = np.percentile(all_sims, 50, axis=0)
    
    return times, cri_lo, median, cri_hi

## CrI ODE Model
def delavenne_ode_model(y, t, Cl, Vc, Vp, Q):
    # Exact same structure as Lanoiselee (2-comp linear)
    # but kept separate in case parameters differ significantly in scale
    AcH, ApH = y
    dAcH = -(Cl / Vc) * AcH + Q * (ApH / Vp - AcH / Vc)
    dApH = Q * (AcH / Vc - ApH / Vp)
    return [dAcH, dApH]

@st.cache_data(show_spinner=False)
def get_delavenne_cri(initial_bolus, additional_boluses, patient_weight,
                      n_pat=250, seed=DEFAULT_SEED):
    """Interindividual-variability band for the Delavenne model.

    Weight covariates on Vc (exponent 1.0) and Cl (exponent 0.767) are applied
    before the random effects. Variability is taken from MODEL_PARAMETERS so
    this function and the calibration cannot drift apart: Cl 0.221, Vc 0.119,
    with Vp and Q held fixed because the source reports no IIV for them.

    `seed` makes the band reproducible (EB-6); it is between-patient spread,
    not uncertainty in the published estimates (EB-2).
    """
    rng = np.random.default_rng(seed)
    
    # 1. Calculate Population Means based on Covariates (Weight)
    # Convert to standard units (mL, mL/min)
    pop_Vc_mL = (3.1 * (patient_weight / 70.0)**1.0) * 1000.0
    pop_Cl_mLmin = ((0.841 * (patient_weight / 70.0)**0.767) * 1000.0) / 60.0
    pop_Vp_mL = 2.23 * 1000.0
    pop_Q_mLmin = (4.67 * 1000.0) / 60.0

    # 2. Variability (omega SDs), order [Cl, Vc, Vp, Q] to match the loop below.
    # Vp and Q carry no reported IIV and are held fixed.
    _iiv = MODEL_PARAMETERS["delavenne"].iiv
    omega = np.array([_iiv["Cl_L_h"], _iiv["Vc_L"], _iiv["Vp_L"], _iiv["Q_L_h"]])
    
    pop_params = [pop_Cl_mLmin, pop_Vc_mL, pop_Vp_mL, pop_Q_mLmin]
    
    times = np.linspace(0, 120, 121)
    all_sims = np.zeros((n_pat, len(times)))

    # Sort boluses
    boluses = sorted([b for b in additional_boluses if 0 < b[0] < 120], key=lambda x: x[0])
    event_times = [0] + [b[0] for b in boluses] + [120]

    for i in range(n_pat):
        # Apply Log-Normal variability
        # If omega is 0, rng.normal(0, 0) returns 0, so exp(0) = 1 (no change)
        indiv_p = [val * np.exp(rng.normal(0, omega[idx]))
                   for idx, val in enumerate(pop_params)]
        
        curr_state = [initial_bolus, 0.0] 
        full_traj_y = []

        # Piecewise solving
        for j in range(len(event_times) - 1):
            t_start = event_times[j]
            t_end = event_times[j+1]
            segment_t = times[(times >= t_start) & (times <= t_end)]
            
            if len(segment_t) > 0:
                sol = odeint(delavenne_ode_model, curr_state, segment_t, args=tuple(indiv_p))
                
                if not full_traj_y:
                    full_traj_y.extend(sol[:, 0])
                else:
                    full_traj_y.extend(sol[1:, 0])

                last_state = sol[-1, :]
                
                # Add bolus if event time matches
                if j < len(boluses):
                    curr_state = [last_state[0] + boluses[j][1], last_state[1]]
                else:
                    curr_state = last_state

        # Interpolate to fixed grid
        all_sims[i, :] = np.interp(times, np.linspace(0, 120, len(full_traj_y)), full_traj_y)

    cri_lo = np.percentile(all_sims, 2.5, axis=0)
    cri_hi = np.percentile(all_sims, 97.5, axis=0)
    median = np.percentile(all_sims, 50, axis=0)
    
    return times, cri_lo, median, cri_hi

def jia_ode_model(y, t, Cl, Vc, Vp, Q):
    AcH, ApH = y
    dAcH = -(Cl / Vc) * AcH + Q * (ApH / Vp - AcH / Vc)
    dApH = Q * (AcH / Vc - ApH / Vp)
    return [dAcH, dApH]

@st.cache_data(show_spinner=False)
def get_jia_cri(initial_bolus, additional_boluses, patient_weight,
                n_pat=250, seed=DEFAULT_SEED):
    """Interindividual-variability band for the paediatric Jia model.

    RESOLVED DISCREPANCY (EB-6): the submitted version's docstring gave the
    variability as CL 0.176, Vc 0.114, Q 0.0573, Vp 0.111 while its code used
    [0.073, 0.081, 0.144, 0.318] -- different numbers in a different order --
    and additionally took their square root, so it treated them as variances
    where the other two models treated theirs as standard deviations. The values
    are now read from MODEL_PARAMETERS (the docstring set, as omega SDs, for
    consistency with the other models) and are flagged UNVERIFIED there until
    checked against the source publication.

    Note also that the Jia implementation carries no weight covariate, so this
    band is identical for a 3 kg neonate and a 70 kg adult (EB-4).
    """
    rng = np.random.default_rng(seed)
    # 1. Population Means (mL and mL/min)
    pop_Vc_mL = 3.04 * 1000.0
    pop_Cl_mLmin = (1.18 * 1000.0) / 60.0
    pop_Vp_mL = 8.01 * 1000.0
    pop_Q_mLmin = (0.171 * 1000.0) / 60.0

    # 2. Variability (omega SDs), order [Cl, Vc, Vp, Q]. See the docstring:
    # the submitted code and its own docstring disagreed on these values.
    _iiv = MODEL_PARAMETERS["jia"].iiv
    omega = np.array([_iiv["Cl_L_h"], _iiv["Vc_L"], _iiv["Vp_L"], _iiv["Q_L_h"]])
    pop_params = [pop_Cl_mLmin, pop_Vc_mL, pop_Vp_mL, pop_Q_mLmin]
    
    times = np.linspace(0, 120, 121)
    all_sims = np.zeros((n_pat, len(times)))

    boluses = sorted([b for b in additional_boluses if 0 < b[0] < 120], key=lambda x: x[0])
    event_times = [0] + [b[0] for b in boluses] + [120]

    for i in range(n_pat):
        # Apply Log-Normal variability to all 4 parameters
        # omega entries are SDs, as for the other two models -- no square root.
        indiv_p = [val * np.exp(rng.normal(0, omega[idx]))
                   for idx, val in enumerate(pop_params)]
        
        curr_state = [initial_bolus, 0.0] 
        full_traj_y = []

        for j in range(len(event_times) - 1):
            t_start, t_end = event_times[j], event_times[j+1]
            segment_t = times[(times >= t_start) & (times <= t_end)]
            
            if len(segment_t) > 0:
                sol = odeint(jia_ode_model, curr_state, segment_t, args=tuple(indiv_p))
                if not full_traj_y:
                    full_traj_y.extend(sol[:, 0])
                else:
                    full_traj_y.extend(sol[1:, 0])

                last_state = sol[-1, :]
                if j < len(boluses):
                    curr_state = [last_state[0] + boluses[j][1], last_state[1]]
                else:
                    curr_state = last_state

        all_sims[i, :] = np.interp(times, np.linspace(0, 120, len(full_traj_y)), full_traj_y)

    return times, np.percentile(all_sims, 2.5, axis=0), \
           np.percentile(all_sims, 50, axis=0), \
           np.percentile(all_sims, 97.5, axis=0)

# ==========================================
# 2. SIDEBAR
# ==========================================
st.sidebar.header("Configuration")

model_idx = MODEL_KEYS.index(canonical_model_name(st.session_state.model_choice))
model_choice = st.sidebar.selectbox(
    "Reference Model", MODEL_KEYS, index=model_idx,
    format_func=display_name, key="k_model")
st.session_state.model_choice = model_choice

st.sidebar.header("1. Institutional Baseline")
ibw_base = st.sidebar.slider("Baseline IBW (kg)", 40, 120, step=5, 
                             value=st.session_state["ibw_base"], key="s_ibw")
st.session_state["ibw_base"] = ibw_base

h_base = st.sidebar.slider("Baseline Bolus (IU/kg)", 250, 600, step=10, 
                           value=st.session_state["h_base"], key="s_h")
st.session_state["h_base"] = h_base

p_base = st.sidebar.slider("Baseline Prime (IU)", 0, 10000, step=500, 
                           value=st.session_state["p_base"], key="s_p")
st.session_state["p_base"] = p_base

t_to_base = st.sidebar.slider("Baseline Time to CPB (min)", 5, 40, step=1, 
                              value=st.session_state["t_to_base"], key="s_tto")
st.session_state["t_to_base"] = t_to_base

t_on_base = st.sidebar.slider("Baseline Time on CPB (min)", 15, 120, step=5, 
                              value=st.session_state["t_on_base"], key="s_ton")
st.session_state["t_on_base"] = t_on_base

with st.sidebar.expander("Simulation Settings"):
    st.caption("Standard Deviations for PDF/Diagnostics")
    ibw_sd = st.number_input("IBW SD (kg)", value=10.0)
    t_to_sd = st.number_input("Time to CPB SD (min)", value=0.25 * t_to_base)
    t_on_sd = st.number_input("Time on CPB SD (min)", value=15.0)
    if st.button("Regenerate Lookup Tables"):
        generate_v2_table_deterministic()

st.sidebar.divider()
st.sidebar.header("2. Plot My Patient")
pat_ibw = st.sidebar.number_input("Patient IBW (kg)", value=float(ibw_base), step=5.0)
pat_h_kg = st.sidebar.number_input("Patient Bolus (IU/kg)", value=float(h_base), step=50.0)
pat_p_hep = st.sidebar.number_input("Patient Prime (IU)", value=float(p_base), step=500.0)
pat_t_to = st.sidebar.number_input("Patient Time to CPB (min)", value=float(t_to_base), step=1.0)
pat_t_on = st.sidebar.number_input("Patient Time on CPB (min)", value=float(t_on_base), step=1.0)

# ==========================================
# 3. TABS
# ==========================================
tab_compare, tab_clinical, tab_topup, tab_nomogram, tab_diagnostics = st.tabs(["Compare Models", "Decay Curves", "Top-up Simulation", "Interactive Nomogram", "Diagnostics & PDF"])

# Load Table for selected model
k_table = load_k_table(model_choice)
k_stats = get_k_stats(h_base, ibw_base, t_to_base, t_on_base, p_base, k_table,
                      model_name=model_choice)
st.sidebar.caption(describe_k_table(k_table, model_choice))
k_mu, k_lo, k_hi = k_stats['mu'], k_stats['lo'], k_stats['hi']

with tab_compare:
    # Models to compare
    # Colour follows the model and is fixed, so a model keeps its hue whichever
    # subset is displayed. Same slots and order as the supplementary figures.
    comp_models = MODEL_KEYS
    SERIES_COLOUR = dict(zip(
        ["lanoiselee", "delavenne", "jia", "meesters", "prodose", "prodose-2"],
        ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
    CRI_MODELS = CRI_MODELS_ALL

    # Prepare data inputs based on sidebar "Plot My Patient" values
    c_ibw = pat_ibw
    c_bolus_kg = pat_h_kg
    c_bolus_total = c_bolus_kg * c_ibw
    c_prime = pat_p_hep
    c_t_to = pat_t_to
    c_t_on = pat_t_on
    c_end_time = c_t_to + c_t_on

    st.markdown("##### Select models to display")
    visibility_map = {}
    for col, model in zip(st.columns(len(comp_models)), comp_models):
        with col:
            visibility_map[model] = st.checkbox(display_name(model), value=True,
                                                key=f"show_{model}")

    st.markdown("##### 95% interindividual-variability bands")
    st.caption("Between-patient spread under the published variability. Not "
               "uncertainty in the published parameter estimates, and not a "
               "predictive interval for an individual (EB-2).")
    cri_map = {}
    for col, model in zip(st.columns(len(CRI_MODELS) + 1), CRI_MODELS):
        with col:
            cri_map[model] = st.checkbox(f"{display_name(model)} band", value=False,
                                         disabled=not visibility_map[model],
                                         key=f"cri_{model}")

    st.divider()

    # Create layout
    col_comp_plot, col_comp_data = st.columns([3, 1])
    
    with col_comp_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        t_plot = np.linspace(0, 120, 121)

        for model in comp_models:
            if not visibility_map[model]:
                continue
            colour = SERIES_COLOUR[model]
            if cri_map.get(model):
                if model == "lanoiselee":
                    t_c, lo_c, _, hi_c = get_lanoiselee_cri(c_bolus_total, [(c_t_to, c_prime)])
                elif model == "delavenne":
                    t_c, lo_c, _, hi_c = get_delavenne_cri(c_bolus_total, [(c_t_to, c_prime)], c_ibw)
                else:
                    t_c, lo_c, _, hi_c = get_jia_cri(c_bolus_total, [(c_t_to, c_prime)], c_ibw)
                ax.fill_between(t_c, lo_c, hi_c, color=colour, alpha=0.15, lw=0)

            y = [get_reference_remaining(model, t, c_bolus_total, c_prime, c_ibw, c_t_to)
                 for t in t_plot]
            ax.plot(t_plot, y, color=colour, lw=2, label=display_name(model))

        ax.axvline(x=c_end_time, color='black', linestyle=':', label="End of CPB")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Heparin Amount (IU)")
        ax.legend(loc='upper right', fontsize='small', ncol=2)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col_comp_data:
        st.subheader("Remaining heparin")
        st.caption(f"At end of CPB ({c_end_time:.0f} min)")

        results = []
        for model in comp_models:
            if not visibility_map[model]:
                continue
            rem = get_reference_remaining(model, c_end_time, c_bolus_total,
                                          c_prime, c_ibw, c_t_to)
            band = None
            if model in CRI_MODELS:
                try:
                    if model == "lanoiselee":
                        t_c, lo, _, hi = get_lanoiselee_cri(c_bolus_total, [(c_t_to, c_prime)])
                    elif model == "delavenne":
                        t_c, lo, _, hi = get_delavenne_cri(c_bolus_total, [(c_t_to, c_prime)], c_ibw)
                    else:
                        t_c, lo, _, hi = get_jia_cri(c_bolus_total, [(c_t_to, c_prime)], c_ibw)
                    i = np.abs(t_c - c_end_time).argmin()
                    band = f"{lo[i]:,.0f} to {hi[i]:,.0f}"
                except Exception:
                    band = None
            results.append((model, rem, SERIES_COLOUR[model], band))

        results.sort(key=lambda r: r[1], reverse=True)

        if not results:
            st.info("No models selected.")
        for model, dose, colour, band in results:
            band_html = (f'<p style="margin:0; font-size:0.78em; color:#888;">'
                         f'95% interindividual range: {band} IU</p>' if band else "")
            st.markdown(
                f"""
                <div style="border-left: 5px solid {colour};
                            padding-left: 10px; margin-bottom: 10px;
                            background-color: rgba(255,255,255,0.05);
                            border-radius: 0 5px 5px 0;">
                    <p style="margin:0; font-size:0.9em; color:gray;">{display_name(model)}</p>
                    <p style="margin:0; font-size:1.2em; font-weight:bold;">{dose:,.0f} IU</p>
                    {band_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        spread = [r[1] for r in results]
        if len(spread) > 1:
            st.caption(
                f"Between-model spread at this timepoint: {min(spread):,.0f} to "
                f"{max(spread):,.0f} IU ({100 * (max(spread) - min(spread)) / max(spread):.0f}% "
                "of the highest). The reference models disagree with each other by "
                "more than the nomogram disagrees with any one of them, which is "
                "why model selection is an institutional decision."
            )

with tab_clinical:
    pat_total_time = pat_t_to + pat_t_on

    if k_table is None:
        st.warning(f"Lookup table for {display_name(model_choice)} not found. Please click 'Regenerate Lookup Tables' in sidebar.")
    
    d0_baseline = (h_base * ibw_base) + p_base
    
    # Calculate limits for chart axis
    # Max likely: Bolus + Prime, Min likely: ~10% of that
    d0_max = d0_baseline * 1.4
    d0_min = d0_baseline * 0.6
    # Residual limits
    r_max = d0_baseline 
    r_min = d0_baseline * 0.3 
    
    m_modulus = 10 / (np.log(d0_max) - np.log(d0_min)) # Basic scaling for drawing

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader(f"Heparin Decay ({display_name(model_choice)} Benchmark)")
        t_plot = np.linspace(0, 120, 200)
        pat_d0 = (pat_h_kg * pat_ibw) + pat_p_hep
        
        # Simplified Model trajectories
        y_simp_mu = pat_d0 * np.exp(-k_mu * t_plot)
        y_simp_hi = pat_d0 * np.exp(-k_lo * t_plot) 
        y_simp_lo = pat_d0 * np.exp(-k_hi * t_plot)
        
        # Reference Model trajectory
        y_ref_pat = [get_reference_remaining(model_choice, ti, pat_h_kg*pat_ibw, pat_p_hep, pat_ibw, pat_t_to, pat_t_on) for ti in t_plot]
        
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.fill_between(t_plot, y_simp_lo, y_simp_hi, color='tab:blue', alpha=0.15, label='95% CrI (Simplified Model)')
        ax1.plot(t_plot, y_simp_mu, color='tab:blue', lw=1.5, label=f'Simplified (k={k_mu:.4f})')
        ax1.plot(t_plot, y_ref_pat, color='tab:orange', ls='--', lw=2.0, label=f'Reference ({display_name(model_choice)})')
        
        # --- NEW: Initialize CrI storage for Results column ---
        ref_cri = None

        if model_choice == "lanoiselee":
            reversal_threshold = 0.625 * pat_d0
            ax1.axhline(y=reversal_threshold, color='green', ls=':', lw=1.5, label="0.625:1 ratio")

            init_bolus = pat_ibw * pat_h_kg
            t_mc, lo, med, hi = get_lanoiselee_cri(init_bolus, [(pat_t_to, pat_p_hep)])
            ax1.fill_between(t_mc, lo, hi, color='orange', alpha=0.1, label='Lanoiselee 95% CrI')
            idx = np.abs(t_mc - pat_total_time).argmin()
            ref_cri = f"{lo[idx]:,.0f} to {hi[idx]:,.0f}" # Capture value at end
            
        elif model_choice == "delavenne":
            init_bolus = pat_ibw * pat_h_kg
            t_mc, lo, med, hi = get_delavenne_cri(init_bolus, [(pat_t_to, pat_p_hep)], pat_ibw)
            ax1.fill_between(t_mc, lo, hi, color='purple', alpha=0.1, label='Delavenne 95% CrI')
            idx = np.abs(t_mc - pat_total_time).argmin()
            ref_cri = f"{lo[idx]:,.0f} to {hi[idx]:,.0f}" # Capture value at end
            
            
        elif model_choice == "jia":
            init_bolus = pat_ibw * pat_h_kg
            t_mc, lo, med, hi = get_jia_cri(init_bolus, [(pat_t_to, pat_p_hep)], pat_ibw)
            ax1.fill_between(t_mc, lo, hi, color='green', alpha=0.1, label='Jia 95% CrI')
            idx = np.abs(t_mc - pat_total_time).argmin()
            ref_cri = f"{lo[idx]:,.0f} to {hi[idx]:,.0f}" # Capture value at end

        # Point at end of CPB
        end_time = pat_t_to + pat_t_on
        ref_end = get_reference_remaining(model_choice, end_time, pat_h_kg*pat_ibw, pat_p_hep, pat_ibw, pat_t_to, pat_t_on)
        simp_end = pat_d0 * np.exp(-k_mu * end_time)
        ax1.scatter(end_time, simp_end, color='red', zorder=5)
        
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Heparin Amount (IU)")
        ax1.legend(fontsize='small')
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    with col2:
        st.subheader("Results")
        pat_total_time = pat_t_to + pat_t_on
        res_simp = pat_d0 * np.exp(-k_mu * pat_total_time)
        # Simplified model also has a CrI based on k_lo and k_hi
        res_simp_lo = pat_d0 * np.exp(-k_hi * pat_total_time)
        res_simp_hi = pat_d0 * np.exp(-k_lo * pat_total_time)
        
        res_ref = ref_end

        # ONE sign convention, defined in agreement.py and used everywhere:
        # difference = reference model - nomogram, so a positive number means
        # the nomogram reads LOW. The submitted dashboard computed
        # (nomogram - reference) here while the Bland-Altman analysis computed
        # (reference - nomogram), which is how the same result appeared as
        # -5.5 IU in the manuscript body and +5.62 IU in the Figure 2 legend
        # (EB-3, R1 Figure 2).
        diff = difference(res_ref, res_simp)
        pct_err = (diff / res_ref) * 100 if res_ref > 0 else 0
        
        # Display Simplified Metric
        st.metric("Simplified Model Prediction", f"{res_simp:,.0f} IU")
        st.caption(f"95% CrI: {res_simp_lo:,.0f} to {res_simp_hi:,.0f} IU")
        
        # Display Reference Metric
        st.metric(f"{display_name(model_choice)} Prediction", f"{res_ref:,.0f} IU")
        if ref_cri:
            st.caption(f"95% CrI: {ref_cri} IU")
        
        st.metric("Difference (reference - nomogram)", f"{diff:+.0f} IU",
                  f"{pct_err:.1f}%", delta_color="inverse")
        st.caption(SIGN_CONVENTION_LABEL)
        
        st.markdown(f"""
        **Parameters Used:**
        * Model: {display_name(model_choice)}
        * Calibrated Decay ($k$): {k_mu:.5f}
        * Patient Total Load: {pat_d0:,.0f} IU
        """)

with tab_topup:
    st.subheader("Dynamic Top-up Simulation (Nomogram Axis Reset)")
    st.write("This tab validates resetting the time axis of the static nomogram by comparing it to the continuous superposition of the selected reference model.")

    # UI Controls
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        t_topup = st.slider("Time of Top-up (minutes from initial dose)", min_value=30, max_value=180, value=60, step=5)
    with col_param2:
        bolus_val = st.number_input("Top-up Bolus (IU)", min_value=1000, max_value=10000, value=5000, step=1000)
    
    # 1. Fetch exact parameters from the current session state
    ibw = st.session_state["ibw_base"]
    h_base = st.session_state["h_base"]
    initial_bolus = h_base * ibw
    prime = st.session_state["p_base"]
    t_to = st.session_state["t_to_base"]
    t_on = st.session_state["t_on_base"]
    model_choice = st.session_state.model_choice
    
    # 2. Extract the exact 'k' used for the simplified model in the rest of the app
    k_table = load_k_table(model_choice)
    k_stats = get_k_stats(h_base, ibw, t_to, t_on, prime, k_table,
                          model_name=model_choice)
    k_mu = k_stats['mu']
    
    initial_total = initial_bolus + prime
    
    # 3. Time vectors
    t_phase1 = np.linspace(0, t_topup, 50)
    t_post = np.linspace(0, 120, 50) 
    t_phase2 = t_topup + t_post 
    
    # --- SIMPLIFIED MODEL (Axis Reset Logic) ---
    # Phase 1 decay (before top-up)
    simp_phase1 = initial_total * np.exp(-k_mu * t_phase1)
    
    # Residual exactly at top-up
    residual_at_topup = initial_total * math.exp(-k_mu * t_topup)
    new_total_load = residual_at_topup + bolus_val
    
    # Phase 2 decay (time axis reset to 0 for t_post)
    simp_phase2 = new_total_load * np.exp(-k_mu * t_post) 
    
    # --- REFERENCE MODEL (exact superposition) ---
    #
    # Superposition is exact for the models that are linear in dose. PRODOSE and
    # PRODOSE-2 are not: they make the slow-pool half-life a function of IU/kg.
    # The submitted code handled PRODOSE-2 with a hand-written special case but
    # passed the top-up through the ordinary PRODOSE trajectory, which gave a
    # 5,000 IU top-up in a 70 kg patient the elimination half-life of a
    # 71 IU/kg induction dose instead of the patient's own. Both are now handled
    # in one place, by holding the dose-dependent covariate at the index bolus
    # (see nomogram_core.reference_amount_with_topups).
    topups = ((float(t_topup), float(bolus_val)),)

    ref_phase1 = [get_reference_remaining(model_choice, t, initial_bolus, prime, ibw, t_to)
                  for t in t_phase1]
    ref_phase2 = [
        reference_amount_with_topups(model_choice, t_abs, initial_bolus, prime,
                                     ibw, t_to, topups)
        for t_abs in t_phase2
    ]

    # 4. Plotting
    fig_topup, ax = plt.subplots(figsize=(10, 5))
    
    # Simplified Model Curves
    ax.plot(t_phase1, simp_phase1, 'b-', linewidth=2, label=f"Simplified (k={k_mu:.4f})")
    ax.plot(t_phase2, simp_phase2, 'b--', linewidth=2, label="Simplified (Axis Reset)")
    
    # Reference Model Curves
    ax.plot(t_phase1, ref_phase1, 'r-', linewidth=2, alpha=0.7, label=f"Reference: {display_name(model_choice)}")
    ax.plot(t_phase2, ref_phase2, 'r--', linewidth=2, alpha=0.7, label="Reference (Analytical Superposition)")
    
    # Annotations
    ax.axvline(x=t_topup, color='gray', linestyle=':', label=f"Top-up ({bolus_val} IU)")
    ax.annotate('', xy=(t_topup, new_total_load), xytext=(t_topup, residual_at_topup),
                arrowprops=dict(arrowstyle="->", color='blue', lw=1.5))
    
    ax.set_xlabel("Total Elapsed Time (min)")
    ax.set_ylabel("Remaining Heparin Load (IU)")
    ax.set_title(f"Validation of Nomogram Axis-Reset post {bolus_val} IU Top-up")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig_topup)

    # 5. Concordance output -- mean and worst case across the whole window, not
    # just the final point (EB-5). The full size x timing x repeat grid is
    # produced by topup.py and reported in the manuscript.
    ref_arr = np.asarray(ref_phase2, dtype=float)
    simp_arr = np.asarray(simp_phase2, dtype=float)
    diff_arr = difference(ref_arr, simp_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_arr = np.where(ref_arr > 0, 100.0 * diff_arr / ref_arr, np.nan)
    worst_i = int(np.nanargmax(np.abs(pct_arr)))
    final_pct = pct_arr[-1]

    st.info(
        f"**Concordance check** (difference = {display_name(model_choice)} minus nomogram). "
        f"At the end of the plotted window the axis-reset estimate differs from "
        f"the {display_name(model_choice)} model by **{final_pct:+.1f}%** "
        f"({diff_arr[-1]:+,.0f} IU). Across the whole post-top-up window the mean "
        f"absolute divergence is **{np.nanmean(np.abs(pct_arr)):.1f}%** and the "
        f"worst case is **{pct_arr[worst_i]:+.1f}%** "
        f"({diff_arr[worst_i]:+,.0f} IU, at t = {t_phase2[worst_i]:.0f} min)."
    )

with tab_nomogram:
    st.subheader("Interactive Nomogram")
        
    col1, col2 = st.columns([1, 1])
        
    with col1:
        # Basic setup
        d0_baseline = (h_base * ibw_base) + p_base
        res_baseline = d0_baseline * np.exp(-k_mu * (t_to_base + t_on_base))
        
        # Scaling factors
        d0_min, d0_max = d0_baseline * 0.6, d0_baseline * 1.4
        r_min, r_max = res_baseline * 0.6, res_baseline * 1.4
        m_modulus = 10 / (np.log(d0_max) - np.log(d0_min))

        # Drawn by the same module that renders the printed PDF, from the same
        # decay constant, so the screen and the print cannot disagree. The
        # previous version was a second, independent implementation fed by a
        # different k -- the defect EB-3 was about, surviving in the drawing layer.
        geom = nomogram_geometry_for(k_mu, h_base, p_base, ibw_base)

        ref_lo = ref_hi = None
        try:
            if model_choice in CRI_MODELS_ALL:
                init_bolus = pat_ibw * pat_h_kg
                if model_choice == "lanoiselee":
                    t_mc, lo, _, hi = get_lanoiselee_cri(init_bolus, [(pat_t_to, pat_p_hep)])
                elif model_choice == "delavenne":
                    t_mc, lo, _, hi = get_delavenne_cri(init_bolus, [(pat_t_to, pat_p_hep)], pat_ibw)
                else:
                    t_mc, lo, _, hi = get_jia_cri(init_bolus, [(pat_t_to, pat_p_hep)], pat_ibw)
                idx = np.abs(t_mc - (pat_t_to + pat_t_on)).argmin()
                ref_lo, ref_hi = float(lo[idx]), float(hi[idx])
        except Exception:
            ref_lo = ref_hi = None

        fig_nomo, ax_nomo = plt.subplots(figsize=(6.2, 7.6))
        draw_nomogram(
            geom, ax=ax_nomo,
            patient={"total": pat_d0, "elapsed": pat_t_to + pat_t_on},
            residual_band=(ref_lo, ref_hi) if ref_lo is not None else None,
            title=f"{display_name(model_choice)}  ·  k = {k_mu:.5f} /min",
        )
        st.pyplot(fig_nomo)
        if ref_lo is not None:
            st.caption(
                f"The shaded bar spans the {display_name(model_choice)} model's "
                f"95% interindividual-variability range at this patient's reversal "
                f"time ({ref_lo:,.0f} to {ref_hi:,.0f} IU). It describes spread "
                "between patients, not uncertainty in the published parameters, "
                "and it is a feature of this interactive tool only: the printed "
                "nomogram's time axis is spaced for one decay constant, so a "
                "different constant is a different chart rather than a wider line."
            )
        
    with col2:
        st.markdown("""
          ### How to Use the Nomogram ###
                      
            1. Identify the **total heparin administered** (initial bolus + CPB prime).  
            2. Locate this value on the **left axis**.  
            3. Identify the **elapsed time** from heparin administration to protamine dosing (time to CPB + time on CPB).  
            4. Locate this value on the **middle axis**.  
            5. Use a straightedge to connect the two points (in this example, red line is drawn for the data from "Plot My Patient".  
            6. The intersection with the **residual heparin axis** provides the estimated remaining heparin load.
            7. **Diagnostics & PDF** tab allows you to generate a PDF Nomogram and look into agreement diagnostics with the benchmark model.
        """)
    
with tab_diagnostics:
    # 'Ground truth' removed throughout: the published models are reference
    # models used as the approximation target, not ground truth (EB-1, R1 p11 L51).
    st.info(f"Diagnostics will run using **{display_name(model_choice)}** as the reference model "
            "(the approximation target, not ground truth). The comparison cohort is "
            "an internal resample from the same assumed distributions, not an "
            "external validation cohort.")
    diag_seed = st.number_input(
        "Random seed", value=int(DEFAULT_SEED), step=1,
        help="Every simulation in the run is drawn from this seed, so the same "
             "seed always reproduces the same constants, statistics and figures. "
             "The seed is printed in the PDF footer and recorded with the results "
             "(EB-6).")
    if st.button("Run Full Diagnostics & Generate PDF"):
        with st.spinner(f"Simulating against {display_name(model_choice)}..."):
            res = run_nomogram(h_base, t_to_base, p_base, ibw_base, ibw_sd,
                               t_to_sd, t_on_base, t_on_sd,
                               model_choice, seed=diag_seed)
            
            (pdf_path, k_best, bias, loa_low, loa_high, fig_ba, mean_k, ci_low, ci_high, 
             k_post_fig, scatter_fig, sim_df, summary_text, sens_df, sens_fig, tornado_fig, metadata, diagnostics) = res
             
             
            sub = st.tabs(["Nomogram", "Agreement", "Diagnostics", "Uncertainty", "Sensitivity", "Data", "Summary"])

            with sub[0]: # Nomogram
                with open(pdf_path, "rb") as f:
                    st.download_button("Download Nomogram PDF", f, file_name=f"nomogram_{display_name(model_choice)}.pdf")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                      ### How to Use the Nomogram
                      
                      1. Identify the **total heparin administered** (initial bolus + CPB prime).  
                      2. Locate this value on the **left axis**.  
                      3. Identify the **elapsed time** from heparin administration to protamine dosing (time to CPB + time on CPB).  
                      4. Locate this value on the **middle axis**.  
                      5. Use a straightedge to connect the two points.  
                      6. The intersection with the **residual heparin axis** provides the estimated remaining heparin load.
                    """)
                
                with col2:
                    st.image("Figure.png")

            with sub[1]:
                plot_col1, plot_col2 = st.columns(2)
                with plot_col1:
                    st.pyplot(fig_ba)
                with plot_col2:
                    st.pyplot(scatter_fig)

            with sub[2]: # Diagnostics
                agreement = metadata["agreement"]
                thr = metadata["threshold_iu"]

                st.subheader("Approximation performance")
                st.caption(metadata["sign_convention"].replace("_", " ") +
                           " -- positive means the nomogram reads low")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Absolute error (IU)**")
                    st.metric("Bias", f"{agreement['bias']:+.1f}",
                              help=f"95% CI {agreement['bias_ci_low']:+.1f} to "
                                   f"{agreement['bias_ci_high']:+.1f}")
                    st.metric("RMSE", f"{agreement['rmse_iu']:.1f}")
                    st.metric("MAE", f"{agreement['mae_iu']:.1f}")
                    st.metric("Largest", f"{agreement['max_abs_error_iu']:.0f}")
                with c2:
                    st.markdown("**Relative error (%)**")
                    st.metric("Mean % error", f"{agreement['mean_pct_error']:+.2f}%")
                    st.metric("MAPE", f"{agreement['mean_abs_pct_error']:.2f}%")
                    st.metric("Median absolute",
                              f"{agreement['median_abs_pct_error']:.2f}%")
                    st.metric("Largest", f"{agreement['max_abs_pct_error']:.1f}%")
                with c3:
                    st.markdown("**Proportional bias and coverage**")
                    st.metric("BA regression slope",
                              f"{agreement['prop_bias_slope']:+.4f}",
                              help="Regression of the difference on the mean. A "
                                   "non-zero slope means the error varies "
                                   "systematically with the residual load, which "
                                   "a near-zero mean bias can hide entirely.")
                    st.metric("Slope p-value", f"{agreement['prop_bias_slope_p']:.2g}")
                    st.metric(f"Within {thr:,.0f} IU",
                              f"{agreement['pct_within_threshold_iu']:.1f}%",
                              help=f"{thr / 100:.0f} mg of protamine at a "
                                   f"1 mg : 100 IU ratio")
                    st.metric("Within 10%",
                              f"{agreement['pct_within_threshold_pct']:.1f}%")

                if agreement["prop_bias_significant"]:
                    st.warning(
                        f"Proportional bias is statistically detectable "
                        f"(slope {agreement['prop_bias_slope']:+.4f}, "
                        f"p = {agreement['prop_bias_slope_p']:.2g}): the predicted "
                        f"difference runs from "
                        f"{agreement['prop_bias_predicted_diff_at_p5']:+.0f} IU at the "
                        f"low end of the range to "
                        f"{agreement['prop_bias_predicted_diff_at_p95']:+.0f} IU at the "
                        f"high end. The mean bias alone does not describe this."
                    )

                st.caption(
                    f"R-squared = {agreement['descriptive_r_squared']:.3f}, reported as "
                    "a descriptor of the scatter only. R-squared is not a measure of "
                    "agreement and is not used as one."
                )

                st.subheader("Performance stratified by elapsed time and residual load")
                st.dataframe(metadata["strata"], use_container_width=True)

            with sub[3]: # Uncertainty
                st.subheader("Monte Carlo sampling precision of k")
                col1, col2 = st.columns(2)

                with col1:
                    st.pyplot(k_post_fig, use_container_width=True)

                with col2:
                    st.write(f"**Decay constant used throughout:** `{k_best:.5f}` /min "
                             f"(apparent half-life {np.log(2) / k_best:.0f} min)")
                    st.write(f"**2.5th-97.5th percentile across replicate cohorts:** "
                             f"`[{ci_low:.5f}, {ci_high:.5f}]`")
                    st.warning(
                        "This interval is **Monte Carlo sampling precision only**. It "
                        "measures how precisely k is pinned down by a cohort of this "
                        "size drawn from these fixed, investigator-specified "
                        "distributions, with the published pharmacokinetic parameters "
                        "held at their point estimates.\n\n"
                        "It does **not** quantify uncertainty in those published "
                        "parameters, in the institutional input estimates, in model "
                        "selection, in the distributional assumptions, in structural "
                        "misspecification, or in an individual patient's prediction. "
                        "It is not a predicted margin of error and must not be read "
                        "as one."
                    )
                    st.caption(
                        "Previously labelled a bootstrap and a posterior: neither is "
                        "accurate. No dataset is resampled and no prior is specified; "
                        "a fresh synthetic cohort is simulated for each replicate."
                    )

                st.subheader("Uncertainty in the published parameters (EB-2)")
                if model_choice in CRI_MODELS_ALL:
                    with st.spinner("Propagating published parameter uncertainty..."):
                        pu = parameter_uncertainty(
                            _spec(h_base, p_base, ibw_base, ibw_sd, t_to_base,
                                  t_to_sd, t_on_base, t_on_sd),
                            model_choice, seed=diag_seed, n_draws=120, n_sim=400)
                    if pu.available:
                        pk = summarise_k(pu.draws["k"])
                        mc_w, pu_w = ci_high - ci_low, pk["k_p97_5"] - pk["k_p2_5"]
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Sampling precision",
                                  f"{mc_w:.5f}", help=f"{ci_low:.5f} to {ci_high:.5f}")
                        c2.metric("Parameter uncertainty", f"{pu_w:.5f}",
                                  help=f"{pk['k_p2_5']:.5f} to {pk['k_p97_5']:.5f}")
                        c3.metric("Ratio", f"{pu_w / mc_w:.0f}x wider")
                        st.info(
                            f"Propagating the uncertainty **{display_name(model_choice)} "
                            f"actually reports** ({pu.reason}) gives an interval "
                            f"**{pu_w / mc_w:.0f} times wider** than the sampling "
                            f"interval above. Only this second interval speaks to how "
                            f"well the reference model itself is known."
                        )
                        st.dataframe(
                            pd.DataFrame([{"parameter": k_, "uncertainty from": v}
                                          for k_, v in (pu.provenance or {}).items()]),
                            use_container_width=True, hide_index=True)
                else:
                    st.caption(
                        f"{display_name(model_choice)} is published as a closed-form "
                        "expression without an estimation-uncertainty table, so no "
                        "parameter-uncertainty interval can be propagated for it. "
                        "Reported as such rather than substituted."
                    )

            with sub[4]: # Sensitivity
                st.subheader(f"Input sensitivity, ±{metadata['variation']:.0%} "
                             "one factor at a time")
                st.dataframe(sens_df, use_container_width=True)
                plot_col1, plot_col2 = st.columns(2)
                with plot_col1:
                    st.pyplot(sens_fig)
                with plot_col2:
                    st.pyplot(tornado_fig)

                st.subheader("Robustness of k to the objective function")
                st.caption(
                    "The primary objective is |bias| + 0.1 × (limits-of-agreement "
                    "width). The one-tenth weight is a choice, not a derived "
                    "quantity, so k is re-derived here under RMSE, MAPE, MAE and an "
                    "asymmetric clinical loss that penalises under-estimation of "
                    "residual heparin 2:1."
                )
                st.dataframe(
                    metadata["objectives"][["objective_label", "k", "half_life_min",
                                            "pct_change_vs_primary"]],
                    use_container_width=True)
                spread = metadata["objectives"]["pct_change_vs_primary"].abs().max()
                st.metric("Largest movement in k across objectives", f"{spread:.2f}%")

            with sub[5]: # Data
                st.download_button("Download CSV", sim_df.to_csv(index=False).encode('utf-8'), "sim_data.csv")
                st.dataframe(sim_df.head())

            with sub[6]: # Summary
                st.markdown(summary_text)
