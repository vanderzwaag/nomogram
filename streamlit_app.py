import numpy as np
from scipy.integrate import odeint
import streamlit as st
import matplotlib.pyplot as plt
import math
import pickle
import itertools
from Nomogram_Models import run_nomogram, prodose_dose, lanoiselee_dose, delavenne_dose, delavenne_response, jia_response, generate_v2_table_deterministic

# ==========================================
# 1. SETUP & UTILS
# ==========================================
st.set_page_config(layout="wide", page_title="Heparin Decay Dashboard")
APP_VERSION = "v1.2.0"

# Define your "hardcoded" defaults
DEFAULTS = {
    "ibw_base": 70,
    "h_base": 400,
    "p_base": 5000,
    "t_to_base": 15,
    "t_on_base": 60,
    "model_choice": "Lanoiselee"
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Use these exact lists in both files
heparin_grid = [250, 300, 350, 400, 450, 500, 550, 600]
ibw_grid      = [40, 55, 70, 85, 100, 115]
time_to_grid  = [5, 15, 25, 35] 
time_on_grid  = [30, 60, 90, 120]
prime_grid    = [0, 5000, 10000]

@st.cache_data
def load_k_table(model_name):
    filename = f"k_table_v2_{model_name.lower()}.pkl"
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def nearest(value, grid):
    return min(grid, key=lambda x: abs(x - value))

def get_k_stats(hpkg, ibw_val, tto, ton, prime_val, table):
    if table is None:
        return {'mu': 0.007, 'lo': 0.006, 'hi': 0.008}
    
    # 1. Force values to the nearest grid point used during generation
    hpkg_n  = nearest(hpkg, heparin_grid)
    ibw_n   = nearest(ibw_val, ibw_grid)
    tto_n   = nearest(tto, time_to_grid)
    ton_n   = nearest(ton, time_on_grid)
    prime_n = nearest(prime_val, prime_grid)
    
    lookup_key = (hpkg_n, ibw_n, tto_n, ton_n, prime_n)
    stats = table.get(lookup_key)
    
    if stats:
        return stats
    else:
        # If we reach here, the table exists but this specific combo isn't in it
        st.error(f"Entry missing for key: {lookup_key}")
        return {'mu': 0.007, 'lo': 0.006, 'hi': 0.008}

# Wrapper to get remaining amount from the REFERENCE model for plotting
def get_reference_remaining(model_name, t, bolus, prime, ibw, t_to, t_on):
    # This function returns remaining amount at time t (where t is total time from bolus)
    # The helper functions in Nomogram_130 return FINAL dose remaining after t_to + t_on.
    # We need a trajectory function.
    
    if model_name == "PRODOSE":
        # Prodose trajectory logic
        k2 = 0.693 / (26 + 0.323 * (bolus / ibw))
        if t <= t_to:
            term1 = bolus * 0.1 * math.exp(-0.0693 * t)
            term2 = bolus * 0.9 * math.exp(-k2 * t)
            return term1 + term2
        else:
            t_on_curr = t - t_to
            term1 = bolus * 0.1 * math.exp(-0.0693 * t)
            term2 = bolus * 0.9 * math.exp(-k2 * t)
            # Prime added at t_to, decays for t_on_curr
            term3 = prime * math.exp(-k2 * t_on_curr)
            return term1 + term2 + term3
    
    elif model_name == "Meesters":
        k2 = 0.693 / 250
        if t <= t_to:
            term1 = bolus * 0.1 * math.exp(-0.0693 * t)
            term2 = bolus * 0.9 * math.exp(-k2 * t)
            return term1 + term2
        else:
            t_on_curr = t - t_to
            term1 = bolus * 0.1 * math.exp(-0.0693 * t)
            term2 = bolus * 0.9 * math.exp(-k2 * t)
            # Prime added at t_to, decays for t_on_curr
            term3 = prime * math.exp(-k2 * t_on_curr)
            return term1 + term2 + term3

    elif model_name == "Lanoiselee":
        # Lanoiselee trajectory logic
        # Nomogram_130 has lanoiselee_response(dose, t)
        from Nomogram_Models import lanoiselee_response
        
        val_bolus = lanoiselee_response(bolus, t)
        val_prime = 0.0
        if t > t_to:
            val_prime = lanoiselee_response(prime, t - t_to)
            
        return val_bolus + val_prime
      
    elif model_name == "Delavenne":
        # Uses delavenne_response which handles weight covariates
        val_bolus = delavenne_response(bolus, t, ibw)
        val_prime = 0.0
        if t > t_to:
            val_prime = delavenne_response(prime, t - t_to, ibw)
        return val_bolus + val_prime
    
    elif model_name == "Jia":
        val_bolus = jia_response(bolus, t, ibw)
        val_prime = 0.0
        if t > t_to:
            val_prime = delavenne_response(prime, t - t_to, ibw)
        return val_bolus + val_prime
      
    return 0.0

def reset_to_defaults():
    for key, val in DEFAULTS.items():
        st.session_state[key] = val

def lanoiselee_model(y, t, Cl, Vc, Vp, Q):
    AcH, ApH = y
    dAcH = -(Cl / Vc) * AcH + Q * (ApH / Vp - AcH / Vc)
    dApH = Q * (AcH / Vc - ApH / Vp)
    return [dAcH, dApH]
  
def get_lanoiselee_cri(initial_bolus, additional_boluses, n_pat=250):
    """
    additional_boluses: list of tuples [(time, dose), ...]
    """
    # Population Means (from your R script)
    pop_params = {
        'Cl': 1500.18 / 60,
        'Vc': 4011.12,
        'Vp': 1458.59,
        'Q': 287.57 / 60
    }
    # Variability (Omegas)
    omega = np.array([0.0983, 0.111, 0.395, 0.21])
    
    times = np.linspace(0, 120, 121)
    all_sims = np.zeros((n_pat, len(times)))

    # Sort additional boluses by time and filter ones within our time range
    boluses = sorted([b for b in additional_boluses if 0 < b[0] < 120], key=lambda x: x[0])
    # Create a list of 'event' times: [0, bolus_t1, bolus_t2, ..., 120]
    event_times = [0] + [b[0] for b in boluses] + [120]

    for i in range(n_pat):
        # Apply Log-Normal variability
        indiv_p = [val * np.exp(np.random.normal(0, omega[idx])) 
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

def get_delavenne_cri(initial_bolus, additional_boluses, patient_weight, n_pat=250):
    """
    Simulates Delavenne model with Covariates (Weight) and Inter-patient Variability.
    
    Variability from table:
    Vc: 0.119 (approx 11.9% variability on log scale)
    Cl: 0.221 (approx 22.1% variability on log scale)
    Vp, Q: Not listed (assumed fixed for this simulation)
    """
    
    # 1. Calculate Population Means based on Covariates (Weight)
    # Convert to standard units (mL, mL/min)
    pop_Vc_mL = (3.1 * (patient_weight / 70.0)**1.0) * 1000.0
    pop_Cl_mLmin = ((0.841 * (patient_weight / 70.0)**0.767) * 1000.0) / 60.0
    pop_Vp_mL = 2.23 * 1000.0
    pop_Q_mLmin = (4.67 * 1000.0) / 60.0

    # 2. Variability (Omega)
    # Order: [Cl, Vc, Vp, Q] to match loop assignment below
    # Vp and Q have 0 noise based on the provided table
    omega = np.array([0.221, 0.119, 0.0, 0.0])
    
    pop_params = [pop_Cl_mLmin, pop_Vc_mL, pop_Vp_mL, pop_Q_mLmin]
    
    times = np.linspace(0, 120, 121)
    all_sims = np.zeros((n_pat, len(times)))

    # Sort boluses
    boluses = sorted([b for b in additional_boluses if 0 < b[0] < 120], key=lambda x: x[0])
    event_times = [0] + [b[0] for b in boluses] + [120]

    for i in range(n_pat):
        # Apply Log-Normal variability
        # If omega is 0, np.random.normal(0, 0) returns 0, so exp(0)=1 (No change)
        indiv_p = [val * np.exp(np.random.normal(0, omega[idx])) 
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

def get_jia_cri(initial_bolus, additional_boluses, patient_weight, n_pat=250):
    """
    Simulates Jia model with Inter-patient Variability (Omega).
    Variability from table: 
    CL: 0.176, Vc: 0.114, Q: 0.0573, Vp: 0.111
    """
    # 1. Population Means (mL and mL/min)
    pop_Vc_mL = 3.04 * 1000.0
    pop_Cl_mLmin = (1.18 * 1000.0) / 60.0
    pop_Vp_mL = 8.01 * 1000.0
    pop_Q_mLmin = (0.171 * 1000.0) / 60.0

    # 2. Variability (Omega) from the "IIV" column in your table
    # Order: [Cl, Vc, Vp, Q]
    omega = np.array([0.073, 0.081, 0.144, 0.318])
    pop_params = [pop_Cl_mLmin, pop_Vc_mL, pop_Vp_mL, pop_Q_mLmin]
    
    times = np.linspace(0, 120, 121)
    all_sims = np.zeros((n_pat, len(times)))

    boluses = sorted([b for b in additional_boluses if 0 < b[0] < 120], key=lambda x: x[0])
    event_times = [0] + [b[0] for b in boluses] + [120]

    for i in range(n_pat):
        # Apply Log-Normal variability to all 4 parameters
        indiv_p = [val * np.exp(np.random.normal(0, np.sqrt(omega[idx]))) 
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

st.sidebar.button("Reset to Defaults", on_click=reset_to_defaults)

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

# 5. Selectbox (Handled similarly)
model_idx = 0 if st.session_state.model_choice == "PRODOSE" else 1
model_choice = st.sidebar.selectbox("Reference Model", ["Delavenne", "Jia", "Lanoiselee", "Meesters", "PRODOSE"], 
                                    index=model_idx, key="k_model")
st.session_state.model_choice = model_choice

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
tab_compare, tab_clinical, tab_topup, tab_nomogram, tab_diagnostics = st.tabs(["🆚 Compare Models", "🚀 Decay Curves", "Top-up Simulation", "📐 Interactive Nomogram", "🔬 Diagnostics & PDF"])

# Load Table for selected model
k_table = load_k_table(model_choice)
k_stats = get_k_stats(h_base, ibw_base, t_to_base, t_on_base, p_base, k_table)
k_mu, k_lo, k_hi = k_stats['mu'], k_stats['lo'], k_stats['hi']

with tab_compare:
    # Models to compare
    comp_models = ["Lanoiselee", "Delavenne", "Jia", "Meesters", "PRODOSE"]
    
    # Prepare data inputs based on sidebar "Plot My Patient" values
    c_ibw = pat_ibw
    c_bolus_kg = pat_h_kg
    c_bolus_total = c_bolus_kg * c_ibw
    c_prime = pat_p_hep
    c_t_to = pat_t_to
    c_t_on = pat_t_on
    c_end_time = c_t_to + c_t_on
    
    # --- NEW: Model Selection Row ---
    st.markdown("##### Select Models to Display")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    
    with m_col1:
        show_lan = st.checkbox("Lanoiselee", value=True)
    with m_col2:
        show_del = st.checkbox("Delavenne", value=True)
    with m_col3:
        show_jia = st.checkbox("Jia", value=True)
    with m_col4:
        show_mee = st.checkbox("Meesters", value=True)
    with m_col5:
        show_pro = st.checkbox("PRODOSE", value=True)

    # 2. Credible Intervals Toggles row (Existing)
    st.markdown("##### 95% Credible Intervals (CrI) Display")
    t_col1, t_col2, t_col3, _ = st.columns([1, 1, 1, 1.5])
    
    # Only show CrI checkbox if the parent model is actually selected (optional UI polish)
    with t_col1:
        show_lan_cri = st.checkbox("Lanoiselee CrI", value=False, disabled=not show_lan)
    with t_col2:
        show_del_cri = st.checkbox("Delavenne CrI", value=False, disabled=not show_del)
    with t_col3:
        show_jia_cri = st.checkbox("Jia CrI", value=False, disabled=not show_jia)
    
    st.divider()

    # Create layout
    col_comp_plot, col_comp_data = st.columns([3, 1])
    
    with col_comp_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        t_plot = np.linspace(0, 120, 121)
    
        # --- Lanoiselee ---
        if show_lan:
            if show_lan_cri:
                t_lan, lo_lan, med_lan, hi_lan = get_lanoiselee_cri(c_bolus_total, [(c_t_to, c_prime)])
                ax.fill_between(t_lan, lo_lan, hi_lan, color='tab:blue', alpha=0.15)
            
            y_lan = [get_reference_remaining("Lanoiselee", t, c_bolus_total, c_prime, c_ibw, c_t_to, c_t_on) for t in t_plot]
            ax.plot(t_plot, y_lan, color='tab:blue', lw=2, label='Lanoiselee')
    
        # --- Delavenne ---
        if show_del:
            if show_del_cri:
                t_del, lo_del, med_del, hi_del = get_delavenne_cri(c_bolus_total, [(c_t_to, c_prime)], c_ibw)
                ax.fill_between(t_del, lo_del, hi_del, color='purple', alpha=0.15)
            
            y_del = [get_reference_remaining("Delavenne", t, c_bolus_total, c_prime, c_ibw, c_t_to, c_t_on) for t in t_plot]
            ax.plot(t_plot, y_del, color='purple', lw=2, label='Delavenne')
    
        # --- Jia ---
        if show_jia:
            if show_jia_cri:
                t_jia, lo_jia, med_jia, hi_jia = get_jia_cri(c_bolus_total, [(c_t_to, c_prime)], c_ibw)
                ax.fill_between(t_jia, lo_jia, hi_jia, color='green', alpha=0.15)
            
            y_jia = [get_reference_remaining("Jia", t, c_bolus_total, c_prime, c_ibw, c_t_to, c_t_on) for t in t_plot]
            ax.plot(t_plot, y_jia, color='green', lw=2, label='Jia')
    
        # --- Meesters ---
        if show_mee:
            y_mee = [get_reference_remaining("Meesters", t, c_bolus_total, c_prime, c_ibw, c_t_to, c_t_on) for t in t_plot]
            ax.plot(t_plot, y_mee, color='tab:red', lw=2, linestyle='--', label='Meesters')
    
        # --- PRODOSE ---
        if show_pro:
            y_pro = [get_reference_remaining("PRODOSE", t, c_bolus_total, c_prime, c_ibw, c_t_to, c_t_on) for t in t_plot]
            ax.plot(t_plot, y_pro, color='tab:orange', lw=2, linestyle='--', label='PRODOSE')
    
        # Plot Visuals
        ax.axvline(x=c_end_time, color='black', linestyle=':', label="End of CPB")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Heparin Amount (IU)")
        ax.legend(loc='upper right', fontsize='small', ncol=2)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
with col_comp_data:
        st.subheader("Remaining Heparin")
        st.caption(f"Calculated at End of CPB ({c_end_time:.0f} min)")
        
        results = []
        
        # Color Map
        color_map = {
            "Lanoiselee": "#1f77b4",
            "Delavenne": "purple",
            "Jia": "green",
            "Meesters": "#d62728",
            "PRODOSE": "#ff7f0e"
        }

        # Visibility Map
        visibility_map = {
            "Lanoiselee": show_lan,
            "Delavenne": show_del,
            "Jia": show_jia,
            "Meesters": show_mee,
            "PRODOSE": show_pro
        }
        
        for model in comp_models:
            if visibility_map[model]:
                rem_dose = get_reference_remaining(model, c_end_time, c_bolus_total, c_prime, c_ibw, c_t_to, c_t_on)
                
                cri_text = None
                try:
                    # We need the time array (t_mc) to find the correct index
                    if model == "Lanoiselee":
                        t_mc, lo, _, hi = get_lanoiselee_cri(c_bolus_total, [(c_t_to, c_prime)])
                    elif model == "Delavenne":
                        t_mc, lo, _, hi = get_delavenne_cri(c_bolus_total, [(c_t_to, c_prime)], c_ibw)
                    elif model == "Jia":
                        t_mc, lo, _, hi = get_jia_cri(c_bolus_total, [(c_t_to, c_prime)], c_ibw)
                    else:
                        t_mc = None
        
                    if t_mc is not None:
                        # --- FIX: Find the index where t_mc is closest to c_end_time ---
                        idx = np.abs(t_mc - c_end_time).argmin()
                        cri_text = f"{lo[idx]:,.0f} to {hi[idx]:,.0f}"
                        
                except Exception:
                    cri_text = "N/A"
        
                results.append((model, rem_dose, color_map[model], cri_text))
        
        # Sort by remaining dose
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Display
        if not results:
            st.info("No models selected.")
        else:
            for model_name, dose, color, cri in results:
                # Build the CrI HTML string if data exists
                cri_html = f'<p style="margin:0; font-size: 0.8em; color: #888;">95% CrI: {cri} IU</p>' if cri else ""
                
                st.markdown(
                    f"""
                    <div style="
                        border-left: 5px solid {color}; 
                        padding-left: 10px; 
                        margin-bottom: 10px; 
                        background-color: rgba(255,255,255,0.05); 
                        border-radius: 0 5px 5px 0;">
                        <p style="margin:0; font-size: 0.9em; color: gray;">{model_name}</p>
                        <p style="margin:0; font-size: 1.2em; font-weight: bold;">{dose:,.0f} IU</p>
                        {cri_html}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
with tab_clinical:
    pat_total_time = pat_t_to + pat_t_on

    if k_table is None:
        st.warning(f"Lookup table for {model_choice} not found. Please click 'Regenerate Lookup Tables' in sidebar.")
    
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
        st.subheader(f"Heparin Decay ({model_choice} Benchmark)")
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
        ax1.plot(t_plot, y_ref_pat, color='tab:orange', ls='--', lw=2.0, label=f'Reference ({model_choice})')
        
        # --- NEW: Initialize CrI storage for Results column ---
        ref_cri = None

        if model_choice == "Lanoiselee":
            reversal_threshold = 0.625 * pat_d0
            ax1.axhline(y=reversal_threshold, color='green', ls=':', lw=1.5, label="0.625:1 ratio")

            init_bolus = pat_ibw * pat_h_kg
            t_mc, lo, med, hi = get_lanoiselee_cri(init_bolus, [(pat_t_to, pat_p_hep)])
            ax1.fill_between(t_mc, lo, hi, color='orange', alpha=0.1, label='Lanoiselee 95% CrI')
            idx = np.abs(t_mc - pat_total_time).argmin()
            ref_cri = f"{lo[idx]:,.0f} to {hi[idx]:,.0f}" # Capture value at end
            
        elif model_choice == "Delavenne":
            init_bolus = pat_ibw * pat_h_kg
            t_mc, lo, med, hi = get_delavenne_cri(init_bolus, [(pat_t_to, pat_p_hep)], pat_ibw)
            ax1.fill_between(t_mc, lo, hi, color='purple', alpha=0.1, label='Delavenne 95% CrI')
            idx = np.abs(t_mc - pat_total_time).argmin()
            ref_cri = f"{lo[idx]:,.0f} to {hi[idx]:,.0f}" # Capture value at end
            
            
        elif model_choice == "Jia":
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
        
        diff = res_simp - res_ref
        pct_err = (diff / res_ref) * 100 if res_ref > 0 else 0
        
        # Display Simplified Metric
        st.metric("Simplified Model Prediction", f"{res_simp:,.0f} IU")
        st.caption(f"95% CrI: {res_simp_lo:,.0f} to {res_simp_hi:,.0f} IU")
        
        # Display Reference Metric
        st.metric(f"{model_choice} Prediction", f"{res_ref:,.0f} IU")
        if ref_cri:
            st.caption(f"95% CrI: {ref_cri} IU")
        
        st.metric("Difference", f"{diff:+.0f} IU", f"{pct_err:.1f}%", delta_color="inverse")
        
        st.markdown(f"""
        **Parameters Used:**
        * Model: {model_choice}
        * Calibrated Decay ($k$): {k_mu:.5f}
        * Patient Total Load: {pat_d0:,.0f} IU
        """)

# Assuming tab_topup is created via st.tabs(...)

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
    k_stats = get_k_stats(h_base, ibw, t_to, t_on, prime, k_table)
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
    
    # --- REFERENCE MODEL (Superposition Logic) ---
    # Phase 1
    ref_phase1 = [get_reference_remaining(model_choice, t, initial_bolus, prime, ibw, t_to, t_on) for t in t_phase1]
    
    # Phase 2
    ref_phase2 = []
    for t_abs in t_phase2:
        # Original dose continuing to decay naturally
        orig_decay = get_reference_remaining(model_choice, t_abs, initial_bolus, prime, ibw, t_to, t_on)
        
        # Top-up dose decay (Given directly on CPB, so t_to=0, prime=0)
        time_since_topup = t_abs - t_topup
        topup_decay = get_reference_remaining(model_choice, time_since_topup, bolus_val, 0, ibw, 0, t_on)
        
        ref_phase2.append(orig_decay + topup_decay)

    # 4. Plotting
    fig_topup, ax = plt.subplots(figsize=(10, 5))
    
    # Simplified Model Curves
    ax.plot(t_phase1, simp_phase1, 'b-', linewidth=2, label=f"Simplified (k={k_mu:.4f})")
    ax.plot(t_phase2, simp_phase2, 'b--', linewidth=2, label="Simplified (Axis Reset)")
    
    # Reference Model Curves
    ax.plot(t_phase1, ref_phase1, 'r-', linewidth=2, alpha=0.7, label=f"Reference: {model_choice}")
    ax.plot(t_phase2, ref_phase2, 'r--', linewidth=2, alpha=0.7, label="Reference (Superposition)")
    
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

    # 5. Concordance Output
    final_simp = simp_phase2[-1]
    final_ref = ref_phase2[-1]
    error_pct = abs(final_simp - final_ref) / final_ref * 100 if final_ref > 0 else 0
    
    st.info(f"**Concordance Check:** 120 minutes post-top-up, the simplified nomogram reset method differs from the **{model_choice}** continuous model by **{error_pct:.1f}%**.")
        
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

        def draw_nomo():
            fig2, ax2 = plt.subplots(figsize=(7, 8), dpi=100)
            xl, xm, xr = 0, 1, 2
                
            def draw_axis(x, v_min, v_max, rev, side):
                ax2.vlines(x, 0, 10, color='black', lw=1.0)
                for v in np.arange((int(v_min//1000)+1)*1000, v_max, 1000):
                    y = m_modulus * (np.log(v) - np.log(v_min))
                    if rev: y = 10 - y
                    if 0 <= y <= 10:
                        is_maj = (v % 5000 == 0)
                        ax2.hlines(y, x-0.02, x+0.02, lw=0.8 if is_maj else 0.4)
                        if is_maj: ax2.text(x+(0.08 if side=='r' else -0.08), y, f"{v:,.0f}", ha=('left' if side=='r' else 'right'), va='center', fontsize=8)
    
            draw_axis(xl, d0_min, d0_max, False, 'l') 
            draw_axis(xr, r_min, r_max, True, 'r')
    
            # --- Middle Time Axis ---
            y_l_ref = m_modulus * (np.log(d0_baseline) - np.log(d0_min))
            y_r_0 = 10 - (m_modulus * (np.log(d0_baseline * np.exp(-k_mu * 0)) - np.log(r_min)))
            y_r_120 = 10 - (m_modulus * (np.log(d0_baseline * np.exp(-k_mu * 120)) - np.log(r_min)))
            ax2.vlines(xm, (y_l_ref + y_r_120)/2, (y_l_ref + y_r_0)/2, color='black', lw=0.8)
              
            for t_mark in range(0, 121, 5):
                y_r_ref = 10 - (m_modulus * (np.log(d0_baseline * np.exp(-k_mu * t_mark)) - np.log(r_min)))
                y_m = (y_l_ref + y_r_ref) / 2
                if 0 <= y_m <= 10:
                    is_maj = (t_mark % 15 == 0)
                    ax2.hlines(y_m, xm-(0.04 if is_maj else 0.02), xm+(0.04 if is_maj else 0.02), lw=0.7 if is_maj else 0.4)
                    if is_maj: ax2.text(xm + 0.07, y_m, f"{t_mark}m", fontsize=7, va='center')
    
            # Patient straightedge
            y_l_pat = m_modulus * (np.log(pat_d0) - np.log(d0_min))
            pat_time = pat_t_to + pat_t_on
            y_r_mu_p = 10 - (m_modulus * (np.log(pat_d0 * np.exp(-k_mu * pat_time)) - np.log(r_min)))
                
            # --- 1. Simplified Model Uncertainty Fan ---
            y_r_hi_p = 10 - (m_modulus * (np.log(pat_d0 * np.exp(-k_lo * pat_time)) - np.log(r_min)))
            y_r_lo_p = 10 - (m_modulus * (np.log(pat_d0 * np.exp(-k_hi * pat_time)) - np.log(r_min)))
            ax2.fill([xl, xr, xr], [y_l_pat, y_r_lo_p, y_r_hi_p], color='tab:red', alpha=0.1, label="Simplified CrI")
            ax2.plot([xl, xr], [y_l_pat, y_r_mu_p], color='red', lw=1.2, zorder=5)
            
            # --- 2. Reference Model Uncertainty (Shaded Area on Axis) ---
            # Re-calculating the raw numeric values for current patient
            ref_lo, ref_hi = None, None
            init_bolus = pat_ibw * pat_h_kg
            try:
                if model_choice == "Lanoiselee":
                    _, lo, _, hi = get_lanoiselee_cri(init_bolus, [(pat_t_to, pat_p_hep)])
                    idx = np.abs(t_mc - pat_total_time).argmin()
                    ref_lo, ref_hi = lo[idx], hi[idx]
                elif model_choice == "Delavenne":
                    _, lo, _, hi = get_delavenne_cri(init_bolus, [(pat_t_to, pat_p_hep)], pat_ibw)
                    idx = np.abs(t_mc - pat_total_time).argmin()
                    ref_lo, ref_hi = lo[idx], hi[idx]
                elif model_choice == "Jia":
                    _, lo, _, hi = get_jia_cri(init_bolus, [(pat_t_to, pat_p_hep)], pat_ibw)
                    ref_lo, ref_hi = lo[-1], hi[-1]
            except:
                pass

# --- 2. Reference Model Uncertainty (Shaded Area on Axis) ---
            # (Calculation of ref_lo, ref_hi and idx remains as before)
            
            if ref_lo is not None:
                # Calculate raw Y coordinates
                y_raw_lo = 10 - (m_modulus * (np.log(ref_lo) - np.log(r_min)))
                y_raw_hi = 10 - (m_modulus * (np.log(ref_hi) - np.log(r_min)))
                
                # Clip the coordinates for the shaded fill area
                y_fill_lo = np.clip(y_raw_lo, 0, 10)
                y_fill_hi = np.clip(y_raw_hi, 0, 10)
                
                # Draw the shaded bar (clipped to axis limits)
                ax2.fill_betweenx([y_fill_hi, y_fill_lo], xr+0.05, xr+0.18, 
                                 color='orange', alpha=0.3, label=f'{model_choice} CrI')
                
                # Add arrows if the CrI goes beyond the scale
                # Note: On this axis, y=10 is the bottom (lower dose) and y=0 is the top (higher dose)
                arrow_props = dict(arrowstyle='->', color='orange', lw=1.5)
                
                # If HI bound (higher dose) is above the top of axis (y < 0)
                if y_raw_hi < 0:
                    ax2.annotate('', xy=(xr+0.115, 0), xytext=(xr+0.115, 0.5), arrowprops=arrow_props)
                
                # If LO bound (lower dose) is below the bottom of axis (y > 10)
                if y_raw_lo > 10:
                    ax2.annotate('', xy=(xr+0.115, 10), xytext=(xr+0.115, 9.5), arrowprops=arrow_props)

                # Centered label
                y_text = (y_fill_lo + y_fill_hi) / 2
                ax2.text(xr+0.22, y_text, f"Ref CrI\n({model_choice})", 
                         fontsize=7, color='orange', fontweight='bold', va='center')
                         
            ax2.text(xl, 10.4, "Initial Dose", ha='center', fontsize=9, fontweight='bold')
            ax2.text(xr, 10.4, "Residual", ha='center', fontsize=9, fontweight='bold')
            ax2.axis('off')
            return fig2
            
        st.pyplot(draw_nomo())
        
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
    st.info(f"Diagnostics will run using **{model_choice}** as the ground truth.")
    if st.button("Run Full Diagnostics & Generate PDF"):
        with st.spinner(f"Simulating against {model_choice}..."):
            res = run_nomogram(h_base, t_to_base, p_base, ibw_base, ibw_sd, 
                               t_to_sd, t_on_base, t_on_sd, 
                               model_choice)
            
            (pdf_path, k_best, bias, loa_low, loa_high, fig_ba, mean_k, ci_low, ci_high, 
             k_post_fig, scatter_fig, sim_df, summary_text, sens_df, sens_fig, tornado_fig, metadata, diagnostics) = res
             
             
            sub = st.tabs(["Nomogram", "Agreement", "Diagnostics", "Uncertainty", "Sensitivity", "Data", "Summary"])

            with sub[0]: # Nomogram
                with open(pdf_path, "rb") as f:
                    st.download_button("Download Nomogram PDF", f, file_name=f"nomogram_{model_choice}.pdf")
                
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

            with sub[2]: # Diagnostics - FIXED FORMATTING
                st.subheader("Model Performance Metrics")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("RMSE (IU)", f"{diagnostics['RMSE (IU)']:.2f}")
                    st.metric("MAE (IU)", f"{diagnostics['MAE (IU)']:.2f}")
                    st.metric("R² Score", f"{diagnostics['R²']:.3f}")
                with c2:
                    st.metric("Within ±5%", f"{diagnostics['% within ±5%']:.1f}%")
                    st.metric("Within ±10%", f"{diagnostics['% within ±10%']:.1f}%")

            with sub[3]: # Uncertainty - FIXED TEXT
                st.subheader("Bayesian Inference of k")
                col1, col2 = st.columns(2)

                with col1:
                    st.pyplot(k_post_fig, use_container_width=True)
                
                with col2:
                    st.write(f"**Posterior Mean:** `{mean_k:.5f}`")
                    st.write(f"**95% Credible Interval:** `[{ci_low:.5f}, {ci_high:.5f}]`")

            with sub[4]: # Sensitivity - FIXED PLOTS
                st.subheader("Parameter Sensitivity")
                st.dataframe(sens_df)
                plot_col1, plot_col2 = st.columns(2)
                with plot_col1:
                    st.pyplot(sens_fig)
                with plot_col2:
                    st.pyplot(tornado_fig)

            with sub[5]: # Data
                st.download_button("Download CSV", sim_df.to_csv(index=False).encode('utf-8'), "sim_data.csv")
                st.dataframe(sim_df.head())

            with sub[6]: # Summary
                st.markdown(summary_text)
