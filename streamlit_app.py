import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import math
import pickle
import itertools

st.title("Streamlit Heparin Decay Nomogram")

# ==========================================
# 1. CORE DATA & LOOKUP FUNCTIONS (DEFINED FIRST)
# ==========================================

# Grid definitions (Must match your table generation)
heparin_grid = [250, 300, 350, 400, 450, 500, 550, 600]
ibw_grid = [40, 55, 70, 85, 100, 115]
time_to_grid = [5, 10, 15, 20, 25, 30, 35, 40]
time_on_grid = [15, 30, 45, 60, 75, 90, 105, 120]
prime_grid = [0, 2000, 4000, 6000, 8000, 10000]

@st.cache_data
def load_k_table():
    try:
        with open("k_table_v2.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

k_table = load_k_table()

def nearest(value, grid):
    return min(grid, key=lambda x: abs(x - value))

def get_k_stats(hpkg, ibw_val, tto, ton, prime_val):
    """Retrieves the {mu, lo, hi} dict from the V2 table."""
    if k_table is None: 
        return {'mu': 0.007, 'lo': 0.006, 'hi': 0.008} # Fallback
    
    hpkg_n = nearest(hpkg, heparin_grid)
    ibw_n = nearest(ibw_val, ibw_grid)
    tto_n = nearest(tto, time_to_grid)
    ton_n = nearest(ton, time_on_grid)
    prime_n = nearest(prime_val, prime_grid)
    
    stats = k_table.get((hpkg_n, ibw_n, tto_n, ton_n, prime_n))
    return stats if stats else {'mu': 0.007, 'lo': 0.006, 'hi': 0.008}

def prodose_remaining(t, heparin_bolus, heparin_prime, ibw_val, time_to_cpb):
    k2 = 0.693 / (26 + 0.323 * (heparin_bolus / ibw_val))
    if t <= time_to_cpb:
        term1 = heparin_bolus * 0.1 * math.exp(-0.0693 * t)
        term2 = heparin_bolus * 0.9 * math.exp(-k2 * t)
        return term1 + term2
    else:
        t_on = t - time_to_cpb
        term1 = heparin_bolus * 0.1 * math.exp(-0.0693 * t)
        term2 = heparin_bolus * 0.9 * math.exp(-k2 * t)
        term3 = heparin_prime * math.exp(-k2 * t_on)
        return term1 + term2 + term3

# ==========================================
# 2. SIDEBAR INPUTS
# ==========================================
st.set_page_config(layout="wide")

st.sidebar.header("1. Institutional Baseline")
ibw_base = st.sidebar.slider("Baseline IBW (kg)", 40, 120, 70)
h_base = st.sidebar.slider("Baseline Bolus (IU/kg)", 250, 600, 300, step=50)
p_base = st.sidebar.slider("Baseline Prime (IU)", 0, 10000, 5000, step=500)
t_to_base = st.sidebar.slider("Baseline Time to CPB (min)", 5, 40, 15)
t_on_base = st.sidebar.slider("Baseline Time on CPB (min)", 15, 120, 60)

st.sidebar.divider()
st.sidebar.header("2. Plot My Patient")
pat_ibw = st.sidebar.number_input("Patient IBW (kg)", value=float(ibw_base), step=1.0)
pat_h_kg = st.sidebar.number_input("Patient Bolus (IU/kg)", value=float(h_base), step=50.0)
pat_p_hep = st.sidebar.number_input("Patient Prime (IU)", value=float(p_base), step=500.0)
pat_t_to = st.sidebar.number_input("Patient Time to CPB (min)", value=float(t_to_base), step=1.0)
pat_t_on = st.sidebar.number_input("Patient Time on CPB (min)", value=float(t_on_base), step=1.0)

show_patient = st.sidebar.checkbox("Plot my patient", value=True)

# ==========================================
# 3. CALCULATIONS & PLOTTING
# ==========================================

# Baseline stats for axes and ribbon
k_stats = get_k_stats(h_base, ibw_base, t_to_base, t_on_base, p_base)
k_mu, k_lo, k_hi = k_stats['mu'], k_stats['lo'], k_stats['hi']

d0_baseline = (h_base * ibw_base) + p_base
res_baseline = d0_baseline * np.exp(-k_mu * (t_to_base + t_on_base))

# Axis Scaling
d0_min, d0_max = d0_baseline * 0.6, d0_baseline * 1.4
r_min, r_max = res_baseline * 0.6, res_baseline * 1.4
m_modulus = 10 / (np.log(d0_max) - np.log(d0_min))

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Heparin Decay Graph")
    t_plot = np.linspace(0, 120, 400)
    
    # 1. Patient starting dose (D0)
    pat_d0 = (pat_h_kg * pat_ibw) + pat_p_hep
    
    # 2. Simplified Model: Uses PATIENT D0 but BASELINE k
    y_simp_mu = pat_d0 * np.exp(-k_mu * t_plot)
    y_simp_hi = pat_d0 * np.exp(-k_lo * t_plot)  # Slower decay
    y_simp_lo = pat_d0 * np.exp(-k_hi * t_plot)  # Faster decay
    
    # 3. PRODOSE Model: Fully dynamic (Uses Patient D0 and PRODOSE math)
    y_prod_pat = [prodose_remaining(ti, pat_h_kg*pat_ibw, pat_p_hep, pat_ibw, pat_t_to) for ti in t_plot]
    
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    
    # Draw Simplified Ribbon (Patient D0 with Institutional Decay)
    ax1.fill_between(t_plot, y_simp_lo, y_simp_hi, color='tab:blue', alpha=0.15, label='Inst. Baseline CrI')
    ax1.plot(t_plot, y_simp_mu, color='tab:blue', lw=1.0, label='Inst. Baseline Median')
    
    # Draw Patient PRODOSE
    ax1.plot(t_plot, y_prod_pat, color='tab:orange', ls='--', lw=1.5, label='Patient (PRODOSE)')
    
    if show_patient:
        # Mark the specific point at the patient's current time
        pat_total_time = pat_t_to + pat_t_on
        pat_res_simp = pat_d0 * np.exp(-k_mu * pat_total_time)
        ax1.scatter(pat_total_time, pat_res_simp, color='black', zorder=5, label='Nomogram Prediction')

    ax1.set_xlim(0, 120)
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Heparin Amount (IU)")
    ax1.legend(fontsize='small')
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

with col2:
    st.subheader("Interactive Nomogram")
    
    def draw_nomo():
        fig2, ax2 = plt.subplots(figsize=(7, 8), dpi=100)
        xl, xm, xr = 0, 1, 2

        # --- Helper: Draw Logarithmic Axes (Left and Right) ---
        def draw_axis(x, v_min, v_max, rev, side):
            ax2.vlines(x, 0, 10, color='black', lw=1.0) # Thinner main axis
            for v in np.arange((int(v_min//1000)+1)*1000, v_max, 1000):
                y = m_modulus * (np.log(v) - np.log(v_min))
                if rev: y = 10 - y
                if 0 <= y <= 10:
                    is_major = (v % 5000 == 0)
                    ax2.hlines(y, x-0.02, x+0.02, lw=0.8 if is_major else 0.4, color='black')
                    if is_major:
                        ax2.text(x+(0.08 if side=='r' else -0.08), y, f"{v:,.0f}", 
                                 ha=('left' if side=='r' else 'right'), 
                                 va='center', fontsize=8)

        draw_axis(xl, d0_min, d0_max, False, 'l') 
        draw_axis(xr, r_min, r_max, True, 'r')    
        
        ax2.text(xl, 10.4, "Initial Dose", ha='center', fontsize=9, fontweight='bold')
        ax2.text(xr, 10.4, "Residual", ha='center', fontsize=9, fontweight='bold')

        # --- Middle Time Axis (xm) Logic ---
        # 1. Calculate the Y extents for 0m and 120m to clip the spine
        y_l_ref = m_modulus * (np.log(d0_baseline) - np.log(d0_min))
        
        y_r_0 = 10 - (m_modulus * (np.log(d0_baseline * np.exp(-k_mu * 0)) - np.log(r_min)))
        y_r_120 = 10 - (m_modulus * (np.log(d0_baseline * np.exp(-k_mu * 120)) - np.log(r_min)))
        
        y_m_top = (y_l_ref + y_r_0) / 2
        y_m_bottom = (y_l_ref + y_r_120) / 2
        
        # Draw the spine only between calculated ticks
        ax2.vlines(xm, y_m_bottom, y_m_top, color='black', lw=0.8)
        
        # 2. Draw the ticks
        for t_mark in range(0, 121, 5):
            y_r_ref = 10 - (m_modulus * (np.log(d0_baseline * np.exp(-k_mu * t_mark)) - np.log(r_min)))
            y_m = (y_l_ref + y_r_ref) / 2
            
            if 0 <= y_m <= 10:
                is_major = (t_mark % 15 == 0)
                tick_size = 0.04 if is_major else 0.02
                ax2.hlines(y_m, xm - tick_size, xm + tick_size, color='black', lw=0.7 if is_major else 0.4)
                
                if is_major:
                    ax2.text(xm + 0.07, y_m, f"{t_mark}m", fontsize=7, va='center')

        # --- Patient Simulation ---
        if show_patient:
            pat_d0 = (pat_h_kg * pat_ibw) + pat_p_hep
            pat_time = pat_t_to + pat_t_on
            y_l_pat = m_modulus * (np.log(pat_d0) - np.log(d0_min))
            
            res_mu = pat_d0 * np.exp(-k_mu * pat_time)
            res_hi_se = pat_d0 * np.exp(-k_lo * pat_time) 
            res_lo_se = pat_d0 * np.exp(-k_hi * pat_time) 
            
            y_r_mu = 10 - (m_modulus * (np.log(res_mu) - np.log(r_min)))
            y_r_hi = 10 - (m_modulus * (np.log(res_hi_se) - np.log(r_min)))
            y_r_lo = 10 - (m_modulus * (np.log(res_lo_se) - np.log(r_min)))
            
            ax2.fill([xl, xr, xr], [y_l_pat, y_r_lo, y_r_hi], color='red', alpha=0.1)
            ax2.plot([xl, xr], [y_l_pat, y_r_mu], color='red', lw=1.2, zorder=5) # Thinner patient line
            
            #pat_prod = prodose_remaining(pat_time, pat_h_kg*pat_ibw, pat_p_hep, pat_ibw, pat_t_to)
            #y_r_prod = 10 - (m_modulus * (np.log(pat_prod) - np.log(r_min)))
            #ax2.scatter(xr, y_r_prod, color='tab:orange', marker='D', s=30, edgecolors='black', lw=0.5, zorder=6)

        ax2.set_xlim(-0.4, 2.4)
        ax2.set_ylim(-0.2, 11)
        ax2.axis('off')
        return fig2

    st.pyplot(draw_nomo())

# --- Calculation for Footer Statistics ---
pat_d0 = (pat_h_kg * pat_ibw) + pat_p_hep
pat_total_time = pat_t_to + pat_t_on

# 1. Simplified Model Statistics
res_mu = pat_d0 * np.exp(-k_mu * pat_total_time)
res_hi_bound = pat_d0 * np.exp(-k_lo * pat_total_time)  # Slower decay = Higher residual
res_lo_bound = pat_d0 * np.exp(-k_hi * pat_total_time)  # Faster decay = Lower residual

# 2. PRODOSE Reference
res_prod_final = prodose_remaining(pat_total_time, pat_h_kg*pat_ibw, pat_p_hep, pat_ibw, pat_t_to)

# 3. Model Comparison (Bias)
diff = res_mu - res_prod_final
perc_diff = (diff / res_prod_final) * 100

# --- Render Footer ---
st.divider()

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Baseline Kinetics ($k$)", f"{k_mu:.5f}")
    st.caption(f"95% CrI: [{k_lo:.5f} – {k_hi:.5f}]")

with col_b:
    st.metric("Simplified Residual", f"{res_mu:,.0f} IU")
    st.caption(f"95% CrI: {res_lo_bound:,.0f} – {res_hi_bound:,.0f} IU")

with col_c:
    st.metric("Patient PRODOSE", f"{res_prod_final:,.0f} IU")
    # Color coding the difference based on significance
    delta_color = "inverse" if abs(perc_diff) > 10 else "normal"
    st.metric("Model Delta", f"{diff:+.0f} IU", f"{perc_diff:.1f}% bias", delta_color=delta_color)

st.write(f"**Clinical Note:** The Simplified model is currently {'overestimating' if diff > 0 else 'underestimating'} the PRODOSE reference by **{abs(diff):,.0f} IU**.")
