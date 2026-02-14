import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import math
import pickle
import itertools
import io
from Nomogram_130 import run_nomogram

# ==========================================
# 1. CORE MATH & DATA LOOKUP
# ==========================================
st.set_page_config(layout="wide", page_title="Heparin Decay Dashboard")
APP_VERSION = "v1.1.0"

# Lookup grids for Tab 1's uncertainty ribbon
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
    if k_table is None: 
        return {'mu': 0.007, 'lo': 0.006, 'hi': 0.008} 
    hpkg_n, ibw_n = nearest(hpkg, heparin_grid), nearest(ibw_val, ibw_grid)
    tto_n, ton_n = nearest(tto, time_to_grid), nearest(ton, time_on_grid)
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
# 2. SHARED SIDEBAR
# ==========================================
st.sidebar.header("1. Institutional Baseline")
ibw_base = st.sidebar.slider("Baseline IBW (kg)", 40, 120, 70, step=5)
h_base = st.sidebar.slider("Baseline Bolus (IU/kg)", 250, 600, 300, step=10)
p_base = st.sidebar.slider("Baseline Prime (IU)", 0, 10000, 5000, step=500)
t_to_base = st.sidebar.slider("Baseline Time to CPB (min)", 5, 40, 15, step=1)
t_on_base = st.sidebar.slider("Baseline Time on CPB (min)", 15, 120, 60, step=5)

with st.sidebar.expander("Add SDs to generate a PDF Nomogram"):
    ibw_sd = st.number_input("IBW SD (kg)", value=10.0)
    t_to_sd = st.number_input("Time to CPB SD (min)", value=0.25 * t_to_base)
    t_on_sd = st.number_input("Time on CPB SD (min)", value=15.0)

st.sidebar.divider()
st.sidebar.header("2. Plot My Patient")
pat_ibw = st.sidebar.number_input("Patient IBW (kg)", value=float(ibw_base), step=0.1)
pat_h_kg = st.sidebar.number_input("Patient Bolus (IU/kg)", value=float(h_base), step=1.0)
pat_p_hep = st.sidebar.number_input("Patient Prime (IU)", value=float(p_base), step=100.0)
pat_t_to = st.sidebar.number_input("Patient Time to CPB (min)", value=float(t_to_base), step=1.0)
pat_t_on = st.sidebar.number_input("Patient Time on CPB (min)", value=float(t_on_base), step=1.0)

# ==========================================
# 3. TABS
# ==========================================
tab_clinical, tab_diagnostics = st.tabs(["🚀 Educational Dashboard", "🔬 Download PDF and Run Diagnostics"])

with tab_clinical:
    k_stats = get_k_stats(h_base, ibw_base, t_to_base, t_on_base, p_base)
    k_mu, k_lo, k_hi = k_stats['mu'], k_stats['lo'], k_stats['hi']
    
    d0_baseline = (h_base * ibw_base) + p_base
    res_baseline = d0_baseline * np.exp(-k_mu * (t_to_base + t_on_base))
    
    # Scaling factors
    d0_min, d0_max = d0_baseline * 0.6, d0_baseline * 1.4
    r_min, r_max = res_baseline * 0.6, res_baseline * 1.4
    m_modulus = 10 / (np.log(d0_max) - np.log(d0_min))

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Heparin Decay Graph")
        t_plot = np.linspace(0, 120, 400)
        pat_d0 = (pat_h_kg * pat_ibw) + pat_p_hep
        y_simp_mu = pat_d0 * np.exp(-k_mu * t_plot)
        y_simp_hi = pat_d0 * np.exp(-k_lo * t_plot)
        y_simp_lo = pat_d0 * np.exp(-k_hi * t_plot)
        y_prod_pat = [prodose_remaining(ti, pat_h_kg*pat_ibw, pat_p_hep, pat_ibw, pat_t_to) for ti in t_plot]
        
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.fill_between(t_plot, y_simp_lo, y_simp_hi, color='tab:blue', alpha=0.15, label='Inst. 95% CrI')
        ax1.plot(t_plot, y_simp_mu, color='tab:blue', lw=1.0, label='Inst. Baseline Median')
        ax1.plot(t_plot, y_prod_pat, color='tab:orange', ls='--', lw=1.5, label='Patient (PRODOSE)')
        ax1.scatter(pat_t_to + pat_t_on, pat_d0 * np.exp(-k_mu * (pat_t_to+pat_t_on)), color='black', zorder=5)
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Heparin (IU)")
        ax1.legend(fontsize='small')
        st.pyplot(fig1)

    with col2:
        st.subheader("Interactive Nomogram")
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

            # --- RESTORED TIME AXIS (MIDDLE) ---
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
            
            # Uncertainty fan on right axis
            y_r_hi_p = 10 - (m_modulus * (np.log(pat_d0 * np.exp(-k_lo * pat_time)) - np.log(r_min)))
            y_r_lo_p = 10 - (m_modulus * (np.log(pat_d0 * np.exp(-k_hi * pat_time)) - np.log(r_min)))
            ax2.fill([xl, xr, xr], [y_l_pat, y_r_lo_p, y_r_hi_p], color='red', alpha=0.1)
            ax2.plot([xl, xr], [y_l_pat, y_r_mu_p], color='red', lw=1.2, zorder=5)
            
            ax2.text(xl, 10.4, "Initial Dose", ha='center', fontsize=9, fontweight='bold')
            ax2.text(xr, 10.4, "Residual", ha='center', fontsize=9, fontweight='bold')
            ax2.axis('off')
            return fig2
        st.pyplot(draw_nomo())
        
    # --- Render Footer ---
    
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
    
    st.divider()
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("Baseline Kinetics ($k$)", f"{k_mu:.5f}")
        st.caption(f"95% CrI: [{k_lo:.5f} – {k_hi:.5f}]")
    
    with col_b:
        st.metric("Nomogram estimation", f"{res_mu:,.0f} IU")
        st.caption(f"95% CrI: {res_lo_bound:,.0f} – {res_hi_bound:,.0f} IU")
    
    with col_c:
        st.metric("PRODOSE estimation", f"{res_prod_final:,.0f} IU")
        # Color coding the difference based on significance
        delta_color = "inverse" if abs(perc_diff) > 10 else "normal"
        st.metric("Model Delta", f"{diff:+.0f} IU", f"{perc_diff:.1f}% bias", delta_color=delta_color)

# --- TAB 2: DIAGNOSTICS & PDF (INTERNAL TABS) ---
with tab_diagnostics:
    if st.button("Generate Nomogram & Run Diagnostics"):
        with st.spinner("Simulating..."):
            res = run_nomogram(h_base, t_to_base, p_base, ibw_base, ibw_sd, t_to_base, t_to_sd, t_on_base, t_on_sd)
            (pdf_path, k_best, bias, loa_low, loa_high, fig_ba, mean_k, ci_low, ci_high, 
             k_post_fig, scatter_fig, sim_df, summary_text, sens_df, sens_fig, tornado_fig, metadata, diagnostics) = res

            sub = st.tabs(["Nomogram", "Agreement", "Diagnostics", "Uncertainty", "Sensitivity", "Data", "Summary"])
            
            with sub[0]: # Nomogram
                with open(pdf_path, "rb") as f:
                    st.download_button("Download Nomogram PDF", f, file_name="heparin_nomogram.pdf")
                
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

st.sidebar.markdown(f"**Version:** {APP_VERSION}")
