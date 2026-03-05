import math
import numpy as np
from pynomo.nomographer import Nomographer
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from io import BytesIO
from reportlab.lib.units import cm
import pickle
import itertools
import pandas as pd

APP_VERSION = "v1.2.0"

# ==========================================
# MODEL DEFINITIONS
# ==========================================

def prodose_dose(heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb):
    """
    PRODOSE formula.
    All times in minutes, doses in IU, ibw in kg.
    """
    term1 = heparin_bolus * 0.1 * math.exp(-0.0693 * (time_to_cpb + time_on_cpb))
    k2 = 0.693 / (26 + 0.323 * (heparin_bolus / ibw))
    term2 = heparin_bolus * 0.9 * math.exp(-k2 * (time_on_cpb + time_to_cpb))
    term3 = heparin_prime * math.exp(-k2 * time_on_cpb)
    return term1 + term2 + term3
  
def meesters_dose(heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb):
    """
    Meesters formula.
    All times in minutes, doses in IU, ibw in kg.
    """
    term1 = heparin_bolus * 0.1 * math.exp(-0.0693 * (time_to_cpb + time_on_cpb))
    k2 = 0.693 / 250
    term2 = heparin_bolus * 0.9 * math.exp(-k2 * (time_on_cpb + time_to_cpb))
    term3 = heparin_prime * math.exp(-k2 * time_on_cpb)
    return term1 + term2 + term3

def get_lanoiselee_params():
    """
    Returns parameters derived from Lanoiselee_Model.R.
    Cl = 25.003 mL/min
    Vc = 4011.12 mL
    Vp = 1458.59 mL
    Q  = 4.7928 mL/min
    """
    Cl = 25.003
    Vc = 4011.12
    Vp = 1458.59
    Q = 4.7928
    
    k10 = Cl / Vc
    k12 = Q / Vc
    k21 = Q / Vp
    
    # Analytical solution constants for 2-compartment model
    sum_k = k10 + k12 + k21
    root = math.sqrt(sum_k**2 - 4 * k10 * k21)
    alpha = (sum_k + root) / 2
    beta = (sum_k - root) / 2
    
    return alpha, beta, k21, Vc

def lanoiselee_response(dose, t):
    """
    Calculates Amount in Central Compartment (Ac) at time t for a single bolus.
    Analytical solution: A(t) = D * ( A*exp(-alpha*t) + B*exp(-beta*t) )
    """
    if t < 0: return 0.0
    alpha, beta, k21, Vc = get_lanoiselee_params()
    
    # Coefficients
    A_coeff = (alpha - k21) / (alpha - beta)
    B_coeff = (k21 - beta) / (alpha - beta)
    
    return dose * (A_coeff * math.exp(-alpha * t) + B_coeff * math.exp(-beta * t))

def lanoiselee_dose(heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb):
    """
    Lanoiselee model response.
    Note: The R script uses fixed population parameters, so IBW is not used 
    for scaling kinetics, only for the initial bolus calculation if provided.
    """
    # Bolus at t=0
    total_time = time_to_cpb + time_on_cpb
    val_bolus = lanoiselee_response(heparin_bolus, total_time)
    
    # Prime at t=time_to_cpb (decays only during time_on_cpb)
    val_prime = lanoiselee_response(heparin_prime, time_on_cpb)
    
    return val_bolus + val_prime

def get_delavenne_params(weight_kg):
    """
    Returns parameters derived from Delavenne Model.
    Units converted to mL and mL/min to match existing system.
    
    Source: Delavenne et al. (Image provided)
    """
    # Base Parameters (Population means for 70kg)
    # Vc = 3.1 L, Vp = 2.23 L, Q = 4.67 L/h, Cl = 0.841 L/h
    
    # Covariate: Weight on Vc (Exponent 1 fixed)
    # Vc_indiv = Vc_pop * (WT/70)^1
    Vc_L = 3.1 * (weight_kg / 70.0)**1.0
    
    # Covariate: Weight on Cl (Exponent 0.767)
    # Cl_indiv = Cl_pop * (WT/70)^0.767
    Cl_L_h = 0.841 * (weight_kg / 70.0)**0.767
    
    # Fixed parameters (No covariate listed)
    Vp_L = 2.23
    Q_L_h = 4.67

    # Convert to mL and mL/min
    Vc = Vc_L * 1000.0
    Vp = Vp_L * 1000.0
    Cl = (Cl_L_h * 1000.0) / 60.0
    Q  = (Q_L_h * 1000.0) / 60.0
    
    # Calculate micro-constants
    k10 = Cl / Vc
    k12 = Q / Vc
    k21 = Q / Vp
    
    # Analytical solution constants (A*exp(-alpha*t) + B*exp(-beta*t))
    sum_k = k10 + k12 + k21
    root = math.sqrt(sum_k**2 - 4 * k10 * k21)
    alpha = (sum_k + root) / 2
    beta = (sum_k - root) / 2
    
    return alpha, beta, k21, Vc

def delavenne_response(dose, t, weight_kg):
    """
    Calculates Amount in Central Compartment (Ac) at time t.
    Requires weight for parameter scaling.
    """
    if t < 0: return 0.0
    alpha, beta, k21, Vc = get_delavenne_params(weight_kg)
    
    # Coefficients
    A_coeff = (alpha - k21) / (alpha - beta)
    B_coeff = (k21 - beta) / (alpha - beta)
    
    return dose * (A_coeff * math.exp(-alpha * t) + B_coeff * math.exp(-beta * t))

def delavenne_dose(heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb):
    """
    Delavenne model response for Total Dose.
    Note: Delavenne uses Actual Body Weight (ABW), but we pass 'ibw' here 
    if that is the variable holding the patient weight in your system.
    If you have a separate Actual Weight variable, pass that instead.
    """
    # Assuming 'ibw' argument holds the patient's weight in kg used for dosing
    weight = ibw 
    
    # Bolus at t=0
    total_time = time_to_cpb + time_on_cpb
    val_bolus = delavenne_response(heparin_bolus, total_time, weight)
    
    # Prime at t=time_to_cpb
    val_prime = delavenne_response(heparin_prime, time_on_cpb, weight)
    
    return val_bolus + val_prime
  
def get_jia_params(weight_kg):
    """
    Returns parameters derived from Jia Model.
    Values from population estimates: Cl=1.18, Vc=3.04, Q=0.171, Vp=8.01.
    """
    # Note: If specific weight exponents are found later, apply here.
    # Currently using fixed population means.
    Vc_L = 3.04
    Cl_L_h = 1.18
    Vp_L = 8.01
    Q_L_h = 0.171

    # Convert to mL and mL/min
    Vc = Vc_L * 1000.0
    Vp = Vp_L * 1000.0
    Cl = (Cl_L_h * 1000.0) / 60.0
    Q  = (Q_L_h * 1000.0) / 60.0
    
    # Calculate micro-constants
    k10 = Cl / Vc
    k12 = Q / Vc
    k21 = Q / Vp
    
    sum_k = k10 + k12 + k21
    root = math.sqrt(sum_k**2 - 4 * k10 * k21)
    alpha = (sum_k + root) / 2
    beta = (sum_k - root) / 2
    
    return alpha, beta, k21, Vc

def jia_response(dose, t, weight_kg):
    """Calculates Amount in Central Compartment (Ac) at time t for Jia."""
    if t < 0: return 0.0
    alpha, beta, k21, Vc = get_jia_params(weight_kg)
    
    # Analytical coefficients
    A_coeff = (alpha - k21) / (alpha - beta)
    B_coeff = (k21 - beta) / (alpha - beta)
    
    return dose * (A_coeff * math.exp(-alpha * t) + B_coeff * math.exp(-beta * t))

def jia_dose(heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb):
    weight = ibw 
    total_time = time_to_cpb + time_on_cpb
    val_bolus = jia_response(heparin_bolus, total_time, weight)
    val_prime = jia_response(heparin_prime, time_on_cpb, weight)
    return val_bolus + val_prime

def get_reference_dose(model_name, h_bolus, h_prime, ibw, t_to, t_on):
    if model_name.lower() == "prodose":
        return prodose_dose(h_bolus, h_prime, ibw, t_to, t_on)
    elif model_name.lower() == "lanoiselee":
        return lanoiselee_dose(h_bolus, h_prime, ibw, t_to, t_on)
    elif model_name.lower() == "meesters":
        return meesters_dose(h_bolus, h_prime, ibw, t_to, t_on)
    elif model_name.lower() == "delavenne":
        return delavenne_dose(h_bolus, h_prime, ibw, t_to, t_on)
    elif model_name.lower() == "jia":
        return jia_dose(h_bolus, h_prime, ibw, t_to, t_on)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def simplified_model_dose(heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb, k):
    """
    Simplified protamine model using a single decay constant k.
    """
    total_heparin = heparin_bolus + heparin_prime
    effective_time = time_to_cpb + time_on_cpb
    return total_heparin * math.exp(-k * effective_time)

# ==========================================
# STATS & OPTIMIZATION
# ==========================================

def bland_altman_stats(ref, test):
    ref = np.asarray(ref)
    test = np.asarray(test)
    diff = ref - test
    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd
    return bias, loa_low, loa_high

def bland_altman_score(ref, test):
    bias, loa_low, loa_high = bland_altman_stats(ref, test)
    width = loa_high - loa_low
    return abs(bias) + 0.1 * width

@st.cache_data(show_spinner=False)
def find_best_k(initial_dose_per_kg, prime_heparin, ibw_mean, ibw_sd, 
                t_to_mean, t_to_sd, t_on_mean, t_on_sd, 
                model_name, n_sim=1000):

    ibw = np.random.normal(loc=ibw_mean, scale=ibw_sd, size=n_sim)
    heparin_bolus = initial_dose_per_kg * ibw
    
    time_to_cpb_samples = np.clip(np.random.normal(loc=t_to_mean, scale=t_to_sd, size=n_sim), 0, None)
    time_on_cpb_samples = np.clip(np.random.normal(loc=t_on_mean, scale=t_on_sd, size=n_sim), 0, None)

    # Reference Doses
    ref_doses = [
        get_reference_dose(model_name, h, prime_heparin, w, t_to, t_on)
        for h, w, t_to, t_on in zip(heparin_bolus, ibw, time_to_cpb_samples, time_on_cpb_samples)
    ]

    def objective(k):
        test_doses = [
            simplified_model_dose(h, prime_heparin, w, t_to, t_on, k)
            for h, w, t_to, t_on in zip(heparin_bolus, ibw, time_to_cpb_samples, time_on_cpb_samples)
        ]
        return bland_altman_score(ref_doses, test_doses)

    result = minimize_scalar(objective, bounds=(0.001, 0.03), method='bounded')
    return result.x

@st.cache_data(show_spinner=False)
def bootstrap_k_distribution(initial_dose_per_kg, prime_heparin, ibw_mean, ibw_sd,
                             t_to_mean, t_to_sd, t_on_mean, t_on_sd,
                             model_name, n_boot=200, n_sim=1000):

    k_values = []
    for _ in range(n_boot):
        # Resample logic is identical to find_best_k, effectively wrapped here
        # Just calling find_best_k repeatedly is cleaner but might be slower due to overhead.
        # We'll inline the simulation loop for speed as in original code.

        ibw = np.random.normal(loc=ibw_mean, scale=ibw_sd, size=n_sim) # Resample population center?
        # Actually standard bootstrap usually resamples the *dataset*, but here we simulate.
        # We generate a new synthetic cohort every time.

        heparin_bolus = initial_dose_per_kg * ibw
        time_to_cpb_samples = np.clip(np.random.normal(loc=t_to_mean, scale=t_to_sd, size=n_sim), 0, None)
        time_on_cpb_samples = np.clip(np.random.normal(loc=t_on_mean, scale=t_on_sd, size=n_sim), 0, None)

        ref_doses = [
            get_reference_dose(model_name, h, prime_heparin, w, t_to, t_on)
            for h, w, t_to, t_on in zip(heparin_bolus, ibw, time_to_cpb_samples, time_on_cpb_samples)
        ]

        def objective(k):
            test_doses = [
                simplified_model_dose(h, prime_heparin, w, t_to, t_on, k)
                for h, w, t_to, t_on in zip(heparin_bolus, ibw, time_to_cpb_samples, time_on_cpb_samples)
            ]
            return bland_altman_score(ref_doses, test_doses)

        result = minimize_scalar(objective, bounds=(0.001, 0.03), method='bounded')
        k_values.append(result.x)

    return np.array(k_values)

# @st.cache_data(show_spinner=False)  
# def bootstrap_k_distribution(initial_dose_per_kg,
#                              prime_heparin,
#                              ibw_mean,
#                              ibw_sd,
#                              t_to_mean,
#                              t_to_sd,
#                              t_on_mean,
#                              t_on_sd,
#                              model_name,
#                              n_boot=200,
#                              n_sim=1000):
# 
#     k_values = []
# 
#     for _ in range(n_boot):
#         # Resample simulation inputs
#         ibw = np.random.normal(loc=80.0, scale=10.0, size=n_sim)
#         heparin_bolus = initial_dose_per_kg * ibw
# 
#         time_to_cpb_samples = np.random.normal(
#             loc=t_to_mean,
#             scale=t_to_sd,
#             size=n_sim
#         )
# 
#         time_on_cpb_samples = np.random.normal(
#             loc=t_on_mean,
#            scale=t_on_sd,
#             size=n_sim
#         )
# 
# 
#         time_to_cpb_samples = np.clip(time_to_cpb_samples, 0, None)
#         time_on_cpb_samples = np.clip(time_on_cpb_samples, 0, None)
# 
#         # Compute PRODOSE reference
#         ref_doses = [
#             prodose_dose(h, prime_heparin, w, t_to, t_on)
#             for h, w, t_to, t_on in zip(
#                 heparin_bolus,
#                 ibw,
#                 time_to_cpb_samples,
#                 time_on_cpb_samples
#             )
#         ]
# 
#         # Objective for optimizer
#         def objective(k):
#             test_doses = [
#                 simplified_model_dose(h, prime_heparin, w, t_to, t_on, k)
#                 for h, w, t_to, t_on in zip(
#                     heparin_bolus,
#                     ibw,
#                     time_to_cpb_samples,
#                     time_on_cpb_samples
#                 )
#             ]
#             return bland_altman_score(ref_doses, test_doses)
# 
#         # Optimize k
#         result = minimize_scalar(objective, bounds=(0.001, 0.02), method='bounded')
#         k_values.append(result.x)
# 
#     return np.array(k_values)

def summarize_k_distribution(k_values):
    mean_k = np.mean(k_values)
    ci_low = np.percentile(k_values, 2.5)
    ci_high = np.percentile(k_values, 97.5)
    return mean_k, ci_low, ci_high

# ==========================================
# NOMOGRAM PDF GENERATION
# ==========================================

def build_nomogram(k_value):
    output_filename = "heparin_dose_decay_nomogram.pdf"
    main_params = {
        'filename': output_filename,
        'paper_height': 27.0,
        'paper_width': 15.0,
        'block_params': [{
            'block_type': 'type_1',
            'f1_params': {
                'u_min': 20, 'u_max': 55,
                'function': lambda dose: math.log(dose),
                'title': r'Initial Heparin Dose (*1000 IU)',
                'tick_levels': 3, 'tick_text_levels': 2,
            },
            'f2_params': {
                'u_min': 0, 'u_max': 120,
                'function': lambda t, k=k_value: -k * t,
                'title': r'Time (minutes)',
                'tick_levels': 2, 'tick_text_levels': 1,
            },
            'f3_params': {
                'u_min': 10, 'u_max': 45,
                'function': lambda remaining: -math.log(remaining),
                'title': r'Remaining Heparin (*1000 IU)',
                'tick_levels': 3, 'tick_text_levels': 2,
            },
        }]
    }
    Nomographer(main_params)
    return output_filename

def center_pdf_on_a4(input_pdf, output_pdf):
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    a4_width, a4_height = map(float, A4)
    for page in reader.pages:
        orig_width = float(page.mediabox.width)
        orig_height = float(page.mediabox.height)
        x_margin = (a4_width - orig_width) / 2.0
        y_margin = (a4_height - orig_height) / 2.0
        page.mediabox.lower_left = (float(page.mediabox.lower_left[0])-x_margin, float(page.mediabox.lower_left[1])-y_margin)
        page.mediabox.upper_right = (float(page.mediabox.upper_right[0])+x_margin, float(page.mediabox.upper_right[1])+y_margin)
        writer.add_page(page)
    with open(output_pdf, "wb") as f:
        writer.write(f)

def add_footer_to_pdf(input_pdf, output_pdf, footer_text):
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    for page in reader.pages:
        packet = BytesIO()
        can = canvas.Canvas(packet, pagesize=A4)
        can.setFont("Helvetica", 8)
        lines = footer_text.split('\n')
        y_position = 50 # Example starting Y coordinate near the bottom
        for line in lines:
            can.drawString(0, y_position, line) # c is your reportlab canvas
            y_position -= 15 # move down for the next line
        can.save()
        packet.seek(0)
        overlay = PdfReader(packet).pages[0]
        page.merge_page(overlay)
        writer.add_page(page)
    with open(output_pdf, "wb") as f:
        writer.write(f)

# ==========================================
# PLOTTING & UTILS
# ==========================================

def bland_altman_plot(ref, test, model_label="Reference"):
    ref = np.asarray(ref)
    test = np.asarray(test)
    diff = ref - test
    mean_vals = (ref + test) / 2
    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mean_vals, diff, alpha=0.4)
    ax.axhline(bias, color='red', linestyle='--', label=f'Bias = {bias:.2f}')
    ax.axhline(loa_low, color='green', linestyle='--', label=f'LOA Low = {loa_low:.2f}')
    ax.axhline(loa_high, color='green', linestyle='--', label=f'LOA High = {loa_high:.2f}')
    ax.set_xlabel(f"Mean of {model_label} and Simplified (IU)")
    ax.set_ylabel(f"Diff ({model_label} - Simplified) (IU)")
    ax.set_title(f"Bland–Altman: {model_label} vs Simplified")
    ax.legend()
    return fig

@st.cache_resource(show_spinner=False)
def plot_k_posterior(k_values):
    fig, ax = plt.subplots(figsize=(6, 4))
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(k_values)
        xs = np.linspace(min(k_values)*0.9, max(k_values)*1.1, 200)
        ax.plot(xs, kde(xs), color='darkblue', lw=2)
    except:
        pass # Fallback if singular
    ax.hist(k_values, bins=20, density=True, alpha=0.3, color='steelblue')
    ax.set_title("Posterior Distribution of k")
    ax.set_xlabel("k")
    return fig

@st.cache_resource(show_spinner=False)
def plot_sensitivity(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    for param in df["Parameter"].unique():
        subset = df[df["Parameter"] == param]
        ax.plot(["low", "high"], subset["k_best"], marker="o", label=param)
    ax.set_title("Sensitivity Analysis")
    ax.set_ylabel("k value")
    ax.legend()
    return fig

@st.cache_resource(show_spinner=False)
def plot_tornado(df):
    params = []
    low_vals = []
    high_vals = []
    spreads = []

    # 1. Extract values
    for param in df["Parameter"].unique():
        subset = df[df["Parameter"] == param]
        k_low = subset[subset["Direction"] == "low"]["k_best"].values[0]
        k_high = subset[subset["Direction"] == "high"]["k_best"].values[0]
        
        params.append(param)
        low_vals.append(k_low)
        high_vals.append(k_high)
        spreads.append(abs(k_high - k_low))

    # 2. Bind into a DataFrame for guaranteed aligned sorting
    plot_df = pd.DataFrame({
        "Parameter": params,
        "k_low": low_vals,
        "k_high": high_vals,
        "spread": spreads
    })

    # 3. Sort ascending (so the largest spread is plotted last, at the top)
    plot_df = plot_df.sort_values(by="spread", ascending=True).reset_index(drop=True)

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(plot_df))
    
    widths = np.abs(plot_df["k_high"] - plot_df["k_low"])
    lefts = np.minimum(plot_df["k_low"], plot_df["k_high"])
    
    ax.barh(y_pos, widths, left=lefts, height=0.6, color="steelblue", alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["Parameter"])
    ax.set_xlabel("k value")
    ax.set_title("Tornado Plot")
    
    return fig

def clinical_summary(k_best, bias, loa_low, loa_high, ibw_mean, model_name):
    return f"""
    **Clinical Interpretation ({model_name} Benchmark)**

    • The calibrated decay constant is **k = {k_best:.5f}**.
    
    • The simplified model shows a **bias of {bias:.1f} IU** relative to {model_name}.
    
    • The 95% limits of agreement are **{loa_low:.1f} to {loa_high:.1f} IU**.
    
    • Population IBW mean: **{ibw_mean:.0f} kg**.
    """

# ==========================================
# 4. SENSITIVITY ANALYSIS & PLOTTING
# ==========================================

@st.cache_data(show_spinner=False)
def sensitivity_analysis(initial_dose_per_kg,
                         prime_heparin,
                         ibw_mean, ibw_sd,
                         t_to_mean, t_to_sd,
                         t_on_mean, t_on_sd,
                         model_name="PRODOSE",
                         variation=0.2):
    """
    Performs One-Factor-At-A-Time (OFAT) sensitivity analysis.
    Varies each input parameter by +/- 'variation' (default 20%) 
    and observes the impact on the calibrated k value.
    """
    
    # Base parameters centered on the simulation inputs
    base_params = {
        "ibw_mean": ibw_mean,
        "ibw_sd": ibw_sd,
        "t_to_mean": t_to_mean,
        "t_to_sd": t_to_sd,
        "t_on_mean": t_on_mean,
        "t_on_sd": t_on_sd
    }

    results = []
    
    # Visual progress bar for the user
    total_steps = len(base_params) * 2 
    progress_text = "Running Sensitivity Analysis..."
    my_bar = st.progress(0, text=progress_text)
    step_count = 0

    for param_name, base_value in base_params.items():
        for direction, factor in [("low", 1 - variation), ("high", 1 + variation)]:
            # Create a copy of params and modify just one
            current_params = base_params.copy()
            current_params[param_name] = base_value * factor

            # Re-run calibration with the modified parameter set
            k_mod = find_best_k(
                initial_dose_per_kg,
                prime_heparin,
                current_params["ibw_mean"],
                current_params["ibw_sd"],
                current_params["t_to_mean"],
                current_params["t_to_sd"],
                current_params["t_on_mean"],
                current_params["t_on_sd"],
                model_name=model_name,
                n_sim=500  # Lower simulation count for speed during sensitivity
            )

            results.append({
                "Parameter": param_name,
                "Direction": direction,
                "Modified value": current_params[param_name],
                "k_best": k_mod
            })
            
            step_count += 1
            my_bar.progress(step_count / total_steps, text=f"Sensitivity: {param_name} ({direction})")
    
    my_bar.empty() # Clear progress bar when done
    return pd.DataFrame(results)

@st.cache_resource(show_spinner=False)
def plot_sensitivity(df):
    """
    Plots a line graph showing how k changes for Low vs High inputs.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for param in df["Parameter"].unique():
        subset = df[df["Parameter"] == param]
        # Plot markers connected by a line
        ax.plot(["Low (-20%)", "High (+20%)"], subset["k_best"], marker="o", label=param)

    ax.set_title("Sensitivity of k to Model Inputs")
    ax.set_ylabel("Calibrated k value")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

@st.cache_resource(show_spinner=False)
# def plot_tornado(df):
#     """
#     Creates a Tornado Diagram to visualize relative impact of parameters.
#     """
#     # Group by parameter to find range (High k - Low k)
#     params = []
#     low_vals = []
#     high_vals = []
# 
#     for param in df["Parameter"].unique():
#         subset = df[df["Parameter"] == param]
#         # We assume the loop order (low then high) or filter explicitly
#         try:
#             k_low = subset[subset["Direction"] == "low"]["k_best"].values[0]
#             k_high = subset[subset["Direction"] == "high"]["k_best"].values[0]
#         except IndexError:
#             continue
# 
#         params.append(param)
#         low_vals.append(k_low)
#         high_vals.append(k_high)
# 
#     low_vals = np.array(low_vals)
#     high_vals = np.array(high_vals)
#     
#     # Calculate center and width for bars
#     # Note: Tornado plots usually center on the 'Base Case k', 
#     # but centering on the average of the range works for simple visual comparison.
#     base = (low_vals + high_vals) / 2
#     half_range = np.abs(high_vals - low_vals) / 2
# 
#     # Create Plot
#     fig, ax = plt.subplots(figsize=(7, 5))
#     y_pos = np.arange(len(params))
# 
#     # The bar starts at the minimum value and extends the full range
#     rects = ax.barh(y_pos, half_range * 2, left=np.minimum(low_vals, high_vals), 
#                     height=0.6, color="steelblue", alpha=0.7, align='center')
# 
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(params)
#     ax.set_xlabel("Calibrated k value")
#     ax.set_title("Tornado Plot: Impact of Parameter Uncertainty")
#     
#     plt.tight_layout()
#     return fig

# ==========================================
# UPDATED MAIN RUNNER
# ==========================================

def run_nomogram(initial_dose_per_kg, t_to_mean, prime_heparin, 
                 ibw_mean, ibw_sd, t_to_sd, t_on_mean, t_on_sd, 
                 model_name):
    
    # 1. Calibrate Best k
    k_best = find_best_k(initial_dose_per_kg, prime_heparin, ibw_mean, ibw_sd, 
                         t_to_mean, t_to_sd, t_on_mean, t_on_sd, model_name)
    
    # 2. Bootstrap for Uncertainty
    k_boot = bootstrap_k_distribution(initial_dose_per_kg, prime_heparin, ibw_mean, ibw_sd,
                                      t_to_mean, t_to_sd, t_on_mean, t_on_sd, model_name)
    mean_k, ci_low, ci_high = summarize_k_distribution(k_boot)

    # 3. Sensitivity Analysis (RESTORED)
    sens_df = sensitivity_analysis(
        initial_dose_per_kg, prime_heparin, 
        ibw_mean, ibw_sd, 
        t_to_mean, t_to_sd, 
        t_on_mean, t_on_sd, 
        model_name=model_name
    )
    sens_fig = plot_sensitivity(sens_df)
    tornado_fig = plot_tornado(sens_df)

    # 4. Generate Diagnostics Data (Simulation)
    n_sim = 1000
    ibw = np.random.normal(ibw_mean, ibw_sd, n_sim)
    heparin_bolus = initial_dose_per_kg * ibw
    t_to = np.clip(np.random.normal(t_to_mean, t_to_sd, n_sim), 0, None)
    t_on = np.clip(np.random.normal(t_on_mean, t_on_sd, n_sim), 0, None)
    
    ref_doses = [get_reference_dose(model_name, h, prime_heparin, w, t1, t2) 
                 for h, w, t1, t2 in zip(heparin_bolus, ibw, t_to, t_on)]
    
    test_doses = [simplified_model_dose(h, prime_heparin, w, t1, t2, k_best) 
                  for h, w, t1, t2 in zip(heparin_bolus, ibw, t_to, t_on)]
    
    # 5. Pack Data for Dashboard
    sim_df = pd.DataFrame({
        "IBW": ibw, "Ref_Dose": ref_doses, "Simp_Dose": test_doses, "TimeOn": t_on
    })
    
    # Metrics
    rmse = np.sqrt(np.mean((np.array(test_doses) - np.array(ref_doses))**2))
    mae = np.mean(np.abs(np.array(test_doses) - np.array(ref_doses)))
    r2 = np.corrcoef(test_doses, ref_doses)[0, 1] ** 2
    
    diagnostics = {
        "RMSE (IU)": rmse, "MAE (IU)": mae, "R²": r2,
        "% within ±5%": np.mean(np.abs(np.array(test_doses) - np.array(ref_doses)) <= 0.05 * np.array(ref_doses)) * 100,
        "% within ±10%": np.mean(np.abs(np.array(test_doses) - np.array(ref_doses)) <= 0.10 * np.array(ref_doses)) * 100
    }

    bias, loa_low, loa_high = bland_altman_stats(ref_doses, test_doses)
    
    # 6. Generate PDF & Plots
    pdf_path_raw = build_nomogram(mean_k) # Uses Nomographer
    center_pdf_on_a4("heparin_dose_decay_nomogram.pdf", "nomogram_a4.pdf")
    footer_text = f"Generated for parameters: Initial heparin: {initial_dose_per_kg}, heparin in prime: {prime_heparin},\ntime to CPB: {t_to_mean}±{t_to_sd}, time on CPB: {t_on_mean}±{t_on_sd}, IBW: {ibw_mean}±{ibw_sd}\nReference model: {model_name}. k={mean_k:.5f}. {APP_VERSION}"
    pdf_path = "nomogram_final.pdf"
    add_footer_to_pdf("nomogram_a4.pdf", pdf_path, footer_text)

    fig_ba = bland_altman_plot(ref_doses, test_doses, model_label=model_name)
    k_post_fig = plot_k_posterior(k_boot)
    
    scatter_fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(ref_doses, test_doses, c=t_on, cmap="viridis", alpha=0.8)
    ax.plot([min(ref_doses), max(ref_doses)], [min(ref_doses), max(ref_doses)], "r--")
    ax.set_xlabel(f"{model_name} Dose")
    ax.set_ylabel("Simplified Dose")
    plt.colorbar(sc, label="Time on CPB")

    summary_text = clinical_summary(mean_k, bias, loa_low, loa_high, ibw_mean, model_name)

    return (pdf_path, k_best, bias, loa_low, loa_high, fig_ba, mean_k, ci_low, ci_high, 
            k_post_fig, scatter_fig, sim_df, summary_text, sens_df, sens_fig, tornado_fig, "", diagnostics)

# ==========================================
# GRID GENERATION
# ==========================================

# Use these exact lists in both files
heparin_grid = [250, 300, 350, 400, 450, 500, 550, 600]
ibw_grid      = [40, 55, 70, 85, 100, 115]
time_to_grid  = [5, 15, 25, 35] 
time_on_grid  = [30, 60, 90, 120]
prime_grid    = [0, 5000, 10000]

def generate_v2_table_deterministic():
    """
    Generates lookup tables for BOTH models.
    """
    combos = list(itertools.product(heparin_grid, ibw_grid, time_to_grid, time_on_grid, prime_grid))
    total = len(combos)
    
    models = ["delavenne", "jia", "prodose", "lanoiselee", "meesters"]
    
    main_prog = st.progress(0.0)
    status_text = st.empty()
    
    for model in models:
        k_table = {}
        status_text.text(f"Generating table for {model}...")
        
        for i, (hpkg, ibw_m, tto_m, ton_m, p_hep) in enumerate(combos):
            ibw_sd_val = 10.0
            tto_sd_val = 0.25 * tto_m
            ton_sd_val = 15.0
            
            # Helper to run optim
            def get_k(t_on_val):
                return find_best_k(hpkg, p_hep, ibw_m, ibw_sd_val, tto_m, tto_sd_val, t_on_val, ton_sd_val, 
                                   model_name=model, n_sim=200)

            k_mu = get_k(ton_m)
            k_lo = get_k(ton_m - 2*ton_sd_val) # Slow elimination scenario
            k_hi = get_k(ton_m + 2*ton_sd_val) # Fast elimination scenario
            
            vals = [k_mu, k_lo, k_hi]
            k_table[(hpkg, ibw_m, tto_m, ton_m, p_hep)] = {
                'mu': k_mu, 'lo': min(vals), 'hi': max(vals)
            }
            
            if i % 100 == 0:
                main_prog.progress((i + 1) / total)

        filename = f"k_table_v2_{model}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(k_table, f)
        st.success(f"Saved {filename}")
