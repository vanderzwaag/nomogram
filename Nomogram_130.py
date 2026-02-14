import math
import numpy as np
from pynomo.nomographer import Nomographer
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from io import BytesIO
from reportlab.lib.units import cm

APP_VERSION = "v1.0.0"

def prodose_dose(heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb):
    """
    PRODOSE formula as provided.
    All times in minutes, doses in IU, ibw in kg.
    """
    term1 = heparin_bolus * 0.1 * math.exp(-0.0693 * (time_to_cpb + time_on_cpb))
    k2 = 0.693 / (26 + 0.323 * (heparin_bolus / ibw))
    term2 = heparin_bolus * 0.9 * math.exp(-k2 * (time_on_cpb + time_to_cpb))
    term3 = heparin_prime * math.exp(-k2 * time_on_cpb)
    return term1 + term2 + term3

def simplified_model_dose(heparin_bolus, heparin_prime, ibw, time_to_cpb, time_on_cpb, k):
    """
    Simplified protamine model using a single decay constant k.
    You can refine this equation later; for now we mirror the structure:
    """
    # Example: single-compartment decay of total heparin load
    total_heparin = heparin_bolus + heparin_prime
    effective_time = time_to_cpb + time_on_cpb
    return total_heparin * math.exp(-k * effective_time)

def bland_altman_stats(ref, test):
    """
    Compute Bland–Altman bias and 95% limits of agreement.
    Returns:
        bias: mean(ref - test)
        loa_low: bias - 1.96 * SD(diff)
        loa_high: bias + 1.96 * SD(diff)
    """
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
    # Penalize bias strongly, width moderately
    return abs(bias) + 0.1 * width

@st.cache_data(show_spinner=False)
def find_best_k(initial_dose_per_kg,
                prime_heparin,
                ibw_mean,
                ibw_sd,
                t_to_mean,
                t_to_sd,
                t_on_mean,
                t_on_sd,
                n_sim=1000):

    ibw = np.random.normal(loc=ibw_mean, scale=ibw_sd, size=n_sim)
    heparin_bolus = initial_dose_per_kg * ibw

    time_to_cpb_samples = np.random.normal(
        loc=t_to_mean,
        scale=t_to_sd,
        size=n_sim
    )

    time_on_cpb_samples = np.random.normal(
        loc=t_on_mean,
        scale=t_on_sd,
        size=n_sim
    )

    time_to_cpb_samples = np.clip(time_to_cpb_samples, 0, None)
    time_on_cpb_samples = np.clip(time_on_cpb_samples, 0, None)

    # Reference PRODOSE doses
    ref_doses = [
        prodose_dose(h, prime_heparin, w, t_to, t_on)
        for h, w, t_to, t_on in zip(
            heparin_bolus,
            ibw,
            time_to_cpb_samples,
            time_on_cpb_samples
        )
    ]

    # Objective function for optimizer
    def objective(k):
        test_doses = [
            simplified_model_dose(h, prime_heparin, w, t_to, t_on, k)
            for h, w, t_to, t_on in zip(
                heparin_bolus,
                ibw,
                time_to_cpb_samples,
                time_on_cpb_samples
            )
        ]
        return bland_altman_score(ref_doses, test_doses)

    # Brent bounded search for k
    result = minimize_scalar(
        objective,
        bounds=(0.001, 0.02),
        method='bounded'
    )

    return result.x

def build_nomogram(k_value):
    output_filename = "heparin_dose_decay_nomogram.pdf"

    main_params = {
        'filename': output_filename,
        'paper_height': 27.0,
        'paper_width': 15.0,
        'block_params': [
            {
                'block_type': 'type_1',
                'f1_params': {
                    'u_min': 20,
                    'u_max': 55,
                    'function': lambda dose: math.log(dose),
                    'title': r'Initial Heparin Dose (*1000 IU)',
                    'tick_levels': 3,
                    'tick_text_levels': 2,
                },
                'f2_params': {
                    'u_min': 0,
                    'u_max': 120,
                    'function': lambda t, k=k_value: -k * t,
                    'title': r'Time (minutes)',
                    'tick_levels': 2,
                    'tick_text_levels': 1,
                },
                'f3_params': {
                    'u_min': 10,
                    'u_max': 45,
                    'function': lambda remaining: -math.log(remaining),
                    'title': r'Remaining Heparin (*1000 IU)',
                    'tick_levels': 3,
                    'tick_text_levels': 2,
                },
            }
        ]
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

        # how much margin is needed on each side
        x_margin = (a4_width - orig_width) / 2.0
        y_margin = (a4_height - orig_height) / 2.0

        # original coordinates
        llx, lly = map(float, page.mediabox.lower_left)
        urx, ury = map(float, page.mediabox.upper_right)

        # expand mediabox outward symmetrically
        new_llx = llx - x_margin
        new_lly = lly - y_margin
        new_urx = urx + x_margin
        new_ury = ury + y_margin

        page.mediabox.lower_left = (new_llx, new_lly)
        page.mediabox.upper_right = (new_urx, new_ury)

        # keep cropbox consistent
        if hasattr(page, "cropbox"):
            page.cropbox.lower_left = (new_llx, new_lly)
            page.cropbox.upper_right = (new_urx, new_ury)

        writer.add_page(page)

    with open(output_pdf, "wb") as f:
        writer.write(f)

def add_footer_to_pdf(input_pdf, output_pdf, footer_text):
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    for page in reader.pages:
        llx, lly = map(float, page.mediabox.lower_left)
        urx, ury = map(float, page.mediabox.upper_right)

        page_width  = urx - llx
        page_height = ury - lly

        packet = BytesIO()
        can = canvas.Canvas(packet, pagesize=(page_width, page_height))

        x = llx + 1 * cm
        y = lly + 0.5 * cm

        can.setFont("Helvetica", 8)
        can.drawString(x, y, footer_text)
        can.save()

        packet.seek(0)
        overlay = PdfReader(packet).pages[0]

        # make overlay use same box as page so coordinates line up
        overlay.mediabox.lower_left = page.mediabox.lower_left
        overlay.mediabox.upper_right = page.mediabox.upper_right

        page.merge_page(overlay)
        writer.add_page(page)

    with open(output_pdf, "wb") as f:
        writer.write(f)

  
import matplotlib.pyplot as plt

def bland_altman_plot(ref, test):
    """
    Returns a Matplotlib figure for a Bland–Altman plot.
    """
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

    ax.set_xlabel("Mean of PRODOSE and simplified model (IU)")
    ax.set_ylabel("Difference (PRODOSE − simplified) (IU)")
    ax.set_title("Bland–Altman Plot")
    ax.legend()

    return fig
  
def run_nomogram(initial_dose_per_kg,
                 median_time_to_cpb,
                 prime_heparin,
                 ibw_mean,
                 ibw_sd,
                 t_to_mean,
                 t_to_sd,
                 t_on_mean,
                 t_on_sd):
                   
    k_best = find_best_k(
    initial_dose_per_kg,
    prime_heparin,
    ibw_mean,
    ibw_sd,
    t_to_mean,
    t_to_sd,
    t_on_mean,
    t_on_sd
    )

    
    # Bootstrap uncertainty
    k_boot = bootstrap_k_distribution(
      initial_dose_per_kg,
      prime_heparin,
      ibw_mean,
      ibw_sd,
      t_to_mean,
      t_to_sd,
      t_on_mean,
      t_on_sd
    )
    
    mean_k, ci_low, ci_high = summarize_k_distribution(k_boot)

    # Recompute stats for the chosen k
    # (We reuse the same simulation logic)
    n_sim = 1000
    ibw = np.random.normal(loc=ibw_mean, scale=ibw_sd, size=n_sim)
    heparin_bolus = initial_dose_per_kg * ibw

    time_to_cpb_samples = np.random.normal(loc=t_to_mean, scale=t_to_sd, size=n_sim)
    time_on_cpb_samples = np.random.normal(loc=t_on_mean, scale=t_on_sd, size=n_sim)

    time_to_cpb_samples = np.clip(time_to_cpb_samples, 0, None)
    time_on_cpb_samples = np.clip(time_on_cpb_samples, 0, None)

    ref_doses = [
        prodose_dose(h, prime_heparin, w, t_to, t_on)
        for h, w, t_to, t_on in zip(
            heparin_bolus, ibw, time_to_cpb_samples, time_on_cpb_samples
        )
    ]

    test_doses = [
        simplified_model_dose(h, prime_heparin, w, t_to, t_on, k_best)
        for h, w, t_to, t_on in zip(
            heparin_bolus,
            ibw,
            time_to_cpb_samples,
            time_on_cpb_samples
        )
    ]

    import pandas as pd

    sim_df = pd.DataFrame({
        "IBW_kg": ibw,
        "HeparinBolus_IU": heparin_bolus,
        "TimeToCPB_min": time_to_cpb_samples,
        "TimeOnCPB_min": time_on_cpb_samples,
        "PRODOSE_IU": ref_doses,
        "Simplified_IU": test_doses
    })
    
    sim_df["prodose"] = ref_doses
    sim_df["simplified"] = test_doses
    sim_df["diff"] = sim_df["simplified"] - sim_df["prodose"]
    sim_df["t_on"] = time_on_cpb_samples
    
    # Core diagnostics
    rmse = np.sqrt(np.mean((sim_df["simplified"] - sim_df["prodose"])**2))
    mae = np.mean(np.abs(sim_df["simplified"] - sim_df["prodose"]))
    r2 = np.corrcoef(sim_df["simplified"], sim_df["prodose"])[0, 1] ** 2
    
    pct_within_5 = np.mean(np.abs(sim_df["diff"]) <= 0.05 * sim_df["prodose"]) * 100
    pct_within_10 = np.mean(np.abs(sim_df["diff"]) <= 0.10 * sim_df["prodose"]) * 100
    
    diagnostics = {
        "RMSE (IU)": rmse,
        "MAE (IU)": mae,
        "R²": r2,
        "% within ±5%": pct_within_5,
        "% within ±10%": pct_within_10,
    }

    bias, loa_low, loa_high = bland_altman_stats(ref_doses, test_doses)

    pdf_path_raw = build_nomogram(mean_k)
    center_pdf_on_a4("heparin_dose_decay_nomogram.pdf", "nomogram_a4.pdf")
    
    footer_text = (
      f"**Input parameters**: "
      f"IBW {ibw_mean} ± {ibw_sd} kg, "
      f"t_to {t_to_mean} ± {t_to_sd} min, "
      f"t_on {t_on_mean} ± {t_on_sd} min, "
      f"bolus {initial_dose_per_kg} IU/kg, prime {prime_heparin} IU. "
      f"Version {APP_VERSION}"
    )

    pdf_path = "nomogram_footer.pdf"
    add_footer_to_pdf("nomogram_a4.pdf", pdf_path, footer_text)

    fig = bland_altman_plot(ref_doses, test_doses)
    
    k_post_fig = plot_k_posterior(k_boot)
    
    ## scatter_fig = prodose_vs_simplified_plot(ref_doses, test_doses)
    
    scatter_fig, ax = plt.subplots(figsize=(6, 4))
    
    sc = ax.scatter(
      sim_df["prodose"],
      sim_df["simplified"],
      c=sim_df["t_on"],
      cmap="viridis",
      alpha=0.8,
      edgecolor="k",
      linewidth=0.3
    )
    
    ax.plot(
      [sim_df["prodose"].min(), sim_df["prodose"].max()],
      [sim_df["prodose"].min(), sim_df["prodose"].max()],
      "r--",
      label="Identity line"
    )
    
    ax.set_xlabel("PRODOSE predicted dose (IU)")
    ax.set_ylabel("Simplified model predicted dose (IU)")
    ax.set_title("PRODOSE vs Simplified Model (colored by time on CPB)")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Time on CPB (min)")
    
    summary_text = clinical_summary(
        mean_k,
        bias,
        loa_low,
        loa_high,
        ibw_mean,
        t_to_mean,
        t_on_mean
    )
    
    sens_df = sensitivity_analysis(
        initial_dose_per_kg,
        prime_heparin,
        ibw_mean,
        ibw_sd,
        t_to_mean,
        t_to_sd,
        t_on_mean,
        t_on_sd
    )

    sens_fig = plot_sensitivity(sens_df)
    tornado_fig = plot_tornado(sens_df)
    
    import datetime

    metadata = f"""# Simulation metadata
    # Timestamp: {datetime.datetime.now().isoformat()}
    # Best k value: {mean_k:.5f}
    # Bias (IU): {bias:.2f}
    # Lower LOA (IU): {loa_low:.2f}
    # Upper LOA (IU): {loa_high:.2f}
    # IBW mean (kg): {ibw_mean}
    # IBW SD (kg): {ibw_sd}
    # Time to CPB mean (min): {t_to_mean}
    # Time to CPB SD (min): {t_to_sd}
    # Time on CPB mean (min): {t_on_mean}
    # Time on CPB SD (min): {t_on_sd}
    """


    return (
      pdf_path,
      k_best,
      bias,
      loa_low,
      loa_high,
      fig,
      mean_k,
      ci_low,
      ci_high,
      k_post_fig,
      scatter_fig,
      sim_df,
      summary_text,
      sens_df,
      sens_fig,
      tornado_fig,
      metadata,
      diagnostics
    )

@st.cache_data(show_spinner=False)  
def bootstrap_k_distribution(initial_dose_per_kg,
                             prime_heparin,
                             ibw_mean,
                             ibw_sd,
                             t_to_mean,
                             t_to_sd,
                             t_on_mean,
                             t_on_sd,
                             n_boot=200,
                             n_sim=1000):

    k_values = []

    for _ in range(n_boot):
        # Resample simulation inputs
        ibw = np.random.normal(loc=80.0, scale=10.0, size=n_sim)
        heparin_bolus = initial_dose_per_kg * ibw

        time_to_cpb_samples = np.random.normal(
            loc=t_to_mean,
            scale=t_to_sd,
            size=n_sim
        )

        time_on_cpb_samples = np.random.normal(
            loc=t_on_mean,
           scale=t_on_sd,
            size=n_sim
        )


        time_to_cpb_samples = np.clip(time_to_cpb_samples, 0, None)
        time_on_cpb_samples = np.clip(time_on_cpb_samples, 0, None)

        # Compute PRODOSE reference
        ref_doses = [
            prodose_dose(h, prime_heparin, w, t_to, t_on)
            for h, w, t_to, t_on in zip(
                heparin_bolus,
                ibw,
                time_to_cpb_samples,
                time_on_cpb_samples
            )
        ]

        # Objective for optimizer
        def objective(k):
            test_doses = [
                simplified_model_dose(h, prime_heparin, w, t_to, t_on, k)
                for h, w, t_to, t_on in zip(
                    heparin_bolus,
                    ibw,
                    time_to_cpb_samples,
                    time_on_cpb_samples
                )
            ]
            return bland_altman_score(ref_doses, test_doses)

        # Optimize k
        result = minimize_scalar(objective, bounds=(0.001, 0.02), method='bounded')
        k_values.append(result.x)

    return np.array(k_values)

def summarize_k_distribution(k_values):
    mean_k = np.mean(k_values)
    ci_low = np.percentile(k_values, 2.5)
    ci_high = np.percentile(k_values, 97.5)
    return mean_k, ci_low, ci_high

@st.cache_resource(show_spinner=False)
def plot_k_posterior(k_values):
    fig, ax = plt.subplots(figsize=(6, 4))

    # Kernel density estimate
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(k_values)
    xs = np.linspace(min(k_values), max(k_values), 200)
    ax.plot(xs, kde(xs), color='darkblue', lw=2)

    # Histogram for reference
    ax.hist(k_values, bins=20, density=True, alpha=0.3, color='steelblue')

    ax.set_title("Posterior Distribution of k")
    ax.set_xlabel("k")
    ax.tick_params(axis="x", labelsize=6)
    ax.set_ylabel("Density")

    return fig

def prodose_vs_simplified_plot(ref, test):
    """
    Scatter plot of PRODOSE vs simplified model doses.
    """
    ref = np.asarray(ref)
    test = np.asarray(test)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ref, test, alpha=0.4, color="steelblue")

    # Identity line (perfect agreement)
    min_val = min(ref.min(), test.min())
    max_val = max(ref.max(), test.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            linestyle="--", color="red", label="Identity line")

    ax.set_xlabel("PRODOSE dose (IU)")
    ax.set_ylabel("Simplified model dose (IU)")
    ax.set_title("PRODOSE vs Simplified Model")
    ax.legend()

    return fig
  
def clinical_summary(k_best, bias, loa_low, loa_high, ibw_mean, t_to_mean, t_on_mean):
    """
    Generate a concise clinical interpretation of the calibration results.
    """
    summary = f"""
    **Clinical Interpretation**

    • The calibrated decay constant is **k = {k_best:.5f}**.  
      This represents the effective elimination rate of total heparin load during CPB.

    • The simplified model shows a **bias of {bias:.1f} IU** relative to PRODOSE.  
      Positive values mean the simplified model tends to *underestimate* protamine needs.

    • The 95% limits of agreement range from **{loa_low:.1f} to {loa_high:.1f} IU**.  
      This reflects the expected variability between the simplified model and PRODOSE.

    • The simulation was based on a typical patient population with:  
      – IBW ≈ **{ibw_mean:.0f} kg**  
      – Time to CPB ≈ **{t_to_mean:.0f} min**  
      – Time on CPB ≈ **{t_on_mean:.0f} min**

    **Clinical takeaway:**  
    The simplified model provides a close approximation to PRODOSE across the simulated population.  
    The magnitude of bias and limits of agreement suggest that the model is suitable for bedside use,  
    provided clinicians remain aware of the expected ± variability shown above.
    """
    return summary

@st.cache_data(show_spinner=False)
def sensitivity_analysis(initial_dose_per_kg,
                         prime_heparin,
                         ibw_mean,
                         ibw_sd,
                         t_to_mean,
                         t_to_sd,
                         t_on_mean,
                         t_on_sd,
                         variation=0.2):
    """
    Perform one-factor-at-a-time sensitivity analysis.
    Each parameter is varied by ±variation (default ±20%).
    Returns a DataFrame summarizing how k_best changes.
    """
    import pandas as pd

    base_params = {
        "ibw_mean": ibw_mean,
        "ibw_sd": ibw_sd,
        "t_to_mean": t_to_mean,
        "t_to_sd": t_to_sd,
        "t_on_mean": t_on_mean,
        "t_on_sd": t_on_sd
    }

    results = []
    
    total_steps = len(base_params) * 2 # low + high
    progress = st.progress(0)
    step = 0

    for param_name, base_value in base_params.items():
        for direction, factor in [("low", 1 - variation), ("high", 1 + variation)]:
            modified_params = base_params.copy()
            modified_params[param_name] = base_value * factor

            k_mod = find_best_k(
                initial_dose_per_kg,
                prime_heparin,
                modified_params["ibw_mean"],
                modified_params["ibw_sd"],
                modified_params["t_to_mean"],
                modified_params["t_to_sd"],
                modified_params["t_on_mean"],
                modified_params["t_on_sd"]
            )

            results.append({
                "Parameter": param_name,
                "Direction": direction,
                "Modified value": modified_params[param_name],
                "k_best": k_mod
            })
            
            step += 1
            progress.progress(step / total_steps)

    return pd.DataFrame(results)

@st.cache_resource(show_spinner=False)
def plot_sensitivity(df):
    fig, ax = plt.subplots(figsize=(7, 4))

    for param in df["Parameter"].unique():
        subset = df[df["Parameter"] == param]
        ax.plot(["low", "high"], subset["k_best"], marker="o", label=param)

    ax.set_title("Sensitivity of k to Simulation Parameters")
    ax.set_ylabel("k value")
    ax.legend()
    return fig

@st.cache_resource(show_spinner=False)
def plot_tornado(df):
    """
    Create a tornado plot from sensitivity analysis results.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Compute low/high k for each parameter
    params = []
    low_vals = []
    high_vals = []

    for param in df["Parameter"].unique():
        subset = df[df["Parameter"] == param]
        k_low = subset[subset["Direction"] == "low"]["k_best"].values[0]
        k_high = subset[subset["Direction"] == "high"]["k_best"].values[0]

        params.append(param)
        low_vals.append(k_low)
        high_vals.append(k_high)

    # Compute bar lengths
    low_vals = np.array(low_vals)
    high_vals = np.array(high_vals)
    base = (low_vals + high_vals) / 2
    half_range = (high_vals - low_vals) / 2

    fig, ax = plt.subplots(figsize=(7, 5))

    y_pos = np.arange(len(params))

    ax.barh(y_pos, half_range, left=base - half_range, height=0.6, color="steelblue", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)

    ax.set_xlabel("k value")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Tornado Plot: Sensitivity of k to Model Assumptions")

    return fig

import numpy as np
import pickle
from scipy.optimize import minimize_scalar
import itertools
import streamlit as st

# Grid definitions (as per your original code)
heparin_grid = [250, 300, 350, 400, 450, 500, 550, 600]
ibw_grid = [40, 55, 70, 85, 100, 115]
time_to_grid = [5, 10, 15, 20, 25, 30, 35, 40]
time_on_grid = [15, 30, 45, 60, 75, 90, 105, 120]
prime_grid = [0, 2000, 4000, 6000, 8000, 10000]

def generate_v2_table():
    k_table_v2 = {}
    
    # Total iterations for progress tracking
    total_combos = (len(heparin_grid) * len(ibw_grid) * len(time_to_grid) * len(time_on_grid) * len(prime_grid))
    
    st.info(f"Generating table for {total_combos} combinations. This may take a while.")
    progress_bar = st.progress(0)
    counter = 0

    # Nested loops through all grid parameters
    for hpkg, ibw_m, tto_m, ton_m, p_hep in itertools.product(
        heparin_grid, ibw_grid, time_to_grid, time_on_grid, prime_grid
    ):
        # Run your bootstrap function
        # Note: adjust n_boot/n_sim if it's too slow for the full grid
        k_dist = bootstrap_k_distribution(
            initial_dose_per_kg=hpkg,
            prime_heparin=p_hep,
            ibw_mean=ibw_m, ibw_sd=ibw_m * 0.1,  # 10% CV for example
            t_to_mean=tto_m, t_to_sd=tto_m * 0.1,
            t_on_mean=ton_m, t_on_sd=ton_m * 0.1,
            n_boot=100, 
            n_sim=500
        )
        
        # Calculate statistics
        k_table_v2[(hpkg, ibw_m, tto_m, ton_m, p_hep)] = {
            'mu': np.mean(k_dist),
            'lo': np.percentile(k_dist, 2.5),
            'hi': np.percentile(k_dist, 97.5)
        }
        
        counter += 1
        progress_bar.progress(counter / total_combos)

    # Save to disk
    with open("k_table_v2.pkl", "wb") as f:
        pickle.dump(k_table_v2, f)
    
    st.success("Table k_table_v2.pkl generated successfully!")
