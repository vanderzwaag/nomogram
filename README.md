# Heparin decay nomogram pipeline

A simulation-calibrated informatics pipeline that fits a single-compartment
exponential approximation to published pharmacokinetic models of unfractionated
heparin, and renders the result as an institution-specific decay nomogram.

> **Research and educational instrument.** Not for clinical use, for the care of
> any patient, or to determine or influence any drug dose. See
> [Intended purpose](#intended-purpose) below and the [LICENSE](LICENSE).

Supporting code for manuscript **JCVA-D-26-01511**. Every number the manuscript
reports regenerates from one seeded command at a tagged commit.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22846042.svg)](https://doi.org/10.5281/zenodo.22846042)
[![tests](https://github.com/vanderzwaag/nomogram/actions/workflows/tests.yml/badge.svg)](https://github.com/vanderzwaag/nomogram/actions/workflows/tests.yml)

---

## What it does

Published pharmacokinetic models of unfractionated heparin during
cardiopulmonary bypass are biexponential and require a computer to evaluate. A
nomogram does not.

This pipeline asks how well a single exponential — one decay constant *k*,
calibrated to an institution's own dosing, weight and bypass-time distributions
— can approximate a chosen reference model, and quantifies where it fails. The
answer differs sharply by model: for Lanoiselée and Jia the approximation holds
to under 0.5% mean absolute error; for Delavenne it does not hold at all
(10.3%, and 72% after repeated supplemental boluses).

**What it does not do.** It represents the pharmacokinetic central-compartment
amount only. It does not model the pharmacodynamic layer — anti-factor Xa
activity or activated clotting time — nor peripheral-compartment heparin,
antithrombin, protamine pharmacology or heparin rebound. Converting a residual
amount into a protamine dose requires applying an institutional ratio, which
this pipeline does not supply.

## Installation

Python 3.10 or newer. Pure Python — no system packages, no LaTeX, no compiler.

```bash
git clone https://github.com/vanderzwaag/nomogram.git
cd nomogram

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python -m pip install -r requirements.txt
```

To reproduce the published numbers exactly, install the versions the archived
run used instead:

```bash
python -m pip install -r requirements-frozen.txt
```

Verify the installation:

```bash
python -m pytest tests/ -q         # the full suite, ~10 s
```

## Running it

```bash
python run_analysis.py --seed 20260912 --outdir outputs/frozen
python make_supplementary_figures.py
```

`outputs/frozen/summary.md` holds every value in manuscript-ready tables.
`outputs/frozen/manifest.json` records the seed, the git commit, package
versions and every analysis decision.

To check a run against the archive:

```bash
python check_frozen.py outputs/frozen
```

The numbers agree to a relative tolerance of 1e-9 by default, and the check
prints the largest difference it found. Measured drift against an environment a
major release behind (numpy 2.2.6, scipy 1.15.3, pandas 2.3.3) is 5e-14 at
worst, in a p-value of 5e-77.

`--exact` requires byte-identical files. That holds when you re-run on the same
machine in the pinned environment, and it is worth being able to check, but it
is a claim about a machine rather than about the analysis: the last bit of a
transcendental function is not specified by IEEE 754, so a different libm moves
a result by one unit in the last place and everything downstream follows.

`python run_analysis.py --help` lists the options: the cohort, the calibration
objective, the evaluation mode, the prime-timing convention and the sample
sizes are all switchable, and each is recorded in the manifest.

### The dashboard

```bash
python -m pip install streamlit
streamlit run streamlit_app.py
```

This opens a local instance in your browser at `http://localhost:8501`. **There
is no hosted deployment** — the dashboard runs only on the machine you start it
on, and nothing it computes leaves that machine. It is not required for any
result in the manuscript: `run_analysis.py` produces all of those without it.

## Reference models

| Model | Structure | Status |
|---|---|---|
| Lanoiselée | Two-compartment, PK/PD | active |
| Delavenne | Two-compartment with weight covariates, PK/PD | active |
| Jia | Two-compartment, adult (small-bodied population) | active |
| Meesters | Closed-form biexponential | active |
| PRODOSE | Closed-form biexponential, dose-dependent | active |
| PRODOSE-2 | Closed-form with circuit dilution | **withheld — pending validation and peer review** |

A withheld model keeps its implementation, parameters and tests; it simply does
not appear in the pipeline. Moving one in or out is a single line in
`nomogram_core.PENDING_MODELS`.

Published parameter values, their provenance and their reported uncertainty
are recorded per model in `nomogram_core.MODEL_PARAMETERS`.

## Layout

```
├── .github/workflows/ CI: the suite, each test file alone, and a check that
│                      the reported numbers are unchanged
├── data/k_tables/     calibrated k per grid node, one CSV per model
├── outputs/frozen/    the archived run: tables, figures, manifest, summary
├── tests/             the test suite
└── *.py               the pipeline; entry points are run_analysis.py,
                       make_supplementary_figures.py and streamlit_app.py
```

| File | Role |
|---|---|
| `nomogram_core.py` | The six models, trajectory and endpoint forms, and the simplified model. No Streamlit or plotting imports, so it runs in CI. |
| `calibration.py` | Seeded cohort sampling, objective functions, endpoint vs trajectory calibration, both uncertainty intervals, sensitivity analyses. |
| `agreement.py` | Sign convention, Bland–Altman with CIs, proportional bias, relative errors, stratification, threshold coverage. |
| `parameter_spaces.py` | Canonical cohorts, the calibration grid, boundary corners, shifted institutions. |
| `topup.py` | Supplemental-bolus grid with worst-case reporting. |
| `benchmarks.py` | Implementation verification and reproduction of published values. |
| `nomogram_render.py` | Nomogram geometry and rendering, shared by the printed PDF and the interactive chart. |
| `k_table_io.py` | Reads and writes `data/k_tables/*.csv`, the decay-constant lookup tables, each with a JSON provenance header. No pickles: unpickling a file shipped in a repository executes whatever is in it. |
| `run_analysis.py` | Regenerates every reported number from one seeded command. |
| `check_frozen.py` | Checks a fresh run against the archive: numbers equal to a tolerance by default, byte-identical with `--exact`. |
| `make_supplementary_figures.py` | Main-text Figure 2 and supplementary figures S1–S4. |
| `nomogram_models.py` | Dashboard-facing wrappers and figures. |
| `streamlit_app.py` | The dashboard. |
| `tests/` | The test suite. |

## How the nomogram works

The chart solves one relation:

```
R = D₀ · exp(−k·t)        ⟺        log₁₀ R = log₁₀ D₀ − k·t / ln 10
```

which is an alignment chart with three parallel axes — total load on the left,
elapsed time in the middle, residual on the right. Connect the total load and
the elapsed time with a straightedge and read the residual where the line
crosses the third axis.

The residual scale runs *downwards*. That is not a stylistic choice: the dose
terms only cancel out of the time axis if one outer scale is reversed, so the
chart would not work otherwise. `tests/test_nomogram_render.py` asserts the
alignment property directly rather than trusting the derivation.

A printed nomogram cannot carry an uncertainty band. *k* is baked into the
spacing of the time axis, so a different constant is a differently-spaced chart,
not a wider line. The band is a feature of the interactive tool only.

## Reproducibility

Every draw of randomness comes from an explicit seeded generator, and each stage
of the analysis draws from its own independent stream derived from the master
seed, so adding a stage cannot shift the numbers produced by the others.

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) records what each analysis decision
is and why, per reviewer comment, together with the defects corrected in the
course of making the analysis reproducible.

**Verification.** Automatic checks confirm each closed-form solution actually
solves its own differential equations and satisfies the identities they imply.
Six landmark values from a published simulation (Delavenne Figure 3) are
reproduced to within 3.0%. No participant-level data from any source study is
held, so no model is re-fitted and no prediction is compared with an
observation; no external or predictive validation is claimed. What is
demonstrated is faithful implementation of the published models — not that the
published models are correct.

## Intended purpose

This software is a **research and educational instrument**. Its purpose is to
demonstrate and teach an informatics method under simulated conditions.

It is **not intended for clinical use**, for the care or management of any
patient, or to determine, inform or influence any drug dose. Within that
intended purpose it is not a medical device.

Nothing here has been cleared, approved or evaluated by any regulatory
authority. Outputs have not been validated against measured anti-factor Xa
activity, activated clotting time or clinical reversal in patients, and no
clinical benefit has been demonstrated. Anyone using it outside the intended
purpose does so on their own responsibility, and may bring the tool within
medical-device oversight in their jurisdiction; the authors accept no
responsibility for such use and make no assessment of how any jurisdiction would
classify it.

## Licence

[MIT](LICENSE) — © 2026 Stanislaw Vander Zwaag, Imre Kukel, Asen Petrov,
Jens Fassl.

## Citing

Please cite both the manuscript and the software. Machine-readable metadata is
in [CITATION.cff](CITATION.cff); release history is in
[CHANGELOG.md](CHANGELOG.md).

**Manuscript.** Vander Zwaag S, Kukel I, Petrov A, Fassl J. From
multi-compartment models to the bedside: a simulation-calibrated informatics
pipeline for generating institution-specific heparin decay nomograms. *Journal
of Cardiothoracic and Vascular Anesthesia*. Manuscript JCVA-D-26-01511, under
revision — replace with volume, pages and DOI on acceptance.

**Software.** Vander Zwaag S, Kukel I, Petrov A, Fassl J. *Heparin decay
nomogram pipeline*, version 2.0.0. Zenodo, 2026.
doi:[10.5281/zenodo.22846043](https://doi.org/10.5281/zenodo.22846043)

Zenodo mints two DOIs, and which one to use depends on what you mean:

| DOI | Resolves to | Use it when |
|---|---|---|
| [10.5281/zenodo.22846043](https://doi.org/10.5281/zenodo.22846043) | release 2.0.0, permanently | the point is which numbers were produced |
| [10.5281/zenodo.22846042](https://doi.org/10.5281/zenodo.22846042) | the latest release, always | the point is the software in general |

Cite the version DOI in a paper reporting results from this pipeline: 2.0.0
changes every statistic 1.x reported, so a citation without a version is
ambiguous about which numbers are meant.

## Generative AI declaration

Claude (Anthropic) was used for code refactoring, test authoring and editing.
All analytical decisions, parameter values and interpretations are the authors'.
