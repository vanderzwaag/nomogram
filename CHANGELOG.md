# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

Every release is reproducible: the analysis regenerates byte-for-byte from a
single seeded command at the tagged commit. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

---

## [2.0.0] — 2026-09-13

A major release. It carries breaking changes to the public API, changes every
statistic the project reports, removes one reference model from the active
pipeline, and corrects two errors in previously published parameter values.
**Results produced with 1.x cannot be compared with results from this release.**

Prepared in response to peer review of manuscript JCVA-D-26-01511. Reviewer
comment identifiers (EB-1…EB-8, R1, R2) are given where a change answers one.

### Fixed — defects that changed reported numbers

- **A table the pipeline no longer produces was still in the frozen archive.**
  `00_verification_source.csv` dated from the first version of the runner, when
  the benchmark table was a placeholder; five of its six rows read "PENDING -
  expected value not yet transcribed from source" and its expected column was
  empty. It was superseded by `00_source_printed_value_checks.csv` and should
  have gone with it, but the runner writes into the output directory without
  clearing it, so the obsolete file persisted. The manifest now records which
  CSVs a run wrote, and the runner warns about any others it finds.

- **No random seed anywhere (EB-3, R1 Figure 2).** Nine calls to the global
  `np.random` ran unseeded, so every execution drew a different synthetic
  cohort and produced different constants and statistics. Because the
  calibration objective minimises the absolute bias, the residual bias came out
  at a few IU with a sign that changed from run to run: re-running the 1.x
  procedure twelve times gives bias values from −6.95 to +7.42 IU, positive in
  42% of runs. This is the root cause of the manuscript reporting −5.5 IU in
  the body and +5.62 IU in the Figure 2 legend for one analysis. All randomness
  now flows from explicit seeded `numpy.random.Generator` instances, with the
  seed recorded in the run manifest.
- **Three different decay constants in circulation for one configuration.** A
  lookup table built at `n_sim=200`, a fresh single-cohort fit at `n_sim=1000`
  used for the agreement statistics, and the replicate mean printed on the PDF.
  One constant — the replicate mean — is now used for the statistics, the
  figures, the printed nomogram and its footer alike.
- **Two opposite difference orderings.** The dashboard computed
  nomogram − reference while the Bland–Altman analysis computed
  reference − nomogram. One convention is now defined in `agreement.py` and
  enforced everywhere: `difference = reference model − nomogram`, so a positive
  value means the nomogram under-estimates residual heparin.
- **Lanoiselée interindividual variability was the wrong column.** The values
  `0.0983 / 0.111 / 0.395 / 0.21` are that column's parenthetical %RSE figures
  (9.83, 11.1, 39.5, 21.0) divided by 100 — the precision of the variability
  estimates, not the variability. All four matched to the digit. The correct
  values are `0.27 / 0.22 / 0.74 / 0.41`, about 2.1× larger, so every credible
  band drawn for this model in 1.x was roughly half the width it should have
  been.
- **Jia variability was wrong three times over.** The values
  `[0.073, 0.081, 0.144, 0.318]` are the *population-mean* %RSE column
  (7.25, 8.09, 14.40, 31.8) divided by 100, in the source's printed row order
  Cl, Vc, **Q, Vp** — so Vp and Q also received each other's values — and a
  square root was then applied, which would have been correct for the ω² column
  but not for these. The peripheral volume was given 38% interindividual
  variability where the source fixes it at zero. The function's own docstring
  gave a third set (`0.176 / 0.114 / 0.0573 / 0.111`) matching no column under
  any transformation. Corrected to √ of the published ω²:
  `0.3493 / 0.3240 / 0.3127`, and zero for Vp. Delavenne was transcribed
  correctly and is unchanged — it is the control that established the other two
  were errors rather than a different convention.
- **Superposition was invalid for the dose-dependent models (EB-5).** PRODOSE
  and PRODOSE-2 make the slow-pool half-life a function of IU/kg and are
  therefore not linear in dose. The 1.x top-up analysis evaluated a 5,000 IU
  supplemental bolus in a 70 kg patient using the elimination half-life that a
  71 IU/kg induction dose would have had. Both models now hold the
  dose-dependent covariate at the index bolus.
- **No lookup table was ever shipped for PRODOSE-2.** `get_k_stats` fell back
  to a hard-coded `k = 0.007` with no warning; the calibrated value is about
  0.0042, a 67% error in the decay constant, invisible to the user. The silent
  fallback is gone.
- **The Nomogram tab showed a bundled screenshot.** `Figure.png` was a static
  picture of a chart with a straightedge drawn on it, displayed beside the PDF
  the user had just generated. It could not follow the reference model, the
  calibrated constant or the dose range they had selected, so it was guaranteed
  to disagree with the chart next to it. The tab now draws the geometry object
  the PDF was rendered from, with the straightedge placed at the cohort
  entered; the image is removed from the repository.
- **Two independent nomogram implementations fed by two different constants.**
  The printed PDF was drawn by PyNomo from the replicate mean; the on-screen
  chart was a separate matplotlib implementation reading `k_mu` from the lookup
  table. Both now come from `nomogram_render`.
- **`invert_yaxis()` inside a shared-axis loop** in the sensitivity figure
  applied once per call, so with six models it cancelled out and the rows only
  appeared in the intended order by accident.
- **Two spellings of the Lanoiselée `Q` parameter** — 4.7928 mL/min in the
  endpoint model, 287.57/60 = 4.79283 in the credible-interval simulation.
  Unified on the unpublished-rounding value; worth about 0.02 IU at the
  reversal timepoint.
- **Model equations were duplicated** between `streamlit_app.py` (trajectory)
  and `nomogram_models.py` (endpoint). Both now come from `nomogram_core`, and
  a test asserts they agree at the reversal timepoint.

### Changed — breaking

- **PRODOSE-2 is withheld from the active pipeline**, pending validation and
  peer review. The project now reports **five** reference models, not six. The
  model is declared in `nomogram_core.PENDING_MODELS`; its implementation,
  parameters and tests remain and are still exercised. `MODEL_NAMES` is now the
  active set and `ALL_MODEL_NAMES` the full one.
- **The Jia model is adult, not paediatric (EB-4, R1 p11 L46).** Its central
  volume of 3.04 L is within 2% of Delavenne's adult 3.1 L, where a 10 kg child
  would be near 0.4 L; the source is an adult cardiac-surgical study in a
  relatively small-bodied population. The paediatric grid, paediatric cohorts,
  paediatric institutions and the allometric-scaling option built for it have
  been removed. A `small_bodied_adults` institution (mean 58 kg) tests
  transportability to the body size Jia was derived in.
- **The calibration target is stated, and the trajectory claim withdrawn
  (EB-4).** Calibration evaluates one endpoint per simulated patient, the
  residual amount at `t = time to CPB + time on CPB`. Trajectory calibration is
  available as `--evaluation-mode` and reported as a sensitivity analysis: it
  raises *k* by 18% (Jia) to 55% (Delavenne).
- **The peer-review password gate is removed (EB-6).** Describing the pipeline
  as open source while putting the dashboard behind a password was a
  contradiction, and the code is now public.
- **PyNomo, PyX, LaTeX, Ghostscript, ReportLab and PyPDF2 are no longer
  required.** `packages.txt` is gone. The nomogram is drawn by
  `nomogram_render` in matplotlib, which also made the printed chart testable —
  it was the only output the suite could not cover.
- **The lookup tables are regenerated deterministically**, each carrying a
  `__metadata__` record of seed, grid, sample size and calibration conventions.
  Tables without that record are reported as unreproducible rather than used
  silently. Only active models get one: the table for the withheld PRODOSE-2 is
  no longer shipped, since the generator cannot rebuild a table for a model the
  pipeline does not offer.
- **`Nomogram_Models.py` is renamed `nomogram_models.py`.** Module names are
  lowercase by convention (PEP 8), and a capitalised one is a filename that
  differs only in case from its own package style -- a hazard on
  case-insensitive filesystems. `import Nomogram_Models` becomes
  `import nomogram_models`.
- **API:** `bootstrap_k_distribution` renamed to `monte_carlo_k_distribution`
  (no dataset is resampled; the old name is aliased). `find_best_k`,
  `sensitivity_analysis` and `run_nomogram` now take a `seed`.
  `build_nomogram` no longer takes PyNomo parameters. `center_pdf_on_a4` and
  `add_footer_to_pdf` are removed.

### Added

- **Parameter-uncertainty propagation (EB-2).** The published PK parameters are
  resampled from the uncertainty each source actually reports — %RSE for
  Lanoiselée and Delavenne, the published bootstrap 95% CI for Jia — and the
  calibration repeated. The resulting interval is **11× (Delavenne), 183×
  (Lanoiselée) and 345× (Jia) wider** than the Monte Carlo sampling interval
  1.x reported as a "predicted margin of error". Delavenne's weight-on-clearance
  exponent (0.767, 29% RSE) is now a parameter rather than a literal, so its
  uncertainty propagates; because the covariate is centred on 70 kg it
  contributes nothing at the canonical cohort and up to ±25% on clearance at
  the ends of the weight grid.
- **Agreement statistics (EB-3):** Bland–Altman with confidence intervals on
  the bias and both limits; proportional-bias regression of difference on mean;
  relative errors alongside absolute; stratification by elapsed time, residual
  load, bypass duration and weight; coverage against a stated clinically
  relevant threshold (1,000 IU = one 10 mg protamine increment at 1 mg : 100 IU).
  R² is retained only as a descriptor of scatter.
- **Objective-function sensitivity (EB-3, R1 p9 L44).** *k* re-derived under
  RMSE, MAE, MAPE and an asymmetric loss penalising under-estimation 2:1, on the
  same cohort; plus the limits-of-agreement weight varied from 0 to 0.5.
  Largest movement 5.8% (Delavenne, asymmetric); ≤0.2% for Lanoiselée and Jia.
- **Coverage and transportability (EB-4).** All 2,304 grid nodes and the 32
  boundary corners; nine deliberately shifted simulated institutions.
- **Top-up grid (EB-5).** 18 dosing histories per model — 2,500/5,000/10,000 IU
  at 30/60/90/120 min plus repeated boluses — evaluated to 240 min, with worst
  case reported beside the mean. Prime timing is stated as a convention and its
  effect quantified: matching the reference convention raises *k* by 3.3–3.5%.
- **Correlated-input sensitivity (R2).** Gaussian copula over weight, time to
  bypass and time on bypass at correlations of 0.3 and 0.6.
- **Implementation verification (EB-6).** 25 automatic checks: each closed form
  is confirmed to solve its own differential equations numerically and to
  satisfy the identities they imply — A(0) = dose, ∫A·dt = dose·Vc/Cl, terminal
  slope = −β, dose linearity, prime timing.
- **Reproduction of a published simulation (EB-6).** Six landmark values from
  Delavenne Figure 3, worst deviation 3.0%. Derived PK constants (Vc, Vss, Cl,
  distribution and terminal half-lives, MRT) tabulated for every model.
- **`run_analysis.py`** regenerates every reported number from one seeded
  command into a dated directory with a manifest recording seed, commit,
  package versions and every analysis decision.
- **`make_supplementary_figures.py`** renders main-text Figure 2 and
  supplementary figures S1–S4 from the frozen run, as PDF and 600 dpi PNG.
  Figure 2 was previously drawn only inside the dashboard and never written to
  disk, so the figure the manuscript showed could not be traced to a run; it is
  now rebuilt from the same seeded test cohort as the agreement table and
  therefore cannot quote different numbers from the text. PDF timestamps are
  suppressed so re-running reproduces the figures byte-for-byte.
- **165 automated tests**, including an end-to-end execution of the dashboard
  against a recording Streamlit stub.
- **New modules:** `nomogram_core`, `calibration`, `agreement`,
  `parameter_spaces`, `topup`, `benchmarks`, `nomogram_render`.
- **`REPRODUCIBILITY.md`**, `LICENSE` (MIT), `CITATION.cff` and this changelog.
- **The lookup tables move to `data/k_tables/` and ship as CSV, not pickles.** `pickle.load` executes
  arbitrary code, so a pickled table in a public repository is a file every
  user of a clone or fork has to trust, for data that is 2,304 rows of five
  integers and three floats. Each table is now CSV with a commented JSON header
  carrying its provenance, readable in any spreadsheet and diffable in review.
  Floats are written with `repr`, which round-trips exactly in Python: the
  converted tables hold values bit-identical to the pickles they replace, and a
  test asserts the round trip. `k_table_io.py` is the single reader and writer,
  and resolves the directory from its own location, so the dashboard finds its
  tables whatever directory it is launched from.
- **Continuous integration** on every push: the suite on Python 3.10 and 3.11,
  every test file again in isolation, a check that the reported numbers are
  unchanged on current library releases, and a separate job that regenerates
  the archive byte-for-byte in the pinned environment. The reproducibility
  claim is now asserted rather than asserted about.
- **`check_frozen.py`** compares a run against the archive, to a relative
  tolerance by default and byte-for-byte with `--exact`. The distinction is
  real: byte identity holds only in the environment `requirements-frozen.txt`
  pins, while agreement of the numbers holds anywhere. Drift between the
  archive's environment and one a major release behind is 5e-14 at worst.
- **`requirements-frozen.txt`** pinning the exact package versions the archived
  run used, so the published numbers can be reproduced to the last digit.
  `requirements.txt` installs current releases and no longer pulls in Streamlit:
  the dashboard is optional and is not needed for any reported result.

### Documentation

- Parameter provenance is recorded per model in
  `nomogram_core.MODEL_PARAMETERS`: source, published values, IIV, %RSE,
  bootstrap interval where available, and notes on anything unverified.
- The pharmacodynamic layers neither implementation includes are recorded in
  `PD_PARAMETERS` for reference. The two PK/PD reference models disagree with
  each other by 90–125 s in predicted ACT at the same anti-factor Xa
  concentration, with maximal responses of 520 s and 836 s — which is why the
  pipeline stops at the central-compartment amount.
- The `k` search interval is stated as 0.001–0.03 min⁻¹ (apparent half-life
  23–693 min), and nodes where the optimisation reaches a bound are flagged.
- Regulatory language moderated (EB-7): the intended purpose is stated
  consistently as a research and educational instrument, and no assessment is
  offered of how any jurisdiction would classify bedside use.

---

## [1.0.0] — 2026

The version submitted with manuscript JCVA-D-26-01511. Preserved on the `main`
branch as of commit `def13e5`.

- Streamlit dashboard comparing six published models of unfractionated heparin
  disposition against a calibrated single-compartment approximation.
- Per-model lookup tables of the calibrated decay constant across a grid of
  institutional inputs.
- PyNomo-rendered printed nomogram, and a separate interactive nomogram.
- Monte Carlo credible bands for the three population pharmacokinetic models.

> Internal version strings in this release read `v1.2.0`; "1.0" is the release
> designation, not what the code reported.

Release 2.0.0 is archived at
doi:[10.5281/zenodo.22846043](https://doi.org/10.5281/zenodo.22846043).

[2.0.0]: https://github.com/vanderzwaag/nomogram/releases/tag/v2.0.0
[1.0.0]: https://github.com/vanderzwaag/nomogram/tree/main
