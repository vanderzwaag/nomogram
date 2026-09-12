# Frozen analysis run

- Generated: 2026-09-12T19:35:14.203768+00:00
- Commit: `e7a4fc6-dirty`  **working tree dirty -- commit before quoting these numbers**
- Seed: `20260912`
- Cohort: grid_snapped_70_15_60: 400 IU/kg, prime 5000 IU, IBW 70+/-10 kg, time to CPB 15+/-3.75 min, time on CPB 60+/-15 min [adult]
- Sign convention: Difference = reference model - nomogram (positive = nomogram under-estimates)
- Calibration endpoint: `reversal_endpoint`
- Prime timing: `lumped_t0`
- Clinical threshold: 1000 IU (10 mg protamine at 1 mg:100 IU) or 10%
- Software: python 3.11.15, numpy 2.4.6, scipy 1.17.1, pandas 3.0.5

## Calibrated decay constant, all six models (EB-8)

| Model | Population | k (/min) | MC sampling interval | Apparent t1/2 (min) |
|---|---|---|---|---|
| Delavenne | adult | 0.01046 | 0.01035 to 0.01057 | 66 |
| Jia | adult | 0.00715 | 0.00715 to 0.00716 | 97 |
| Lanoiselee | adult | 0.00703 | 0.00703 to 0.00703 | 99 |
| Meesters | adult | 0.00386 | 0.00384 to 0.00387 | 180 |
| PRODOSE | adult | 0.00550 | 0.00548 to 0.00551 | 126 |
| PRODOSE-2 | adult | 0.00419 | 0.00418 to 0.00421 | 165 |

The interval is a Monte Carlo sampling-precision interval. It does **not** quantify uncertainty in the published PK parameters, in the institutional input estimates, in model selection, or in an individual patient's prediction (EB-2).

## Agreement on an internal resample (EB-3)

| Model | Bias (IU) | 95% LoA (IU) | MAPE (%) | Max abs error (%) | Prop-bias slope | p | Within threshold (%) |
|---|---|---|---|---|---|---|---|
| Delavenne | -12.84 | -3714.2 to +3688.6 | 10.33 | 48.1 | -0.1093 | 2e-07 | 40.3 |
| Jia | +3.99 | -200.6 to +208.5 | 0.44 | 2.1 | +0.0008 | 0.43 | 100.0 |
| Lanoiselee | +4.80 | -176.4 to +186.0 | 0.38 | 2.0 | -0.0085 | 3e-21 | 100.0 |
| Meesters | +9.99 | -806.1 to +826.1 | 1.35 | 4.7 | -0.0706 | 5e-77 | 98.2 |
| PRODOSE | +9.59 | -696.6 to +715.8 | 1.32 | 4.8 | -0.0813 | 2.8e-147 | 99.3 |
| PRODOSE-2 | +15.79 | -953.0 to +984.6 | 1.65 | 7.2 | -0.1320 | 2.1e-246 | 96.1 |

The test cohort is a fresh draw from the same data-generating process, i.e. an internal resample, not an external validation cohort (EB-1).

## Robustness of k to the objective function (EB-3)

| Model | asymmetric | bland_altman | mae | mape | rmse | max deviation from primary |
|---|---|---|---|---|---|---|
| delavenne | 0.00985 | 0.01046 | 0.01034 | 0.01039 | 0.01028 | 5.8% |
| jia | 0.00714 | 0.00715 | 0.00716 | 0.00716 | 0.00716 | 0.2% |
| lanoiselee | 0.00702 | 0.00703 | 0.00704 | 0.00704 | 0.00704 | 0.2% |
| meesters | 0.00378 | 0.00386 | 0.00383 | 0.00382 | 0.00384 | 2.1% |
| prodose | 0.00543 | 0.00550 | 0.00547 | 0.00545 | 0.00548 | 1.4% |
| prodose-2 | 0.00412 | 0.00420 | 0.00418 | 0.00416 | 0.00418 | 2.0% |

## Calibration endpoint: single timepoint versus trajectory (EB-4)

| Model | reversal_endpoint | trajectory_cpb | trajectory_full |
|---|---|---|---|
| delavenne | 0.01049 | 0.01395 | 0.01624 |
| jia | 0.00715 | 0.00698 | 0.00835 |
| lanoiselee | 0.00703 | 0.00690 | 0.00828 |
| meesters | 0.00386 | 0.00442 | 0.00570 |
| prodose | 0.00550 | 0.00602 | 0.00741 |
| prodose-2 | 0.00419 | 0.00497 | 0.00645 |

The run used `reversal_endpoint`: One point per patient: residual heparin at the reversal timepoint t = time to CPB + time on CPB. This is what the submitted analysis calibrated against; with it, the manuscript may claim agreement in residual amount at reversal, NOT reproduction of entire trajectories.

## Prime-timing convention (EB-5)

| Model | cpb_onset | lumped_t0 | change (%) |
|---|---|---|---|
| delavenne | 0.01081 | 0.01044 | +3.5 |
| jia | 0.00739 | 0.00715 | +3.4 |
| lanoiselee | 0.00726 | 0.00703 | +3.4 |
| meesters | 0.00398 | 0.00385 | +3.3 |
| prodose | 0.00567 | 0.00549 | +3.3 |
| prodose-2 | 0.00432 | 0.00418 | +3.3 |

## Implementation verification (EB-6)

Each model is checked against the equations and parameters of its source publication: the closed-form solutions are confirmed to solve the corresponding differential equations numerically, and to satisfy the identities those equations imply (initial condition, area under the curve equal to dose x Vc / Cl, terminal slope equal to -beta, dose linearity, and the prime-timing convention). The derived pharmacokinetic constants each source's parameters imply are tabulated so a reader can compare them with the source directly. For Delavenne the implementation additionally reproduces the published Figure 3 simulation -- a 70 kg patient given either 350 IU/kg plus hourly 5,000 IU boluses, or 300 IU/kg followed by a 55 IU/kg/h infusion -- to within the precision with which that figure can be read (worst deviation 3%). That figure contains no observed data, so it is a pure model prediction and needs no participant-level data to reproduce. The Lanoiselee paper's corresponding diagnostic is a prediction-corrected visual predictive check, which cannot serve as a benchmark: its ordinate carries prediction-corrected observations rather than model predictions, and its bands can only be regenerated from the original dataset and its full design. No participant-level data from any source study is held by the authors, so the models are not re-fitted and their predictions are not compared with observed measurements; no such external or predictive validation is claimed. What is demonstrated is that the implementation faithfully reproduces the published model, not that the published model is correct.

Derived constants at 70 kg, for comparison against each source publication:

| Model | Vc (L) | Vss (L) | Cl (L/h) | Distribution t1/2 (min) | Terminal t1/2 (min) | MRT (min) |
|---|---|---|---|---|---|---|
| Delavenne | 3.10 | 5.33 | 0.841 | 11.2 | 272 | 380 |
| Jia | 3.04 | 11.05 | 1.180 | 93.0 | 2245 | 562 |
| Lanoiselee | 4.01 | 5.47 | 1.500 | 84.3 | 278 | 219 |

| Model | Fast pool | Fast t1/2 (min) | Slow t1/2 (min) | |
|---|---|---|---|---|
| Meesters | 10% | 10.0 | 250 | fixed |
| PRODOSE | 10% | 10.0 | 155 | quoted at 400 IU/kg; depends on dose per kg |
| PRODOSE-2 | 10% | 2.4 | 171 | quoted at 400 IU/kg; depends on dose per kg |

25 automatic implementation checks, all passing.

Reproduction of a simulation published in a source (Delavenne Figure 3, 70 kg patient). The figure contains no observed data, so it is a pure model prediction and needs no participant-level data to reproduce; the tolerance reflects the precision with which the figure can be read.

| Published value | Expected (read off) | Reproduced | Deviation |
|---|---|---|---|
| Panel C, peak anti-Xa immediately after the 350 IU/kg bolus | 7.80 anti-Xa IU/mL | 7.90 | 1.3% |
| Panel C, trough before the first hourly 5,000 IU top-up | 3.70 anti-Xa IU/mL | 3.79 | 2.4% |
| Panel C, peak immediately after the first hourly top-up | 5.20 anti-Xa IU/mL | 5.36 | 3.0% |
| Panel A, peak anti-Xa immediately after the 300 IU/kg bolus | 6.80 anti-Xa IU/mL | 6.77 | 0.4% |
| Panel A, anti-Xa plateau under the 55 IU/kg/h infusion at 2 h | 4.00 anti-Xa IU/mL | 4.04 | 0.9% |
| Panel A, anti-Xa at the end of the 6 h infusion | 4.30 anti-Xa IU/mL | 4.28 | 0.4% |

The Lanoiselee paper's corresponding diagnostic is a prediction-corrected visual predictive check, which cannot serve as a benchmark: its ordinate carries prediction-corrected observations rather than model predictions, and its bands can only be regenerated from the original dataset and its full design.

## Two intervals on k, which must not be conflated (EB-2)

The sampling-precision interval repeats the simulation with fresh cohorts drawn from the same fixed, investigator-specified distributions, holding the published pharmacokinetic parameters at their point estimates. The parameter-uncertainty interval instead resamples those published parameters from their reported uncertainty. The first measures Monte Carlo noise; only the second speaks to how well the reference models themselves are known.

| Model | k | Sampling-precision interval | Parameter-uncertainty interval | Width ratio | Source |
|---|---|---|---|---|---|
| Delavenne | 0.01046 | 0.01035 to 0.01057 | 0.00923 to 0.01165 | 11x wider | published %RSE |
| Jia | 0.00715 | 0.00715 to 0.00716 | 0.00549 to 0.00883 | 345x wider | published bootstrap 95% CI |
| Lanoiselee | 0.00703 | 0.00703 to 0.00703 | 0.00626 to 0.00777 | 183x wider | published %RSE |
| Meesters | 0.00386 | 0.00384 to 0.00387 | not propagated | -- | closed-form expression; none published |
| PRODOSE | 0.00550 | 0.00548 to 0.00551 | not propagated | -- | closed-form expression; none published |
| PRODOSE-2 | 0.00419 | 0.00418 to 0.00421 | not propagated | -- | closed-form expression; none published |

Every interval above is propagated from uncertainty the source publications actually report: the asymptotic %RSE for Lanoiselee and Delavenne, and the non-parametric bootstrap interval for Jia, which publishes one. For Delavenne the draw includes the weight exponent on clearance (0.767, 29% RSE); because that covariate is centred on 70 kg its contribution is zero at the canonical cohort's IBW and reaches about 25% on clearance at the ends of the weight grid, so it moves the boundary and transportability results rather than this table.

## Coverage of the full input range and its boundaries (EB-4)

| Model | Evaluation | Nodes | Mean abs error (%) | Max abs error (%) | Worst node |
|---|---|---|---|---|---|
| delavenne | full_range | 2304 | 24.35 | 67.4 | 600 IU/kg, 115 kg, prime 10000, 35+120 min |
| jia | full_range | 2304 | 1.99 | 11.5 | 250 IU/kg, 40 kg, prime 10000, 35+30 min |
| lanoiselee | full_range | 2304 | 1.86 | 11.4 | 250 IU/kg, 40 kg, prime 10000, 35+30 min |
| meesters | full_range | 2304 | 3.90 | 15.6 | 600 IU/kg, 115 kg, prime 10000, 35+120 min |
| prodose | full_range | 2304 | 8.79 | 29.7 | 600 IU/kg, 115 kg, prime 10000, 35+120 min |
| prodose-2 | full_range | 2304 | 6.33 | 25.6 | 600 IU/kg, 115 kg, prime 10000, 35+120 min |
| delavenne | boundary | 32 | 32.87 | 67.4 | 600 IU/kg, 115 kg, prime 10000, 35+120 min |
| jia | boundary | 32 | 2.75 | 11.5 | 250 IU/kg, 40 kg, prime 10000, 35+30 min |
| lanoiselee | boundary | 32 | 2.30 | 11.4 | 250 IU/kg, 40 kg, prime 10000, 35+30 min |
| meesters | boundary | 32 | 5.27 | 15.6 | 600 IU/kg, 115 kg, prime 10000, 35+120 min |
| prodose | boundary | 32 | 14.63 | 29.7 | 600 IU/kg, 115 kg, prime 10000, 35+120 min |
| prodose-2 | boundary | 32 | 9.43 | 25.6 | 600 IU/kg, 115 kg, prime 10000, 35+120 min |

## Supplemental-bolus grid, worst case and mean (EB-5)

| Model | Scenarios | Mean abs error (%) | Worst abs error (%) | Worst abs error (IU) | Worst-case scenario |
|---|---|---|---|---|---|
| delavenne | 18 | 35.27 | 72.1 | 11619 | 3 x 10000 IU at 45/90/135 min |
| prodose-2 | 18 | 6.56 | 16.2 | 3279 | 3 x 10000 IU at 45/90/135 min |
| meesters | 18 | 6.51 | 16.0 | 3482 | 3 x 10000 IU at 45/90/135 min |
| prodose | 18 | 5.91 | 15.2 | 2217 | 3 x 10000 IU at 45/90/135 min |
| lanoiselee | 18 | 1.33 | 6.6 | 524 | 10000 IU at 30 min |
| jia | 18 | 1.56 | 3.0 | 495 | 3 x 10000 IU at 45/90/135 min |

## Transportability to shifted institutions (EB-4)

Mean absolute percentage error when the canonically calibrated k is applied unchanged to a shifted population. Adult models are not run All six models are adult and are evaluated in the same populations; the small_bodied_adults row matches the body size the Jia model was derived in (EB-4, R1 p11 L46).

| Institution | delavenne | jia | lanoiselee | meesters | prodose | prodose-2 |
|---|---|---|---|---|---|---|
| heavier_population | 12.34 | 0.53 | 0.47 | 1.36 | 1.34 | 2.19 |
| high_dose | 9.63 | 0.73 | 0.71 | 1.53 | 8.30 | 5.50 |
| lighter_population | 14.51 | 0.57 | 0.55 | 1.41 | 1.44 | 2.51 |
| long_bypass | 30.94 | 1.08 | 0.70 | 5.31 | 5.08 | 5.58 |
| low_dose_no_prime | 9.98 | 1.81 | 1.72 | 2.41 | 12.14 | 7.38 |
| short_bypass | 17.00 | 0.60 | 0.44 | 2.33 | 2.14 | 2.67 |
| slow_start | 15.77 | 2.19 | 2.51 | 3.23 | 3.75 | 3.57 |
| small_bodied_adults | 12.70 | 0.52 | 0.49 | 1.38 | 1.39 | 2.21 |
| wide_case_mix | 21.92 | 1.10 | 1.24 | 3.54 | 3.55 | 4.05 |

## Most influential inputs, +/-20% (EB-8)

| Model | ibw_mean | ibw_sd | t_on_mean | t_on_sd | t_to_mean | t_to_sd |
|---|---|---|---|---|---|---|
| delavenne | 15.16 | 1.64 | 14.45 | 1.54 | 4.43 | 1.61 |
| jia | 0.72 | 0.09 | 0.70 | 0.11 | 0.57 | 0.08 |
| lanoiselee | 0.66 | 0.06 | 0.33 | 0.08 | 0.68 | 0.06 |
| meesters | 1.65 | 0.46 | 5.39 | 0.57 | 1.99 | 0.64 |
| prodose | 1.29 | 0.31 | 3.54 | 0.39 | 1.58 | 0.43 |
| prodose-2 | 5.36 | 0.38 | 5.32 | 0.62 | 1.87 | 0.61 |

## Worked example (EB-8)

lanoiselee, 70 kg, 400 IU/kg bolus + 5000 IU prime = 33,000 IU total load, k = 0.00703/min:

| Elapsed (min) | Nomogram residual (IU) | Reference residual (IU) | Protamine at 1 mg:100 IU (mg) |
|---|---|---|---|
| 75 | 19,477 | 19,464 | 195 |
| 60 | 21,643 | 21,663 | 216 |
| 90 | 17,528 | 17,508 | 175 |
| 120 | 14,195 | 14,220 | 142 |

Reading the residual load off the nomogram requires no calculation; converting it into a protamine dose requires applying the institutional ratio (EB-8).

## Outstanding author decisions

- **EB-4 / R1 p11 L46** -- Correct the manuscript's description of the Jia model
- **EB-1** -- State the assumption under which central-compartment amount maps to a protamine dose
