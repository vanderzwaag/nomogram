# Frozen analysis run

- Generated: 2026-09-12T16:57:32.359490+00:00
- Commit: `def13e5-dirty`  **working tree dirty -- commit before quoting these numbers**
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
| Jia | paediatric | 0.00716 | 0.00716 to 0.00717 | 97 |
| Lanoiselee | adult | 0.00703 | 0.00703 to 0.00703 | 99 |
| Meesters | adult | 0.00386 | 0.00384 to 0.00387 | 180 |
| PRODOSE | adult | 0.00550 | 0.00548 to 0.00551 | 126 |
| PRODOSE-2 | adult | 0.00419 | 0.00418 to 0.00421 | 165 |

The interval is a Monte Carlo sampling-precision interval. It does **not** quantify uncertainty in the published PK parameters, in the institutional input estimates, in model selection, or in an individual patient's prediction (EB-2).

## Agreement on an internal resample (EB-3)

| Model | Bias (IU) | 95% LoA (IU) | MAPE (%) | Max abs error (%) | Prop-bias slope | p | Within threshold (%) |
|---|---|---|---|---|---|---|---|
| Delavenne | -12.84 | -3714.2 to +3688.6 | 10.33 | 48.1 | -0.1093 | 2e-07 | 40.3 |
| Jia | +1.13 | -53.7 to +56.0 | 0.88 | 5.5 | -0.0024 | 0.007 | 100.0 |
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

## Two intervals on k, which must not be conflated (EB-2)

The sampling-precision interval repeats the simulation with fresh cohorts drawn from the same fixed, investigator-specified distributions, holding the published pharmacokinetic parameters at their point estimates. The parameter-uncertainty interval instead resamples those published parameters from their reported uncertainty. The first measures Monte Carlo noise; only the second speaks to how well the reference models themselves are known.

| Model | k | Sampling-precision interval | Parameter-uncertainty interval | Width ratio | Source |
|---|---|---|---|---|---|
| Delavenne | 0.01046 | 0.01035 to 0.01057 | 0.00828 to 0.01296 | 22x wider | iiv |
| Jia | 0.00716 | 0.00716 to 0.00717 | 0.00465 to 0.01021 | 379x wider | iiv |
| Lanoiselee | 0.00703 | 0.00703 to 0.00703 | 0.00514 to 0.00913 | 483x wider | iiv |
| Meesters | 0.00386 | 0.00384 to 0.00387 | not propagated | -- | Meesters is a closed-form expression published without parameter uncertainty; no interval can be propagated for it |
| PRODOSE | 0.00550 | 0.00548 to 0.00551 | not propagated | -- | PRODOSE is a closed-form expression published without parameter uncertainty; no interval can be propagated for it |
| PRODOSE-2 | 0.00419 | 0.00418 to 0.00421 | not propagated | -- | PRODOSE-2 is a closed-form expression published without parameter uncertainty; no interval can be propagated for it |

**Caveat.** No %RSE has yet been transcribed from the source publications, so interindividual variability was substituted. IIV describes spread between patients, not uncertainty in the published estimate, so the parameter-uncertainty interval above is an over-estimate and must be described as such until the published RSEs are supplied.

## Coverage of the full input range and its boundaries (EB-4)

| Model | Evaluation | Nodes | Mean abs error (%) | Max abs error (%) | Worst node |
|---|---|---|---|---|---|
| delavenne | full_range | 2304 | 24.35 | 67.4 | 600 IU/kg, 115 kg, prime 10000, 35+120 min |
| jia | full_range | 675 | 2.45 | 11.9 | 300 IU/kg, 3 kg, prime 2500, 25+30 min |
| lanoiselee | full_range | 2304 | 1.86 | 11.4 | 250 IU/kg, 40 kg, prime 10000, 35+30 min |
| meesters | full_range | 2304 | 3.90 | 15.6 | 600 IU/kg, 115 kg, prime 10000, 35+120 min |
| prodose | full_range | 2304 | 8.79 | 29.7 | 600 IU/kg, 115 kg, prime 10000, 35+120 min |
| prodose-2 | full_range | 2304 | 6.33 | 25.6 | 600 IU/kg, 115 kg, prime 10000, 35+120 min |
| delavenne | boundary | 32 | 32.87 | 67.4 | 600 IU/kg, 115 kg, prime 10000, 35+120 min |
| jia | boundary | 32 | 3.00 | 11.9 | 300 IU/kg, 3 kg, prime 2500, 25+30 min |
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
| jia | 18 | 1.40 | 2.7 | 458 | 3 x 10000 IU at 45/90/135 min |

## Transportability to shifted institutions (EB-4)

Mean absolute percentage error when the canonically calibrated k is applied unchanged to a shifted population. Adult models are not run in paediatric populations, or the paediatric model in adult ones, so those cells are marked n/a rather than pooled (EB-4).

| Institution | delavenne | jia | lanoiselee | meesters | prodose | prodose-2 |
|---|---|---|---|---|---|---|
| heavier_population | 12.34 | n/a | 0.47 | 1.36 | 1.34 | 2.19 |
| high_dose | 9.63 | n/a | 0.71 | 1.53 | 8.30 | 5.50 |
| infant | n/a | 0.91 | n/a | n/a | n/a | n/a |
| lighter_population | 14.51 | n/a | 0.55 | 1.41 | 1.44 | 2.51 |
| long_bypass | 30.94 | n/a | 0.70 | 5.31 | 5.08 | 5.58 |
| low_dose_no_prime | 9.98 | n/a | 1.72 | 2.41 | 12.14 | 7.38 |
| neonatal | n/a | 0.79 | n/a | n/a | n/a | n/a |
| older_child | n/a | 0.82 | n/a | n/a | n/a | n/a |
| short_bypass | 17.00 | n/a | 0.44 | 2.33 | 2.14 | 2.67 |
| slow_start | 15.77 | n/a | 2.51 | 3.23 | 3.75 | 3.57 |
| wide_case_mix | 21.92 | n/a | 1.24 | 3.54 | 3.55 | 4.05 |

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

- **EB-2** -- Transcribe the published %RSE for these parameters, or state that the parameter-uncertainty interval could not be propagated
- **EB-6** -- Supply the expected values for the per-source benchmark checks
- **EB-4** -- Confirm the paediatric parameter space for Jia
- **EB-1** -- State the assumption under which central-compartment amount maps to a protamine dose
