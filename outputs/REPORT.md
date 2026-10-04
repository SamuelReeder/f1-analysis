# F1 driver skill vs car performance: qualifying-pace ratings

Data through **Azerbaijan Grand Prix 2026** (event 2026-15), window 2010-2026, 15,656 qualifying lap times. Model `quali-v1`, fit `20261002T002001Z`.

Ratings are **seconds per 90-second lap relative to the average driver / car entered at the latest event**; positive = faster. Scope: one-lap qualifying pace only.

**Headline driver rating: pace in the current car** (what teammate comparisons measure directly). **Portable skill** (pace expected to carry over to another team) is **experimental**: it fails 2 of the gates below (synthetic_current_grid, sensitivity).

## Acceptance gates

| gate | criterion | result |
|---|---|---|
| convergence | R-hat < 1.05 for every rating (portable skill, team-specific effect, car) and hyperparameter; no more than 1 divergence per 1,000 draws | PASS |
| synthetic_calibration | clean synthetic data, independent truths: mean 90% interval coverage of true driver and car values 85-97% | PASS |
| synthetic_current_grid | clean synthetic data, independent truths: portable skill's current-grid rank correlation >= 0.8 and 90% coverage >= 80% (means over truths) | FAIL |
| synthetic_current_grid_in_team | the same for pace in the current car | PASS |
| forecast_vs_baselines | teammate-gap forecasts beat the static two-way model, naive and zero baselines on RMSE, pooled over all cutoffs | PASS |
| forecast_calibration | session-level 90% forecast intervals cover 85-97% | PASS |
| new_pairings | new teammate pairings: model RMSE below the naive and zero baselines | PASS |
| sensitivity | portable skill: current-grid Spearman >= 0.9 against every sensitivity variant | FAIL |
| sensitivity_in_team | pace in the current car: current-grid Spearman >= 0.9 against every sensitivity variant | PASS |

Changes to the gates: 2026-09-28, before the refit whose results they judge: the two synthetic gates use 8 independent clean truths (before: one truth shared by all scenarios). Two gates were added for the headline rating, pace in the current car (`synthetic_current_grid_in_team`, `sensitivity_in_team`). When they were added, its single-truth rank correlation (0.91) was known; its sensitivity results were not. The convergence gate now also covers the team-specific effect, which is part of the headline rating.

## Drivers in their current car (headline, latest event)

Pace relative to the field's drivers in the car each driver has now: portable skill plus the driver's team-specific effect. This is what teammate comparisons measure directly. In simulation on the real network (8 independent truths) the current-grid order is recovered with rank correlation 0.83 (0.72–0.93) (mean, range over truths). Read ranks as ranges.

| in_team_rank | name | team | in_car | lo90 | hi90 | ranks | p_fastest |
|---|---|---|---|---|---|---|---|
| 1 | Max Verstappen | Red Bull | 0.276 | 0.117 | 0.431 | 1–6 | 0.543 |
| 2 | Isack Hadjar | Red Bull | 0.168 | -0.003 | 0.335 | 2–12 | 0.033 |
| 3 | Carlos Sainz | Williams | 0.135 | -0.057 | 0.324 | 1–14 | 0.106 |
| 4 | Lando Norris | McLaren | 0.116 | -0.096 | 0.323 | 1–16 | 0.099 |
| 5 | Charles Leclerc | Ferrari | 0.094 | -0.099 | 0.278 | 2–16 | 0.049 |
| 6 | Andrea Kimi Antonelli | Mercedes | 0.052 | -0.135 | 0.246 | 2–18 | 0.032 |
| 7 | Gabriel Bortoleto | Audi | 0.041 | -0.148 | 0.236 | 2–18 | 0.029 |
| 8 | Oliver Bearman | Haas F1 Team | 0.037 | -0.167 | 0.242 | 2–19 | 0.028 |
| 9 | Oscar Piastri | McLaren | 0.035 | -0.184 | 0.240 | 3–20 | 0.005 |
| 10 | Lewis Hamilton | Ferrari | 0.024 | -0.168 | 0.212 | 3–19 | 0.003 |
| 11 | Arvid Lindblad | RB F1 Team | 0.021 | -0.142 | 0.187 | 3–18 | 0.006 |
| 12 | Fernando Alonso | Aston Martin | 0.020 | -0.193 | 0.228 | 2–20 | 0.029 |
| 13 | Pierre Gasly | Alpine F1 Team | -0.004 | -0.177 | 0.168 | 4–19 | 0.007 |
| 14 | George Russell | Mercedes | -0.020 | -0.201 | 0.172 | 4–20 | 0.002 |
| 15 | Alexander Albon | Williams | -0.042 | -0.235 | 0.143 | 5–21 | 0.000 |
| 16 | Valtteri Bottas | Cadillac F1 Team | -0.046 | -0.293 | 0.195 | 3–21 | 0.015 |
| 17 | Sergio Pérez | Cadillac F1 Team | -0.048 | -0.291 | 0.195 | 3–21 | 0.012 |
| 18 | Liam Lawson | RB F1 Team | -0.092 | -0.245 | 0.064 | 8–21 | 0.000 |
| 19 | Nico Hülkenberg | Audi | -0.111 | -0.310 | 0.090 | 7–22 | 0.000 |
| 20 | Esteban Ocon | Haas F1 Team | -0.179 | -0.381 | 0.032 | 10–22 | 0.000 |
| 21 | Franco Colapinto | Alpine F1 Team | -0.185 | -0.358 | -0.009 | 12–22 | 0.000 |
| 22 | Lance Stroll | Aston Martin | -0.282 | -0.497 | -0.079 | 15–22 | 0.000 |

## Portable skill (experimental)

`skill` is the part of a driver's pace expected to carry over to another team: pace in the current car minus `team_effect`, the driver's persistent team-specific effect. **Experimental:** in simulation the current-grid order of portable skill is recovered with rank correlation 0.65 (0.53–0.84) (mean, range over 8 truths), and it moves with structural choices (see Sensitivity). The team-specific effect is not established to be car-handling compatibility: team support, role, adaptation or selection would look the same.

| name | team | skill | lo90 | hi90 | ranks | p_fastest | team_effect | events | teams | teammates | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Carlos Sainz | Williams | 0.161 | -0.030 | 0.357 | 1–14 | 0.231 | -0.028 | 247 | 5 | 7 |  |
| Max Verstappen | Red Bull | 0.160 | -0.052 | 0.374 | 1–15 | 0.251 | 0.113 | 248 | 2 | 8 | sensitive to model choice |
| Isack Hadjar | Red Bull | 0.050 | -0.103 | 0.218 | 2–18 | 0.025 | 0.113 | 36 | 2 | 3 | sensitive to model choice |
| Charles Leclerc | Ferrari | 0.050 | -0.177 | 0.277 | 1–20 | 0.072 | 0.040 | 188 | 2 | 5 |  |
| Andrea Kimi Antonelli | Mercedes | 0.045 | -0.123 | 0.222 | 2–19 | 0.031 | 0.006 | 39 | 1 | 1 |  |
| Lando Norris | McLaren | 0.044 | -0.187 | 0.276 | 1–21 | 0.072 | 0.070 | 167 | 1 | 3 | sensitive to model choice |
| Valtteri Bottas | Cadillac F1 Team | 0.041 | -0.177 | 0.257 | 1–20 | 0.061 | -0.083 | 262 | 4 | 6 | sensitive to model choice |
| Fernando Alonso | Aston Martin | 0.030 | -0.194 | 0.258 | 1–21 | 0.065 | -0.012 | 303 | 4 | 6 | sensitive to model choice |
| Lewis Hamilton | Ferrari | 0.017 | -0.189 | 0.215 | 2–21 | 0.026 | 0.005 | 343 | 3 | 5 |  |
| Pierre Gasly | Alpine F1 Team | 0.015 | -0.187 | 0.227 | 2–20 | 0.038 | -0.018 | 193 | 3 | 8 | sensitive to model choice |
| George Russell | Mercedes | 0.003 | -0.204 | 0.211 | 2–21 | 0.029 | -0.023 | 167 | 2 | 5 | sensitive to model choice |
| Oliver Bearman | Haas F1 Team | 0.002 | -0.176 | 0.182 | 3–20 | 0.017 | 0.035 | 42 | 1 | 3 |  |
| Gabriel Bortoleto | Audi | -0.002 | -0.165 | 0.178 | 3–20 | 0.015 | 0.038 | 39 | 1 | 1 |  |
| Oscar Piastri | McLaren | -0.012 | -0.197 | 0.182 | 3–21 | 0.018 | 0.041 | 85 | 1 | 1 | sensitive to model choice |
| Sergio Pérez | Cadillac F1 Team | -0.019 | -0.225 | 0.191 | 2–21 | 0.020 | -0.030 | 298 | 5 | 7 |  |
| Franco Colapinto | Alpine F1 Team | -0.040 | -0.213 | 0.126 | 4–21 | 0.005 | -0.142 | 42 | 2 | 2 | sensitive to model choice |
| Liam Lawson | RB F1 Team | -0.054 | -0.225 | 0.111 | 5–21 | 0.003 | -0.036 | 50 | 2 | 4 | sensitive to model choice |
| Alexander Albon | Williams | -0.066 | -0.285 | 0.151 | 4–22 | 0.007 | 0.021 | 144 | 3 | 6 |  |
| Arvid Lindblad | RB F1 Team | -0.073 | -0.215 | 0.100 | 6–21 | 0.003 | 0.089 | 15 | 1 | 2 | sensitive to model choice |
| Nico Hülkenberg | Audi | -0.084 | -0.281 | 0.117 | 5–22 | 0.005 | -0.026 | 269 | 5 | 11 |  |
| Esteban Ocon | Haas F1 Team | -0.092 | -0.293 | 0.110 | 5–22 | 0.005 | -0.083 | 195 | 4 | 6 | sensitive to model choice |
| Lance Stroll | Aston Martin | -0.194 | -0.459 | 0.053 | 8–22 | 0.003 | -0.086 | 207 | 2 | 7 | sensitive to model choice |

`flag`: rating moves by more than 0.10 s, or rank by 8+ places, across the sensitivity variants below. (The rule set before the results, 0.05 s or 4 places, flagged every driver because it is smaller than the posterior uncertainty, so it was loosened.) `events`: events entered. `teams`: team lineages raced (5+ events). These counts are context, not a precision measure; use the interval.

## Car packages (latest event, track-neutral)

`track_profile` > 0: relatively stronger on high-speed circuits; < 0: on slow/street circuits.

| constructor | rating | lo90 | hi90 | gap_to_fastest | ranks | track_profile |
|---|---|---|---|---|---|---|
| Mercedes | 0.909 | 0.664 | 1.150 | 0.000 | 1–3 | 0.015 |
| Ferrari | 0.722 | 0.484 | 0.960 | -0.207 | 1–4 | -0.021 |
| McLaren | 0.692 | 0.435 | 0.953 | -0.239 | 1–4 | -0.017 |
| Red Bull | 0.570 | 0.354 | 0.782 | -0.376 | 2–5 | -0.038 |
| Alpine F1 Team | 0.252 | 0.024 | 0.487 | -0.686 | 4–7 | 0.040 |
| RB F1 Team | 0.104 | -0.125 | 0.326 | -0.836 | 5–8 | 0.103 |
| Audi | 0.050 | -0.218 | 0.307 | -0.893 | 5–8 | 0.048 |
| Haas F1 Team | -0.185 | -0.441 | 0.080 | -1.124 | 6–9 | 0.013 |
| Williams | -0.533 | -0.781 | -0.266 | -1.474 | 8–9 | -0.022 |
| Aston Martin | -1.246 | -1.626 | -0.900 | -2.191 | 10–11 | -0.155 |
| Cadillac F1 Team | -1.334 | -1.641 | -1.036 | -2.281 | 10–11 | 0.032 |

## Data

- **Entered drivers without a valid lap** (145 of 7,246 driver-events: no time set, or only laps more than 5% off the segment median) are kept. They have no observations, their rating is carried forward by the skill walk, and they count in the field that ratings are relative to.
- **Times missing from Jolpica** (2025-06: every qualifying time blank) are rebuilt from FastF1 lap timing (best non-deleted lap per driver and segment, `extract/quali_fill.py`). On the other events of the same season the method matches Jolpica exactly for 1,014 of 1,015 times. FastF1 also has 8 times that Jolpica does not: laps deleted by race control that FastF1 did not flag, or times removed after the session, which lap timing cannot see. For the filled event, no race-control lap deletion names a lap that was kept.

## Validation

### Forecasting (22 leave-future-out cutoffs, 2013–2026)

Each cutoff is fitted only on data up to that point (including the circuit factor) and forecasts the next season (end-of-season cutoffs) or the rest of the season (after round 8). Established drivers only (≥10 prior events). RMSE and CRPS are in seconds.

| target | model | static two-way | naive raw gaps | zero |
|---|---|---|---|---|
| teammate gap, per segment | 0.547 | 0.560 | 0.566 | 0.577 |
| teammate gap, per pairing (all, n=188) | 0.192 | 0.234 | 0.260 | 0.265 |
| teammate gap, per pairing (continuing, n=156) | 0.173 | 0.180 | 0.196 | 0.264 |
| teammate gap, per pairing (new, n=32) | 0.265 | 0.406 | 0.458 | 0.266 |

Session-level calibration of teammate-gap forecasts: 50% intervals cover 57.1%, 90% intervals cover 94.2%; CRPS 0.264s. Pairing-level 90% intervals cover 84.6%.

Full relative pace (car + driver) within each forecast segment:

| forecaster | rmse_s | spearman |
|---|---|---|
| model | 0.539 | 0.671 |
| static two-way | 0.559 | 0.648 |
| recent form (last 5 events) | 0.562 | 0.628 |

### Synthetic recovery on the real F1 network

Truth drawn from the model's generative process on the real entries, teams and segments (including the drivers entered without a valid lap). Coverage = share of true values inside the 90% interval.

**Independent clean truths (8).** Each has its own truth, its own noise and its own hyperparameters: a different posterior draw from the main fit, so settings such as the size of team-specific effects vary between truths. The gates use the means.

| truth | sd_team_effect | skill_cov90 | car_cov90 | grid_spearman | grid_cov90 | grid_in_team_spearman | grid_in_team_cov90 | cross_team_cov90 | divergences |
|---|---|---|---|---|---|---|---|---|---|
| seed1 | 0.163 | 0.924 | 0.908 | 0.571 | 0.909 | 0.753 | 0.818 | 0.959 | 0 |
| seed2 | 0.136 | 0.864 | 0.906 | 0.835 | 0.955 | 0.919 | 0.909 | 0.955 | 0 |
| seed3 | 0.160 | 0.888 | 0.896 | 0.719 | 0.909 | 0.935 | 0.955 | 0.936 | 0 |
| seed4 | 0.169 | 0.867 | 0.875 | 0.616 | 0.955 | 0.850 | 0.864 | 0.918 | 0 |
| seed5 | 0.216 | 0.892 | 0.869 | 0.527 | 0.818 | 0.715 | 0.773 | 0.818 | 0 |
| seed6 | 0.145 | 0.905 | 0.916 | 0.753 | 0.955 | 0.860 | 1.000 | 0.959 | 0 |
| seed7 | 0.158 | 0.916 | 0.903 | 0.551 | 0.955 | 0.833 | 0.955 | 0.900 | 0 |
| seed8 | 0.183 | 0.861 | 0.885 | 0.619 | 0.909 | 0.815 | 0.909 | 0.891 | 0 |

Mean (range): skill_cov90 0.89 (0.86–0.92); car_cov90 0.89 (0.87–0.92); grid_spearman 0.65 (0.53–0.84); grid_cov90 0.92 (0.82–0.95); grid_in_team_spearman 0.83 (0.72–0.93); grid_in_team_cov90 0.90 (0.77–1.00). Eight truths show the spread between truths; they do not pin down coverage to better than a few percentage points.

**Misspecification scenarios (one shared truth).** These add effects the model does not contain. All share one simulated truth (posterior medians), so they compare scenarios on equal terms but each is a single draw.

| scenario | skill_corr | skill_rmse_s | skill_cov90 | car_corr | car_cov90 | grid_spearman | grid_cov90 | cross_team_cov90 | grid_in_team_spearman | car_latent_share_true/est |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | 0.814 | 0.099 | 0.834 | 0.994 | 0.897 | 0.846 | 0.818 | 0.818 | 0.897 | 0.97/0.98 |
| compat | 0.794 | 0.102 | 0.878 | 0.992 | 0.891 | 0.783 | 0.909 | 0.895 | 0.824 | 0.97/0.98 |
| upgrades | 0.792 | 0.102 | 0.781 | 0.994 | 0.917 | 0.814 | 0.864 | 0.818 | 0.953 | 0.97/0.98 |
| form | 0.752 | 0.110 | 0.924 | 0.993 | 0.919 | 0.767 | 0.955 | 0.959 | 0.892 | 0.97/0.96 |
| transfer_luck | 0.820 | 0.093 | 0.903 | 0.993 | 0.890 | 0.869 | 0.909 | 0.918 | 0.914 | 0.97/0.97 |

`car_latent_share`: within each season, the variance of the synthetic car states divided by the variance of car states plus portable skill states, averaged over seasons (true vs estimated). It covers only those two latent components (not the team-specific effect, weekend effects or noise) and describes the simulated truth, not a share of observed qualifying variation.

### The team-specific effect

**Placebo test.** Team-specific effect SD 0.164% (90% interval 0.119–0.204) vs placebo SD 0.018% (0.002–0.049) when a driver's stint in one team is split in half. P(placebo > team effect) = 0.00. A large team-specific SD with a small placebo SD means relative performance shifts when a driver changes team, much more than with time spent in the same team. The test cannot say why (car handling, team support, role, adaptation or selection).

**What one effect covers.** The main model gives one effect per driver and team lineage, so separate spells with the same lineage share it (9 cases, e.g. gasly (faenza: 2017–2018, 2019–2022); hulkenberg (hinwil: 2013, 2025–2026); hulkenberg (silverstone: 2012, 2014–2016, 2020–2022)). Two variants test this choice: one effect per spell (a return after 3+ events with other teams starts a new one; a one-off stand-in drive does not) and one per regulation era (2014, 2017, 2022 and 2026 start new ones).

Current grid under each variant vs the main model (all data):

| variant | in_team_spearman | in_team_max_shift_s | portable_spearman | portable_max_shift_s |
|---|---|---|---|---|
| per spell | 0.964 | 0.043 | 0.960 | 0.043 |
| per era | 0.964 | 0.069 | 0.853 | 0.126 |

Forecasts of teammate gaps per pairing, variant vs main model on identical targets at 6 cutoffs (end2016, end2019, end2021, end2024, end2025, mid2025). `mse_diff_s2`: mean difference in squared error (variant minus main, s²; negative = variant better), with a bootstrap 95% interval over pairings (pairings at one cutoff share a fit, so it is somewhat too narrow).

| comparison | n | rmse_main_s | rmse_variant_s | mse_diff_s2 | 95% interval | in90_main | in90_variant |
|---|---|---|---|---|---|---|---|
| per spell: all pairings | 49 | 0.221 | 0.216 | -0.0024 | -0.0073 to +0.0005 | 0.816 | 0.837 |
| per spell: first season after a reset | 28 | 0.262 | 0.254 | -0.0038 | -0.0116 to +0.0013 | 0.714 | 0.750 |
| per era: all pairings | 49 | 0.221 | 0.214 | -0.0034 | -0.0124 to +0.0042 | 0.816 | 0.939 |
| per era: first season after a reset | 28 | 0.262 | 0.252 | -0.0050 | -0.0211 to +0.0080 | 0.714 | 0.929 |

Neither variant forecasts clearly better than one effect per lineage.

### Sprint qualifying (pre-registered test)

Sprint qualifying added to the training data up to each cutoff, against the main model on the same forecast targets at 7 of 7 cutoffs (`docs/sprint_qualifying.md`).

| comparison | n | rmse_main_s | rmse_variant_s | mse_diff_s2 | 95% interval | in90_main | in90_variant |
|---|---|---|---|---|---|---|---|
| all pairings | 62 | 0.160 | 0.162 | +0.0006 | -0.0011 to +0.0022 | 0.903 | 0.919 |

| fit | rmse_s | crps_s | cov90 |
|---|---|---|---|
| main | 0.463 | 0.235 | 0.950 |
| variant | 0.463 | 0.236 | 0.952 |

Gate: all fits converged: yes; pairing mse interval below zero: no; lower rmse at 5 of 7 cutoffs: no; session coverage 85 to 97: yes; session crps no higher: no. Not passed: the published model is unchanged.

### Sensitivity of the current driver ranking

| variant | in_team_spearman | in_team_max_shift_s | in_team_max_rank_shift | portable_spearman | portable_max_shift_s | portable_max_rank_shift |
|---|---|---|---|---|---|---|
| sens_start2006 | 0.975 | 0.060 | 4 | 0.877 | 0.106 | 7 |
| sens_notrack | 0.994 | 0.013 | 2 | 0.988 | 0.012 | 3 |
| sens_gausscar | 0.993 | 0.016 | 2 | 0.991 | 0.010 | 3 |
| sens_notransient | 0.997 | 0.016 | 1 | 0.990 | 0.022 | 2 |
| sens_noform | 0.958 | 0.061 | 6 | 0.918 | 0.079 | 7 |
| sens_nocompat | 0.957 | 0.081 | 5 | 0.809 | 0.152 | 9 |
| sens_splitt | 0.989 | 0.023 | 2 | 0.990 | 0.022 | 2 |
| sens_drift_x2 | 0.962 | 0.092 | 5 | 0.918 | 0.142 | 6 |
| sens_drift_half | 0.979 | 0.044 | 4 | 0.919 | 0.082 | 5 |
| sens_team_spell | 0.964 | 0.043 | 6 | 0.960 | 0.043 | 4 |
| sens_team_era | 0.964 | 0.069 | 4 | 0.853 | 0.126 | 8 |

Per-driver pace in the current car (s) under each variant:

| driver_id | main | sens_start2006 | sens_notrack | sens_gausscar | sens_notransient | sens_noform | sens_nocompat | sens_splitt | sens_drift_x2 | sens_drift_half | sens_team_spell | sens_team_era | spread_s | rank_range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| max_verstappen | 0.276 | 0.296 | 0.274 | 0.276 | 0.280 | 0.258 | 0.274 | 0.274 | 0.252 | 0.265 | 0.300 | 0.235 | 0.065 | 0.000 |
| hadjar | 0.168 | 0.181 | 0.166 | 0.170 | 0.165 | 0.182 | 0.138 | 0.162 | 0.166 | 0.153 | 0.189 | 0.126 | 0.063 | 2.000 |
| sainz | 0.135 | 0.133 | 0.134 | 0.141 | 0.144 | 0.141 | 0.133 | 0.132 | 0.154 | 0.129 | 0.142 | 0.154 | 0.025 | 2.000 |
| norris | 0.116 | 0.144 | 0.126 | 0.123 | 0.112 | 0.141 | 0.196 | 0.120 | 0.160 | 0.116 | 0.129 | 0.137 | 0.084 | 2.000 |
| leclerc | 0.094 | 0.111 | 0.087 | 0.081 | 0.104 | 0.061 | 0.056 | 0.087 | 0.079 | 0.105 | 0.087 | 0.108 | 0.054 | 2.000 |
| antonelli | 0.052 | 0.075 | 0.055 | 0.055 | 0.041 | 0.098 | 0.088 | 0.054 | 0.119 | 0.016 | 0.073 | 0.089 | 0.104 | 5.000 |
| bortoleto | 0.041 | 0.023 | 0.041 | 0.038 | 0.036 | 0.045 | 0.042 | 0.032 | 0.034 | 0.059 | -0.000 | 0.032 | 0.059 | 7.000 |
| bearman | 0.037 | 0.031 | 0.035 | 0.042 | 0.035 | 0.046 | 0.016 | 0.035 | 0.040 | 0.043 | 0.036 | 0.046 | 0.029 | 3.000 |
| piastri | 0.035 | 0.059 | 0.044 | 0.041 | 0.034 | 0.043 | 0.111 | 0.040 | 0.061 | 0.043 | 0.045 | 0.061 | 0.076 | 5.000 |
| hamilton | 0.024 | 0.037 | 0.012 | 0.008 | 0.028 | 0.022 | -0.002 | 0.013 | 0.035 | 0.013 | 0.020 | 0.054 | 0.056 | 4.000 |
| arvid_lindblad | 0.021 | 0.042 | 0.024 | 0.025 | 0.010 | 0.028 | 0.010 | 0.009 | 0.008 | 0.020 | -0.003 | -0.015 | 0.057 | 6.000 |
| alonso | 0.020 | -0.040 | 0.010 | 0.015 | 0.020 | -0.041 | -0.024 | 0.019 | -0.032 | 0.010 | 0.007 | -0.009 | 0.062 | 4.000 |
| gasly | -0.004 | 0.006 | -0.002 | -0.002 | -0.017 | 0.052 | 0.045 | -0.001 | 0.050 | -0.015 | 0.004 | 0.043 | 0.069 | 8.000 |
| russell | -0.020 | 0.001 | -0.014 | -0.017 | -0.014 | -0.002 | -0.014 | -0.019 | -0.005 | -0.004 | 0.005 | -0.042 | 0.047 | 4.000 |
| albon | -0.042 | -0.054 | -0.035 | -0.038 | -0.026 | -0.070 | -0.069 | -0.018 | -0.087 | 0.003 | -0.030 | -0.083 | 0.090 | 5.000 |
| bottas | -0.046 | -0.067 | -0.054 | -0.050 | -0.047 | -0.056 | -0.047 | -0.063 | -0.042 | -0.080 | -0.042 | -0.024 | 0.056 | 3.000 |
| perez | -0.048 | -0.079 | -0.050 | -0.055 | -0.055 | -0.048 | -0.053 | -0.049 | -0.044 | -0.080 | -0.046 | -0.044 | 0.037 | 2.000 |
| lawson | -0.092 | -0.083 | -0.096 | -0.094 | -0.107 | -0.064 | -0.112 | -0.087 | -0.091 | -0.100 | -0.107 | -0.140 | 0.076 | 3.000 |
| hulkenberg | -0.111 | -0.137 | -0.114 | -0.115 | -0.103 | -0.154 | -0.118 | -0.100 | -0.144 | -0.081 | -0.154 | -0.114 | 0.073 | 2.000 |
| ocon | -0.179 | -0.179 | -0.176 | -0.175 | -0.171 | -0.198 | -0.195 | -0.191 | -0.202 | -0.162 | -0.174 | -0.153 | 0.049 | 1.000 |
| colapinto | -0.185 | -0.176 | -0.187 | -0.184 | -0.193 | -0.153 | -0.122 | -0.182 | -0.139 | -0.196 | -0.181 | -0.116 | 0.080 | 2.000 |
| stroll | -0.282 | -0.325 | -0.295 | -0.294 | -0.274 | -0.336 | -0.346 | -0.275 | -0.374 | -0.259 | -0.295 | -0.345 | 0.114 | 0.000 |

Per-driver portable skill (s) under each variant:

| driver_id | main | sens_start2006 | sens_notrack | sens_gausscar | sens_notransient | sens_noform | sens_nocompat | sens_splitt | sens_drift_x2 | sens_drift_half | sens_team_spell | sens_team_era | spread_s | rank_range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| max_verstappen | 0.160 | 0.216 | 0.157 | 0.161 | 0.151 | 0.190 | 0.274 | 0.153 | 0.189 | 0.111 | 0.175 | 0.286 | 0.175 | 1.000 |
| hadjar | 0.050 | 0.114 | 0.043 | 0.046 | 0.047 | 0.055 | 0.138 | 0.043 | 0.093 | 0.024 | 0.047 | 0.065 | 0.115 | 3.000 |
| sainz | 0.161 | 0.170 | 0.159 | 0.165 | 0.156 | 0.198 | 0.133 | 0.153 | 0.202 | 0.107 | 0.162 | 0.135 | 0.096 | 3.000 |
| norris | 0.044 | 0.101 | 0.054 | 0.052 | 0.045 | 0.069 | 0.196 | 0.041 | 0.098 | 0.045 | 0.050 | 0.151 | 0.155 | 4.000 |
| leclerc | 0.050 | 0.101 | 0.044 | 0.040 | 0.064 | 0.023 | 0.056 | 0.046 | 0.061 | 0.055 | 0.042 | 0.117 | 0.094 | 5.000 |
| antonelli | 0.045 | 0.068 | 0.047 | 0.045 | 0.035 | 0.075 | 0.088 | 0.046 | 0.104 | 0.009 | 0.049 | 0.020 | 0.095 | 7.000 |
| bortoleto | -0.002 | 0.015 | -0.000 | -0.000 | -0.005 | 0.023 | 0.042 | -0.008 | 0.017 | 0.001 | -0.011 | 0.011 | 0.053 | 5.000 |
| bearman | 0.002 | 0.010 | 0.002 | 0.005 | -0.002 | 0.021 | 0.016 | 0.004 | 0.024 | -0.003 | -0.000 | 0.022 | 0.027 | 4.000 |
| piastri | -0.012 | 0.038 | -0.010 | -0.012 | -0.007 | -0.015 | 0.111 | -0.005 | 0.021 | -0.003 | -0.011 | 0.039 | 0.126 | 9.000 |
| hamilton | 0.017 | 0.027 | 0.005 | 0.010 | 0.012 | 0.040 | -0.002 | 0.012 | 0.044 | -0.011 | 0.024 | 0.022 | 0.055 | 7.000 |
| arvid_lindblad | -0.073 | 0.002 | -0.074 | -0.077 | -0.076 | -0.071 | 0.010 | -0.079 | -0.059 | -0.082 | -0.087 | -0.060 | 0.097 | 10.000 |
| alonso | 0.030 | -0.041 | 0.027 | 0.032 | 0.023 | -0.007 | -0.024 | 0.040 | -0.013 | -0.006 | 0.035 | -0.014 | 0.082 | 8.000 |
| gasly | 0.015 | 0.015 | 0.020 | 0.021 | 0.003 | 0.088 | 0.045 | 0.020 | 0.071 | 0.003 | 0.014 | 0.034 | 0.085 | 8.000 |
| russell | 0.003 | 0.004 | 0.004 | 0.007 | 0.014 | 0.015 | -0.014 | 0.007 | 0.013 | 0.019 | -0.023 | 0.002 | 0.042 | 8.000 |
| albon | -0.066 | -0.059 | -0.060 | -0.074 | -0.053 | -0.111 | -0.069 | -0.056 | -0.108 | -0.015 | -0.057 | -0.042 | 0.095 | 4.000 |
| bottas | 0.041 | -0.009 | 0.037 | 0.040 | 0.042 | 0.016 | -0.047 | 0.027 | -0.001 | 0.031 | 0.049 | 0.009 | 0.096 | 10.000 |
| perez | -0.019 | -0.067 | -0.018 | -0.018 | -0.027 | -0.016 | -0.053 | -0.022 | -0.024 | -0.032 | -0.013 | -0.064 | 0.055 | 3.000 |
| lawson | -0.054 | -0.089 | -0.052 | -0.047 | -0.066 | -0.027 | -0.112 | -0.056 | -0.080 | -0.041 | -0.077 | -0.143 | 0.115 | 5.000 |
| hulkenberg | -0.084 | -0.123 | -0.079 | -0.081 | -0.080 | -0.112 | -0.118 | -0.062 | -0.115 | -0.057 | -0.042 | -0.099 | 0.082 | 4.000 |
| ocon | -0.092 | -0.123 | -0.092 | -0.093 | -0.078 | -0.143 | -0.195 | -0.105 | -0.167 | -0.039 | -0.091 | -0.123 | 0.156 | 3.000 |
| colapinto | -0.040 | -0.070 | -0.034 | -0.035 | -0.041 | -0.042 | -0.122 | -0.038 | -0.055 | -0.021 | -0.037 | -0.105 | 0.101 | 4.000 |
| stroll | -0.194 | -0.300 | -0.197 | -0.198 | -0.172 | -0.273 | -0.346 | -0.179 | -0.336 | -0.112 | -0.187 | -0.275 | 0.234 | 0.000 |

### Data window: 2010 vs 2006 start

Same model fitted from 2006 (adding 2006–09 Q1/Q2 low-fuel times) vs from 2010, on identical forecast targets at 6 cutoffs (end2016, end2019, end2022, end2024, end2025, mid2025).

| comparison | n | rmse_main_s | rmse_variant_s | mse_diff_s2 | 95% interval | in90_main | in90_variant |
|---|---|---|---|---|---|---|---|
| 2006 vs 2010: all pairings | 48 | 0.208 | 0.208 | -0.0002 | -0.0021 to +0.0014 | 0.854 | 0.875 |
| 2006 vs 2010: new pairings | 15 | 0.278 | 0.273 | -0.0030 | -0.0082 to +0.0018 | 0.800 | 0.800 |

| window | rmse_s | crps_s | cov90 | pace_rmse_s | pace_spearman |
|---|---|---|---|---|---|
| 2010 | 0.502 | 0.253 | 0.943 | 0.514 | 0.590 |
| 2006 | 0.500 | 0.253 | 0.944 | 0.514 | 0.582 |

### Residual checks (main fit)

- Correlation of teammates' standardised residuals in the same segment: -0.179 (Q1 -0.138, Q2 -0.308, Q3 -0.125; n = 6,936 pairs). Clearly positive values would mean shared car effects within a segment are not fully captured; negative values arise when a segment's two residuals are pulled apart by the shared car-segment effect.
- Lag-1 autocorrelation of a pairing's teammate-gap residual across events: 0.022 (n = 3,353). Positive values would mean relative form persists beyond what the skill walk captures.
- Standardised residual quantiles 1/5/50/95/99%: -3.55 / -1.75 / 0.02 / 1.40 / 2.25. The slow side has the heavier tail (compromised laps); the variant with a wider slow side (`sens_splitt`) changes the rankings little (rank correlation 0.99 in the current car, 0.99 portable).

### Convergence

6000 posterior draws; divergences: 0; max R-hat skill 1.009, team-specific effect 1.007, car 1.029, hyperparameters 1.038; min ESS skill 383, team-specific effect 530, car 178.

## What the ratings can and cannot say

- **Car-package ratings are well determined in simulation.** Recovery correlation 0.989 (0.981–0.995) over 8 independent truths, 90% coverage 0.89 (0.87–0.92). In the simulated truth, car states vary far more than portable skill states within a season; that describes the simulation, not a share of observed qualifying variation.
- **Teammate comparisons and forecasts** beat a static two-way model, raw teammate gaps and a zero-gap baseline on every forecast target above, including brand-new pairings, with 90% intervals covering 94% per segment.
- **Pace in the current car (headline).** Current-grid rank correlation in simulation 0.83 (0.72–0.93); against the sensitivity variants its rank correlation is 0.96–1.00.
- **Portable skill is experimental.** Current-grid rank correlation in simulation 0.65 (0.53–0.84), and against the sensitivity variants 0.81–0.99. Performance relative to teammates has a large team-specific part, which limits how well the network separates portable skill. Read portable ranks as ranges.
- **The team-specific effect is associated with the team, not proven to be car compatibility.** Support, role, adaptation or selection would produce the same pattern.
- **Drivers who have only raced for one team** (e.g. Piastri, Antonelli, Bortoleto): their portable skill depends on the model's assumption about how team-specific effects are distributed. Simulation follows the model's own assumptions, so it cannot check this.
- **Structural sensitivities of portable skill:** whether the team-specific effect is modelled (0.81), how much history is used (0.88), what one team-specific effect covers (per spell 0.96, per era 0.85); the weekend-form term and the assumed rate of skill drift (0.92–0.92). Details of the car and noise model matter least (≥ 0.99).
- **Misspecification scenarios** (extra team-specific effects, unequal upgrades, persistent form, lucky seasons before a team change) give portable current-grid rank correlations of 0.77–0.87 on their one shared truth, inside the spread between independent clean truths (0.53–0.84). Single draws cannot rank these violations, and none of them falls outside that spread.
- **Scope:** one-lap qualifying pace. Racing qualities are in the Racing section below.

## Racing (stage 2)

**Racing validation needs regeneration.** Outputs without current provenance are excluded from this report and cannot enter the championship. Earlier pass/fail results and equal-car standings are historical results, not validated results of the corrected pipeline. See README, Review fixes and regeneration.

Affected files: `battles/summary.json`, `benchmark/summary.json`, `championship/summary.json`, `consistency/summary.json`, `firstlap/summary.json`, `pitstops/summary.json`, `race/multi_heldout.json`, `race/multi_summary.json`, `reliability/summary.json`, `wet/summary.json`.

Every racing quality has a standalone test on held-out data and, where it has held-out draws, the equal-car championship's entry test. Racing outcomes run from 2010; lap-level evidence from 2018 (FastF1), and from 2010 through Jolpica's lap data with a weaker observation model.

### Summary of the racing gates

`own gate`: the quality's standalone test on held-out data (baselines keep the car and context terms). `overall rating`: the nested entry test of the equal-car championship's race stage (held-out finishing orders, with vs without the quality, carrying its posterior draws; mean difference in log predictive density per race and its 95% interval). A quality can fail its own gate and still enter. Team qualities are set to the average in the equal-car championship.

| quality | kind | own gate | overall rating |
|---|---|---|---|
| overtaking and defending | driver | not feasible | not tested (no held-out test) |

### Racing data

- **FastF1 (2018 onward):** 187 races, 29 sprint races: lap times, positions, tyres, track status, race-control messages, weather.
- **Jolpica lap-by-lap data (2010-2017):** from Jolpica's CSV database dump (uploaded 2026-09-13, sha256 matching the published value). Against the API pages on the 30 races both have: 32,874 laps, lap times equal 100.0%, positions equal 100.0%; pit stops on the same lap 100.0%.
- **What Jolpica lacks is inferred** (oldlaps.py), checked against FastF1 on 186 races: positions agree 99.3%; a gap under 1 s at the line agrees 99.4%; neutralised laps (from the field's lap times) precision 0.73, recall 0.75; inferred pit laps precision 0.84, recall 0.59 (against recorded stops 2011-2017: 0.86 / 0.74; inference is used only for 2010). No tyre compounds or speed traps before 2018: those races enter through a weaker observation model (stated per model).
- **Wet races before 2018** from hourly precipitation at the circuit (Open-Meteo archive, ERA5): precipitation >= 0.2 mm from start to +2 h. Against FastF1's wet flag on 187 races: precision 0.64, recall 0.60, a weak proxy; models give proxy-wet races their own coefficient. Threshold sweep in `outputs/analysis/conditions_check.json`.
- **Power-unit suppliers:** 180 team-seasons from each season's Wikipedia entry list (powerunits.py; pages saved with hashes). One badge that names no supplier is set by hand (2017 Toro Rosso: Renault).
- **Telemetry:** per-lap coasting summaries for 138 dry races (extract/telemetry.py), aligned to the laps by matching FastF1's finish-line speed trap; 3 races miss it by more than 5 km/h (2020-01, 2026-08, 2026-11) and are not used.

### Event timeline (outcomes from 2010, race evidence from 2018)

8,141 rows over 341 races: neutralisations, retirements and other outcomes, and race-control evidence (incidents, penalties, stoppages, off-track moments), plus pit anomalies, suspected damage and possible team orders. Rules produce evidence, not verdicts. Full audit: `outputs/timeline/AUDIT.md`.

- **Retirement causes.** 1214 retirements: 939 from a coded status, 228 inferred from race evidence, 47 from class base rates (uncoded, no race evidence). Jolpica codes causes through 2022 but records almost every later retirement only as "Retired"; those get probabilities from a model of the cause class given race-control and lap evidence, fitted on 315 coded retirements. Held out a season at a time, its log loss is 0.761 against 0.879 for base rates (accuracy 0.65 vs 0.43): informative, far from certain.
- **Who caused an incident** is split by stated assumptions (the audit lists them), not estimates.

### Overtaking and defending

Synthetic feasibility at the estimated effect sizes (at the estimated size, mean over 3 truths: corr >= 0.7 and 90% coverage 85-97%, for both attacker and defender effects): attacker correlation 0.68, coverage 87%; defender correlation 0.50, coverage 87%: **not feasible**.
