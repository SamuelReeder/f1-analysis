# F1 driver skill vs car performance: qualifying-pace ratings

Data through **Azerbaijan Grand Prix 2026** (event 2026-15), window 2010-2026, 15,656 qualifying lap times. Model `quali-v1`, fit `20260928T052937Z`.

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

Pace relative to the field's drivers in the car each driver has now: portable skill plus the driver's team-specific effect. This is what teammate comparisons measure directly. In simulation on the real network (8 independent truths) the current-grid order is recovered with rank correlation 0.83 (0.74–0.92) (mean, range over truths). Read ranks as ranges.

| in_team_rank | name | team | in_car | lo90 | hi90 | ranks | p_fastest |
|---|---|---|---|---|---|---|---|
| 1 | Max Verstappen | Red Bull | 0.275 | 0.118 | 0.435 | 1–6 | 0.544 |
| 2 | Isack Hadjar | Red Bull | 0.169 | 0.001 | 0.340 | 2–12 | 0.035 |
| 3 | Carlos Sainz | Williams | 0.132 | -0.055 | 0.325 | 1–15 | 0.102 |
| 4 | Lando Norris | McLaren | 0.119 | -0.097 | 0.326 | 1–16 | 0.102 |
| 5 | Charles Leclerc | Ferrari | 0.087 | -0.101 | 0.268 | 2–17 | 0.045 |
| 6 | Andrea Kimi Antonelli | Mercedes | 0.056 | -0.130 | 0.250 | 2–18 | 0.033 |
| 7 | Gabriel Bortoleto | Audi | 0.043 | -0.158 | 0.236 | 2–18 | 0.030 |
| 8 | Oscar Piastri | McLaren | 0.036 | -0.179 | 0.243 | 3–20 | 0.004 |
| 9 | Oliver Bearman | Haas F1 Team | 0.035 | -0.167 | 0.243 | 2–19 | 0.033 |
| 10 | Arvid Lindblad | RB F1 Team | 0.023 | -0.148 | 0.191 | 3–18 | 0.007 |
| 11 | Fernando Alonso | Aston Martin | 0.021 | -0.194 | 0.232 | 2–19 | 0.027 |
| 12 | Lewis Hamilton | Ferrari | 0.017 | -0.170 | 0.204 | 4–19 | 0.003 |
| 13 | Pierre Gasly | Alpine F1 Team | -0.004 | -0.173 | 0.166 | 4–19 | 0.008 |
| 14 | George Russell | Mercedes | -0.015 | -0.199 | 0.169 | 4–20 | 0.002 |
| 15 | Alexander Albon | Williams | -0.046 | -0.229 | 0.140 | 5–21 | 0.000 |
| 16 | Sergio Pérez | Cadillac F1 Team | -0.050 | -0.289 | 0.201 | 3–21 | 0.012 |
| 17 | Valtteri Bottas | Cadillac F1 Team | -0.051 | -0.295 | 0.194 | 3–21 | 0.013 |
| 18 | Liam Lawson | RB F1 Team | -0.094 | -0.248 | 0.066 | 8–21 | 0.000 |
| 19 | Nico Hülkenberg | Audi | -0.112 | -0.307 | 0.087 | 7–22 | 0.000 |
| 20 | Esteban Ocon | Haas F1 Team | -0.176 | -0.383 | 0.028 | 10–22 | 0.000 |
| 21 | Franco Colapinto | Alpine F1 Team | -0.185 | -0.359 | -0.015 | 12–22 | 0.000 |
| 22 | Lance Stroll | Aston Martin | -0.279 | -0.490 | -0.072 | 15–22 | 0.000 |

## Portable skill (experimental)

`skill` is the part of a driver's pace expected to carry over to another team: pace in the current car minus `team_effect`, the driver's persistent team-specific effect. **Experimental:** in simulation the current-grid order of portable skill is recovered with rank correlation 0.70 (0.51–0.89) (mean, range over 8 truths), and it moves with structural choices (see Sensitivity). The team-specific effect is not established to be car-handling compatibility: team support, role, adaptation or selection would look the same.

| name | team | skill | lo90 | hi90 | ranks | p_fastest | team_effect | events | teams | teammates | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Max Verstappen | Red Bull | 0.166 | -0.046 | 0.378 | 1–15 | 0.270 | 0.108 | 247 | 2 | 8 | sensitive to model choice |
| Carlos Sainz | Williams | 0.159 | -0.033 | 0.355 | 1–14 | 0.225 | -0.027 | 245 | 5 | 7 | sensitive to model choice |
| Charles Leclerc | Ferrari | 0.054 | -0.167 | 0.285 | 1–20 | 0.073 | 0.031 | 188 | 2 | 5 |  |
| Andrea Kimi Antonelli | Mercedes | 0.050 | -0.120 | 0.228 | 2–18 | 0.031 | 0.006 | 39 | 1 | 1 |  |
| Isack Hadjar | Red Bull | 0.049 | -0.100 | 0.216 | 2–17 | 0.021 | 0.113 | 36 | 2 | 3 | sensitive to model choice |
| Lando Norris | McLaren | 0.045 | -0.185 | 0.282 | 1–20 | 0.070 | 0.069 | 167 | 1 | 3 | sensitive to model choice |
| Valtteri Bottas | Cadillac F1 Team | 0.038 | -0.179 | 0.251 | 1–20 | 0.055 | -0.084 | 262 | 4 | 6 | sensitive to model choice |
| Fernando Alonso | Aston Martin | 0.030 | -0.198 | 0.263 | 1–21 | 0.064 | -0.010 | 302 | 4 | 6 | sensitive to model choice |
| Pierre Gasly | Alpine F1 Team | 0.021 | -0.191 | 0.225 | 2–21 | 0.033 | -0.022 | 192 | 3 | 8 |  |
| Lewis Hamilton | Ferrari | 0.015 | -0.192 | 0.220 | 2–21 | 0.028 | 0.001 | 343 | 3 | 5 |  |
| George Russell | Mercedes | 0.004 | -0.202 | 0.210 | 2–21 | 0.025 | -0.021 | 167 | 2 | 5 | sensitive to model choice |
| Oliver Bearman | Haas F1 Team | 0.002 | -0.186 | 0.188 | 3–20 | 0.020 | 0.033 | 41 | 1 | 3 |  |
| Gabriel Bortoleto | Audi | 0.001 | -0.163 | 0.176 | 3–20 | 0.015 | 0.037 | 38 | 1 | 1 |  |
| Oscar Piastri | McLaren | -0.012 | -0.198 | 0.188 | 3–21 | 0.017 | 0.043 | 85 | 1 | 1 | sensitive to model choice |
| Sergio Pérez | Cadillac F1 Team | -0.021 | -0.231 | 0.190 | 3–21 | 0.021 | -0.028 | 297 | 5 | 7 |  |
| Franco Colapinto | Alpine F1 Team | -0.040 | -0.212 | 0.125 | 4–21 | 0.005 | -0.140 | 42 | 2 | 2 | sensitive to model choice |
| Liam Lawson | RB F1 Team | -0.053 | -0.230 | 0.109 | 5–21 | 0.003 | -0.037 | 50 | 2 | 4 | sensitive to model choice |
| Alexander Albon | Williams | -0.069 | -0.287 | 0.143 | 4–22 | 0.011 | 0.024 | 143 | 3 | 6 |  |
| Arvid Lindblad | RB F1 Team | -0.073 | -0.217 | 0.106 | 6–21 | 0.003 | 0.087 | 15 | 1 | 2 | sensitive to model choice |
| Nico Hülkenberg | Audi | -0.083 | -0.284 | 0.115 | 5–22 | 0.005 | -0.026 | 268 | 5 | 11 |  |
| Esteban Ocon | Haas F1 Team | -0.094 | -0.307 | 0.111 | 5–22 | 0.006 | -0.080 | 194 | 4 | 6 | sensitive to model choice |
| Lance Stroll | Aston Martin | -0.196 | -0.463 | 0.058 | 8–22 | 0.002 | -0.081 | 205 | 2 | 7 | sensitive to model choice |

`flag`: rating moves by more than 0.10 s, or rank by 8+ places, across the sensitivity variants below. (The rule set before the results, 0.05 s or 4 places, flagged every driver because it is smaller than the posterior uncertainty, so it was loosened.) `events`: events entered. `teams`: team lineages raced (5+ events). These counts are context, not a precision measure; use the interval.

## Car packages (latest event, track-neutral)

`track_profile` > 0: relatively stronger on high-speed circuits; < 0: on slow/street circuits.

| constructor | rating | lo90 | hi90 | gap_to_fastest | ranks | track_profile |
|---|---|---|---|---|---|---|
| Mercedes | 0.906 | 0.671 | 1.151 | 0.000 | 1–3 | 0.016 |
| Ferrari | 0.731 | 0.495 | 0.962 | -0.197 | 1–4 | -0.022 |
| McLaren | 0.693 | 0.432 | 0.954 | -0.234 | 1–4 | -0.017 |
| Red Bull | 0.570 | 0.363 | 0.786 | -0.364 | 2–5 | -0.038 |
| Alpine F1 Team | 0.253 | 0.016 | 0.492 | -0.686 | 4–7 | 0.040 |
| RB F1 Team | 0.105 | -0.125 | 0.325 | -0.838 | 5–8 | 0.104 |
| Audi | 0.050 | -0.214 | 0.309 | -0.895 | 5–8 | 0.048 |
| Haas F1 Team | -0.182 | -0.448 | 0.084 | -1.125 | 6–9 | 0.014 |
| Williams | -0.534 | -0.773 | -0.267 | -1.474 | 8–9 | -0.022 |
| Aston Martin | -1.249 | -1.637 | -0.898 | -2.192 | 10–11 | -0.156 |
| Cadillac F1 Team | -1.335 | -1.648 | -1.026 | -2.273 | 10–11 | 0.034 |

## Data

- **Entered drivers without a valid lap** (115 of 7,216 driver-events: no time set, or only laps more than 5% off the segment median) are kept. They have no observations, their rating is carried forward by the skill walk, and they count in the field that ratings are relative to.
- **Times missing from Jolpica** (2025-06: every qualifying time blank) are rebuilt from FastF1 lap timing (best non-deleted lap per driver and segment, `extract/quali_fill.py`). On the other events of the same season the method matches Jolpica exactly for 1,014 of 1,015 times. FastF1 also has 8 times that Jolpica does not: laps deleted by race control that FastF1 did not flag, or times removed after the session, which lap timing cannot see. For the filled event, no race-control lap deletion names a lap that was kept.

## Validation

### Forecasting (25 leave-future-out cutoffs, 2013–2026)

Each cutoff is fitted only on data up to that point (including the circuit factor) and forecasts the next season (end-of-season cutoffs) or the rest of the season (after round 8). Established drivers only (≥10 prior events). RMSE and CRPS are in seconds.

| target | model | static two-way | naive raw gaps | zero |
|---|---|---|---|---|
| teammate gap, per segment | 0.549 | 0.570 | 0.570 | 0.579 |
| teammate gap, per pairing (all, n=210) | 0.188 | 0.245 | 0.262 | 0.264 |
| teammate gap, per pairing (continuing, n=167) | 0.171 | 0.182 | 0.200 | 0.265 |
| teammate gap, per pairing (new, n=43) | 0.243 | 0.408 | 0.425 | 0.264 |

Session-level calibration of teammate-gap forecasts: 50% intervals cover 56.4%, 90% intervals cover 94.1%; CRPS 0.267s. Pairing-level 90% intervals cover 86.7%.

Full relative pace (car + driver) within each forecast segment:

| forecaster | rmse_s | spearman |
|---|---|---|
| model | 0.544 | 0.658 |
| static two-way | 0.564 | 0.636 |
| recent form (last 5 events) | 0.566 | 0.625 |

### Synthetic recovery on the real F1 network

Truth drawn from the model's generative process on the real entries, teams and segments (including the drivers entered without a valid lap). Coverage = share of true values inside the 90% interval.

**Independent clean truths (8).** Each has its own truth, its own noise and its own hyperparameters: a different posterior draw from the main fit, so settings such as the size of team-specific effects vary between truths. The gates use the means.

| truth | sd_team_effect | skill_cov90 | car_cov90 | grid_spearman | grid_cov90 | grid_in_team_spearman | grid_in_team_cov90 | cross_team_cov90 | divergences |
|---|---|---|---|---|---|---|---|---|---|
| seed1 | 0.127 | 0.946 | 0.890 | 0.894 | 0.909 | 0.916 | 0.818 | 0.977 | 0 |
| seed2 | 0.111 | 0.836 | 0.888 | 0.799 | 0.955 | 0.827 | 0.909 | 0.936 | 0 |
| seed3 | 0.188 | 0.922 | 0.897 | 0.615 | 0.909 | 0.832 | 0.773 | 0.882 | 0 |
| seed4 | 0.173 | 0.922 | 0.890 | 0.514 | 0.864 | 0.828 | 0.909 | 0.891 | 0 |
| seed5 | 0.198 | 0.869 | 0.863 | 0.626 | 0.864 | 0.745 | 0.682 | 0.823 | 0 |
| seed6 | 0.173 | 0.951 | 0.917 | 0.691 | 0.909 | 0.862 | 0.955 | 0.891 | 0 |
| seed7 | 0.144 | 0.918 | 0.873 | 0.857 | 0.909 | 0.857 | 0.636 | 0.868 | 0 |
| seed8 | 0.139 | 0.817 | 0.792 | 0.605 | 0.864 | 0.806 | 0.773 | 0.764 | 0 |

Mean (range): skill_cov90 0.90 (0.82–0.95); car_cov90 0.88 (0.79–0.92); grid_spearman 0.70 (0.51–0.89); grid_cov90 0.90 (0.86–0.95); grid_in_team_spearman 0.83 (0.74–0.92); grid_in_team_cov90 0.81 (0.64–0.95). Eight truths show the spread between truths; they do not pin down coverage to better than a few percentage points.

**Misspecification scenarios (one shared truth).** These add effects the model does not contain. All share one simulated truth (posterior medians), so they compare scenarios on equal terms but each is a single draw.

| scenario | skill_corr | skill_rmse_s | skill_cov90 | car_corr | car_cov90 | grid_spearman | grid_cov90 | cross_team_cov90 | grid_in_team_spearman | car_latent_share_true/est |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | 0.851 | 0.099 | 0.847 | 0.993 | 0.898 | 0.761 | 0.818 | 0.836 | 0.938 | 0.96/0.97 |
| compat | 0.830 | 0.111 | 0.792 | 0.993 | 0.904 | 0.797 | 0.773 | 0.759 | 0.810 | 0.96/0.98 |
| upgrades | 0.832 | 0.105 | 0.810 | 0.993 | 0.893 | 0.819 | 0.727 | 0.800 | 0.929 | 0.96/0.97 |
| form | 0.811 | 0.114 | 0.889 | 0.991 | 0.892 | 0.793 | 0.909 | 0.950 | 0.907 | 0.96/0.95 |
| transfer_luck | 0.827 | 0.105 | 0.840 | 0.992 | 0.876 | 0.861 | 0.909 | 0.877 | 0.918 | 0.96/0.96 |

`car_latent_share`: within each season, the variance of the synthetic car states divided by the variance of car states plus portable skill states, averaged over seasons (true vs estimated). It covers only those two latent components (not the team-specific effect, weekend effects or noise) and describes the simulated truth, not a share of observed qualifying variation.

### The team-specific effect

**Placebo test.** Team-specific effect SD 0.162% (90% interval 0.114–0.205) vs placebo SD 0.018% (0.001–0.048) when a driver's stint in one team is split in half. P(placebo > team effect) = 0.00. A large team-specific SD with a small placebo SD means relative performance shifts when a driver changes team, much more than with time spent in the same team. The test cannot say why (car handling, team support, role, adaptation or selection).

**What one effect covers.** The main model gives one effect per driver and team lineage, so separate spells with the same lineage share it (9 cases, e.g. gasly (faenza: 2017–2018, 2019–2022); hulkenberg (hinwil: 2013, 2025–2026); hulkenberg (silverstone: 2012, 2014–2016, 2020–2022)). Two variants test this choice: one effect per spell (a return after 3+ events with other teams starts a new one; a one-off stand-in drive does not) and one per regulation era (2014, 2017, 2022 and 2026 start new ones).

Current grid under each variant vs the main model (all data):

| variant | in_team_spearman | in_team_max_shift_s | portable_spearman | portable_max_shift_s |
|---|---|---|---|---|
| per spell | 0.949 | 0.055 | 0.967 | 0.030 |
| per era | 0.942 | 0.078 | 0.888 | 0.131 |

Forecasts of teammate gaps per pairing, variant vs main model on identical targets at 6 cutoffs (end2016, end2019, end2021, end2024, end2025, mid2025). `mse_diff_s2`: mean difference in squared error (variant minus main, s²; negative = variant better), with a bootstrap 95% interval over pairings (pairings at one cutoff share a fit, so it is somewhat too narrow).

| comparison | n | rmse_main_s | rmse_variant_s | mse_diff_s2 | 95% interval | in90_main | in90_variant |
|---|---|---|---|---|---|---|---|
| per spell: all pairings | 49 | 0.219 | 0.213 | -0.0029 | -0.0076 to -0.0001 | 0.837 | 0.816 |
| per spell: first season after a reset | 28 | 0.259 | 0.250 | -0.0048 | -0.0124 to -0.0001 | 0.750 | 0.714 |
| per era: all pairings | 49 | 0.219 | 0.212 | -0.0033 | -0.0120 to +0.0039 | 0.837 | 0.939 |
| per era: first season after a reset | 28 | 0.259 | 0.250 | -0.0048 | -0.0202 to +0.0075 | 0.750 | 0.929 |

Forecasts favour one effect per spell slightly (interval below zero, but too narrow as noted). The main model keeps one effect per lineage in this version; switching is a change for the next model version, with its own refit and validation.

### Sensitivity of the current driver ranking

| variant | in_team_spearman | in_team_max_shift_s | in_team_max_rank_shift | portable_spearman | portable_max_shift_s | portable_max_rank_shift |
|---|---|---|---|---|---|---|
| sens_start2006 | 0.983 | 0.051 | 3 | 0.863 | 0.111 | 7 |
| sens_notrack | 0.982 | 0.026 | 3 | 0.992 | 0.016 | 2 |
| sens_gausscar | 0.984 | 0.027 | 3 | 0.992 | 0.015 | 2 |
| sens_notransient | 0.991 | 0.026 | 2 | 0.964 | 0.026 | 3 |
| sens_noform | 0.925 | 0.079 | 6 | 0.922 | 0.097 | 7 |
| sens_nocompat | 0.962 | 0.064 | 5 | 0.813 | 0.148 | 9 |
| sens_splitt | 0.984 | 0.028 | 4 | 0.991 | 0.023 | 3 |
| sens_drift_x2 | 0.933 | 0.108 | 6 | 0.922 | 0.145 | 7 |
| sens_drift_half | 0.967 | 0.048 | 5 | 0.929 | 0.077 | 6 |
| sens_team_spell | 0.949 | 0.055 | 7 | 0.967 | 0.030 | 4 |
| sens_team_era | 0.942 | 0.078 | 5 | 0.888 | 0.131 | 6 |

Per-driver pace in the current car (s) under each variant:

| driver_id | main | sens_start2006 | sens_notrack | sens_gausscar | sens_notransient | sens_noform | sens_nocompat | sens_splitt | sens_drift_x2 | sens_drift_half | sens_team_spell | sens_team_era | spread_s | rank_range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| max_verstappen | 0.275 | 0.300 | 0.284 | 0.290 | 0.275 | 0.282 | 0.280 | 0.284 | 0.269 | 0.272 | 0.305 | 0.243 | 0.063 | 0.000 |
| hadjar | 0.169 | 0.179 | 0.181 | 0.183 | 0.158 | 0.199 | 0.146 | 0.176 | 0.183 | 0.162 | 0.197 | 0.130 | 0.069 | 2.000 |
| sainz | 0.132 | 0.124 | 0.137 | 0.135 | 0.141 | 0.143 | 0.135 | 0.137 | 0.166 | 0.125 | 0.142 | 0.155 | 0.042 | 2.000 |
| norris | 0.119 | 0.152 | 0.093 | 0.104 | 0.132 | 0.131 | 0.168 | 0.098 | 0.145 | 0.099 | 0.101 | 0.131 | 0.074 | 3.000 |
| leclerc | 0.087 | 0.091 | 0.093 | 0.090 | 0.104 | 0.085 | 0.070 | 0.098 | 0.088 | 0.110 | 0.101 | 0.113 | 0.043 | 3.000 |
| antonelli | 0.056 | 0.080 | 0.053 | 0.054 | 0.049 | 0.075 | 0.091 | 0.056 | 0.115 | 0.012 | 0.071 | 0.092 | 0.103 | 6.000 |
| bortoleto | 0.043 | 0.028 | 0.029 | 0.037 | 0.039 | 0.038 | 0.038 | 0.030 | 0.021 | 0.061 | -0.012 | 0.018 | 0.073 | 8.000 |
| piastri | 0.036 | 0.072 | 0.010 | 0.019 | 0.062 | 0.033 | 0.086 | 0.014 | 0.044 | 0.022 | 0.016 | 0.051 | 0.076 | 6.000 |
| bearman | 0.035 | 0.023 | 0.030 | 0.042 | 0.030 | 0.032 | 0.011 | 0.029 | 0.039 | 0.040 | 0.039 | 0.038 | 0.031 | 5.000 |
| arvid_lindblad | 0.023 | 0.043 | 0.030 | 0.030 | 0.016 | 0.044 | 0.012 | 0.018 | 0.020 | 0.031 | 0.009 | -0.013 | 0.057 | 6.000 |
| alonso | 0.021 | -0.030 | 0.004 | 0.010 | 0.008 | -0.058 | -0.023 | 0.015 | -0.048 | 0.010 | 0.003 | -0.011 | 0.079 | 6.000 |
| hamilton | 0.017 | 0.021 | 0.025 | 0.024 | 0.022 | 0.046 | 0.008 | 0.026 | 0.039 | 0.021 | 0.032 | 0.060 | 0.052 | 5.000 |
| gasly | -0.004 | 0.012 | 0.004 | 0.010 | -0.018 | 0.039 | 0.040 | 0.003 | 0.063 | -0.008 | 0.010 | 0.049 | 0.081 | 7.000 |
| russell | -0.015 | 0.013 | -0.014 | -0.020 | -0.002 | -0.025 | -0.006 | -0.014 | -0.007 | -0.013 | 0.003 | -0.036 | 0.049 | 3.000 |
| albon | -0.046 | -0.060 | -0.032 | -0.046 | -0.021 | -0.068 | -0.067 | -0.018 | -0.086 | 0.002 | -0.032 | -0.076 | 0.088 | 5.000 |
| perez | -0.050 | -0.081 | -0.044 | -0.043 | -0.071 | -0.032 | -0.047 | -0.041 | -0.033 | -0.085 | -0.049 | -0.038 | 0.053 | 4.000 |
| bottas | -0.051 | -0.072 | -0.049 | -0.046 | -0.061 | -0.042 | -0.042 | -0.056 | -0.045 | -0.081 | -0.043 | -0.015 | 0.065 | 3.000 |
| lawson | -0.094 | -0.082 | -0.084 | -0.084 | -0.097 | -0.051 | -0.099 | -0.078 | -0.082 | -0.094 | -0.094 | -0.142 | 0.092 | 4.000 |
| hulkenberg | -0.112 | -0.128 | -0.124 | -0.128 | -0.103 | -0.158 | -0.121 | -0.107 | -0.163 | -0.082 | -0.165 | -0.126 | 0.084 | 3.000 |
| ocon | -0.176 | -0.188 | -0.179 | -0.176 | -0.173 | -0.202 | -0.200 | -0.193 | -0.204 | -0.161 | -0.170 | -0.156 | 0.048 | 1.000 |
| colapinto | -0.185 | -0.169 | -0.183 | -0.175 | -0.191 | -0.163 | -0.134 | -0.179 | -0.126 | -0.198 | -0.175 | -0.107 | 0.091 | 3.000 |
| stroll | -0.279 | -0.326 | -0.293 | -0.306 | -0.289 | -0.353 | -0.344 | -0.281 | -0.387 | -0.258 | -0.301 | -0.355 | 0.130 | 0.000 |

Per-driver portable skill (s) under each variant:

| driver_id | main | sens_start2006 | sens_notrack | sens_gausscar | sens_notransient | sens_noform | sens_nocompat | sens_splitt | sens_drift_x2 | sens_drift_half | sens_team_spell | sens_team_era | spread_s | rank_range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| max_verstappen | 0.166 | 0.216 | 0.164 | 0.173 | 0.144 | 0.201 | 0.280 | 0.164 | 0.203 | 0.114 | 0.180 | 0.296 | 0.183 | 1.000 |
| hadjar | 0.049 | 0.115 | 0.053 | 0.061 | 0.050 | 0.068 | 0.146 | 0.050 | 0.101 | 0.035 | 0.053 | 0.074 | 0.111 | 2.000 |
| sainz | 0.159 | 0.167 | 0.155 | 0.156 | 0.151 | 0.201 | 0.135 | 0.155 | 0.212 | 0.103 | 0.165 | 0.135 | 0.108 | 3.000 |
| norris | 0.045 | 0.115 | 0.030 | 0.033 | 0.066 | 0.060 | 0.168 | 0.023 | 0.076 | 0.037 | 0.035 | 0.139 | 0.145 | 7.000 |
| leclerc | 0.054 | 0.092 | 0.053 | 0.056 | 0.066 | 0.048 | 0.070 | 0.058 | 0.064 | 0.060 | 0.069 | 0.121 | 0.072 | 4.000 |
| antonelli | 0.050 | 0.077 | 0.048 | 0.050 | 0.036 | 0.066 | 0.091 | 0.053 | 0.098 | 0.012 | 0.048 | 0.024 | 0.085 | 5.000 |
| bortoleto | 0.001 | 0.017 | -0.005 | 0.002 | -0.002 | 0.024 | 0.038 | -0.009 | 0.009 | 0.004 | -0.016 | -0.000 | 0.054 | 5.000 |
| piastri | -0.012 | 0.045 | -0.021 | -0.020 | 0.007 | -0.011 | 0.086 | -0.014 | 0.002 | -0.007 | -0.017 | 0.028 | 0.108 | 9.000 |
| bearman | 0.002 | 0.005 | -0.002 | 0.003 | -0.010 | 0.014 | 0.011 | -0.000 | 0.021 | -0.006 | 0.003 | 0.017 | 0.032 | 5.000 |
| arvid_lindblad | -0.073 | -0.004 | -0.076 | -0.075 | -0.069 | -0.061 | 0.012 | -0.080 | -0.054 | -0.073 | -0.080 | -0.056 | 0.092 | 11.000 |
| alonso | 0.030 | -0.036 | 0.023 | 0.028 | 0.017 | -0.024 | -0.023 | 0.034 | -0.024 | -0.010 | 0.024 | -0.019 | 0.070 | 8.000 |
| hamilton | 0.015 | 0.011 | 0.011 | 0.009 | 0.004 | 0.043 | 0.008 | 0.017 | 0.043 | -0.009 | 0.032 | 0.033 | 0.053 | 6.000 |
| gasly | 0.021 | 0.016 | 0.026 | 0.029 | 0.008 | 0.076 | 0.040 | 0.024 | 0.085 | 0.008 | 0.027 | 0.040 | 0.077 | 7.000 |
| russell | 0.004 | 0.009 | 0.008 | 0.008 | 0.025 | -0.006 | -0.006 | 0.010 | 0.017 | 0.022 | -0.024 | 0.014 | 0.049 | 8.000 |
| albon | -0.069 | -0.064 | -0.060 | -0.072 | -0.043 | -0.096 | -0.067 | -0.054 | -0.091 | -0.015 | -0.064 | -0.045 | 0.081 | 4.000 |
| perez | -0.021 | -0.065 | -0.013 | -0.013 | -0.046 | 0.005 | -0.047 | -0.012 | -0.010 | -0.030 | -0.016 | -0.057 | 0.069 | 5.000 |
| bottas | 0.038 | -0.018 | 0.045 | 0.045 | 0.039 | 0.031 | -0.042 | 0.040 | 0.006 | 0.031 | 0.054 | 0.016 | 0.096 | 11.000 |
| lawson | -0.053 | -0.084 | -0.044 | -0.047 | -0.072 | -0.025 | -0.099 | -0.049 | -0.074 | -0.040 | -0.069 | -0.136 | 0.111 | 5.000 |
| hulkenberg | -0.083 | -0.119 | -0.095 | -0.098 | -0.074 | -0.130 | -0.121 | -0.076 | -0.138 | -0.065 | -0.053 | -0.106 | 0.085 | 4.000 |
| ocon | -0.094 | -0.131 | -0.082 | -0.091 | -0.081 | -0.146 | -0.200 | -0.113 | -0.162 | -0.036 | -0.093 | -0.128 | 0.164 | 3.000 |
| colapinto | -0.040 | -0.066 | -0.036 | -0.036 | -0.038 | -0.044 | -0.134 | -0.038 | -0.047 | -0.021 | -0.037 | -0.101 | 0.113 | 5.000 |
| stroll | -0.196 | -0.307 | -0.197 | -0.211 | -0.183 | -0.293 | -0.344 | -0.182 | -0.341 | -0.119 | -0.196 | -0.286 | 0.225 | 0.000 |

### Data window: 2010 vs 2006 start

Same model fitted from 2006 (adding 2006–09 Q1/Q2 low-fuel times) vs from 2010, on identical forecast targets at 6 cutoffs (end2016, end2019, end2022, end2024, end2025, mid2025).

| comparison | n | rmse_main_s | rmse_variant_s | mse_diff_s2 | 95% interval | in90_main | in90_variant |
|---|---|---|---|---|---|---|---|
| 2006 vs 2010: all pairings | 48 | 0.207 | 0.205 | -0.0008 | -0.0030 to +0.0012 | 0.875 | 0.854 |
| 2006 vs 2010: new pairings | 15 | 0.275 | 0.268 | -0.0034 | -0.0101 to +0.0023 | 0.800 | 0.800 |

| window | rmse_s | crps_s | cov90 | pace_rmse_s | pace_spearman |
|---|---|---|---|---|---|
| 2010 | 0.501 | 0.253 | 0.942 | 0.509 | 0.602 |
| 2006 | 0.500 | 0.253 | 0.938 | 0.508 | 0.600 |

### Residual checks (main fit)

- Correlation of teammates' standardised residuals in the same segment: -0.179 (Q1 -0.137, Q2 -0.307, Q3 -0.125; n = 6,936 pairs). Clearly positive values would mean shared car effects within a segment are not fully captured; negative values arise when a segment's two residuals are pulled apart by the shared car-segment effect.
- Lag-1 autocorrelation of a pairing's teammate-gap residual across events: 0.022 (n = 3,353). Positive values would mean relative form persists beyond what the skill walk captures.
- Standardised residual quantiles 1/5/50/95/99%: -3.55 / -1.75 / 0.02 / 1.40 / 2.24. The slow side has the heavier tail (compromised laps); the variant with a wider slow side (`sens_splitt`) changes the rankings little (rank correlation 0.98 in the current car, 0.99 portable).

### Convergence

6000 posterior draws; divergences: 0; max R-hat skill 1.010, team-specific effect 1.010, car 1.020, hyperparameters 1.015; min ESS skill 209, team-specific effect 368, car 308.

## What the ratings can and cannot say

- **Car-package ratings are well determined in simulation.** Recovery correlation 0.989 (0.976–0.994) over 8 independent truths, 90% coverage 0.88 (0.79–0.92). In the simulated truth, car states vary far more than portable skill states within a season; that describes the simulation, not a share of observed qualifying variation.
- **Teammate comparisons and forecasts** beat a static two-way model, raw teammate gaps and a zero-gap baseline on every forecast target above, including brand-new pairings, with 90% intervals covering 94% per segment.
- **Pace in the current car (headline).** Current-grid rank correlation in simulation 0.83 (0.74–0.92); against the sensitivity variants its rank correlation is 0.93–0.99.
- **Portable skill is experimental.** Current-grid rank correlation in simulation 0.70 (0.51–0.89), and against the sensitivity variants 0.81–0.99. Performance relative to teammates has a large team-specific part, which limits how well the network separates portable skill. Read portable ranks as ranges.
- **The team-specific effect is associated with the team, not proven to be car compatibility.** Support, role, adaptation or selection would produce the same pattern.
- **Drivers who have only raced for one team** (e.g. Piastri, Antonelli, Bortoleto): their portable skill depends on the model's assumption about how team-specific effects are distributed. Simulation follows the model's own assumptions, so it cannot check this.
- **Structural sensitivities of portable skill:** whether the team-specific effect is modelled (0.81), how much history is used (0.86), what one team-specific effect covers (per spell 0.97, per era 0.89); the weekend-form term and the assumed rate of skill drift (0.92–0.93). Details of the car and noise model matter least (≥ 0.96).
- **Misspecification scenarios** (extra team-specific effects, unequal upgrades, persistent form, lucky seasons before a team change) give portable current-grid rank correlations of 0.79–0.86 on their one shared truth, inside the spread between independent clean truths (0.51–0.89). Single draws cannot rank these violations, and none of them falls outside that spread.
- **Scope:** one-lap qualifying pace. Racing qualities are in the Racing section below.

## Racing (stage 2)

Every racing quality has a standalone test on held-out data and, where it has held-out draws, the equal-car championship's entry test. Racing outcomes run from 2010; lap-level evidence from 2018 (FastF1), and from 2010 through Jolpica's lap data with a weaker observation model.

### Summary of the racing gates

`own gate`: the quality's standalone test on held-out data (baselines keep the car and context terms). `overall rating`: the nested entry test of the equal-car championship's race stage (held-out finishing orders, with vs without the quality, carrying its posterior draws; mean difference in log predictive density per race and its 95% interval). A quality can fail its own gate and still enter. Team qualities are set to the average in the equal-car championship.

| quality | kind | own gate | overall rating |
|---|---|---|---|
| race-specific pace (beyond the qualifying link) | driver | pass | enters (+0.246, +0.156 to +0.341) |
| tyre degradation | driver | pass | does not enter (-0.002, -0.009 to +0.005) |
| consistency (lap-time spread) | driver | fail | does not enter (+0.030, -0.002 to +0.062) |
| wet-weather pace | driver | fail | not tested (the championship simulates dry races) |
| error rate (own-error retirements) | driver | fail | does not enter (its own held-out test is the incident stage's entry test) |
| first-lap performance | driver | fail | enters (+0.049, +0.017 to +0.086) |
| overtaking and defending | driver | not feasible | not tested (no held-out test) |
| unexplained results effect | driver | fail | does not enter (its own test is on held-out finishing orders) |
| mechanical reliability (team-season) | team | fail | equalised |
| mechanical reliability (power-unit supplier) | team | fail | equalised |
| pit stops (team operations) | team | pass | equalised |

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

### Race pace and tyre degradation

One joint lap-level model across seasons (racemulti.py): lap = race trend + compounds + degradation + dirty air + car (team x race) + gamma x qualifying pace (stage 1) + race-specific driver pace (lasting, per season, per race) + driver degradation, with AR(1) Student-t errors within stints. Laps before 2018 have unknown compounds: a stint offset replaces the compound terms, with their own noise scale.

Full fit: 237,055 clean laps in 259 dry races (fastf1 122,325, jolpica 114,730), 2010–2026; gamma 0.97 (90% interval 0.88–1.08); SD of lasting race-specific pace 0.079 (90% interval 0.059–0.102)% of lap time; max R-hat 1.009, 0 divergences.

Held out (fit on every season before S (from 2010); predict season S's teammate gaps (pair-season means of stage A gaps, >= 4 races, both drivers seen in training); 15 training fits, max R-hat 1.040, 2 of them (2015, 2019) on the second attempt of the fixed retry rule (new seed, doubled warmup); squared error in %², negative = better, bootstrap interval over pair-seasons):

| test | result |
|---|---|
| qualifying link vs zero | -0.0444 (95% interval -0.0694 to -0.0235; 129 pair-seasons) |
| + race-specific pace vs qualifying link | -0.0055 (95% interval -0.0100 to -0.0017; 129 pair-seasons) |
| degradation effects vs zero | -2.7e-05 (95% interval -5.5e-05 to -1.8e-06; 129 pair-seasons) |

- **Race-specific pace** **passes** its gate (interval below zero).
- **Degradation** **passes** its gate.

The same held-out predictions split by era (descriptive, added after the gate result; the gates are the pooled intervals above):

| held out | race-specific pace vs qualifying link | degradation vs zero |
|---|---|---|
| held out 2012-2017 (Jolpica targets) | -0.0017 (95% interval -0.0042 to +5.1e-04; 50 pair-seasons) | -5.8e-05 (95% interval -1.2e-04 to -8.8e-07; 50 pair-seasons) |
| held out 2018 on (FastF1 targets) | -0.0079 (95% interval -0.0154 to -0.0016; 79 pair-seasons) | -6.8e-06 (95% interval -2.6e-05 to +1.2e-05; 79 pair-seasons) |

Neither era alone carries both results: where an interval excludes zero in one era it does not in the other. Before 2018 compounds are unknown, so a teammate difference in degradation there can include a difference in compound choice.

Top 15 by race-specific pace (full fit, 2010–2026; lasting race pace beyond the qualifying link, % of lap time (positive = faster; 0.1% is about 0.09 s on a 90 s lap); identified only from differences between teammates, linked across teams by drivers who move; all 83 drivers in `outputs/race/multi_drivers.csv`; ordered by median, and most intervals overlap):

| driver | median | 90% interval | seasons | clean laps |
|---|---|---|---|---|
| Max Verstappen | +0.1578 | +0.0786 to +0.2305 | 12 | 8221 |
| Sergio Pérez | +0.1282 | +0.0611 to +0.1950 | 15 | 10015 |
| Lance Stroll | +0.1131 | +0.0419 to +0.1843 | 10 | 6290 |
| Daniel Ricciardo | +0.1112 | +0.0431 to +0.1778 | 14 | 8607 |
| Sebastian Vettel | +0.0877 | +0.0199 to +0.1625 | 13 | 8745 |
| Nicholas Latifi | +0.0867 | -0.0106 to +0.1856 | 3 | 1867 |
| Jean-Éric Vergne | +0.0850 | -0.0132 to +0.1959 | 3 | 1748 |
| Charles Pic | +0.0701 | -0.0399 to +0.1920 | 2 | 1282 |
| Logan Sargeant | +0.0608 | -0.0471 to +0.1682 | 2 | 1160 |
| Kimi Räikkönen | +0.0556 | -0.0193 to +0.1327 | 10 | 6757 |
| Guanyu Zhou | +0.0517 | -0.0435 to +0.1491 | 3 | 2258 |
| Charles Leclerc | +0.0477 | -0.0254 to +0.1203 | 9 | 6116 |
| Michael Schumacher | +0.0461 | -0.0556 to +0.1621 | 3 | 1642 |
| Daniil Kvyat | +0.0438 | -0.0404 to +0.1294 | 6 | 3635 |
| Jenson Button | +0.0432 | -0.0371 to +0.1243 | 8 | 4750 |

Top 15 by degradation (full fit, 2010–2026; lasting change in pace per lap of tyre age, % of lap time (positive = loses less pace as the tyres age); identified only from differences between teammates, linked across teams by drivers who move; all 83 drivers in `outputs/race/multi_drivers.csv`; ordered by median, and most intervals overlap):

| driver | median | 90% interval | seasons | clean laps |
|---|---|---|---|---|
| Nico Rosberg | +0.0103 | +0.0051 to +0.0158 | 7 | 4708 |
| Fernando Alonso | +0.0083 | +0.0044 to +0.0122 | 15 | 9851 |
| Lando Norris | +0.0053 | +0.0006 to +0.0098 | 8 | 5914 |
| Daniel Ricciardo | +0.0051 | +0.0014 to +0.0090 | 14 | 8607 |
| Jenson Button | +0.0051 | +0.0002 to +0.0099 | 8 | 4750 |
| Vitantonio Liuzzi | +0.0047 | -0.0023 to +0.0120 | 2 | 981 |
| Paul di Resta | +0.0045 | -0.0016 to +0.0110 | 4 | 2190 |
| Stoffel Vandoorne | +0.0045 | -0.0023 to +0.0117 | 3 | 1167 |
| Michael Schumacher | +0.0045 | -0.0019 to +0.0112 | 3 | 1642 |
| Mark Webber | +0.0040 | -0.0019 to +0.0101 | 4 | 2637 |
| Liam Lawson | +0.0037 | -0.0024 to +0.0102 | 4 | 1784 |
| Andrea Kimi Antonelli | +0.0033 | -0.0027 to +0.0100 | 2 | 1409 |
| Robert Kubica | +0.0032 | -0.0030 to +0.0100 | 3 | 1492 |
| Bruno Senna | +0.0031 | -0.0034 to +0.0102 | 3 | 1387 |
| Yuki Tsunoda | +0.0031 | -0.0022 to +0.0084 | 6 | 3506 |

Variants of the same held-out test:

| variant | race-specific pace vs qualifying link | degradation vs zero |
|---|---|---|
| 2018 onward only (FastF1) | -0.0111 (95% interval -0.0220 to -0.0029; 63 pair-seasons) | -6.5e-06 (95% interval -2.2e-05 to +7.9e-06; 63 pair-seasons) |
| races with aligned telemetry, no coasting term | -0.0107 (95% interval -0.0214 to -0.0025; 62 pair-seasons) | -2.9e-06 (95% interval -1.7e-05 to +1.1e-05; 62 pair-seasons) |
| same races, laps and targets adjusted for coasting | -0.0104 (95% interval -0.0205 to -0.0024; 62 pair-seasons) | -2.4e-06 (95% interval -1.7e-05 to +1.2e-05; 62 pair-seasons) |

The coasting term (telemetry use 1: lift-and-coast seconds per lap) costs 0.02–0.02% of lap time per second of coasting across the training fits; adjusting for it does not change the race-specific pace conclusion.

The earlier two-stage shortcut (per-race estimates, then a model across races) was not accepted by the joint-model check on 2024 (rule: at least 90% of drivers within a quarter of a posterior SD, for both quantities); its driver numbers are not published and this joint model replaces it.

### Reliability and driver errors (2010 onward)

Competing risks per lap (7,188 starts, 1,214 retirements); uncertain causes count fractionally. Mechanical risk has era, power-unit supplier-season and team-season terms; own-error risk has team-season, traffic and driver terms; every cause has wet-race terms (48 FastF1 wet races, 36 proxy-wet races before 2018). Traffic: 1.75 (90% interval 1.18–2.27) on the log own-error rate per unit share of laps in traffic; wet races (FastF1) 0.16 (90% interval -0.17–0.46) own error, 0.17 (90% interval -0.29–0.59) caused by another driver.

Held-out seasons, paired log predictive density per race:

- **Driver own-error effects** (baseline keeps every other term): +0.003 (95% interval -0.002 to +0.010): **does not pass**.
- **Team-season mechanical effects** (second half of each season from its first half, supplier terms kept): -0.003 (95% interval -0.024 to +0.019): **does not pass**.
- **Power-unit supplier-season effects** (team-season terms kept): -0.015 (95% interval -0.055 to +0.023): **does not pass**.

### Pit stops (team operations)

Pit-lane time relative to the race's median stop (11,864 stops, 323 races, 2011–2026; jolpica 5,953, fastf1 5,911); excluded: relative time outside [-5.0, 20.0] s, or red flag on the in-lap. Lasting team SD 0.66 (90% interval 0.49–0.96) s, team-season SD 0.21 (90% interval 0.19–0.23) s; Student-t with very heavy tails (nu 0.97: slow stops). Held out (second half of each season from its first half): team terms +1.609 (95% interval +0.860 to +2.307): **passes**.

2026, seconds per stop relative to the race median (negative = faster):

| team | rel_s_median | rel_s_q05 | rel_s_q95 | stops | share_slow_3s |
|---|---|---|---|---|---|
| brackley | -0.695 | -1.047 | -0.348 | 46 | 0.065 |
| mclaren | -0.497 | -0.864 | -0.153 | 45 | 0.156 |
| ferrari | -0.496 | -0.853 | -0.147 | 49 | 0.102 |
| red_bull | -0.447 | -0.806 | -0.102 | 46 | 0.087 |
| silverstone | -0.324 | -0.711 | 0.051 | 40 | 0.200 |
| williams | -0.320 | -0.675 | 0.023 | 58 | 0.172 |
| faenza | -0.287 | -0.648 | 0.066 | 40 | 0.050 |
| enstone | -0.265 | -0.609 | 0.083 | 44 | 0.045 |
| hinwil | -0.212 | -0.589 | 0.134 | 47 | 0.128 |
| haas | 0.207 | -0.138 | 0.554 | 46 | 0.174 |
| cadillac | 0.574 | 0.160 | 1.016 | 47 | 0.149 |

### Consistency

Each driver's robust spread of clean-lap residuals in the stage-A regression, compared with the teammate's (2,323 teammate races in 259 races; median spread 0.45% of lap time). Driver SD on the log scale 0.043 (90% interval 0.029–0.060); split-half correlation of driver means 0.28. Held out against no driver effect: -2.2e-04 (95% interval -0.0027 to +0.0024; 128 pair-seasons): **does not pass**.

### Wet-weather pace (experimental)

Laps on intermediates or wets, each compared with the field on the same lap (121 teammate races in 15 wet races, 2018 onward). Wet gaps follow qualifying gaps with gamma 1.64 (90% interval 1.21–2.08) (larger than in the dry). A driver-specific wet effect, held out against the qualifying link: squared error -0.046 per race (95% interval -0.192 to +0.072; 13 races): **does not pass**.

### First-lap performance

Positions gained from the grid slot to the end of lap 1 (7,537 starts: 343 races from 2010, 29 sprint races). Slot, side of the grid, start tyre, sprint, a lasting team term, team-season car and driver effects. Driver SD 0.36 (90% interval 0.28–0.46) places, lasting team SD 0.80 (90% interval 0.56–1.18), team-season SD 0.34 (90% interval 0.28–0.41).

Held-out seasons (paired log predictive density per race or sprint):

| model | driver effects, all | drivers who changed team | drivers who stayed |
|---|---|---|---|
| with lasting team term (final) | +0.005 (95% interval -0.062 to +0.073) | -0.017 (95% interval -0.048 to +0.016); 1275 starts | +0.020 (95% interval -0.040 to +0.080) |
| without it (first build's structure) | +0.330 (95% interval +0.229 to +0.430) | +0.032 (95% interval -0.013 to +0.078) | +0.303 (95% interval +0.211 to +0.394) |

The lasting team term itself (no driver effects in either model): +0.464 (95% interval +0.328 to +0.596). 90% predictive intervals cover 92%.

- **Standalone ranking gate** (overall interval above zero, interval for drivers in a new team not entirely below zero; the second condition was written down after the first build's result): **does not pass**. The lasting team term was added after the first build failed; the decision rests on the final run with 2010-2017 starts, fixed in advance (docs/racing_approach.md, decisions fixed before the final runs).

### Overtaking and defending

16,967 battle episodes and 6,432 passes (by source: fastf1 8,891 episodes, jolpica 7,237 episodes, sprint 839 episodes).
Synthetic feasibility at the estimated effect sizes (at the estimated size, mean over 3 truths: corr >= 0.7 and 90% coverage 85-97%, for both attacker and defender effects): attacker correlation 0.68, coverage 87%; defender correlation 0.50, coverage 87%: **not feasible**.
Not rated: reported as opportunity counts (`outputs/battles/drivers.csv`).

### Results benchmark

Rank-ordered logit on full classifications, held-out seasons 2014–2026 (267 races):

| model | log_lik_per_race | spearman | teammate_h2h |
|---|---|---|---|
| ratings | -41.394 | 0.615 | 0.613 |
| ratings_results | -41.306 | 0.577 | 0.601 |
| grid | -40.417 | 0.621 | 0.654 |
| grid_ratings | -40.305 | 0.648 | 0.664 |

- A driver results effect over the stage-1 ratings (the unexplained contribution): +0.087 (95% interval -0.035 to +0.205).
- Grid + ratings vs grid alone: +0.112 (95% interval +0.032 to +0.196).

### Equal-car championship

2,000 simulated 24-race seasons with equal cars, each from one posterior draw: qualifying (stage 1, pace in the current car, with weekend form and session noise) -> grid -> race given the grid (ranking model fitted on real finishing orders) -> retirements (reliability model, equal mechanical risk, driver error rates at the average) -> points. Race stage on held-out finishing orders: vs grid alone +1.826 (95% interval +1.475 to +2.170); vs ratings alone +1.180 (95% interval +0.901 to +1.464).

Entry tests (race stage with vs without the quality, fitted from 2010 up to each held-out season, the quality's own held-out draws for that season):

| quality | held-out seasons | held-out races | result | enters |
|---|---|---|---|---|
| first_lap | 2012–2026 | 306 | +0.049 (95% interval +0.017 to +0.086) | yes |
| race_specific_pace | 2012–2026 | 306 | +0.246 (95% interval +0.156 to +0.341) | yes |
| degradation | 2012–2026 | 306 | -0.002 (95% interval -0.009 to +0.005) | no |
| consistency | 2012–2026 | 306 | +0.030 (95% interval -0.002 to +0.062) | no |
| overtaking_attack |  |  | no held-out draws (outputs/battles/heldout_effects.npz) | no |
| overtaking_defend |  |  | no held-out draws (outputs/battles/heldout_effects.npz) | no |

Qualities in the simulation: qualifying pace, first_lap, race_specific_pace.

| name | points_per_race | p_title | ranks |
|---|---|---|---|
| Max Verstappen | 11.477 | 0.646 | 1–5 |
| Isack Hadjar | 6.804 | 0.035 | 2–16 |
| Carlos Sainz | 6.488 | 0.070 | 1–16 |
| Lando Norris | 6.367 | 0.067 | 1–18 |
| Charles Leclerc | 6.088 | 0.043 | 2–17 |
| Arvid Lindblad | 4.818 | 0.008 | 3–19 |
| Andrea Kimi Antonelli | 4.744 | 0.019 | 2–20 |
| Fernando Alonso | 4.730 | 0.018 | 2–20 |
| Sergio Pérez | 4.690 | 0.024 | 2–21 |
| Oliver Bearman | 4.627 | 0.022 | 2–21 |
| Gabriel Bortoleto | 4.579 | 0.015 | 2–20 |
| Oscar Piastri | 4.512 | 0.010 | 3–21 |

Contribution breakdown (expected points per race lost when a quality is set to the field average, the others kept; and all at once). Conditional model estimates: correlated qualities have no unique allocation, and setting a quality to the average in every draw also removes its uncertainty, which shifts expected points slightly through the convex points scale.

| name | points_per_race | loss_qualifying_pace_at_average | loss_first_lap_at_average | loss_race_specific_pace_at_average | loss_all_at_average |
|---|---|---|---|---|---|
| Max Verstappen | 11.454 | 5.701 | 0.235 | 1.445 | 7.102 |
| Isack Hadjar | 6.925 | 2.992 | -0.027 | -0.309 | 2.745 |
| Carlos Sainz | 6.499 | 2.410 | 0.050 | -0.104 | 2.366 |
| Lando Norris | 6.401 | 2.172 | -0.079 | 0.137 | 2.233 |
| Charles Leclerc | 6.117 | 1.599 | 0.139 | 0.312 | 1.998 |
| Arvid Lindblad | 4.780 | 0.512 | 0.023 | 0.224 | 0.751 |
| Andrea Kimi Antonelli | 4.731 | 1.026 | -0.140 | -0.231 | 0.683 |
| Sergio Pérez | 4.702 | -0.272 | 0.014 | 0.800 | 0.643 |
| Fernando Alonso | 4.687 | 0.602 | 0.071 | 0.008 | 0.690 |
| Oliver Bearman | 4.636 | 0.880 | -0.102 | -0.203 | 0.591 |
| Gabriel Bortoleto | 4.545 | 0.905 | -0.105 | -0.285 | 0.544 |
| Oscar Piastri | 4.537 | 0.768 | 0.030 | -0.335 | 0.488 |
