# F1 driver skill vs car performance: qualifying-pace ratings

Data through **Azerbaijan Grand Prix 2026** (event 2026-15), window 2010-2026, 15,611 qualifying lap times. Model `quali-v1`.

Ratings are **seconds per 90-second lap relative to the average driver / car at the latest event**; positive = faster. Scope: one-lap qualifying pace only.

## Acceptance gates

| gate | criterion | result |
|---|---|---|
| convergence | R-hat < 1.05 for every rating; no more than 1 divergence per 1,000 draws | PASS |
| synthetic_calibration | clean synthetic data: 90% intervals cover 85-97% of true driver and car values | PASS |
| synthetic_current_grid | clean synthetic data: current-grid rank correlation >= 0.8 and 90% coverage >= 80% | FAIL |
| forecast_vs_baselines | teammate-gap forecasts beat the static two-way model, naive and zero baselines on RMSE, pooled over all cutoffs | PASS |
| forecast_calibration | session-level 90% forecast intervals cover 85-97% | PASS |
| new_pairings | new teammate pairings: model RMSE below the naive and zero baselines | PASS |
| sensitivity | current driver ranking Spearman >= 0.9 against every sensitivity variant | FAIL |

## Drivers (latest event)

`skill` is **portable ability**: what the driver would bring to any car, excluding their fit with the current team. `team_fit` is the estimated driver-team compatibility for the current team (added to skill in that car). Drivers who have only driven for one team cannot separate the two from teammate comparisons, so their skill is pulled towards the average and their intervals are wide.

| name | team | skill | lo90 | hi90 | ranks | p_fastest | team_fit | events | teammates | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| Carlos Sainz | Williams | 0.162 | -0.021 | 0.357 | 1–13 | 0.232 | -0.028 | 241 | 7 |  |
| Max Verstappen | Red Bull | 0.158 | -0.051 | 0.372 | 1–15 | 0.239 | 0.112 | 241 | 8 | sensitive to model choice |
| Charles Leclerc | Ferrari | 0.048 | -0.178 | 0.269 | 1–21 | 0.062 | 0.040 | 187 | 5 |  |
| Lando Norris | McLaren | 0.047 | -0.189 | 0.295 | 1–21 | 0.084 | 0.072 | 166 | 3 | sensitive to model choice |
| Isack Hadjar | Red Bull | 0.047 | -0.104 | 0.217 | 2–18 | 0.022 | 0.114 | 33 | 3 | sensitive to model choice |
| Andrea Kimi Antonelli | Mercedes | 0.046 | -0.123 | 0.220 | 2–18 | 0.029 | 0.001 | 38 | 1 | sensitive to model choice |
| Valtteri Bottas | Cadillac F1 Team | 0.035 | -0.180 | 0.252 | 1–20 | 0.058 | -0.085 | 262 | 6 | sensitive to model choice |
| Fernando Alonso | Aston Martin | 0.035 | -0.193 | 0.267 | 1–20 | 0.069 | -0.016 | 299 | 6 | sensitive to model choice |
| Pierre Gasly | Alpine F1 Team | 0.027 | -0.185 | 0.246 | 1–20 | 0.052 | -0.015 | 189 | 8 | sensitive to model choice |
| Lewis Hamilton | Ferrari | 0.014 | -0.189 | 0.215 | 2–21 | 0.028 | 0.004 | 339 | 5 |  |
| George Russell | Mercedes | 0.003 | -0.213 | 0.213 | 2–21 | 0.026 | -0.023 | 166 | 5 |  |
| Gabriel Bortoleto | Audi | -0.002 | -0.169 | 0.179 | 3–20 | 0.016 | 0.035 | 37 | 1 |  |
| Oliver Bearman | Haas F1 Team | -0.009 | -0.183 | 0.179 | 3–20 | 0.014 | 0.041 | 39 | 3 |  |
| Oscar Piastri | McLaren | -0.011 | -0.200 | 0.190 | 3–21 | 0.018 | 0.045 | 84 | 1 | sensitive to model choice |
| Sergio Pérez | Cadillac F1 Team | -0.019 | -0.227 | 0.192 | 3–21 | 0.020 | -0.030 | 295 | 7 |  |
| Franco Colapinto | Alpine F1 Team | -0.034 | -0.202 | 0.123 | 5–21 | 0.004 | -0.129 | 42 | 2 |  |
| Liam Lawson | RB F1 Team | -0.055 | -0.234 | 0.114 | 5–21 | 0.003 | -0.034 | 49 | 4 |  |
| Alexander Albon | Williams | -0.069 | -0.293 | 0.147 | 4–22 | 0.009 | 0.022 | 139 | 6 |  |
| Arvid Lindblad | RB F1 Team | -0.079 | -0.220 | 0.095 | 6–21 | 0.002 | 0.091 | 15 | 2 | sensitive to model choice |
| Nico Hülkenberg | Audi | -0.083 | -0.289 | 0.117 | 5–22 | 0.006 | -0.029 | 267 | 11 |  |
| Esteban Ocon | Haas F1 Team | -0.085 | -0.295 | 0.119 | 4–22 | 0.007 | -0.099 | 191 | 6 | sensitive to model choice |
| Lance Stroll | Aston Martin | -0.192 | -0.459 | 0.066 | 7–22 | 0.002 | -0.086 | 199 | 7 | sensitive to model choice |

`flag`: rating moves by more than 0.10 s, or rank by 8+ places, across the sensitivity variants below. (The rule set before the results, 0.05 s or 4 places, flagged every driver because it is smaller than the posterior uncertainty, so it was loosened.)

### Drivers in their current car (skill + team fit)

What teammate comparisons measure directly, and what a driver delivers in the car they have now. It is better determined than portable skill: in simulation on the real network it is recovered with rank correlation 0.91, against 0.75.

| in_team_rank | name | team | in_team | lo90 | hi90 |
|---|---|---|---|---|---|
| 1 | Max Verstappen | Red Bull | 0.272 | 0.121 | 0.425 |
| 2 | Isack Hadjar | Red Bull | 0.165 | -0.003 | 0.331 |
| 3 | Carlos Sainz | Williams | 0.136 | -0.049 | 0.321 |
| 4 | Lando Norris | McLaren | 0.122 | -0.097 | 0.346 |
| 5 | Charles Leclerc | Ferrari | 0.089 | -0.098 | 0.270 |
| 6 | Andrea Kimi Antonelli | Mercedes | 0.046 | -0.144 | 0.247 |
| 7 | Oscar Piastri | McLaren | 0.036 | -0.185 | 0.255 |
| 8 | Gabriel Bortoleto | Audi | 0.036 | -0.161 | 0.239 |
| 9 | Oliver Bearman | Haas F1 Team | 0.035 | -0.167 | 0.243 |
| 10 | Fernando Alonso | Aston Martin | 0.022 | -0.188 | 0.235 |
| 11 | Lewis Hamilton | Ferrari | 0.020 | -0.167 | 0.202 |
| 12 | Arvid Lindblad | RB F1 Team | 0.019 | -0.149 | 0.187 |
| 13 | Pierre Gasly | Alpine F1 Team | 0.015 | -0.155 | 0.181 |
| 14 | George Russell | Mercedes | -0.023 | -0.215 | 0.167 |
| 15 | Alexander Albon | Williams | -0.046 | -0.232 | 0.139 |
| 16 | Valtteri Bottas | Cadillac F1 Team | -0.052 | -0.290 | 0.188 |
| 17 | Sergio Pérez | Cadillac F1 Team | -0.053 | -0.283 | 0.192 |
| 18 | Liam Lawson | RB F1 Team | -0.094 | -0.252 | 0.066 |
| 19 | Nico Hülkenberg | Audi | -0.114 | -0.314 | 0.089 |
| 20 | Franco Colapinto | Alpine F1 Team | -0.168 | -0.341 | 0.003 |
| 21 | Esteban Ocon | Haas F1 Team | -0.188 | -0.397 | 0.021 |
| 22 | Lance Stroll | Aston Martin | -0.282 | -0.493 | -0.067 |

## Car packages (latest event, track-neutral)

`track_profile` > 0: relatively stronger on high-speed circuits; < 0: on slow/street circuits.

| constructor | rating | lo90 | hi90 | gap_to_fastest | ranks | track_profile |
|---|---|---|---|---|---|---|
| Mercedes | 0.914 | 0.670 | 1.159 | 0.000 | 1–3 | 0.015 |
| Ferrari | 0.726 | 0.488 | 0.955 | -0.209 | 1–4 | -0.023 |
| McLaren | 0.687 | 0.426 | 0.957 | -0.250 | 1–4 | -0.018 |
| Red Bull | 0.573 | 0.368 | 0.783 | -0.376 | 2–4 | -0.039 |
| Alpine F1 Team | 0.232 | 0.002 | 0.467 | -0.714 | 4–7 | 0.040 |
| RB F1 Team | 0.106 | -0.118 | 0.319 | -0.841 | 5–8 | 0.104 |
| Audi | 0.054 | -0.206 | 0.310 | -0.894 | 5–8 | 0.047 |
| Haas F1 Team | -0.176 | -0.446 | 0.090 | -1.130 | 6–9 | 0.014 |
| Williams | -0.531 | -0.779 | -0.270 | -1.483 | 8–9 | -0.022 |
| Aston Martin | -1.253 | -1.631 | -0.894 | -2.205 | 10–11 | -0.154 |
| Cadillac F1 Team | -1.331 | -1.633 | -1.029 | -2.278 | 10–11 | 0.032 |

## Validation

### Forecasting (25 leave-future-out cutoffs, 2013–2026)

Each cutoff is fitted only on data up to that point (including the circuit factor) and forecasts the next season (end-of-season cutoffs) or the rest of the season (after round 8). Established drivers only (≥10 prior events). RMSE and CRPS are in seconds.

| target | model | static two-way | naive raw gaps | zero |
|---|---|---|---|---|
| teammate gap, per segment | 0.550 | 0.571 | 0.571 | 0.580 |
| teammate gap, per pairing (all, n=209) | 0.188 | 0.246 | 0.262 | 0.265 |
| teammate gap, per pairing (continuing, n=166) | 0.171 | 0.181 | 0.199 | 0.265 |
| teammate gap, per pairing (new, n=43) | 0.244 | 0.408 | 0.425 | 0.264 |

Session-level calibration of teammate-gap forecasts: 50% intervals cover 56.8%, 90% intervals cover 94.0%; CRPS 0.267s. Pairing-level 90% intervals cover 86.6%.

Full relative pace (car + driver) within each forecast segment:

| forecaster | rmse_s | spearman |
|---|---|---|
| model | 0.544 | 0.665 |
| static two-way | 0.564 | 0.636 |
| recent form (last 5 events) | 0.567 | 0.625 |

### Synthetic recovery on the real F1 network

Truth drawn from the model's generative process on the real entries, teams and segments; misspecified scenarios add effects the model does not contain. Coverage = share of true values inside the 90% interval.

| scenario | skill_corr | skill_rmse_s | skill_cov90 | car_corr | car_cov90 | grid_spearman | grid_cov90 | cross_team_cov90 | grid_in_team_spearman | car_share_true/est |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | 0.783 | 0.105 | 0.883 | 0.995 | 0.930 | 0.749 | 0.909 | 0.918 | 0.909 | 0.97/0.97 |
| compat | 0.756 | 0.108 | 0.893 | 0.994 | 0.917 | 0.741 | 0.909 | 0.927 | 0.679 | 0.97/0.97 |
| upgrades | 0.782 | 0.103 | 0.878 | 0.994 | 0.929 | 0.790 | 0.909 | 0.909 | 0.840 | 0.97/0.97 |
| form | 0.709 | 0.119 | 0.897 | 0.993 | 0.921 | 0.739 | 0.955 | 0.955 | 0.814 | 0.97/0.96 |
| transfer_luck | 0.731 | 0.114 | 0.870 | 0.993 | 0.911 | 0.640 | 0.864 | 0.859 | 0.828 | 0.97/0.97 |

### Is driver-team compatibility real? (placebo test)

Compatibility SD 0.161% (90% interval 0.121–0.204) vs placebo SD 0.020% (0.002–0.048) when a driver's stint in one team is split in half. P(placebo > compatibility) = 0.00. Performance shifts when a driver changes team, not with time spent in the same team, so the effect is team-specific.

### Sensitivity of the current driver ranking

| variant | spearman | max_abs_shift_s | max_rank_shift |
|---|---|---|---|
| sens_start2006 | 0.863 | 0.095 | 7 |
| sens_notrack | 0.994 | 0.016 | 2 |
| sens_gausscar | 0.990 | 0.022 | 2 |
| sens_notransient | 0.981 | 0.020 | 3 |
| sens_noform | 0.914 | 0.072 | 7 |
| sens_nocompat | 0.806 | 0.153 | 9 |
| sens_splitt | 0.988 | 0.026 | 2 |
| sens_drift_x2 | 0.910 | 0.146 | 6 |
| sens_drift_half | 0.937 | 0.069 | 6 |

Per-driver rating (s) under each variant:

| driver_id | main | sens_start2006 | sens_notrack | sens_gausscar | sens_notransient | sens_noform | sens_nocompat | sens_splitt | sens_drift_x2 | sens_drift_half | spread_s | rank_range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sainz | 0.162 | 0.170 | 0.153 | 0.159 | 0.152 | 0.203 | 0.127 | 0.150 | 0.187 | 0.106 | 0.097 | 3.000 |
| max_verstappen | 0.158 | 0.212 | 0.174 | 0.177 | 0.140 | 0.192 | 0.283 | 0.174 | 0.214 | 0.123 | 0.160 | 1.000 |
| leclerc | 0.048 | 0.096 | 0.057 | 0.052 | 0.058 | 0.013 | 0.059 | 0.063 | 0.050 | 0.068 | 0.083 | 7.000 |
| norris | 0.047 | 0.103 | 0.045 | 0.040 | 0.044 | 0.052 | 0.173 | 0.028 | 0.093 | 0.046 | 0.145 | 4.000 |
| hadjar | 0.047 | 0.108 | 0.054 | 0.060 | 0.046 | 0.061 | 0.144 | 0.061 | 0.109 | 0.040 | 0.104 | 2.000 |
| antonelli | 0.046 | 0.065 | 0.048 | 0.047 | 0.043 | 0.071 | 0.079 | 0.054 | 0.106 | 0.006 | 0.100 | 4.000 |
| bottas | 0.035 | -0.017 | 0.037 | 0.033 | 0.052 | 0.013 | -0.045 | 0.024 | -0.005 | 0.025 | 0.097 | 11.000 |
| alonso | 0.035 | -0.027 | 0.021 | 0.027 | 0.022 | 0.000 | -0.030 | 0.028 | -0.021 | -0.011 | 0.065 | 8.000 |
| gasly | 0.027 | 0.033 | 0.020 | 0.027 | 0.007 | 0.099 | 0.042 | 0.018 | 0.085 | 0.002 | 0.096 | 8.000 |
| hamilton | 0.014 | 0.011 | 0.007 | 0.005 | 0.009 | 0.030 | -0.005 | 0.012 | 0.017 | -0.010 | 0.040 | 6.000 |
| russell | 0.003 | -0.012 | -0.001 | 0.001 | 0.011 | -0.001 | -0.022 | -0.003 | 0.001 | 0.013 | 0.035 | 6.000 |
| bortoleto | -0.002 | 0.012 | 0.007 | 0.003 | -0.005 | 0.021 | 0.048 | 0.000 | 0.027 | 0.001 | 0.053 | 5.000 |
| bearman | -0.009 | 0.014 | -0.007 | -0.004 | 0.002 | 0.009 | 0.018 | -0.006 | 0.009 | -0.006 | 0.027 | 5.000 |
| piastri | -0.011 | 0.028 | -0.012 | -0.018 | -0.007 | -0.020 | 0.093 | -0.003 | 0.020 | -0.002 | 0.113 | 10.000 |
| perez | -0.019 | -0.073 | -0.022 | -0.024 | -0.037 | -0.013 | -0.052 | -0.034 | -0.031 | -0.038 | 0.060 | 4.000 |
| colapinto | -0.034 | -0.056 | -0.041 | -0.043 | -0.032 | -0.026 | -0.122 | -0.048 | -0.042 | -0.024 | 0.098 | 5.000 |
| lawson | -0.055 | -0.081 | -0.054 | -0.049 | -0.065 | -0.024 | -0.097 | -0.056 | -0.071 | -0.048 | 0.073 | 3.000 |
| albon | -0.069 | -0.056 | -0.064 | -0.069 | -0.049 | -0.098 | -0.078 | -0.051 | -0.103 | -0.012 | 0.091 | 4.000 |
| arvid_lindblad | -0.079 | 0.006 | -0.067 | -0.069 | -0.075 | -0.070 | 0.019 | -0.069 | -0.043 | -0.074 | 0.098 | 11.000 |
| hulkenberg | -0.083 | -0.130 | -0.083 | -0.089 | -0.080 | -0.119 | -0.112 | -0.070 | -0.120 | -0.057 | 0.074 | 2.000 |
| ocon | -0.085 | -0.117 | -0.095 | -0.100 | -0.067 | -0.129 | -0.198 | -0.112 | -0.160 | -0.038 | 0.159 | 3.000 |
| stroll | -0.192 | -0.287 | -0.205 | -0.215 | -0.182 | -0.262 | -0.346 | -0.203 | -0.338 | -0.123 | 0.223 | 0.000 |

### Data window: 2010 vs 2006 start

Same model fitted from 2006 (adding 2006–09 Q1/Q2 low-fuel times) vs from 2010, scored on identical forecast targets at 6 cutoffs (end2016, end2019, end2022, end2024, end2025, mid2025); 47 matched pairings.

| window | pairing_rmse_s | pairing_new_rmse_s | pairing_in90 | session_rmse_s | session_crps_s | pace_rmse_s | pace_spearman |
|---|---|---|---|---|---|---|---|
| 2010 | 0.207 | 0.273 | 0.872 | 0.503 | 0.254 | 0.512 | 0.595 |
| 2006 | 0.207 | 0.271 | 0.872 | 0.502 | 0.254 | 0.516 | 0.595 |

The two windows forecast equally well, so the data cannot say which is right. The pre-specified 2010 window (clean low-fuel qualifying) is kept. The rating shifts between windows (at most 0.1 s, see the sensitivity table) are genuine uncertainty and are well inside the drivers' 90% intervals.

### Convergence

6000 posterior draws; divergences: 0; max R-hat skill 1.021, car 1.013, hyperparameters 1.041; min ESS skill 287, car 367.

## What the ratings can and cannot say

- **Car-package ratings are well determined.** Synthetic recovery correlation ≥ 0.99 in every scenario, with calibrated intervals. Car differences dominate qualifying pace: about 97% of the variance between entries, recovered without bias.
- **Teammate comparisons and forecasts are reliable.** The model beats a static two-way model, raw teammate gaps and a zero-gap baseline on every forecast target, including brand-new pairings, with calibrated intervals.
- **Portable driver skill is only moderately determined.** Performance relative to teammates has a large team-specific part: driver-team fit with SD about 0.16%, against about 0.10% for differences in portable ability. That limits how well the network can rank the current grid (synthetic rank correlation 0.75). Intervals remain honest (coverage about 90%), so read ranks as ranges, not positions.
- **Drivers who have only raced for one team** (e.g. Piastri, Antonelli, Bortoleto) have portable skill that teammate data cannot separate from team fit. Their ratings lean on the prior and on their teammate's links elsewhere.
- **The main structural sensitivities** are whether team fit is modelled and how much history is used. Details of the car model and noise model barely matter (rank correlation ≥ 0.98).
- **Main residual risk:** drivers who change team after an unusually lucky or unlucky season. In simulation this is the violation that degrades the ranking most.
- **Scope:** one-lap qualifying pace only. Race pace, tyre management, starts and racecraft are not measured yet.
