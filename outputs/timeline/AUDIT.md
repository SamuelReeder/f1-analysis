# Race event timeline: audit (timeline-v1)

8,141 rows over 341 races (outcomes from 2010, race-control and lap evidence from 2018). Built by `python -m f1rank.timeline` from `data/processed/race_*.parquet` (FastF1) and Jolpica results. Rules produce evidence, not verdicts.

## Rows by kind and season

| kind | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| did_not_start | 4 | 1 | 1 | 1 | 1 | 4 | 3 | 1 | 0 | 0 | 2 | 2 | 0 | 3 | 3 | 3 | 7 |
| disqualified | 0 | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 3 | 2 | 0 | 1 | 0 | 2 | 2 | 6 | 0 |
| incident_noted | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 149 | 226 | 144 | 127 | 184 | 170 | 210 | 265 | 209 |
| off_track | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 25 | 41 | 249 | 227 | 324 | 440 | 349 | 325 | 278 |
| penalty | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 29 | 27 | 41 | 44 | 72 | 97 | 65 | 105 | 74 |
| pit_anomaly | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 42 | 53 | 36 | 42 | 48 | 67 | 53 | 73 | 46 |
| possible_team_order | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 25 | 36 | 25 | 36 | 38 | 32 | 39 | 50 | 41 |
| red_flag | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 8 | 4 | 7 | 3 | 1 | 3 |
| retirement | 114 | 76 | 87 | 63 | 82 | 73 | 79 | 94 | 82 | 58 | 54 | 55 | 73 | 60 | 49 | 51 | 64 |
| safety_car | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 16 | 20 | 15 | 21 | 18 | 14 | 18 | 11 |
| stoppage | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 25 | 15 | 3 | 7 | 9 | 0 | 5 | 0 | 0 |
| suspected_damage | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 19 | 14 | 3 | 23 | 18 | 17 | 14 | 4 |
| vsc | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 9 | 7 | 12 | 22 | 10 | 9 | 13 | 19 |
| weather_change | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 25 | 11 | 10 | 26 | 36 | 24 | 28 | 43 | 30 |
| withdrew | 0 | 1 | 1 | 0 | 2 | 3 | 2 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| yellow | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 102 | 91 | 71 | 84 | 113 | 96 | 86 | 117 | 109 |

## Retirement causes

1214 retirements; cause basis: status code 939, inferred from evidence 228, class base rates 47; reviewed: 0.

Mean cause probabilities by basis:

| basis | p_mechanical | p_own_error | p_other_driver | p_external | p_team_ops | p_unknown |
|---|---|---|---|---|---|---|
| class base rates | 0.479 | 0.303 | 0.018 | 0.041 | 0.000 | 0.159 |
| inferred from evidence | 0.505 | 0.228 | 0.065 | 0.024 | 0.000 | 0.178 |
| status code | 0.505 | 0.159 | 0.094 | 0.020 | 0.020 | 0.201 |

### Cause model (status "Retired")

Multinomial logistic model of the class (mechanical, incident, other) given evidence (lap1, contact_msg, multi_car, neutralised, in_pits, slow_before), fitted on 315 retirements with coded statuses (mechanical 151, incident 147, other 17). Leave-one-season-out: log loss 0.761 vs 0.879 for the training base rates; accuracy 0.65 vs 0.43. It is applied to retirements coded only as "Retired" (almost all from 2023), whose evidence rates are compared below: the model assumes the evidence means the same in those seasons.

| held-out season | n | log_loss_model | log_loss_base_rates |
|---|---|---|---|
| 2018 | 77 | 0.873 | 0.997 |
| 2019 | 55 | 0.889 | 0.867 |
| 2020 | 52 | 0.732 | 0.918 |
| 2021 | 52 | 0.637 | 0.846 |
| 2022 | 73 | 0.656 | 0.769 |
| 2023 | 6 | 0.776 | 0.768 |

Calibration of P(mechanical), held-out seasons:

| bin | n | mean_predicted | observed |
|---|---|---|---|
| (-0.001, 0.2] | 49 | 0.092 | 0.061 |
| (0.2, 0.4] | 65 | 0.299 | 0.292 |
| (0.4, 0.6] | 77 | 0.499 | 0.584 |
| (0.6, 0.8] | 124 | 0.707 | 0.677 |

Evidence rates among retirements, by season:

| season | lap1 | contact_msg | multi_car | neutralised | in_pits | slow_before |
|---|---|---|---|---|---|---|
| 2018 | 0.190 | 0.051 | 0.405 | 0.405 | 0.405 | 0.152 |
| 2019 | 0.034 | 0.052 | 0.362 | 0.466 | 0.534 | 0.121 |
| 2020 | 0.130 | 0.093 | 0.389 | 0.556 | 0.426 | 0.167 |
| 2021 | 0.182 | 0.018 | 0.364 | 0.455 | 0.545 | 0.145 |
| 2022 | 0.137 | 0.151 | 0.370 | 0.534 | 0.438 | 0.055 |
| 2023 | 0.100 | 0.100 | 0.233 | 0.367 | 0.517 | 0.133 |
| 2024 | 0.224 | 0.163 | 0.408 | 0.531 | 0.469 | 0.163 |
| 2025 | 0.216 | 0.059 | 0.275 | 0.588 | 0.412 | 0.137 |
| 2026 | 0.047 | 0.047 | 0.219 | 0.484 | 0.547 | 0.188 |

### Attribution of the incident class (assumptions, not estimates)

| evidence | own_error | external | unknown | other_driver |
|---|---|---|---|---|
| single_car | 0.700 | 0.100 | 0.200 | 0.000 |
| penalised | 0.700 | 0.000 | 0.200 | 0.100 |
| other_penalised | 0.100 | 0.000 | 0.200 | 0.700 |
| multi_car | 0.300 | 0.000 | 0.400 | 0.300 |

`penalised`: a collision-related penalty for the car in race control's messages from two laps before the retirement on. Penalties decided after the race are not in the messages, so their cases fall under `multi_car`.

## Changes since the previous build

```
{
 "previous_build": "timeline-v1",
 "added": 0,
 "removed": 0,
 "cause_changed_by_0.05_plus": []
}
```
