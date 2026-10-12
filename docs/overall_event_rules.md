# Overall Grand Prix scoring

The [frozen rule record](overall_event_rules.json) identifies the source documents,
their downloaded hashes and the executable constants in
[scoring.py](../f1rank/overall/scoring.py). This is a scoring specification for
supplied outcomes, not a fitted model or publication approval.

The [FIA points rules](https://api.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_a_general_provisions_-_iss_03_-_2026-06-25.pdf)
select a distance-based points column only after the leader meets the consecutive
unneutralised-lap requirement. It applies regardless of the reason the event ends.
Dead heats share points from the occupied places. Championship countback compares
race results and then qualifying results. The
[classification rules](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_09_-_2026-10-01.pdf)
order cars by completed distance and line crossing, with a rounded-down minimum
distance for classification. Retired cars can remain classified.

The scorer requires scheduled laps, the leader's end-signal distance and the
longest uninterrupted run of complete laps without safety-car procedures. Final
classified lap counts are separate from the leader's end-signal distance: an
adjudicated countback must not silently change the points column. A field-wide
neutralisation indicator does not determine consecutive racing laps. Missing,
impossible and misaligned metadata raise errors.

The field supplied to the scorer already reflects official exclusions and
penalties. It does not determine those decisions. Dead heats use competition
ranks and cannot acquire a driver-identity advantage. The simulator's continuous
crossing draws still assign no probability to exact dead heats. Residual
championship ties expose rank intervals and shared title mass; this is the
model's representation of an unresolved tie, not an invented sporting tiebreak.

The nested simulator calls this scorer with an explicit race-control assumption
in each scenario. It still refuses all-retired draws without a full-distance
finisher: a points calculator does not supply the missing race-stoppage model.
Its retirement model also does not yet generate lapped running cars or correlated
incidents. The frozen reference calendar is unchanged and cannot supply unknown
race-control inputs. These remain requirements for coherent calibrated simulation.

The known-outcome fixtures in
[test_overall_event_rules.py](../tests/test_overall_event_rules.py) cover distance
boundaries, insufficient racing laps, classification/countback distance,
classified retirees, dead heats across the points cutoff, field permutations,
championship countback and invalid inputs. Software fixtures are distinct from
posterior recovery, real-data calibration and evidence of driver ability.
