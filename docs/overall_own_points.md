# Exact own-driver scoring for explanatory probes

This computational definition precedes implementation and measurement. The
baseline continues through the complete existing simulator, including its
classification, points and championship countback. An explanatory intervention
needs only the affected driver's points, so it need not compute the other
drivers' classifications or point totals.

The proposed specialization accepts the native race-order generator's complete
crossing-order permutation for every draw and intervened driver. Distances are
the existing shared completed-lap arrays. Validate axes, integer distances and
complete permutations before use. A repeated or missing competitor must be
rejected. Require the existing full-distance-finisher assumption; this does not
provide a shortened-race or lapping model.

For the affected driver, its classification position is one plus the number of
competitors with more completed laps plus the number with the same completed
laps that precede it in crossing order. These sets are disjoint. Locate its
crossing position and gather distances into crossing order; no other driver's
classification position is required. A strict crossing-order permutation and
distance comparisons cannot create a classification dead heat, so the existing
dead-heat sharing rule reduces to the points for the occupied position.

Eligibility still uses the existing scorer's integer classification numerator
and denominator applied to the field's maximum completed distance. Use the
original `gp_points_table` with the unchanged reference scheduled/leader distance
and explicit consecutive-racing-lap assumption. Return zero for an ineligible
driver or a position outside that table. Do not copy or modify the points values
or substitute an assumed race-control condition.

This function is specific to the simulator's complete, strict generated orders.
It is not a replacement for general race adjudication or scoring of supplied
dead heats, penalties, disqualifications or partial classifications. The original
general scorer remains authoritative. The helper's accepted input domain must
be explicit and checked; callers cannot supply arbitrary point tables.

Before connecting it, compare its output exactly with the original full-field
scorer's corresponding diagonal for exhaustive small-field distance/order cases
crossing classification thresholds. Also cover full-size generated fields,
no-points conditions, invalid inventories and unsupported all-retired outcomes.
Then rerun the complete paired-simulator equivalence and controller checks and
the unchanged maximum-size resource block. Bind this helper and definition in
the summary/controller source identities. Preserve all measurements and
failures. No model, random stream, workload, precision bound or scientific
acceptance rule changes.
