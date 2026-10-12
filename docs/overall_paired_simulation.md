# Equivalent paired explanatory simulation

The baseline remains the existing complete simulator, with its scenario
validation, retirement failures, point allocation and sporting tie handling.
The explanatory kernel replays the same seed and block address and consumes
the same event random numbers. It does not alter the parameter draws, reference
calendar, field, simulation count, precision rules or intervention definition.

Within each event, weekend deviations, retirement uniforms and race-order
Gumbels are shared by all the paired interventions, as they were when each
intervention restarted the same simulator. Save the random-generator state
immediately before qualifying. Replay that state for each driver's qualifying
intervention with the original qualifying function. Its resulting grid applies
to both that driver's qualifying-only and combined probes. The race-only probe
uses the baseline grid. Keep the original race-strength expression and its
operation order; adding a delta to an already rounded strength is not required
to give an identical result.

Score each intervention through the existing full-field race-order and points
functions and retain only the intervened driver's season points. Explanations
do not consume the intervention's championship ranks or title probability, so
their championship countback is omitted. Baseline countback remains unchanged.
Retirement uniforms and hazards do not depend on the state intervention, so a
baseline all-retired failure cannot become an acceptable explanatory season.

Before connecting this kernel to the summary/controller, compare every probe
and the complete baseline against the original separate simulator calls.
Exercise varying correlated states and observation coefficients, supported
qualifying formats, both saved-array floating precisions, reordered entrants,
no-op interventions, retirement and no-points contexts, and original failure
cases. These are exact equality checks on synthetic arrays, not tests of
scientific validity. Keep source/dependency fingerprints in the connected
summary and frozen controller protocol.

Measure the unchanged fixed resource workload on both source versions and
compare output fingerprints. Preserve both measurements and any failures.
Performance evidence applies to that local block only. The full posterior
workload, ranking precision, scientific acceptance and publication remain open.
