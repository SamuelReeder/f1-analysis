# Equivalent paired explanatory simulation

The baseline remains the existing complete simulator, with its scenario
validation, retirement failures, point allocation and sporting tie handling.
The explanatory kernel replays the same seed and block address and consumes
the same event random numbers. It does not alter the parameter draws, reference
calendar, field, simulation count, precision rules or intervention definition.

Within each event, weekend deviations, retirement uniforms and race-order
Gumbels are shared by all the paired interventions, as they were when each
intervention restarted the same simulator. Record the normal and Student-t
draws while the original qualifying function constructs the baseline grid.
Replay each draw lane's exact random values for every driver's qualifying
intervention with that same qualifying function. Its resulting grid applies
to both that driver's qualifying-only and combined probes. The race-only probe
uses the baseline grid. Keep the original race-strength expression and its
operation order; adding a delta to an already rounded strength is not required
to give an identical result.

Order each intervention with the existing full-field race-order function.
The [own-driver scorer](overall_own_points.md) computes exactly the affected
driver's points under the existing classification and points rules, without
constructing unused classifications for its competitors. Explanations
do not consume the intervention's championship ranks or title probability, so
their championship countback is omitted. Baseline countback remains unchanged.
Retirement uniforms and hazards do not depend on the state intervention, so a
baseline all-retired failure cannot become an acceptable explanatory season.

## Batched interventions

This batching definition precedes the batched implementation and measurement.
It follows the original separate calls and the shared-randomness kernel already
measured in the immutable resource record. It changes neither the registered
stream address nor the existing draw and replicate block sizes.

Within an event, add an intervention axis in sorted driver-identity order. Each
entry on that axis neutralizes only the corresponding driver's state. Flatten
the draw/intervention axes only while calling the original qualifying function,
repeating each lane's parameters and saved random values in the same order.
Check the random method, requested shape and Student-t degrees of freedom on
replay, and require that all recorded requests are consumed. There are no new
random draws or alternative seeds on this intervention axis.

Pass the batched strengths and shared Gumbels to the original race-order
function. With shared completion distances, retain every driver's own points
from its own intervention. The original batched full-field scorer is preserved
as a recorded comparison; the equivalent own-driver specialization removes
unneeded work. Keep season additions in their original event order. Broadcasting
inputs must not mutate them or pair different draw lanes. The independent
replicate loop, baseline season ranks and complete field remain unchanged.

Before adoption, compare the replayed qualifying grids against individual calls
from a saved generator state as well as complete baseline/probe seasons against
the original simulator. Require exact output fingerprints on the unchanged
maximum-size resource workload. Preserve failures; do not change values,
workload or tolerances in response to a disagreement.

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
