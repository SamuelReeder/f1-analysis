# Reviewing the interrupted profile's continuation

The [reviewer](../analysis/overall_continuation_review.py) is prepared for use
after the separately registered continuation stops. Its software tests and
execution against terminal evidence remain pending while the fit occupies the
memory budget. It does not change the running fit or its registration.

Before using it, preserve and commit the continuation's raw results, generated
truth, wrapper resource output and a copy of its exit-status file. Preserve its
log and cache metadata as terminal execution evidence, including exceptions and
partial archives. Large posterior archives remain in the ignored cache.
The reviewer requires the original service to be authoritatively terminal and
runs in a separate capped scope with no populated sibling fitting jobs.

It revalidates the frozen continuation inputs, original interruption review and
completed saved-array audit. Earlier cases retain their original results and
completion records. The interrupted case retains its missing original completion
and historical running receipt; its continuation is recorded separately. Every
original failed attempt remains visible. Settings, seed order, original scientific
receipt fields, historical metadata bytes, saved archive hashes and wrapper exit
must agree. A failure after sampling still withholds combined completion even if
the sampler produced a converged receipt.

Only newly recorded attempts need their full saved arrays and coordinate and
driver-contrast diagnostics reproduced again. Reuse of earlier audits requires
unchanged bound inputs and archive bytes. The verifier records every new attempt,
including failed convergence, missing archives, exceptions and unequal reports.
It uses the original storage inventory and numerical checkers without revised
tolerances or acceptance rules. Recorded divergence counts are inputs to report
reproduction; sampler extra fields were not saved and cannot be independently
recomputed.

Audit execution success, combined recorded convergence and independently checked
combined computational evidence are separate fields. Even when those checks all
succeed, the original uninterrupted profile remains incomplete. Full recovery,
simulation-based calibration, predictive validation and ranking publication still
need their own required evidence. The reviewer never writes an original summary,
changes an acceptance rule or updates the public ranking.
