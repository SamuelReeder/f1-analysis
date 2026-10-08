# Authorized archive computation follow-up

The user approved the separate computational follow-up proposed in
`docs/championship_followup.md` and said an initial result need not be perfect.
This authorizes the documented change to computation after the exhausted archive
fit. It does not turn that failed study into a pass or authorize a GitHub push,
merge, or release. The previous archive registration and failure remain intact.

The candidate, model likelihood and priors, archive dates, preprocessing, training
cutoffs, baselines, convergence thresholds, and predictive gate are exactly those
of the original archive study. This remains an earlier-era transport test, with
the limitations and prior-use audit in `docs/championship_archive.md`.

The only sampler change is the final stage of the existing `artifacts.RETRY`
recipe applied to `racemulti.fit` defaults: 1,500 warm-up iterations, 2,000 samples
per chain, four chains, target acceptance 0.98, seed 3. Use this recipe once for
each archive race-pace fold; no additional retries or seed search. Initialization
continues to use the existing prior-median rule. The wrapper calls the existing
`racemulti.arrays` and `racemulti.model` directly, preserving their likelihood and
priors. Qualifying and race-stage fits keep their original registered rules.

Verify every qualifying and pace training fold before scoring any race outcomes.
An exhausted fit terminates this new study, without removing a season. If all
training fits pass, evaluate the complete fixed comparison once. Do not inspect
partial scores to modify the study. Record all attempts, including the preserved
failure that motivated this explicitly authorized computational change.

The already prepared lap and weather data are frozen, hashed inputs. Qualifying
checkpoints with the identical training design may be copied and verified into a
separate cache namespace. Every new checkpoint lives under
`outputs/fits/championship_archive_conservative/`. Research evidence lives under
`outputs/analysis/championship_archive_conservative/`. Canonical model sources,
racing outputs, and dashboard data are untouched by this experiment.

Before fitting, commit this plan and its analysis module, then generate and commit
the immutable `docs/championship_archive_conservative_protocol.json`. The
registration fixes sources, inputs, sampler settings, and the previous failure.
Operational interruptions can resume only under that identical registration with
matching caches; statistical failure cannot resume. Record a pass, failure, or
inconclusive result without promising success. A pass supports the fixed model on
this earlier archive; publication integration must retain that scope and the
original failed prediction test. This module never switches on a ranking itself.
