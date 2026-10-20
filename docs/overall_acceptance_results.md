# Independent convergence evaluator verification

The [acceptance evaluator](overall_acceptance.md) independently recomputes the
complete final-fit parameter and current-driver contrast diagnostics. It derives
the retained-field inventory from the actual model and checks the saved archive
against it. A receipt's success flag cannot override a failed calculation or hide
a missing field. This implements convergence evidence only; it does not accept
the scientific model or authorize publication.

The [committed verification record](../outputs/championship/development/overall-acceptance-convergence-checks-v1/verification.json)
records **96 passing focused checks**. They compare the inventory with the actual
NumPyro collector, exercise matching and contradictory numerical reports, and
reject incomplete fields or inputs changed during evaluation. Driver contrasts
can fail even when individual parameter diagnostics pass; that failure is retained.
Connected bounded-diagnostic, archive and candidate-simulation checks also pass.

These tests use invented traces and fake samplers. Fixed prior draws supply
shapes and dtypes only. No MCMC or real-data acceptance decision was run. Earlier
fit attempts remain in the required sequence and their sampler extras remain
checked, but this evaluator independently recomputes parameter reports only for
the final attempt. Functional expected-point and rank precision remain separate.

Pytest reports **132.94 seconds**. The external resource log separately reports
**2:01.34** elapsed and **726,068 KiB** peak resident memory. These clock readings
are preserved separately. The active scope record confirms the configured local
**3G** memory and **200%** CPU limits. These software-check measurements do not
establish full-fit resources on the intended runner.

The failed full-network profile remains failed. Complete scientific definitions,
recovery and calibration, historical source availability, predictive comparisons,
independent prospective evidence and functional precision are still required.
Accepted canonical dashboard production, full refresh integration and approved
rollout remain incomplete under the [full plan](overall_model_plan.md).
