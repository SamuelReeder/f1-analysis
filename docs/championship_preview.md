# Provisional equal-car preview

The user asked for overall driver results and said the first result need not be
perfect. Provide a separate exploratory simulation from the already converged
current-data fits. This does not achieve the still-unresolved goal of reliable
predictive improvement and does not alter any validation result or enable the
production Overall view.

Use the already fixed candidate: current qualifying pace plus career-level race
pace beyond qualifying. Reuse the current, provenance-checked qualifying and full
race-pace posteriors. Fit the race-stage coefficients on the original covered
training seasons with the championship's existing full-fit settings and retry
ladder. Require convergence before making estimates; the failed archive draws
cannot be used. This is a descriptive current-data fit, not another validation
attempt or an amendment to the failed archive registrations.

Reuse the championship simulation function and its existing assumptions: equal
average cars and mechanical risk, qualifying pace generates the grid, gated driver
error effects only, and current driver-team compatibility retained in the headline.
Keep the fixed race-pace contribution even though it has not passed the combined
publication test, and state that explicitly. No result from this preview can turn
a failed validation gate into a pass.

Use the existing simulation length, scoring system, retirement-distance assumption,
and reported season-rank percentiles from `f1rank/championship.py`. Fix the simulation
random seed to the same starting seed used there. Give a driver without race-pace
draws the current field mean, then center the field as the original code does.
Record the input hashes, coefficient-fit diagnostics, and every failed validation
reference. The preview must display its unvalidated status alongside the estimates.

Write only under `outputs/analysis/championship_preview/`, with large posterior
checkpoints under the ignored `outputs/fits/championship_preview/`. Do not replace
canonical championship outputs, edit dashboard payloads, or publish to GitHub.
The HTML and CSV are a local research preview, not a production release. They show
conditional model estimates and do not establish a universal best driver.
