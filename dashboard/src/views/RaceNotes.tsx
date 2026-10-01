import { Badge } from "../components";
import { pct } from "../lib";
import type {
  RacePace as RaceData,
  RaceValidation as Validation,
} from "../types";

export function RaceValidation({
  validation,
  driver,
}: {
  validation: Validation;
  driver: boolean;
}) {
  return (
    <section className="panel race-validation">
      <div className="panel-heading">
        <div>
          <h2>Held-out prediction</h2>
          <p>
            {validation.n_races} later races · refitted model without{" "}
            {driver ? "driver" : "car"} ratings as the baseline
          </p>
        </div>
        <Badge tone={validation.passed ? "green" : "amber"}>
          {validation.passed ? "Passed" : "Not established"}
        </Badge>
      </div>
      <dl className="race-validation-values">
        <div>
          <dt>Prediction error</dt>
          <dd>
            {(validation.rmse * 0.9).toFixed(3)}s{" "}
            <small>
              vs {(validation.baseline_rmse * 0.9).toFixed(3)}s baseline
            </small>
          </dd>
        </div>
        <div>
          <dt>90% interval coverage</dt>
          <dd>
            {pct(validation.coverage90)} <small>target 85–95%</small>
          </dd>
        </div>
        <div>
          <dt>Interval width</dt>
          <dd>
            {(validation.mean_interval_width * 0.9).toFixed(3)}s{" "}
            <small>
              vs {(validation.baseline_interval_width * 0.9).toFixed(3)}s
              baseline
            </small>
          </dd>
        </div>
      </dl>
      <div className="table-foot">
        <span>Seconds per 90-second lap. Lower error is better.</span>
        <span>
          {validation.n_predictions}{" "}
          {driver ? "teammate comparisons" : "team-race predictions"}
        </span>
      </div>
    </section>
  );
}

export function RaceMethodology({ data }: { data?: RaceData | null }) {
  return (
    <section className="panel race-method">
      <details>
        <summary>Race-pace methodology</summary>
        <p>
          The model fits clean laps from dry races since 2018. It estimates a
          lasting driver contribution and a season-specific change in form, plus
          a separate car contribution for each team and season. Teammates share
          the car term; drivers changing teams connect the comparisons.
        </p>
        <p>
          The table reports total race pace at a reference tyre age of 10 laps.
          It is estimated independently of qualifying. Driver ratings combine
          lasting pace and current-season form. Car ratings describe the current
          season across its sampled circuits. Neither is a prediction of
          finishing position or championship points.
        </p>
        <p>
          Lap number accounts for the common fuel and track trend. Compound,
          tyre age, traffic and whether a driver is unpressured also enter the
          model. Separate race-day deviations prevent a single weekend from
          being treated as lasting ability. A Student-t error model allows
          unusually slow laps, with correlation between consecutive laps of a
          stint.
        </p>
        <p>
          Wet races, pit laps, neutralised laps, unreliable timing and
          incident-affected windows are excluded. Mechanical retirements leave
          their earlier usable laps in the fit. Each entry needs at least two
          usable races in the current season to appear in the table.
        </p>
        <p>
          Validation trains through round 10 in 2024 and 2025, and round 7 in
          2026, then predicts later dry races in each season. The comparison
          models are fitted again with only the driver ratings or only the car
          ratings removed. Publication requires lower prediction error with a
          95% interval below zero, 85–95% coverage of the 90% prediction
          intervals, and narrower intervals than the baseline. Error uncertainty
          is resampled by race.
        </p>
        <p>
          Driver testing uses teammate pace gaps. Car testing uses observed
          team-average pace, without subtracting this model’s driver estimates
          from the target. Shared uncertainty in the per-race estimates is
          retained by resampling whole bootstrap draws across drivers and teams.
        </p>
        <p>
          The 90% pace and rank intervals come from joint posterior draws.
          Driver and car estimates are centred separately on the currently rated
          field. The displayed rank orders the medians; wide rank intervals
          indicate that the data do not establish a precise order.
        </p>
        <p>
          Fuel loads, strategy, team priority and how hard a driver pushes are
          only partly observed. Persistent differences can be attributed to a
          driver or car by the model. These are conditional performance
          estimates, not an exact measurement of innate skill. Reliability, pit
          stops, starts and overtaking are outside this metric.
        </p>
        {data && (
          <p className="mono">
            Fit {data.fit_id} · {data.n_laps.toLocaleString()} clean laps ·{" "}
            {data.n_races} dry races · R-hat{" "}
            {data.diagnostics.rhat_max.toFixed(3)}
          </p>
        )}
      </details>
    </section>
  );
}
