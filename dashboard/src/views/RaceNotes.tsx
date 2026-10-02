import { Badge } from "../components";
import { date, pct, signed } from "../lib";
import { Equation, InlineMath, RatingMath } from "../ModelMath";
import type {
  CarState,
  FeatureTest,
  RacePace as RaceData,
  RaceValidation as Validation,
} from "../types";

const source = "https://github.com/SamuelReeder/f1-analysis/blob/main/";

export function RaceValidation({
  validation,
  driver,
  title = "Held-out prediction",
  note,
}: {
  validation: Validation;
  driver: boolean;
  title?: string;
  note?: React.ReactNode;
}) {
  return (
    <section className="panel race-validation">
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          <p>
            {note ?? (
              <>
                {validation.n_races} later races · refitted model without{" "}
                {driver ? "driver" : "car"} ratings as the baseline
              </>
            )}
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
        <Equation
          label="Race pace transformation"
          tex={String.raw`\begin{aligned}
            y_i &= -100\ln\!\left(\frac{t_i}{\widetilde t_r}\right) \\
            L_i &= \frac{\operatorname{lap}_i-\overline{\operatorname{lap}}_r}{10} \\
            A_i &= \operatorname{tyre\ age}_i-10
          \end{aligned}`}
        />
        <p>
          <InlineMath tex="t_i" /> is lap time and{" "}
          <InlineMath tex={String.raw`\widetilde t_r`} /> is the median filtered
          clean lap for race <InlineMath tex="r" />. Positive pace means faster.{" "}
          <InlineMath tex="L_i" /> measures lap number in units of ten laps,
          centred within the race; <InlineMath tex="A_i" /> measures tyre age
          relative to the 10-lap reference. Indices <InlineMath tex="d" />,{" "}
          <InlineMath tex="j" />, <InlineMath tex="s" /> and{" "}
          <InlineMath tex="c" /> identify driver, team lineage, season and
          compound.
        </p>
        <p>
          The table reports total race pace at a reference tyre age of 10 laps.
          It is estimated independently of qualifying. Driver ratings combine
          lasting pace and current-season form. Car ratings describe the current
          season across its sampled circuits. Neither is a prediction of
          finishing position or championship points.
        </p>
        <Equation
          label="Total race-pace components"
          tex={String.raw`\begin{aligned}
            D_{ds}&=\operatorname{centre}_s(S_d+f_{ds}) \\
            C_{js}&=(B_s\gamma_s)_j
          \end{aligned}`}
        />
        <p>
          <InlineMath tex="S_d" /> is lasting driver pace and{" "}
          <InlineMath tex="f_{ds}" /> is season form. The centring operator
          subtracts the mean across modelled driver-season entries in that
          season. <InlineMath tex="B_s" /> is an orthonormal zero-sum contrast
          matrix for that season’s teams, with learned coordinates{" "}
          <InlineMath tex={String.raw`\gamma_s`} />. It constrains car
          contributions to sum to zero. The published tables then centre the
          driver and car draws on their respective currently eligible fields.
        </p>
        <p>
          Lap number accounts for the common fuel and track trend. Compound,
          tyre age, traffic and whether a driver is unpressured also enter the
          model. Separate race-day deviations prevent a single weekend from
          being treated as lasting ability. A Student-t error model allows
          unusually slow laps, with correlation between consecutive laps of a
          stint.
        </p>
        <Equation
          label="Clean-lap race model"
          tex={String.raw`\begin{aligned}
            \mu_i &= \alpha_r+\beta_r L_i+D_{ds}+C_{js} \\
                  &\quad+\chi_{rc}+(\delta_{rc}+v_{dr})A_i \\
                  &\quad+\boldsymbol\tau_r^\mathsf{T}\mathbf H_i+u_{dr}+w_{jr}
          \end{aligned}`}
        />
        <dl className="math-terms">
          <div>
            <dt>
              <InlineMath tex={String.raw`\alpha_r+\beta_r L_i`} />
            </dt>
            <dd>Race baseline and common lap-number trend.</dd>
          </div>
          <div>
            <dt>
              <InlineMath tex={String.raw`\chi_{rc}`} />
            </dt>
            <dd>
              Compound offset; zero for the most-used compound in that race.
            </dd>
          </div>
          <div>
            <dt>
              <InlineMath tex={String.raw`\delta_{rc}`} />
            </dt>
            <dd>Race- and compound-specific tyre-age slope.</dd>
          </div>
          <div>
            <dt>
              <InlineMath tex="v_{dr}" />
            </dt>
            <dd>
              Driver’s extra tyre-age slope for this race, centred across
              drivers.
            </dd>
          </div>
          <div>
            <dt>
              <InlineMath tex={String.raw`\mathbf H_i`} />
            </dt>
            <dd>
              Indicators for a car ahead within 1s, from 1–2s, and no recorded
              car within 5s ahead or behind.
            </dd>
          </div>
          <div>
            <dt>
              <InlineMath tex={String.raw`\boldsymbol\tau_r`} />
            </dt>
            <dd>
              Race-specific coefficients for those three traffic indicators.
            </dd>
          </div>
          <div>
            <dt>
              <InlineMath tex="u_{dr},\ w_{jr}" />
            </dt>
            <dd>
              Driver-day and shared car-day deviations, each centred within the
              race.
            </dd>
          </div>
        </dl>
        <Equation
          label="Correlated lap residuals"
          tex={String.raw`\begin{aligned}
            \varepsilon_i &= y_i-\mu_i \\
            \varepsilon_i\mid\varepsilon_{i-1} &\sim t_\nu(\rho\varepsilon_{i-1},\sigma) \\
            \varepsilon_i &\sim t_\nu\!\left(0,\frac{\sigma}{\sqrt{1-\rho^2}}\right)\quad\text{at a reset}
          \end{aligned}`}
        />
        <p>
          Consecutive laps are linked only within the same driver’s stint and
          race, with no missing lap number. A new stint or a gap resets the
          residual using the displayed initial scale.{" "}
          <InlineMath tex={String.raw`\rho`} /> measures correlation;{" "}
          <InlineMath tex={String.raw`\nu`} /> controls tail weight. Student-t’s
          second argument is its location and its third is its scale, not its
          variance. Race-day effects and tyre-age slopes are excluded from the
          headline rating at the reference tyre age.
        </p>
        <details className="method-priors">
          <summary>Race model priors and estimation</summary>
          <Equation
            label="Driver, car and residual priors"
            tex={String.raw`\begin{aligned}
              S_d^{\rm raw} &\sim\mathcal N(0,0.5^2) \\
              S_d &= S_d^{\rm raw}-\overline S^{\rm raw} \\
              f_{ds} &= \operatorname{centre}_s(\sigma_f z_{ds}) \\
              z_{ds} &\sim\mathcal N(0,1) \\
              \sigma_f &\sim\operatorname{HalfNormal}(0.2) \\
              \gamma_s &\sim\mathcal N(\mathbf0,1.5^2 I) \\
              \rho &\sim\operatorname{Uniform}(-0.5,0.95) \\
              \sigma &\sim\operatorname{HalfNormal}(1) \\
              \nu &\sim\operatorname{Gamma}(2,0.1)
            \end{aligned}`}
          />
          <p>
            Normal distributions are written as mean and variance; Gamma uses
            shape and rate. Baseline, lap trend, compound offset, tyre slope and
            traffic coefficients have zero-mean normal priors with standard
            deviations 3, 2, 3, 0.3 and 1 respectively, in percentage-point pace
            units per indicated covariate unit. Driver-day, car-day and extra
            tyre-slope effects use centred normal draws with learned scales
            drawn from HalfNormal(0.3), HalfNormal(0.4) and HalfNormal(0.05).
          </p>
          <Equation
            label="Joint Bayesian fit"
            tex={String.raw`p(\theta\mid y)\propto p(\theta)\prod_i p(\varepsilon_i\mid\varepsilon_{i-1},\theta)`}
          />
          <p>
            The product uses the reset distribution where laps are not linked.
            NumPyro’s No-U-Turn Sampler estimates the joint posterior in four
            chains. The prior scales regularise driver and car estimates with
            little evidence; they do not resolve every ambiguity between them.
          </p>
        </details>
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
        <Equation
          label="Prediction error and improvement"
          tex={String.raw`\begin{aligned}
            \operatorname{RMSE} &= \sqrt{\frac{1}{N}\sum_{k=1}^{N}(o_k-\widehat p_k)^2} \\
            d_r &= \frac{1}{n_r}\sum_{k\in r}\bigl[(o_k-\widehat p_{k,\rm full})^2 \\
                &\hspace{3em}{}-(o_k-\widehat p_{k,\rm base})^2\bigr] \\
            \Delta_{\rm MSE} &= \frac{1}{R}\sum_{r=1}^{R}d_r
          \end{aligned}`}
        />
        <p>
          <InlineMath tex="o_k" /> is the held-out observed pace estimate and{" "}
          <InlineMath tex={String.raw`\widehat p_k`} /> is its posterior-mean
          prediction. <InlineMath tex="N" /> counts predictions,{" "}
          <InlineMath tex="R" /> counts races and <InlineMath tex="n_r" />{" "}
          counts predictions in race <InlineMath tex="r" />. RMSE weights
          predictions equally; improvement weights races equally. The test
          resamples whole races 4,000 times to form the 95% interval for{" "}
          <InlineMath tex={String.raw`\Delta_{\rm MSE}`} />. A negative
          difference favours the full model. Calculations use percentage-point
          pace; displayed RMSE and interval widths are multiplied by 0.9 to give
          reference-lap seconds.
        </p>
        <Equation
          label="Prediction calibration and publication gate"
          tex={String.raw`\begin{aligned}
            C_{90} &= \frac{1}{N}\sum_k\mathbf1\{l_k\le o_k\le h_k\} \\
            W_{90} &= \frac{1}{N}\sum_k(h_k-l_k) \\
            \text{publish if }&R\ge12 \\
                             &Q_{0.975}(\Delta_{\rm MSE}^{*})<0 \\
                             &0.85\le C_{90}\le0.95 \\
                             &W_{90,\rm full}<W_{90,\rm base}
          \end{aligned}`}
        />
        <p>
          <InlineMath tex="l_k" /> and <InlineMath tex="h_k" /> bound the 90%
          predictive interval, including future race-day variation and
          uncertainty in the observed target. <InlineMath tex="C_{90}" /> is its
          observed coverage and <InlineMath tex="W_{90}" /> its average width.
          The star denotes bootstrap resamples. Every condition must pass
          independently for the driver table and the car table.
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
        <RatingMath />
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
        <div className="method-links">
          <a href={`${source}f1rank/race_total_model.py`}>
            Race model specification
          </a>
          <a href={`${source}f1rank/race_total.py`}>
            Validation and rating calculations
          </a>
        </div>
      </details>
    </section>
  );
}

const FEATURE_LABELS: Record<FeatureTest["feature"], string> = {
  longrun: "Race-fuel pace in practice",
  traps: "Speed traps in practice",
  upgrades: "Upgrades declared to the FIA",
};
const FEATURE_ORDER: FeatureTest["feature"][] = [
  "longrun",
  "traps",
  "upgrades",
];

// The pre-registered weekend-information tests (docs/race_features.md): team rows on the
// car view, the teammate row on the driver view.
export function FeatureTests({
  results,
  driver,
}: {
  results: FeatureTest[];
  driver: boolean;
}) {
  const rows = FEATURE_ORDER.flatMap((feature) => {
    const result = results.find((r) => r.feature === feature);
    const v = result?.metrics[driver ? "drivers" : "cars"];
    return result && v ? [{ feature, result, v }] : [];
  });
  if (!rows.length) return null;
  const dates = [...new Set(rows.map((r) => date(r.result.recorded_at)))];
  return (
    <section
      className="panel race-validation"
      aria-labelledby="feature-tests-title"
    >
      <div className="panel-heading">
        <div>
          <h2 id="feature-tests-title">Tests of weekend information</h2>
          <p>
            Pre-registered · recorded {dates.join(", ")} · the race model plus
            one piece of information known before each race, on the same{" "}
            {rows[0].v.n_races} later races, against the model without it ·{" "}
            <a href={`${source}${rows[0].result.document}`}>
              decisions and results
            </a>
          </p>
        </div>
      </div>
      <div
        className="table-scroll"
        tabIndex={0}
        role="region"
        aria-label="Tests of weekend information; scroll for all columns"
      >
        <table className="forecast-table">
          <thead>
            <tr>
              <th>Information</th>
              <th>Result</th>
              <th className="numeric">Error</th>
              <th className="numeric">Error change, 95% interval</th>
              <th className="numeric">90% coverage</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ feature, v }) => (
              <tr key={feature}>
                <td>{FEATURE_LABELS[feature]}</td>
                <td>
                  <Badge tone={v.passed ? "green" : "amber"}>
                    {v.passed ? "Passed" : "Not established"}
                  </Badge>
                </td>
                <td className="numeric">
                  {(v.rmse * 0.9).toFixed(3)}s{" "}
                  <small>vs {(v.baseline_rmse * 0.9).toFixed(3)}s</small>
                </td>
                <td className="numeric">
                  {signed(v.mse_difference_ci95[0], 4)} to{" "}
                  {signed(v.mse_difference_ci95[1], 4)}
                </td>
                <td className="numeric">{pct(v.coverage90)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-foot">
        <span>
          Error in seconds per 90-second lap, vs the model without the
          information. The error change is the race-level difference in squared
          error (percent²); a test passes when its whole interval is below zero
          and coverage is 85–95%. Each test is run once, with no correction for
          running several; a pass would make the information a candidate for a
          new, separately tested model, not change these tables.
        </span>
      </div>
    </section>
  );
}

// The pre-registered follow-up for the withheld car table (docs/race_car_state.md).
export function CarStateTest({ result }: { result: CarState }) {
  return (
    <RaceValidation
      validation={result.cars}
      driver={false}
      title="Follow-up test: in-season car development"
      note={
        <>
          Pre-registered · recorded {date(result.recorded_at)} · same{" "}
          {result.cars.n_races} races and driver-only baseline, with the car
          allowed to change from race to race ·{" "}
          <a href={`${source}${result.document}`}>decisions and result</a>
        </>
      }
    />
  );
}
