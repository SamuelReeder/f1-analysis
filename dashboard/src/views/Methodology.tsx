import type { Dataset } from "../types";

const source = "https://github.com/SamuelReeder/f1-analysis/blob/main/";

export default function Methodology({ data }: { data: Dataset }) {
  return (
    <section className="panel methodology" aria-labelledby="methodology-title">
      <div className="panel-heading">
        <div>
          <h2 id="methodology-title">Methodology</h2>
          <p>{data.meta.model_version} · qualifying pace</p>
        </div>
      </div>
      <p className="method-intro">
        A time-varying Bayesian model estimates driver and car contributions to
        qualifying pace. Teammates provide comparisons within a car; drivers
        changing teams connect those comparisons across the grid.
      </p>
      <details>
        <summary>Data and lap filtering</summary>
        <div className="method-body">
          <p>
            This release uses {data.meta.n_lap_times.toLocaleString()}{" "}
            qualifying times from {data.events.length} events in{" "}
            {data.meta.window}. Each observation is a driver’s recorded time in
            Q1, Q2 or Q3, sourced from Jolpica. Missing sessions can be
            supplemented from FastF1 timing. Race results and telemetry do not
            enter these qualifying ratings.
          </p>
          <p>
            Times are converted to relative pace within each qualifying segment:
            <code className="method-formula">
              pace = −100 × ln(time / segment median)
            </code>
            Positive values mean faster laps. Times roughly 5% slower than the
            segment median are excluded, as are segments with fewer than four
            usable times. An entrant without a valid time keeps a model estimate
            based on their other appearances rather than receiving a zero score.
          </p>
          <p>
            A separate baseline for each segment accounts for common track
            conditions and the different fields in Q2 and Q3. A heavy-tailed
            Student-t error model and a separate noise scale per segment reduce
            the influence of unusual laps and variable sessions. They cannot
            identify every instance of traffic, damage or a missed final run.
          </p>
          <a href={`${source}f1rank/design.py`}>Data preparation code</a>
        </div>
      </details>
      <details>
        <summary>Separating driver and car</summary>
        <div className="method-body">
          <p>
            The model assumes teammates share a car package. Their pace
            difference therefore provides evidence about the drivers. When a
            driver moves, comparisons with their old and new teammates help
            place both teams on the same scale. All qualifying observations are
            fitted together.
          </p>
          <p>
            The fitted pace includes persistent driver skill, a driver–team
            effect, persistent car development, circuit suitability, temporary
            driver and car weekend effects, and a shared car effect within each
            segment. Temporary effects allow an exceptional or poor weekend
            without forcing the same-sized change into the lasting rating.
          </p>
          <p>
            This is a statistical separation, not a controlled equal-car test.
            Unequal upgrades or team support can resemble a driver difference.
            Sparse links between teams leave several explanations compatible
            with the data. Shared priors pull weakly supported estimates towards
            the population and limit how far a small sample can move a rating.
          </p>
          <a href={`${source}f1rank/model.py`}>Model specification</a>
        </div>
      </details>
      <details>
        <summary>How ratings change over time</summary>
        <div className="method-body">
          <p>
            Driver skill follows a gradual random walk, with separate variation
            within and between seasons, an experience curve, and an age effect
            after 32. These effects are estimated from the field; they do not
            prescribe an individual driver’s career path.
          </p>
          <p>
            Car performance can move more abruptly to accommodate upgrades. Some
            performance carries into the next season, with weaker carryover and
            larger possible changes at the 2014, 2017, 2022 and 2026 regulation
            resets. Team rebrands are linked through their underlying lineage.
          </p>
          <p>
            Circuit suitability is represented by a learned axis from slower
            street circuits to faster circuits, with a separate loading for each
            team-season. This is a simplified track adjustment, not a full
            aerodynamic or power-unit model.
          </p>
        </div>
      </details>
      <details>
        <summary>What each rating measures</summary>
        <div className="method-body">
          <dl>
            <div>
              <dt>In-team driver pace</dt>
              <dd>
                Portable skill plus the persistent effect associated with that
                driver and team lineage. It excludes the shared car contribution
                and temporary weekend form.
              </dd>
            </div>
            <div>
              <dt>Portable skill · experimental</dt>
              <dd>
                The driver component after removing the driver–team effect. It
                aims to describe ability that transfers between teams, but the
                separation is sensitive to model assumptions.
              </dd>
            </div>
            <div>
              <dt>Driver–team effect</dt>
              <dd>
                A persistent association for a driver and team lineage,
                including separate spells at that team. It cannot distinguish
                car-handling compatibility from support, role, adaptation or
                selection.
              </dd>
            </div>
            <div>
              <dt>Car pace</dt>
              <dd>
                The persistent car-package contribution at a neutral circuit
                profile, excluding temporary weekend effects. The
                circuit-adjusted history adds the estimated track contribution.
                Reliability and pit stops are separate qualities.
              </dd>
            </div>
          </dl>
          <p>
            Ratings are centred on the average driver or car entered at each
            event and converted to seconds per 90-second reference lap. A
            +0.100s rating means an estimated tenth gained against that field
            average. It is not a literal lap-time prediction at every circuit.
          </p>
          <a href={`${source}f1rank/ratings.py`}>Rating definitions in code</a>
        </div>
      </details>
      <details>
        <summary>Intervals, ranks and head-to-head comparisons</summary>
        <div className="method-body">
          <p>
            The fit produces {data.meta.diagnostics.n_draws.toLocaleString()}{" "}
            posterior samples: plausible sets of ratings given the data and
            model assumptions. The headline value is the median. The 90% pace
            interval spans the 5th to 95th percentiles; it describes uncertainty
            in estimated ability, not the spread of a driver’s individual laps.
          </p>
          <p>
            Drivers or cars are ranked within each joint sample to obtain rank
            ranges and probabilities of being fastest or in the top three. The
            table itself is sorted by median pace. Wide rank ranges mean nearby
            positions are poorly resolved; these probabilities are not race-win
            probabilities.
          </p>
          <p>
            A head-to-head gap subtracts the two ratings within each shared
            sample, preserving their dependence. It does not subtract the ends
            of their separate intervals. Gap intervals and car comparison
            probabilities use {data.comparison_draws} shared samples; driver
            comparison probabilities use all{" "}
            {data.meta.diagnostics.n_draws.toLocaleString()}.
          </p>
        </div>
      </details>
      <details>
        <summary>Validation and known limits</summary>
        <div className="method-body">
          <p>
            Publication checks require matching input and output fingerprints
            and a converged fit. Convergence checks whether the sampler has
            explored the fitted model consistently; it does not establish that
            the model’s assumptions are correct.
          </p>
          <p>
            Research validation fits only past data and forecasts later teammate
            gaps, comparing errors with static driver-and-car ratings, raw
            teammate gaps and a zero-gap baseline. Simulated seasons with known
            driver and car values test whether the model can recover them on
            F1’s actual network of teammates and team changes. Sensitivity runs
            change the driver evolution and team-effect assumptions.
          </p>
          <p>
            Portable skill failed the recorded ranking-recovery and sensitivity
            gates, so it remains experimental. The validation results above are
            historical research, not a fresh validation of this release.
            Qualifying pace is only one part of driver performance: the
            dashboard does not yet publish an overall racing-skill ranking or an
            equal-car championship. Each racing quality’s current status appears
            above.
          </p>
          <a href={`${source}outputs/REPORT.md`}>
            Research report and acceptance gates
          </a>
        </div>
      </details>
      <details>
        <summary>Updates and historical estimates</summary>
        <div className="method-body">
          <p>
            New rankings require a data refresh, model refit, checked export and
            publication. The browser checks for a published release every 30
            seconds; it does not refit the model. A website-only deployment can
            change the interface while retaining the same ratings.
          </p>
          <p>
            History charts use the latest fit, so later evidence can revise an
            earlier estimate. Their reference is the field at each event, which
            also changes. They are neither historical forecasts nor absolute
            comparisons across eras. Publication snapshots preserve the
            estimates that were actually published at the time.
          </p>
        </div>
      </details>
    </section>
  );
}
