import type { Dataset } from "../types";
import { Equation, InlineMath, RatingMath } from "../ModelMath";

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
            Times are converted to relative pace within each qualifying segment.
          </p>
          <Equation
            label="Qualifying pace transformation"
            tex={String.raw`y_i = -100\ln\!\left(\frac{t_i}{\widetilde t_q}\right)`}
          />
          <p>
            <InlineMath tex="t_i" /> is the recorded time and{" "}
            <InlineMath tex={String.raw`\widetilde t_q`} /> is the median in
            that event’s qualifying segment <InlineMath tex="q" />. Positive{" "}
            <InlineMath tex="y_i" /> means faster. Times roughly 5% slower than
            the segment median are excluded, as are segments with fewer than
            four usable times. An entrant without a valid time keeps a model
            estimate based on their other appearances rather than receiving a
            zero score.
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
          <Equation
            label="Qualifying observation model"
            tex={String.raw`\begin{aligned}
              y_i &\sim t_\nu(\mu_i,\sigma_q) \\
              \mu_i &= \alpha_q + D_{de} + K_{dje} + C_{je} \\
                    &\quad + T_{je} + W_{je} + F_{de} + B_{jq}
            \end{aligned}`}
          />
          <p>
            Indices <InlineMath tex="d" />, <InlineMath tex="j" /> and{" "}
            <InlineMath tex="e" /> identify the driver, team lineage and event.{" "}
            <InlineMath tex={String.raw`t_\nu(\mu,\sigma)`} /> is a Student-t
            distribution with location <InlineMath tex={String.raw`\mu`} />,
            scale <InlineMath tex={String.raw`\sigma`} /> and learned tail
            parameter <InlineMath tex={String.raw`\nu`} />. Its scale is not its
            standard deviation.
          </p>
          <dl className="math-terms">
            <div>
              <dt>
                <InlineMath tex={String.raw`\alpha_q`} />
              </dt>
              <dd>Baseline for this event’s Q1, Q2 or Q3 segment.</dd>
            </div>
            <div>
              <dt>
                <InlineMath tex="D_{de}" />
              </dt>
              <dd>Persistent, time-varying driver skill.</dd>
            </div>
            <div>
              <dt>
                <InlineMath tex="K_{dje}" />
              </dt>
              <dd>
                Persistent driver–team effect, centred on the event’s drivers.
              </dd>
            </div>
            <div>
              <dt>
                <InlineMath tex="C_{je}" />
              </dt>
              <dd>Persistent car-package pace.</dd>
            </div>
            <div>
              <dt>
                <InlineMath tex="T_{je}" />
              </dt>
              <dd>Car suitability for the event’s circuit.</dd>
            </div>
            <div>
              <dt>
                <InlineMath tex="W_{je}" />
              </dt>
              <dd>Car weekend deviation shared by teammates.</dd>
            </div>
            <div>
              <dt>
                <InlineMath tex="F_{de}" />
              </dt>
              <dd>Temporary driver weekend form.</dd>
            </div>
            <div>
              <dt>
                <InlineMath tex="B_{jq}" />
              </dt>
              <dd>
                Car effect shared by teammates within the qualifying segment.
              </dd>
            </div>
          </dl>
          <p>
            Temporary effects allow an exceptional or poor weekend without
            forcing the same-sized change into the lasting rating.
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
          <Equation
            label="Driver evolution"
            tex={String.raw`\begin{aligned}
              w_{de} &= w_{d,e^-} + \eta_{de},\quad \eta_{de}\sim\mathcal N(0,v_{de}) \\
              v_{de} &= \begin{cases}
                \sigma_{\rm race}^{2}\Delta e & \text{within a season} \\
                \sigma_{\rm season}^{2}\max(\Delta s,1) & \text{between seasons}
              \end{cases} \\
              D_{de} &= \operatorname{centre}_e\!\bigl[
                w_{de}+g(1-e^{-n_{de}/\lambda}) \\
                &\hspace{5em}{}+a\max(\operatorname{age}_{de}-32,0)\bigr]
            \end{aligned}`}
          />
          <p>
            <InlineMath tex="e^-" /> is the driver’s previous appearance;{" "}
            <InlineMath tex={String.raw`\Delta e`} /> and{" "}
            <InlineMath tex={String.raw`\Delta s`} /> count intervening events
            and seasons. <InlineMath tex="n_{de}" /> is prior race experience.
            The model learns experience gain <InlineMath tex="g" />, its
            timescale <InlineMath tex={String.raw`\lambda`} /> and age slope{" "}
            <InlineMath tex="a" />. A driver’s first walk state has prior{" "}
            <InlineMath tex={String.raw`\mathcal N(0,\sigma_{\rm level}^2)`} />.
            The operator{" "}
            <InlineMath tex={String.raw`\operatorname{centre}_e`} /> subtracts
            the event’s field mean, separately for drivers and cars.
          </p>
          <p>
            Car performance can move more abruptly to accommodate upgrades. Some
            performance carries into the next season, with weaker carryover and
            larger possible changes at the 2014, 2017, 2022 and 2026 regulation
            resets. Team rebrands are linked through their underlying lineage.
          </p>
          <Equation
            label="Car development"
            tex={String.raw`\begin{aligned}
              c_{je} &= \kappa_e c_{j,e^-}+\xi_{je},\quad C_{je}=\operatorname{centre}_e(c_{je}) \\
              \kappa_e &= \begin{cases}
                1 & \text{within a season} \\
                \rho & \text{ordinary season start} \\
                \rho_{\rm reset} & \text{regulation reset}
              \end{cases} \\
              \xi_{je} &\sim \begin{cases}
                t_4(0,\sigma_{\rm car,race}) & \text{within a season} \\
                \mathcal N(0,\sigma_{\rm car,season}^{2}) & \text{ordinary season start} \\
                \mathcal N(0,\sigma_{\rm car,reset}^{2}) & \text{regulation reset}
              \end{cases}
            \end{aligned}`}
          />
          <p>
            <InlineMath tex="c" /> is the uncentred car state. The first state
            of each team in the window has prior{" "}
            <InlineMath tex={String.raw`\mathcal N(0,1.5^2)`} />. Within-season
            innovations use four-degree-of-freedom Student-t tails; season
            changes use Gaussian jumps. Carryover has priors{" "}
            <InlineMath tex={String.raw`\rho\sim\operatorname{Beta}(8,2)`} />{" "}
            and{" "}
            <InlineMath
              tex={String.raw`\rho_{\rm reset}\sim\operatorname{Beta}(3,3)`}
            />
            .
          </p>
          <p>
            Circuit suitability is represented by a learned axis from slower
            street circuits to faster circuits, with a separate loading for each
            team-season. This is a simplified track adjustment, not a full
            aerodynamic or power-unit model.
          </p>
          <Equation
            label="Circuit suitability"
            tex={String.raw`T_{je}=\operatorname{centre}_e\!\left[\ell_{js}\,x_{\operatorname{circuit}(e)}\right]`}
          />
          <p>
            <InlineMath tex="x" /> is the circuit factor estimated from training
            data and then held fixed during the fit.{" "}
            <InlineMath tex={String.raw`\ell_{js}`} /> is the learned loading
            for team <InlineMath tex="j" /> in season <InlineMath tex="s" />.
          </p>
        </div>
      </details>
      <details>
        <summary>Bayesian estimation and priors</summary>
        <div className="method-body">
          <Equation
            label="Posterior distribution"
            tex={String.raw`p(\theta\mid y)\propto p(y\mid\theta)\,p(\theta)`}
          />
          <p>
            <InlineMath tex={String.raw`\theta`} /> contains the unknown model
            parameters. The likelihood{" "}
            <InlineMath tex={String.raw`p(y\mid\theta)`} /> scores how well a
            parameter set explains the laps; the prior{" "}
            <InlineMath tex={String.raw`p(\theta)`} /> supplies regularisation.
            NumPyro’s No-U-Turn Sampler draws from the resulting posterior in
            four chains. Driver, car and nuisance parameters are estimated
            jointly.
          </p>
          <Equation
            label="Hierarchical effects and segment noise"
            tex={String.raw`\begin{aligned}
              z_k &\sim\mathcal N(0,1),\quad h_k=\sigma_h z_k \\
              \sigma_h &\sim\operatorname{HalfNormal}(b_h) \\
              \sigma_q &= \sigma_0\exp(\tau u_q),\quad u_q\sim\mathcal N(0,1) \\
              \nu &\sim\operatorname{Gamma}(2,0.1)
            \end{aligned}`}
          />
          <p>
            Effects <InlineMath tex="h_k" /> share a learned scale, then are
            centred within their comparison group where applicable. Normal
            distributions use mean and variance; HalfNormal is a positive scale
            prior; Gamma uses shape and rate. In percentage-point units,{" "}
            <InlineMath tex="b_h" /> is 0.5 for initial driver levels, 0.1 for
            driver–team effects, and 0.2 for driver weekend, car weekend and car
            segment effects. Driver walk scales use HalfNormal(0.03) within
            seasons and HalfNormal(0.2) between seasons. Car innovation scales
            use 0.15, 0.5 and 1.0 for within-season, season-start and reset
            changes. Both <InlineMath tex={String.raw`\sigma_0`} /> and{" "}
            <InlineMath tex={String.raw`\tau`} /> have HalfNormal(0.5) priors.
          </p>
          <p>
            The experience gain has prior{" "}
            <InlineMath tex={String.raw`g\sim\mathcal N(0,1)`} />, the age slope{" "}
            <InlineMath tex={String.raw`a\sim\mathcal N(0,0.2^2)`} />, and the
            experience timescale{" "}
            <InlineMath
              tex={String.raw`\ln\lambda\sim\mathcal N(\ln20,0.7^2)`}
            />
            . Neither trend is forced to have a particular sign. Segment
            baselines have prior{" "}
            <InlineMath tex={String.raw`\mathcal N(0,2^2)`} />; circuit loadings
            use a shared HalfNormal(0.5) scale prior. The linked specification
            includes every prior and centring constraint.
          </p>
          <a href={`${source}f1rank/model.py`}>
            Full model and prior specification
          </a>
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
          <Equation
            label="Qualifying rating components"
            tex={String.raw`\begin{aligned}
              x_{\rm in\ team} &= D_{de}+K_{dje} \\
              x_{\rm portable} &= D_{de} \\
              x_{\rm car} &= C_{je},\quad x_{\rm at\ circuit}=C_{je}+T_{je}
            \end{aligned}`}
          />
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
          <RatingMath />
          <p>
            A head-to-head gap subtracts the two ratings within each shared
            sample, preserving their dependence. It does not subtract the ends
            of their separate intervals. Gap intervals and car comparison
            probabilities use {data.comparison_draws} shared samples; driver
            comparison probabilities use all{" "}
            {data.meta.diagnostics.n_draws.toLocaleString()}.
          </p>
          <Equation
            label="Paired comparison"
            tex={String.raw`\begin{aligned}
              \Delta_{AB}^{(m)} &= r_A^{(m)}-r_B^{(m)} \\
              P(A\text{ faster than }B) &\approx\frac{1}{M}\sum_{m=1}^{M}\mathbf{1}\{\Delta_{AB}^{(m)}>0\}
            \end{aligned}`}
          />
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
            Qualifying pace is only one part of driver performance. The
            experimental Overall (equal car) view publishes simulated championship
            outcomes only when its current provenance and combined validation
            pass; otherwise it explains why the ranking is withheld. Its evidence
            lists admitted qualities and excluded test seasons. Each racing
            quality’s current status appears on Model health.
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
