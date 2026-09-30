import { useState } from "react";
import {
  Check,
  ShieldCheck,
  TriangleAlert,
  Info,
  Layers3,
  ArrowDownToLine,
} from "lucide-react";
import type { Dataset, RunStatus, Release } from "../types";
import { date, pct, download, signed } from "../lib";
import { Badge, Empty, PageHeading } from "../components";

const gateNames: Record<string, string> = {
  convergence: "Sampler convergence",
  synthetic_calibration: "Simulation interval coverage",
  synthetic_current_grid: "Portable skill: ranking recovery",
  synthetic_current_grid_in_team: "In-team pace: ranking recovery",
  forecast_vs_baselines: "Forecasts versus simpler models",
  forecast_calibration: "Forecast interval coverage",
  new_pairings: "New teammate pairings",
  sensitivity: "Portable skill: model sensitivity",
  sensitivity_in_team: "In-team pace: model sensitivity",
};
export default function Health({
  data,
  run,
  release,
}: {
  data: Dataset;
  run?: RunStatus;
  release?: Release;
}) {
  const d = data.meta.diagnostics;
  const days = Math.max(
    0,
    Math.floor(
      (Date.now() - new Date(data.meta.data_as_of.date).getTime()) / 86400000,
    ),
  );
  const validation = data.validation.lfo_summary?.data;
  const gates = data.validation.gates?.data;
  const [snapshot, setSnapshot] = useState(0);
  const snap = data.snapshots[snapshot];
  return (
    <>
      <PageHeading
        eyebrow="TRANSPARENCY, BUILT IN"
        title="Behind the rankings."
      >
        What is published, how it was checked, and where the model still has
        limits.
      </PageHeading>
      <div className="health-status panel">
        <div className="health-icon">
          <ShieldCheck size={27} />
        </div>
        <div>
          <h2>Qualifying publication checks passed</h2>
          <p>
            Source and output fingerprints matched at publication. The published
            fit passed its convergence checks.
          </p>
        </div>
        <Badge tone="amber">Racing revalidation pending</Badge>
      </div>
      <div className="stat-grid four">
        <div className="stat-card">
          <div className="stat-label">DATA THROUGH</div>
          <div className="health-number">{date(data.meta.data_as_of.date)}</div>
          <p className="stat-sub">
            {days} days ago · {data.meta.data_as_of.race_name}
          </p>
        </div>
        <div className="stat-card">
          <div className="stat-label">MAXIMUM R-HAT</div>
          <div className="health-number">
            {d.rhat_max.toFixed(3)}
            <Check size={18} />
          </div>
          <p className="stat-sub">Publication threshold &lt; 1.05</p>
        </div>
        <div className="stat-card">
          <div className="stat-label">DIVERGENCES</div>
          <div className="health-number">
            {d.divergences}
            <Check size={18} />
          </div>
          <p className="stat-sub">
            Across {d.n_draws.toLocaleString()} posterior draws
          </p>
        </div>
        <div className="stat-card">
          <div className="stat-label">LATEST REFRESH</div>
          <div className="health-number status-word">
            {run?.state === "ok"
              ? "Published"
              : run?.state === "running"
                ? "Running"
                : run?.state === "failed"
                  ? "Failed"
                  : "Unknown"}
          </div>
          <p className="stat-sub">
            {run?.finished_at
              ? date(run.finished_at)
              : run?.stage || "Status unavailable"}
            {run?.duration_s !== undefined &&
              ` · ${run.duration_s.toFixed(1)}s`}
          </p>
        </div>
      </div>
      {run?.error && (
        <div className="notice warning">
          <TriangleAlert size={18} />
          <div>
            <strong>Refresh stopped at {run.stage}</strong>
            <p>{run.error}</p>
            <p>
              The current release remains available. Resolve the error before
              running another refresh.
            </p>
          </div>
        </div>
      )}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Model readiness</h2>
            <p>Availability is tracked separately for each quality.</p>
          </div>
          <Badge>Evidence before rankings</Badge>
        </div>
        <div className="readiness">
          <div>
            <span className="readiness-icon good">
              <Check size={17} />
            </span>
            <div>
              <strong>Qualifying pace</strong>
              <p>
                Driver pace in the current team and track-neutral car
                performance.
              </p>
            </div>
            <Badge tone="green">Published</Badge>
          </div>
          <div>
            <span className="readiness-icon caution">
              <Info size={17} />
            </span>
            <div>
              <strong>Portable driver skill</strong>
              <p>
                Available with an experimental label; sensitive to structural
                model choices.
              </p>
            </div>
            <Badge tone="amber">Experimental</Badge>
          </div>
          {data.racing.map((r) => (
            <div key={r.name}>
              <span className="readiness-icon">
                <Layers3 size={16} />
              </span>
              <div>
                <strong>{r.name}</strong>
                <p>{r.reason}</p>
                {r.detail && (
                  <details>
                    <summary>Technical detail</summary>
                    <code>{r.detail}</code>
                  </details>
                )}
              </div>
              <Badge tone={r.status === "passed" ? "green" : ""}>
                {r.status === "stale"
                  ? "Needs regeneration"
                  : r.status === "passed"
                    ? "Checks passed"
                    : r.status === "experimental"
                      ? "Experimental"
                      : "Unavailable"}
              </Badge>
            </div>
          ))}
        </div>
      </section>
      <div className="health-columns">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Historical validation</h2>
              <p>Recorded research results · not a fresh validation run</p>
            </div>
          </div>
          <div className="validation-note">
            These summaries predate dashboard publication and have no
            validation-run manifest. They provide research context, not proof
            that a new fit has passed all tests.
          </div>
          {gates ? (
            <div className="gate-list">
              {Object.entries(gates).map(([k, ok]) => (
                <div key={k}>
                  <span>{gateNames[k] || k}</span>
                  <span className={ok ? "gate-pass" : "gate-fail"}>
                    {ok ? <Check size={15} /> : <Info size={15} />}{" "}
                    {ok ? "Recorded pass" : "Recorded fail"}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <Empty title="No validation summary">
              Historical checks have not been recorded.
            </Empty>
          )}
        </section>
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Forecasting benchmark</h2>
              <p>Historical teammate-pairing error · lower is better</p>
            </div>
          </div>
          {validation ? (
            <div className="benchmark">
              <div className="benchmark-head">
                <strong>
                  {validation.pairing_all.rmse_pred.toFixed(3)}
                  <span>s</span>
                </strong>
                <div>
                  model error
                  <br />
                  <span>Across {validation.n_cutoffs} historical cutoffs</span>
                </div>
              </div>
              {[
                { name: "Full model", v: validation.pairing_all.rmse_pred },
                {
                  name: "Static driver + car",
                  v: validation.pairing_all.rmse_akm,
                },
                {
                  name: "Raw teammate gaps",
                  v: validation.pairing_all.rmse_naive,
                },
                { name: "Zero baseline", v: validation.pairing_all.rmse_zero },
              ].map((r, i) => (
                <div className="benchmark-row" key={r.name}>
                  <div>
                    <span>{r.name}</span>
                    <strong>{r.v.toFixed(3)}s</strong>
                  </div>
                  <div className="benchmark-track">
                    <span
                      style={{
                        width: `${(r.v / 0.35) * 100}%`,
                        background: i ? "#c6cfbf" : "#356b50",
                      }}
                    />
                  </div>
                </div>
              ))}
              <p className="small-note">
                RMSE in seconds on a 90-second lap. Session-level 90% intervals
                covered {pct(validation.teammate_session.cov90)} of held-out
                observations. These recorded summaries have not been revalidated
                by this dashboard.
              </p>
            </div>
          ) : (
            <Empty title="Benchmark unavailable">
              Run the forecasting evaluation to add results.
            </Empty>
          )}
        </section>
      </div>
      <section className="panel method-panel">
        <div className="panel-heading">
          <div>
            <h2>How to read these estimates</h2>
            <p>A statistical separation, with visible assumptions.</p>
          </div>
        </div>
        <div className="method-grid">
          <article>
            <span>01</span>
            <h3>Teammates provide the anchor</h3>
            <p>
              Drivers share a car package with their teammate. Drivers changing
              teams connect those comparisons across the grid. A time-varying
              Bayesian model estimates the driver and car contributions.
            </p>
          </article>
          <article>
            <span>02</span>
            <h3>Uncertainty belongs in the result</h3>
            <p>
              Dots show posterior medians. The 90% intervals describe
              uncertainty in estimated pace, not the variability of a single
              lap. Overlapping rank ranges mean a precise ordering is not
              established.
            </p>
          </article>
          <article>
            <span>03</span>
            <h3>The scope stays explicit</h3>
            <p>
              These are qualifying ratings. The headline includes a driver’s
              team-specific effect; portable skill remains experimental. Neither
              is yet an overall measure of racing ability.
            </p>
          </article>
          <article>
            <span>04</span>
            <h3>History can be revised</h3>
            <p>
              New comparisons can change the interpretation of old performances.
              Trend charts use the latest fit; immutable publication snapshots
              preserve what was shown at the time.
            </p>
          </article>
        </div>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Publication archive</h2>
            <p>As-published estimates · kept separately from revised trends</p>
          </div>
          {snap && (
            <button
              className="button"
              onClick={() =>
                download(
                  snap.file,
                  JSON.stringify(snap, null, 2),
                  "application/json",
                )
              }
            >
              <ArrowDownToLine size={15} />
              Snapshot
            </button>
          )}
        </div>
        {snap ? (
          <div className="archive">
            <label>
              Published version
              <select
                value={snapshot}
                onChange={(e) => setSnapshot(Number(e.target.value))}
              >
                {data.snapshots.map((s, i) => (
                  <option key={s.file} value={i}>
                    {s.event.event_id} · {s.event.race_name} ·{" "}
                    {date(s.generated_at)}
                  </option>
                ))}
              </select>
            </label>
            <div className="archive-top">
              {snap.drivers.slice(0, 5).map((r, i) => (
                <div key={String(r.driver_id)}>
                  <span>{i + 1}</span>
                  <strong>{r.name}</strong>
                  <b>{signed(Number(r.in_team_median_s))}s</b>
                </div>
              ))}
            </div>
            <p className="small-note">
              Showing the first five drivers by headline qualifying pace in this
              publication. Only snapshots with explicit headline / portable
              metric labels are included.
            </p>
          </div>
        ) : (
          <Empty title="No compatible snapshots">
            New qualifying publications will appear here.
          </Empty>
        )}
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Release details</h2>
            <p>Reproducible source identity and update timing.</p>
          </div>
          <button
            className="button"
            onClick={() =>
              download(
                "f1-provenance.json",
                JSON.stringify(
                  { release, meta: data.meta, provenance: data.provenance },
                  null,
                  2,
                ),
                "application/json",
              )
            }
          >
            <ArrowDownToLine size={15} />
            Provenance
          </button>
        </div>
        <dl className="release-details">
          <div>
            <dt>Release</dt>
            <dd>{release?.release || "Loading"}</dd>
          </div>
          <div>
            <dt>Model / fit</dt>
            <dd>
              {data.meta.model_version} / {data.meta.fit_id}
            </dd>
          </div>
          <div>
            <dt>Ratings generated</dt>
            <dd>{new Date(data.meta.generated_at).toLocaleString()}</dd>
          </div>
          <div>
            <dt>Dashboard published</dt>
            <dd>
              {release
                ? new Date(release.published_at).toLocaleString()
                : "Unknown"}
            </dd>
          </div>
          <div>
            <dt>Verified inputs / outputs</dt>
            <dd>
              {Object.keys(data.provenance.inputs).length} /{" "}
              {Object.keys(data.provenance.outputs).length} files
            </dd>
          </div>
          <div>
            <dt>Refresh policy</dt>
            <dd>
              Checks for publications every 30 seconds. Model fitting runs
              separately.
            </dd>
          </div>
        </dl>
      </section>
    </>
  );
}
