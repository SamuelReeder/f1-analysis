import { useState } from "react";
import {
  Check,
  ShieldCheck,
  TriangleAlert,
  Info,
  Layers3,
  ArrowDownToLine,
  Activity,
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
  const driverRace = Boolean(data.race_pace?.drivers.length);
  const carRace = Boolean(data.race_pace?.cars.length);
  const raceStatus =
    driverRace && carRace
      ? "Race pace published"
      : driverRace
        ? "Driver race pace published"
        : carRace
          ? "Car race pace published"
          : data.race_pace
            ? "Race pace unavailable"
            : "Racing validation pending";
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
      <PageHeading title="Model health" />
      <div className="health-status panel">
        <div className="health-icon">
          <ShieldCheck size={27} />
        </div>
        <div>
          <h2>Qualifying publication checks passed</h2>
          <p>Source checksums matched · fit converged</p>
        </div>
        <Badge tone={driverRace || carRace ? "green" : "amber"}>
          {raceStatus}
        </Badge>
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
      <Updates data={data} />
      {run?.error && (
        <div className="notice warning">
          <TriangleAlert size={18} />
          <div>
            <strong>Refresh stopped at {run.stage}</strong>
            <p>{run.error}</p>
            <p>Showing the last published release.</p>
          </div>
        </div>
      )}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Model readiness</h2>
          </div>
        </div>
        <div className="readiness">
          {data.race_pace &&
            (["drivers", "cars"] as const).map((kind) => (
              <div key={kind}>
                <span
                  className={`readiness-icon ${data.race_pace!.validation.metrics[kind].passed ? "good" : "caution"}`}
                >
                  <Activity size={17} />
                </span>
                <div>
                  <strong>
                    {kind === "drivers" ? "Driver" : "Car"} race pace
                  </strong>
                  <p>
                    {data.race_pace!.validation.metrics[kind].passed
                      ? "Prediction and uncertainty checks passed"
                      : "Publication checks not passed"}
                  </p>
                </div>
                <a href={`#${kind}/race`}>View race pace</a>
              </div>
            ))}
          <div>
            <span className="readiness-icon good">
              <Check size={17} />
            </span>
            <div>
              <strong>Qualifying pace</strong>
              <p>In-team driver pace and track-neutral car pace</p>
            </div>
            <Badge tone="green">Published</Badge>
          </div>
          <div>
            <span className="readiness-icon caution">
              <Info size={17} />
            </span>
            <div>
              <strong>Portable driver skill</strong>
              <p>Sensitive to model assumptions</p>
            </div>
            <Badge tone="amber">Experimental</Badge>
          </div>
        </div>
        <details className="research-group">
          <summary>
            In research ({data.racing.length}) · not published as ratings
          </summary>
          <div className="readiness">
            {data.racing.map((r) => (
              <div key={r.name}>
                <span className="readiness-icon">
                  <Layers3 size={16} />
                </span>
                <div>
                  <strong>{r.name}</strong>
                  <details>
                    <summary>Status details</summary>
                    <p>{r.reason}</p>
                    {r.detail && <code>{r.detail}</code>}
                  </details>
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
        </details>
      </section>
      <div className="health-columns">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Historical validation</h2>
            </div>
          </div>
          <div className="validation-note">
            Earlier research runs; not revalidated for this release. No
            validation-run manifest.
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
                        background: i ? "var(--line-strong)" : "var(--accent)",
                      }}
                    />
                  </div>
                </div>
              ))}
              <p className="small-note">
                RMSE per 90s lap · held-out session coverage (90% intervals):{" "}
                {pct(validation.teammate_session.cov90)}
              </p>
            </div>
          ) : (
            <Empty title="Benchmark unavailable">
              Run the forecasting evaluation to add results.
            </Empty>
          )}
        </section>
      </div>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Publication archive</h2>
            <p>Original published estimates</p>
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
            <p className="small-note">Top 5 · in-team qualifying pace</p>
          </div>
        ) : (
          <Empty title="No compatible snapshots">
            New qualifying publications will appear here.
          </Empty>
        )}
        <RaceArchive data={data} />
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Release details</h2>
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
            <dt>Dataset</dt>
            <dd>
              {data.meta.n_lap_times.toLocaleString()} qualifying times ·{" "}
              {data.events.length} events · {data.meta.window}
            </dd>
          </div>
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
            <dd>Publication check every 30s; model fitting runs separately.</dd>
          </div>
        </dl>
      </section>
    </>
  );
}

function Updates({ data }: { data: Dataset }) {
  const r = data.refresh;
  const minutes = (s: number) =>
    s >= 90 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`;
  const total = r?.stages.reduce((t, s) => t + s.seconds, 0) || 0;
  return (
    <section className="panel" aria-labelledby="updates-title">
      <div className="panel-heading">
        <div>
          <h2 id="updates-title">Updates</h2>
          <p>
            Checked every Monday and Tuesday at 06:00 UTC; a new race triggers a
            full refit
          </p>
        </div>
        {r?.run_url && (
          <a className="button" href={r.run_url}>
            Run log
          </a>
        )}
      </div>
      {r ? (
        <dl className="release-details">
          <div>
            <dt>Latest refit</dt>
            <dd>
              {r.event ? `${r.event.race_name} (${r.event.event_id})` : "—"}
            </dd>
          </div>
          <div>
            <dt>Finished</dt>
            <dd>
              {new Date(r.finished_at).toLocaleString()} ·{" "}
              {r.trigger === "schedule"
                ? "scheduled"
                : r.trigger === "workflow_dispatch"
                  ? "started manually"
                  : "fitting machine"}
            </dd>
          </div>
          <div>
            <dt>Duration</dt>
            <dd>
              {minutes(total)} ·{" "}
              {r.stages
                .filter((s) => s.seconds >= 60)
                .map((s) => `${s.stage} ${minutes(s.seconds)}`)
                .join(", ")}
            </dd>
          </div>
          <div>
            <dt>Race timing</dt>
            <dd>
              {r.races ? "Extracted and refitted" : "Not part of this run"}
            </dd>
          </div>
        </dl>
      ) : (
        <Empty title="No recorded refit">
          The first scheduled refresh will record its stages here.
        </Empty>
      )}
    </section>
  );
}

function RaceArchive({ data }: { data: Dataset }) {
  const snaps = [...(data.race_snapshots || [])].reverse();
  const [i, setI] = useState(0);
  const snap = snaps[i];
  if (!snap) return null;
  return (
    <div className="archive race-archive">
      <label>
        Race pace version
        <select value={i} onChange={(e) => setI(Number(e.target.value))}>
          {snaps.map((s, k) => (
            <option key={`${s.grid_as_of}-${s.fit_id}`} value={k}>
              {s.data_as_of.event_id} · {s.data_as_of.race_name} ·{" "}
              {date(s.generated_at)}
            </option>
          ))}
        </select>
      </label>
      {snap.passed.drivers ? (
        <>
          <div className="archive-top">
            {snap.drivers.slice(0, 5).map((r, k) => (
              <div key={r.id}>
                <span>{k + 1}</span>
                <strong>{r.name}</strong>
                <b>{signed(r.pace.median)}s</b>
              </div>
            ))}
          </div>
          <p className="small-note">
            Top 5 · dry-race pace
            {snap.passed.cars ? "" : " · car table withheld in this version"}
          </p>
        </>
      ) : (
        <p className="small-note">
          Both race tables were withheld in this version.
        </p>
      )}
      <button
        className="button"
        onClick={() =>
          download(
            `race_${snap.grid_as_of}_${snap.fit_id}.json`,
            JSON.stringify(snap, null, 2),
            "application/json",
          )
        }
      >
        <ArrowDownToLine size={15} />
        Race snapshot
      </button>
    </div>
  );
}
