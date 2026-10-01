import { useState } from "react";
import { ArrowDownToLine, Search } from "lucide-react";
import { Badge, Band, Empty } from "../components";
import { color, date, download, pct, signed } from "../lib";
import type { RacePace as RaceData, RaceEntity, RankingKind } from "../types";
import RankingHeader from "./RankingHeader";

export default function RacePace({ data, kind }: {
  data?: RaceData | null;
  kind: RankingKind;
}) {
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState("");
  const rows = [...(data?.[kind] || [])].sort(
    (a, b) => b.pace.median - a.pace.median,
  );
  const active = rows.find((r) => r.id === selected) || rows[0];
  const shown = rows.filter((r) =>
    `${r.name} ${r.team || ""}`.toLowerCase().includes(search.toLowerCase()),
  );
  const driver = kind === "drivers";
  const validation = data?.validation.metrics[kind];
  const failedChecks = validation
    ? [
        validation.n_races < 12 && "enough held-out races",
        !validation.improves && "prediction improvement",
        !validation.calibrated && "interval coverage",
        !validation.sharper && "narrower prediction intervals",
      ]
        .filter(Boolean)
        .join(", ")
    : "";
  const domain = [
    Math.min(0, ...rows.map((r) => r.pace.q05)),
    Math.max(0, ...rows.map((r) => r.pace.q95)),
  ];
  function exportRows() {
    if (!data) return;
    const headers = [
      "rank",
      "name",
      "team",
      "metric",
      "season",
      "pace_seconds_per_90s",
      "lower_90",
      "upper_90",
      "rank_low",
      "rank_high",
      "races",
      "clean_laps",
      "as_of_event",
      "fit_id",
    ];
    const values = rows.map((r, i) => [
      i + 1,
      r.name,
      r.team || r.name,
      `total_race_pace_${kind}`,
      data.season,
      r.pace.median,
      r.pace.q05,
      r.pace.q95,
      r.pace.rank_lo,
      r.pace.rank_hi,
      r.races,
      r.laps,
      data.data_as_of.event_id,
      data.fit_id,
    ]);
    download(
      `f1-race-${kind}-${data.data_as_of.event_id}.csv`,
      [headers, ...values]
        .map((row) =>
          row
            .map((v) => `"${String(v ?? "").replaceAll('"', '""')}"`)
            .join(","),
        )
        .join("\n"),
    );
  }
  return (
    <>
      <RankingHeader
        kind={kind}
        discipline="race"
        action={
          rows.length > 0 && (
            <button className="button" onClick={exportRows}>
              <ArrowDownToLine size={16} />
              Export race rankings
            </button>
          )
        }
      />
      <div className="event-line">
        <Badge tone={data ? "green" : "amber"}>Dry races</Badge>
        {data && (
          <span>
            {data.data_as_of.race_name}{" "}
            <span className="muted">· {date(data.data_as_of.date)}</span>
          </span>
        )}
        <span className="muted">
          {data ? `${data.season} season estimates` : "No checked release"}
        </span>
      </div>
      {!data ? (
        <section className="panel">
          <Empty title="Race-pace validation is pending">
            {driver ? "Driver" : "Car"} rankings appear here when their prediction
            and uncertainty checks pass.
          </Empty>
        </section>
      ) : (
        <>
          {rows.length > 0 ? (
            <div className="ranking-layout">
              <section className="panel ranking-panel">
                <div className="panel-heading">
                  <div>
                    <h2>{driver ? "Driver race pace" : "Car race pace"}</h2>
                    <p>
                      {driver
                        ? "Car performance accounted for"
                        : "Driver contribution accounted for"}{" "}
                      · tyre age 10 laps
                    </p>
                  </div>
                </div>
                <div className="table-tools">
                  <label className="search">
                    <Search size={16} />
                    <input
                      aria-label="Search race rankings"
                      placeholder={
                        driver ? "Search a driver…" : "Search a constructor…"
                      }
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                    />
                  </label>
                  <span className="table-count">
                    {shown.length} {kind}
                  </span>
                </div>
                <div
                  className="table-scroll"
                  tabIndex={0}
                  role="region"
                  aria-label="Race pace rankings; scroll for all entries"
                >
                  <table className="rank-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>{driver ? "DRIVER" : "CONSTRUCTOR"}</th>
                        <th className="numeric">PACE / 90s</th>
                        <th className="range-col">90% PACE INTERVAL</th>
                        <th className="numeric">90% RANK RANGE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {shown.map((r) => (
                        <tr
                          key={r.id}
                          className={r.id === active.id ? "selected-row" : ""}
                        >
                          <td>
                            <span
                              className={`rank ${rows.indexOf(r) < 3 ? "top-rank" : ""}`}
                            >
                              {String(rows.indexOf(r) + 1).padStart(2, "0")}
                            </span>
                          </td>
                          <td>
                            <button
                              className="entity-button"
                              aria-label={`Inspect race pace for ${r.name}`}
                              aria-pressed={r.id === active.id}
                              onClick={() => setSelected(r.id)}
                            >
                              <span
                                className="team-stripe"
                                style={{ background: color(r.lineage || r.id) }}
                              />
                              <span>
                                <strong>{r.name}</strong>
                                <small>
                                  {r.team || `${r.races} dry races`}
                                </small>
                              </span>
                            </button>
                          </td>
                          <td className="numeric pace-number">
                            {signed(r.pace.median)}
                            <small>s</small>
                          </td>
                          <td
                            className="range-col"
                            style={{
                              color:
                                r.id === active.id
                                  ? "var(--accent)"
                                  : "var(--muted)",
                            }}
                          >
                            <Band value={r.pace} domain={domain} compact />
                          </td>
                          <td className="numeric">
                            <span className="rank-range">
                              {r.pace.rank_lo}–{r.pace.rank_hi}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {!shown.length && (
                  <Empty title="No matching entries">Try another name.</Empty>
                )}
                <div className="table-foot">
                  <span>
                    Relative to the rated {driver ? "drivers" : "cars"}
                  </span>
                  <span>Positive = faster · ranked by median</span>
                </div>
              </section>
              <RaceDetail row={active} driver={driver} />
            </div>
          ) : (
            <section className="panel">
              <Empty
                title={
                  validation?.passed
                    ? "Not enough current-season race data"
                    : `${driver ? "Driver" : "Car"} race-pace ranking withheld`
                }
              >
                {validation?.passed
                  ? "Each entry needs two usable dry races this season before it appears in the table."
                  : `Publication checks not met: ${failedChecks}.`}
              </Empty>
            </section>
          )}
          {data[driver ? "unrated_drivers" : "unrated_cars"].length > 0 && (
            <p className="muted">
              {data[driver ? "unrated_drivers" : "unrated_cars"].length} {kind}{" "}
              have fewer than two usable dry races this season.
            </p>
          )}
          {validation && (
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
                      vs {(validation.baseline_interval_width * 0.9).toFixed(3)}
                      s baseline
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
          )}
        </>
      )}
      <section className="panel race-method">
        <details>
          <summary>Race-pace methodology</summary>
          <p>
            The model fits clean laps from dry races since 2018. It estimates a
            lasting driver contribution and a season-specific change in form,
            plus a separate car contribution for each team and season. Teammates
            share the car term; drivers changing teams connect the comparisons.
          </p>
          <p>
            The table reports total race pace at a reference tyre age of 10
            laps. It is estimated independently of qualifying. Driver ratings
            combine lasting pace and current-season form. Car ratings describe
            the current season across its sampled circuits. Neither is a
            prediction of finishing position or championship points.
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
            intervals, and narrower intervals than the baseline. Error
            uncertainty is resampled by race.
          </p>
          <p>
            Driver testing uses teammate pace gaps. Car testing uses observed
            team-average pace, without subtracting this model’s driver estimates
            from the target. Shared uncertainty in the per-race estimates is
            retained by resampling whole bootstrap draws across drivers and
            teams.
          </p>
          <p>
            The 90% pace and rank intervals come from joint posterior draws.
            Driver and car estimates are centred separately on the currently
            rated field. The displayed rank orders the medians; wide rank
            intervals indicate that the data do not establish a precise order.
          </p>
          <p>
            Fuel loads, strategy, team priority and how hard a driver pushes are
            only partly observed. Persistent differences can be attributed to a
            driver or car by the model. These are conditional performance
            estimates, not an exact measurement of innate skill. Reliability,
            pit stops, starts and overtaking are outside this metric.
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
    </>
  );
}

function RaceDetail({ row, driver }: { row: RaceEntity; driver: boolean }) {
  return (
    <aside className="panel race-detail">
      <span className="eyebrow">{driver ? "DRIVER" : "CAR"} RACE PACE</span>
      <h2>{row.name}</h2>
      <p className="muted">{row.team || "Season estimate"}</p>
      <div className="race-detail-pace">
        {signed(row.pace.median)}
        <small>s</small>
      </div>
      <p className="muted">per 90-second lap · positive = faster</p>
      <dl className="race-detail-values">
        <div>
          <dt>90% pace interval</dt>
          <dd>
            {signed(row.pace.q05)} to {signed(row.pace.q95)}s
          </dd>
        </div>
        <div>
          <dt>90% rank range</dt>
          <dd>
            {row.pace.rank_lo}–{row.pace.rank_hi}
          </dd>
        </div>
        <div>
          <dt>Probability fastest</dt>
          <dd>{pct(row.pace.p_fastest)}</dd>
        </div>
        <div>
          <dt>Usable races this season</dt>
          <dd>{row.races}</dd>
        </div>
        <div>
          <dt>Clean laps this season</dt>
          <dd>{row.laps.toLocaleString()}</dd>
        </div>
      </dl>
    </aside>
  );
}
