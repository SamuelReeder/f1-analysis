import { useState } from "react";
import { Check, X } from "lucide-react";
import type { Dataset } from "../types";
import { PageHeading, Empty, Badge, useWidth } from "../components";
import { circuitCode, date, niceTicks, pct, signed, SERIES } from "../lib";

// Scored forecasts: each event is predicted by a fit that stops at the event before it.
export default function Forecasts({ data }: { data: Dataset }) {
  const asof = data.asof;
  const [hover, setHover] = useState<number | null>(null);
  const [ref, W] = useWidth(930);
  const names = new Map(data.catalog.drivers.map((d) => [d.id, d.name]));
  const codes = new Map(data.drivers.map((d) => [d.id, d.code]));
  const label = (id: string) =>
    codes.get(id) ||
    (names.get(id) || id).split(" ").at(-1)!.slice(0, 3).toUpperCase();
  if (!asof?.pooled) {
    return (
      <>
        <PageHeading title="Track record" />
        <Empty title="No scored forecasts yet">
          Forecasts appear after a fit that excludes a race has been scored
          against it.
        </Empty>
      </>
    );
  }
  const p = asof.pooled;
  const events = asof.events.filter((e) => e.n_pairs > 0);
  const circuitOf = new Map(data.events.map((e) => [e.event_id, e.circuit_id]));
  const vals = events.flatMap((e) => [e.rmse!, e.rmse_naive!]);
  const hi = Math.max(...vals) * 1.15;
  const step = (W - 95) / Math.max(events.length - 1, 1);
  const x = (i: number) => 65 + i * step;
  const every = Math.max(1, Math.ceil(30 / step));
  const y = (v: number) => 230 - (v / hi) * 200;
  const path = (key: "rmse" | "rmse_naive") =>
    events.map((e, i) => `${i ? "L" : "M"}${x(i)},${y(e[key]!)}`).join(" ");
  const latest = asof.latest;
  const pairs = (latest?.pairs || []).filter((r) => r.segment === "Q1");
  return (
    <>
      <PageHeading title="Track record">
        How well the qualifying model predicts the next race it has not seen
      </PageHeading>
      <div className="stat-grid four">
        <div className="stat-card">
          <div className="stat-label">TEAMMATE GAP ERROR</div>
          <div className="health-number">
            {p.rmse.toFixed(3)}
            <small>s</small>
          </div>
          <p className="stat-sub">
            vs {p.rmse_naive.toFixed(3)}s for last season’s gap ·{" "}
            {p.rmse_zero.toFixed(3)}s for no gap
          </p>
        </div>
        <div className="stat-card">
          <div className="stat-label">90% INTERVAL COVERAGE</div>
          <div className="health-number">{pct(p.coverage90)}</div>
          <p className="stat-sub">
            Share of real gaps inside the interval · target 90%
          </p>
        </div>
        <div className="stat-card">
          <div className="stat-label">PREDICTED ORDER</div>
          <div className="health-number">{p.order_spearman.toFixed(2)}</div>
          <p className="stat-sub">
            Mean rank correlation with each session’s real order (1 = exact)
          </p>
        </div>
        <div className="stat-card">
          <div className="stat-label">SCORED</div>
          <div className="health-number">{p.n_events}</div>
          <p className="stat-sub">
            events · {p.n_pairs.toLocaleString()} teammate gaps
          </p>
        </div>
      </div>
      <section className="panel trend-panel" aria-labelledby="error-title">
        <div className="panel-heading">
          <div>
            <h2 id="error-title">Error by race</h2>
            <p>
              Teammate gap error per qualifying, seconds per 90s lap · lower is
              better
            </p>
          </div>
          <div className="chart-legend" aria-hidden="true">
            <span>
              <i style={{ background: SERIES.driver }} />
              Model
            </span>
            <span>
              <i style={{ background: "var(--line-strong)" }} />
              Last season’s gap
            </span>
          </div>
        </div>
        <div className="chart-wrap">
          <svg
            viewBox={`0 0 ${W} 270`}
            ref={ref}
            className="trend-chart"
            role="group"
            aria-label="Forecast error by race for the model and the last-season baseline. Focus a race for values."
            onMouseLeave={() => setHover(null)}
          >
            {niceTicks(0, hi).map((v) => (
              <g key={v}>
                <line
                  x1="65"
                  x2={W - 30}
                  y1={y(v)}
                  y2={y(v)}
                  stroke="var(--line)"
                />
                <text x="50" y={y(v) + 4} textAnchor="end">
                  {v.toFixed(2)}
                </text>
              </g>
            ))}
            <path
              d={path("rmse_naive")}
              fill="none"
              stroke="var(--line-strong)"
              strokeWidth="2"
            />
            <path
              d={path("rmse")}
              fill="none"
              stroke={SERIES.driver}
              strokeWidth="2"
            />
            {events.map((e, i) => (
              <g key={e.event.event_id}>
                <circle
                  cx={x(i)}
                  cy={y(e.rmse_naive!)}
                  r="4"
                  fill="var(--line-strong)"
                  stroke="var(--surface)"
                  strokeWidth="2"
                />
                <circle
                  cx={x(i)}
                  cy={y(e.rmse!)}
                  r="4"
                  fill={SERIES.driver}
                  stroke="var(--surface)"
                  strokeWidth="2"
                />
                {i % every === 0 && (
                  <text x={x(i)} y="250" textAnchor="middle">
                    {String(Number(e.event.event_id.slice(5))).padStart(2, "0")}
                  </text>
                )}
                {every === 1 && (
                  <text
                    x={x(i)}
                    y="264"
                    textAnchor="middle"
                    className="axis-sub"
                  >
                    {circuitCode(circuitOf.get(e.event.event_id))}
                  </text>
                )}
                <rect
                  x={x(i) - step / 2}
                  y="20"
                  width={step}
                  height="215"
                  fill="transparent"
                  tabIndex={0}
                  role="button"
                  aria-label={`${e.event.race_name}: model ${e.rmse!.toFixed(3)} seconds, last season's gap ${e.rmse_naive!.toFixed(3)} seconds`}
                  onMouseEnter={() => setHover(i)}
                  onFocus={() => setHover(i)}
                  onBlur={() => setHover(null)}
                />
              </g>
            ))}
            {hover !== null && (
              <line
                x1={x(hover)}
                x2={x(hover)}
                y1="20"
                y2="230"
                stroke="var(--muted)"
                pointerEvents="none"
              />
            )}
          </svg>
          <div className="chart-readout" aria-live="polite">
            {hover !== null ? (
              <>
                <strong>{events[hover].event.race_name}</strong>
                <span>model {events[hover].rmse!.toFixed(3)}s</span>
                <span>
                  last season’s gap {events[hover].rmse_naive!.toFixed(3)}s
                </span>
                <span>coverage {pct(events[hover].coverage90!)}</span>
                <span>order {events[hover].order_spearman?.toFixed(2)}</span>
              </>
            ) : (
              <span>Race rounds · select one for values</span>
            )}
          </div>
        </div>
        <details className="table-view">
          <summary>Show as table</summary>
          <table>
            <thead>
              <tr>
                <th>Race</th>
                <th className="numeric">Gaps</th>
                <th className="numeric">Model</th>
                <th className="numeric">Last season</th>
                <th className="numeric">No gap</th>
                <th className="numeric">Coverage</th>
                <th className="numeric">Order</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e) => (
                <tr key={e.event.event_id}>
                  <td>{e.event.race_name}</td>
                  <td className="numeric">{e.n_pairs}</td>
                  <td className="numeric">{e.rmse!.toFixed(3)}</td>
                  <td className="numeric">{e.rmse_naive!.toFixed(3)}</td>
                  <td className="numeric">{e.rmse_zero!.toFixed(3)}</td>
                  <td className="numeric">{pct(e.coverage90!)}</td>
                  <td className="numeric">{e.order_spearman?.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      </section>
      {latest && (
        <section className="panel chart-panel" aria-labelledby="latest-title">
          <div className="panel-heading">
            <div>
              <h2 id="latest-title">
                Latest forecast · {latest.event.race_name}
              </h2>
              <p>
                Q1 teammate gaps predicted from races through{" "}
                {latest.trained_through.race_name} (
                {date(latest.trained_through.date)})
              </p>
            </div>
          </div>
          <div
            className="forecast-scroll"
            tabIndex={0}
            role="region"
            aria-label="Latest teammate gap forecasts"
          >
            <table className="forecast-table">
              <thead>
                <tr>
                  <th>Teammates</th>
                  <th className="numeric">Predicted</th>
                  <th className="numeric">90% interval</th>
                  <th className="numeric">Actual</th>
                  <th>Inside</th>
                </tr>
              </thead>
              <tbody>
                {pairs.map((r) => {
                  const inside = r.q05 <= r.observed && r.observed <= r.q95;
                  return (
                    <tr key={`${r.a}-${r.b}`}>
                      <td>
                        {label(r.a)} vs {label(r.b)}
                        {!r.established && (
                          <Badge tone="amber">Not scored</Badge>
                        )}
                      </td>
                      <td className="numeric">{signed(r.predicted)}s</td>
                      <td className="numeric">
                        {signed(r.q05)} to {signed(r.q95)}
                      </td>
                      <td className="numeric">{signed(r.observed)}s</td>
                      <td>
                        {inside ? (
                          <span className="gate-pass">
                            <Check size={14} /> Yes
                          </span>
                        ) : (
                          <span className="gate-fail">
                            <X size={14} /> No
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="panel-foot">
            <span>
              Positive = first driver faster. Pairs with a driver who had fewer
              than 10 earlier qualifying sessions are shown but not scored.
            </span>
          </div>
        </section>
      )}
      <details className="panel metric-notes forecast-method">
        <summary>How forecasts are made and scored</summary>
        <p>
          For each race, the qualifying model is refitted on every qualifying
          session up to the race before it. That fit predicts the gap between
          teammates in each segment of the next qualifying, including a new
          one-weekend form draw for each driver and the fitted session noise,
          and the order of the whole field from car, circuit and driver terms.
          The race’s own lap times never enter its forecast.
        </p>
        <p>
          The error is the root mean square difference between predicted and
          real teammate gaps, in seconds per 90-second lap. “Last season’s gap”
          repeats each pairing’s mean gap from their latest season together (or
          each driver’s gap to past teammates). Coverage is the share of real
          gaps inside the 90% predictive interval. Records for 2026 races before
          October 2026 were computed retrospectively with the same code; later
          records are added by the weekly refresh after each race. The longer
          historical benchmark (25 cutoffs since 2013) is on Model health.
        </p>
      </details>
    </>
  );
}
