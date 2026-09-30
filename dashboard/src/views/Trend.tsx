import { useEffect, useState } from "react";
import type { Dataset, Metric, Estimate } from "../types";
import { signed, shortName } from "../lib";
import { Empty } from "../components";

export default function Trend({
  data,
  car,
  metric,
  initial,
}: {
  data: Dataset;
  car: boolean;
  metric: Metric;
  initial: string;
}) {
  const entities = car ? data.catalog.cars : data.catalog.drivers;
  const current = car ? data.cars : data.drivers;
  const [a, setA] = useState(initial);
  const [b, setB] = useState(current.find((e) => e.id !== initial)?.id || "");
  const latestYear = data.events.at(-1)!.season;
  const [season, setSeason] = useState(String(latestYear));
  const [bands, setBands] = useState(true);
  const [hover, setHover] = useState<number | null>(null);
  const [circuit, setCircuit] = useState(false);
  useEffect(() => {
    setA(initial);
    if (b === initial) setB(entities.find((e) => e.id !== initial)?.id || "");
  }, [initial]);
  const events = data.events.filter(
    (e) => season === "all" || e.season === Number(season),
  );
  const history = car ? data.history.cars : data.history.drivers;
  const key = car ? (circuit ? "at_circuit" : "pace") : metric;
  const series = [a, b].filter(Boolean).map((id, i) => ({
    id,
    name: entities.find((e) => e.id === id)?.name || id,
    color: i === 0 ? "#286b55" : "#a88144",
    points: events.map(
      (e) => history[id]?.find((p) => p.event === e.event_id)?.[key],
    ),
  }));
  const values = series.flatMap((s) =>
    s.points.filter((p): p is Estimate => !!p),
  );
  const low = Math.min(0, ...values.map((v) => (bands ? v.q05 : v.median)));
  const high = Math.max(0, ...values.map((v) => (bands ? v.q95 : v.median)));
  const pad = Math.max((high - low) * 0.14, 0.05);
  const ymin = low - pad;
  const ymax = high + pad;
  const x = (i: number) => 65 + (i / Math.max(events.length - 1, 1)) * 835;
  const y = (v: number) => 255 - ((v - ymin) / (ymax - ymin)) * 215;
  const segments = (points: (Estimate | undefined)[]) => {
    const groups: { v: Estimate; i: number }[][] = [];
    points.forEach((v, i) => {
      if (!v) return;
      if (!points[i - 1]) groups.push([]);
      groups.at(-1)!.push({ v, i });
    });
    return groups;
  };
  return (
    <section className="panel trend-panel">
      <div className="panel-heading">
        <div>
          <h2>Pace history</h2>
          <p>Revised estimates</p>
        </div>
        <label className="inline-label">
          Season
          <select
            aria-label="Trend season"
            value={season}
            onChange={(e) => {
              setSeason(e.target.value);
              setHover(null);
            }}
          >
            {[...new Set(data.events.map((e) => e.season))]
              .reverse()
              .map((s) => (
                <option key={s}>{s}</option>
              ))}
            <option value="all">All seasons</option>
          </select>
        </label>
      </div>
      <div className="trend-controls">
        <label>
          <span className="series-dot" style={{ background: "#286b55" }} />
          <select
            aria-label="First trend entry"
            value={a}
            onChange={(e) => {
              setA(e.target.value);
              if (e.target.value === b) setB("");
            }}
          >
            {entities
              .filter((e) => e.id !== b)
              .map((e) => (
                <option value={e.id} key={e.id}>
                  {e.name}
                </option>
              ))}
          </select>
        </label>
        <label>
          <span className="series-dot" style={{ background: "#a88144" }} />
          <select
            aria-label="Second trend entry"
            value={b}
            onChange={(e) => setB(e.target.value)}
          >
            <option value="">No comparison</option>
            {entities
              .filter((e) => e.id !== a)
              .map((e) => (
                <option value={e.id} key={e.id}>
                  {e.name}
                </option>
              ))}
          </select>
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={bands}
            onChange={(e) => setBands(e.target.checked)}
          />
          90% intervals
        </label>
        {car && (
          <label className="checkbox">
            <input
              type="checkbox"
              checked={circuit}
              onChange={(e) => setCircuit(e.target.checked)}
            />
            Circuit-adjusted
          </label>
        )}
      </div>
      {!values.length ? (
        <Empty title="No estimates in this season">
          Choose another season or entry.
        </Empty>
      ) : (
        <div className="chart-wrap">
          <div className="chart-unit">
            SECONDS / 90s LAP <span>↑ Faster</span>
          </div>
          <svg
            viewBox="0 0 930 300"
            className="trend-chart"
            role="group"
            aria-label={`Historical pace for ${series.map((s) => s.name).join(" and ")}, ${season}. Higher is faster. Focus on an event to inspect values.`}
            onMouseLeave={() => setHover(null)}
          >
            {[0, 1, 2, 3, 4].map((i) => {
              const v = ymin + ((ymax - ymin) * i) / 4;
              return (
                <g key={i}>
                  <line x1="65" x2="900" y1={y(v)} y2={y(v)} stroke="#e8ebe5" />
                  <text x="50" y={y(v) + 4} textAnchor="end">
                    {signed(v, 2)}
                  </text>
                </g>
              );
            })}
            <line
              x1="65"
              x2="900"
              y1={y(0)}
              y2={y(0)}
              stroke="#a7b3a7"
              strokeDasharray="4 4"
            />
            {series.map((s) => (
              <g key={s.id}>
                {segments(s.points).map((seg, j) => (
                  <g key={j}>
                    {bands && (
                      <path
                        d={`M${seg.map((p) => `${x(p.i)},${y(p.v.q95)}`).join(" L")} L${[
                          ...seg,
                        ]
                          .reverse()
                          .map((p) => `${x(p.i)},${y(p.v.q05)}`)
                          .join(" L")} Z`}
                        fill={s.color}
                        opacity=".10"
                      />
                    )}
                    <path
                      d={`M${seg.map((p) => `${x(p.i)},${y(p.v.median)}`).join(" L")}`}
                      fill="none"
                      stroke={s.color}
                      strokeWidth="2.5"
                      strokeLinejoin="round"
                    />
                    {seg.length === 1 && (
                      <circle
                        cx={x(seg[0].i)}
                        cy={y(seg[0].v.median)}
                        r="4"
                        fill={s.color}
                      />
                    )}
                  </g>
                ))}
              </g>
            ))}
            {events.map((e, i) => (
              <g key={e.event_id}>
                {(events.length < 26 ||
                  i % Math.ceil(events.length / 9) === 0 ||
                  i === events.length - 1) && (
                  <text x={x(i)} y="282" textAnchor="middle">
                    {season === "all"
                      ? e.season
                      : String(e.round).padStart(2, "0")}
                  </text>
                )}
                <rect
                  x={x(i) - Math.max(2, 417 / events.length)}
                  y="32"
                  width={Math.max(4, 834 / events.length)}
                  height="230"
                  fill="transparent"
                  tabIndex={0}
                  role="button"
                  aria-label={`${e.race_name} ${e.season}: ${series.map((s) => (s.points[i] ? `${s.name} ${signed(s.points[i]!.median)} seconds` : `${s.name} no estimate`)).join(", ")}`}
                  onFocus={() => setHover(i)}
                  onBlur={() => setHover(null)}
                  onMouseEnter={() => setHover(i)}
                  onClick={() => setHover(i)}
                />
              </g>
            ))}
            {hover !== null && (
              <g pointerEvents="none">
                <line
                  x1={x(hover)}
                  x2={x(hover)}
                  y1="32"
                  y2="255"
                  stroke="#68786a"
                  strokeDasharray="3 3"
                />
                {series.map(
                  (s) =>
                    s.points[hover] && (
                      <circle
                        key={s.id}
                        cx={x(hover)}
                        cy={y(s.points[hover]!.median)}
                        r="5"
                        fill={s.color}
                        stroke="white"
                        strokeWidth="2"
                      />
                    ),
                )}
              </g>
            )}
          </svg>
          <div className="chart-readout" aria-live="polite">
            {hover !== null ? (
              <>
                <strong>
                  {events[hover].race_name} · {events[hover].season}
                </strong>
                {series.map((s) => (
                  <span key={s.id} style={{ color: s.color }}>
                    {shortName(s.name)}:{" "}
                    {s.points[hover]
                      ? `${signed(s.points[hover]!.median)}s`
                      : "no estimate"}
                  </span>
                ))}
              </>
            ) : (
              <span>
                {season === "all" ? "Race events" : "Race rounds"} · select an
                event for values
              </span>
            )}
          </div>
        </div>
      )}
      <div className="panel-foot">
        <span>Relative to each event’s field, revised using later data.</span>
      </div>
    </section>
  );
}
