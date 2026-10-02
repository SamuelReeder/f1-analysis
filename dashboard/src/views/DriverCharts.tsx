import { useCallback, useRef, useState } from "react";
import type { Dataset, Estimate } from "../types";
import { color, niceTicks, signed, shortName, SERIES } from "../lib";
import { Empty, useWidth } from "../components";

// Each driver's expected qualifying pace at an average circuit, split into the car's
// part and the driver's part. Both are relative to the field, so they add.
export function PaceBreakdown({
  data,
  selected,
  onSelect,
}: {
  data: Dataset;
  selected: string;
  onSelect: (id: string) => void;
}) {
  const [hover, setHover] = useState<string | null>(null);
  const [ref, W] = useWidth();
  const rows = data.breakdown || [];
  if (!rows.length) return null;
  const names = new Map(data.drivers.map((d) => [d.id, d]));
  const cars = new Map(data.cars.map((c) => [c.id, c.name]));
  const extent = rows.flatMap((r) => [
    0,
    r.car,
    r.car + r.driver,
    r.total.q05,
    r.total.q95,
  ]);
  const lo = Math.min(...extent) - 0.1;
  const hi = Math.max(...extent) + 0.1;
  const left = W < 520 ? 96 : 150;
  const width = W - left - 20;
  const x = (v: number) => left + ((v - lo) / (hi - lo)) * width;
  const rowH = 30;
  const top = 26;
  const height = top + rows.length * rowH + 30;
  const focus = rows.find((r) => r.id === (hover || selected)) || rows[0];
  const fd = names.get(focus.id);
  return (
    <section className="panel chart-panel" aria-labelledby="breakdown-title">
      <div className="panel-heading">
        <div>
          <h2 id="breakdown-title">Car and driver</h2>
          <p>
            Expected qualifying pace at an average circuit, split into the car
            and the driver
          </p>
        </div>
        <div className="chart-legend" aria-hidden="true">
          <span>
            <i style={{ background: SERIES.car }} />
            Car
          </span>
          <span>
            <i style={{ background: SERIES.driver }} />
            Driver
          </span>
          <span>
            <i className="legend-total" />
            Total, 90% interval
          </span>
        </div>
      </div>
      <div className="chart-wrap">
        <svg
          viewBox={`0 0 ${W} ${height}`}
          ref={ref}
          className="breakdown-chart"
          role="group"
          aria-label="Car and driver contributions to expected qualifying pace for each driver. Focus a row for values."
          onMouseLeave={() => setHover(null)}
        >
          {niceTicks(lo, hi).map((v) => (
            <g key={v}>
              <line
                x1={x(v)}
                x2={x(v)}
                y1={top - 6}
                y2={height - 26}
                stroke="var(--line)"
              />
              <text x={x(v)} y={height - 10} textAnchor="middle">
                {signed(v, 2)}
              </text>
            </g>
          ))}
          <line
            x1={x(0)}
            x2={x(0)}
            y1={top - 6}
            y2={height - 26}
            stroke="var(--line-strong)"
          />
          {rows.map((r, i) => {
            const d = names.get(r.id);
            const cy = top + i * rowH + rowH / 2;
            const end = r.car + r.driver;
            const active = r.id === focus.id;
            return (
              <g key={r.id} opacity={active || !hover ? 1 : 0.45}>
                <rect
                  x={0}
                  y={cy - rowH / 2}
                  width={W}
                  height={rowH}
                  fill={active ? "var(--accent-soft)" : "transparent"}
                />
                <rect
                  x={2}
                  y={cy - 7}
                  width={3}
                  height={14}
                  fill={color(r.team)}
                />
                <text x={12} y={cy + 4} className="row-label">
                  {d ? shortName(d.name) : r.id}
                </text>
                {/* Waterfall: the car moves the driver from 0, then the driver
                    part starts where the car ends, one step lower. */}
                <rect
                  x={Math.min(x(0), x(r.car))}
                  y={cy - 9}
                  width={Math.max(Math.abs(x(r.car) - x(0)), 1)}
                  height={6}
                  rx={1.5}
                  fill={SERIES.car}
                />
                <rect
                  x={Math.min(x(r.car), x(end))}
                  y={cy - 1}
                  width={Math.max(Math.abs(x(end) - x(r.car)), 1)}
                  height={6}
                  rx={1.5}
                  fill={SERIES.driver}
                />
                <line
                  x1={x(r.total.q05)}
                  x2={x(r.total.q95)}
                  y1={cy + 10}
                  y2={cy + 10}
                  stroke="var(--ink)"
                  strokeWidth="1.5"
                />
                <circle
                  cx={x(r.total.median)}
                  cy={cy + 10}
                  r="3.5"
                  fill="var(--ink)"
                  stroke="var(--surface)"
                  strokeWidth="1.5"
                />
                <rect
                  x={0}
                  y={cy - rowH / 2}
                  width={W}
                  height={rowH}
                  fill="transparent"
                  tabIndex={0}
                  role="button"
                  aria-label={`${d?.name || r.id}: car ${signed(r.car)}, driver ${signed(r.driver)}, total ${signed(r.total.median)} seconds`}
                  onMouseEnter={() => setHover(r.id)}
                  onFocus={() => setHover(r.id)}
                  onBlur={() => setHover(null)}
                  onClick={() => onSelect(r.id)}
                />
              </g>
            );
          })}
        </svg>
        <div className="chart-readout" aria-live="polite">
          <strong>{fd?.name || focus.id}</strong>
          <span>
            {cars.get(focus.team) || fd?.team} car {signed(focus.car)}s
          </span>
          <span>driver {signed(focus.driver)}s</span>
          <span>
            total {signed(focus.total.median)}s ({signed(focus.total.q05)} to{" "}
            {signed(focus.total.q95)})
          </span>
        </div>
      </div>
      <details className="table-view">
        <summary>Show as table</summary>
        <table>
          <thead>
            <tr>
              <th>Driver</th>
              <th>Car</th>
              <th className="numeric">Car</th>
              <th className="numeric">Driver</th>
              <th className="numeric">Total</th>
              <th className="numeric">90% interval</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td>{names.get(r.id)?.name || r.id}</td>
                <td>{cars.get(r.team) || r.team}</td>
                <td className="numeric">{signed(r.car)}</td>
                <td className="numeric">{signed(r.driver)}</td>
                <td className="numeric">{signed(r.total.median)}</td>
                <td className="numeric">
                  {signed(r.total.q05)} to {signed(r.total.q95)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
      <div className="panel-foot">
        <span>
          Seconds per 90-second lap against an average driver in an average car.
          The driver part is in-team pace; the total’s interval combines both
          parts draw by draw.
        </span>
      </div>
    </section>
  );
}

// Published qualifying pace (in the current car) against published race pace.
export function QualiVsRace({
  data,
  selected,
  onSelect,
}: {
  data: Dataset;
  selected: string;
  onSelect: (id: string) => void;
}) {
  const [hover, setHover] = useState<string | null>(null);
  const [widthRef, W] = useWidth();
  const svg = useRef<SVGSVGElement | null>(null);
  const svgRef = useCallback(
    (element: SVGSVGElement | null) => {
      svg.current = element;
      widthRef(element);
    },
    [widthRef],
  );
  const race = new Map((data.race_pace?.drivers || []).map((r) => [r.id, r]));
  const points = data.drivers
    .filter((d) => race.has(d.id))
    .map((d) => ({
      id: d.id,
      name: d.name,
      code: d.code,
      lineage: d.lineage,
      q: d.headline,
      r: race.get(d.id)!.pace,
    }));
  if (points.length < 3) return null;
  const span = (vals: number[]) => {
    const lo = Math.min(0, ...vals);
    const hi = Math.max(0, ...vals);
    const pad = (hi - lo) * 0.08;
    return [lo - pad, hi + pad];
  };
  const [xlo, xhi] = span(points.flatMap((p) => [p.q.q05, p.q.q95]));
  const [ylo, yhi] = span(points.flatMap((p) => [p.r.q05, p.r.q95]));
  const H = W < 520 ? 320 : 380;
  const [left, right, top, bottom] = [70, W - 30, 30, H - 50];
  const X = (v: number) => left + ((v - xlo) / (xhi - xlo)) * (right - left);
  const Y = (v: number) => bottom - ((v - ylo) / (yhi - ylo)) * (bottom - top);
  const focus = points.find((p) => p.id === (hover || selected)) || points[0];
  // Label greedily (focused driver first, then the most isolated) and skip a label
  // that would collide; the readout and each point's accessible name carry the rest.
  const placed: { x: number; y: number; w: number }[] = [];
  const labels = new Map<
    string,
    { x: number; y: number; anchor: "start" | "end" }
  >();
  const centre = {
    x: points.reduce((t, p) => t + X(p.q.median), 0) / points.length,
    y: points.reduce((t, p) => t + Y(p.r.median), 0) / points.length,
  };
  const spread = (p: (typeof points)[number]) =>
    Math.hypot(X(p.q.median) - centre.x, Y(p.r.median) - centre.y);
  [...points]
    .sort((a, b) =>
      a.id === focus.id ? -1 : b.id === focus.id ? 1 : spread(b) - spread(a),
    )
    .forEach((p) => {
      const px = X(p.q.median);
      const py = Y(p.r.median);
      const w = p.code.length * 6.5;
      const free = (x: number, y: number) =>
        placed.every(
          (o) => x + w < o.x || x > o.x + o.w || Math.abs(y - o.y) > 11,
        ) &&
        points.every(
          (o) =>
            o.id === p.id ||
            X(o.q.median) < x - 6 ||
            X(o.q.median) > x + w + 6 ||
            Y(o.r.median) < y - 15 ||
            Y(o.r.median) > y + 5,
        );
      for (const [dx, dy, anchor] of [
        [8, -6, "start"],
        [8, 13, "start"],
        [-8, -6, "end"],
        [-8, 13, "end"],
      ] as const) {
        const left = anchor === "start" ? px + dx : px + dx - w;
        if (free(left, py + dy)) {
          placed.push({ x: left, y: py + dy, w });
          labels.set(p.id, { x: px + dx, y: py + dy, anchor });
          break;
        }
      }
    });
  const nearest = (e: React.MouseEvent) => {
    const m = svg.current?.getScreenCTM();
    if (!m) return null;
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());
    let best: string | null = null;
    let distance = 30;
    points.forEach((p) => {
      const d = Math.hypot(X(p.q.median) - pt.x, Y(p.r.median) - pt.y);
      if (d < distance) [best, distance] = [p.id, d];
    });
    return best;
  };
  const whisker = (p: (typeof points)[number]) => (
    <g pointerEvents="none">
      <line
        x1={X(p.q.q05)}
        x2={X(p.q.q95)}
        y1={Y(p.r.median)}
        y2={Y(p.r.median)}
        stroke="var(--ink)"
        opacity=".55"
      />
      <line
        x1={X(p.q.median)}
        x2={X(p.q.median)}
        y1={Y(p.r.q05)}
        y2={Y(p.r.q95)}
        stroke="var(--ink)"
        opacity=".55"
      />
    </g>
  );
  return (
    <section className="panel chart-panel" aria-labelledby="scatter-title">
      <div className="panel-heading">
        <div>
          <h2 id="scatter-title">Qualifying and race pace</h2>
          <p>
            Each driver’s published qualifying pace against their dry-race pace
          </p>
        </div>
      </div>
      <div className="chart-wrap">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="scatter-chart"
          role="group"
          aria-label="Driver qualifying pace against driver race pace. Focus a point for values."
          ref={svgRef}
          onMouseMove={(e) => setHover(nearest(e))}
          onMouseLeave={() => setHover(null)}
          onClick={(e) => {
            const id = nearest(e);
            if (id) onSelect(id);
          }}
        >
          {niceTicks(xlo, xhi).map((v) => (
            <g key={`x${v}`}>
              <line
                x1={X(v)}
                x2={X(v)}
                y1={top}
                y2={bottom}
                stroke="var(--line)"
              />
              <text x={X(v)} y={bottom + 18} textAnchor="middle">
                {signed(v, 2)}
              </text>
            </g>
          ))}
          {niceTicks(ylo, yhi).map((v) => (
            <g key={`y${v}`}>
              <line
                x1={left}
                x2={right}
                y1={Y(v)}
                y2={Y(v)}
                stroke="var(--line)"
              />
              <text x={60} y={Y(v) + 3} textAnchor="end">
                {signed(v, 2)}
              </text>
            </g>
          ))}
          <line
            x1={X(0)}
            x2={X(0)}
            y1={top}
            y2={bottom}
            stroke="var(--line-strong)"
          />
          <line
            x1={left}
            x2={right}
            y1={Y(0)}
            y2={Y(0)}
            stroke="var(--line-strong)"
          />
          <text
            x={right - 4}
            y={top + 14}
            textAnchor="end"
            className="quadrant"
          >
            Faster than average in both
          </text>
          <text x={left + 4} y={bottom - 8} className="quadrant">
            Slower than average in both
          </text>
          <text
            x={(left + right) / 2}
            y={H - 8}
            textAnchor="middle"
            className="axis-title"
          >
            Driver qualifying pace, s per 90s lap →
          </text>
          <text
            x={16}
            y={(top + bottom) / 2}
            textAnchor="middle"
            className="axis-title"
            transform={`rotate(-90 16 ${(top + bottom) / 2})`}
          >
            Driver race pace, s per 90s lap →
          </text>
          {whisker(focus)}
          {points.map((p) => {
            const active = p.id === focus.id;
            return (
              <g key={p.id}>
                <circle
                  cx={X(p.q.median)}
                  cy={Y(p.r.median)}
                  r={active ? 6 : 5}
                  fill={color(p.lineage)}
                  stroke={active ? "var(--ink)" : "var(--surface)"}
                  strokeWidth="2"
                />
                {labels.has(p.id) && (
                  <text
                    x={labels.get(p.id)!.x}
                    y={labels.get(p.id)!.y}
                    textAnchor={labels.get(p.id)!.anchor}
                    className={active ? "point-label active" : "point-label"}
                  >
                    {p.code}
                  </text>
                )}
                <circle
                  cx={X(p.q.median)}
                  cy={Y(p.r.median)}
                  r="12"
                  fill="transparent"
                  tabIndex={0}
                  role="button"
                  aria-label={`${p.name}: qualifying ${signed(p.q.median)}, race ${signed(p.r.median)} seconds`}
                  onFocus={() => setHover(p.id)}
                  onBlur={() => setHover(null)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") onSelect(p.id);
                  }}
                />
              </g>
            );
          })}
        </svg>
        <div className="chart-readout" aria-live="polite">
          <strong>{focus.name}</strong>
          <span>
            qualifying {signed(focus.q.median)}s ({signed(focus.q.q05)} to{" "}
            {signed(focus.q.q95)})
          </span>
          <span>
            race {signed(focus.r.median)}s ({signed(focus.r.q05)} to{" "}
            {signed(focus.r.q95)})
          </span>
        </div>
      </div>
      <div className="panel-foot">
        <span>
          Separate models, both for the driver with the car estimated
          separately: one-lap qualifying pace (in-team, as in the ranking) and
          dry-race pace at tyre age 10 laps. Both are relative to the field; the
          lines show the focused driver’s 90% intervals. Race pace data as of{" "}
          {data.race_pace?.data_as_of.race_name}.
        </span>
      </div>
    </section>
  );
}

// Driver race pace in each season since 2018 (revised with all data).
export function RaceHistory({
  data,
  initial,
}: {
  data: Dataset;
  initial: string;
}) {
  const seasons = (data.race_pace?.seasons || []).filter(
    (s) => s.drivers.length,
  );
  const drivers = new Map<string, string>();
  seasons.forEach((s) => s.drivers.forEach((d) => drivers.set(d.id, d.name)));
  const options = [...drivers].sort((a, b) => a[1].localeCompare(b[1]));
  const [a, setA] = useState(initial);
  const [b, setB] = useState("");
  const [hover, setHover] = useState<number | null>(null);
  const [ref, W] = useWidth(930);
  if (!seasons.length) return null;
  const pick = (id: string) =>
    seasons.map((s) => s.drivers.find((d) => d.id === id));
  const series = [a, b].filter(Boolean).map((id, i) => ({
    id,
    name: drivers.get(id) || id,
    color: i ? "var(--ink)" : "var(--accent)",
    dash: i ? "7 4" : undefined,
    rows: pick(id),
  }));
  const vals = series.flatMap((s) =>
    s.rows.filter(Boolean).flatMap((r) => [r!.pace.q05, r!.pace.q95]),
  );
  const lo = Math.min(0, ...vals) - 0.05;
  const hi = Math.max(0, ...vals) + 0.05;
  const step = (W - 95) / Math.max(seasons.length - 1, 1);
  const x = (i: number) => 65 + i * step;
  const y = (v: number) => 255 - ((v - lo) / (hi - lo)) * 215;
  const line = (rows: (typeof series)[number]["rows"]) => {
    const parts: string[] = [];
    rows.forEach((r, i) => {
      if (!r) return;
      parts.push(`${rows[i - 1] ? "L" : "M"}${x(i)},${y(r.pace.median)}`);
    });
    return parts.join(" ");
  };
  const est = (e?: Estimate) =>
    e
      ? `${signed(e.median)}s (${signed(e.q05)} to ${signed(e.q95)})`
      : "not rated";
  return (
    <section className="panel trend-panel" aria-labelledby="race-history-title">
      <div className="panel-heading">
        <div>
          <h2 id="race-history-title">Race pace by season</h2>
          <p>Revised estimates · drivers with two or more clean dry races</p>
        </div>
      </div>
      <div className="trend-controls">
        <label>
          <span
            className="series-dot"
            style={{ background: "var(--accent)" }}
          />
          <select
            aria-label="First race-history driver"
            value={a}
            onChange={(e) => {
              setA(e.target.value);
              if (e.target.value === b) setB("");
            }}
          >
            {options.map(([id, name]) => (
              <option key={id} value={id}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span className="series-dot series-dashed" />
          <select
            aria-label="Second race-history driver"
            value={b}
            onChange={(e) => setB(e.target.value)}
          >
            <option value="">No comparison</option>
            {options
              .filter(([id]) => id !== a)
              .map(([id, name]) => (
                <option key={id} value={id}>
                  {name}
                </option>
              ))}
          </select>
        </label>
      </div>
      {!vals.length ? (
        <Empty title="No race-pace seasons for this driver">
          Choose another driver.
        </Empty>
      ) : (
        <div className="chart-wrap">
          <div className="chart-unit">
            SECONDS / 90s LAP <span>↑ Faster</span>
          </div>
          <svg
            viewBox={`0 0 ${W} 300`}
            ref={ref}
            className="trend-chart"
            role="group"
            aria-label={`Race pace by season for ${series.map((s) => s.name).join(" and ")}. Focus a season for values.`}
            onMouseLeave={() => setHover(null)}
          >
            {niceTicks(lo, hi).map((v) => (
              <g key={v}>
                <line
                  x1="65"
                  x2={W - 30}
                  y1={y(v)}
                  y2={y(v)}
                  stroke="var(--line)"
                />
                <text x="50" y={y(v) + 4} textAnchor="end">
                  {signed(v, 2)}
                </text>
              </g>
            ))}
            <line
              x1="65"
              x2={W - 30}
              y1={y(0)}
              y2={y(0)}
              stroke="var(--line-strong)"
            />
            {series.map((s) => (
              <g key={s.id}>
                {s.rows.map(
                  (r, i) =>
                    r && (
                      <line
                        key={i}
                        x1={x(i)}
                        x2={x(i)}
                        y1={y(r.pace.q05)}
                        y2={y(r.pace.q95)}
                        stroke={s.color}
                        strokeWidth="6"
                        opacity=".18"
                        strokeLinecap="round"
                      />
                    ),
                )}
                <path
                  d={line(s.rows)}
                  fill="none"
                  stroke={s.color}
                  strokeWidth="2"
                  strokeDasharray={s.dash}
                />
                {s.rows.map(
                  (r, i) =>
                    r && (
                      <circle
                        key={i}
                        cx={x(i)}
                        cy={y(r.pace.median)}
                        r="4"
                        fill={s.color}
                        stroke="var(--surface)"
                        strokeWidth="2"
                      />
                    ),
                )}
              </g>
            ))}
            {seasons.map((s, i) => (
              <g key={s.season}>
                <text x={x(i)} y="282" textAnchor="middle">
                  {s.season}
                </text>
                <rect
                  x={x(i) - step / 2}
                  y="32"
                  width={step}
                  height="230"
                  fill="transparent"
                  tabIndex={0}
                  role="button"
                  aria-label={`${s.season}: ${series.map((ser) => `${ser.name} ${est(ser.rows[i]?.pace)}`).join(", ")}`}
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
                y1="32"
                y2="255"
                stroke="var(--muted)"
                pointerEvents="none"
              />
            )}
          </svg>
          <div className="chart-readout" aria-live="polite">
            {hover !== null ? (
              <>
                <strong>{seasons[hover].season}</strong>
                {series.map((s) => (
                  <span key={s.id}>
                    {shortName(s.name)}
                    {s.rows[hover] ? ` (${s.rows[hover]!.team})` : ""}:{" "}
                    {est(s.rows[hover]?.pace)}
                  </span>
                ))}
              </>
            ) : (
              <span>Seasons · select one for values</span>
            )}
          </div>
        </div>
      )}
      <div className="panel-foot">
        <span>
          Each season is centred on that season’s rated drivers, so values
          compare drivers within a season, not absolute speed across years. The
          held-out checks cover mid-season windows in 2024–2026.
        </span>
      </div>
    </section>
  );
}
