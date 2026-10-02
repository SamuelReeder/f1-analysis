import { useEffect, useState } from "react";
import type { Dataset, Metric, Estimate } from "../types";
import {
  signed,
  shortName,
  circuitCode,
  niceTicks,
  REGULATION_RESETS,
} from "../lib";
import { Empty, useWidth } from "../components";

type Mode = "revised" | "asof";
type Scale = "pace" | "rank";

// Ranks are drawn as negative values so that, as for pace, higher is faster.
const asRank = (e?: Estimate): Estimate | undefined =>
  e &&
  e.rank_median !== undefined &&
  e.rank_lo !== undefined &&
  e.rank_hi !== undefined
    ? { ...e, median: -e.rank_median, q05: -e.rank_hi, q95: -e.rank_lo }
    : undefined;
const rankText = (e: Estimate) =>
  `P${Math.round(-e.median)} (P${-e.q95}–P${-e.q05})`;

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
  const [ref, W] = useWidth(930);
  const [circuit, setCircuit] = useState(false);
  const asof = data.asof?.series[car ? "cars" : "drivers"];
  const [mode, setMode] = useState<Mode>("revised");
  const [scale, setScale] = useState<Scale>("pace");
  // Only the revised history carries a rank distribution at every event.
  const ranked = scale === "rank" && mode === "revised";
  useEffect(() => {
    setA(initial);
    if (b === initial) setB(entities.find((e) => e.id !== initial)?.id || "");
  }, [initial]);
  // Seasons with at least two after-race points (the first record of a season is
  // trained through the previous season's finale, which alone is not a history).
  // The latest published estimate is itself the value after the latest race.
  const asofEvents = new Set(
    Object.values(asof || {}).flatMap((points) => points.map((p) => p.event)),
  );
  if (asofEvents.size) asofEvents.add(data.events.at(-1)!.event_id);
  const asofSeasons = new Set(
    [...asofEvents]
      .map((e) => e.slice(0, 4))
      .filter((s, _, all) => all.filter((x) => x === s).length >= 2),
  );
  const hasAsof = asofSeasons.size > 0;
  const seasons = [...new Set(data.events.map((e) => e.season))]
    .reverse()
    .filter((s) => mode === "revised" || asofSeasons.has(String(s)));
  const events = data.events.filter(
    (e) => season === "all" || e.season === Number(season),
  );
  const key = car
    ? circuit && mode === "revised" && !ranked
      ? "at_circuit"
      : "pace"
    : metric;
  const history = car ? data.history.cars : data.history.drivers;
  const lastEvent = data.events.at(-1)!.event_id;
  // After the latest race the published estimate is itself the as-of value.
  const source =
    mode === "asof"
      ? Object.fromEntries(
          entities.map((e) => [
            e.id,
            [
              ...(asof?.[e.id] || []),
              ...(history[e.id] || []).filter((p) => p.event === lastEvent),
            ],
          ]),
        )
      : history;
  const series = [a, b].filter(Boolean).map((id, i) => {
    const points = events.map((e) =>
      source[id]?.find((p) => p.event === e.event_id),
    );
    // A change of team (or of a car's entrant name) is marked where it first appears.
    const changes: { i: number; team: string }[] = [];
    let previous = "";
    points.forEach((p, j) => {
      if (!p) return;
      if (previous && p.team !== previous) changes.push({ i: j, team: p.team });
      previous = p.team;
    });
    return {
      id,
      name: entities.find((e) => e.id === id)?.name || id,
      color: i === 0 ? "var(--accent)" : "var(--ink)",
      dash: i === 0 ? undefined : "7 4",
      points: points.map((p) => {
        const e = p?.[key] as Estimate | undefined;
        return ranked ? asRank(e) : e;
      }),
      teams: points.map((p) => p?.team),
      changes,
    };
  });
  const values = series.flatMap((s) =>
    s.points.filter((p): p is Estimate => !!p),
  );
  // Pace keeps zero (the field average) in view; ranks run from P1 down.
  const low = Math.min(
    ranked ? -1 : 0,
    ...values.map((v) => (bands ? v.q05 : v.median)),
  );
  const high = ranked
    ? -1
    : Math.max(0, ...values.map((v) => (bands ? v.q95 : v.median)));
  const pad = Math.max((high - low) * 0.14, ranked ? 0.5 : 0.05);
  const ymin = low - pad;
  const ymax = high + pad;
  const step = (W - 95) / Math.max(events.length - 1, 1);
  const x = (i: number) => 65 + i * step;
  // Round labels need ~30px each; circuit codes only when every round is labelled.
  // Across all seasons, label each season's first race, thinned to fit.
  const every = Math.max(1, Math.ceil(30 / step));
  const firsts = events
    .map((e, i) => (i === 0 || events[i - 1].season !== e.season ? i : -1))
    .filter((i) => i >= 0);
  const seasonEvery = Math.max(1, Math.ceil((firsts.length * 34) / (W - 95)));
  const labelled = (i: number) =>
    season === "all"
      ? firsts.indexOf(i) >= 0 && firsts.indexOf(i) % seasonEvery === 0
      : i % every === 0 || (i === events.length - 1 && i % every >= every / 2);
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
  const resets =
    season === "all"
      ? REGULATION_RESETS.map((s) => ({
          season: s,
          i: events.findIndex((e) => e.season === s),
        })).filter((r) => r.i > 0)
      : [];
  const ticks = ranked
    ? [
        ...new Set([
          -1,
          ...niceTicks(ymin, ymax).filter((v) => Number.isInteger(v) && v < -1),
        ]),
      ]
    : niceTicks(ymin, ymax);
  const describe = (e: Estimate) =>
    ranked
      ? rankText(e)
      : `${signed(e.median)}s (${signed(e.q05)} to ${signed(e.q95)})`;
  const switchMode = (next: Mode) => {
    setMode(next);
    setHover(null);
    if (next === "asof") {
      setSeason(String([...asofSeasons].sort().at(-1) || latestYear));
      setCircuit(false);
    }
  };
  return (
    <section className="panel trend-panel">
      <div className="panel-heading">
        <div>
          <h2>Pace history</h2>
          <p>
            {mode === "revised"
              ? "Revised estimates"
              : "Estimates after each race"}
            {ranked ? " · rank in each event’s field" : ""}
          </p>
        </div>
        <div className="trend-heading-controls">
          {mode === "revised" && (
            <div className="segmented" aria-label="Chart scale">
              <button
                onClick={() => {
                  setScale("pace");
                  setHover(null);
                }}
                aria-pressed={scale === "pace"}
              >
                Pace
              </button>
              <button
                onClick={() => {
                  setScale("rank");
                  setHover(null);
                }}
                aria-pressed={scale === "rank"}
              >
                Rank
              </button>
            </div>
          )}
          {hasAsof && (
            <div className="segmented" aria-label="History type">
              <button
                onClick={() => switchMode("revised")}
                aria-pressed={mode === "revised"}
              >
                Revised
              </button>
              <button
                onClick={() => switchMode("asof")}
                aria-pressed={mode === "asof"}
              >
                After each race
              </button>
            </div>
          )}
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
              {seasons.map((s) => (
                <option key={s}>{s}</option>
              ))}
              {mode === "revised" && <option value="all">All seasons</option>}
            </select>
          </label>
        </div>
      </div>
      <div className="trend-controls">
        <label>
          <span
            className="series-dot"
            style={{ background: "var(--accent)" }}
          />
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
          <span className="series-dot series-dashed" />
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
        {car && mode === "revised" && !ranked && (
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
            {ranked ? "RANK IN THE FIELD" : "SECONDS / 90s LAP"}{" "}
            <span>↑ Faster</span>
          </div>
          <svg
            viewBox={`0 0 ${W} 312`}
            ref={ref}
            className="trend-chart"
            role="group"
            aria-label={`${mode === "asof" ? "Estimates after each race" : ranked ? "Revised rank" : "Revised pace"} for ${series.map((s) => s.name).join(" and ")}, ${season}. Higher is faster. Focus on an event to inspect values.`}
            onMouseLeave={() => setHover(null)}
          >
            {ticks.map((v, i) => {
              return (
                <g key={i}>
                  <line
                    x1="65"
                    x2={W - 30}
                    y1={y(v)}
                    y2={y(v)}
                    stroke="var(--line)"
                  />
                  <text x="50" y={y(v) + 4} textAnchor="end">
                    {ranked ? `P${-v}` : signed(v, 2)}
                  </text>
                </g>
              );
            })}
            {!ranked && (
              <line
                x1="65"
                x2={W - 30}
                y1={y(0)}
                y2={y(0)}
                stroke="var(--line-strong)"
              />
            )}
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
                      strokeDasharray={s.dash}
                      strokeLinejoin="round"
                    />
                    {(seg.length === 1 || mode === "asof") &&
                      seg.map((p) => (
                        <circle
                          key={p.i}
                          cx={x(p.i)}
                          cy={y(p.v.median)}
                          r="4"
                          fill={s.color}
                          stroke="var(--surface)"
                          strokeWidth="2"
                        />
                      ))}
                  </g>
                ))}
              </g>
            ))}
            {resets.map((r) => (
              <g key={r.season} className="chart-marker">
                <line
                  x1={x(r.i) - step / 2}
                  x2={x(r.i) - step / 2}
                  y1="8"
                  y2="255"
                  stroke="var(--line-strong)"
                />
                <text x={x(r.i) - step / 2 + 4} y="16">
                  New rules {r.season}
                </text>
              </g>
            ))}
            {series.flatMap((s, k) =>
              s.changes.map((c) => (
                <g key={`${s.id}-${c.i}`} className="chart-marker">
                  <line
                    x1={x(c.i)}
                    x2={x(c.i)}
                    y1={k ? 252 : 32}
                    y2={k ? 262 : 42}
                    stroke={s.color}
                    strokeWidth="2"
                  />
                  <text x={Math.min(x(c.i) + 4, W - 130)} y={k ? 272 : 28}>
                    {shortName(s.name)} → {c.team}
                  </text>
                </g>
              )),
            )}
            {events.map((e, i) => (
              <g key={e.event_id}>
                {labelled(i) && (
                  <text x={x(i)} y="290" textAnchor="middle">
                    {season === "all"
                      ? e.season
                      : String(e.round).padStart(2, "0")}
                  </text>
                )}
                {season !== "all" && every === 1 && (
                  <text
                    x={x(i)}
                    y="304"
                    textAnchor="middle"
                    className="axis-sub"
                  >
                    {circuitCode(e.circuit_id)}
                  </text>
                )}
                <rect
                  x={x(i) - step / 2}
                  y="32"
                  width={Math.max(step, 2)}
                  height="230"
                  fill="transparent"
                  tabIndex={0}
                  role="button"
                  aria-label={`${e.race_name} ${e.season}: ${series.map((s) => (s.points[i] ? `${s.name} ${ranked ? rankText(s.points[i]!) : `${signed(s.points[i]!.median)} seconds`}` : `${s.name} no estimate`)).join(", ")}`}
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
                  stroke="var(--muted)"
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
                        stroke="var(--surface)"
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
                  <span key={s.id}>
                    {shortName(s.name)}
                    {s.teams[hover] ? ` (${s.teams[hover]})` : ""}:{" "}
                    {s.points[hover]
                      ? describe(s.points[hover]!)
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
        <span>
          {mode === "revised"
            ? ranked
              ? "Median rank in each event’s field with its 90% range, revised using later data."
              : "Relative to each event’s field, revised using later data."
            : "Each point uses only races up to that event, from a separate fit per round; the revised history also uses later races."}
        </span>
      </div>
    </section>
  );
}
