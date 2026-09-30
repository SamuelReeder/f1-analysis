import { useState } from "react";
import {
  ArrowDownToLine,
  ArrowRight,
  CircleHelp,
  Flag,
  Info,
  Layers3,
  Search,
  TrendingUp,
  X,
} from "lucide-react";
import type { Dataset, Driver, Car, Metric, Estimate } from "../types";
import { Badge, Empty, Band, PageHeading, MetricControl } from "../components";
import {
  color,
  signed,
  pct,
  date,
  metricLabel,
  shortName,
  exportCsv,
} from "../lib";
import Trend from "./Trend";

export default function Rankings({
  data,
  car,
  metric,
  setMetric,
}: {
  data: Dataset;
  car: boolean;
  metric: Metric;
  setMetric: (m: Metric) => void;
}) {
  const rows = [...(car ? data.cars : data.drivers)].sort(
    (a, b) => val(b).median - val(a).median,
  );
  function val(r: Driver | Car) {
    return car ? (r as Car).pace : (r as Driver)[metric];
  }
  const [selected, setSelected] = useState(rows[0].id);
  const [search, setSearch] = useState("");
  const [team, setTeam] = useState("all");
  const selectedRow = rows.find((r) => r.id === selected) || rows[0];
  const top = rows[0];
  const topValue = val(top);
  const filtered = rows.filter(
    (r) =>
      `${r.name} ${car ? "" : (r as Driver).team}`
        .toLowerCase()
        .includes(search.toLowerCase()) &&
      (team === "all" || (r as Driver).team === team),
  );
  const domain = [
    Math.min(...rows.map((r) => val(r).q05), 0),
    Math.max(...rows.map((r) => val(r).q95), 0),
  ];
  const e = data.meta.data_as_of;
  return (
    <>
      <PageHeading
        eyebrow={`${e.event_id.slice(0, 4)} · ROUND ${Number(e.event_id.slice(5))} · QUALIFYING`}
        title={car ? "The machinery, measured." : "The drivers, separated."}
        action={
          <button
            className="button"
            onClick={() => exportCsv(rows, car, metric, data)}
          >
            <ArrowDownToLine size={16} />
            Export rankings
          </button>
        }
      >
        {car
          ? "Track-neutral car performance, with the driver contribution accounted for."
          : "Estimated driver pace with car performance accounted for. Read the ranges, not just the rank."}
      </PageHeading>
      <div className="event-line">
        <Badge tone="green">Published estimates</Badge>
        <span>
          Through {e.race_name} <span className="muted">· {date(e.date)}</span>
        </span>
        <a href="#health">
          View evidence <ArrowRight size={14} />
        </a>
      </div>
      <div className="stat-grid">
        <div className="stat-card">
          <div className="stat-label">
            {car ? "LEADING CAR ESTIMATE" : "LEADING DRIVER ESTIMATE"}
            <Flag size={15} />
          </div>
          <div className="stat-name">
            {top.name}
            <span className="tiny-tag">P1</span>
          </div>
          <div className="stat-sub">
            <strong>{signed(topValue.median)}s</strong> relative to the field
            average
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">
            PROBABILITY OF BEING FASTEST
            <TrendingUp size={15} />
          </div>
          <div className="stat-number">
            {pct(topValue.p_fastest)}
            <span>for {car ? top.name : shortName(top.name)}</span>
          </div>
          <div className="stat-sub">
            Model uncertainty · not a race-win probability
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">
            THE EVIDENCE BASE
            <Layers3 size={15} />
          </div>
          <div className="stat-number">
            {data.meta.n_lap_times.toLocaleString()}
            <span>qualifying times</span>
          </div>
          <div className="stat-sub">
            {data.events.length} events · {data.meta.window} · {rows.length}{" "}
            current {car ? "cars" : "drivers"}
          </div>
        </div>
      </div>
      <div className="ranking-layout">
        <section className="panel ranking-panel">
          <div className="panel-heading">
            <div>
              <h2>{car ? "Car performance" : "Driver rankings"}</h2>
              <p>
                {car
                  ? "Average circuit · qualifying pace"
                  : metricLabel(metric)}
              </p>
            </div>
            {!car && <MetricControl metric={metric} setMetric={setMetric} />}
          </div>
          {!car && metric === "portable" && (
            <div className="inline-warning">
              <Info size={16} />
              <span>
                Experimental: portable skill is sensitive to model choices and
                has weaker ranking recovery.
              </span>
            </div>
          )}
          <div className="table-tools">
            <label className="search">
              <Search size={16} />
              <input
                aria-label={car ? "Search cars" : "Search drivers"}
                placeholder={car ? "Search a constructor…" : "Search a driver…"}
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
              {search && (
                <button aria-label="Clear search" onClick={() => setSearch("")}>
                  <X size={14} />
                </button>
              )}
            </label>
            {!car && (
              <select
                aria-label="Filter by team"
                value={team}
                onChange={(e) => setTeam(e.target.value)}
              >
                <option value="all">All teams</option>
                {[...new Set(data.drivers.map((r) => r.team))]
                  .sort()
                  .map((t) => (
                    <option key={t}>{t}</option>
                  ))}
              </select>
            )}
            <span className="table-count">
              {filtered.length} {car ? "cars" : "drivers"}
            </span>
          </div>
          <div
            className="table-scroll"
            tabIndex={0}
            role="region"
            aria-label="Rankings; scroll for all entries"
          >
            <table className="rank-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>{car ? "CONSTRUCTOR" : "DRIVER"}</th>
                  <th className="numeric">PACE / 90s</th>
                  <th className="range-col">90% PACE INTERVAL</th>
                  <th className="numeric">RANK RANGE</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r) => {
                  const v = val(r);
                  const place = rows.indexOf(r) + 1;
                  return (
                    <tr
                      key={r.id}
                      className={r.id === selectedRow.id ? "selected-row" : ""}
                    >
                      <td>
                        <span
                          className={`rank ${place <= 3 ? "top-rank" : ""}`}
                        >
                          {String(place).padStart(2, "0")}
                        </span>
                      </td>
                      <td>
                        <button
                          className="entity-button"
                          onClick={() => setSelected(r.id)}
                          aria-label={`Inspect ${r.name}`}
                          aria-pressed={r.id === selectedRow.id}
                        >
                          <span
                            className="team-stripe"
                            style={{
                              background: color(
                                car ? r.id : (r as Driver).lineage,
                              ),
                            }}
                          />
                          <span>
                            <strong>{r.name}</strong>
                            <small>
                              {car ? "Car package" : (r as Driver).team}
                              {!car &&
                                !(r as Driver).has_time &&
                                " · carried forward"}
                            </small>
                          </span>
                        </button>
                      </td>
                      <td
                        className={`numeric pace-number ${v.median > 0 ? "positive" : ""}`}
                      >
                        {signed(v.median)}
                        <small>s</small>
                      </td>
                      <td
                        className="range-col"
                        style={{
                          color:
                            r.id === selectedRow.id ? "#35795d" : "#819485",
                        }}
                      >
                        <Band value={v} domain={domain} compact />
                      </td>
                      <td className="numeric">
                        <span className="rank-range">
                          {v.rank_lo}–{v.rank_hi}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {!filtered.length && (
            <Empty title="No matching entries">
              Try a different name or team.
            </Empty>
          )}
          <div className="table-foot">
            <span>
              <span className="legend-dot" />
              Median <span className="legend-line" />
              90% interval
            </span>
            <span>Positive = faster · ranked by median</span>
          </div>
        </section>
        <div className="detail-column">
          <Detail
            row={selectedRow}
            value={val(selectedRow)}
            car={car}
            rank={rows.indexOf(selectedRow) + 1}
            metric={metric}
          />
          <div className="context-card">
            <CircleHelp size={19} />
            <h3>What does +0.100s mean?</h3>
            <p>
              An estimated advantage of one tenth on a reference 90-second lap,
              relative to the average {car ? "car" : "driver"} in this field.
            </p>
            <p>
              {car
                ? "The car rating covers the whole package. Pit crew performance and reliability are separate qualities."
                : "The headline includes a persistent team-specific driver effect. It is not a definitive ranking of pure, transferable skill."}
            </p>
            <a href="#health">
              Understand the model <ArrowRight size={14} />
            </a>
          </div>
        </div>
      </div>
      <Trend data={data} car={car} metric={metric} initial={selectedRow.id} />
      <div className="notice roadmap">
        <Layers3 size={20} />
        <div>
          <strong>Beyond qualifying</strong>
          <p>
            Race pace, tyre management, racecraft and an equal-car championship
            are in development. Racing estimates remain hidden until their
            results are regenerated and checked.
          </p>
        </div>
        <a href="#health">
          See readiness <ArrowRight size={16} />
        </a>
      </div>
    </>
  );
}

function Detail({
  row,
  value,
  car,
  rank,
  metric,
}: {
  row: Driver | Car;
  value: Estimate;
  car: boolean;
  rank: number;
  metric: Metric;
}) {
  return (
    <section className="panel detail-card" aria-label="Selected entry">
      <div className="detail-kicker">
        {car ? "CONSTRUCTOR" : "DRIVER"} SPOTLIGHT<span>#{rank}</span>
      </div>
      <div
        className="monogram"
        style={{ borderColor: color(car ? row.id : (row as Driver).lineage) }}
      >
        {car ? row.name.slice(0, 3).toUpperCase() : (row as Driver).code}
      </div>
      <h2>{row.name}</h2>
      <p>
        {car
          ? "Track-neutral qualifying pace"
          : `${(row as Driver).team} · ${metricLabel(metric)}`}
      </p>
      <div className="detail-pace">
        {signed(value.median)}
        <span>s</span>
      </div>
      <div className="detail-caption">
        per 90-second lap · above / below average
      </div>
      <div className="detail-rule" />
      <dl className="detail-stats">
        <div>
          <dt>90% pace interval</dt>
          <dd>
            {signed(value.q05)} to {signed(value.q95)}s
          </dd>
        </div>
        <div>
          <dt>90% rank range</dt>
          <dd>
            {value.rank_lo}–{value.rank_hi}
          </dd>
        </div>
        <div>
          <dt>Probability fastest</dt>
          <dd>{pct(value.p_fastest)}</dd>
        </div>
        <div>
          <dt>Probability top 3</dt>
          <dd>{pct(value.p_top3)}</dd>
        </div>
      </dl>
      {!car && (
        <>
          <div className="detail-rule" />
          <div className="evidence-grid">
            <div>
              <strong>{(row as Driver).evidence.events}</strong>
              <span>events</span>
            </div>
            <div>
              <strong>{(row as Driver).evidence.teammates}</strong>
              <span>teammates</span>
            </div>
            <div>
              <strong>{(row as Driver).evidence.teams}</strong>
              <span>team lineages</span>
            </div>
          </div>
          <p className="small-note">
            Evidence counts provide context; they are not confidence scores.
          </p>
        </>
      )}
      {car && (
        <p className="small-note">
          Estimated gap to the fastest car:{" "}
          {Math.abs((row as Car).gap_to_best.median).toFixed(3)}s. Calculated
          across joint model samples.
        </p>
      )}
    </section>
  );
}
