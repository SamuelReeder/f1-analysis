import { useState } from "react";
import { ArrowDownToLine, ArrowRight, Info, Search, X } from "lucide-react";
import type { Dataset, Driver, Car, Metric, Estimate } from "../types";
import { Badge, Empty, Band, PageHeading, MetricControl } from "../components";
import { color, signed, pct, date, metricLabel, exportCsv } from "../lib";
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
        title={car ? "Car rankings" : "Driver rankings"}
        action={
          <button
            className="button"
            onClick={() => exportCsv(rows, car, metric, data)}
          >
            <ArrowDownToLine size={16} />
            Export rankings
          </button>
        }
      />
      <div className="event-line">
        <Badge tone="green">Qualifying</Badge>
        <span>
          Round {Number(e.event_id.slice(5))} · {e.race_name}{" "}
          <span className="muted">· {date(e.date)}</span>
        </span>
        <a href="#health">
          Model details <ArrowRight size={14} />
        </a>
      </div>
      <div className="ranking-layout">
        <section className="panel ranking-panel">
          <div className="panel-heading">
            <div>
              <h2>Qualifying pace</h2>
              <p>
                {car
                  ? "Average circuit"
                  : metric === "headline"
                    ? "Includes driver–team effect"
                    : "Experimental transferable skill"}
              </p>
            </div>
            {!car && <MetricControl metric={metric} setMetric={setMetric} />}
          </div>
          {!car && metric === "portable" && (
            <div className="inline-warning">
              <Info size={16} />
              <span>
                Experimental: sensitive to model assumptions; weaker ranking
                recovery in simulations.
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
                  <th className="numeric">90% RANK RANGE</th>
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
                            r.id === selectedRow.id
                              ? "var(--accent)"
                              : "var(--muted)",
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
          <details className="metric-notes">
            <summary>Metric definitions</summary>
            <dl>
              <div>
                <dt>Pace</dt>
                <dd>
                  Seconds gained per 90-second lap against the average{" "}
                  {car ? "car" : "driver"} in the field.
                </dd>
              </div>
              <div>
                <dt>{car ? "Car rating" : "In-team pace"}</dt>
                <dd>
                  {car
                    ? "Qualifying performance at an average circuit, excluding reliability and pit stops."
                    : "Driver qualifying pace including a persistent driver–team effect. Portable skill excludes that effect and remains experimental."}
                </dd>
              </div>
              <div>
                <dt>90% intervals</dt>
                <dd>
                  Uncertainty in estimated pace and rank. Probabilities refer to
                  qualifying ability, not race results.
                </dd>
              </div>
            </dl>
          </details>
        </div>
      </div>
      <Trend data={data} car={car} metric={metric} initial={selectedRow.id} />
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
        {car ? "CONSTRUCTOR" : "DRIVER"}
        <span>#{rank}</span>
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
        seconds / 90s lap · positive = faster
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
        {car && (
          <div>
            <dt>Gap to fastest car</dt>
            <dd>{Math.abs((row as Car).gap_to_best.median).toFixed(3)}s</dd>
          </div>
        )}
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
        </>
      )}
    </section>
  );
}
