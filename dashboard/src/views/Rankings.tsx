import { useState } from "react";
import { ArrowDownToLine, ArrowRight, Info, Search, X } from "lucide-react";
import type {
  Dataset,
  Driver,
  Car,
  Discipline,
  Metric,
  Estimate,
} from "../types";
import { Badge, Empty, Band, MetricControl } from "../components";
import { color, signed, pct, date, exportCsv, exportRaceCsv } from "../lib";
import Trend from "./Trend";
import RankingHeader from "./RankingHeader";
import { RaceMethodology, RaceValidation } from "./RaceNotes";

interface RankingEntry {
  id: string;
  name: string;
  code: string;
  lineage: string;
  team: string;
  teamKey: string;
  carried?: boolean;
  pace: Estimate;
  gap?: number;
  evidence: { label: string; value: number }[];
}

export default function Rankings({
  data,
  car,
  discipline,
  metric,
  setMetric,
}: {
  data: Dataset;
  car: boolean;
  discipline: Discipline;
  metric: Metric;
  setMetric: (m: Metric) => void;
}) {
  const race = discipline === "race";
  const kind = car ? "cars" : "drivers";
  const raceData = data.race_pace;
  const qualifying = [...(car ? data.cars : data.drivers)].sort(
    (a, b) => qualifyingValue(b).median - qualifyingValue(a).median,
  );
  function qualifyingValue(r: Driver | Car) {
    return car ? (r as Car).pace : (r as Driver)[metric];
  }
  const raceRows = [...(raceData?.[kind] || [])].sort(
    (a, b) => b.pace.median - a.pace.median,
  );
  const rows: RankingEntry[] = race
    ? raceRows.map((r) => {
        const driver = data.drivers.find((d) => d.id === r.id);
        const constructor = data.cars.find(
          (c) => c.id === (car ? r.id : r.lineage),
        );
        return {
          id: r.id,
          name: car ? constructor?.name || r.name : r.name,
          code: r.code || driver?.code || r.name.slice(0, 3).toUpperCase(),
          lineage: r.lineage || driver?.lineage || r.id,
          team:
            driver && driver.lineage === r.lineage
              ? driver.team
              : constructor?.name || r.team || "",
          teamKey: r.lineage || driver?.lineage || r.team || "",
          pace: r.pace,
          evidence: [
            { label: "dry races this season", value: r.races },
            { label: "clean laps this season", value: r.laps },
          ],
        };
      })
    : qualifying.map((r) => {
        const driver = r as Driver;
        return {
          id: r.id,
          name: r.name,
          code: car ? r.name.slice(0, 3).toUpperCase() : driver.code,
          lineage: car ? r.id : driver.lineage,
          team: car ? "" : driver.team,
          teamKey: car ? "" : driver.lineage,
          carried: !car && !driver.has_time,
          pace: qualifyingValue(r),
          gap: car ? Math.abs((r as Car).gap_to_best.median) : undefined,
          evidence: car
            ? []
            : [
                { label: "events", value: driver.evidence.events },
                { label: "teammates", value: driver.evidence.teammates },
                { label: "team lineages", value: driver.evidence.teams },
              ],
        };
      });
  // This component stays mounted when the metric changes, keeping the user's
  // selected entry and filters while the estimates update underneath them.
  const [selected, setSelected] = useState(rows[0]?.id || "");
  const [search, setSearch] = useState("");
  const [team, setTeam] = useState("all");
  const selectedRow = rows.find((r) => r.id === selected) || rows[0];
  const teams = new Map(data.drivers.map((d) => [d.lineage, d.team]));
  rows.forEach((r) => {
    if (r.teamKey && r.team) teams.set(r.teamKey, r.team);
  });
  const filtered = rows.filter(
    (r) =>
      `${r.name} ${r.team}`.toLowerCase().includes(search.toLowerCase()) &&
      (team === "all" || r.teamKey === team),
  );
  const domain = [
    Math.min(0, ...rows.map((r) => r.pace.q05)),
    Math.max(0, ...rows.map((r) => r.pace.q95)),
  ];
  const e = race ? raceData?.data_as_of : data.meta.data_as_of;
  const validation = raceData?.validation.metrics[kind];
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
  const emptyTitle = !raceData
    ? "Race-pace validation is pending"
    : validation?.passed
      ? "Not enough current-season race data"
      : `${car ? "Car" : "Driver"} race-pace ranking withheld`;
  const emptyReason = !raceData
    ? `${car ? "Car" : "Driver"} rankings appear here when their prediction and uncertainty checks pass.`
    : validation?.passed
      ? "Each entry needs two usable dry races this season before it appears in the table."
      : `Publication checks not met: ${failedChecks}.`;
  return (
    <>
      <RankingHeader
        kind={kind}
        discipline={discipline}
        action={
          <button
            className="button"
            disabled={!rows.length}
            onClick={() => {
              if (race && raceData) exportRaceCsv(raceRows, kind, raceData);
              else if (!race) exportCsv(qualifying, car, metric, data);
            }}
          >
            <ArrowDownToLine size={16} />
            Export rankings
          </button>
        }
      />
      <div className="event-line ranking-event">
        <Badge tone={!race || validation?.passed ? "green" : "amber"}>
          {race ? "Dry races" : "Qualifying"}
        </Badge>
        {e ? (
          <span>
            Round {Number(e.event_id.slice(5))} · {e.race_name}{" "}
            <span className="muted">· {date(e.date)}</span>
          </span>
        ) : (
          <span className="muted">No checked release</span>
        )}
        <a href="#health">
          Model details <ArrowRight size={14} />
        </a>
      </div>
      <div className="ranking-layout">
        <section className="panel ranking-panel">
          <div className="panel-heading">
            <div>
              <h2>{race ? "Race pace" : "Qualifying pace"}</h2>
              <p>
                {race
                  ? "Dry races · tyre age 10 laps"
                  : car
                    ? "Driver contribution accounted for"
                    : metric === "headline"
                      ? "Includes driver–team effect"
                      : "Experimental transferable skill"}
              </p>
            </div>
            {!race && !car ? (
              <MetricControl metric={metric} setMetric={setMetric} />
            ) : (
              <span className="ranking-context">
                {race ? "Season estimate" : "Average circuit"}
              </span>
            )}
          </div>
          {!race && !car && metric === "portable" && (
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
                disabled={!rows.length}
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
                disabled={!rows.length}
                onChange={(e) => setTeam(e.target.value)}
              >
                <option value="all">All teams</option>
                {[...teams]
                  .sort((a, b) => a[1].localeCompare(b[1]))
                  .map(([id, name]) => (
                    <option key={id} value={id}>
                      {name}
                    </option>
                  ))}
              </select>
            )}
            <span className="table-count">
              {filtered.length} {kind}
            </span>
          </div>
          {rows.length > 0 ? (
            <>
              <div
                className="table-scroll"
                tabIndex={0}
                role="region"
                aria-label="Rankings; scroll for all entries"
              >
                <table className="rank-table">
                  <colgroup>
                    <col className="position-column" />
                    <col className="name-column" />
                    <col className="pace-column" />
                    <col className="range-col" />
                    <col className="rank-column" />
                  </colgroup>
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
                      const v = r.pace;
                      const place = rows.indexOf(r) + 1;
                      return (
                        <tr
                          key={r.id}
                          className={
                            r.id === selectedRow?.id ? "selected-row" : ""
                          }
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
                              aria-pressed={r.id === selectedRow?.id}
                            >
                              <span
                                className="team-stripe"
                                style={{ background: color(r.lineage) }}
                              />
                              <span>
                                <strong>{r.name}</strong>
                                <small>
                                  {car ? "Car package" : r.team}
                                  {r.carried && " · carried forward"}
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
                                r.id === selectedRow?.id
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
            </>
          ) : (
            <Empty title={emptyTitle}>{emptyReason}</Empty>
          )}
          {rows.length > 0 && (
            <div className="table-foot">
              <span>
                <span className="legend-dot" />
                Median <span className="legend-line" />
                90% interval
              </span>
              <span>Positive = faster · ranked by median</span>
            </div>
          )}
        </section>
        <div className="detail-column">
          {selectedRow && (
            <Detail
              row={selectedRow}
              car={car}
              rank={rows.indexOf(selectedRow) + 1}
            />
          )}
          <MetricDefinitions car={car} race={race} />
        </div>
      </div>
      {race ? (
        <>
          {!!raceData?.[car ? "unrated_cars" : "unrated_drivers"].length && (
            <p className="muted">
              {raceData[car ? "unrated_cars" : "unrated_drivers"].length} {kind}{" "}
              have fewer than two usable dry races this season.
            </p>
          )}
          {validation && (
            <RaceValidation validation={validation} driver={!car} />
          )}
          <RaceMethodology data={raceData} />
        </>
      ) : (
        selectedRow && (
          <Trend
            data={data}
            car={car}
            metric={metric}
            initial={selectedRow.id}
          />
        )
      )}
    </>
  );
}

function Detail({
  row,
  car,
  rank,
}: {
  row: RankingEntry;
  car: boolean;
  rank: number;
}) {
  const value = row.pace;
  return (
    <section className="panel detail-card" aria-label="Selected entry">
      <div className="detail-kicker">
        {car ? "CONSTRUCTOR" : "DRIVER"}
        <span>#{rank}</span>
      </div>
      <div className="monogram" style={{ borderColor: color(row.lineage) }}>
        {row.code}
      </div>
      <h2>{row.name}</h2>
      <p>{car ? "Car package" : row.team || "Driver estimate"}</p>
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
          <dd>{value.p_fastest === undefined ? "—" : pct(value.p_fastest)}</dd>
        </div>
        <div>
          <dt>Probability top 3</dt>
          <dd>{value.p_top3 === undefined ? "—" : pct(value.p_top3)}</dd>
        </div>
        {row.gap !== undefined && (
          <div>
            <dt>Gap to fastest car</dt>
            <dd>{row.gap.toFixed(3)}s</dd>
          </div>
        )}
      </dl>
      {row.evidence.length > 0 && (
        <>
          <div className="detail-rule" />
          <div className="evidence-grid">
            {row.evidence.map((e) => (
              <div key={e.label}>
                <strong>{e.value.toLocaleString()}</strong>
                <span>{e.label}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function MetricDefinitions({ car, race }: { car: boolean; race: boolean }) {
  return (
    <details className="metric-notes">
      <summary>Metric definitions</summary>
      <dl>
        <div>
          <dt>Pace</dt>
          <dd>
            Seconds gained per 90-second lap against the average{" "}
            {car ? "car" : "driver"} in the {race ? "rated " : ""}field.
          </dd>
        </div>
        <div>
          <dt>{race ? "Season pace" : car ? "Car rating" : "In-team pace"}</dt>
          <dd>
            {race
              ? car
                ? "Current-season car pace across the sampled dry circuits, with driver contribution estimated separately."
                : "Lasting driver pace plus current-season form, with car performance estimated separately."
              : car
                ? "Qualifying performance at an average circuit, excluding reliability and pit stops."
                : "Driver qualifying pace including a persistent driver–team effect. Portable skill excludes that effect and remains experimental."}
          </dd>
        </div>
        <div>
          <dt>90% intervals</dt>
          <dd>
            Uncertainty in estimated pace and rank. Probabilities refer to{" "}
            {race ? "dry-race pace" : "qualifying ability"}, not race results. A
            dash means the estimate is unavailable.
          </dd>
        </div>
      </dl>
    </details>
  );
}
