import { useState } from "react";
import { ArrowLeftRight, Info } from "lucide-react";
import type { Dataset, Metric, Car, Driver } from "../types";
import { metricLabel, shortName, pct, signed } from "../lib";
import { PageHeading, MetricControl, Badge, Band } from "../components";

export default function Compare({
  data,
  metric,
  setMetric,
}: {
  data: Dataset;
  metric: Metric;
  setMetric: (m: Metric) => void;
}) {
  const [car, setCar] = useState(false);
  const entities = car ? data.cars : data.drivers;
  const [a, setA] = useState(data.drivers[0].id);
  const [b, setB] = useState(data.drivers[1].id);
  const first = entities.find((r) => r.id === a) || entities[0];
  const second =
    entities.find((r) => r.id === b && r.id !== first.id) ||
    entities.find((r) => r.id !== first.id)!;
  const pair =
    data.comparisons[car ? "cars" : metric][`${first.id}|${second.id}`];
  const value = (r: Driver | Car) =>
    car ? (r as Car).pace : (r as Driver)[metric];
  const domain = [
    Math.min(value(first).q05, value(second).q05, 0),
    Math.max(value(first).q95, value(second).q95, 0),
  ];
  return (
    <>
      <PageHeading eyebrow="A CLOSER COMPARISON" title="Head to head.">
        Compare underlying qualifying pace, with uncertainty carried through.
      </PageHeading>
      <div className="comparison-toolbar">
        <div className="segmented">
          <button
            aria-pressed={!car}
            onClick={() => {
              setCar(false);
              setA(data.drivers[0].id);
              setB(data.drivers[1].id);
            }}
          >
            Drivers
          </button>
          <button
            aria-pressed={car}
            onClick={() => {
              setCar(true);
              setA(data.cars[0].id);
              setB(data.cars[1].id);
            }}
          >
            Cars
          </button>
        </div>
        {!car && <MetricControl metric={metric} setMetric={setMetric} />}
        <Badge tone={metric === "portable" && !car ? "amber" : "green"}>
          {car ? "Track-neutral pace" : metricLabel(metric)}
        </Badge>
      </div>
      <section className="panel comparison-panel">
        <div className="comparison-entries">
          <div>
            <label htmlFor="compare-a">FIRST {car ? "CAR" : "DRIVER"}</label>
            <select
              id="compare-a"
              value={first.id}
              onChange={(e) => setA(e.target.value)}
            >
              {entities
                .filter((r) => r.id !== second.id)
                .map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.name}
                  </option>
                ))}
            </select>
            <p>{car ? "Car package pace" : (first as Driver).team}</p>
          </div>
          <button
            className="swap"
            aria-label="Swap comparison"
            onClick={() => {
              setA(second.id);
              setB(first.id);
            }}
          >
            <ArrowLeftRight size={20} />
          </button>
          <div>
            <label htmlFor="compare-b">SECOND {car ? "CAR" : "DRIVER"}</label>
            <select
              id="compare-b"
              value={second.id}
              onChange={(e) => setB(e.target.value)}
            >
              {entities
                .filter((r) => r.id !== first.id)
                .map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.name}
                  </option>
                ))}
            </select>
            <p>{car ? "Car package pace" : (second as Driver).team}</p>
          </div>
        </div>
        <div className="matchup">
          <div>
            <span className="eyebrow">PROBABILITY OF HIGHER PACE</span>
            <div className="matchup-prob">{pct(pair.p_ahead)}</div>
            <p>
              {shortName(first.name)} ahead of {shortName(second.name)}
            </p>
          </div>
          <div className="matchup-gap">
            <span className="eyebrow">ESTIMATED PACE DIFFERENCE</span>
            <div>
              {signed(pair.median)}
              <small>s</small>
            </div>
            <p>
              90% interval: {signed(pair.q05)} to {signed(pair.q95)}s
            </p>
          </div>
        </div>
        <div className="probability-track">
          <div style={{ width: pct(pair.p_ahead) }} />
        </div>
        <div className="probability-labels">
          <span>
            {shortName(first.name)} · {pct(pair.p_ahead)}
          </span>
          <span>
            {shortName(second.name)} · {pct(1 - pair.p_ahead)}
          </span>
        </div>
        <div className="comparison-bands">
          {[first, second].map((r) => (
            <div key={r.id}>
              <strong>{r.name}</strong>
              <Band value={value(r)} domain={domain} />
              <span>{signed(value(r).median)}s</span>
            </div>
          ))}
        </div>
        <div className="notice comparison-note">
          <Info size={18} />
          <p>
            {pair.q05 <= 0 && pair.q95 >= 0
              ? "The gap’s 90% interval includes zero: the ordering is uncertain."
              : "The gap’s 90% interval excludes zero: the model finds a clearer pace difference."}{" "}
            This is a comparison of lasting qualifying pace, not a prediction of
            the next session or race result.
          </p>
        </div>
        <div className="panel-foot">
          Gap intervals use {data.comparison_draws} shared posterior samples.{" "}
          {car
            ? "Car probabilities use the same samples."
            : `Driver probabilities use all ${data.meta.diagnostics.n_draws.toLocaleString()} posterior samples.`}
        </div>
      </section>
      <section className="panel matrix-panel">
        <div className="panel-heading">
          <div>
            <h2>The whole field, at a glance</h2>
            <p>
              Probability the row has higher pace than the column. Select a cell
              to compare.
            </p>
          </div>
          <div className="matrix-key">
            Less likely <span /> More likely
          </div>
        </div>
        <div className="matrix-scroll">
          <table className="matrix">
            <thead>
              <tr>
                <th aria-label="Row versus column" />
                {entities.map((e) => (
                  <th key={e.id} title={e.name}>
                    {car
                      ? e.name.slice(0, 3).toUpperCase()
                      : (e as Driver).code}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {entities.map((r) => (
                <tr key={r.id}>
                  <th title={r.name}>
                    {car
                      ? r.name.slice(0, 3).toUpperCase()
                      : (r as Driver).code}
                  </th>
                  {entities.map((c) => {
                    const p =
                      data.comparisons[car ? "cars" : metric][`${r.id}|${c.id}`]
                        .p_ahead;
                    return (
                      <td key={c.id}>
                        {r.id === c.id ? (
                          <span className="matrix-diagonal">—</span>
                        ) : (
                          <button
                            style={{
                              background:
                                p >= 0.9
                                  ? "#2d6146"
                                  : p >= 0.75
                                    ? "#c4dcc3"
                                    : p >= 0.5
                                      ? "#e6efe1"
                                      : p >= 0.25
                                        ? "#f7f1e5"
                                        : "#efdbc0",
                              color: p >= 0.9 ? "#fff" : "#263f2d",
                            }}
                            title={`${r.name} ahead of ${c.name}: ${pct(p)}`}
                            aria-label={`${r.name} ahead of ${c.name}: ${pct(p)}`}
                            onClick={() => {
                              setA(r.id);
                              setB(c.id);
                              window.scrollTo({ top: 0, behavior: "smooth" });
                            }}
                          >
                            {Math.round(p * 100)}
                          </button>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel-foot">
          Percent probability · the table does not imply certainty about a
          complete finishing order.
        </div>
      </section>
    </>
  );
}
