import { useState, type CSSProperties } from "react";
import { Badge, Empty } from "../components";
import { date, pct, signed } from "../lib";
import type { CurrentOverallResult } from "../types";
import { currentOverallAvailable, currentOverallEvidence, currentOverallReason,
  currentOverallState, overallCheckLabels, overallRequiredChecks } from "../overall";
import { performanceScale, PerformanceKey } from "../performance";
import RankingHeader from "./RankingHeader";

export function CurrentOverallMethodology() {
  return <section className="panel overall-notes" aria-labelledby="overall-method-title">
    <h2 id="overall-method-title">About the equal-car comparison</h2>
    <p>This experimental comparison estimates current qualifying and race performance together.
      It places the latest field in equal machinery and team conditions, over a common fixed
      mix of circuits and race conditions. Car and team advantages are excluded from the driver score;
      uncertainty about separating those effects remains.</p>
    <p>Drivers are ordered by expected points per race. The expected-points range describes
      uncertainty about their underlying performance. The season finish range also includes
      race-to-race variation. Title chances describe this reference scenario, not the real championship.</p>
    <p>Qualifying generates a grid, current race performance predicts the race, and completion
      distance determines classification and points. Championship ties use race countback,
      then qualifying countback; unresolved ties share title probability.</p>
    <p>A ranking requires current sources, stable fits, known-truth recovery, predictive checks,
      calibrated uncertainty, new prospective evidence and adequate simulation precision.
      The old model’s combined gate cannot establish this replacement.</p>
  </section>;
}

export function CurrentOverallSummary({ result }: { result: CurrentOverallResult }) {
  const available = currentOverallAvailable(result);
  const evidence = currentOverallEvidence(result);
  const rows = available ? [...result.standings].sort((a, b) => b.expected_points.mean - a.expected_points.mean) : [];
  const scale = performanceScale(rows.map(row => row.expected_points.mean));
  return <>
    <p className="briefing-units">Experimental · equal machinery and team support</p>
    {evidence && <p className="briefing-source">Recorded {date(evidence.recorded_at)} · Data through {evidence.data_as_of}</p>}
    {available ? <>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Equal-car summary; scroll for all entries">
        <table className="briefing-ranking briefing-overall-table" aria-label="Equal-car championship summary">
          <thead><tr><th scope="col">Driver</th><th scope="col" className="numeric">Points per race</th>
            <th scope="col" className="numeric">Expected points range</th><th scope="col" className="numeric">Season finish range</th></tr></thead>
          <tbody>{rows.map(row => <tr className="briefing-ranking-row" key={row.driver_id}
            style={{ "--performance-color": scale(row.expected_points.mean) } as CSSProperties}>
            <th scope="row">{row.name}</th><td className="numeric performance-value">{row.expected_points.mean.toFixed(2)}</td>
            <td className="numeric">{row.expected_points.q05.toFixed(2)}–{row.expected_points.q95.toFixed(2)}</td>
            <td className="numeric">{row.season_rank.q05}–{row.season_rank.q95}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <PerformanceKey points />
      <p className="briefing-units">Expected-points ranges describe uncertainty about performance.
        Season finish ranges additionally include racing variation. Both are 90% ranges.</p>
    </> : <div className="briefing-withheld"><h2>Overall ranking {currentOverallState(result)}</h2>
      <p>{currentOverallReason(result)}</p></div>}
    <div className="briefing-actions"><a href="#drivers/overall">Full standings, explanations & validation</a></div>
  </>;
}

export default function CurrentOverall({ result }: { result: CurrentOverallResult }) {
  const [selected, setSelected] = useState("");
  const available = currentOverallAvailable(result);
  const evidence = currentOverallEvidence(result);
  const rows = available ? [...result.standings].sort((a, b) => b.expected_points.mean - a.expected_points.mean) : [];
  const scale = performanceScale(rows.map(row => row.expected_points.mean));
  const driver = rows.find(row => row.driver_id === selected) || rows[0];
  const explanation = available && result.contributions.find(row => row.driver_id === driver?.driver_id);
  const labels: Record<string, string> = { qualifying_pace: "Qualifying performance", total_race_pace: "Current race performance", all: "Both together" };
  return <>
    <RankingHeader kind="drivers" discipline="overall" />
    <div className="event-line ranking-event"><Badge tone="amber">Experimental · equal car</Badge>
      {evidence ? <span>Recorded {date(evidence.recorded_at)} · Data through {evidence.data_as_of}</span>
        : <span className="muted">No verified Overall release</span>}</div>
    <div className="ranking-layout overall-layout">
      <section className="panel ranking-panel">
        <div className="panel-heading"><div><h2>Overall (equal car)</h2><p>Current performance · equal machinery and team support</p></div></div>
        {available ? <>
          <div className="table-scroll" tabIndex={0} role="region" aria-label="Overall rankings; scroll for all entries">
            <table className="rank-table overall-table" aria-label="Equal-car championship standings">
              <thead><tr><th scope="col">Driver</th><th scope="col" className="numeric">Points per race</th>
                <th scope="col" className="numeric">Expected points range</th><th scope="col" className="numeric">Title chance</th>
                <th scope="col" className="numeric">Season finish range</th></tr></thead>
              <tbody>{rows.map(row => <tr key={row.driver_id} className={row.driver_id === driver?.driver_id ? "selected-row" : ""}
                style={{ "--performance-color": scale(row.expected_points.mean) } as CSSProperties}>
                <td><button className="entity-button" onClick={() => setSelected(row.driver_id)}
                  aria-label={`Inspect ${row.name} explanation`} aria-pressed={row.driver_id === driver?.driver_id}><strong>{row.name}</strong></button></td>
                <td className="numeric performance-value">{row.expected_points.mean.toFixed(2)}</td>
                <td className="numeric">{row.expected_points.q05.toFixed(2)}–{row.expected_points.q95.toFixed(2)}</td>
                <td className="numeric">{pct(row.p_title)}</td><td className="numeric">{row.season_rank.q05}–{row.season_rank.q95}</td>
              </tr>)}</tbody>
            </table>
          </div>
          <PerformanceKey points />
          <p className="overall-caption">Both ranges contain the central 90% of their respective estimates.
            The expected-points range reflects uncertainty about performance; the season finish range
            also includes racing variation. Select a driver for their explanation.</p>
        </> : <Empty title={`Overall ranking ${currentOverallState(result)}`}>{currentOverallReason(result)}</Empty>}
      </section>
      {explanation && driver && <section className="panel overall-contributions" aria-labelledby="contribution-title">
        <h2 id="contribution-title">What explains the estimate?</h2><h3>{driver.name}</h3>
        <p>Expected points per race lost when each performance component is set to the field average.</p>
        <dl>{Object.entries(explanation.losses).map(([name, loss]) => <div key={name}><dt>{labels[name] || name.replaceAll("_", " ")}</dt><dd>{signed(loss, 2)}</dd></div>)}</dl>
        <p>Each component is changed separately using the same model draws and conditions.
          Correlated components have no unique allocation of credit; these values should not be added.</p>
        <h3>Simulation precision</h3>
        <p>Numerical standard error of expected points: {driver.expected_points.mcse.toFixed(3)} points per race.
          This measures simulation error, separately from uncertainty about the driver and variation between seasons.</p>
      </section>}
    </div>
    {evidence && <section className="panel overall-notes" aria-labelledby="overall-validation-title">
      <h2 id="overall-validation-title">Evidence required for this ranking</h2>
      <dl className="overall-decisions">{overallRequiredChecks.map(key => <div key={key}>
        <dt>{overallCheckLabels[key]} · {evidence.checks[key]?.passed === true ? "Passed" : "Not established"}</dt>
        <dd>{typeof evidence.checks[key]?.reason === "string" ? evidence.checks[key].reason : "Complete evidence has not been supplied."}</dd>
      </div>)}</dl>
      <p>Reference scenario: {evidence.scenario.label} · {evidence.scenario.races} races.</p>
      {Array.isArray(result.limitations) && result.limitations.length > 0 && <>
        <h3>Limits of the result</h3><ul>{result.limitations.filter(x => typeof x === "string").map((item, i) => <li key={i}>{item}</li>)}</ul>
      </>}
    </section>}
    <CurrentOverallMethodology />
  </>;
}
