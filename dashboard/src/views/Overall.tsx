import { useState } from "react";
import { Badge, Empty } from "../components";
import { date, pct, signed } from "../lib";
import type { OverallResult } from "../types";
import RankingHeader from "./RankingHeader";

const labels: Record<string, string> = {
  qualifying_pace: "Qualifying pace",
  first_lap: "First-lap performance",
  race_specific_pace: "Race-specific pace",
  degradation: "Tyre management",
  consistency: "Consistency",
  overtaking_attack: "Overtaking",
  overtaking_defend: "Defending",
  all: "All components together",
};
const qualityName = (name: string) => labels[name] || name.replaceAll("_", " ");

export function OverallMethodology() {
  return (
    <section className="panel overall-notes" aria-labelledby="overall-method-title">
      <h2 id="overall-method-title">About the equal-car championship</h2>
      <p>
        This experimental scenario simulates seasons with equal average machinery
        and mechanical retirement risk. Qualifying pace sets the grid, a race
        model predicts finishing order, and retirements affect points. The
        headline keeps each driver’s estimated fit with their current team; it
        does not isolate a universal or purely transferable ability.
      </p>
      <p>
        Racing qualities enter only after tests alongside the other qualities
        and a separate test of the complete selection procedure on later,
        held-out seasons. Qualifying inputs for each test season use only earlier
        seasons. Tests exclude seasons whose qualifying fits failed to converge;
        those races may still train later folds.
      </p>
      <p>
        This is a simplified grid-to-finish simulation. It does not model passing,
        traffic, pit strategy or wet conditions lap by lap. Pit operations and wet
        pace are not separate driver contributions here. Driver-specific error
        rates affect retirements only if their separate validation passes.
        A quality’s own standalone ranking gate is separate from championship entry.
      </p>
      <p>
        Standings are ordered by expected points per race. Title probability is
        the share of simulated seasons won; rank ranges describe simulated season
        outcomes, including race variability and uncertainty. They are not a
        forecast of the real championship. No table is published until current
        provenance and the combined validation gate both pass.
      </p>
    </section>
  );
}

export default function Overall({ result }: { result?: OverallResult }) {
  const [selected, setSelected] = useState("");
  const evidence = result?.status === "established" || result?.status === "not established"
    ? result.evidence : undefined;
  const established = result?.status === "established" &&
    evidence?.combined_validation.gate === true && result.standings.length > 0;
  const rows = established ? result.standings : [];
  const driver = rows.find((r) => r.driver_id === selected) || rows[0];
  const contribution = established
    ? result.contributions.find((r) => r.driver_id === driver?.driver_id)
    : undefined;
  const state = result?.status === "established" && !established
    ? "not established" : result?.status || "unavailable";
  return (
    <>
      <RankingHeader kind="drivers" discipline="overall" />
      <div className="event-line ranking-event">
        <Badge tone="amber">Experimental · equal car</Badge>
        {evidence ? (
          <span>
            Recorded {date(evidence.recorded_at)}
            {evidence.data_as_of && ` · Data through ${evidence.data_as_of}`}
          </span>
        ) : <span className="muted">No verified championship release</span>}
      </div>
      <div className="ranking-layout overall-layout">
        <section className="panel ranking-panel">
          <div className="panel-heading">
            <div>
              <h2>Overall (equal car)</h2>
              <p>Expected points · driver–team effects retained</p>
            </div>
          </div>
          {established ? (
            <>
              <div className="table-scroll" tabIndex={0} role="region"
                aria-label="Overall rankings; scroll for all entries">
                <table className="rank-table overall-table" aria-label="Equal-car championship standings">
                  <colgroup><col className="overall-driver" /><col /><col /><col /></colgroup>
                  <thead><tr>
                    <th scope="col">Driver</th>
                    <th scope="col" className="numeric">Points per race</th>
                    <th scope="col" className="numeric">Title probability</th>
                    <th scope="col" className="numeric">Rank range</th>
                  </tr></thead>
                  <tbody>{rows.map((row) => (
                    <tr key={row.driver_id} className={row.driver_id === driver?.driver_id ? "selected-row" : ""}>
                      <td><button className="entity-button" onClick={() => setSelected(row.driver_id)}
                        aria-label={`Inspect ${row.name} contributions`} aria-pressed={row.driver_id === driver?.driver_id}>
                        <strong>{row.name}</strong>
                      </button></td>
                      <td className="numeric">{row.points_per_race.toFixed(2)}</td>
                      <td className="numeric">{pct(row.p_title)}</td>
                      <td className="numeric"><span className="rank-range">{row.rank_lo}–{row.rank_hi}</span></td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
              <p className="overall-caption">
                {evidence.simulated_seasons.toLocaleString()} simulated seasons · {evidence.n_races_simulated} races per season.
                Select a driver to inspect their contributions. Overlapping rank ranges leave positions uncertain.
              </p>
            </>
          ) : (
            <Empty title={`Overall ranking ${state}`}>
              {result?.status === "established"
                ? "This release does not contain a validated overall ranking."
                : result?.reason || "No verified championship results are available in this release yet."}
            </Empty>
          )}
        </section>
        {contribution && driver && (
          <section className="panel overall-contributions" aria-labelledby="contribution-title">
            <h2 id="contribution-title">Contribution breakdown</h2>
            <h3>{driver.name}</h3>
            <p>Expected points per race lost when a component is set to the field average.</p>
            <dl>{Object.entries(contribution.losses).map(([quality, loss]) => (
              <div key={quality}><dt>{qualityName(quality)}</dt><dd>{signed(loss, 2)}</dd></div>
            ))}</dl>
            <p>
              A negative loss means points improve at the field average. Each
              component is changed alone; “all” changes them together. Correlated
              qualities have no unique allocation of credit, so these values
              should not be added. Driver-error retirement effects are held fixed
              in this breakdown.
            </p>
          </section>
        )}
      </div>
      {evidence && (
        <section className="panel overall-notes" aria-labelledby="quality-entry-title">
          <h2 id="quality-entry-title">Which racing qualities entered?</h2>
          <p>
            Entered: {evidence.qualities_entered.length
              ? evidence.qualities_entered.map(qualityName).join(", ")
              : "No additional racing qualities"}.
          </p>
          <dl className="overall-decisions">{evidence.quality_decisions.map((decision) => (
            <div key={decision.quality}>
              <dt>{qualityName(decision.quality)} · {decision.entered ? "Entered" : decision.tested ? "Not entered" : "Not tested"}</dt>
              <dd>{decision.reason}</dd>
            </div>
          ))}</dl>
          <p>{evidence.driver_error_rates_used
            ? "Retirements use driver-specific own-error rates because their separate gate passed."
            : "Retirements use the field-average own-error rate; driver-specific error effects are excluded."}</p>
          <h3>Excluded test seasons</h3>
          <p>Qualifying fits did not converge, so these seasons were left out of the recorded held-out tests:</p>
          <dl className="overall-decisions">
            <div><dt>Combined quality-selection validation</dt>
              <dd>{evidence.excluded_test_seasons.combined_validation.join(", ") || "None recorded"}</dd></div>
            <div><dt>Race-stage validation</dt>
              <dd>{evidence.excluded_test_seasons.heldout_race_stage.join(", ") || "None recorded"}</dd></div>
          </dl>
          <p>Exclusion applies to testing. These races can still be used to train later seasons’ fits.</p>
        </section>
      )}
      <OverallMethodology />
    </>
  );
}
