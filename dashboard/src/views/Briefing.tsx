import type { CSSProperties, ReactNode } from "react";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { date, pct, signed } from "../lib";
import type { Dataset, Estimate, RaceValidation } from "../types";
import { performanceScale, PerformanceKey } from "../performance";
import "./Briefing.css";

export const BRIEFING_CHAPTERS = [
  { id: "qualifying", label: "Drivers" },
  { id: "cars", label: "Cars" },
  { id: "race", label: "Race pace" },
  { id: "overall", label: "Equal car" },
  { id: "evidence", label: "The evidence" },
] as const;

type PaceEntry = { id: string; name: string; team?: string; pace: Estimate; carried?: boolean };
const rankRange = (pace: Estimate) =>
  pace.rank_lo != null && pace.rank_hi != null ? `${pace.rank_lo}–${pace.rank_hi}` : "Not available";
const qualityLabels: Record<string, string> = {
  qualifying_pace: "Qualifying pace", first_lap: "First-lap performance",
  race_specific_pace: "Race-specific pace", degradation: "Tyre management",
  consistency: "Consistency", overtaking_attack: "Overtaking", overtaking_defend: "Defending",
};
const qualityLabel = (value: string) => qualityLabels[value] || value.replaceAll("_", " ");

function Chapter({ title, source, description, children }: {
  title: string; source: ReactNode; description: string; children: ReactNode;
}) {
  return <section className="briefing-stage">
    <header className="briefing-heading">
      <h1>{title}</h1>
      <p className="briefing-source">{source}</p>
      <p className="briefing-description">{description}</p>
    </header>
    {children}
  </section>;
}

function PaceTable({ rows, label, cars = false, compact = false }: {
  rows: PaceEntry[]; label: string; cars?: boolean; compact?: boolean;
}) {
  const ordered = [...rows].sort((a, b) => b.pace.median - a.pace.median);
  const scale = performanceScale(rows.map((row) => row.pace.median));
  const lo = Math.min(0, ...ordered.map((row) => row.pace.q05));
  const hi = Math.max(0, ...ordered.map((row) => row.pace.q95));
  const x = (value: number) => 6 + (value - lo) / (hi - lo || 1) * 228;
  if (!ordered.length) return <p className="briefing-empty">No estimates are available in this release.</p>;
  return <><table className={`briefing-ranking briefing-pace-table${compact ? " briefing-compact-table" : ""}`} aria-label={label}>
    <thead><tr>
      <th scope="col">{cars ? "Car" : "Driver"}</th>
      <th scope="col" className="numeric">Pace <span className="briefing-th-unit">s / 90s</span></th>
      <th scope="col" className="briefing-interval-column">90% pace interval</th>
      <th scope="col" className="numeric">90% rank range</th>
      {!compact && <th scope="col" className="numeric briefing-probability-column">Chance fastest</th>}
    </tr></thead>
    <tbody>{ordered.map((row) => <tr className="briefing-ranking-row" key={row.id} style={{ "--performance-color": scale(row.pace.median) } as CSSProperties}>
      <th scope="row"><strong>{row.name}</strong>{row.team && <small>{row.team}</small>}{row.carried && <small className="briefing-carried">Carried forward · no timed lap</small>}</th>
      <td className="numeric briefing-pace-value performance-value">{signed(row.pace.median)}</td>
      <td className="briefing-interval-column">
        <div className="briefing-interval-cell">
          <span>{signed(row.pace.q05)}</span>
          <svg viewBox="0 0 240 24" aria-hidden="true">
            <line x1={x(0)} x2={x(0)} y1="1" y2="23" stroke="#777" strokeDasharray="2 3" />
            <line className="briefing-interval-line" x1={x(row.pace.q05)} x2={x(row.pace.q95)} y1="12" y2="12" stroke="var(--performance-color)" strokeWidth="3" strokeLinecap="round" />
            <circle className="briefing-interval-point" cx={x(row.pace.median)} cy="12" r="4" fill="var(--performance-color)" stroke="#000" strokeWidth="1.5" />
          </svg>
          <span>{signed(row.pace.q95)}</span>
        </div>
      </td>
      <td className="numeric">{rankRange(row.pace)}</td>
      {!compact && <td className="numeric briefing-probability-column">{row.pace.p_fastest == null ? "—" : pct(row.pace.p_fastest)}</td>}
    </tr>)}</tbody>
  </table><PerformanceKey /></>;
}

function QualifyingChapter({ data, cars = false }: { data: Dataset; cars?: boolean }) {
  const rows: PaceEntry[] = cars
    ? data.cars.map((car) => ({ id: car.id, name: car.name, pace: car.pace }))
    : data.drivers.map((driver) => ({ id: driver.id, name: driver.name, team: driver.team, pace: driver.headline, carried: !driver.has_time }));
  const leader = [...rows].sort((a, b) => b.pace.median - a.pace.median)[0];
  const title = cars ? "Car qualifying pace" : "Driver qualifying pace";
  return <Chapter title={title}
    source={<>{data.meta.data_as_of.race_name} · {date(data.meta.data_as_of.date)}</>}
    description={cars ? "Average circuit · driver contribution accounted for" : "Current team · driver–team effects retained"}>
    <PaceTable rows={rows} label={title} cars={cars} />
    <p className="briefing-units">Seconds relative to the field average per 90-second lap; higher is faster. Ranges show uncertainty. Chance fastest is the probability of the highest estimated pace.</p>
    <p className="briefing-takeaway">{leader ? <><strong>{leader.name}</strong> has the highest central estimate. Rank ranges show the uncertainty in the order.</> : "No qualifying estimates are available."}</p>
    <div className="briefing-actions">
      <a href={cars ? "#cars" : "#drivers"}>Full rankings, history & downloads <ArrowRight size={14} aria-hidden="true" /></a>
      <a href={cars ? "#compare/cars" : "#compare"}>Compare two {cars ? "cars" : "drivers"} <ArrowRight size={14} aria-hidden="true" /></a>
    </div>
    <details className="briefing-details"><summary>What this rating measures</summary><p>{cars
      ? "The car estimate separates machinery from driver contribution at an average circuit. It describes qualifying performance, not race performance."
      : "Teammate comparisons help separate driver pace from machinery. The headline retains each driver’s estimated fit with their current team; it is not purely transferable skill. Estimates without a timed lap in the latest event are marked as carried forward."}</p></details>
  </Chapter>;
}

function failedRaceReason(validation?: RaceValidation) {
  if (!validation) return "No current validation result is available.";
  const reasons = [!validation.improves && "prediction improvement", !validation.calibrated && "interval coverage", !validation.sharper && "narrower intervals"].filter(Boolean);
  return reasons.length ? `Publication checks not met: ${reasons.join(", ")}.` : "The combined publication check has not passed; see the full validation details.";
}

function RaceChapter({ data }: { data: Dataset }) {
  const race = data.race_pace;
  return <Chapter title="Dry-race pace"
    source={race ? <>{race.data_as_of.race_name} · {date(race.data_as_of.date)}</> : "No checked race-pace release"}
    description="Clean, dry laps · fuel effects accounted for · reference tyre age of 10 laps">
    <div className="briefing-race-panels">{(["drivers", "cars"] as const).map((kind) => {
      const validation = race?.validation.metrics[kind];
      const passed = validation?.passed === true;
      const entries = passed ? race?.[kind] || [] : [];
      const title = kind === "drivers" ? "Driver race pace" : "Car race pace";
      const rows = entries.map((row) => ({ ...row, name: kind === "cars" ? data.cars.find((car) => car.id === row.id)?.name || row.name : row.name }));
      const leader = [...rows].sort((a, b) => b.pace.median - a.pace.median)[0];
      return <section className="briefing-race-section" key={kind} aria-label={title}>
        <div className="briefing-section-heading"><h2>{title}</h2><span>{passed ? "Validation passed" : "Withheld"}</span></div>
        {entries.length ? <PaceTable rows={rows} label={title} cars={kind === "cars"} compact />
          : <p className="briefing-empty">{passed ? "The checks passed, but no current entries meet the data requirements." : failedRaceReason(validation)}</p>}
        {leader && <p className="briefing-takeaway"><strong>{leader.name}</strong> has the highest central estimate. Positions remain uncertain.</p>}
        <div className="briefing-actions"><a href={`#${kind}/race`}>Full {kind === "drivers" ? "driver" : "car"} results & checks <ArrowRight size={14} aria-hidden="true" /></a></div>
      </section>;
    })}</div>
    <p className="briefing-units">Seconds relative to the field average per 90-second lap; higher is faster. Driver and car rankings pass separate prediction and uncertainty checks.</p>
  </Chapter>;
}

function OverallChapter({ data }: { data: Dataset }) {
  const result = data.overall;
  const evidence = result?.status === "established" || result?.status === "not established" ? result.evidence : null;
  const established = result?.status === "established" && evidence?.combined_validation.gate === true && result.standings.length > 0;
  const rows = established ? [...result.standings].sort((a, b) => b.points_per_race - a.points_per_race) : [];
  const scale = performanceScale(rows.map((row) => row.points_per_race));
  const state = result?.status === "established" && !established ? "not established" : result?.status || "unavailable";
  const reason = result?.status === "established" && !established ? "This release does not contain a validated overall ranking." : result?.reason || "No verified championship results are available in this release yet.";
  return <Chapter title="Overall (equal car)"
    source={evidence ? <>Recorded {date(evidence.recorded_at)}{evidence.data_as_of && <> · Data through {evidence.data_as_of}</>}</> : "No verified championship release"}
    description="Experimental · equal average machinery · driver–team effects retained">
    {established ? <>
      <table className="briefing-ranking briefing-overall-table" aria-label="Equal-car championship summary">
        <thead><tr><th scope="col">Driver</th><th scope="col" className="numeric">Points per race</th><th scope="col" className="numeric">Title probability</th><th scope="col" className="numeric">Rank range</th></tr></thead>
        <tbody>{rows.map((row) => <tr className="briefing-ranking-row" key={row.driver_id} style={{ "--performance-color": scale(row.points_per_race) } as CSSProperties}>
          <th scope="row">{row.name}</th><td className="numeric performance-value">{row.points_per_race.toFixed(2)}</td>
          <td className="numeric">{pct(row.p_title)}</td><td className="numeric">{row.rank_lo}–{row.rank_hi}</td>
        </tr>)}</tbody>
      </table>
      <PerformanceKey points />
      <p className="briefing-units">{evidence.simulated_seasons.toLocaleString()} simulated seasons · {evidence.n_races_simulated} races per season. Title probability is the share won; rank ranges describe simulated season outcomes.</p>
      <p className="briefing-takeaway"><strong>{rows[0].name}</strong> has the most expected points per race in this equal-car simulation.</p>
    </> : <div className="briefing-withheld"><h2>Overall ranking {state}</h2><p>{reason}</p></div>}
    {evidence && <section className="briefing-quality-summary" aria-label="Quality selection">
      <h2>Additional racing qualities</h2>
      <p>{evidence.qualities_entered.length ? evidence.qualities_entered.map(qualityLabel).join(" · ") : "No additional racing qualities entered."}</p>
      <p>Tested but not entered: {evidence.qualities_not_entered.length ? evidence.qualities_not_entered.map(qualityLabel).join(" · ") : "None recorded"}.</p>
      <p>Excluded test seasons — combined validation: {evidence.excluded_test_seasons.combined_validation.join(", ") || "None recorded"}; race stage: {evidence.excluded_test_seasons.heldout_race_stage.join(", ") || "None recorded"}.</p>
    </section>}
    <p className="briefing-units">This simplified scenario excludes lap-by-lap passing, traffic, pit strategy and wet conditions. It is not a forecast of the real championship.</p>
    <div className="briefing-actions"><a href="#drivers/overall">Full standings, contributions & validation <ArrowRight size={14} aria-hidden="true" /></a></div>
  </Chapter>;
}

function EvidenceChapter({ data }: { data: Dataset }) {
  const pooled = data.asof?.pooled;
  const comparison = pooled?.vs_naive;
  const verdict = !comparison ? "Baseline comparison unavailable" : comparison.ci95[1] < 0 ? "Better than last season’s gap" : comparison.ci95[0] > 0 ? "Worse than last season’s gap" : "No clear difference from last season’s gap";
  const latest = [...(data.asof?.events || [])].filter((event) => event.n_pairs > 0).sort((a, b) => b.event.date.localeCompare(a.event.date))[0];
  const maxError = pooled ? Math.max(pooled.rmse, pooled.rmse_naive, pooled.rmse_zero) || 1 : 1;
  return <Chapter title="Qualifying prediction checks"
    source={latest ? <>Scored through {latest.event.race_name} · {date(latest.event.date)}</> : "No scored-event date is available"}
    description="Teammate gaps predicted using only data from before each scored race">
    {pooled ? <>
      <dl className="briefing-score-grid">
        <div><dt>Teammate gap error</dt><dd>{pooled.rmse.toFixed(3)}<small> s</small></dd></div>
        <div><dt>90% interval coverage</dt><dd>{pct(pooled.coverage90)}</dd></div>
        <div><dt>Scored events</dt><dd>{pooled.n_events}</dd></div>
        <div><dt>Scored teammate gaps</dt><dd>{pooled.n_pairs.toLocaleString()}</dd></div>
      </dl>
      <section className="briefing-error-chart" aria-label="Prediction error comparison"><h2>Prediction error · lower is better</h2>{[
        { label: "Qualifying model", value: pooled.rmse },
        { label: "Last season’s gap", value: pooled.rmse_naive },
        { label: "No-gap prediction", value: pooled.rmse_zero },
      ].map((entry, index) => <div className="briefing-error-row" key={entry.label}>
        <span>{entry.label}</span><div className="briefing-error-track" aria-hidden="true"><span className={index === 0 ? "briefing-error-model" : ""} style={{ width: `${entry.value / maxError * 100}%` }} /></div><strong>{entry.value.toFixed(3)} s</strong>
      </div>)}</section>
      <p className="briefing-takeaway"><strong>{verdict}.</strong> {comparison ? <>Model minus last season’s mean squared error: {signed(comparison.mse_difference, 4)} s²; event-level 95% interval {signed(comparison.ci95[0], 4)} to {signed(comparison.ci95[1], 4)} s². An interval crossing zero leaves the difference uncertain.</> : "The error values alone do not establish an improvement without a comparison interval."}</p>
      <p className="briefing-units">Interval coverage is the share of real gaps inside the prediction interval. These historical checks use earlier-only data and are distinct from forecasts actually published before a race.</p>
    </> : <p className="briefing-empty">No scored qualifying forecasts are available in this release.</p>}
    <div className="briefing-actions"><a href="#forecasts">Full forecasting record <ArrowRight size={14} aria-hidden="true" /></a><a href="#methodology">Methodology <ArrowRight size={14} aria-hidden="true" /></a></div>
  </Chapter>;
}

export default function Briefing({ data, chapter }: { data: Dataset; chapter: string }) {
  const index = Math.max(0, BRIEFING_CHAPTERS.findIndex((item) => item.id === chapter));
  const current = BRIEFING_CHAPTERS[index];
  const previous = BRIEFING_CHAPTERS[index - 1];
  const next = BRIEFING_CHAPTERS[index + 1];
  return <div className="briefing-view">
    <nav className="briefing-chapters" aria-label="Briefing chapters">{BRIEFING_CHAPTERS.map((item) => <a key={item.id} href={`#briefing/${item.id}`} aria-current={current.id === item.id ? "step" : undefined}>{item.label}</a>)}</nav>
    {current.id === "qualifying" && <QualifyingChapter data={data} />}
    {current.id === "cars" && <QualifyingChapter data={data} cars />}
    {current.id === "race" && <RaceChapter data={data} />}
    {current.id === "overall" && <OverallChapter data={data} />}
    {current.id === "evidence" && <EvidenceChapter data={data} />}
    <nav className="briefing-controls" aria-label="Briefing controls">
      {previous ? <a href={`#briefing/${previous.id}`}><ArrowLeft size={14} aria-hidden="true" />Previous: {previous.label}</a> : <span />}
      <a href={`#briefing/${next?.id || "qualifying"}`}>{next ? `Next: ${next.label}` : "Back to the start"}<ArrowRight size={14} aria-hidden="true" /></a>
    </nav>
  </div>;
}
