import type { ReactNode } from "react";
import { ArrowLeft, ArrowRight, ArrowUpRight, Check, CircleDashed, Layers3 } from "lucide-react";
import { date, pct, signed } from "../lib";
import type { Dataset, Estimate, RaceValidation } from "../types";
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
  qualifying_pace: "Qualifying pace",
  first_lap: "First-lap performance",
  race_specific_pace: "Race-specific pace",
  degradation: "Tyre management",
  consistency: "Consistency",
  overtaking_attack: "Overtaking",
  overtaking_defend: "Defending",
};
const qualityLabel = (value: string) => qualityLabels[value] || value.replaceAll("_", " ");

function Drilldown({ href, children }: { href: string; children: ReactNode }) {
  return <a className="briefing-drilldown" href={href}>{children}<ArrowUpRight size={17} aria-hidden="true" /></a>;
}

function Narrative({ label, title, takeaway, context, source, children }: {
  label: string; title: string; takeaway: ReactNode; context: ReactNode; source?: ReactNode; children?: ReactNode;
}) {
  return <div className="briefing-narrative">
    <p className="briefing-eyebrow">{label}</p>
    <h1>{title}</h1>
    <p className="briefing-takeaway">{takeaway}</p>
    <div className="briefing-reading"><h2>How to read this</h2><p>{context}</p></div>
    {children && <div className="briefing-actions">{children}</div>}
    {source && <p className="briefing-source">{source}</p>}
  </div>;
}

function PaceRanking({ rows, title, subtitle }: { rows: PaceEntry[]; title: string; subtitle: string }) {
  const top = [...rows].sort((a, b) => b.pace.median - a.pace.median).slice(0, 5);
  const lo = Math.min(0, ...top.map((row) => row.pace.q05));
  const hi = Math.max(0, ...top.map((row) => row.pace.q95));
  const span = hi - lo || 1;
  const position = (value: number) => `${((value - lo) / span) * 100}%`;
  return <section className="briefing-paper briefing-pace-card" aria-label={title}>
    <div className="briefing-card-heading"><div><p className="briefing-card-kicker">THE LEADING ESTIMATES</p><h2>{title}</h2></div><Layers3 size={22} aria-hidden="true" /></div>
    <p className="briefing-card-subtitle">{subtitle}</p>
    {top.length ? <>
      <div className="briefing-ranking-labels" aria-hidden="true"><span>Estimate & uncertainty</span><span>Faster <ArrowRight size={13} /></span></div>
      <ol className="briefing-ranking">
        {top.map((row) => <li className="briefing-ranking-row" key={row.id}>
          <div className="briefing-row-heading"><div><strong>{row.name}</strong>{row.team && <span className="briefing-row-team">{row.team}</span>}</div><span className="briefing-pace-value">{signed(row.pace.median)}<small> s</small></span></div>
          <div className="briefing-interval" aria-hidden="true">
            <span className="briefing-zero" style={{ left: position(0) }} />
            <span className="briefing-band" style={{ left: position(row.pace.q05), width: `${((row.pace.q95 - row.pace.q05) / span) * 100}%` }} />
            <span className="briefing-dot" style={{ left: position(row.pace.median) }} />
          </div>
          <div className="briefing-row-detail"><span>Rank range <strong>{rankRange(row.pace)}</strong></span><span className="briefing-row-interval">90% interval: {signed(row.pace.q05)} to {signed(row.pace.q95)} s</span></div>
          {row.carried && <p className="briefing-carried">Carried forward · no timed lap at this event</p>}
        </li>)}
      </ol>
      <p className="briefing-card-foot">Seconds per 90-second reference lap. Higher is faster. Bars show the uncertainty around each estimate.</p>
    </> : <div className="briefing-empty"><CircleDashed size={32} aria-hidden="true" /><h3>No estimates in this release</h3><p>The full results will appear when a checked release is available.</p></div>}
  </section>;
}

function QualifyingChapter({ data, cars = false }: { data: Dataset; cars?: boolean }) {
  const rows: PaceEntry[] = cars
    ? data.cars.map((car) => ({ id: car.id, name: car.name, pace: car.pace }))
    : data.drivers.map((driver) => ({ id: driver.id, name: driver.name, team: driver.team, pace: driver.headline, carried: !driver.has_time }));
  const leader = [...rows].sort((a, b) => b.pace.median - a.pace.median)[0];
  return <div className="briefing-stage">
    <Narrative label={cars ? "THE CAR CONTRIBUTION" : "QUALIFYING · CURRENT TEAM"}
      title={cars ? "The machinery\nbehind the pace." : "Who stands out\nin qualifying?"}
      takeaway={leader ? <><strong>{leader.name}</strong> has the highest central {cars ? "car-pace" : "driver-pace"} estimate in this release. The rank ranges show the uncertainty behind that order.</> : "This release has no qualifying estimates to summarize yet."}
      context={cars ? "This separates the car contribution from the driver contribution at an average circuit. It describes machinery in qualifying, with driver effects accounted for." : "Teammate comparisons help separate driver pace from machinery. The headline retains each driver’s estimated fit with their current team; it is not a measure of purely transferable skill."}
      source={<>Qualifying data through {data.meta.data_as_of.race_name} · {date(data.meta.data_as_of.date)}</>}>
      <Drilldown href={cars ? "#cars" : "#drivers"}>{cars ? "Explore all car rankings" : "Explore all driver rankings"}</Drilldown>
      <a className="briefing-text-link" href={cars ? "#compare/cars" : "#compare"}>Compare two {cars ? "cars" : "drivers"} <ArrowRight size={14} aria-hidden="true" /></a>
    </Narrative>
    <PaceRanking rows={rows} title={cars ? "Qualifying car pace" : "Qualifying driver pace"}
      subtitle={cars ? "Average circuit · driver effects accounted for" : "In current team · driver–team effect retained"} />
  </div>;
}

function failedRaceReason(validation?: RaceValidation) {
  if (!validation) return "No current validation result is available.";
  const reasons = [!validation.improves && "prediction improvement", !validation.calibrated && "interval coverage", !validation.sharper && "narrower intervals"].filter(Boolean);
  return reasons.length ? `Publication checks not met: ${reasons.join(", ")}.` : "The combined publication check has not passed; see the full validation details.";
}

function RaceChapter({ data }: { data: Dataset }) {
  const race = data.race_pace;
  const driverPassed = race?.validation.metrics.drivers.passed === true;
  const carPassed = race?.validation.metrics.cars.passed === true;
  const driverLeader = driverPassed ? [...(race?.drivers || [])].sort((a, b) => b.pace.median - a.pace.median)[0] : undefined;
  const carLeader = carPassed ? [...(race?.cars || [])].sort((a, b) => b.pace.median - a.pace.median)[0] : undefined;
  const carName = carLeader ? data.cars.find((car) => car.id === carLeader.id)?.name || carLeader.name : "";
  const takeaway = driverLeader
    ? <><strong>{driverLeader.name}</strong> has the highest central driver race-pace estimate.{carLeader && <> <strong>{carName}</strong> has the highest car estimate.</>} These are dry-race estimates, with uncertain positions.</>
    : carLeader
      ? <><strong>{carName}</strong> has the highest central car race-pace estimate. The driver ranking {driverPassed ? "has no eligible entries yet" : "is withheld until its own checks pass"}.</>
      : <>The race-pace picture {driverPassed || carPassed ? "has no eligible entries to summarize" : "is not established yet"}. Driver and car rankings earn publication separately.</>;
  return <div className="briefing-stage briefing-stage-race">
    <Narrative label="RACE PACE · SEPARATE CHECKS" title={"One lap is only\npart of the story."} takeaway={takeaway}
      context="Race pace uses clean, dry laps with fuel and tyre effects accounted for. Driver and car estimates must each pass their own prediction and uncertainty checks; qualifying strength alone does not establish race strength."
      source={race ? <>Race data through {race.data_as_of.race_name} · {date(race.data_as_of.date)}</> : "No checked race-pace release is available."}>
      <Drilldown href="#drivers/race">Explore driver race pace</Drilldown>
      <a className="briefing-text-link" href="#cars/race">Explore car race pace <ArrowRight size={14} aria-hidden="true" /></a>
    </Narrative>
    <div className="briefing-race-panels">
      {(["drivers", "cars"] as const).map((kind) => {
        const validation = race?.validation.metrics[kind];
        const passed = validation?.passed === true;
        const entries = passed ? (race?.[kind] || []) : [];
        const title = kind === "drivers" ? "Driver race pace" : "Car race pace";
        return <section key={kind} className="briefing-paper briefing-race-card" aria-label={title}>
          <div className="briefing-card-heading"><h2>{title}</h2><span className={`briefing-state ${passed ? "briefing-state-pass" : ""}`}>{passed ? <Check size={13} aria-hidden="true" /> : <CircleDashed size={13} aria-hidden="true" />}{passed ? "Checks passed" : "Withheld"}</span></div>
          {entries.length ? <ol className="briefing-ranking briefing-race-ranking">{[...entries].sort((a, b) => b.pace.median - a.pace.median).slice(0, 3).map((row) => <li className="briefing-ranking-row" key={row.id}>
            <div className="briefing-row-heading"><strong>{kind === "cars" ? data.cars.find((car) => car.id === row.id)?.name || row.name : row.name}</strong><span className="briefing-pace-value">{signed(row.pace.median)}<small> s</small></span></div>
            <div className="briefing-row-detail"><span>Rank range <strong>{rankRange(row.pace)}</strong></span></div>
          </li>)}</ol> : <p className="briefing-race-reason">{passed ? "The checks passed, but no current entries meet the data requirements." : failedRaceReason(validation)}</p>}
          <a className="briefing-card-link" href={`#${kind}/race`}>View {kind === "drivers" ? "driver" : "car"} results & checks <ArrowUpRight size={15} aria-hidden="true" /></a>
        </section>;
      })}
      <p className="briefing-panel-caption">Pace in seconds per 90-second reference lap · higher is faster. Rank ranges preserve uncertainty.</p>
    </div>
  </div>;
}

function OverallChapter({ data }: { data: Dataset }) {
  const result = data.overall;
  const evidence = result?.status === "established" || result?.status === "not established" ? result.evidence : null;
  const established = result?.status === "established" && evidence?.combined_validation.gate === true && result.standings.length > 0;
  const rows = established ? [...result.standings].sort((a, b) => b.points_per_race - a.points_per_race) : [];
  const leader = rows[0];
  const state = result?.status === "established" && !established ? "not established" : result?.status || "unavailable";
  const reason = result?.status === "established" && !established ? "This release does not contain a validated overall ranking." : result?.reason || "No verified championship results are available in this release yet.";
  return <div className="briefing-stage">
    <Narrative label="EXPERIMENTAL · EQUAL CAR" title={"The same car.\nA different question."}
      takeaway={leader ? <><strong>{leader.name}</strong> has the most expected points per race in this equal-car simulation. The rank range shows how uncertain season outcomes remain.</> : <>An overall ranking is <strong>{state}</strong>. The championship stays withheld until current results pass the combined validation check.</>}
      context="Simulated seasons give drivers equal average machinery and mechanical retirement risk. Estimated driver–team effects remain. Only racing qualities supported by the entry tests can join the championship."
      source={evidence ? <>Championship recorded {date(evidence.recorded_at)}{evidence.data_as_of && <> · Data through {evidence.data_as_of}</>}</> : "No verified championship release"}>
      <Drilldown href="#drivers/overall">Explore the equal-car championship</Drilldown>
      <p className="briefing-caveat">This simplified scenario excludes lap-by-lap passing, traffic, pit strategy and wet conditions. It is not a forecast of the real championship.</p>
    </Narrative>
    <section className="briefing-paper briefing-overall-card" aria-label="Equal-car championship summary">
      <div className="briefing-card-heading"><div><p className="briefing-card-kicker">THE CHAMPIONSHIP QUESTION</p><h2>Overall, in equal machinery</h2></div><Layers3 size={22} aria-hidden="true" /></div>
      <p className="briefing-card-subtitle">Experimental · driver–team effects retained</p>
      {established ? <>
        <ol className="briefing-ranking briefing-overall-ranking">{rows.slice(0, 5).map((row) => <li className="briefing-ranking-row" key={row.driver_id}>
          <div className="briefing-row-heading"><strong>{row.name}</strong><span className="briefing-pace-value">{row.points_per_race.toFixed(2)}<small> pts/race</small></span></div>
          <div className="briefing-row-detail"><span>Title probability <strong>{pct(row.p_title)}</strong></span><span>Rank range <strong>{row.rank_lo}–{row.rank_hi}</strong></span></div>
        </li>)}</ol>
        <p className="briefing-card-foot">{evidence.simulated_seasons.toLocaleString()} simulated seasons · {evidence.n_races_simulated} races per season. Title probability is the share of simulated seasons won.</p>
      </> : <div className="briefing-withheld"><div className="briefing-withheld-symbol" aria-hidden="true"><CircleDashed size={48} strokeWidth={1} /></div><p className="briefing-card-kicker">EVIDENCE BEFORE ORDER</p><h3>Overall ranking {state}</h3><p>{reason}</p><span className="briefing-state">Ranking withheld</span></div>}
      {evidence && <div className="briefing-quality-summary"><h3>Qualities admitted by the tests</h3><p>{evidence.qualities_entered.length ? evidence.qualities_entered.map(qualityLabel).join(" · ") : "No additional racing qualities entered."}</p><p>Not entered: {evidence.qualities_not_entered.length ? evidence.qualities_not_entered.map(qualityLabel).join(" · ") : "None recorded"}.</p><p>Excluded test seasons — combined validation: {evidence.excluded_test_seasons.combined_validation.join(", ") || "None recorded"}; race stage: {evidence.excluded_test_seasons.heldout_race_stage.join(", ") || "None recorded"}.</p><a className="briefing-card-link" href="#drivers/overall">See contributions, exclusions & decisions <ArrowUpRight size={15} aria-hidden="true" /></a></div>}
    </section>
  </div>;
}

function EvidenceChapter({ data }: { data: Dataset }) {
  const pooled = data.asof?.pooled;
  const comparison = pooled?.vs_naive;
  const verdict = !comparison ? "Baseline comparison unavailable" : comparison.ci95[1] < 0 ? "Better than last season’s gap" : comparison.ci95[0] > 0 ? "Worse than last season’s gap" : "No clear difference from last season’s gap";
  const latest = [...(data.asof?.events || [])].filter((event) => event.n_pairs > 0).sort((a, b) => b.event.date.localeCompare(a.event.date))[0];
  const maxError = pooled ? Math.max(pooled.rmse, pooled.rmse_naive, pooled.rmse_zero) || 1 : 1;
  return <div className="briefing-stage">
    <Narrative label="OUT-OF-SAMPLE · QUALIFYING" title={"An estimate earns\nits place here."}
      takeaway={pooled ? comparison ? <><strong>{verdict}.</strong> That is what the recorded event-level uncertainty comparison supports, across {pooled.n_events} scored events.</> : <>The model has been scored across <strong>{pooled.n_events} events</strong>, but a baseline uncertainty comparison is unavailable. The error values alone do not establish an improvement.</> : "There are no scored qualifying forecasts in this release. Predictive accuracy is not established here yet."}
      context="For each scored race, the qualifying model uses data that stops before that race. The check compares predicted teammate gaps with observed gaps. Lower prediction error is better; interval coverage shows how often real gaps fell inside the model’s uncertainty range."
      source={latest ? <>Scored through {latest.event.race_name} · {date(latest.event.date)}</> : "No scored-event date is available."}>
      <Drilldown href="#forecasts">Explore the forecasting record</Drilldown>
      <a className="briefing-text-link" href="#methodology">Read the full methodology <ArrowRight size={14} aria-hidden="true" /></a>
      <p className="briefing-caveat">These are qualifying checks. Race pace and the experimental equal-car championship have separate publication gates.</p>
    </Narrative>
    <section className="briefing-paper briefing-evidence-card" aria-label="Qualifying prediction evidence">
      <div className="briefing-card-heading"><div><p className="briefing-card-kicker">THE MODEL’S TRACK RECORD</p><h2>Tested against unseen races</h2></div><Layers3 size={22} aria-hidden="true" /></div>
      {pooled ? <>
        <dl className="briefing-score-grid"><div><dt>Teammate gap error</dt><dd>{pooled.rmse.toFixed(3)}<small> s</small></dd></div><div><dt>90% interval coverage</dt><dd>{pct(pooled.coverage90)}</dd></div><div><dt>Scored events</dt><dd>{pooled.n_events}</dd></div></dl>
        <div className="briefing-error-chart"><h3>Prediction error · lower is better</h3>{[
          { label: "Qualifying model", value: pooled.rmse },
          { label: "Last season’s gap", value: pooled.rmse_naive },
          { label: "No-gap prediction", value: pooled.rmse_zero },
        ].map((entry, index) => <div className="briefing-error-row" key={entry.label}><div><span>{entry.label}</span><strong>{entry.value.toFixed(3)} s</strong></div><div className="briefing-error-track" aria-hidden="true"><span className={index === 0 ? "briefing-error-model" : ""} style={{ width: `${entry.value / maxError * 100}%` }} /></div></div>)}</div>
        <div className="briefing-evidence-verdict"><h3>{verdict}</h3>{comparison ? <p>Model minus last season’s mean squared error: {signed(comparison.mse_difference, 4)} s². The event-level 95% interval is {signed(comparison.ci95[0], 4)} to {signed(comparison.ci95[1], 4)} s². An interval crossing zero leaves the difference uncertain.</p> : <p>The release has no uncertainty interval for comparison with last season’s gap, so the error values alone do not establish an improvement.</p>}</div>
        <p className="briefing-card-foot">{pooled.n_pairs.toLocaleString()} scored teammate gaps. Historical checks were reconstructed from earlier-only data; they are distinct from forecasts actually published before a race.</p>
      </> : <div className="briefing-empty"><CircleDashed size={32} aria-hidden="true" /><h3>No scored forecasts yet</h3><p>A checked forecasting record will appear here when it is included in the published data.</p></div>}
    </section>
  </div>;
}

export default function Briefing({ data, chapter }: { data: Dataset; chapter: string }) {
  const index = Math.max(0, BRIEFING_CHAPTERS.findIndex((item) => item.id === chapter));
  const current = BRIEFING_CHAPTERS[index];
  const previous = BRIEFING_CHAPTERS[index - 1];
  const next = BRIEFING_CHAPTERS[index + 1];
  return <div className="briefing-view">
    <div className="briefing-topline"><span className="briefing-kicker"><span aria-hidden="true" />THE GRID, EXPLAINED</span><span className="briefing-intro-label">A guided look at the evidence</span></div>
    <nav className="briefing-chapters" aria-label="Briefing chapters">{BRIEFING_CHAPTERS.map((item, i) => <a key={item.id} href={`#briefing/${item.id}`} aria-current={current.id === item.id ? "step" : undefined}><span className="briefing-chapter-number" aria-hidden="true">{String(i + 1).padStart(2, "0")}</span><span>{item.label}</span></a>)}</nav>
    {current.id === "qualifying" && <QualifyingChapter data={data} />}
    {current.id === "cars" && <QualifyingChapter data={data} cars />}
    {current.id === "race" && <RaceChapter data={data} />}
    {current.id === "overall" && <OverallChapter data={data} />}
    {current.id === "evidence" && <EvidenceChapter data={data} />}
    <nav className="briefing-controls" aria-label="Briefing controls"><div>{previous ? <a className="briefing-previous" href={`#briefing/${previous.id}`}><ArrowLeft size={16} aria-hidden="true" />Previous: {previous.label}</a> : <span className="briefing-control-note">The rankings. The context. The confidence.</span>}</div><a className="briefing-next" href={`#briefing/${next?.id || "qualifying"}`}><span>{next ? "CONTINUE THE BRIEFING" : "THE BRIEFING IS COMPLETE"}<strong>{next ? `Next: ${next.label}` : "Back to the start"}</strong></span><ArrowRight size={23} aria-hidden="true" /></a></nav>
  </div>;
}
