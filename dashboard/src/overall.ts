import contract from "../../contracts/overall.json";
import type { CurrentOverallEvidence, CurrentOverallResult, OverallResult } from "./types";

export const overallCheckLabels: Record<string, string> = {
  provenance: "Current source and data",
  convergence: "Stable model estimates",
  recovery: "Recovery of known driver and car effects",
  chronological_prediction: "Predictions using only earlier races",
  grid_calibration: "Qualifying predictions and uncertainty",
  race_calibration: "Race predictions and uncertainty",
  retirement_calibration: "Completion and retirement predictions",
  prospective_prediction: "New predictions recorded before their outcomes",
  simulation_precision: "Simulation precision",
};
export const overallRequiredChecks = contract.required_checks;
export const isCurrentOverall = (result?: OverallResult): result is CurrentOverallResult => result?.schema_version === 2;
export const isSupportedOverall = (result?: OverallResult): boolean =>
  !result || isCurrentOverall(result) || (
    (result.schema_version === undefined || result.schema_version === 1) &&
    !("model_version" in result) && !("target" in result)
  );
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const text = (v: unknown): v is string => typeof v === "string" && v.trim().length > 0;
const bounded = (v: unknown, low: number, high: number): v is number => finite(v) && v >= low && v <= high;
const positiveInteger = (v: unknown): v is number => finite(v) && Number.isInteger(v) && v > 0;

export function currentOverallEvidence(result: CurrentOverallResult): CurrentOverallEvidence | null {
  const e = result.evidence;
  if (!e || !text(e.fit_id) || !text(e.recorded_at) || !text(e.data_as_of) ||
      !e.scenario || !text(e.scenario.id) || !text(e.scenario.label) || !positiveInteger(e.scenario.races) ||
      !positiveInteger(e.posterior_draws) || !positiveInteger(e.replicates_per_draw) ||
      !Array.isArray(e.field) || e.field.length < 2 || !e.field.every(text) || new Set(e.field).size !== e.field.length ||
      !e.checks || typeof e.checks !== "object") return null;
  return e;
}

// This is a display safeguard in addition to the producer's artifact checks.
// The legacy combined gate can never stand in for the successor's evidence.
export function currentOverallAvailable(result: CurrentOverallResult): boolean {
  const e = currentOverallEvidence(result);
  if (result.status !== "established" || result.model_version !== contract.model_version ||
      result.target !== contract.target || !e || !Array.isArray(result.standings) ||
      !Array.isArray(result.contributions) || !Array.isArray(result.limitations) ||
      !result.limitations.every(text) || result.standings.length !== e.field.length ||
      !overallRequiredChecks.every(key => e.checks[key]?.passed === true && text(e.checks[key].source) && text(e.checks[key].reason))) return false;
  const ids = result.standings.map(row => row?.driver_id);
  if (new Set(ids).size !== ids.length || !ids.every(id => e.field.includes(id))) return false;
  const n = e.field.length;
  const rowsValid = result.standings.every(row => {
    const p = row?.expected_points, rank = row?.season_rank;
    return text(row?.name) && p && rank && bounded(p.mean, 0, contract.points[0]) &&
      bounded(p.q05, 0, contract.points[0]) && bounded(p.q95, p.q05, contract.points[0]) &&
      finite(p.posterior_sd) && p.posterior_sd >= 0 &&
      bounded(p.mcse, 0, contract.max_conditional_points_mcse) &&
      bounded(p.max_conditional_mcse, p.mcse, contract.max_conditional_points_mcse) &&
      bounded(row.p_title, 0, 1) && bounded(rank.q05, 1, n) &&
      bounded(rank.q95, rank.q05, n) && bounded(rank.median, rank.q05, rank.q95);
  });
  if (!rowsValid || Math.abs(result.standings.reduce((sum, row) => sum + row.p_title, 0) - 1) >
      contract.title_probability_sum_tolerance) return false;
  const explanations = result.contributions;
  return explanations.length === n && new Set(explanations.map(row => row?.driver_id)).size === n &&
    explanations.every(row => row && ids.includes(row.driver_id) && row.fit_id === e.fit_id &&
      row.scenario_id === e.scenario.id && row.losses &&
      contract.required_explanations.every(key => finite(row.losses[key])) &&
      Object.values(row.losses).every(finite));
}

export function currentOverallState(result: CurrentOverallResult): string {
  if (result.status === "established") return currentOverallAvailable(result) ? "established" : "not established";
  return ["not established", "stale", "unavailable"].includes(result.status) ? result.status : "unavailable";
}
export function currentOverallReason(result: CurrentOverallResult): string {
  return result.status === "established" && !currentOverallAvailable(result)
    ? "This release does not contain a complete, validated Overall ranking."
    : text(result.reason) ? result.reason : "No verified Overall result is available yet.";
}
