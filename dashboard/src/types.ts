export type Metric = "headline" | "portable";
export type View =
  "drivers" | "cars" | "compare" | "forecasts" | "methodology" | "health";
export type RankingKind = "drivers" | "cars";
export type Discipline = "qualifying" | "race";
export interface Estimate {
  q05: number;
  q25?: number;
  median: number;
  q75?: number;
  q95: number;
  rank_lo?: number;
  rank_hi?: number;
  rank_median?: number;
  p_fastest?: number;
  p_top3?: number;
}
export interface Driver {
  id: string;
  name: string;
  code: string;
  team: string;
  lineage: string;
  has_time: boolean;
  headline: Estimate;
  portable: Estimate;
  team_effect: Estimate;
  evidence: {
    events: number;
    seasons: number;
    teammates: number;
    teams: number;
  };
}
export interface Car {
  id: string;
  name: string;
  pace: Estimate;
  gap_to_best: Estimate;
}
export interface Event {
  event_id: string;
  season: number;
  round: number;
  date: string;
  race_name: string;
  circuit_id?: string;
}
export interface Point {
  event: string;
  team: string;
  headline?: Estimate;
  portable?: Estimate;
  pace?: Estimate;
  at_circuit?: Estimate;
}
export interface Pair extends Estimate {
  p_ahead: number;
}
export interface RaceEntity {
  id: string;
  name: string;
  team?: string;
  lineage?: string;
  code?: string;
  pace: Estimate;
  races: number;
  laps: number;
  last_race: string;
}
export interface RaceValidation {
  passed: boolean;
  improves: boolean;
  calibrated: boolean;
  sharper: boolean;
  n_races: number;
  n_predictions: number;
  rmse: number;
  baseline_rmse: number;
  mse_difference: number;
  mse_difference_ci95: number[];
  coverage90: number;
  mean_interval_width: number;
  baseline_interval_width: number;
}
export interface RacePace {
  model: string;
  fit_id: string;
  season: number;
  data_as_of: Event;
  first_event: string;
  grid_as_of?: string;
  n_laps: number;
  n_races: number;
  drivers: RaceEntity[];
  cars: RaceEntity[];
  seasons?: { season: number; drivers: RaceEntity[]; cars: RaceEntity[] }[];
  unrated_drivers: string[];
  unrated_cars: string[];
  diagnostics: Pick<
    Dataset["meta"]["diagnostics"],
    "rhat_max" | "divergences" | "n_draws" | "converged"
  >;
  validation: {
    folds: string[];
    design: string;
    metrics: { drivers: RaceValidation; cars: RaceValidation };
  };
}
export interface Dataset {
  schema_version: number;
  meta: {
    model_version: string;
    fit_id: string;
    generated_at: string;
    data_as_of: { event_id: string; race_name: string; date: string };
    window: string;
    n_lap_times: number;
    diagnostics: {
      rhat_max: number;
      divergences: number;
      n_draws: number;
      converged: boolean;
      elapsed_s: number;
    };
  };
  drivers: Driver[];
  cars: Car[];
  race_pace?: RacePace | null;
  breakdown?: BreakdownRow[];
  asof?: Asof | null;
  refresh?: RefreshRecord | null;
  race_snapshots?: RaceSnapshot[];
  events: Event[];
  catalog: {
    drivers: { id: string; name: string }[];
    cars: { id: string; name: string }[];
  };
  history: { drivers: Record<string, Point[]>; cars: Record<string, Point[]> };
  comparisons: Record<Metric | "cars", Record<string, Pair>>;
  comparison_draws: number;
  racing: {
    name: string;
    status: string;
    reason: string;
    detail?: string;
    source: string;
  }[];
  validation: {
    gates?: { data: Record<string, boolean>; recorded_at: string | null };
    lfo_summary?: {
      data: {
        n_cutoffs: number;
        pairing_all: {
          rmse_pred: number;
          rmse_akm: number;
          rmse_naive: number;
          rmse_zero: number;
        };
        teammate_session: { cov90: number };
      };
    };
  };
  provenance: {
    export_manifest: string;
    inputs: Record<string, string>;
    outputs: Record<string, string>;
  };
  snapshots: {
    file: string;
    fit_id: string;
    generated_at: string;
    event: { race_name: string; event_id: string };
    drivers: Record<string, string | number>[];
    cars: Record<string, string | number>[];
  }[];
}
export interface RunStatus {
  state: "ok" | "running" | "failed";
  stage: string;
  started_at: string;
  finished_at?: string;
  duration_s?: number;
  error?: string;
}
export interface Release {
  release: string;
  url: string;
  published_at: string;
}
export interface BreakdownRow {
  id: string;
  team: string;
  car: number;
  driver: number;
  total: Estimate;
}
// Estimates after each race have the same shape as the revised history.
export type AsofPoint = Point;
export interface ForecastPair {
  segment: string;
  team: string;
  a: string;
  b: string;
  predicted: number;
  q05: number;
  q95: number;
  observed: number;
  naive: number | null;
  established: boolean;
}
export interface ForecastEvent {
  event: { event_id: string; race_name: string; date: string };
  trained_through: string;
  fit_id: string;
  n_pairs: number;
  rmse: number | null;
  rmse_naive: number | null;
  rmse_zero: number | null;
  coverage90: number | null;
  order_spearman: number | null;
}
export interface Asof {
  pooled: {
    n_events: number;
    n_pairs: number;
    rmse: number;
    rmse_naive: number;
    rmse_zero: number;
    coverage90: number;
    order_spearman: number;
  } | null;
  events: ForecastEvent[];
  series: {
    drivers: Record<string, AsofPoint[]>;
    cars: Record<string, AsofPoint[]>;
  };
  latest: {
    event: { event_id: string; race_name: string; date: string };
    trained_through: { event_id: string; race_name: string; date: string };
    pairs: ForecastPair[];
    order: { segment: string; n: number; spearman: number }[];
  } | null;
}
export interface RefreshRecord {
  started_at: string;
  finished_at: string;
  races: boolean;
  event: { event_id: string; race_name: string; date: string } | null;
  trigger: string;
  run_url: string | null;
  stages: { stage: string; seconds: number }[];
}
export interface RaceSnapshot {
  model: string;
  fit_id: string;
  data_as_of: Event;
  grid_as_of: string;
  generated_at: string;
  passed: { drivers: boolean; cars: boolean };
  drivers: RaceEntity[];
  cars: RaceEntity[];
}
