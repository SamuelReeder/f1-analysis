import type { Car, Driver, Dataset, Metric } from "./types";

const colors: Record<string, string> = {
  brackley: "#218981",
  ferrari: "#cf4549",
  mclaren: "#d37b2c",
  red_bull: "#4677ba",
  milton_keynes: "#4677ba",
  williams: "#5684c7",
  enstone: "#a065a5",
  silverstone: "#548775",
  sauber: "#707b69",
  faenza: "#6b83ab",
  haas: "#9b7773",
  cadillac: "#888a99",
};
export const color = (id: string) => colors[id] || "#7c8795";
export const signed = (n: number, digits = 3) =>
  `${n > 0 ? "+" : n < 0 ? "−" : ""}${Math.abs(n).toFixed(digits)}`;
export const pct = (n: number = 0) => `${(n * 100).toFixed(1)}%`;
export const date = (s: string) =>
  new Date(s.length === 10 ? `${s}T12:00:00` : s).toLocaleDateString("en", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
export const shortName = (name: string) => name.split(" ").slice(-1)[0];
export const metricLabel = (m: Metric) =>
  m === "headline" ? "Pace in current team" : "Portable skill";
export function download(name: string, body: string, type = "text/csv") {
  const url = URL.createObjectURL(new Blob([body], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}
export function exportCsv(
  rows: (Driver | Car)[],
  car: boolean,
  metric: Metric,
  data: Dataset,
) {
  const headers = [
    "rank",
    "name",
    "team",
    "metric",
    "pace_seconds_per_90s",
    "lower_90",
    "upper_90",
    "rank_low",
    "rank_high",
    "probability_fastest",
    "as_of_event",
    "fit_id",
  ];
  const body = rows.map((r, i) => {
    const v = car ? (r as Car).pace : (r as Driver)[metric];
    return [
      i + 1,
      r.name,
      car ? r.name : (r as Driver).team,
      car ? "track_neutral_car_pace" : metric,
      v.median,
      v.q05,
      v.q95,
      v.rank_lo,
      v.rank_hi,
      v.p_fastest,
      data.meta.data_as_of.event_id,
      data.meta.fit_id,
    ];
  });
  download(
    `f1-${car ? "cars" : metric}-${data.meta.data_as_of.event_id}.csv`,
    [headers, ...body]
      .map((r) =>
        r.map((v) => `"${String(v ?? "").replaceAll('"', '""')}"`).join(","),
      )
      .join("\n"),
  );
}
