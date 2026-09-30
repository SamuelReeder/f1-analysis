import { Layers3 } from "lucide-react";
import type { Estimate, Metric } from "./types";
import { signed } from "./lib";

export function Badge({
  children,
  tone = "",
}: {
  children: React.ReactNode;
  tone?: string;
}) {
  return (
    <span className={`badge ${tone}`}>
      <span className="badge-dot" />
      {children}
    </span>
  );
}
export function Empty({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="empty">
      <Layers3 size={28} />
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
export function Band({
  value,
  domain,
  compact = false,
}: {
  value: Estimate;
  domain: number[];
  compact?: boolean;
}) {
  const pos = (v: number) =>
    8 + ((v - domain[0]) / (domain[1] - domain[0])) * 184;
  return (
    <svg
      className={`band ${compact ? "compact" : ""}`}
      viewBox="0 0 200 26"
      role="img"
      aria-label={`Pace ${signed(value.median)} seconds, 90 percent interval ${signed(value.q05)} to ${signed(value.q95)}`}
    >
      <line
        x1={pos(0)}
        x2={pos(0)}
        y1="0"
        y2="26"
        stroke="#d5dbd4"
        strokeDasharray="2 3"
      />
      <line
        x1={pos(value.q05)}
        x2={pos(value.q95)}
        y1="13"
        y2="13"
        stroke="currentColor"
        strokeWidth="3"
        opacity=".28"
      />
      <line
        x1={pos(value.q25 ?? value.q05)}
        x2={pos(value.q75 ?? value.q95)}
        y1="13"
        y2="13"
        stroke="currentColor"
        strokeWidth="6"
        strokeLinecap="round"
        opacity=".65"
      />
      <circle
        cx={pos(value.median)}
        cy="13"
        r="4.5"
        fill="currentColor"
        stroke="white"
        strokeWidth="1.5"
      />
    </svg>
  );
}

export function PageHeading({
  title,
  children,
  action,
}: {
  title: string;
  children?: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <h1>{title}</h1>
        {children && <p>{children}</p>}
      </div>
      {action}
    </div>
  );
}
export function MetricControl({
  metric,
  setMetric,
}: {
  metric: Metric;
  setMetric: (m: Metric) => void;
}) {
  return (
    <div className="segmented" aria-label="Driver metric">
      <button
        onClick={() => setMetric("headline")}
        aria-pressed={metric === "headline"}
      >
        In-team pace
      </button>
      <button
        onClick={() => setMetric("portable")}
        aria-pressed={metric === "portable"}
      >
        Portable skill <span>Experimental</span>
      </button>
    </div>
  );
}
