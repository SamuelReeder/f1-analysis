// Relative performance colors are derived from central estimates, never ranks
// or uncertainty bounds. Keep the complete field when applying table filters.
const stops = [[248, 113, 113], [251, 191, 36], [74, 222, 128]];
const rgb = (channels: number[]) => `rgb(${channels.join(", ")})`;

export function performanceScale(values: number[]) {
  const finite = values.filter(Number.isFinite);
  const low = Math.min(...finite);
  const high = Math.max(...finite);
  return (value: number) => {
    if (!Number.isFinite(value) || !finite.length) return "var(--muted)";
    const position = high === low ? .5 : Math.max(0, Math.min(1, (value - low) / (high - low)));
    const segment = position < .5 ? 0 : 1;
    const fraction = position * 2 - segment;
    return rgb(stops[segment].map((channel, index) =>
      Math.round(channel + (stops[segment + 1][index] - channel) * fraction)));
  };
}

export function PerformanceKey({ points = false }: { points?: boolean }) {
  return <p className="performance-key">
    <span className="performance-key-scale">Lower
      <span aria-hidden="true" style={{ background: `linear-gradient(to right, ${stops.map(rgb).join(", ")})` }} />
      Higher
    </span>
    <span>Color compares {points ? "expected points" : "central pace estimates"} within this field, not certainty.</span>
  </p>;
}
