import { PageHeading } from "../components";
import type { Discipline, RankingKind } from "../types";

export default function RankingHeader({
  kind,
  discipline,
  action,
}: {
  kind: RankingKind;
  discipline: Discipline;
  action?: React.ReactNode;
}) {
  return (
    <>
      <PageHeading
        title={kind === "drivers" ? "Drivers" : "Cars"}
        action={action}
      />
      <nav
        className="ranking-metrics"
        aria-label={`${kind === "drivers" ? "Driver" : "Car"} ranking metric`}
      >
        <a
          href={`#${kind}`}
          aria-current={discipline === "qualifying" ? "page" : undefined}
        >
          Qualifying pace
        </a>
        <a
          href={`#${kind}/race`}
          aria-current={discipline === "race" ? "page" : undefined}
        >
          Race pace
        </a>
        {kind === "drivers" && (
          <a href="#drivers/overall" aria-current={discipline === "overall" ? "page" : undefined}>
            Overall (equal car)
          </a>
        )}
      </nav>
    </>
  );
}
