import type { Dataset } from "../types";
import { PageHeading } from "../components";
import Methodology from "./Methodology";
import { OverallMethodology } from "./Overall";
import { RaceMethodology } from "./RaceNotes";

export default function MethodologyPage({ data }: { data: Dataset }) {
  const refresh = data.refresh;
  return (
    <>
      <PageHeading title="Methodology">
        What each number means, how it is estimated and how it is checked
      </PageHeading>
      <section className="panel read-guide" aria-labelledby="guide-title">
        <div className="panel-heading">
          <div>
            <h2 id="guide-title">How to read the ratings</h2>
          </div>
        </div>
        <dl>
          <div>
            <dt>Pace in seconds</dt>
            <dd>
              Pace ratings are seconds per 90-second lap against the average of
              the field at that event. +0.20 means two tenths faster than an
              average driver (or car); 0 is the field average.
            </dd>
          </div>
          <div>
            <dt>Driver and car are separate</dt>
            <dd>
              Teammates share a car, so the gap between them measures the
              drivers; drivers who change teams link those comparisons across
              the grid. The car rating is what remains of a team’s pace. A
              driver’s expected pace is their rating plus their car’s.
            </dd>
          </div>
          <div>
            <dt>Ranges, not just ranks</dt>
            <dd>
              Each estimate has a 90% interval and a 90% rank range. Tables are
              ordered by the middle estimate; when rank ranges overlap, the data
              do not clearly separate those positions.
            </dd>
          </div>
          <div>
            <dt>Qualifying and race pace</dt>
            <dd>
              Qualifying pace comes from one-lap times since 2010. Race pace
              comes from clean dry-race laps since 2018, adjusted for tyre
              compound and age, lap number and traffic. They are separate
              models, each checked on races it did not see before it is
              published.
            </dd>
          </div>
          <div>
            <dt>Overall (equal car) · experimental</dt>
            <dd>
              The equal-car championship compares expected points per race,
              title probabilities and simulated season rank ranges. It keeps
              driver–team effects and only admits racing qualities supported by
              its entry tests and combined held-out validation. Missing, stale
              or failed evidence leaves the ranking withheld.
            </dd>
          </div>
          <div>
            <dt>Revised and after each race</dt>
            <dd>
              Revised history uses every race, including later ones. “After each
              race” shows what the data up to that race supported; the Track
              record page scores how well those estimates predicted the next
              qualifying.
            </dd>
          </div>
          <div>
            <dt>Updates</dt>
            <dd>
              A scheduled job checks for a new race every Monday and Tuesday
              (06:00 UTC), refits both models and republishes once the
              convergence and provenance checks pass. A table that fails its
              prediction checks stays withheld.
              {refresh?.event &&
                ` The latest completed refit includes the ${refresh.event.race_name}.`}
            </dd>
          </div>
        </dl>
      </section>
      <Methodology data={data} />
      <RaceMethodology data={data.race_pace} />
      <OverallMethodology />
    </>
  );
}
