import { useState, useRef, useEffect, useCallback } from "react";
import {
  Activity,
  ArrowLeft,
  ArrowLeftRight,
  ArrowRight,
  BookOpen,
  LayoutDashboard,
  ChevronRight,
  Target,
  Gauge,
  LoaderCircle,
  RefreshCw,
  TriangleAlert,
  Users,
} from "lucide-react";
import type {
  Dataset,
  Discipline,
  Metric,
  Release,
  RunStatus,
  View,
} from "./types";
import Rankings from "./views/Rankings";
import Briefing, { BRIEFING_CHAPTERS } from "./views/Briefing";
import Overall from "./views/Overall";
import Compare from "./views/Compare";
import Health from "./views/Health";
import Forecasts from "./views/Forecasts";
import MethodologyPage from "./views/MethodologyPage";

const navigation = [
  { id: "briefing", name: "Briefing", icon: LayoutDashboard },
  { id: "drivers", name: "Drivers", icon: Users },
  { id: "cars", name: "Cars", icon: Gauge },
  { id: "compare", name: "Head to head", icon: ArrowLeftRight },
  { id: "forecasts", name: "Track record", icon: Target },
  { id: "methodology", name: "Methodology", icon: BookOpen },
  { id: "health", name: "Model health", icon: Activity },
] as const;
function getRoute(): { view: View; discipline: Discipline; chapter: string } {
  const [page, metric] = window.location.hash.slice(1).split("/");
  if (page === "race") return { view: "drivers", discipline: "race", chapter: "qualifying" };
  const view = navigation.some((n) => n.id === page)
    ? (page as View)
    : "briefing";
  return {
    view,
    chapter: BRIEFING_CHAPTERS.some((c) => c.id === metric) ? metric : "qualifying",
    discipline:
      (view === "drivers" || view === "cars") && metric === "race"
        ? "race"
        : view === "drivers" && metric === "overall"
          ? "overall"
          : "qualifying",
  };
}

export default function App() {
  const [{ view, discipline, chapter }, setRoute] = useState(getRoute);
  const [data, setData] = useState<Dataset>();
  const [release, setRelease] = useState<Release>();
  const releaseRef = useRef("");
  const [run, setRun] = useState<RunStatus>();
  const [error, setError] = useState("");
  const [checking, setChecking] = useState(false);
  const [checked, setChecked] = useState("");
  const busy = useRef(false);
  const mounted = useRef(true);
  const refresh = useCallback(async () => {
    if (busy.current) return;
    busy.current = true;
    setChecking(true);
    try {
      const get = async (p: string) => {
        const r = await fetch(`${import.meta.env.BASE_URL}data/${p}`, {
          cache: "no-store",
          signal: AbortSignal.timeout(15000),
        });
        if (!r.ok)
          throw new Error("Published data is unavailable. Try again shortly.");
        return r.json();
      };
      const pointer: Release & { schema_version: number } =
        await get("latest.json");
      if (
        pointer.schema_version !== 1 ||
        !/^[a-f0-9]{20}$/.test(pointer.release) ||
        pointer.url !== `releases/${pointer.release}.json`
      )
        throw new Error("This dataset format is not supported.");
      if (pointer.release !== releaseRef.current) {
        const next: Dataset = await get(pointer.url);
        if (
          next.schema_version !== 1 ||
          !next.drivers?.length ||
          !next.cars?.length ||
          !next.meta?.diagnostics?.converged ||
          !next.history ||
          !next.comparisons
        )
          throw new Error(
            "The dataset is incomplete. The previous release is still shown.",
          );
        const race = next.race_pace;
        if (
          race &&
          (race.model !== "total-dry-race-pace-v1" ||
            !race.diagnostics?.converged ||
            race.diagnostics.rhat_max >= 1.05 ||
            !Array.isArray(race.drivers) ||
            !Array.isArray(race.cars) ||
            !race.validation?.metrics?.drivers ||
            !race.validation?.metrics?.cars ||
            (race.drivers.length > 0 &&
              !race.validation.metrics.drivers.passed) ||
            (race.cars.length > 0 && !race.validation.metrics.cars.passed))
        )
          throw new Error(
            "The race dataset is incomplete. The previous release is still shown.",
          );
        if (mounted.current) {
          setData(next);
          releaseRef.current = pointer.release;
        }
      }
      const status = await get("status.json");
      if (mounted.current) {
        setRelease(pointer);
        setRun(status);
        setError("");
        setChecked(new Date().toISOString());
      }
    } catch (e) {
      if (mounted.current)
        setError(
          e instanceof Error ? e.message : "Unable to check for updates.",
        );
    } finally {
      busy.current = false;
      if (mounted.current) setChecking(false);
    }
  }, []);
  useEffect(() => {
    mounted.current = true;
    refresh();
    const timer = setInterval(refresh, 30000);
    const nav = (event?: HashChangeEvent) => {
      if (window.location.hash === "#main") return;
      const route = getRoute();
      if (window.location.hash === "#race") {
        window.history.replaceState(null, "", "#drivers/race");
      }
      setRoute(route);
      if (event) requestAnimationFrame(() => {
        document.getElementById("main")?.focus({ preventScroll: true });
        window.scrollTo({ top: 0 });
      });
    };
    nav();
    window.addEventListener("hashchange", nav);
    return () => {
      mounted.current = false;
      clearInterval(timer);
      window.removeEventListener("hashchange", nav);
    };
  }, [refresh]);
  const [metric, setMetric] = useState<Metric>("headline");
  const returnChapter = view === "drivers" && discipline === "overall" ? "overall"
    : (view === "drivers" || view === "cars") && discipline === "race" ? "race"
    : view === "cars" || (view === "compare" && chapter === "cars") ? "cars"
    : view === "forecasts" || view === "health" || view === "methodology" ? "evidence" : "qualifying";
  return (
    <div className="shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="site-header">
        <a href="#briefing" className="site-brand" aria-label="F1 Analysis — briefing">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span><strong>F1 <span>ANALYSIS</span></strong><small>Beyond the finishing order</small></span>
        </a>
        <nav className="site-navigation" aria-label="Main navigation">
          {navigation.map((n) => (
            <a
              href={`#${n.id}${n.id === "drivers" && discipline === "overall" ? "/overall" : (n.id === "drivers" || n.id === "cars") && discipline === "race" ? "/race" : ""}`}
              key={n.id}
              className={view === n.id ? "active" : ""}
              aria-current={view === n.id ? "page" : undefined}
            >
              <n.icon size={16} aria-hidden="true" />
              <span>{n.name}</span>
            </a>
          ))}
        </nav>
      </header>
      <div className="workspace">
        <header className="topbar">
          <span className="breadcrumb">
            F1 Analysis <ChevronRight size={14} />
            <strong>{navigation.find((n) => n.id === view)?.name}</strong>
          </span>
          <div className="top-actions">
            {data && (
              <span className="season-pill">
                {data.meta.data_as_of.event_id.slice(0, 4)} SEASON
              </span>
            )}
            <button
              className="icon-button"
              title="Check for new published results"
              aria-label="Check for updates"
              onClick={refresh}
              disabled={checking}
            >
              <RefreshCw size={17} className={checking ? "spin" : ""} />
            </button>
          </div>
        </header>
        <main id="main" tabIndex={-1}>
          {!data ? (
            <div className="loading panel">
              {error ? (
                <>
                  <TriangleAlert />
                  <h1>Waiting for a published dataset</h1>
                  <p>{error}</p>
                  <button onClick={refresh} className="button primary">
                    Try again
                  </button>
                </>
              ) : (
                <>
                  <LoaderCircle className="spin" />
                  <h1>Loading rankings</h1>
                </>
              )}
            </div>
          ) : (
            <>
              {(error || run?.state === "failed") && (
                <div className="notice warning" role="status">
                  <TriangleAlert size={18} />
                  <div>
                    <strong>
                      {error
                        ? "Update check unavailable"
                        : "The latest refresh failed"}
                    </strong>
                    <p>{error || "Showing the last published release."}</p>
                  </div>
                  <a href="#health">
                    View health <ArrowRight size={14} />
                  </a>
                </div>
              )}
              {run?.state === "running" && (
                <div className="notice" role="status">
                  <LoaderCircle size={17} className="spin" />
                  Refresh in progress: {run.stage}. Showing the last release.
                </div>
              )}
              {view !== "briefing" && (
                <div className="explore-context">
                  <a href={`#briefing/${returnChapter}`}><ArrowLeft size={15} aria-hidden="true" />Back to briefing</a>
                  <span>Explore the full results</span>
                </div>
              )}
              {view === "briefing" ? (
                <Briefing data={data} chapter={chapter} />
              ) : view === "drivers" && discipline === "overall" ? (
                <Overall result={data.overall} />
              ) : view === "drivers" || view === "cars" ? (
                <Rankings
                  key={view}
                  data={data}
                  car={view === "cars"}
                  discipline={discipline}
                  metric={metric}
                  setMetric={setMetric}
                />
              ) : view === "compare" ? (
                <Compare key={chapter} data={data} car={chapter === "cars"} metric={metric} setMetric={setMetric} />
              ) : view === "forecasts" ? (
                <Forecasts data={data} />
              ) : view === "methodology" ? (
                <MethodologyPage data={data} />
              ) : (
                <Health data={data} run={run} release={release} />
              )}
              <footer className="page-footer">
                <span>
                  Independent estimates. Not affiliated with Formula 1.
                </span>
                <span title={checked}>Update check every 30s</span>
              </footer>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
