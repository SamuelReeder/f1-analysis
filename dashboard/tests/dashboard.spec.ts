import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import fs from "node:fs/promises";
import type { Dataset, RacePace, OverallResult } from "../src/types";
let current: Dataset;

test.beforeEach(async ({ page, request }) => {
  const pointer = await (await request.get("data/latest.json")).json();
  current = await (await request.get(`data/${pointer.url}`)).json();
  await page.goto("./");
  await expect(page).toHaveTitle("F1 Analysis");
  await expect(
    page.getByRole("heading", { name: "Drivers", level: 1 }),
  ).toBeVisible();
});

test("rankings, search, team filter, driver details and metric-specific CSV", async ({
  page,
}) => {
  await expect(page.locator(".rank-table tbody tr")).toHaveCount(
    current.drivers.length,
  );
  await page.getByText("Metric definitions", { exact: true }).click();
  await expect(
    page.getByText("Driver qualifying pace including a persistent"),
  ).toBeVisible();
  await page.getByLabel("Search drivers").fill("norris");
  await expect(page.locator(".rank-table tbody tr")).toHaveCount(1);
  await page.getByRole("button", { name: "Inspect Lando Norris" }).click();
  await expect(
    page
      .getByRole("region", { name: "Selected entry" })
      .getByRole("heading", { name: "Lando Norris" }),
  ).toBeVisible();
  await page.getByLabel("Clear search").click();
  await page.getByLabel("Filter by team").selectOption({ label: "Ferrari" });
  await expect(page.locator(".rank-table tbody tr")).toHaveCount(2);
  await page
    .getByRole("button", { name: "Portable skill Experimental" })
    .click();
  await expect(
    page.getByText("Experimental: sensitive to model assumptions"),
  ).toBeVisible();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export rankings" }).click();
  const download = await downloadPromise;
  const csv = await fs.readFile((await download.path())!, "utf8");
  expect(download.suggestedFilename()).toContain("portable");
  expect(csv).toContain('"portable"');
  expect(csv.trim().split("\n")).toHaveLength(current.drivers.length + 1); // full ranking, irrespective of search filter
});

test("season trends support absent entrants and car circuit adjustment", async ({
  page,
}) => {
  await page.getByLabel("Trend season").selectOption("2010");
  await expect(
    page.getByRole("heading", { name: "No estimates in this season" }),
  ).toBeVisible();
  await page.getByLabel("First trend entry").selectOption("vettel");
  await expect(page.locator(".trend-chart")).toBeVisible();
  await page.getByLabel("Trend season").selectOption("2026");
  await expect(page.locator(".trend-chart")).toBeVisible();
  await page.getByRole("link", { name: "Cars", exact: true }).click();
  await expect(page.locator(".rank-table tbody tr")).toHaveCount(
    current.cars.length,
  );
  const before = await page
    .locator(".trend-chart path")
    .first()
    .getAttribute("d");
  await page.getByLabel("Circuit-adjusted").check();
  const after = await page
    .locator(".trend-chart path")
    .first()
    .getAttribute("d");
  expect(after).not.toBe(before);
  await expect(
    page.getByRole("heading", { name: "Pace history" }),
  ).toBeVisible();
});

test("driver charts split car and driver, and history can show estimates after each race", async ({
  page,
}) => {
  const top = current.breakdown![0];
  const row = page.getByRole("button", {
    name: new RegExp(
      `^${current.drivers.find((d) => d.id === top.id)!.name}: car `,
    ),
  });
  await row.focus();
  await expect(
    page
      .locator("#breakdown-title")
      .locator("xpath=../../..")
      .locator(".chart-readout"),
  ).toContainText("total");
  await page.getByText("Show as table").first().click();
  await expect(
    page.locator(".table-view").first().locator("tbody tr"),
  ).toHaveCount(current.breakdown!.length);
  const before = await page
    .locator(".trend-chart path")
    .first()
    .getAttribute("d");
  await page.getByRole("button", { name: "After each race" }).click();
  await expect(page.getByText("Estimates after each race")).toBeVisible();
  const after = await page
    .locator(".trend-chart path")
    .first()
    .getAttribute("d");
  expect(after).not.toBe(before);
  await expect(
    page.getByLabel("Trend season").locator("option"),
  ).not.toContainText(["All seasons"]);
});

test("revised history can be shown as rank in each event's field", async ({
  page,
}) => {
  const scale = page.getByLabel("Chart scale");
  await scale.getByRole("button", { name: "Rank" }).click();
  await expect(scale.getByRole("button", { name: "Rank" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  const panel = page.locator(".trend-panel").filter({ has: scale });
  await expect(panel.getByText("RANK IN THE FIELD")).toBeVisible();
  const point = panel.locator("rect[role='button']").last();
  await expect(point).toHaveAttribute("aria-label", /P\d+ \(P\d+–P\d+\)/);
  await point.focus();
  await expect(panel.locator(".chart-readout")).toContainText(
    /P\d+ \(P\d+–P\d+\)/,
  );
  await page.getByRole("button", { name: "After each race" }).click();
  await expect(page.getByLabel("Chart scale")).toHaveCount(0);
});

test("track record shows the forecast published before the next race", async ({
  page,
}) => {
  const upcoming = current.forecast?.next;
  test.skip(!upcoming, "No forecast for an upcoming race in this release");
  await page.getByRole("link", { name: "Track record", exact: true }).click();
  const panel = page.locator("section").filter({
    has: page.getByRole("heading", {
      name: `Next race · ${upcoming!.event.race_name}`,
    }),
  });
  await expect(panel).toBeVisible();
  await expect(
    panel.getByRole("region", { name: /teammate gaps/ }).locator("tbody tr"),
  ).toHaveCount(upcoming!.pairs.length);
  const cars = panel
    .getByRole("region", { name: /car order/ })
    .locator("tbody tr");
  await expect(cars).toHaveCount(upcoming!.cars.length);
  await expect(cars.first()).toContainText(upcoming!.cars[0].name);
});

test("track record reports pooled forecast checks and the latest forecast", async ({
  page,
}) => {
  await page.getByRole("link", { name: "Track record", exact: true }).click();
  const pooled = current.asof!.pooled!;
  const card = page.locator(".stat-card").filter({
    hasText: "Teammate gap error",
  });
  await expect(card).toContainText(`${pooled.rmse.toFixed(3)}s`);
  // each baseline comparison is stated only as far as its interval allows
  await expect(card).toContainText(
    `${pooled.rmse_zero.toFixed(3)}s for no gap`,
  );
  const verdict = (d?: { ci95: [number, number] }) =>
    !d
      ? null
      : d.ci95[1] < 0
        ? "(model better)"
        : d.ci95[0] > 0
          ? "(model worse)"
          : "(no clear difference)";
  for (const [d, label] of [
    [pooled.vs_naive, "for last season’s gap"],
    [pooled.vs_zero, "for no gap"],
  ] as const) {
    const v = verdict(d);
    if (v) await expect(card).toContainText(`${label} ${v}`);
  }
  await expect(
    page.locator(".chart-legend").filter({ hasText: "No gap" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: `Latest forecast · ${current.asof!.latest!.event.race_name}`,
    }),
  ).toBeVisible();
  await page.getByText("Show as table").click();
  await expect(page.locator(".table-view tbody tr")).toHaveCount(
    current.asof!.events.filter((e) => e.n_pairs > 0).length,
  );
});

test("comparisons reverse correctly and matrix cells select the actual pair", async ({
  page,
}) => {
  await page.getByRole("link", { name: "Head to head", exact: true }).click();
  const before = parseFloat(
    (await page.locator(".matchup-prob").innerText()).replace("%", ""),
  );
  await page.getByRole("button", { name: "Swap comparison" }).click();
  const after = parseFloat(
    (await page.locator(".matchup-prob").innerText()).replace("%", ""),
  );
  expect(before + after).toBeCloseTo(100, 1);
  await page
    .getByRole("button", { name: /^Lando Norris ahead of Charles Leclerc:/ })
    .click();
  await expect(page.getByLabel("FIRST DRIVER")).toHaveValue("norris");
  await expect(page.getByLabel("SECOND DRIVER")).toHaveValue("leclerc");
  await expect(page.locator(".matrix button.teammate")).toHaveCount(
    current.drivers.flatMap((d) =>
      current.drivers.filter((o) => o.id !== d.id && o.lineage === d.lineage),
    ).length,
  );
  await page.getByRole("button", { name: "Cars", exact: true }).click();
  await expect(page.getByLabel("FIRST CAR")).toHaveValue(current.cars[0].id);
  await expect(page.locator(".matrix tbody tr")).toHaveCount(
    current.cars.length,
  );
  await page.getByText("Calculation details", { exact: true }).click();
  await expect(
    page.getByText("Gap intervals:", { exact: false }),
  ).toBeVisible();
});

test("health discloses unavailable racing, historical gates and snapshots", async ({
  page,
}) => {
  await page.getByRole("link", { name: "Model health", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Qualifying publication checks passed" }),
  ).toBeVisible();
  await expect(
    page.getByText("Needs regeneration", { exact: true }),
  ).toHaveCount(current.racing.filter((r) => r.status === "stale").length);
  await expect(page.getByText("Recorded fail", { exact: true })).toHaveCount(
    Object.values(current.validation.gates?.data || {}).filter((v) => !v)
      .length,
  );
  await expect(page.locator(".archive-top>div").first()).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Updates", exact: true }),
  ).toBeVisible();
  // Research models are grouped and collapsed below the published ones.
  await expect(page.getByText("Status details").first()).toBeHidden();
  await page.getByText(/^In research \(\d+\)/).click();
  await expect(page.getByText("Status details").first()).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Methodology", exact: true }),
  ).toHaveCount(0);
});

test("methodology has its own page with a reading guide and equations", async ({
  page,
}) => {
  await page.getByRole("link", { name: "Methodology", exact: true }).click();
  await expect(page).toHaveURL(/#methodology$/);
  await expect(
    page.getByRole("heading", { name: "How to read the ratings" }),
  ).toBeVisible();
  await expect(
    page.getByText("A time-varying Bayesian model estimates"),
  ).toBeVisible();
  await page.getByText("Data and lap filtering", { exact: true }).click();
  await expect(
    page.getByRole("region", {
      name: "Qualifying pace transformation",
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.locator(".method-equation math").first()).toBeAttached();
  await page.getByText("What each rating measures", { exact: true }).click();
  await expect(
    page.getByText("The driver component after removing"),
  ).toBeVisible();
  await expect(
    page.getByText("This dataset format is not supported."),
  ).toHaveCount(0);
});

test("failed update keeps the previous good data visible", async ({ page }) => {
  await page.route("**/data/latest.json", (route) =>
    route.fulfill({ status: 503, body: "unavailable" }),
  );
  await page.getByLabel("Check for updates").click();
  await expect(
    page.getByText("Update check unavailable", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".rank-table tbody tr")).toHaveCount(
    current.drivers.length,
  );
});

test("publisher failure is surfaced without hiding rankings", async ({
  page,
}) => {
  await page.route("**/data/status.json", (route) =>
    route.fulfill({
      json: {
        state: "failed",
        stage: "export",
        error: "Fit did not converge",
        started_at: "2026-09-30T00:00:00Z",
        finished_at: "2026-09-30T01:00:00Z",
      },
    }),
  );
  await page.getByLabel("Check for updates").click();
  await expect(
    page.getByText("The latest refresh failed", { exact: true }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Model health", exact: true }).click();
  await expect(
    page.getByText("Fit did not converge", { exact: true }),
  ).toBeVisible();
});

test("a new release is adopted by the refresh check", async ({
  page,
  request,
}) => {
  const pointer = await (await request.get("data/latest.json")).json();
  const data = await (await request.get(`data/${pointer.url}`)).json();
  const nextId = "1234567890abcdef1234";
  data.drivers[0].name = "Updated driver estimate";
  await page.route("**/data/latest.json", (route) =>
    route.fulfill({
      json: { ...pointer, release: nextId, url: `releases/${nextId}.json` },
    }),
  );
  await page.route(`**/data/releases/${nextId}.json`, (route) =>
    route.fulfill({ json: data }),
  );
  await page.getByLabel("Check for updates").click();
  await expect(
    page.getByRole("button", { name: "Inspect Updated driver estimate" }),
  ).toBeVisible();
});

test("initial unavailable dataset has a working retry", async ({ page }) => {
  await page.route("**/data/latest.json", (route) =>
    route.fulfill({ status: 404, body: "{}" }),
  );
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Waiting for a published dataset" }),
  ).toBeVisible();
  await page.unroute("**/data/latest.json");
  await page.getByRole("button", { name: "Try again" }).click();
  await expect(
    page.getByRole("heading", { name: "Drivers", level: 1 }),
  ).toBeVisible();
});

function raceFixture(): RacePace {
  const gate = {
    passed: true,
    improves: true,
    calibrated: true,
    sharper: true,
    n_races: 24,
    n_predictions: 200,
    rmse: 0.2,
    baseline_rmse: 0.3,
    mse_difference: -0.05,
    mse_difference_ci95: [-0.08, -0.02],
    coverage90: 0.9,
    mean_interval_width: 0.4,
    baseline_interval_width: 0.6,
  };
  const row = (id: string, name: string, median: number) => ({
    id,
    name,
    pace: {
      q05: median - 0.1,
      median,
      q95: median + 0.1,
      rank_lo: 1,
      rank_hi: 2,
      p_fastest: 0.6,
    },
    races: 10,
    laps: 300,
    last_race: "2026-15",
  });
  return {
    model: "total-dry-race-pace-v1",
    fit_id: "test-fixture",
    season: 2026,
    data_as_of: current.events[current.events.length - 1],
    first_event: "2018-01",
    n_laps: 100000,
    n_races: 130,
    diagnostics: current.meta.diagnostics,
    drivers: [
      row("test_a", "Test Driver A", 0.1),
      row("test_b", "Test Driver B", -0.1),
    ],
    cars: [
      row("test_car_a", "Test Car A", 0.2),
      row("test_car_b", "Test Car B", -0.2),
    ],
    unrated_drivers: [],
    unrated_cars: [],
    validation: {
      folds: ["2024-10", "2025-10", "2026-07"],
      design: "Test fixture",
      metrics: { drivers: gate, cars: gate },
    },
  };
}

test("ranking metrics support browser history, direct links and switching entity", async ({
  page,
}) => {
  const mainNav = page.getByRole("navigation", { name: "Main navigation" });
  await expect(mainNav.getByRole("link")).toHaveCount(6);
  await expect(mainNav.getByRole("link", { name: "Race pace" })).toHaveCount(0);
  await page
    .getByRole("button", { name: "Portable skill Experimental" })
    .click();
  await page.getByRole("link", { name: "Race pace", exact: true }).click();
  await expect(page).toHaveURL(/#drivers\/race$/);
  await expect(
    mainNav.getByRole("link", { name: "Drivers", exact: true }),
  ).toHaveAttribute("aria-current", "page");
  await expect(
    page.getByRole("heading", { name: "Drivers", level: 1 }),
  ).toBeVisible();
  await page.goBack();
  await expect(
    page.getByRole("link", { name: "Qualifying pace", exact: true }),
  ).toHaveAttribute("aria-current", "page");
  await expect(
    page.getByRole("button", { name: "Portable skill Experimental" }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.goForward();
  await expect(
    page.getByRole("link", { name: "Race pace", exact: true }),
  ).toHaveAttribute("aria-current", "page");
  await mainNav.getByRole("link", { name: "Cars", exact: true }).click();
  await expect(page).toHaveURL(/#cars\/race$/);
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Cars", level: 1 }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Race pace", exact: true }),
  ).toHaveAttribute("aria-current", "page");
  await page
    .getByRole("link", { name: "Qualifying pace", exact: true })
    .click();
  await expect(page).toHaveURL(/#cars$/);
  await expect(page.locator(".rank-table tbody tr")).toHaveCount(
    current.cars.length,
  );
});

for (const width of [1440, 390]) {
  test(`switching pace at ${width}px keeps the selection, filters and layout`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 1000 });
    const fixture = raceFixture();
    const driver = current.drivers.find((d) => d.id === "norris")!;
    fixture.drivers = fixture.drivers.map((r, i) => {
      const d = i ? driver : current.drivers[0];
      return {
        ...r,
        id: d.id,
        name: d.name,
        code: d.code,
        lineage: d.lineage,
        team: d.team,
      };
    });
    await page.route("**/data/releases/*.json", (route) =>
      route.fulfill({ json: { ...current, race_pace: fixture } }),
    );
    await page.reload();
    await page
      .getByLabel("Filter by team")
      .selectOption({ label: driver.team });
    await page.getByLabel("Search drivers").fill("norris");
    await page.getByRole("button", { name: `Inspect ${driver.name}` }).click();
    await page.evaluate(() => document.fonts.ready);
    const positions = () =>
      page.evaluate(() =>
        [
          ".event-line",
          ".ranking-panel",
          ".table-tools",
          ".rank-table thead",
          ".detail-card",
          ".detail-pace",
        ].map((selector) => {
          const rect = document
            .querySelector(selector)!
            .getBoundingClientRect();
          return {
            selector,
            x: rect.x,
            y: rect.y + window.scrollY,
            width: rect.width,
          };
        }),
      );
    const before = await positions();
    const qualifyingPace = await page.locator(".detail-pace").innerText();
    await page.getByRole("link", { name: "Race pace", exact: true }).click();
    await expect(
      page.getByRole("heading", { name: "Race pace", exact: true }),
    ).toBeVisible();
    await expect(page.getByLabel("Search drivers")).toHaveValue("norris");
    await expect(page.getByLabel("Filter by team")).toHaveValue(driver.lineage);
    await expect(
      page
        .getByRole("region", { name: "Selected entry" })
        .getByRole("heading", { name: driver.name }),
    ).toBeVisible();
    await expect(page.locator(".rank-table tbody tr")).toHaveCount(1);
    expect(await page.locator(".detail-pace").innerText()).not.toBe(
      qualifyingPace,
    );
    const after = await positions();
    before.forEach((rect, i) => {
      for (const key of ["x", "y", "width"] as const) {
        expect(
          Math.abs(rect[key] - after[i][key]),
          `${rect.selector}: ${key}`,
        ).toBeLessThan(1);
      }
    });
    await page
      .getByRole("link", { name: "Qualifying pace", exact: true })
      .click();
    await expect(page.locator(".detail-pace")).toHaveText(qualifyingPace);
    await expect(page.getByLabel("Search drivers")).toHaveValue("norris");
    await page.getByLabel("Clear search").click();
    await expect(page.locator(".rank-table tbody tr")).toHaveCount(2);
  });
}

test("race rankings follow their entity pages, with search, CSV and accessible detail", async ({
  page,
}) => {
  const fixture = raceFixture();
  await page.route("**/data/releases/*.json", (route) =>
    route.fulfill({ json: { ...current, race_pace: fixture } }),
  );
  await page.goto("./#race");
  await expect(page).toHaveURL(/#drivers\/race$/);
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Race pace", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".rank-table tbody tr")).toHaveCount(2);
  await page.getByLabel("Search drivers").fill("Driver B");
  await expect(page.locator(".rank-table tbody tr")).toHaveCount(1);
  await page.getByRole("button", { name: "Inspect Test Driver B" }).click();
  await expect(
    page
      .getByRole("region", { name: "Selected entry" })
      .getByRole("heading", { name: "Test Driver B" }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Cars", exact: true }).click();
  await expect(page.locator(".rank-table tbody tr")).toHaveCount(2);
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export rankings" }).click();
  const file = await pending;
  const csv = await fs.readFile((await file.path())!, "utf8");
  expect(csv).toContain("total_race_pace_cars");
  expect(csv).toContain('"test-fixture"');
  expect(csv.trim().split("\n")).toHaveLength(3);
  await page.getByText("Race-pace methodology", { exact: true }).click();
  await expect(
    page.getByText("It is estimated independently of qualifying.", {
      exact: false,
    }),
  ).toBeVisible();
  const desktop = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(desktop.violations).toEqual([]);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  const mobile = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(mobile.violations).toEqual([]);
});

test("a withheld car race ranking does not hide a supported driver table", async ({
  page,
}) => {
  const fixture = raceFixture();
  fixture.cars = [];
  fixture.validation.metrics.cars = {
    ...fixture.validation.metrics.cars,
    passed: false,
    improves: false,
  };
  await page.route("**/data/releases/*.json", (route) =>
    route.fulfill({ json: { ...current, race_pace: fixture } }),
  );
  await page.goto("./#race");
  await expect(page).toHaveURL(/#drivers\/race$/);
  await page.reload();
  await expect(page.locator(".rank-table tbody tr")).toHaveCount(2);
  await page.getByRole("link", { name: "Cars", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Car race-pace ranking withheld" }),
  ).toBeVisible();
  await expect(page.locator(".rank-table")).toHaveCount(0);
  if (current.car_state)
    await expect(
      page.getByRole("heading", {
        name: "Follow-up test: in-season car development",
      }),
    ).toBeVisible();
  // every recorded weekend-information test has a team row on the car view
  const features = current.race_features ?? [];
  if (features.length)
    await expect(
      page
        .locator("section")
        .filter({
          has: page.getByRole("heading", {
            name: "Tests of weekend information",
          }),
        })
        .locator("tbody tr"),
    ).toHaveCount(features.length);
  await expect(
    page.getByRole("button", { name: "Export rankings" }),
  ).toBeDisabled();
  await page.getByRole("link", { name: "Model health", exact: true }).click();
  await expect(
    page.getByText("Driver race pace published", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Racing validation pending", { exact: true }),
  ).toHaveCount(0);
});

test("an unsupported race ranking cannot replace the previous release", async ({
  page,
}) => {
  const race = raceFixture();
  race.validation.metrics.cars = {
    ...race.validation.metrics.cars,
    passed: false,
  };
  const id = "abcdef1234567890abcd";
  await page.route("**/data/latest.json", (route) =>
    route.fulfill({
      json: {
        schema_version: 1,
        release: id,
        url: `releases/${id}.json`,
        published_at: "2026-09-30T00:00:00Z",
      },
    }),
  );
  await page.route(`**/data/releases/${id}.json`, (route) =>
    route.fulfill({ json: { ...current, race_pace: race } }),
  );
  await page.getByLabel("Check for updates").click();
  await expect(
    page.getByText(
      "The race dataset is incomplete. The previous release is still shown.",
    ),
  ).toBeVisible();
  await expect(page.locator(".rank-table tbody tr")).toHaveCount(
    current.drivers.length,
  );
});

for (const view of [
  "drivers",
  "drivers/race",
  "drivers/overall",
  "cars",
  "cars/race",
  "compare",
  "forecasts",
  "methodology",
  "health",
]) {
  test(`${view} has no browser errors, no mobile overflow, and accessible controls`, async ({
    page,
  }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`./#${view}`);
    await expect(page.locator(".page-heading")).toBeVisible();
    // Audit the expanded equations as well as the ranking controls.
    await page
      .locator(".methodology details, .race-method details")
      .evaluateAll((details) =>
        details.forEach((detail) => detail.setAttribute("open", "")),
      );
    if (view === "methodology" || view.endsWith("/race")) {
      await expect(
        page.locator(".method-equation math").first(),
      ).toBeAttached();
    }
    const audit = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(
      audit.violations.map((v) => ({
        id: v.id,
        targets: v.nodes.map((n) => n.target),
      })),
    ).toEqual([]);
    await page.setViewportSize({ width: 390, height: 844 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
    const mobileAudit = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(
      mobileAudit.violations.map((v) => ({
        id: v.id,
        targets: v.nodes.map((n) => n.target),
      })),
    ).toEqual([]);
    await page.screenshot({
      path: `test-results/${view.replace("/", "-")}-mobile.png`,
      fullPage: true,
    });
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.screenshot({
      path: `test-results/${view.replace("/", "-")}-desktop.png`,
      fullPage: true,
    });
    expect(errors).toEqual([]);
  });
}

// Entirely synthetic equal-car outcomes; historical championship files are never fixtures.
function overallFixture(passed = true): OverallResult {
  return {
    status: passed ? "established" : "not established",
    reason: passed
      ? "Combined held-out validation passed; this equal-car scenario remains experimental."
      : "The combined held-out validation has not established an overall ranking.",
    standings: passed ? [
      { driver_id: "alpha", name: "Fixture Driver Alpha", points_per_race: 8.25, p_title: .75, rank_median: 1, rank_lo: 1, rank_hi: 2 },
      { driver_id: "beta", name: "Fixture Driver Beta", points_per_race: 6.5, p_title: .25, rank_median: 2, rank_lo: 1, rank_hi: 2 },
    ] : [],
    contributions: passed ? [
      { driver_id: "alpha", name: "Fixture Driver Alpha", points_per_race: 8.2,
        losses: { qualifying_pace: .35, first_lap: .2, all: .45 } },
      { driver_id: "beta", name: "Fixture Driver Beta", points_per_race: 6.4,
        losses: { qualifying_pace: -.2, first_lap: .05, all: -.1 } },
    ] : [],
    evidence: {
      recorded_at: "2004-01-01T00:00:00Z", fit_id: "synthetic-overall-fixture", data_as_of: "2003-05",
      qualities_entered: passed ? ["first_lap"] : [], qualities_selected: ["first_lap"],
      qualities_not_entered: passed ? ["consistency"] : ["first_lap", "consistency"],
      quality_decisions: [
        { quality: "first_lap", entered: passed, tested: true, reason: passed
          ? "Passed the conditional entry test and the combined held-out validation."
          : "Selected by the conditional entry test, but the combined held-out validation did not pass." },
        { quality: "consistency", entered: false, tested: true,
          reason: "Did not establish an improvement when tested alongside the other candidate qualities." },
        { quality: "overtaking_attack", entered: false, tested: false, reason: "Not tested: no held-out draws" },
      ],
      entry_tests: { first_lap: { enters: true }, consistency: { enters: false }, overtaking_attack: { not_run: "no held-out draws" } },
      excluded_test_seasons: { combined_validation: [2001], heldout_race_stage: [2001, 2002] },
      combined_validation: { gate: passed }, heldout_race_stage: {},
      simulated_seasons: 40, n_races_simulated: 5, driver_error_rates_used: false,
    },
  };
}

for (const passed of [true, false]) {
  test(`equal-car ${passed ? "passed" : "not-established"} fixture is accessible and fits phone widths`, async ({ page }) => {
    const fixture = overallFixture(passed);
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.route("**/data/releases/*.json", (route) => route.fulfill({ json: { ...current, overall: fixture } }));
    await page.goto("./#drivers/overall");
    await page.reload();
    const link = page.getByRole("link", { name: "Overall (equal car)", exact: true });
    await expect(link).toHaveAttribute("aria-current", "page");
    await expect(page.getByText("Experimental · equal car", { exact: true })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Which racing qualities entered?" })).toBeVisible();
    await expect(page.getByText("Consistency · Not entered", { exact: true })).toBeVisible();
    await expect(page.getByText("Not tested: no held-out draws", { exact: true })).toBeVisible();
    await expect(page.getByText("2001, 2002", { exact: true })).toBeVisible();
    await expect(page.getByText("2001", { exact: true })).toBeVisible();
    const table = page.getByRole("table", { name: "Equal-car championship standings" });
    if (passed) {
      await expect(table.locator("tbody tr")).toHaveCount(fixture.standings.length);
      await expect(table.locator("tbody tr").first()).toContainText(["Fixture Driver Alpha", "8.25", "75.0%", "1–2"].join(""));
      const panel = page.getByRole("region", { name: "Contribution breakdown" });
      await expect(panel.getByRole("heading", { name: "Fixture Driver Alpha" })).toBeVisible();
      await expect(panel.getByText("+0.35", { exact: true })).toBeVisible();
      await page.getByRole("button", { name: "Inspect Fixture Driver Beta contributions" }).click();
      await expect(panel.getByRole("heading", { name: "Fixture Driver Beta" })).toBeVisible();
      await expect(panel.getByText("−0.20", { exact: true })).toBeVisible();
      await expect(page.getByText("40 simulated seasons · 5 races per season.", { exact: false })).toBeVisible();
    } else {
      await expect(page.getByRole("heading", { name: "Overall ranking not established" })).toBeVisible();
      await expect(page.getByText(fixture.reason, { exact: true })).toBeVisible();
      await expect(page.getByText(/Selected by the conditional entry test, but/)).toBeVisible();
      await expect(page.getByRole("table")).toHaveCount(0);
      await expect(page.getByRole("region", { name: "Contribution breakdown" })).toHaveCount(0);
    }
    for (const width of [1440, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      const layout = await page.evaluate(() => ({
        width: window.innerWidth,
        pageWidth: document.documentElement.scrollWidth,
        outside: [...document.querySelectorAll("body *")].filter((element) =>
          element.getBoundingClientRect().right > window.innerWidth).map((element) =>
          ({ tag: element.tagName, className: element.className, right: element.getBoundingClientRect().right })),
      }));
      expect(layout.pageWidth, JSON.stringify(layout)).toBeLessThanOrEqual(width);
      const audit = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
      expect(audit.violations.map((v) => ({ id: v.id, targets: v.nodes.map((n) => n.target) }))).toEqual([]);
      if (passed) await expect(table.getByRole("columnheader", { name: "Rank range" })).toBeVisible();
    }
    await page.screenshot({ path: `test-results/overall-${passed ? "passed" : "withheld"}-mobile.png`, fullPage: true });
    expect(errors).toEqual([]);
  });
}

test("equal-car direct links and browser history stay under Drivers", async ({ page }) => {
  await page.getByRole("link", { name: "Overall (equal car)", exact: true }).click();
  await expect(page).toHaveURL(/#drivers\/overall$/);
  await expect(page.getByRole("navigation", { name: "Main navigation" }).getByRole("link", { name: "Drivers", exact: true }))
    .toHaveAttribute("aria-current", "page");
  await page.reload();
  await expect(page.getByRole("link", { name: "Overall (equal car)", exact: true })).toHaveAttribute("aria-current", "page");
  await page.getByRole("link", { name: "Race pace", exact: true }).click();
  await expect(page).toHaveURL(/#drivers\/race$/);
  await page.goBack();
  await expect(page.getByRole("link", { name: "Overall (equal car)", exact: true })).toHaveAttribute("aria-current", "page");
  await page.getByRole("navigation", { name: "Main navigation" }).getByRole("link", { name: "Cars", exact: true }).click();
  await expect(page.getByRole("link", { name: "Overall (equal car)", exact: true })).toHaveCount(0);
});

for (const state of ["stale", "unavailable", "legacy"] as const) {
  test(`equal-car ${state} release never falls back to a ranking`, async ({ page }) => {
    const fixture = { ...overallFixture(), status: state === "legacy" ? "unavailable" : state, evidence: null };
    await page.route("**/data/releases/*.json", (route) => route.fulfill({ json: { ...current, overall: state === "legacy" ? undefined : fixture } }));
    await page.goto("./#drivers/overall");
    await page.reload();
    await expect(page.getByRole("heading", { name: `Overall ranking ${state === "legacy" ? "unavailable" : state}` })).toBeVisible();
    await expect(page.getByRole("table")).toHaveCount(0);
    await expect(page.getByRole("region", { name: "Contribution breakdown" })).toHaveCount(0);
    await expect(page.getByRole("heading", { name: "Which racing qualities entered?" })).toHaveCount(0);
  });
}

test("a failed overall gate hides even accidentally supplied standings", async ({ page }) => {
  const fixture = overallFixture();
  fixture.evidence!.combined_validation.gate = false;
  await page.route("**/data/releases/*.json", (route) => route.fulfill({ json: { ...current, overall: fixture } }));
  await page.goto("./#drivers/overall");
  await page.reload();
  await expect(page.getByRole("heading", { name: "Overall ranking not established" })).toBeVisible();
  await expect(page.getByRole("table")).toHaveCount(0);
  await expect(page.getByRole("region", { name: "Contribution breakdown" })).toHaveCount(0);
});
