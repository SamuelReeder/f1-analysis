import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import fs from "node:fs/promises";
import type { Dataset, RacePace } from "../src/types";
let current: Dataset;

test.beforeEach(async ({ page, request }) => {
  const pointer = await (await request.get("data/latest.json")).json();
  current = await (await request.get(`data/${pointer.url}`)).json();
  await page.goto("./");
  await expect(page).toHaveTitle("F1 Analysis");
  await expect(
    page.getByRole("heading", { name: "Driver rankings" }),
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
  await page.getByLabel("Filter by team").selectOption("Ferrari");
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
  await page.getByRole("link", { name: "Car rankings", exact: true }).click();
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
  await expect(page.locator(".archive-top>div")).toHaveCount(5);
  await expect(
    page.getByRole("heading", { name: "Methodology", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("A time-varying Bayesian model estimates"),
  ).toBeVisible();
  await page.getByText("Data and lap filtering", { exact: true }).click();
  await expect(
    page.getByText("pace = −100 × ln(time / segment median)"),
  ).toBeVisible();
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
    page.getByRole("heading", { name: "Driver rankings" }),
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

test("race pace has independent driver and car tables, search, CSV and accessible detail", async ({
  page,
}) => {
  const fixture = raceFixture();
  await page.route("**/data/releases/*.json", (route) =>
    route.fulfill({ json: { ...current, race_pace: fixture } }),
  );
  await page.goto("./#race");
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Driver race pace", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".rank-table tbody tr")).toHaveCount(2);
  await page.getByLabel("Search race rankings").fill("Driver B");
  await expect(page.locator(".rank-table tbody tr")).toHaveCount(1);
  await page
    .getByRole("button", { name: "Inspect race pace for Test Driver B" })
    .click();
  await expect(
    page
      .locator(".race-detail")
      .getByRole("heading", { name: "Test Driver B" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Cars", exact: true }).click();
  await expect(page.locator(".rank-table tbody tr")).toHaveCount(2);
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export race rankings" }).click();
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
  await page.reload();
  await expect(page.locator(".rank-table tbody tr")).toHaveCount(2);
  await page.getByRole("button", { name: "Cars", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Car race-pace ranking withheld" }),
  ).toBeVisible();
  await expect(page.locator(".rank-table")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Export race rankings" }),
  ).toHaveCount(0);
});

for (const view of ["drivers", "cars", "race", "compare", "health"]) {
  test(`${view} has no browser errors, no mobile overflow, and accessible controls`, async ({
    page,
  }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`./#${view}`);
    await expect(page.locator(".page-heading")).toBeVisible();
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
      path: `test-results/${view}-mobile.png`,
      fullPage: true,
    });
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.screenshot({
      path: `test-results/${view}-desktop.png`,
      fullPage: true,
    });
    expect(errors).toEqual([]);
  });
}
