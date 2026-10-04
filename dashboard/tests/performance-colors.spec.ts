import { test, expect, type Locator, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import type { Dataset, Estimate, OverallResult, RacePace } from "../src/types";

let current: Dataset;
const red = "rgb(248, 113, 113)";
const amber = "rgb(251, 191, 36)";
const green = "rgb(74, 222, 128)";
const samples = [
  { name: "Spectrum Center Wide", median: 0, width: 4, color: amber },
  { name: "Spectrum Tail", median: -1, width: 2.6, color: red },
  { name: "Spectrum Lead", median: 1, width: .1, color: green },
  { name: "Spectrum Quarter", median: -.5, width: .1, color: "rgb(250, 152, 75)" },
  { name: "Spectrum Center Narrow", median: 0, width: .02, color: amber },
];

// Published data supplies only the surrounding schema; every tested ranking,
// uncertainty range and validation outcome is synthetic.
test.beforeEach(async ({ request }) => {
  const pointer = await (await request.get("data/latest.json")).json();
  current = await (await request.get(`data/${pointer.url}`)).json();
});

function estimate(median: number, width: number): Estimate {
  return { median, q05: median - width, q95: median + width, rank_lo: 1, rank_hi: 5, p_fastest: .2, p_top3: .6 };
}

function fixture(): Dataset {
  const data = structuredClone(current);
  data.drivers = samples.map((sample, index) => ({
    id: `spectrum_${index}`, name: sample.name, code: `S${index}`, team: `Spectrum Team ${index}`, lineage: `spectrum_team_${index}`,
    has_time: true, headline: estimate(sample.median, sample.width), portable: estimate(sample.median, sample.width),
    team_effect: estimate(0, .05), evidence: { events: 5, seasons: 1, teammates: 1, teams: 1 },
  }));
  data.cars = samples.map((sample, index) => ({
    id: `spectrum_team_${index}`, name: `Car ${sample.name}`, pace: estimate(sample.median, sample.width), gap_to_best: estimate(1 - sample.median, .1),
  }));
  data.catalog = { drivers: data.drivers.map(({ id, name }) => ({ id, name })), cars: data.cars.map(({ id, name }) => ({ id, name })) };
  data.history = { drivers: {}, cars: {} };
  data.breakdown = [];
  const gate = {
    passed: true, improves: true, calibrated: true, sharper: true,
    n_races: 18, n_predictions: 160, rmse: .2, baseline_rmse: .3,
    mse_difference: -.05, mse_difference_ci95: [-.08, -.02], coverage90: .9,
    mean_interval_width: .4, baseline_interval_width: .6,
  };
  const race: RacePace = {
    model: "total-dry-race-pace-v1", fit_id: "synthetic-colors", season: 2003,
    data_as_of: { event_id: "2003-05", season: 2003, round: 5, date: "2003-05-04", race_name: "Fixture Grand Prix" },
    first_event: "2001-01", n_laps: 90000, n_races: 90, diagnostics: data.meta.diagnostics,
    drivers: data.drivers.map(driver => ({ ...driver, pace: driver.headline, races: 10, laps: 300, last_race: "2003-05" })),
    cars: data.cars.map(car => ({ ...car, races: 10, laps: 300, last_race: "2003-05" })),
    unrated_drivers: [], unrated_cars: [],
    validation: { folds: ["2001-10", "2002-10"], design: "Synthetic held-out test", metrics: { drivers: gate, cars: gate } },
  };
  const overall: OverallResult = {
    status: "established", reason: "Synthetic combined validation passed.",
    standings: data.drivers.map((driver, index) => ({
      driver_id: driver.id, name: driver.name, points_per_race: samples[index].median + 2,
      p_title: .2, rank_median: 3, rank_lo: 1, rank_hi: 5,
    })),
    contributions: [],
    evidence: {
      recorded_at: "2004-01-01T00:00:00Z", fit_id: "synthetic-colors", data_as_of: "2003-05",
      qualities_entered: [], qualities_selected: [], qualities_not_entered: [], quality_decisions: [], entry_tests: {},
      excluded_test_seasons: { combined_validation: [], heldout_race_stage: [] },
      combined_validation: { gate: true }, heldout_race_stage: {}, simulated_seasons: 40, n_races_simulated: 5,
      driver_error_rates_used: false,
    },
  };
  return { ...data, race_pace: race, overall };
}

async function supply(page: Page, data = fixture()) {
  await page.route("**/data/releases/*.json", route => route.fulfill({ json: data }));
}

async function expectColors(rows: Locator, prefix = "") {
  await expect(rows).toHaveCount(samples.length);
  for (const sample of samples) {
    const row = rows.filter({ hasText: `${prefix}${sample.name}` });
    await expect(row.locator(".performance-value")).toHaveCSS("color", sample.color);
  }
}

async function expectIntervals(rows: Locator, prefix = "") {
  for (const sample of samples) {
    const row = rows.filter({ hasText: `${prefix}${sample.name}` });
    await expect(row.locator(".briefing-interval-line")).toHaveCSS("stroke", sample.color);
    await expect(row.locator(".briefing-interval-point")).toHaveCSS("fill", sample.color);
  }
}

test("median pace controls a continuous spectrum, with identical colors across views and filters", async ({ page }) => {
  await supply(page);
  await page.goto("./#briefing/qualifying");
  const briefingRows = page.locator(".briefing-ranking-row");
  await expectColors(briefingRows);
  await expectIntervals(briefingRows);
  // These tied medians have radically different intervals and occupy different
  // rows, so equal colors must not depend on uncertainty or ordinal position.
  await expect(page.locator(".performance-key")).toContainText(/lower/i);
  await expect(page.locator(".performance-key")).toContainText(/higher/i);
  await expect(page.locator(".performance-key")).toContainText(/uncertainty|certainty/i);
  await page.goto("./#drivers");
  const rows = page.locator(".rank-table tbody tr");
  await expectColors(rows);
  for (const sample of samples) {
    const interval = rows.filter({ hasText: sample.name }).locator("td.range-col");
    await expect(interval).toHaveCSS("color", sample.color);
  }
  await page.getByLabel("Search drivers").fill("Spectrum Quarter");
  await expect(rows).toHaveCount(1);
  await expect(rows.locator(".performance-value")).toHaveCSS("color", samples[3].color);
  await page.getByLabel("Clear search").click();
  await page.getByLabel("Filter by team").selectOption("spectrum_team_1");
  await expect(rows).toHaveCount(1);
  await expect(rows.locator(".performance-value")).toHaveCSS("color", red);
  await page.getByRole("button", { name: "Inspect Spectrum Tail", exact: true }).click();
  await expect(rows.locator("td.range-col")).toHaveCSS("color", red);
});

test("car, dry-race and equal-car tables use their primary estimate consistently", async ({ page }) => {
  await supply(page);
  for (const route of ["briefing/cars", "cars"]) {
    await page.goto(`./#${route}`);
    const rows = page.locator(route.startsWith("briefing") ? ".briefing-ranking-row" : ".rank-table tbody tr");
    await expectColors(rows, "Car ");
    if (route.startsWith("briefing")) await expectIntervals(rows, "Car ");
  }
  await page.goto("./#briefing/race");
  for (const [name, prefix] of [["Driver race pace", ""], ["Car race pace", "Car "]] as const) {
    const rows = page.getByRole("table", { name, exact: true }).locator("tbody tr");
    await expectColors(rows, prefix);
    await expectIntervals(rows, prefix);
  }
  for (const route of ["drivers/race", "cars/race", "briefing/overall", "drivers/overall"]) {
    await page.goto(`./#${route}`);
    const rows = page.locator(route.startsWith("briefing") ? ".briefing-ranking-row" : ".rank-table tbody tr");
    await expectColors(rows, route === "cars/race" ? "Car " : "");
  }
});

test("equal-valued and single-entry fields use a finite neutral color", async ({ page }) => {
  for (const count of [5, 1]) {
    const data = fixture();
    data.drivers = data.drivers.slice(0, count).map(driver => ({ ...driver, headline: estimate(.4, .2) }));
    await page.unroute("**/data/releases/*.json");
    await supply(page, data);
    for (const route of ["briefing/qualifying", "drivers"]) {
      await page.goto(`./#${route}`);
      await page.reload();
      const rows = page.locator(route.startsWith("briefing") ? ".briefing-ranking-row" : ".rank-table tbody tr");
      await expect(rows).toHaveCount(count);
      for (const value of await rows.locator(".performance-value").all()) {
        await expect(value).toHaveCSS("color", amber);
      }
      if (route.startsWith("briefing")) {
        for (const line of await rows.locator(".briefing-interval-line").all()) {
          await expect(line).toHaveCSS("stroke", amber);
        }
      }
    }
  }
});

test("colored estimates remain legible on black at phone width", async ({ page }) => {
  await supply(page);
  await page.setViewportSize({ width: 390, height: 844 });
  for (const route of ["briefing/qualifying", "drivers", "briefing/overall", "drivers/overall"]) {
    await page.goto(`./#${route}`);
    const rows = page.locator(route.startsWith("briefing") ? ".briefing-ranking-row" : ".rank-table tbody tr");
    await expectColors(rows);
    await expect(page.locator("body")).toHaveCSS("background-color", "rgb(0, 0, 0)");
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
    const accessibility = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
    expect(accessibility.violations).toEqual([]);
  }
});
