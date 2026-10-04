import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import type { Dataset, OverallResult, RacePace } from "../src/types";

let current: Dataset;
let pointer: { schema_version: number; release: string; url: string; published_at: string };

// Read a published release only for the surrounding app schema. All scenario
// outcomes below are synthetic; no historical championship files are fixtures.
test.beforeEach(async ({ request }) => {
  pointer = await (await request.get("data/latest.json")).json();
  current = await (await request.get(`data/${pointer.url}`)).json();
});

async function supply(page: Page, data: Dataset) {
  await page.route("**/data/releases/*.json", (route) => route.fulfill({ json: data }));
}

function overallFixture(): OverallResult {
  return {
    status: "established",
    reason: "The synthetic combined held-out validation passed.",
    standings: [
      { driver_id: "fixture_beta", name: "Briefing Driver Beta", points_per_race: 7.25, p_title: .18, rank_median: 2, rank_lo: 1, rank_hi: 3 },
      { driver_id: "fixture_alpha", name: "Briefing Driver Alpha", points_per_race: 9.75, p_title: .82, rank_median: 1, rank_lo: 1, rank_hi: 2 },
    ],
    contributions: [],
    evidence: {
      recorded_at: "2004-01-01T00:00:00Z", fit_id: "synthetic-briefing", data_as_of: "2003-05",
      qualities_entered: ["first_lap"], qualities_selected: ["first_lap"], qualities_not_entered: ["consistency"],
      quality_decisions: [
        { quality: "first_lap", entered: true, tested: true, reason: "Passed the synthetic conditional test and combined validation." },
        { quality: "consistency", entered: false, tested: true, reason: "Did not establish an improvement in the synthetic test." },
      ],
      entry_tests: { first_lap: { enters: true }, consistency: { enters: false } },
      excluded_test_seasons: { combined_validation: [2001], heldout_race_stage: [2001, 2002] },
      combined_validation: { gate: true }, heldout_race_stage: {},
      simulated_seasons: 40, n_races_simulated: 5, driver_error_rates_used: false,
    },
  };
}

function raceFixture(driversPassed = true, carsPassed = true): RacePace {
  const gate = (passed: boolean) => ({
    passed, improves: passed, calibrated: true, sharper: true,
    n_races: 18, n_predictions: 160, rmse: .2, baseline_rmse: .3,
    mse_difference: -.05, mse_difference_ci95: [-.08, -.02], coverage90: .9,
    mean_interval_width: .4, baseline_interval_width: .6,
  });
  const row = (id: string, name: string, median: number) => ({
    id, name, pace: { q05: median - .1, median, q95: median + .1, rank_lo: 1, rank_hi: 2, p_fastest: .6 },
    races: 10, laps: 300, last_race: "2003-05",
  });
  return {
    model: "total-dry-race-pace-v1", fit_id: "synthetic-briefing", season: 2003,
    data_as_of: { event_id: "2003-05", season: 2003, round: 5, date: "2003-05-04", race_name: "Fixture Grand Prix" },
    first_event: "2001-01", n_laps: 90000, n_races: 90,
    diagnostics: current.meta.diagnostics,
    drivers: driversPassed ? [row("slow_driver", "Fixture Race Follower", -.15), row("fast_driver", "Fixture Race Leader", .15)] : [],
    cars: carsPassed ? [row("slow_car", "Fixture Car Follower", -.25), row("fast_car", "Fixture Car Leader", .25)] : [],
    unrated_drivers: [], unrated_cars: [],
    validation: { folds: ["2001-10", "2002-10"], design: "Synthetic held-out test", metrics: { drivers: gate(driversPassed), cars: gate(carsPassed) } },
  };
}

function evidenceFixture(ci95: [number, number] | null): Dataset {
  const data = structuredClone(current);
  data.asof = {
    events: [], series: { drivers: {}, cars: {} }, latest: null,
    pooled: {
      n_events: 12, n_pairs: 80, rmse: .1, rmse_naive: .2, rmse_zero: .3, coverage90: .9, order_spearman: .7,
      ...(ci95 ? { vs_naive: { mse_difference: -.03, ci95 } } : {}),
    },
  };
  return data;
}

const chapters = [
  ["qualifying", "Drivers"],
  ["cars", "Cars"],
  ["race", "Race pace"],
  ["overall", "Equal car"],
  ["evidence", "The evidence"],
] as const;

test("the default briefing leads with the runtime ranking and its uncertainty", async ({ page }) => {
  await page.goto("./");
  const view = page.locator(".briefing-view");
  await expect(view).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Main navigation" }).getByRole("link", { name: "Briefing", exact: true }))
    .toHaveAttribute("aria-current", "page");
  const ordered = [...current.drivers].sort((a, b) => b.headline.median - a.headline.median);
  const rows = view.locator(".briefing-ranking-row");
  await expect(rows).toHaveCount(Math.min(5, ordered.length));
  await expect(rows.first()).toContainText(ordered[0].name);
  await expect(rows.first()).toContainText(Math.abs(ordered[0].headline.median).toFixed(3));
  await expect(view).toContainText(/current team/i);
  await expect(view).toContainText(/uncertain|uncertainty|intervals/i);
  await expect(view.locator(".briefing-takeaway")).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Briefing chapters" }).getByRole("link", { name: "Drivers", exact: true }))
    .toHaveAttribute("aria-current", /^(step|page)$/);
});

test("chapters advance, survive reload, and respect browser back and forward", async ({ page }) => {
  await page.goto("./#briefing/qualifying");
  const chapterNav = page.getByRole("navigation", { name: "Briefing chapters" });
  const controls = page.getByRole("navigation", { name: "Briefing controls" });
  const next = controls.getByRole("link", { name: /Next: Cars$/ });
  await next.focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/#briefing\/cars$/);
  await expect(page.locator("#main")).toBeFocused();
  await page.reload();
  await expect(chapterNav.getByRole("link", { name: "Cars", exact: true })).toHaveAttribute("aria-current", /^(step|page)$/);
  await expect(page.locator(".briefing-ranking-row").first()).toContainText([...current.cars].sort((a, b) => b.pace.median - a.pace.median)[0].name);
  await expect(page.locator(".briefing-view")).toContainText(/track.neutral|circuit/i);
  await page.goBack();
  await expect(page).toHaveURL(/#briefing\/qualifying$/);
  await page.goForward();
  await expect(page).toHaveURL(/#briefing\/cars$/);
  for (const [chapter, label] of chapters.slice(2)) {
    await chapterNav.getByRole("link", { name: label, exact: true }).click();
    await expect(page).toHaveURL(new RegExp(`#briefing/${chapter}$`));
    await expect(chapterNav.getByRole("link", { name: label, exact: true })).toHaveAttribute("aria-current", /^(step|page)$/);
  }
  await controls.getByRole("link", { name: /Back to the start$/ }).click();
  await expect(page).toHaveURL(/#briefing\/qualifying$/);
});

for (const [chapter, drilldown] of [["qualifying", "drivers"], ["cars", "cars"], ["race", "drivers/race"], ["overall", "drivers/overall"], ["evidence", "forecasts"]] as const) {
  test(`${chapter} offers a granular result and a contextual return`, async ({ page }) => {
    await page.goto(`./#briefing/${chapter}`);
    await page.locator(`.briefing-view a[href="#${drilldown}"]`).first().click();
    await expect(page).toHaveURL(new RegExp(`#${drilldown}$`));
    const back = page.getByRole("link", { name: "Back to briefing", exact: true });
    await expect(back).toHaveAttribute("href", `#briefing/${chapter}`);
    await back.click();
    await expect(page).toHaveURL(new RegExp(`#briefing/${chapter}$`));
    await expect(page.locator(".briefing-view")).toBeVisible();
  });
}

test("car comparison keeps its entity through reload, browser history and briefing return", async ({ page }) => {
  await page.goto("./#briefing/cars");
  await page.locator('.briefing-view a[href="#compare/cars"]').click();
  await expect(page).toHaveURL(/#compare\/cars$/);
  await expect(page.getByLabel("FIRST CAR", { exact: true })).toHaveValue(current.cars[0].id);
  await expect(page.getByLabel("SECOND CAR", { exact: true })).toHaveValue(current.cars[1].id);
  await page.reload();
  await expect(page.getByLabel("FIRST CAR", { exact: true })).toBeVisible();
  await expect(page.getByLabel("SECOND CAR", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Drivers", exact: true }).click();
  await expect(page).toHaveURL(/#compare$/);
  await expect(page.getByLabel("FIRST DRIVER", { exact: true })).toBeVisible();
  await page.goBack();
  await expect(page).toHaveURL(/#compare\/cars$/);
  await expect(page.getByLabel("FIRST CAR", { exact: true })).toBeVisible();
  await expect(page.getByLabel("SECOND CAR", { exact: true })).toBeVisible();
  const back = page.getByRole("link", { name: "Back to briefing", exact: true });
  await expect(back).toHaveAttribute("href", "#briefing/cars");
  await back.click();
  await expect(page).toHaveURL(/#briefing\/cars$/);
  await expect(page.locator(".briefing-view")).toBeVisible();
});

test("a new release updates the briefing conclusion and ranking together", async ({ page }) => {
  await page.goto("./#briefing/qualifying");
  await expect(page.locator(".briefing-ranking-row").first()).toBeVisible();
  const next = structuredClone(current);
  const leader = next.drivers[next.drivers.length - 1];
  leader.name = "Newly published briefing leader";
  const median = Math.max(...next.drivers.map((d) => d.headline.median)) + .45;
  leader.headline = { ...leader.headline, median, q05: median - .1, q95: median + .1, rank_lo: 1, rank_hi: 2 };
  const release = "a1234567890bcdef1234";
  await page.route("**/data/latest.json", (route) => route.fulfill({ json: { ...pointer, release, url: `releases/${release}.json` } }));
  await page.route(`**/data/releases/${release}.json`, (route) => route.fulfill({ json: next }));
  await page.getByLabel("Check for updates").click();
  await expect(page.locator(".briefing-ranking-row").first()).toContainText(leader.name);
  await expect(page.locator(".briefing-takeaway")).toContainText(leader.name);
  await expect(page.locator(".briefing-ranking-row").first()).toContainText(median.toFixed(3));
});

for (const state of ["not established", "stale", "unavailable", "failed gate", "legacy"] as const) {
  test(`the ${state} equal-car briefing never presents untrusted standings`, async ({ page }) => {
    const overall = overallFixture();
    if (state === "failed gate") overall.evidence!.combined_validation.gate = false;
    else if (state !== "legacy") overall.status = state;
    await supply(page, { ...current, overall: state === "legacy" ? undefined : overall });
    await page.goto("./#briefing/overall");
    const view = page.locator(".briefing-view");
    await expect(view).toBeVisible();
    await expect(view.locator(".briefing-ranking")).toHaveCount(0);
    await expect(view).not.toContainText("Briefing Driver Alpha");
    await expect(view).not.toContainText("Briefing Driver Beta");
    await expect(view).toContainText(/not established|unavailable|stale/i);
    await expect(view).toContainText(/experimental/i);
  });
}

test("a passed equal-car briefing explains the runtime outcome and exclusions", async ({ page }) => {
  const overall = overallFixture();
  await supply(page, { ...current, overall });
  await page.goto("./#briefing/overall");
  const view = page.locator(".briefing-view");
  const leader = [...overall.standings].sort((a, b) => b.points_per_race - a.points_per_race)[0];
  const row = view.locator(".briefing-ranking-row").first();
  await expect(row).toContainText(leader.name);
  await expect(row).toContainText(leader.points_per_race.toFixed(2));
  await expect(row).toContainText(`${(leader.p_title * 100).toFixed(1)}%`);
  await expect(row).toContainText(`${leader.rank_lo}–${leader.rank_hi}`);
  await expect(view).toContainText(/experimental/i);
  await expect(view).toContainText(/equal.car/i);
  await expect(view).toContainText(/First.lap/i);
  await expect(view).toContainText(/Consistency/i);
  await expect(view).toContainText("2001");
  await expect(view).toContainText("2002");
});

for (const driversPassed of [true, false]) {
  test(`race briefing independently publishes the ${driversPassed ? "driver" : "car"} evidence`, async ({ page }) => {
    await supply(page, { ...current, race_pace: raceFixture(driversPassed, !driversPassed) });
    await page.goto("./#briefing/race");
    const view = page.locator(".briefing-view");
    await expect(view.getByRole("heading", { name: "Driver race pace", exact: true })).toBeVisible();
    await expect(view.getByRole("heading", { name: "Car race pace", exact: true })).toBeVisible();
    await expect(view.locator(".briefing-ranking")).toHaveCount(1);
    await expect(view.locator(".briefing-ranking-row").first()).toContainText(driversPassed ? "Fixture Race Leader" : "Fixture Car Leader");
    await expect(view).not.toContainText(driversPassed ? "Fixture Car Leader" : "Fixture Race Leader");
    await expect(view).toContainText(/withheld|not established/i);
    await expect(view).toContainText(/dry/i);
  });
}

for (const [ci, verdict] of [
  [[-.08, -.01], "Better than last season’s gap"],
  [[-.08, .02], "No clear difference from last season’s gap"],
  [[.01, .08], "Worse than last season’s gap"],
  [null, "Baseline comparison unavailable"],
] as const) {
  test(`evidence concludes '${verdict}' from the interval, not just headline error`, async ({ page }) => {
    // The raw error is lower in every case, so it cannot determine the verdict.
    const data = evidenceFixture(ci ? [ci[0], ci[1]] : null);
    await supply(page, data);
    await page.goto("./#briefing/evidence");
    await expect(page.locator(".briefing-view")).toContainText(verdict);
    await expect(page.locator(".briefing-view")).toContainText(data.asof!.pooled!.rmse.toFixed(3));
  });
}

for (const [chapter] of chapters) {
  test(`${chapter} briefing is accessible at desktop and phone widths`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await supply(page, { ...current, race_pace: raceFixture(), overall: overallFixture() });
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto(`./#briefing/${chapter}`);
    await expect(page.locator(".briefing-view")).toBeVisible();
    for (const width of [1440, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.evaluate(() => document.fonts.ready);
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
      const audit = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
      expect(audit.violations.map((violation) => ({ id: violation.id, targets: violation.nodes.map((node) => node.target) }))).toEqual([]);
      if (width !== 320) await page.screenshot({ path: `test-results/briefing-${chapter}-${width}.png`, fullPage: true });
    }
    expect(errors).toEqual([]);
  });
}


test("presentation controls stay in view while reading long chapters", async ({ page }) => {
  await page.goto("./#briefing/qualifying");
  const next = page.getByRole("navigation", { name: "Briefing controls" }).getByRole("link", { name: /Next: Cars$/ });
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 800 });
    await page.evaluate(() => window.scrollTo(0, 0));
    await expect(next).toBeInViewport();
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight / 2));
    await expect(next).toBeInViewport();
  }
  await next.click();
  await expect(page).toHaveURL(/#briefing\/cars$/);
  await expect(page.locator("#main")).toBeFocused();
});
