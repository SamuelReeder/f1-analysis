import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import contract from "../../contracts/overall.json" with { type: "json" };
import type { CurrentOverallResult, Dataset } from "../src/types";

let current: Dataset;
test.beforeEach(async ({ request }) => {
  const pointer = await (await request.get("data/latest.json")).json();
  current = await (await request.get(`data/${pointer.url}`)).json();
});

// These invented outcomes exercise the display contract. They are not model
// results, and cannot advance the actual published pointer.
function fixture(): CurrentOverallResult {
  const fit_id = "synthetic-joint-fit", scenario_id = "synthetic-common-calendar";
  return {
    schema_version: 2, model_version: "overall-joint-v1",
    target: "current_equal_machinery_and_team", status: "established",
    reason: "Every required check passed in this synthetic display fixture.",
    evidence: {
      recorded_at: "2004-01-01T00:00:00Z", data_as_of: "2003-05", fit_id,
      field: ["beta", "alpha"], scenario: { id: scenario_id, label: "Synthetic common calendar", races: 12 },
      posterior_draws: 4000, replicates_per_draw: 200,
      checks: Object.fromEntries(contract.required_checks.map(key => [key, {
        passed: true, source: `synthetic/${key}.json`, reason: `Synthetic ${key.replaceAll("_", " ")} evidence.`,
      }])),
    },
    standings: [
      { driver_id: "beta", name: "Fixture Beta", expected_points: {
        mean: 18.5, q05: 18.1, q95: 19.3, posterior_sd: .3, mcse: .002, max_conditional_mcse: .03,
      }, p_title: .2, season_rank: { q05: 1, median: 2, q95: 2 } },
      { driver_id: "alpha", name: "Fixture Alpha", expected_points: {
        mean: 22.5, q05: 21.9, q95: 23.3, posterior_sd: .4, mcse: .002, max_conditional_mcse: .04,
      }, p_title: .8, season_rank: { q05: 1, median: 1, q95: 2 } },
    ],
    contributions: [
      { driver_id: "alpha", fit_id, scenario_id, losses: { qualifying_pace: .8, total_race_pace: 1.2, all: 1.7 } },
      { driver_id: "beta", fit_id, scenario_id, losses: { qualifying_pace: -.2, total_race_pace: -.6, all: -.7 } },
    ],
    limitations: ["Synthetic fixture only; no real driver is evaluated."],
  };
}

async function supply(page: Page, result: unknown) {
  await page.route("**/data/releases/*.json", route => route.fulfill({ json: { ...current, overall: result } }));
}

test("the single Overall view separates performance uncertainty, season variation and numerical error", async ({ page }) => {
  await supply(page, fixture());
  await page.goto("./#drivers/overall");
  const table = page.getByRole("table", { name: "Equal-car championship standings" });
  await expect(table.locator("tbody tr")).toHaveCount(2);
  await expect(table.locator("tbody tr").first()).toContainText("Fixture Alpha");
  await expect(table.getByRole("columnheader", { name: "Expected points range" })).toBeVisible();
  await expect(table.getByRole("columnheader", { name: "Season finish range" })).toBeVisible();
  await expect(page.getByText("Current performance · equal machinery and team support")).toBeVisible();
  const explanation = page.getByRole("region", { name: "What explains the estimate?" });
  await expect(explanation.getByRole("heading", { name: "Fixture Alpha" })).toBeVisible();
  await expect(explanation).toContainText("Numerical standard error");
  await expect(explanation).toContainText("these values should not be added");
  await table.getByRole("button", { name: "Inspect Fixture Beta explanation" }).click();
  await expect(explanation.getByRole("heading", { name: "Fixture Beta" })).toBeVisible();
  await expect(explanation).toContainText("−0.60");
  await expect(page.getByRole("link", { name: "Overall (equal car)", exact: true })).toHaveCount(1);
  await expect(page.getByRole("tab", { name: /preview|methodology/i })).toHaveCount(0);
});

for (const key of contract.required_checks) {
  for (const failure of ["failed", "missing"] as const) {
    test(`${key} evidence ${failure} withholds standings despite an old combined success flag`, async ({ page }) => {
      const result = fixture();
      if (failure === "failed") result.evidence!.checks[key].passed = false;
      else delete result.evidence!.checks[key];
      Object.assign(result.evidence!, { combined_validation: { gate: true } });
      await supply(page, result);
      await page.goto("./#drivers/overall");
      await expect(page.getByRole("heading", { name: "Overall ranking not established" })).toBeVisible();
      await expect(page.getByRole("table", { name: "Equal-car championship standings" })).toHaveCount(0);
      await expect(page.getByRole("region", { name: "What explains the estimate?" })).toHaveCount(0);
      await page.goto("./#briefing/overall");
      await expect(page.getByRole("heading", { name: "Overall ranking not established" })).toBeVisible();
      await expect(page.getByRole("table", { name: "Equal-car championship summary" })).toHaveCount(0);
    });
  }
}

const invalid: [string, (result: CurrentOverallResult) => void][] = [
  ["wrong model", r => Object.assign(r, { model_version: "unregistered-model" })],
  ["wrong target", r => Object.assign(r, { target: "current_team" })],
  ["missing entrant", r => r.standings.pop()],
  ["duplicate entrant", r => r.standings[1].driver_id = r.standings[0].driver_id],
  ["unrecognized entrant", r => r.standings[0].driver_id = "outsider"],
  ["duplicate field", r => r.evidence!.field[1] = r.evidence!.field[0]],
  ["missing evidence", r => r.evidence = null],
  ["incomplete evidence", r => Object.assign(r, { evidence: { checks: {} } })],
  ["invalid check reason", r => Object.assign(r.evidence!.checks.provenance, { reason: { malformed: true } })],
  ["missing evidence source", r => r.evidence!.checks.provenance.source = ""],
  ["missing fit identity", r => r.evidence!.fit_id = ""],
  ["invalid interval", r => r.standings[0].expected_points.q05 = 25],
  ["nonfinite points", r => r.standings[0].expected_points.mean = NaN],
  ["insufficient simulation precision", r => r.standings[0].expected_points.max_conditional_mcse = .1],
  ["missing explanation", r => r.contributions.pop()],
  ["mixed explanation draws", r => r.contributions[0].fit_id = "different-fit"],
  ["mixed explanation scenario", r => r.contributions[0].scenario_id = "different-scenario"],
  ["missing race explanation", r => delete r.contributions[0].losses.total_race_pace],
  ["duplicate explanation", r => r.contributions[1].driver_id = r.contributions[0].driver_id],
  ["invalid title probabilities", r => r.standings[0].p_title = .9],
  ["invalid finish range", r => r.standings[0].season_rank.q95 = 3],
];
for (const [name, mutate] of invalid) {
  test(`${name} cannot publish a claimed established ranking`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    const result = fixture(); mutate(result);
    await supply(page, result);
    await page.goto("./#drivers/overall");
    await expect(page.getByRole("heading", { name: "Overall ranking not established" })).toBeVisible();
    await expect(page.getByRole("table", { name: "Equal-car championship standings" })).toHaveCount(0);
    expect(errors).toEqual([]);
  });
}

for (const status of ["not established", "stale", "unavailable"] as const) {
  test(`${status} hides accidentally retained successor rows`, async ({ page }) => {
    const result = fixture(); result.status = status;
    await supply(page, result);
    await page.goto("./#drivers/overall");
    await expect(page.getByRole("heading", { name: `Overall ranking ${status}` })).toBeVisible();
    await expect(page.getByRole("table", { name: "Equal-car championship standings" })).toHaveCount(0);
  });
}

for (const schema_version of [undefined, 1, 3]) {
  test(`successor with schema ${schema_version} cannot fall back to legacy success`, async ({ page }) => {
    const result = { ...fixture(), schema_version };
    Object.assign(result.evidence!, { combined_validation: { gate: true } });
    await supply(page, result);
    for (const route of ["drivers/overall", "briefing/overall"]) {
      await page.goto(`./#${route}`);
      await expect(page.getByRole("heading", { name: "Overall ranking unavailable" })).toBeVisible();
      await expect(page.getByRole("table")).toHaveCount(0);
    }
  });
}

test("briefing and methodology describe the successor with the same target", async ({ page }) => {
  await supply(page, fixture());
  await page.goto("./#briefing/overall");
  const table = page.getByRole("table", { name: "Equal-car championship summary" });
  await expect(table.locator("tbody tr").first()).toContainText("Fixture Alpha");
  await expect(page.getByText("Experimental · equal machinery and team support")).toBeVisible();
  await page.getByRole("link", { name: "Full standings, explanations & validation" }).click();
  await expect(page.getByRole("table", { name: "Equal-car championship standings" })).toBeVisible();
  await page.goto("./#methodology");
  await expect(page.getByText("performance together under equal machinery and team support.", { exact: false })).toBeVisible();
  await expect(page.getByText("The old model’s combined gate cannot establish this replacement.", { exact: false })).toBeVisible();
});

for (const status of ["established", "not established"] as const) {
  test(`successor ${status} remains accessible on small screens`, async ({ page }) => {
    const result = fixture(); result.status = status;
    await supply(page, result);
    await page.setViewportSize({ width: 390, height: 844 });
    for (const route of ["drivers/overall", "briefing/overall"]) {
      await page.goto(`./#${route}`);
      await expect(page.getByRole("heading", { name: route.startsWith("drivers") ? "Drivers" : "Overall (equal car)", level: 1 })).toBeVisible();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
      const scan = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
      expect(scan.violations).toEqual([]);
    }
  });
}
