import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import contract from "../../contracts/overall.json" with { type: "json" };
import syntheticExport from "../../tests/fixtures/overall_current_contract.json" with { type: "json" };
import { currentOverallAvailable, overallPair } from "../src/overall";
import type { CurrentOverallResult, Dataset } from "../src/types";

let current: Dataset;
test.beforeEach(async ({ request }) => {
  const pointer = await (await request.get("data/latest.json")).json();
  current = await (await request.get(`data/${pointer.url}`)).json();
});

// These invented outcomes exercise the display contract. They are not model
// results, and cannot advance the actual published pointer.
function fixture(): CurrentOverallResult {
  return structuredClone(syntheticExport) as CurrentOverallResult;
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
  ["missing pairwise comparisons", r => r.pairwise = []],
  ["mixed comparison posterior", r => r.pairwise[0].posterior_key = "d".repeat(64)],
  ["mixed comparison simulation", r => r.pairwise[0].simulation_key = "d".repeat(64)],
  ["wrong paired mean", r => r.pairwise[0].points_difference.mean = 3],
  ["self comparison", r => r.pairwise[0].driver_b = "alpha"],
  ["unknown comparison driver", r => r.pairwise[0].driver_b = "outsider"],
  ["missing explanation uncertainty", r => Object.assign(r.contributions[0].losses, { all: 1.7 })],
  ["unregistered explanation", r => r.contributions[0].losses.extra = r.contributions[0].losses.all],
  ["failed explanation precision", r => r.contributions[0].losses.all.conditional_precision_passed = false],
  ["failed paired precision", r => r.pairwise[0].points_difference.conditional_mcse_bound = .1],
  ["falsified precision bound", r => r.pairwise[0].points_difference.conditional_mcse_bound = 0],
  ["insufficient replicate inventory", r => r.evidence!.replicates_per_draw = 200],
  ["mixed explanation posterior", r => r.contributions[0].posterior_key = "d".repeat(64)],
  ["missing simulation identity", r => r.evidence!.simulation_key = ""],
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


test("paired comparisons preserve their narrow shared-draw interval and reverse its direction", async ({ page }) => {
  const result = fixture();
  expect(currentOverallAvailable(result)).toBe(true);
  expect(overallPair(result, "alpha", "beta")?.q05).toBe(3.95);
  expect(overallPair(result, "beta", "alpha")?.q05).toBe(-4.08);
  await supply(page, result);
  await page.goto("./#drivers/overall");
  const pair = page.getByRole("region", { name: "Compare expected performance" });
  await expect(pair).toContainText("+4.00 points per race");
  await expect(pair).toContainText("+3.95 to +4.08");
  await page.getByRole("button", { name: "Inspect Fixture Beta explanation" }).click();
  await expect(pair).toContainText("−4.00 points per race");
  await expect(pair).toContainText("−4.08 to −3.95");
  const explanation = page.getByRole("region", { name: "What explains the estimate?" });
  await expect(explanation).toContainText("(−0.90 to −0.40)");
});
