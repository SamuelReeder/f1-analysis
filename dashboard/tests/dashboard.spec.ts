import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import fs from "node:fs/promises";
import type { Dataset } from "../src/types";
let current: Dataset;

test.beforeEach(async ({ page, request }) => {
  const pointer = await (await request.get("data/latest.json")).json();
  current = await (await request.get(`data/${pointer.url}`)).json();
  await page.goto("./");
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
  await page.getByText("Methodology", { exact: true }).click();
  await expect(
    page.getByText("A time-varying Bayesian model separates"),
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

for (const view of ["drivers", "cars", "compare", "health"]) {
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
