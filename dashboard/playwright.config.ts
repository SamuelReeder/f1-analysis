import { defineConfig } from "@playwright/test";
const staticUrl = process.env.DASHBOARD_TEST_URL;
export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  timeout: 30000,
  use: {
    baseURL: staticUrl || "http://127.0.0.1:4173/",
    trace: "retain-on-failure",
    launchOptions: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
      : {},
  },
  webServer: {
    command: staticUrl
      ? "npm run preview -- --host 127.0.0.1 --port 4180 --strictPort --base /f1-analysis/"
      : ".venv/bin/python -m f1rank.dashboard serve --port 4173",
    cwd: staticUrl ? "." : "..",
    url: staticUrl || "http://127.0.0.1:4173",
    reuseExistingServer: true,
  },
});
