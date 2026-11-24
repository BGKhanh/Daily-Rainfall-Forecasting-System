import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

function validateProdApiBasePlugin(): Plugin {
  return {
    name: "validate-prod-api-base-url",
    apply: "build",
    configResolved(config) {
      const apiBase = (process.env.VITE_API_BASE_URL ?? "").trim();
      const strictRequired =
        process.env.CLOUDFLARE_WORKERS === "1" || process.env.REQUIRE_VITE_API_BASE_URL === "1";

      if (config.command === "build" && !apiBase) {
        const msg =
          "VITE_API_BASE_URL is empty during production Vite build; SPA API requests will target same-origin instead of the decoupled backend.";
        if (strictRequired) {
          throw new Error(`[BUILD_ENV_ERROR] ${msg}`);
        }
        console.warn(`[vite-build-warning] ${msg}`);
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), validateProdApiBasePlugin()],
  server: {
    port: 5173,
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/locations": "http://127.0.0.1:8000",
      "/observations": "http://127.0.0.1:8000",
      "/models": "http://127.0.0.1:8000",
      "/predict": "http://127.0.0.1:8000",
      "/scores": "http://127.0.0.1:8000",
      "/benchmark": "http://127.0.0.1:8000",
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/setupTests.ts",
  },
});
