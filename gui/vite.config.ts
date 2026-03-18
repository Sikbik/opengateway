import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

const guiDir = fileURLToPath(new URL(".", import.meta.url));

function resolveGatewayBinary() {
  if (process.env.OPENGATEWAY_BIN?.trim()) {
    return process.env.OPENGATEWAY_BIN.trim();
  }

  const candidates = [
    path.resolve(guiDir, "../target/debug/opengateway"),
    path.resolve(guiDir, "../target/release/opengateway"),
  ];

  return candidates.find((candidate) => fs.existsSync(candidate)) ?? "opengateway";
}

function execGateway(args: string[]) {
  const binary = resolveGatewayBinary();
  return new Promise<string>((resolve, reject) => {
    execFile(binary, args, { cwd: path.resolve(guiDir, "..") }, (error, stdout, stderr) => {
      if (error) {
        reject(new Error(stderr.trim() || stdout.trim() || error.message));
        return;
      }
      resolve(stdout.trim());
    });
  });
}

function factoryControlBridge(): Plugin {
  return {
    name: "factory-control-bridge",
    configureServer(server) {
      server.middlewares.use("/__factory-control", async (req, res) => {
        try {
          const url = new URL(req.url ?? "/", "http://127.0.0.1");
          const body = await new Promise<string>((resolve, reject) => {
            let raw = "";
            req.on("data", (chunk) => {
              raw += chunk;
            });
            req.on("end", () => resolve(raw));
            req.on("error", reject);
          });

          const writeJson = (payload: string) => {
            res.statusCode = 200;
            res.setHeader("content-type", "application/json");
            res.end(payload);
          };

          if (req.method === "GET" && url.pathname === "/snapshot") {
            writeJson(await execGateway(["gui-snapshot"]));
            return;
          }

          if (req.method === "GET" && url.pathname === "/acp-snapshot") {
            writeJson(await execGateway(["gui-acp-snapshot"]));
            return;
          }

          if (req.method === "POST" && url.pathname === "/acp-bridge/start") {
            const parsed = JSON.parse(body || "{}") as { agent?: string };
            if (!parsed.agent) {
              res.statusCode = 400;
              res.end("agent is required");
              return;
            }
            writeJson(
              await execGateway([
                "gui-acp-bridge-start",
                "--agent",
                parsed.agent,
              ]),
            );
            return;
          }

          if (req.method === "POST" && url.pathname === "/acp-bridge/stop") {
            const parsed = JSON.parse(body || "{}") as { agent?: string };
            if (!parsed.agent) {
              res.statusCode = 400;
              res.end("agent is required");
              return;
            }
            writeJson(
              await execGateway([
                "gui-acp-bridge-stop",
                "--agent",
                parsed.agent,
              ]),
            );
            return;
          }

          if (req.method === "GET" && url.pathname === "/acp-inspect") {
            const sessionId = url.searchParams.get("sessionId")?.trim();
            if (!sessionId) {
              res.statusCode = 400;
              res.end("sessionId is required");
              return;
            }
            writeJson(
              await execGateway([
                "gui-acp-inspect",
                "--session-id",
                sessionId,
                "--limit",
                url.searchParams.get("limit") ?? "12",
              ]),
            );
            return;
          }

          if (req.method === "GET" && url.pathname === "/logs") {
            writeJson(
              await execGateway([
                "gui-logs",
                "--limit",
                url.searchParams.get("limit") ?? "160",
              ]),
            );
            return;
          }

          if (req.method === "POST" && url.pathname === "/start") {
            writeJson(await execGateway(["gui-start"]));
            return;
          }

          if (req.method === "POST" && url.pathname === "/stop") {
            writeJson(await execGateway(["gui-stop"]));
            return;
          }

          if (req.method === "POST" && url.pathname === "/doctor") {
            writeJson(await execGateway(["gui-doctor"]));
            return;
          }

          if (req.method === "POST" && url.pathname === "/sync-factory") {
            writeJson(await execGateway(["gui-sync-factory"]));
            return;
          }

          if (req.method === "POST" && url.pathname === "/droids/model") {
            const parsed = JSON.parse(body || "{}") as { path?: string; model?: string };
            if (!parsed.path || !parsed.model) {
              res.statusCode = 400;
              res.end("path and model are required");
              return;
            }
            writeJson(
              await execGateway([
                "gui-set-droid-model",
                "--path",
                parsed.path,
                "--model",
                parsed.model,
              ]),
            );
            return;
          }

          res.statusCode = 404;
          res.end("not found");
        } catch (error) {
          res.statusCode = 500;
          res.end(error instanceof Error ? error.message : String(error));
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), factoryControlBridge()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
});
