import { invoke } from "@tauri-apps/api/core";

const BRIDGE_PREFIX = "/__factory-control";

function isTauriRuntime() {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

async function fetchBridge<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BRIDGE_PREFIX}${path}`, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `bridge request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function runtimeMode(): "desktop" | "browser" {
  return isTauriRuntime() ? "desktop" : "browser";
}

export async function call<T>(
  command: string,
  args?: Record<string, unknown>,
): Promise<T> {
  if (isTauriRuntime()) {
    return invoke<T>(command, args);
  }

  switch (command) {
    case "load_snapshot":
      return fetchBridge<T>("/snapshot");
    case "tail_logs":
      return fetchBridge<T>(`/logs?limit=${args?.limit ?? 160}`);
    case "start_gateway":
      return fetchBridge<T>("/start", { method: "POST" });
    case "stop_gateway":
      return fetchBridge<T>("/stop", { method: "POST" });
    case "run_doctor":
      return fetchBridge<T>("/doctor", { method: "POST" });
    case "sync_factory":
      return fetchBridge<T>("/sync-factory", { method: "POST" });
    case "set_droid_model":
      return fetchBridge<T>("/droids/model", {
        method: "POST",
        body: JSON.stringify(args ?? {}),
      });
    default:
      throw new Error(`unsupported runtime command: ${command}`);
  }
}
