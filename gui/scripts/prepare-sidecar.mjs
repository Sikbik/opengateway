import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync, spawnSync } from "node:child_process";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const guiDir = join(scriptDir, "..");
const repoDir = join(guiDir, "..");
const binariesDir = join(guiDir, "src-tauri", "binaries");

const targetTriple =
  process.env.TARGET_TRIPLE ||
  process.env.TAURI_ENV_TARGET_TRIPLE ||
  process.env.CARGO_BUILD_TARGET ||
  process.env.npm_config_target ||
  detectHostTarget();

const releaseBuild = !process.argv.includes("--debug");
const profile = releaseBuild ? "release" : "debug";
const binaryName = targetTriple.includes("windows")
  ? "opengateway.exe"
  : "opengateway";

runCargoBuild();

const sourceBinary = join(repoDir, "target", targetTriple, profile, binaryName);
if (!existsSync(sourceBinary)) {
  throw new Error(`built backend not found at ${sourceBinary}`);
}

mkdirSync(binariesDir, { recursive: true });
const sidecarName = targetTriple.includes("windows")
  ? `opengateway-${targetTriple}.exe`
  : `opengateway-${targetTriple}`;
const bundledBinary = join(binariesDir, sidecarName);
copyFileSync(sourceBinary, bundledBinary);

console.log(`Prepared bundled backend: ${bundledBinary}`);

function runCargoBuild() {
  const args = [
    "build",
    "--manifest-path",
    join(repoDir, "Cargo.toml"),
    "--target",
    targetTriple,
  ];
  if (releaseBuild) {
    args.push("--release");
  }

  const result = spawnSync("cargo", args, {
    cwd: repoDir,
    stdio: "inherit",
  });

  if (result.status !== 0) {
    throw new Error("failed to build bundled opengateway backend");
  }
}

function detectHostTarget() {
  const output = execFileSync("rustc", ["-vV"], { encoding: "utf8" });
  const hostLine = output
    .split(/\r?\n/)
    .find((line) => line.startsWith("host: "));
  if (!hostLine) {
    throw new Error("failed to detect Rust host target");
  }
  return hostLine.slice("host: ".length).trim();
}
