param(
  [ValidateSet("auto", "desktop", "check", "build")]
  [string]$Mode = "auto"
)

$ErrorActionPreference = "Stop"

$RootDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$GuiDir = Join-Path $RootDir "gui"

function Require-Command($Name) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "error: $Name is required to run Factory Control"
  }
}

function Ensure-WslBridgeConfig {
  if ($env:OPENGATEWAY_WSL_BRIDGE) {
    return
  }

  if ($RootDir -match '^\\\\wsl\$\\([^\\]+)\\(.+)$') {
    $distro = $Matches[1]
    $workspace = "/" + ($Matches[2] -replace '\\', '/')

    $env:OPENGATEWAY_WSL_BRIDGE = "1"
    if (-not $env:OPENGATEWAY_WSL_DISTRO) {
      $env:OPENGATEWAY_WSL_DISTRO = $distro
    }
    if (-not $env:OPENGATEWAY_WSL_WORKSPACE) {
      $env:OPENGATEWAY_WSL_WORKSPACE = $workspace
    }

    $debugBin = Join-Path $RootDir "target\debug\opengateway"
    $releaseBin = Join-Path $RootDir "target\release\opengateway"
    if (-not $env:OPENGATEWAY_WSL_BIN -and (Test-Path $debugBin)) {
      $env:OPENGATEWAY_WSL_BIN = "$workspace/target/debug/opengateway"
    } elseif (-not $env:OPENGATEWAY_WSL_BIN -and (Test-Path $releaseBin)) {
      $env:OPENGATEWAY_WSL_BIN = "$workspace/target/release/opengateway"
    }
  }
}

Require-Command npm
Require-Command cargo

if (-not (Test-Path (Join-Path $GuiDir "node_modules"))) {
  Write-Host "Installing GUI dependencies..."
  Push-Location $GuiDir
  try {
    npm install
  } finally {
    Pop-Location
  }
}

Ensure-WslBridgeConfig

Push-Location $GuiDir
try {
  switch ($Mode) {
    "auto" { npm run desktop; break }
    "desktop" { npm run desktop; break }
    "check" { npm run desktop:check; break }
    "build" { npm run desktop:build; break }
  }
} finally {
  Pop-Location
}
