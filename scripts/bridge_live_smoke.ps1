# Bridge live-smoke build gate.
#
# 1. Builds the C# bridge mod (fails fast with a clear message if the .NET SDK
#    is missing).
# 2. Regenerates the offline golden fixture and runs the offline compare.
# 3. Optionally records a live trace when -Live is passed and STS2 is running
#    with the bridge mod loaded.
#
# Usage:
#   ./scripts/bridge_live_smoke.ps1
#   ./scripts/bridge_live_smoke.ps1 -Live -Host 127.0.0.1 -Port 9002

[CmdletBinding()]
param(
    [switch]$Live,
    [string]$BridgeHost = "127.0.0.1",
    [int]$Port = 9002
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Project = Join-Path $RepoRoot "bridge_mod/STS2BridgeMod.csproj"

if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    Write-Error "dotnet SDK not found on PATH. Install the .NET SDK to build the bridge mod."
    exit 1
}

Write-Host "==> Building bridge mod ($Project)"
$buildOutput = dotnet build $Project -c Release 2>&1
$buildExit = $LASTEXITCODE
$buildOutput | ForEach-Object { Write-Host $_ }

$compileErrors = $buildOutput | Select-String -Pattern ": error CS"
if ($compileErrors) {
    Write-Error "Bridge mod has C# compile errors."
    exit 1
}
$godotMissing = $buildOutput | Select-String -Pattern "GodotPath is not configured"
if ($buildExit -ne 0 -and $godotMissing) {
    Write-Warning "Bridge mod C# compiled cleanly, but the .pck export was skipped because the Godot 4.5.1 mono editor is not configured. Install it (see bridge_mod/STS2BridgeMod.csproj GodotPath) to produce a loadable mod; parity checks below still run."
}
elseif ($buildExit -ne 0) {
    Write-Error "Bridge mod build failed."
    exit 1
}

Write-Host "==> Regenerating offline golden fixture + self-compare"
python (Join-Path $RepoRoot "scripts/record_bridge_smoke.py")
if ($LASTEXITCODE -ne 0) {
    Write-Error "Offline smoke fixture compare failed."
    exit 1
}

Write-Host "==> Running offline bridge parity tests"
python -m pytest (Join-Path $RepoRoot "tests/test_bridge_live_smoke.py") (Join-Path $RepoRoot "tests/test_bridge_replay_harness.py") -q
if ($LASTEXITCODE -ne 0) {
    Write-Error "Offline bridge parity tests failed."
    exit 1
}

if ($Live) {
    Write-Host "==> Recording live trace from STS2 at ${BridgeHost}:${Port} (game must be in the smoke combat)"
    python (Join-Path $RepoRoot "scripts/record_bridge_smoke.py") --live --host $BridgeHost --port $Port
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Live record + compare reported a mismatch."
        exit 1
    }

    Write-Host "==> Running live bridge smoke test"
    python -m pytest (Join-Path $RepoRoot "tests/test_bridge_live_smoke.py") --run-live-bridge -q
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Live bridge smoke test failed."
        exit 1
    }
}

Write-Host "bridge_live_smoke: OK"
