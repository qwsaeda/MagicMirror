<#
.SYNOPSIS
    MagicMirror one-click packaging script
.DESCRIPTION
    Build frontend + Rust server + package + (optional) Tauri installer
.PARAMETER SkipFrontend
    Skip frontend build (use existing dist/)
.PARAMETER SkipTauri
    Skip Tauri installer build
.PARAMETER WithEnhancer
    Include GFPGAN enhancer model (full ~800MB)
.PARAMETER SkipBuildServer
    Skip Rust server compile (use existing binary)
#>

param(
    [switch]$SkipFrontend,
    [switch]$SkipTauri,
    [switch]$WithEnhancer,
    [switch]$SkipBuildServer
)

$ErrorActionPreference = "Stop"

$ROOT = Split-Path $PSScriptRoot -Parent
$OUT = Join-Path $ROOT "out"
$SERVER_BIN = "magic-server.exe"
$MODELS_SRC = "C:\Users\Administrator\MagicMirror\models"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MagicMirror Packaging Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Frontend build
if (-not $SkipFrontend) {
    Write-Host "`n[1/5] Building frontend..." -ForegroundColor Yellow
    Push-Location $ROOT
    pnpm build
    if ($LASTEXITCODE -ne 0) { throw "Frontend build failed" }
    Pop-Location
    Write-Host "  [OK] dist/ generated" -ForegroundColor Green
} else {
    Write-Host "`n[1/5] Skipping frontend build" -ForegroundColor Yellow
}

# 2. Rust server build
if (-not $SkipBuildServer) {
    Write-Host "`n[2/5] Building Rust server..." -ForegroundColor Yellow
    Push-Location (Join-Path $ROOT "src-server")
    cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "Rust server build failed" }
    Pop-Location
    Write-Host "  [OK] magic-server.exe generated" -ForegroundColor Green
} else {
    Write-Host "`n[2/5] Skipping Rust server build" -ForegroundColor Yellow
}

# 3. Package server distribution
Write-Host "`n[3/5] Packaging server distribution..." -ForegroundColor Yellow
$serverExe = Join-Path $ROOT "src-server\target\release\$SERVER_BIN"
if (-not (Test-Path $serverExe)) {
    throw "server binary not found: $SERVER_BIN"
}

$serverDir = Join-Path $env:TEMP "MagicMirror-server-pkg"
if (Test-Path $serverDir) { Remove-Item $serverDir -Recurse -Force }
New-Item -ItemType Directory -Path $serverDir -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $serverDir "models") -Force | Out-Null

Copy-Item $serverExe (Join-Path $serverDir "server.exe") -Force

if (-not (Test-Path $MODELS_SRC)) {
    throw "Models directory not found: $MODELS_SRC"
}
$models = Get-ChildItem $MODELS_SRC -File | Where-Object {
    # Only include models needed by Rust server
    $_.Name -in @(
        "scrfd_2.5g.onnx",
        "arcface_w600k_r50.onnx",
        "inswapper_128_fp16.onnx",
        "inswapper_weight.bin",
        "gfpgan_1.4.onnx"
    )
}
if (-not $WithEnhancer) {
    $models = $models | Where-Object { $_.Name -notlike "*gfpgan*" }
    Write-Host "  (lite: no GFPGAN)" -ForegroundColor Yellow
}
foreach ($m in $models) {
    Copy-Item $m.FullName (Join-Path $serverDir "models") -Force
    Write-Host "  [OK] models/$($m.Name)" -ForegroundColor DarkGray
}

$bat = @"
@echo off
cd /d "%~dp0"
echo Starting MagicMirror Server on http://localhost:8023
start /b server.exe
echo Server started.
pause
taskkill /f /im server.exe >nul 2>&1
"@
Set-Content -Path (Join-Path $serverDir "start_server.bat") -Value $bat -Encoding ASCII

$zipName = if ($WithEnhancer) { "server-windows-x86_64-full.zip" } else { "server-windows-x86_64-lite.zip" }
$zipPath = Join-Path $OUT $zipName
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
# Compress contents only (no nested server/ folder)
Push-Location $serverDir
Compress-Archive -Path * -DestinationPath $zipPath -Force
Pop-Location
$sizeMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
Write-Host "  [OK] $zipPath ($sizeMB MB)" -ForegroundColor Green

# 4. Tauri installer
if (-not $SkipTauri) {
    Write-Host "`n[4/5] Building Tauri installer..." -ForegroundColor Yellow
    Push-Location $ROOT
    pnpm tauri build
    if ($LASTEXITCODE -ne 0) { throw "Tauri build failed" }
    Pop-Location

    $installer = Get-ChildItem (Join-Path $ROOT "src-tauri\target\release\bundle\nsis") -Filter "*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($installer) {
        Copy-Item $installer.FullName (Join-Path $OUT "MagicMirror_Setup.exe") -Force
        $sizeMB = [math]::Round($installer.Length / 1MB, 2)
        Write-Host "  [OK] out/MagicMirror_Setup.exe ($sizeMB MB)" -ForegroundColor Green
    }
} else {
    Write-Host "`n[4/5] Skipping Tauri installer build" -ForegroundColor Yellow
}

# 5. Full package (all-in-one, extract & run)
if (-not $SkipFrontend -or (Test-Path (Join-Path $ROOT "src-tauri\target\release\MagicMirror.exe"))) {
    Write-Host "`n[5/5] Building full package..." -ForegroundColor Yellow
    $fullDir = Join-Path $OUT "MagicMirror-Full"
    if (Test-Path $fullDir) { Remove-Item $fullDir -Recurse -Force }
    New-Item -ItemType Directory -Path $fullDir -Force | Out-Null

    # Tauri app binary
    $tauriExe = Join-Path $ROOT "src-tauri\target\release\MagicMirror.exe"
    if (Test-Path $tauriExe) {
        Copy-Item $tauriExe (Join-Path $fullDir "MagicMirror.exe") -Force
    }

    # Server binary + models
    Copy-Item $serverExe (Join-Path $fullDir "server.exe") -Force
    Copy-Item -Path (Join-Path $serverDir "models") -Destination $fullDir -Recurse -Force

    # Start scripts
    $startBat = @"
@echo off
cd /d "%~dp0"
echo Starting MagicMirror Server...
start /b server.exe
echo Launching MagicMirror...
start MagicMirror.exe
echo Done.
"@
    Set-Content -Path (Join-Path $fullDir "start.bat") -Value $startBat -Encoding ASCII

    $fullZip = Join-Path $OUT "MagicMirror-Full.zip"
    if (Test-Path $fullZip) { Remove-Item $fullZip -Force }
    Push-Location $fullDir
    Compress-Archive -Path * -DestinationPath $fullZip -Force
    Pop-Location
    $sizeMB = [math]::Round((Get-Item $fullZip).Length / 1MB, 2)
    Write-Host "  [OK] out/MagicMirror-Full.zip ($sizeMB MB)" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Packaging complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Artifacts:"
Get-ChildItem $OUT -File | ForEach-Object {
    $sizeMB = [math]::Round($_.Length / 1MB, 2)
    Write-Host "  - $($_.Name) ($sizeMB MB)"
}
