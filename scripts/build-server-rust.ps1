param(
    [switch]$SkipEnhancer  # Skip gfpgan model to reduce size
)

$ErrorActionPreference = "Stop"

$ROOT = Split-Path $PSScriptRoot -Parent
$MODELS_SRC = "C:\Users\Administrator\MagicMirror\models"
$OUT_DIR = Join-Path $ROOT "out"

Write-Host "Building Rust server..." -ForegroundColor Cyan

# Build Rust server
Set-Location (Join-Path $ROOT "src-server")
cargo build --release
if ($LASTEXITCODE -ne 0) { throw "Rust build failed" }
Set-Location $ROOT

Write-Host "Packaging server..." -ForegroundColor Cyan

# Create output directory
New-Item -ItemType Directory -Path (Join-Path $OUT_DIR "server") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $OUT_DIR "server\models") -Force | Out-Null

# Copy Rust binary
Copy-Item (Join-Path $ROOT "src-server\target\release\magic-server.exe") (Join-Path $OUT_DIR "server\server.exe") -Force

# Copy models (only those needed by Rust server)
$models = Get-ChildItem $MODELS_SRC -File | Where-Object {
    $_.Name -in @(
        "scrfd_2.5g.onnx",
        "arcface_w600k_r50.onnx",
        "inswapper_128_fp16.onnx",
        "inswapper_weight.bin",
        "gfpgan_1.4.onnx"
    )
}
if ($SkipEnhancer) {
    $models = $models | Where-Object { $_.Name -notlike "*gfpgan*" }
    Write-Host "Skipping enhancer model (gfpgan)" -ForegroundColor Yellow
}
foreach ($m in $models) {
    Copy-Item $m.FullName (Join-Path $OUT_DIR "server\models") -Force
}

# Create zip
$zipName = "server-windows-x86_64.zip"
if ($SkipEnhancer) {
    $zipName = "server-windows-x86_64-lite.zip"
}
Compress-Archive -Path (Join-Path $OUT_DIR "server") -DestinationPath (Join-Path $OUT_DIR $zipName) -Force

$sizeMB = [math]::Round((Get-Item (Join-Path $OUT_DIR $zipName)).Length / 1MB, 2)
Write-Host "Done: $zipName ($sizeMB MB)" -ForegroundColor Green
