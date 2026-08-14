# MagicMirror Face Swap Script
# Usage: .\face-swap.ps1
# Requires: Server running on http://localhost:8023

param(
    [string]$OutputDir = ".",
    [switch]$WaitForServer
)

$ErrorActionPreference = "Stop"

$ServerUrl = "http://localhost:8023"

Write-Host "=== MagicMirror Face Swap ===" -ForegroundColor Cyan
Write-Host ""

# Check server status
Write-Host "Checking server..." -ForegroundColor Yellow
try {
    $status = Invoke-RestMethod -Uri "$ServerUrl/status" -TimeoutSec 5
    Write-Host "  Server is running: $($status.status)" -ForegroundColor Green
} catch {
    Write-Host "  Server not responding at $ServerUrl" -ForegroundColor Red
    Write-Host "  Please start the server first:" -ForegroundColor Yellow
    Write-Host "    cd F:\MagicMirror\src-server" -ForegroundColor Gray
    Write-Host "    .\target\release\magic-server.exe" -ForegroundColor Gray
    if ($WaitForServer) {
        Write-Host "  Waiting for server..." -ForegroundColor Yellow
        $maxRetries = 30
        for ($i = 0; $i -lt $maxRetries; $i++) {
            Start-Sleep -Seconds 1
            try {
                $status = Invoke-RestMethod -Uri "$ServerUrl/status" -TimeoutSec 2
                Write-Host "  Server is now running!" -ForegroundColor Green
                break
            } catch {
                if ($i -eq $maxRetries - 1) {
                    Write-Host "  Timed out waiting for server" -ForegroundColor Red
                    exit 1
                }
            }
        }
    } else {
        exit 1
    }
}

# Find a* and b* images
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($OutputDir -eq ".") { $OutputDir = $ScriptDir }

Write-Host ""
Write-Host "Looking for images in: $OutputDir" -ForegroundColor Yellow

$aImages = @()
$bImages = @()

# Find all matching images (jpg and png)
$aJpg = Get-ChildItem -Path $OutputDir -Filter "a*.jpg" -ErrorAction SilentlyContinue | Where-Object { -not $_.PSIsContainer }
$aPng = Get-ChildItem -Path $OutputDir -Filter "a*.png" -ErrorAction SilentlyContinue | Where-Object { -not $_.PSIsContainer }
$bJpg = Get-ChildItem -Path $OutputDir -Filter "b*.jpg" -ErrorAction SilentlyContinue | Where-Object { -not $_.PSIsContainer }
$bPng = Get-ChildItem -Path $OutputDir -Filter "b*.png" -ErrorAction SilentlyContinue | Where-Object { -not $_.PSIsContainer }

if ($aJpg) { $aImages = $aImages + $aJpg }
if ($aPng) { $aImages = $aImages + $aPng }
if ($bJpg) { $bImages = $bImages + $bJpg }
if ($bPng) { $bImages = $bImages + $bPng }

if (-not $aImages) {
    Write-Host "ERROR: No images starting with 'a' found in $OutputDir" -ForegroundColor Red
    Write-Host "  Example: a_face.jpg" -ForegroundColor Gray
    exit 1
}

if (-not $bImages) {
    Write-Host "ERROR: No images starting with 'b' found in $OutputDir" -ForegroundColor Red
    Write-Host "  Example: b_face.jpg" -ForegroundColor Gray
    exit 1
}

Write-Host "Found $($aImages.Count) source image(s) (a*)" -ForegroundColor Green
Write-Host "Found $($bImages.Count) target image(s) (b*)" -ForegroundColor Green
Write-Host ""

# Prepare models
Write-Host "Preparing models..." -ForegroundColor Yellow
try {
    $prepare = Invoke-RestMethod -Uri "$ServerUrl/prepare" -Method POST -TimeoutSec 60
    if (-not $prepare.success) {
        Write-Host "  Model preparation failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Models loaded successfully" -ForegroundColor Green
} catch {
    Write-Host "  Error preparing models: $_" -ForegroundColor Red
    exit 1
}

# Process each combination
$processed = 0
foreach ($aImg in $aImages) {
    foreach ($bImg in $bImages) {
        $aBase = [System.IO.Path]::GetFileNameWithoutExtension($aImg.Name)
        $bBase = [System.IO.Path]::GetFileNameWithoutExtension($bImg.Name)
        $outputName = "c_${aBase}_to_${bBase}.jpg"
        $outputPath = Join-Path $OutputDir $outputName
        
        Write-Host "Processing: $($aImg.Name) -> $($bImg.Name)" -ForegroundColor Cyan
        Write-Host "  Output: $outputPath" -ForegroundColor Gray
        
        $body = @{
            id = "swap-$processed"
            inputImage = $aImg.FullName
            targetFace = $bImg.FullName
        } | ConvertTo-Json
        
        try {
            $result = Invoke-RestMethod -Uri "$ServerUrl/task" -Method POST -Body $body -ContentType "application/json; charset=utf-8" -TimeoutSec 120
            
            # Resolve relative path - server writes to its own directory
            $fullPath = $result.result
            # Try: script dir, server dir, release dir, current location
            $searchPaths = @(
                $OutputDir,
                "F:\MagicMirror\src-server",
                "F:\MagicMirror\src-server\target\release",
                (Get-Location).Path
            )
            foreach ($path in $searchPaths) {
                $candidate = Join-Path $path $result.result
                if (Test-Path $candidate) {
                    $fullPath = $candidate
                    break
                }
            }
            
            # Check if output file exists
            if (Test-Path $fullPath) {
                Copy-Item $fullPath $outputPath -Force
                Write-Host "  Success: $outputPath" -ForegroundColor Green
                $processed++
            } else {
                Write-Host "  Warning: Output file not found at $fullPath" -ForegroundColor Yellow
                Write-Host "  Server response: $result" -ForegroundColor Gray
            }
        } catch {
            Write-Host "  Error: $_" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "=== Complete ===" -ForegroundColor Cyan
Write-Host "Processed: $processed swap(s)" -ForegroundColor Green
Write-Host "Output directory: $OutputDir" -ForegroundColor Green
Write-Host ""
Write-Host "Output files start with 'c_' prefix" -ForegroundColor Yellow
