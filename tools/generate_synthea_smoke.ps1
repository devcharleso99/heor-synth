$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$SyntheaDir = Join-Path $RepoRoot "external\synthea"
$RawDest = Join-Path $RepoRoot "data\raw\synthea\smoke"
$SyntheaOutput = Join-Path $SyntheaDir "output"
$SyntheaCsvOutput = Join-Path $SyntheaOutput "csv"

$Seed = "20260601"
$ReferenceDate = "20260601"
$PopulationSize = "100"

Write-Host ""
Write-Host "HEOR Synth - Generate Synthea smoke dataset"
Write-Host "Repo root: $RepoRoot"
Write-Host "Synthea dir: $SyntheaDir"
Write-Host "Raw destination: $RawDest"
Write-Host ""

if (-not (Test-Path (Join-Path $SyntheaDir "run_synthea.bat"))) {
    throw "Synthea is not set up yet. Run .\tools\setup_synthea.ps1 first."
}

New-Item -ItemType Directory -Force $RawDest | Out-Null

Write-Host "Cleaning previous Synthea output..."
if (Test-Path $SyntheaOutput) {
    Remove-Item $SyntheaOutput -Recurse -Force
}

Write-Host "Cleaning previous smoke CSV files..."
Get-ChildItem $RawDest -Filter *.csv -ErrorAction SilentlyContinue | Remove-Item -Force

Write-Host ""
Write-Host "Generating smoke dataset..."
Push-Location $SyntheaDir

.\run_synthea.bat `
    -s $Seed `
    -r $ReferenceDate `
    -p $PopulationSize `
    --exporter.csv.export=true `
    --exporter.fhir.export=false `
    --exporter.ccda.export=false `
    --exporter.text.export=false

Pop-Location

if (-not (Test-Path $SyntheaCsvOutput)) {
    throw "Expected Synthea CSV output folder not found: $SyntheaCsvOutput"
}

Write-Host ""
Write-Host "Copying CSV files into HEOR Synth raw folder..."
Copy-Item -Path (Join-Path $SyntheaCsvOutput "*.csv") -Destination $RawDest -Force

Write-Host ""
Write-Host "Smoke CSV files copied:"
Get-ChildItem $RawDest -Filter *.csv | Select-Object Name, Length

Write-Host ""
Write-Host "Running raw inventory against smoke scenario..."
Push-Location $RepoRoot
uv run python .\src\cli\main.py raw-inventory --scenario .\configs\scenarios\smoke_synthea.yaml --fail-on-missing
Pop-Location

Write-Host ""
Write-Host "Synthea smoke dataset generation complete."