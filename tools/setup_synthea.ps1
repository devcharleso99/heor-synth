param(
    # Restore the external/synthea working tree to match upstream (like a "validate/repair").
    # WARNING: This discards local changes inside external/synthea.
    [switch]$Repair,

    # Delete and re-clone the Synthea repository.
    # WARNING: This deletes external/synthea entirely.
    [switch]$Reclone
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ExternalDir = Join-Path $RepoRoot "external"
$SyntheaDir = Join-Path $ExternalDir "synthea"
$SyntheaRepoUrl = "https://github.com/synthetichealth/synthea.git"


function Invoke-CmdChecked([string]$CommandLine) {
    # Many tools (java, git, gradle) emit normal informational output on stderr.
    # With `$ErrorActionPreference = "Stop"` that can be surfaced as a terminating error.
    # Running through `cmd /c` with `2>&1` avoids that while still respecting exit codes.
    cmd /c "$CommandLine 2>&1"
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $CommandLine"
    }
}


function Get-OriginDefaultBranch([string]$RepoDir) {
    $refLine = (cmd /c "git -C ""$RepoDir"" symbolic-ref refs/remotes/origin/HEAD 2>&1" | Select-Object -First 1) -join ""
    if ($LASTEXITCODE -eq 0 -and $refLine -match "refs/remotes/origin/(?<branch>.+)$") {
        return $Matches["branch"]
    }

    # Fallbacks if origin/HEAD isn't set.
    cmd /c "git -C ""$RepoDir"" show-ref --verify --quiet refs/remotes/origin/main 1>nul 2>nul"
    if ($LASTEXITCODE -eq 0) { return "main" }

    cmd /c "git -C ""$RepoDir"" show-ref --verify --quiet refs/remotes/origin/master 1>nul 2>nul"
    if ($LASTEXITCODE -eq 0) { return "master" }

    return "master"
}

Write-Host ""
Write-Host "HEOR Synth - Synthea setup"
Write-Host "Repo root: $RepoRoot"
Write-Host "Synthea dir: $SyntheaDir"
Write-Host ""

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git was not found. Install Git first, then rerun this script."
}

if (-not (Get-Command java -ErrorAction SilentlyContinue)) {
    Write-Host "Java was not found."
    Write-Host "Install a Java JDK 17+ first. Recommended command:"
    Write-Host "winget install EclipseAdoptium.Temurin.17.JDK"
    throw "Missing Java JDK 17+."
}

# NOTE: `java -version` writes to stderr. With `$ErrorActionPreference = "Stop"`,
# PowerShell can surface that stderr as a terminating NativeCommandError.
# Use `cmd /c` so we can capture the version line without triggering an error.
$javaVersionLine = (cmd /c "java -version 2>&1" | Select-Object -First 1) -join ""

if ($javaVersionLine -match '"(?<major>\d+)') {
    $majorVersion = [int]$Matches["major"]
} else {
    throw "Could not parse Java version from: $javaVersionLine"
}

if ($majorVersion -lt 17) {
    throw "Java JDK 17+ is required. Found: $javaVersionLine"
}

Write-Host "Java check passed: $javaVersionLine"

New-Item -ItemType Directory -Force $ExternalDir | Out-Null

if ($Reclone -and (Test-Path $SyntheaDir)) {
    Write-Host "Reclone requested. Removing existing Synthea directory..."
    Remove-Item $SyntheaDir -Recurse -Force
}

if ((Test-Path $SyntheaDir) -and (-not (Test-Path (Join-Path $SyntheaDir ".git")))) {
    throw "Synthea directory exists but is not a git repository: $SyntheaDir`nDelete it and rerun, or rerun with -Reclone."
}

if (-not (Test-Path (Join-Path $SyntheaDir ".git"))) {
    Write-Host "Cloning Synthea..."
    Invoke-CmdChecked "git clone $SyntheaRepoUrl ""$SyntheaDir"""
} else {
    if ($Repair) {
        Write-Host "Synthea repo already exists. Repair mode enabled (like a validate/repair)."
        Write-Host "WARNING: This will discard local changes in: $SyntheaDir"

        Invoke-CmdChecked "git -C ""$SyntheaDir"" fetch --all --prune"
        $defaultBranch = Get-OriginDefaultBranch $SyntheaDir

        # Force the working tree to match the upstream default branch.
        Invoke-CmdChecked "git -C ""$SyntheaDir"" checkout -B $defaultBranch origin/$defaultBranch"
        Invoke-CmdChecked "git -C ""$SyntheaDir"" reset --hard origin/$defaultBranch"
        Invoke-CmdChecked "git -C ""$SyntheaDir"" clean -fd"
    } else {
        # "Smart" behavior: don't reclone. Only fast-forward pull if the repo is clean.
        $dirty = cmd /c "git -C ""$SyntheaDir"" status --porcelain 2>&1"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not check Synthea repo status: $SyntheaDir"
        }

        if ($dirty) {
            Write-Host "Synthea repo already exists, but has local changes. Skipping git pull."
            Write-Host "To restore to upstream (discarding local changes), rerun with: .\tools\setup_synthea.ps1 -Repair"
        } else {
            Write-Host "Synthea repo already exists. Pulling latest changes..."
            Invoke-CmdChecked "git -C ""$SyntheaDir"" pull --ff-only"
        }
    }
}

$requiredSyntheaFiles = @(
    "gradlew.bat",
    "run_synthea.bat"
)

$missingSyntheaFiles = @()
foreach ($f in $requiredSyntheaFiles) {
    if (-not (Test-Path (Join-Path $SyntheaDir $f))) {
        $missingSyntheaFiles += $f
    }
}

if ($missingSyntheaFiles.Count -gt 0) {
    throw "Synthea repo appears incomplete (missing: $($missingSyntheaFiles -join ', ')).`nTry rerunning with -Repair, or delete external/synthea and rerun."
}

Write-Host ""
Write-Host "Building Synthea..."
Push-Location $SyntheaDir
Invoke-CmdChecked ".\\gradlew.bat build -x test"
Pop-Location

Write-Host ""
Write-Host "Synthea setup complete."