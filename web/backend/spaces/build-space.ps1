<#
.SYNOPSIS
Assembles a Hugging Face Space from this repository.

.DESCRIPTION
Copies the API and the two files it reads into a cloned Space, preserving the
directory layout the code expects. That layout is not cosmetic: config.py finds
the model by walking up from its own location to the repository root, so the
backend has to sit at web/backend/app and the assets at app/src/main/assets
inside the Space exactly as they do here.

Deliberately copies the backend and two model files only. The landmarker
bundles, the expression model and the avatar motions are served by the frontend
and never opened by this service, and none of the Android application code is
involved - a Space has to be public for a browser to reach it, so what goes in
is worth being deliberate about.

.PARAMETER SpaceDir
The cloned Space working tree, e.g. ..\hf-space

.EXAMPLE
git clone https://huggingface.co/spaces/<user>/isuara-api ..\hf-space
.\build-space.ps1 -SpaceDir ..\..\..\..\hf-space
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$SpaceDir
)

$ErrorActionPreference = 'Stop'

$here     = Split-Path -Parent $MyInvocation.MyCommand.Path   # web/backend/spaces
$backend  = Split-Path -Parent $here                          # web/backend
$repoRoot = (Resolve-Path (Join-Path $backend '..\..')).Path  # repository root

if (-not (Test-Path $SpaceDir)) {
    throw "Space directory not found: $SpaceDir. Clone the Space first."
}
$SpaceDir = (Resolve-Path $SpaceDir).Path

Write-Host "repo:  $repoRoot"
Write-Host "space: $SpaceDir"
Write-Host ''

# source -> destination, both relative to their roots
$items = @(
    @{ From = 'web/backend/app';                               To = 'web/backend/app';                               Dir = $true  },
    @{ From = 'web/backend/requirements.txt';                  To = 'web/backend/requirements.txt';                  Dir = $false },
    @{ From = 'web/backend/Dockerfile';                        To = 'Dockerfile';                                    Dir = $false },
    @{ From = 'web/backend/spaces/README.md';                  To = 'README.md';                                     Dir = $false },
    @{ From = 'app/src/main/assets/bim_lstm_v312_fp16.tflite'; To = 'app/src/main/assets/bim_lstm_v312_fp16.tflite'; Dir = $false },
    @{ From = 'app/src/main/assets/label_map.json';            To = 'app/src/main/assets/label_map.json';            Dir = $false }
)

foreach ($item in $items) {
    $src = Join-Path $repoRoot $item.From
    $dst = Join-Path $SpaceDir $item.To
    if (-not (Test-Path $src)) { throw "missing: $src" }

    New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
    if ($item.Dir) {
        # Remove first, so a file deleted here is deleted in the Space too
        # rather than lingering from an earlier push.
        Remove-Item -Recurse -Force $dst -ErrorAction SilentlyContinue
        Copy-Item -Recurse $src $dst
        # __pycache__ would otherwise be pushed to a public repository.
        Get-ChildItem $dst -Recurse -Directory -Filter '__pycache__' |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    } else {
        Copy-Item -Force $src $dst
    }
    Write-Host ("  {0,-52} -> {1}" -f $item.From, $item.To)
}

# The Dockerfile builds from the repository root and expects that layout; inside
# the Space the Space root IS that root, so it works unchanged. Assert it, since
# a silent mismatch would only surface as a failed build minutes later.
$dockerfile = Get-Content (Join-Path $SpaceDir 'Dockerfile') -Raw
foreach ($needed in @('web/backend/requirements.txt', 'app/src/main/assets/bim_lstm_v312_fp16.tflite')) {
    if ($dockerfile -notmatch [regex]::Escape($needed)) {
        throw "Dockerfile no longer copies $needed - the Space layout and the Dockerfile have diverged."
    }
}

$size = (Get-ChildItem $SpaceDir -Recurse -File |
         Where-Object { $_.FullName -notmatch '\\\.git\\' } |
         Measure-Object Length -Sum).Sum
Write-Host ''
Write-Host ("staged {0:N2} MB" -f ($size / 1MB))
Write-Host ''
Write-Host 'Next:'
Write-Host "  cd `"$SpaceDir`""
Write-Host '  git add -A'
Write-Host '  git commit -m "Deploy API"'
Write-Host '  git push'
