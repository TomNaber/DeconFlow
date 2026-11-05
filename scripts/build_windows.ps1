$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    $python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
    & $python scripts/build_deliverable.py --platform windows @args
    if ($LASTEXITCODE -ne 0) { throw "Build failed (exit code $LASTEXITCODE)." }
} finally {
    Pop-Location
}
