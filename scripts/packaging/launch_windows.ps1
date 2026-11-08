$ErrorActionPreference = 'Stop'
try {
    $cache = Join-Path $env:LOCALAPPDATA 'Batch Deconvolution'
    New-Item -ItemType Directory -Force $cache | Out-Null
    $runtime = Join-Path $cache 'runtime-@BUILD_ID@'
    $env:CONDA_PREFIX = $runtime
    $env:PATH = "$runtime;$runtime\Library\bin;$runtime\Scripts;$env:PATH"
    Remove-Item Env:PYTHONHOME, Env:PYTHONPATH -ErrorAction SilentlyContinue
    if (-not (Test-Path "$runtime\.ready")) {
        $lock = [System.IO.File]::Open("$runtime.lock", 'OpenOrCreate', 'ReadWrite', 'None')
        try {
            if (-not (Test-Path "$runtime\.ready")) {
                Expand-Archive -LiteralPath "$PSScriptRoot\runtime.zip" -DestinationPath $runtime -Force
                & "$runtime\python.exe" "$runtime\Scripts\conda-unpack-script.py"
                if ($LASTEXITCODE -ne 0) { throw 'Runtime setup failed.' }
                & "$runtime\@JAVA_HOME@\bin\java.exe" -version
                if ($LASTEXITCODE -ne 0) { throw 'Bundled Java setup failed.' }
                New-Item -ItemType File "$runtime\.ready" | Out-Null
            }
        } finally { $lock.Dispose() }
    }
    $env:JAVA_HOME = "$runtime\@JAVA_HOME@"
    $env:BFF_JAVA_VERSION = '21'
    $env:BFF_JAVA_FETCH = 'never'
    $env:BIOFORMATS_VERSION = '8.5.0'
    $env:BATCH_DECONVOLUTION_WORKSPACE = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'Batch Deconvolution'
    Set-Location "$PSScriptRoot\app"
    & "$runtime\python.exe" -u main.py @args
    if ($LASTEXITCODE -ne 0) { throw "Application exited with code $LASTEXITCODE." }
} catch {
    Write-Host $_ -ForegroundColor Red
    Read-Host 'Press Enter to close'
    exit 1
}
