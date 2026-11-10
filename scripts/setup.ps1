param([string]$Environment = ".venv")
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
if ([IO.Path]::IsPathRooted($Environment) -or $Environment -notmatch '^\.venv(?:-[a-zA-Z0-9]+)?$') {
    throw "Environment must be .venv or .venv-name inside the project."
}
if (-not (Test-Path "$Environment\Scripts\python.exe")) {
    py -3.13 -m venv $Environment
    if ($LASTEXITCODE -ne 0) { throw "Python 3.13 environment creation failed." }
}
$projectPython = Join-Path $projectRoot "$Environment\Scripts\python.exe"
& $projectPython -m pip install --disable-pip-version-check -r requirements.lock
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& $projectPython -m pip install --no-deps --no-build-isolation -e .
if ($LASTEXITCODE -ne 0) { throw "Application installation failed." }
& $projectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed." }
& $projectPython -m fraud_detection init
if ($LASTEXITCODE -ne 0) { throw "Storage initialization failed." }
Write-Host "Setup complete. Run scripts\verify.ps1 next."
