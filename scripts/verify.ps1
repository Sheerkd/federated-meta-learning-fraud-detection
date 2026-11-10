param([string]$Environment = ".venv")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($Environment -notmatch '^\.venv(?:-[a-zA-Z0-9]+)?$') { throw "Invalid environment name." }
$projectPython = Join-Path (Get-Location) "$Environment\Scripts\python.exe"
& $projectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed." }
& $projectPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
& $projectPython -m fraud_detection doctor
if ($LASTEXITCODE -ne 0) { throw "Functional environment check failed." }
& $projectPython -m fraud_detection flower-smoke
if ($LASTEXITCODE -ne 0) { throw "Flower process check failed." }
