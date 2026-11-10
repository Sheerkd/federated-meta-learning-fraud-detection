param([string]$Environment = ".venv")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($Environment -notmatch '^\.venv(?:-[a-zA-Z0-9]+)?$') { throw "Invalid environment name." }
$projectPython = Join-Path (Get-Location) "$Environment\Scripts\python.exe"
& $projectPython -m fraud_detection dashboard
if ($LASTEXITCODE -ne 0) { throw "Dashboard exited with an error." }
