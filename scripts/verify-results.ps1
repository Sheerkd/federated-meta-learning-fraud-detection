param([string]$Environment = ".venv")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($Environment -notmatch '^\.venv(?:-[a-zA-Z0-9]+)?$') { throw "Invalid environment name." }
$projectPython = Join-Path (Get-Location) "$Environment\Scripts\python.exe"
foreach ($command in @("verify-data", "verify-study", "verify-study-sources")) {
    & $projectPython -m fraud_detection $command
    if ($LASTEXITCODE -ne 0) { throw "Saved-result verification failed: $command" }
}
Write-Host "Saved dataset, study, source runs, and explanations verified."
