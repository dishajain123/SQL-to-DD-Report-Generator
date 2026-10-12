$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = "."
New-Item -ItemType Directory -Force -Path output | Out-Null
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = (Get-Command python -ErrorAction Stop).Source
}
& $py (Join-Path $PSScriptRoot "_formula_check_run.py") *> (Join-Path $PSScriptRoot "output\formula_check_stdout.txt")
exit $LASTEXITCODE
