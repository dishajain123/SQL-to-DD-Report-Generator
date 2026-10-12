$ErrorActionPreference = "Continue"
$root = "C:\Users\dishaj\Downloads\Projects\SQL-to-DD-Report-Generator"
Set-Location $root
$env:PYTHONPATH = "."

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $candidates = @(
        (Get-Command py -ErrorAction SilentlyContinue),
        (Get-Command python -ErrorAction SilentlyContinue)
    )
    foreach ($c in $candidates) {
        if ($c) {
            $py = "py"
            if ($c.Name -eq "py") { $pyArgs = @("-3") } else { $pyArgs = @() }
            break
        }
    }
    if (-not $py -or $py -eq "py") {
        # resolve py -3
        $resolved = & py -3 -c "import sys; print(sys.executable)" 2>&1
        if ($LASTEXITCODE -eq 0) { $py = ($resolved | Select-Object -Last 1).ToString().Trim() }
    }
}

$meta = @()
$meta += "PYTHON: $py"
$meta += "EXISTS: $(Test-Path $py)"

function Run-Capture($argsList, $outFile) {
    $metaPath = $outFile -replace '\.txt$', '_meta.txt'
    if ($py -eq "py" -or (Test-Path $py)) {
        if ($py -eq "py") {
            & py -3 @argsList *> $outFile
        } else {
            & $py @argsList *> $outFile
        }
        $exit = $LASTEXITCODE
    } else {
        "No python found" | Out-File -FilePath $outFile -Encoding utf8
        $exit = 127
    }
    "exit=$exit" | Out-File -FilePath $metaPath -Encoding utf8
    return $exit
}

Run-Capture @("-m", "pytest", "tests\unit\derivation_v2\", "-q", "--tb=short") (Join-Path $root "_pytest_capture.txt")
Run-Capture @("scripts\audit_pro_sp_sequenced.py") (Join-Path $root "_audit_capture.txt")

$meta | Out-File (Join-Path $root "_capture_run_meta.txt") -Encoding utf8
