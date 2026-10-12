# Golden SP snippet tests

Run from the **repository root** (where `pytest.ini` lives):

```powershell
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m pytest tests\golden\test_pro_sps_snippets.py -v
```

Collect only (expect **9** tests):

```powershell
.\.venv\Scripts\python.exe -m pytest tests\golden\test_pro_sps_snippets.py --collect-only -q
```

If you see **`collected 0 items`**, the path or working directory is wrong, or the file
is missing from your tree. If you see **`FileNotFoundError`** under `%TEMP%` or `/tmp`
during teardown, use output capture that avoids fd temp files:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\golden\test_pro_sps_snippets.py -v -s
# or
.\.venv\Scripts\python.exe -m pytest tests\golden\test_pro_sps_snippets.py -v --capture=tee-sys
```

Avoid mixing **WSL paths** with **Windows `.venv\Scripts\python.exe`** (or the reverse);
use one environment consistently.
