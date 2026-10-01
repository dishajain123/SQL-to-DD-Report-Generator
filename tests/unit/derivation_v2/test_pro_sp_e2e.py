"""End-to-end offline generation for sequenced PRO stored procedures (local paths)."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.generate_sample_dd_demo import generate_sample

PRO_SP_ROOT = Path(
    r"C:\Users\dishaj\Downloads\PRO_SPs_Sequenced\PRO_SPs_Sequenced"
)
PRO_SP_FILES = [
    "01_S00_PRO.InsertDataforAssetClassficationRBL.StoredProcedure.sql",
    "06_S01_PRO.Reference_Period_Calculation.StoredProcedure.sql",
    "07_S02_PRO.DPD_Calculation.StoredProcedure.sql",
]


@pytest.mark.parametrize("filename", PRO_SP_FILES)
def test_pro_sequenced_sp_generates_dd_candidates(filename: str, tmp_path: Path) -> None:
    path = PRO_SP_ROOT / filename
    if not path.is_file():
        pytest.skip(f"Local SP not present: {path}")
    summary = generate_sample(path, tmp_path / "out")
    assert summary["rows"] > 0
    assert summary["inventory_complete"]
    assert summary["writes"] > 0
    dd_json = tmp_path / "out" / path.stem / "dd_rows.json"
    assert dd_json.is_file()
    assert len(dd_json.read_text(encoding="utf-8")) > 10
