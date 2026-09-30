"""Batch CLI: mixed encodings in a folder or zip run end to end."""
import json
import zipfile
from pathlib import Path

from app.batch import main
from app.utils import db

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = (ROOT / "samples" / "sql" / "07_DPD_Bucket_Classification.sql").read_text(encoding="utf-8")


def test_batch_runs_utf16_and_cp1252_files(tmp_path, monkeypatch):
    # main() points db.settings at the output dir; restore it after the test.
    monkeypatch.setattr(db, "settings", db.settings)
    src = tmp_path / "in"
    src.mkdir()
    (src / "unicode_le.sql").write_bytes(SAMPLE.encode("utf-16"))  # SSMS "Unicode"
    (src / "ansi.sql").write_bytes(SAMPLE.replace("DESCRIPTION", "DESCRIPCIÓN", 1).encode("cp1252"))

    out = tmp_path / "dist"
    assert main(["--input-dir", str(src), "--output-dir", str(out)]) == 0

    results = {r["file"]: r for r in json.loads((out / "batch_summary.json").read_text(encoding="utf-8"))}
    assert results["unicode_le.sql"]["encoding"] == "utf-16"
    assert results["ansi.sql"]["encoding"] == "cp1252"
    for result in results.values():
        assert not result["error"]
        assert result["rows"] > 0
        assert result["parse_failures"] == []
        assert (Path(result["output_dir"]) / "dd_export.csv").exists()
    assert (out / "batch_summary.md").exists()


def test_batch_reads_a_zip_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "settings", db.settings)
    archive = tmp_path / "PRO_SPs.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("PRO_SPs/proc.sql", SAMPLE.encode("utf-16-le"))  # no BOM
        zf.writestr("PRO_SPs/readme.txt", b"not sql")

    out = tmp_path / "dist"
    assert main(["--input-dir", str(archive), "--output-dir", str(out)]) == 0
    results = json.loads((out / "batch_summary.json").read_text(encoding="utf-8"))
    assert [r["file"] for r in results] == ["PRO_SPs/proc.sql"]
    assert results[0]["encoding"] == "utf-16-le"
