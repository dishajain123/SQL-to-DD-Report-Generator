import pytest

from app.utils.text_encoding import decode_text_bytes


def test_decode_text_bytes_accepts_utf8():
    decoded = decode_text_bytes("SELECT 1;".encode("utf-8"))

    assert decoded.text == "SELECT 1;"
    assert decoded.encoding in {"utf-8", "utf-8-sig"}


def test_decode_text_bytes_accepts_cp1252_sql():
    raw = "SELECT 'café' AS name;".encode("cp1252")

    decoded = decode_text_bytes(raw)

    assert decoded.text == "SELECT 'café' AS name;"
    assert decoded.encoding == "cp1252"


def test_decode_text_bytes_accepts_utf16_sql():
    raw = "SELECT 'hello' AS greeting;".encode("utf-16")

    decoded = decode_text_bytes(raw)

    assert decoded.text == "SELECT 'hello' AS greeting;"
    assert decoded.encoding == "utf-16"


def test_decode_text_bytes_accepts_text_heavy_with_unicode_en_spaces():
    # Reproduces a real upload: SQL pasted from SSMS with formatting that
    # substituted ASCII spaces for U+2002 EN SPACE throughout. Python's
    # str.isprintable() treats U+2002 as non-printable (it only special-cases
    # the ASCII 0x20 space among category Zs characters), which previously
    # pushed this file's printable ratio to ~0.84 -- just under the old 0.85
    # gate -- so a file that decoded successfully was still rejected.
    sql_with_en_spaces = ("SELECT * FROM dbo.Foo WHERE x = 1;\n" * 200)
    raw = sql_with_en_spaces.encode("utf-16")

    decoded = decode_text_bytes(raw)

    assert decoded.encoding == "utf-16"
    assert " " not in decoded.text
    assert "SELECT * FROM dbo.Foo WHERE x = 1;" in decoded.text


SSMS_SQL = "CREATE PROCEDURE PRO.X AS\r\nUPDATE A SET A.Flag = 'Y' FROM PRO.T A\r\n"


@pytest.mark.parametrize(
    "raw,encoding",
    [
        (SSMS_SQL.encode("utf-16-le"), "utf-16-le"),  # BOM-less "Unicode"
        (SSMS_SQL.encode("utf-16-be"), "utf-16-be"),
        (b"\xfe\xff" + SSMS_SQL.encode("utf-16-be"), "utf-16"),
        (b"\xef\xbb\xbf" + SSMS_SQL.encode("utf-8"), "utf-8-sig"),
    ],
)
def test_decode_text_bytes_detects_utf16_and_bom_variants(raw, encoding):
    decoded = decode_text_bytes(raw)
    assert decoded.text == SSMS_SQL
    assert decoded.encoding == encoding
    assert "\x00" not in decoded.text and "﻿" not in decoded.text


def test_read_sql_file_never_returns_utf16_noise_for_cp1252(tmp_path):
    """Regression: write_inventory_scan.read_sql_file tried utf-16 before
    latin-1, and UTF-16 accepts almost any even-length byte string."""
    from app.parsing.write_inventory_scan import read_sql_file

    path = tmp_path / "proc.sql"
    path.write_bytes("SELECT 'café' AS x;".encode("cp1252"))
    assert read_sql_file(path) == "SELECT 'café' AS x;"

    path.write_bytes(SSMS_SQL.encode("utf-16"))
    assert read_sql_file(path) == SSMS_SQL


def test_normalize_sql_text_repairs_utf16_decoded_as_latin1():
    from app.utils.text_encoding import normalize_sql_text

    mis_decoded = SSMS_SQL.encode("utf-16").decode("latin-1")
    assert "\x00" in mis_decoded
    assert normalize_sql_text(mis_decoded) == SSMS_SQL
    assert normalize_sql_text("﻿" + SSMS_SQL) == SSMS_SQL
    assert normalize_sql_text(SSMS_SQL) == SSMS_SQL


def test_decode_text_bytes_still_rejects_binary_garbage():
    raw = bytes(range(256)) * 4

    with pytest.raises(UnicodeDecodeError):
        decode_text_bytes(raw)
