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


def test_decode_text_bytes_still_rejects_binary_garbage():
    raw = bytes(range(256)) * 4

    with pytest.raises(UnicodeDecodeError):
        decode_text_bytes(raw)
