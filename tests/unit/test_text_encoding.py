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
