"""Scanners must stay linear on large production procedures.

Regression: every keyword test did ``re.match(p, text[i:])`` inside
per-character loops, copying the rest of the file on each call (O(n²)). A
~150 KB procedure then sat in "Generating derivation rows" indefinitely.
"""
import time

from app.derivation.v2.phase1_lineage import build_lineage_map
from app.derivation.v2.phase2_mutation_folder import fold_column_mutations
from app.derivation.v2.sql_text import (
    extract_if_else_chains,
    extract_insert_select,
    extract_select_into,
    extract_update_statements,
)


def _large_procedure(blocks: int = 400) -> str:
    parts = ["CREATE PROCEDURE PRO.Big @TIMEKEY INT AS\nBEGIN\nBEGIN TRY\n"]
    for n in range(blocks):
        parts.append(
            f"/* block {n} */\n"
            f"IF @TIMEKEY > {26000 + n}\nBEGIN\n"
            f"  UPDATE A SET A.Col{n % 40} = CASE WHEN B.Flag = 'Y' THEN 'V{n}' ELSE A.Col{n % 40} END\n"
            f"  FROM ##ACCOUNTCAL A INNER JOIN dbo.AdvAcBasicDetail B ON A.AccountEntityID = B.AccountEntityID\n"
            f"  WHERE B.EffectiveFromTimeKey <= @TIMEKEY AND ISNULL(B.Balance, 0) > 0\n"
            f"END\n"
            f"INSERT INTO PRO.ProcessMonitor(UserID, Description) SELECT ORIGINAL_LOGIN(), 'STEP {n}'\n"
        )
    parts.append("END TRY\nBEGIN CATCH\n  SELECT 1\nEND CATCH\nEND\n")
    return "".join(parts)


def test_scanners_are_linear_on_a_large_procedure():
    sql = _large_procedure()
    assert len(sql) > 100_000
    started = time.perf_counter()
    assert len(extract_update_statements(sql)) == 400
    assert len([b for b in extract_if_else_chains(sql) if b.kind == "IF"]) == 400
    assert len(extract_insert_select(sql)) == 400
    extract_select_into(sql)
    lineage = build_lineage_map(sql)
    for column in ("Col0", "Col1", "Col2"):
        assert len(fold_column_mutations(sql, "##ACCOUNTCAL", column, lineage)) == 10
    # The quadratic version took minutes here; linear code takes well under
    # a few seconds even on a slow filesystem / CI box.
    assert time.perf_counter() - started < 30


def test_select_into_is_not_paired_with_a_later_insert_into():
    sql = (
        "SELECT A.Id FROM PRO.Acct A WHERE A.Flag = 'Y'\n"
        "INSERT INTO #Stage (Id) SELECT Id FROM PRO.Acct\n"
        "SELECT Id, Flag INTO #Real FROM PRO.Acct WHERE Flag = 'N'\n"
    )
    targets = [r["target"] for r in extract_select_into(sql)]
    assert targets == ["#Real"]


def test_cached_scanner_results_are_isolated_from_callers():
    sql = _large_procedure(60)
    first = extract_update_statements(sql)
    first[0]["set_clause"] = "tampered"
    first.clear()
    again = extract_update_statements(sql)
    assert len(again) == 60
    assert again[0]["set_clause"] != "tampered"
