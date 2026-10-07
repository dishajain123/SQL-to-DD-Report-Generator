"""Print every folded mutation for one target column and flag fields that look like SQL clauses.

Usage (from the repo root):
    python scripts/debug_mutations.py samples/sql/PRO_SPs_Sequenced/19_S11_PRO.SMA_MARKING.StoredProcedure.sql CUSTOMERCAL SMA_DT

A field that starts with ``FROM`` or contains ``JOIN`` is not an expression; the line marked
``<<< SUSPECT`` names the exact MutationPass field (assigned_expression / where_clause /
outer_condition / join_filter_condition) that carries the stray clause.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from app.derivation.v2.phase1_lineage import build_lineage_map
from app.derivation.v2.phase2_mutation_folder import fold_column_mutations
from app.parsing.write_inventory_scan import read_sql_file

_FIELDS = ("assigned_expression", "where_clause", "outer_condition", "join_filter_condition")
_SUSPECT = re.compile(r"(?is)^\s*FROM\b|\bJOIN\b")


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    path, entity, column = Path(argv[1]), argv[2], argv[3]
    sql = read_sql_file(path)
    lineage = build_lineage_map(sql, None)
    mutations = fold_column_mutations(sql, entity, column, lineage, None)
    print(f"{len(mutations)} mutation(s) for {entity}.{column}")
    for m in mutations:
        print(f"\n#{m.ordinal} stmt={m.statement_index} op={m.operation} pos={m.source_position}")
        for field in _FIELDS:
            value = getattr(m, field, None)
            if not value:
                continue
            flag = "   <<< SUSPECT" if _SUSPECT.search(str(value)) else ""
            print(f"  {field}: {str(value)[:300]!r}{flag}")
        print(f"  raw_sql: {(m.raw_sql or '')[:160]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
