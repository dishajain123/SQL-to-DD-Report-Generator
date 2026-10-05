"""Print how each UPDATE's FROM clause (and derived tables) is parsed.

    python scripts/_debug_derived_alias.py <path-to-sp.sql>
"""
from __future__ import annotations

import sys
from pathlib import Path

from app.derivation.v2.phase1_lineage import build_lineage_map
from app.derivation.v2.phase2_mutation_folder import (
    DerivedTable,
    _mask_derived_tables,
    _parse_update_sources,
)
from app.derivation.v2.sql_text import extract_update_statements

from app.utils.text_encoding import normalize_sql_text, read_sql_file

# The SP files are UTF-16; decode with the same helper the pipeline uses.
sql = normalize_sql_text(read_sql_file(sys.argv[1]))
lineage = build_lineage_map(sql)
for idx, stmt in enumerate(extract_update_statements(sql)):
    from_clause = stmt.get("from_clause") or ""
    if "(" not in from_clause or "SELECT" not in from_clause.upper():
        continue
    print(f"=== stmt #{idx} head={stmt['head']!r}")
    print("SET  :", " ".join((stmt["set_clause"] or "").split())[:200])
    masked, sources = _mask_derived_tables(from_clause)
    print("MASK :", " ".join(masked.split())[:200], "| derived:", list(sources))
    alias_map, joins = _parse_update_sources(stmt["head"], from_clause, lineage, None)
    for key, val in alias_map.items():
        extra = (
            f" projections={list(val.projections)} inner={dict(val.inner_alias_map)}"
            if isinstance(val, DerivedTable)
            else ""
        )
        print(f"  alias {key!r} -> {str(val)!r}{extra}")
