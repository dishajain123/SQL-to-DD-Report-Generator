"""Backward-compatible wrapper for the DD CSV export module.

The canonical implementation lives in :mod:`app.report.dd_export`. This
module stays in place so older imports and tests continue to work while
the codebase uses the clearer, behavior-matching name internally.
"""
from __future__ import annotations

from app.report.dd_export import (
    COLUMNS,
    dd_row_to_dict,
    export_dd_rows,
    export_dd_rows_csv,
    export_reviewed_dd_rows_for_job,
    export_reviewed_dd_rows_for_job_csv,
    merge_dd_rows,
    read_existing_dd_csv,
    read_existing_dd_excel,
)

__all__ = [
    "COLUMNS",
    "dd_row_to_dict",
    "export_dd_rows",
    "export_dd_rows_csv",
    "export_reviewed_dd_rows_for_job",
    "export_reviewed_dd_rows_for_job_csv",
    "merge_dd_rows",
    "read_existing_dd_csv",
    "read_existing_dd_excel",
]
