"""Native set aggregates allowed in generated 4X (subset of app.grammar.platform_library)."""
from __future__ import annotations

# When True, MIN/MAX/SUM/COUNT compile as native 4X function calls
# instead of being lowered to row-level IF trees.
NATIVE_AGGREGATES_ENABLED = True

# Documented native set aggregates in 4X (MIN/MAX optional group list; SUM/COUNT single arg).
PLATFORM_AGGREGATE_FUNCTIONS = frozenset({"MIN", "MAX", "SUM", "COUNT"})

# T-SQL aggregates with no documented 4X equivalent — must not be silently lowered.
UNSUPPORTED_PLATFORM_AGGREGATES = frozenset({"STRING_AGG"})

STRING_AGG_UNSUPPORTED_MESSAGE = (
    "STRING_AGG has no documented native 4X equivalent "
    "(see samples/platform_docs/4x_functions_operators.md); "
    "requires manual review or staging-table passthrough"
)
