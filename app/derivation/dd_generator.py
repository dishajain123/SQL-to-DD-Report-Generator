"""Compatibility facade for the DD generation engine.

The full implementation now lives in
`app.derivation.dd_generation_engine`, which keeps the sprawling
derivation logic in a dedicated module while preserving the historical
import path used by the app and tests.
"""
from __future__ import annotations

from app.derivation import dd_generation_engine as _dd_generation_engine

globals().update(
    {
        name: value
        for name, value in _dd_generation_engine.__dict__.items()
        if not name.startswith("__")
    }
)

