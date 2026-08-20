"""Logging setup used across the application."""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def silence_noisy_third_party_loggers() -> None:
    """Raise the log level of third-party loggers that log routine,
    expected control-flow through Python's `logging` module rather than
    raising exceptions -- so our own `try/except` blocks around their calls
    don't (and can't) suppress them.

    sqlglot is the concrete case here: `sqlglot.parse_one()` falls back to
    a generic `Command` node for syntax it doesn't model (this codebase
    hits that constantly, since T-SQL procedural bodies contain PRINT,
    TRY/CATCH, and multi-branch control flow sqlglot only partially
    supports) and logs a WARNING via `logging.getLogger("sqlglot")` when it
    does. That fallback is expected and already handled by the calling
    code's own logic (see the DML-only guards in
    app/derivation/dd_generation_engine.py and app/parsing/sql_parser.py) --
    it doesn't need to also print a warning on every occurrence. Call this
    once, at process start-up, before any sqlglot usage.
    """
    logging.getLogger("sqlglot").setLevel(logging.ERROR)