"""Shared CLI argument parsers for the Typer entrypoints in `scripts/`."""

from __future__ import annotations


def int_list(s: str) -> list[int]:
    """Parse a comma-separated list of ints, ignoring whitespace and empty items.

    Args:
        s: e.g. ``"1, 2, 5,10"``.

    Returns:
        The parsed ints, e.g. ``[1, 2, 5, 10]``. An empty/whitespace string yields ``[]``.
    """
    return [int(x) for x in str(s).replace(" ", "").split(",") if x != ""]
