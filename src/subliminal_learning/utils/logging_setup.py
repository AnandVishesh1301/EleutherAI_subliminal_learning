"""One-shot loguru configuration for script entry points."""

from __future__ import annotations

import sys

from loguru import logger


def configure_loguru() -> None:
    """Configure loguru to print INFO+ logs to stderr with a compact format."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}",
    )
