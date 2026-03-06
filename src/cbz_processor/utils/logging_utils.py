"""Logging utilities for CBZ processor."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from typing import NoReturn


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(
    log_dir: Path | str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Set up logging with console and file handlers.

    Args:
        log_dir: Directory for log files
        console_level: Logging level for console
        file_level: Logging level for file

    Returns:
        Configured logger instance
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("cbz_processor")
    logger.setLevel(logging.DEBUG)

    console = Console()
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        level=console_level,
    )
    rich_handler.setFormatter(
        logging.Formatter("%(message)s", datefmt="%H:%M:%S"),
    )
    logger.addHandler(rich_handler)

    file_handler = logging.FileHandler(
        log_path / f"processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    error_handler = logging.FileHandler(
        log_path / f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    logger.addHandler(error_handler)

    return logger


def log_error(logger: logging.Logger, msg: str, **kwargs) -> None:
    """Helper to log error with context."""
    logger.error(msg, extra={"context": kwargs})


def log_warning(logger: logging.Logger, msg: str, **kwargs) -> None:
    """Helper to log warning with context."""
    logger.warning(msg, extra={"context": kwargs})


def log_info(logger: logging.Logger, msg: str, **kwargs) -> None:
    """Helper to log info with context."""
    logger.info(msg, extra={"context": kwargs})


def log_debug(logger: logging.Logger, msg: str, **kwargs) -> None:
    """Helper to log debug with context."""
    logger.debug(msg, extra={"context": kwargs})
