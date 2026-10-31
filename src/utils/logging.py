"""Shared logging configuration for NeuroScan AI.

Provides a centralized logging setup using loguru with consistent formatting
across all modules.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: str | Path | None = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """Configure application-wide logging.

    Args:
        log_level: Minimum log level to capture.
        log_file: Path to log file. If None, logs only to stderr.
        rotation: When to rotate the log file.
        retention: How long to keep old log files.
    """
    # Remove default handler
    logger.remove()

    # Console handler with color
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, format=log_format, level=log_level, colorize=True)

    # File handler (no PHI in logs!)
    if log_file:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        logger.add(
            log_file,
            format=file_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="gz",
        )

    logger.info("Logging configured at level {}", log_level)


def get_logger(name: str) -> "logger":
    """Get a named logger instance.

    Args:
        name: Module or component name for the logger.

    Returns:
        A loguru logger instance bound with the given name.
    """
    return logger.bind(name=name)

