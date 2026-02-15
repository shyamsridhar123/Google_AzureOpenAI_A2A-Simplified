"""Centralized logging configuration for the A2A demo project."""

import logging
import os
from typing import Optional


def setup_logging(level: Optional[str] = None, format_string: Optional[str] = None) -> None:
    """
    Configure logging for the entire application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to INFO.
        format_string: Custom format string for log messages. Uses default if not provided.
    """
    # Get logging level from environment or use provided level or default to INFO
    log_level_str = level or os.getenv("LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # Get format string from environment or use provided or default
    log_format = format_string or os.getenv(
        "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure root logger
    logging.basicConfig(level=log_level, format=log_format, force=True)

    # Set lower log level for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Name of the module (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
