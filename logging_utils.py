"""Logging configuration utilities for the LSNPC project.

This module provides a centralized way to configure logging across the entire project.
It supports different logging levels, output formats, and destinations (console, file, etc.).
"""

import logging
import sys
from pathlib import Path
from typing import Literal, Optional, Union


# Default logging format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logging(
    level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO',
    log_file: Optional[Union[str, Path]] = None,
    format_str: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    verbose: bool = False
) -> None:
    """Configure logging for the entire project.
    
    This function should be called once at the start of training scripts
    to ensure consistent logging across all modules.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Default: 'INFO'
        log_file: Optional path to a log file. If provided, logs will also
                  be written to this file. Default: None
        format_str: Log message format string. Default: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format: Date/time format for log timestamps. Default: '%Y-%m-%d %H:%M:%S'
        verbose: If True, sets level to DEBUG. Default: False
    
    Example:
        >>> from logging_utils import setup_logging
        >>> setup_logging(level='INFO', log_file='training.log')
        
        # Now all modules with `logger = logging.getLogger(__name__)` will use this config
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Override with DEBUG if verbose
    if verbose:
        numeric_level = logging.DEBUG
    
    # Create formatter
    formatter = logging.Formatter(format_str, datefmt=date_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__ of the module)
    
    Returns:
        logging.Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
    """
    return logging.getLogger(name)


# Re-export logging constants for convenience
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


__all__ = [
    'setup_logging',
    'get_logger',
    'DEFAULT_FORMAT',
    'DEFAULT_DATE_FORMAT',
    'DEBUG',
    'INFO', 
    'WARNING',
    'ERROR',
    'CRITICAL',
]
