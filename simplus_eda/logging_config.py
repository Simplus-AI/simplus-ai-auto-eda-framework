"""
Logging configuration for Simplus EDA Framework.

This module provides a centralized logging system with configurable levels,
formatters, and handlers for the entire framework.

Features:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Console and file logging
- Colored console output
- Structured log formatting
- Context-aware logging
- Performance tracking
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import functools
import time


# ============================================================================
# Logger Configuration
# ============================================================================

# Default log format
DEFAULT_FORMAT = '[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s'
DETAILED_FORMAT = '[%(asctime)s] %(levelname)-8s [%(name)s:%(funcName)s:%(lineno)d] %(message)s'
SIMPLE_FORMAT = '%(levelname)-8s: %(message)s'

# Date format
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Color codes for console output
COLORS = {
    'DEBUG': '\033[36m',      # Cyan
    'INFO': '\033[32m',       # Green
    'WARNING': '\033[33m',    # Yellow
    'ERROR': '\033[31m',      # Red
    'CRITICAL': '\033[35m',   # Magenta
    'RESET': '\033[0m'        # Reset
}


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with color support for console output.
    """

    def __init__(self, fmt: str = DEFAULT_FORMAT, datefmt: str = DATE_FORMAT,
                 use_colors: bool = True):
        """
        Initialize ColoredFormatter.

        Args:
            fmt: Log format string
            datefmt: Date format string
            use_colors: Whether to use colors in output
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional colors."""
        if self.use_colors:
            levelname = record.levelname
            if levelname in COLORS:
                record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"

        return super().format(record)


def setup_logger(
    name: str = 'simplus_eda',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = DEFAULT_FORMAT,
    use_colors: bool = True,
    propagate: bool = False
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Log message format
        use_colors: Use colored output for console
        propagate: Whether to propagate to parent loggers

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger('my_app', level=logging.DEBUG)
        >>> logger.info("Application started")
        >>> logger.warning("This is a warning")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(log_format, use_colors=use_colors)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(DETAILED_FORMAT, DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = 'simplus_eda') -> logging.Logger:
    """
    Get a logger instance.

    If the logger doesn't exist, it will be created with default settings.

    Args:
        name: Logger name (typically module name)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data...")
    """
    logger = logging.getLogger(name)

    # Set up logger if it doesn't have handlers
    if not logger.handlers:
        setup_logger(name)

    return logger


def configure_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    verbose: bool = False,
    detailed: bool = False
) -> None:
    """
    Configure global logging settings for the framework.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional path to log file
        verbose: Enable verbose logging (DEBUG level)
        detailed: Use detailed log format

    Example:
        >>> configure_logging(level='DEBUG', log_file='eda.log', detailed=True)
        >>> logger = get_logger(__name__)
        >>> logger.debug("Debug information")
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Override with DEBUG if verbose
    if verbose:
        numeric_level = logging.DEBUG

    # Choose format
    log_format = DETAILED_FORMAT if detailed else DEFAULT_FORMAT

    # Set up root simplus_eda logger
    setup_logger(
        name='simplus_eda',
        level=numeric_level,
        log_file=log_file,
        log_format=log_format
    )


# ============================================================================
# Context Manager for Logging
# ============================================================================

class LogContext:
    """
    Context manager for temporary logging configuration.

    Example:
        >>> with LogContext(level='DEBUG'):
        ...     logger.debug("This will be shown")
        >>> logger.debug("This will not be shown (if level was INFO)")
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        level: Optional[int] = None,
        suppress: bool = False
    ):
        """
        Initialize LogContext.

        Args:
            logger: Logger to modify (default: root simplus_eda logger)
            level: Temporary log level
            suppress: Suppress all logging
        """
        self.logger = logger or get_logger()
        self.new_level = level
        self.suppress = suppress
        self.old_level = None
        self.old_disabled = None

    def __enter__(self):
        """Enter context."""
        self.old_level = self.logger.level
        self.old_disabled = self.logger.disabled

        if self.suppress:
            self.logger.disabled = True
        elif self.new_level is not None:
            self.logger.setLevel(self.new_level)

        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original settings."""
        self.logger.setLevel(self.old_level)
        self.logger.disabled = self.old_disabled


# ============================================================================
# Decorators for Logging
# ============================================================================

def log_execution(logger: Optional[logging.Logger] = None, level: int = logging.INFO):
    """
    Decorator to log function execution with timing.

    Args:
        logger: Logger to use (default: function's module logger)
        level: Log level for execution messages

    Example:
        >>> @log_execution()
        ... def process_data(df):
        ...     # ... processing ...
        ...     return result
        >>>
        >>> result = process_data(df)
        INFO: Executing process_data...
        INFO: process_data completed in 1.23s
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.log(level, f"Executing {func_name}...")

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.log(level, f"{func_name} completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"{func_name} failed after {elapsed:.2f}s: {e}")
                raise

        return wrapper
    return decorator


def log_errors(logger: Optional[logging.Logger] = None, reraise: bool = True):
    """
    Decorator to log exceptions with full context.

    Args:
        logger: Logger to use (default: function's module logger)
        reraise: Whether to re-raise the exception after logging

    Example:
        >>> @log_errors()
        ... def risky_operation():
        ...     raise ValueError("Something went wrong")
        >>>
        >>> risky_operation()
        ERROR: Exception in risky_operation: ValueError: Something went wrong
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Exception in {func.__name__}: {type(e).__name__}: {e}",
                    exc_info=True
                )
                if reraise:
                    raise

        return wrapper
    return decorator


# ============================================================================
# Performance Tracking
# ============================================================================

class PerformanceTracker:
    """
    Track and log performance metrics.

    Example:
        >>> tracker = PerformanceTracker()
        >>> tracker.start('data_loading')
        >>> # ... load data ...
        >>> tracker.stop('data_loading')
        >>> tracker.start('analysis')
        >>> # ... analyze ...
        >>> tracker.stop('analysis')
        >>> tracker.report()
        Performance Report:
          data_loading: 1.23s
          analysis: 5.67s
          Total: 6.90s
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize PerformanceTracker.

        Args:
            logger: Logger to use for reporting
        """
        self.logger = logger or get_logger()
        self.timings: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}

    def start(self, operation: str) -> None:
        """
        Start tracking an operation.

        Args:
            operation: Name of the operation
        """
        self.start_times[operation] = time.time()
        self.logger.debug(f"Started: {operation}")

    def stop(self, operation: str) -> float:
        """
        Stop tracking an operation.

        Args:
            operation: Name of the operation

        Returns:
            Elapsed time in seconds
        """
        if operation not in self.start_times:
            self.logger.warning(f"Operation '{operation}' was not started")
            return 0.0

        elapsed = time.time() - self.start_times[operation]
        self.timings[operation] = elapsed
        self.logger.debug(f"Completed: {operation} ({elapsed:.2f}s)")

        del self.start_times[operation]
        return elapsed

    def report(self, level: int = logging.INFO) -> None:
        """
        Report all tracked timings.

        Args:
            level: Log level for the report
        """
        if not self.timings:
            self.logger.log(level, "No performance data to report")
            return

        total = sum(self.timings.values())
        lines = ["Performance Report:"]

        for operation, elapsed in sorted(self.timings.items(), key=lambda x: -x[1]):
            percentage = (elapsed / total * 100) if total > 0 else 0
            lines.append(f"  {operation}: {elapsed:.2f}s ({percentage:.1f}%)")

        lines.append(f"  Total: {total:.2f}s")

        self.logger.log(level, "\n".join(lines))

    def get_timings(self) -> Dict[str, float]:
        """
        Get all recorded timings.

        Returns:
            Dictionary of operation names to elapsed times
        """
        return self.timings.copy()


# ============================================================================
# Initialization
# ============================================================================

# Create default logger
_default_logger = setup_logger('simplus_eda', level=logging.INFO)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'setup_logger',
    'get_logger',
    'configure_logging',
    'LogContext',
    'log_execution',
    'log_errors',
    'PerformanceTracker',
    'ColoredFormatter',
]
