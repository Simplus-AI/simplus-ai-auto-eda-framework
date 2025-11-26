"""
Error recovery mechanisms for Simplus EDA Framework.

This module provides utilities for graceful error handling and recovery,
allowing analyses to continue even when some operations fail.

Features:
- Retry logic with exponential backoff
- Fallback mechanisms
- Safe execution wrappers
- Partial result handling
- Error context preservation
"""

import functools
import time
from typing import Callable, Any, Optional, List, Dict, TypeVar, Tuple
import logging
from simplus_eda.logging_config import get_logger
from simplus_eda.exceptions import SimplusEDAError, AnalysisError

logger = get_logger(__name__)

T = TypeVar('T')


# ============================================================================
# Retry Mechanisms
# ============================================================================

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
    logger: Optional[logging.Logger] = None
):
    """
    Decorator to retry a function on failure with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        logger: Logger for retry messages

    Example:
        >>> @retry(max_attempts=3, delay=1.0, backoff=2.0)
        ... def unstable_operation():
        ...     # May fail occasionally
        ...     return fetch_data_from_api()
    """
    if logger is None:
        logger = get_logger(__name__)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            # All attempts failed
            raise last_exception

        return wrapper
    return decorator


# ============================================================================
# Fallback Mechanisms
# ============================================================================

def with_fallback(
    fallback_value: Any = None,
    fallback_func: Optional[Callable] = None,
    exceptions: Tuple[type, ...] = (Exception,),
    logger: Optional[logging.Logger] = None,
    log_errors: bool = True
):
    """
    Decorator to provide fallback value or function on error.

    Args:
        fallback_value: Value to return on error (if fallback_func not provided)
        fallback_func: Function to call on error (takes exception as argument)
        exceptions: Tuple of exceptions to catch
        logger: Logger for error messages
        log_errors: Whether to log errors

    Example:
        >>> @with_fallback(fallback_value=0.0)
        ... def calculate_mean(data):
        ...     return sum(data) / len(data)  # May fail if data is empty
        >>>
        >>> result = calculate_mean([])  # Returns 0.0 instead of raising error
    """
    if logger is None:
        logger = get_logger(__name__)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}. Using fallback.")

                if fallback_func is not None:
                    return fallback_func(e)
                else:
                    return fallback_value

        return wrapper
    return decorator


# ============================================================================
# Safe Execution
# ============================================================================

class SafeExecutionResult:
    """
    Result of a safe execution attempt.

    Attributes:
        success: Whether execution succeeded
        value: Result value (if successful)
        error: Exception (if failed)
        error_message: Error message
    """

    def __init__(
        self,
        success: bool,
        value: Any = None,
        error: Optional[Exception] = None,
        error_message: Optional[str] = None
    ):
        self.success = success
        self.value = value
        self.error = error
        self.error_message = error_message or (str(error) if error else None)

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success

    def __repr__(self) -> str:
        if self.success:
            return f"SafeExecutionResult(success=True, value={self.value!r})"
        else:
            return f"SafeExecutionResult(success=False, error={self.error_message!r})"


def safe_execute(
    func: Callable[..., T],
    *args,
    default: Any = None,
    exceptions: Tuple[type, ...] = (Exception,),
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> SafeExecutionResult:
    """
    Execute a function safely, catching exceptions and returning result object.

    Args:
        func: Function to execute
        *args: Positional arguments for func
        default: Default value to return on error
        exceptions: Tuple of exceptions to catch
        logger: Logger for error messages
        **kwargs: Keyword arguments for func

    Returns:
        SafeExecutionResult with success status and value or error

    Example:
        >>> result = safe_execute(risky_function, arg1, arg2, default=None)
        >>> if result.success:
        ...     print(f"Success: {result.value}")
        ... else:
        ...     print(f"Failed: {result.error_message}")
    """
    if logger is None:
        logger = get_logger(__name__)

    try:
        value = func(*args, **kwargs)
        return SafeExecutionResult(success=True, value=value)
    except exceptions as e:
        logger.error(f"Safe execution failed for {func.__name__}: {e}")
        return SafeExecutionResult(
            success=False,
            value=default,
            error=e,
            error_message=str(e)
        )


# ============================================================================
# Partial Result Handling
# ============================================================================

class PartialResults:
    """
    Container for partial results when some operations fail.

    Allows analysis to continue even when some columns/operations fail.

    Example:
        >>> results = PartialResults()
        >>> for column in df.columns:
        ...     try:
        ...         stats = compute_stats(df[column])
        ...         results.add_success(column, stats)
        ...     except Exception as e:
        ...         results.add_failure(column, e)
        >>> print(f"Successful: {results.success_count}")
        >>> print(f"Failed: {results.failure_count}")
    """

    def __init__(self):
        """Initialize PartialResults."""
        self.successes: Dict[str, Any] = {}
        self.failures: Dict[str, Exception] = {}
        self.metadata: Dict[str, Any] = {}

    def add_success(self, key: str, value: Any, metadata: Optional[Dict] = None) -> None:
        """
        Add a successful result.

        Args:
            key: Result identifier (e.g., column name)
            value: Result value
            metadata: Optional metadata about the result
        """
        self.successes[key] = value
        if metadata:
            self.metadata[key] = metadata

    def add_failure(self, key: str, error: Exception, metadata: Optional[Dict] = None) -> None:
        """
        Add a failed result.

        Args:
            key: Result identifier
            error: Exception that occurred
            metadata: Optional metadata about the failure
        """
        self.failures[key] = error
        if metadata:
            self.metadata[key] = metadata

    @property
    def success_count(self) -> int:
        """Number of successful operations."""
        return len(self.successes)

    @property
    def failure_count(self) -> int:
        """Number of failed operations."""
        return len(self.failures)

    @property
    def total_count(self) -> int:
        """Total number of operations."""
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.success_count / self.total_count) * 100

    def has_successes(self) -> bool:
        """Check if there are any successful results."""
        return bool(self.successes)

    def has_failures(self) -> bool:
        """Check if there are any failures."""
        return bool(self.failures)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of results.

        Returns:
            Dictionary with success/failure counts and details
        """
        return {
            'total_operations': self.total_count,
            'successful': self.success_count,
            'failed': self.failure_count,
            'success_rate': self.success_rate,
            'successful_keys': list(self.successes.keys()),
            'failed_keys': list(self.failures.keys()),
            'errors': {k: str(v) for k, v in self.failures.items()}
        }

    def __repr__(self) -> str:
        return (f"PartialResults(success={self.success_count}, "
                f"failure={self.failure_count}, "
                f"rate={self.success_rate:.1f}%)")


def execute_partial(
    items: List[Any],
    func: Callable[[Any], T],
    item_name_func: Optional[Callable[[Any], str]] = None,
    logger: Optional[logging.Logger] = None,
    raise_on_all_failed: bool = True
) -> PartialResults:
    """
    Execute a function on multiple items, collecting partial results.

    Args:
        items: List of items to process
        func: Function to apply to each item
        item_name_func: Function to get name from item (default: str)
        logger: Logger for messages
        raise_on_all_failed: Raise error if all items fail

    Returns:
        PartialResults object with successes and failures

    Example:
        >>> columns = df.columns
        >>> results = execute_partial(
        ...     columns,
        ...     lambda col: compute_stats(df[col]),
        ...     item_name_func=lambda col: col
        ... )
        >>> print(f"Processed {results.success_count}/{results.total_count} columns")
    """
    if logger is None:
        logger = get_logger(__name__)

    if item_name_func is None:
        item_name_func = str

    results = PartialResults()

    for item in items:
        item_name = item_name_func(item)
        try:
            value = func(item)
            results.add_success(item_name, value)
        except Exception as e:
            logger.warning(f"Failed to process '{item_name}': {e}")
            results.add_failure(item_name, e)

    # Log summary
    logger.info(
        f"Partial execution completed: {results.success_count}/{results.total_count} succeeded "
        f"({results.success_rate:.1f}%)"
    )

    # Raise error if all failed
    if raise_on_all_failed and not results.has_successes():
        error_summary = "\n".join([f"  - {k}: {v}" for k, v in results.failures.items()])
        raise AnalysisError(
            f"All {results.total_count} operations failed",
            details={'errors': error_summary}
        )

    return results


# ============================================================================
# Context Preservation
# ============================================================================

class ErrorContext:
    """
    Preserve context information when errors occur.

    Useful for debugging and error reporting.

    Example:
        >>> with ErrorContext(operation="data_loading", file="data.csv") as ctx:
        ...     df = pd.read_csv("data.csv")
        ...     ctx.add_info("rows_loaded", len(df))
    """

    def __init__(self, **context):
        """
        Initialize ErrorContext.

        Args:
            **context: Context key-value pairs
        """
        self.context = context
        self.additional_info: Dict[str, Any] = {}

    def add_info(self, key: str, value: Any) -> None:
        """
        Add additional context information.

        Args:
            key: Information key
            value: Information value
        """
        self.additional_info[key] = value

    def get_full_context(self) -> Dict[str, Any]:
        """Get complete context including additional info."""
        return {**self.context, **self.additional_info}

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and enhance exception if one occurred."""
        if exc_type is not None and isinstance(exc_val, SimplusEDAError):
            # Add context to SimplusEDA exceptions
            exc_val.details.update(self.get_full_context())
        return False  # Don't suppress exception


# ============================================================================
# Graceful Degradation
# ============================================================================

def degrade_gracefully(
    primary_func: Callable[..., T],
    fallback_funcs: List[Callable[..., T]],
    logger: Optional[logging.Logger] = None,
    *args,
    **kwargs
) -> T:
    """
    Try primary function, falling back to alternatives if it fails.

    Args:
        primary_func: Primary function to try
        fallback_funcs: List of fallback functions to try in order
        logger: Logger for messages
        *args: Arguments for functions
        **kwargs: Keyword arguments for functions

    Returns:
        Result from first successful function

    Raises:
        Exception: If all functions fail

    Example:
        >>> result = degrade_gracefully(
        ...     lambda: expensive_computation(),
        ...     [
        ...         lambda: faster_approximation(),
        ...         lambda: simple_estimate()
        ...     ]
        ... )
    """
    if logger is None:
        logger = get_logger(__name__)

    # Try primary function
    try:
        return primary_func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Primary function failed: {e}. Trying fallbacks...")

        # Try fallbacks in order
        for i, fallback in enumerate(fallback_funcs, 1):
            try:
                logger.info(f"Trying fallback {i}/{len(fallback_funcs)}...")
                result = fallback(*args, **kwargs)
                logger.info(f"Fallback {i} succeeded")
                return result
            except Exception as fallback_error:
                logger.warning(f"Fallback {i} failed: {fallback_error}")

        # All functions failed
        logger.error("All functions (primary and fallbacks) failed")
        raise AnalysisError(
            "Primary function and all fallbacks failed",
            details={
                'primary_error': str(e),
                'n_fallbacks_tried': len(fallback_funcs)
            }
        )


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'retry',
    'with_fallback',
    'safe_execute',
    'SafeExecutionResult',
    'PartialResults',
    'execute_partial',
    'ErrorContext',
    'degrade_gracefully',
]
