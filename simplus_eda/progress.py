"""
Progress tracking and indicators for the EDA framework.

This module provides progress tracking utilities with tqdm integration,
time estimates, verbose logging, and callback hooks for custom progress tracking.

Features:
- tqdm progress bars for analysis steps
- Time estimation for operations
- Verbose logging with progress updates
- Callback hooks for custom progress tracking
- Thread-safe progress tracking
"""

import time
import threading
from typing import Optional, Callable, Dict, Any, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta

from simplus_eda.logging_config import get_logger

logger = get_logger(__name__)

# Try to import tqdm
try:
    from tqdm import tqdm as tqdm_progress
    from tqdm.auto import tqdm as tqdm_auto
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm_progress = None
    tqdm_auto = None


# ============================================================================
# Progress Callback System
# ============================================================================

@dataclass
class ProgressUpdate:
    """
    Progress update information.

    Attributes:
        operation: Operation name
        current: Current progress value
        total: Total progress value
        percentage: Progress percentage (0-100)
        message: Progress message
        elapsed_time: Elapsed time in seconds
        estimated_remaining: Estimated remaining time in seconds
        metadata: Additional metadata
    """
    operation: str
    current: int
    total: int
    percentage: float
    message: str = ""
    elapsed_time: Optional[float] = None
    estimated_remaining: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation."""
        parts = [
            f"{self.operation}:",
            f"{self.current}/{self.total}",
            f"({self.percentage:.1f}%)"
        ]

        if self.message:
            parts.append(f"- {self.message}")

        if self.estimated_remaining:
            eta = timedelta(seconds=int(self.estimated_remaining))
            parts.append(f"ETA: {eta}")

        return " ".join(parts)


class ProgressCallback:
    """
    Base class for progress callbacks.

    Subclass this to create custom progress handlers.
    """

    def on_start(self, operation: str, total: int, **kwargs):
        """Called when operation starts."""
        pass

    def on_update(self, update: ProgressUpdate):
        """Called on progress update."""
        pass

    def on_complete(self, operation: str, elapsed_time: float):
        """Called when operation completes."""
        pass

    def on_error(self, operation: str, error: Exception):
        """Called on error."""
        pass


class LoggingCallback(ProgressCallback):
    """Progress callback that logs to logger."""

    def __init__(self, log_level: str = 'INFO'):
        """
        Initialize logging callback.

        Args:
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING')
        """
        self.log_level = log_level.upper()

    def on_start(self, operation: str, total: int, **kwargs):
        """Log operation start."""
        logger.log(self.log_level, f"Starting {operation} (total={total})")

    def on_update(self, update: ProgressUpdate):
        """Log progress update."""
        if update.percentage % 25 == 0:  # Log at 0%, 25%, 50%, 75%, 100%
            logger.log(self.log_level, str(update))

    def on_complete(self, operation: str, elapsed_time: float):
        """Log completion."""
        logger.log(self.log_level,
                  f"Completed {operation} in {elapsed_time:.2f}s")


# ============================================================================
# Progress Tracker
# ============================================================================

class ProgressTracker:
    """
    Track progress for operations with support for callbacks and progress bars.

    Example:
        >>> tracker = ProgressTracker(use_tqdm=True)
        >>> with tracker.track('processing', total=100):
        ...     for i in range(100):
        ...         # Do work
        ...         tracker.update(1)
    """

    def __init__(
        self,
        use_tqdm: bool = True,
        verbose: bool = False,
        callbacks: Optional[List[ProgressCallback]] = None
    ):
        """
        Initialize progress tracker.

        Args:
            use_tqdm: Whether to use tqdm progress bars
            verbose: Whether to enable verbose logging
            callbacks: List of progress callbacks
        """
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self.verbose = verbose
        self.callbacks = callbacks or []

        if verbose and not any(isinstance(cb, LoggingCallback) for cb in self.callbacks):
            self.callbacks.append(LoggingCallback())

        self._current_operation: Optional[str] = None
        self._total: int = 0
        self._current: int = 0
        self._start_time: float = 0
        self._pbar: Optional[Any] = None
        self._lock = threading.Lock()

        if self.use_tqdm and not TQDM_AVAILABLE:
            logger.warning("tqdm not available, progress bars disabled")
            self.use_tqdm = False

    def add_callback(self, callback: ProgressCallback):
        """Add a progress callback."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: ProgressCallback):
        """Remove a progress callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    @contextmanager
    def track(
        self,
        operation: str,
        total: int,
        desc: Optional[str] = None,
        **kwargs
    ):
        """
        Context manager for tracking progress.

        Args:
            operation: Operation name
            total: Total progress units
            desc: Description for progress bar
            **kwargs: Additional metadata

        Example:
            >>> with tracker.track('processing', total=100):
            ...     for i in range(100):
            ...         tracker.update(1)
        """
        self.start(operation, total, desc, **kwargs)
        try:
            yield self
        except Exception as e:
            self._on_error(operation, e)
            raise
        finally:
            self.complete()

    def start(
        self,
        operation: str,
        total: int,
        desc: Optional[str] = None,
        **kwargs
    ):
        """
        Start tracking an operation.

        Args:
            operation: Operation name
            total: Total progress units
            desc: Description for progress bar
            **kwargs: Additional metadata
        """
        with self._lock:
            self._current_operation = operation
            self._total = total
            self._current = 0
            self._start_time = time.time()

            # Create tqdm progress bar
            if self.use_tqdm:
                self._pbar = tqdm_auto(
                    total=total,
                    desc=desc or operation,
                    unit='it',
                    **kwargs
                )

            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback.on_start(operation, total, **kwargs)
                except Exception as e:
                    logger.error(f"Error in callback on_start: {e}")

    def update(self, n: int = 1, message: str = "", **metadata):
        """
        Update progress.

        Args:
            n: Progress increment
            message: Progress message
            **metadata: Additional metadata
        """
        with self._lock:
            if not self._current_operation:
                logger.warning("No active operation to update")
                return

            self._current += n

            # Update progress bar
            if self._pbar:
                if message:
                    self._pbar.set_postfix_str(message)
                self._pbar.update(n)

            # Calculate metrics
            elapsed = time.time() - self._start_time
            percentage = (self._current / self._total * 100) if self._total > 0 else 0

            # Estimate remaining time
            if self._current > 0 and percentage < 100:
                rate = self._current / elapsed
                remaining = (self._total - self._current) / rate
            else:
                remaining = None

            # Create update
            update = ProgressUpdate(
                operation=self._current_operation,
                current=self._current,
                total=self._total,
                percentage=percentage,
                message=message,
                elapsed_time=elapsed,
                estimated_remaining=remaining,
                metadata=metadata
            )

            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback.on_update(update)
                except Exception as e:
                    logger.error(f"Error in callback on_update: {e}")

    def complete(self):
        """Complete current operation."""
        with self._lock:
            if not self._current_operation:
                return

            elapsed = time.time() - self._start_time

            # Close progress bar
            if self._pbar:
                self._pbar.close()
                self._pbar = None

            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback.on_complete(self._current_operation, elapsed)
                except Exception as e:
                    logger.error(f"Error in callback on_complete: {e}")

            # Reset state
            self._current_operation = None
            self._total = 0
            self._current = 0

    def _on_error(self, operation: str, error: Exception):
        """Handle error."""
        # Close progress bar
        if self._pbar:
            self._pbar.close()
            self._pbar = None

        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback.on_error(operation, error)
            except Exception as e:
                logger.error(f"Error in callback on_error: {e}")

    def get_current_progress(self) -> Optional[ProgressUpdate]:
        """Get current progress status."""
        with self._lock:
            if not self._current_operation:
                return None

            elapsed = time.time() - self._start_time
            percentage = (self._current / self._total * 100) if self._total > 0 else 0

            if self._current > 0 and percentage < 100:
                rate = self._current / elapsed
                remaining = (self._total - self._current) / rate
            else:
                remaining = None

            return ProgressUpdate(
                operation=self._current_operation,
                current=self._current,
                total=self._total,
                percentage=percentage,
                elapsed_time=elapsed,
                estimated_remaining=remaining
            )


# ============================================================================
# Convenience Functions
# ============================================================================

def progress_bar(
    iterable,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    disable: bool = False,
    **kwargs
):
    """
    Wrap an iterable with a progress bar.

    Args:
        iterable: Iterable to wrap
        desc: Description
        total: Total length (auto-detected if None)
        disable: Whether to disable progress bar
        **kwargs: Additional tqdm arguments

    Returns:
        Wrapped iterable with progress bar

    Example:
        >>> for item in progress_bar(items, desc='Processing'):
        ...     process(item)
    """
    if not TQDM_AVAILABLE or disable:
        return iterable

    return tqdm_auto(iterable, desc=desc, total=total, **kwargs)


@contextmanager
def progress_context(
    desc: str,
    total: int,
    use_tqdm: bool = True,
    verbose: bool = False
):
    """
    Context manager for simple progress tracking.

    Args:
        desc: Description
        total: Total progress units
        use_tqdm: Whether to use tqdm
        verbose: Whether to log progress

    Yields:
        Update function to call with progress increments

    Example:
        >>> with progress_context('Processing', 100) as update:
        ...     for i in range(100):
        ...         # Do work
        ...         update(1)
    """
    tracker = ProgressTracker(use_tqdm=use_tqdm, verbose=verbose)

    with tracker.track(desc, total):
        yield tracker.update


def estimate_time_remaining(
    current: int,
    total: int,
    elapsed_seconds: float
) -> Optional[float]:
    """
    Estimate time remaining for operation.

    Args:
        current: Current progress
        total: Total progress
        elapsed_seconds: Elapsed time in seconds

    Returns:
        Estimated remaining time in seconds, or None if cannot estimate

    Example:
        >>> remaining = estimate_time_remaining(25, 100, 10.0)
        >>> print(f"ETA: {remaining:.1f} seconds")
    """
    if current <= 0 or current >= total:
        return None

    rate = current / elapsed_seconds
    remaining_items = total - current
    return remaining_items / rate


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string

    Example:
        >>> format_time(90)
        '1m 30s'
        >>> format_time(3665)
        '1h 1m 5s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


# ============================================================================
# Progress Decorators
# ============================================================================

def with_progress(
    desc: Optional[str] = None,
    use_tqdm: bool = True,
    verbose: bool = False
):
    """
    Decorator to add progress tracking to a function.

    The function must accept a 'progress_callback' parameter.

    Args:
        desc: Description for progress bar
        use_tqdm: Whether to use tqdm
        verbose: Whether to enable verbose logging

    Example:
        >>> @with_progress(desc='Computing statistics')
        ... def compute_stats(data, progress_callback=None):
        ...     for i, item in enumerate(data):
        ...         # Do work
        ...         if progress_callback:
        ...             progress_callback(1)
        ...     return results
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Extract total if provided
            total = kwargs.pop('progress_total', None)

            if total is None:
                # Try to infer from first arg if it's a sequence
                if args and hasattr(args[0], '__len__'):
                    total = len(args[0])

            if total is None:
                # No progress tracking without total
                logger.warning(f"Cannot track progress for {func.__name__}: total unknown")
                return func(*args, **kwargs)

            # Create tracker
            tracker = ProgressTracker(use_tqdm=use_tqdm, verbose=verbose)
            operation = desc or func.__name__

            with tracker.track(operation, total):
                # Inject progress callback
                kwargs['progress_callback'] = tracker.update
                return func(*args, **kwargs)

        return wrapper
    return decorator


# ============================================================================
# Multi-Step Progress Tracker
# ============================================================================

class MultiStepProgressTracker:
    """
    Track progress across multiple steps with different weights.

    Example:
        >>> tracker = MultiStepProgressTracker([
        ...     ('load', 10),
        ...     ('analyze', 70),
        ...     ('report', 20),
        ... ])
        >>> with tracker.step('load'):
        ...     load_data()
        >>> with tracker.step('analyze'):
        ...     analyze_data()
    """

    def __init__(
        self,
        steps: List[tuple],
        use_tqdm: bool = True,
        verbose: bool = False
    ):
        """
        Initialize multi-step tracker.

        Args:
            steps: List of (step_name, weight) tuples
            use_tqdm: Whether to use tqdm
            verbose: Whether to enable verbose logging
        """
        self.steps = steps
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self.verbose = verbose

        # Calculate total weight
        self.total_weight = sum(weight for _, weight in steps)

        # Create main progress bar
        if self.use_tqdm:
            self._main_pbar = tqdm_auto(
                total=100,
                desc='Overall Progress',
                unit='%',
                bar_format='{l_bar}{bar}| {n:.0f}/{total:.0f}% [{elapsed}<{remaining}]'
            )
        else:
            self._main_pbar = None

        self._current_weight = 0
        self._step_weights = {name: weight for name, weight in steps}

    @contextmanager
    def step(self, step_name: str, desc: Optional[str] = None):
        """
        Context manager for a step.

        Args:
            step_name: Step name (must be in steps list)
            desc: Optional description override

        Yields:
            Progress tracker for this step
        """
        if step_name not in self._step_weights:
            raise ValueError(f"Unknown step: {step_name}")

        weight = self._step_weights[step_name]

        # Log step start
        if self.verbose:
            logger.info(f"Starting step: {step_name}")

        # Create step tracker (no main progress bar for individual steps)
        step_tracker = ProgressTracker(
            use_tqdm=self.use_tqdm,
            verbose=self.verbose
        )

        try:
            yield step_tracker
        finally:
            # Update main progress bar
            progress_increment = (weight / self.total_weight) * 100
            self._current_weight += weight

            if self._main_pbar:
                self._main_pbar.update(progress_increment)

            if self.verbose:
                logger.info(f"Completed step: {step_name}")

    def close(self):
        """Close the progress tracker."""
        if self._main_pbar:
            self._main_pbar.close()


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'ProgressTracker',
    'ProgressUpdate',
    'ProgressCallback',
    'LoggingCallback',
    'MultiStepProgressTracker',
    'progress_bar',
    'progress_context',
    'with_progress',
    'estimate_time_remaining',
    'format_time',
    'TQDM_AVAILABLE',
]
