"""
Memory profiling and optimization utilities.

This module provides tools for monitoring and optimizing memory usage
during data analysis.

Features:
- Memory usage tracking
- Memory optimization recommendations
- Out-of-memory detection
- Memory-efficient operations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import gc
import warnings

from simplus_eda.logging_config import get_logger
from simplus_eda.exceptions import MemoryError as SimplusMemoryError

logger = get_logger(__name__)

# Try to import psutil for system memory info
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MemoryProfiler:
    """
    Track and profile memory usage during operations.

    Example:
        >>> profiler = MemoryProfiler()
        >>> profiler.start('data_loading')
        >>> df = pd.read_csv('large.csv')
        >>> profiler.stop('data_loading')
        >>> profiler.report()
    """

    def __init__(self):
        """Initialize MemoryProfiler."""
        self.measurements = {}
        self.current_operation = None

    def start(self, operation: str) -> None:
        """
        Start tracking memory for an operation.

        Args:
            operation: Operation name
        """
        self.current_operation = operation
        self.measurements[operation] = {
            'start_memory_mb': self._get_current_memory()
        }

    def stop(self, operation: str) -> Dict[str, float]:
        """
        Stop tracking and calculate memory change.

        Args:
            operation: Operation name

        Returns:
            Dictionary with memory statistics
        """
        if operation not in self.measurements:
            logger.warning(f"Operation '{operation}' was not started")
            return {}

        end_memory = self._get_current_memory()
        start_memory = self.measurements[operation]['start_memory_mb']

        memory_change = end_memory - start_memory

        self.measurements[operation].update({
            'end_memory_mb': end_memory,
            'memory_change_mb': memory_change,
            'peak_memory_mb': self._get_peak_memory()
        })

        if memory_change > 100:
            logger.warning(
                f"Operation '{operation}' used {memory_change:.1f} MB of memory"
            )

        self.current_operation = None

        return self.measurements[operation]

    def report(self) -> None:
        """Print memory usage report."""
        if not self.measurements:
            logger.info("No memory measurements recorded")
            return

        print("\nMemory Usage Report:")
        print("="*70)

        for operation, stats in self.measurements.items():
            change = stats.get('memory_change_mb', 0)
            print(f"{operation}:")
            print(f"  Memory change: {change:+.2f} MB")
            if 'peak_memory_mb' in stats:
                print(f"  Peak memory: {stats['peak_memory_mb']:.2f} MB")

        print("="*70)

    def _get_current_memory(self) -> float:
        """Get current process memory usage in MB."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        else:
            return 0.0

    def _get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        else:
            return 0.0


def get_system_memory_info() -> Dict[str, Any]:
    """
    Get system memory information.

    Returns:
        Dictionary with memory statistics

    Example:
        >>> mem_info = get_system_memory_info()
        >>> print(f"Available: {mem_info['available_gb']:.2f} GB")
    """
    if not PSUTIL_AVAILABLE:
        logger.warning("psutil not available, returning dummy values")
        return {
            'total_gb': 0,
            'available_gb': 0,
            'used_gb': 0,
            'percent_used': 0,
            'psutil_available': False
        }

    memory = psutil.virtual_memory()

    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent_used': memory.percent,
        'psutil_available': True
    }


def estimate_dataframe_memory(
    n_rows: int,
    columns: Dict[str, str],
    categorical_cardinality: Optional[Dict[str, int]] = None
) -> float:
    """
    Estimate memory usage of a DataFrame before loading.

    Args:
        n_rows: Number of rows
        columns: Dictionary mapping column names to dtypes
        categorical_cardinality: Cardinality of categorical columns

    Returns:
        Estimated memory in MB

    Example:
        >>> columns = {'id': 'int64', 'name': 'object', 'value': 'float64'}
        >>> mem_mb = estimate_dataframe_memory(1000000, columns)
        >>> print(f"Estimated memory: {mem_mb:.2f} MB")
    """
    # Bytes per row
    bytes_per_row = 0

    for col, dtype in columns.items():
        if dtype in ['int8', 'uint8', 'bool']:
            bytes_per_row += 1
        elif dtype in ['int16', 'uint16']:
            bytes_per_row += 2
        elif dtype in ['int32', 'uint32', 'float32']:
            bytes_per_row += 4
        elif dtype in ['int64', 'uint64', 'float64']:
            bytes_per_row += 8
        elif dtype in ['object', 'string']:
            # Estimate: 50 bytes per string on average
            bytes_per_row += 50
        elif dtype == 'category':
            # Category uses less memory
            if categorical_cardinality and col in categorical_cardinality:
                # Code size + category storage
                bytes_per_row += 1 + (categorical_cardinality[col] * 8) / n_rows
            else:
                bytes_per_row += 2
        else:
            # Unknown type: assume 8 bytes
            bytes_per_row += 8

    total_bytes = n_rows * bytes_per_row
    total_mb = total_bytes / (1024 * 1024)

    return total_mb


def check_memory_available(
    required_mb: float,
    safety_margin: float = 0.2
) -> Tuple[bool, str]:
    """
    Check if sufficient memory is available.

    Args:
        required_mb: Required memory in MB
        safety_margin: Safety margin (0.2 = 20% extra)

    Returns:
        Tuple of (is_available, message)

    Example:
        >>> available, msg = check_memory_available(1000)
        >>> if not available:
        ...     print(f"Warning: {msg}")
    """
    if not PSUTIL_AVAILABLE:
        logger.warning("Cannot check memory (psutil not available)")
        return True, "Memory check unavailable"

    memory = psutil.virtual_memory()
    available_mb = memory.available / (1024 * 1024)
    required_with_margin = required_mb * (1 + safety_margin)

    if available_mb >= required_with_margin:
        return True, f"Sufficient memory available ({available_mb:.0f} MB)"
    else:
        return False, (
            f"Insufficient memory: need {required_with_margin:.0f} MB, "
            f"only {available_mb:.0f} MB available"
        )


def optimize_dataframe_memory(
    df: pd.DataFrame,
    categorical_threshold: int = 50,
    downcast_integers: bool = True,
    downcast_floats: bool = True,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Optimize DataFrame memory usage.

    Args:
        df: DataFrame to optimize
        categorical_threshold: Convert to category if unique values < threshold
        downcast_integers: Downcast integer types
        downcast_floats: Downcast float types
        verbose: Print optimization details

    Returns:
        Tuple of (optimized_df, optimization_report)

    Example:
        >>> df_opt, report = optimize_dataframe_memory(df, verbose=True)
        >>> print(f"Memory saved: {report['memory_saved_mb']:.2f} MB")
    """
    start_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)

    df_opt = df.copy()
    optimizations = []

    # Optimize object columns
    for col in df_opt.select_dtypes(include=['object']).columns:
        n_unique = df_opt[col].nunique()

        if n_unique < categorical_threshold:
            df_opt[col] = df_opt[col].astype('category')
            optimizations.append(f"{col}: object -> category ({n_unique} unique)")

    # Downcast integers
    if downcast_integers:
        for col in df_opt.select_dtypes(include=['int']).columns:
            original_dtype = df_opt[col].dtype
            df_opt[col] = pd.to_numeric(df_opt[col], downcast='integer')
            if df_opt[col].dtype != original_dtype:
                optimizations.append(f"{col}: {original_dtype} -> {df_opt[col].dtype}")

    # Downcast floats
    if downcast_floats:
        for col in df_opt.select_dtypes(include=['float']).columns:
            original_dtype = df_opt[col].dtype
            df_opt[col] = pd.to_numeric(df_opt[col], downcast='float')
            if df_opt[col].dtype != original_dtype:
                optimizations.append(f"{col}: {original_dtype} -> {df_opt[col].dtype}")

    end_memory = df_opt.memory_usage(deep=True).sum() / (1024 * 1024)
    memory_saved = start_memory - end_memory
    percent_saved = (memory_saved / start_memory * 100) if start_memory > 0 else 0

    report = {
        'start_memory_mb': start_memory,
        'end_memory_mb': end_memory,
        'memory_saved_mb': memory_saved,
        'percent_saved': percent_saved,
        'optimizations': optimizations
    }

    if verbose:
        print(f"\nMemory Optimization Report:")
        print(f"  Original: {start_memory:.2f} MB")
        print(f"  Optimized: {end_memory:.2f} MB")
        print(f"  Saved: {memory_saved:.2f} MB ({percent_saved:.1f}%)")
        if optimizations:
            print(f"  Optimizations applied:")
            for opt in optimizations:
                print(f"    - {opt}")

    return df_opt, report


def suggest_optimizations(df: pd.DataFrame) -> List[str]:
    """
    Suggest memory optimization strategies for a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        List of optimization suggestions

    Example:
        >>> suggestions = suggest_optimizations(large_df)
        >>> for suggestion in suggestions:
        ...     print(f"- {suggestion}")
    """
    suggestions = []

    # Check object columns
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        n_unique = df[col].nunique()
        if n_unique < 50:
            suggestions.append(
                f"Convert '{col}' to category (only {n_unique} unique values)"
            )

    # Check for large numeric types
    for col in df.select_dtypes(include=['int64']).columns:
        min_val = df[col].min()
        max_val = df[col].max()

        if min_val >= 0:
            if max_val < 256:
                suggestions.append(f"Downcast '{col}' to uint8")
            elif max_val < 65536:
                suggestions.append(f"Downcast '{col}' to uint16")
        else:
            if min_val >= -128 and max_val <= 127:
                suggestions.append(f"Downcast '{col}' to int8")
            elif min_val >= -32768 and max_val <= 32767:
                suggestions.append(f"Downcast '{col}' to int16")

    # Check for float64 that could be float32
    for col in df.select_dtypes(include=['float64']).columns:
        # Sample check (full check would be expensive)
        sample = df[col].dropna().sample(min(1000, len(df[col].dropna())))
        if (sample == sample.astype('float32')).all():
            suggestions.append(f"Consider downcast '{col}' to float32")

    # Overall memory check
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    if memory_mb > 1000:
        suggestions.append(
            f"Dataset is large ({memory_mb:.0f} MB). Consider using Dask or sampling."
        )

    return suggestions


def force_garbage_collection() -> Dict[str, int]:
    """
    Force garbage collection and return stats.

    Returns:
        Dictionary with GC statistics

    Example:
        >>> gc_stats = force_garbage_collection()
        >>> print(f"Collected {gc_stats['collected']} objects")
    """
    # Get counts before
    before = sum(gc.get_count())

    # Force collection
    collected = gc.collect()

    # Get counts after
    after = sum(gc.get_count())

    logger.debug(f"Garbage collection: collected {collected} objects")

    return {
        'collected': collected,
        'objects_before': before,
        'objects_after': after
    }


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'MemoryProfiler',
    'get_system_memory_info',
    'estimate_dataframe_memory',
    'check_memory_available',
    'optimize_dataframe_memory',
    'suggest_optimizations',
    'force_garbage_collection',
    'PSUTIL_AVAILABLE',
]
