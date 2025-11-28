"""
Chunked and streaming data processing utilities.

This module provides tools for processing large datasets in chunks,
enabling analysis of datasets that don't fit in memory.

Features:
- Chunk-based CSV reading
- Streaming statistics computation
- Incremental aggregation
- Memory-efficient iteration
"""

import pandas as pd
import numpy as np
from typing import Iterator, Callable, Dict, Any, Optional, List, Union
from pathlib import Path

from simplus_eda.logging_config import get_logger, PerformanceTracker
from simplus_eda.exceptions import DataLoadingError, AnalysisError

logger = get_logger(__name__)


class ChunkedDataReader:
    """
    Read and process large files in chunks.

    Example:
        >>> reader = ChunkedDataReader('large_file.csv', chunksize=10000)
        >>> for chunk in reader:
        ...     process(chunk)
    """

    def __init__(
        self,
        file_path: str,
        chunksize: int = 10000,
        **read_kwargs
    ):
        """
        Initialize ChunkedDataReader.

        Args:
            file_path: Path to file
            chunksize: Number of rows per chunk
            **read_kwargs: Additional arguments for pd.read_csv
        """
        self.file_path = Path(file_path)
        self.chunksize = chunksize
        self.read_kwargs = read_kwargs

        if not self.file_path.exists():
            raise DataLoadingError(
                f"File not found",
                file_path=str(file_path)
            )

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over chunks."""
        try:
            logger.debug(f"Reading {self.file_path} in chunks of {self.chunksize}")

            for i, chunk in enumerate(pd.read_csv(
                self.file_path,
                chunksize=self.chunksize,
                **self.read_kwargs
            )):
                logger.debug(f"Processing chunk {i+1} ({len(chunk)} rows)")
                yield chunk

        except Exception as e:
            raise DataLoadingError(
                f"Error reading file in chunks",
                file_path=str(self.file_path),
                original_error=e
            )


class StreamingAggregator:
    """
    Compute statistics incrementally over data chunks.

    Allows computing statistics on large datasets without loading
    all data into memory.

    Example:
        >>> aggregator = StreamingAggregator()
        >>> for chunk in chunks:
        ...     aggregator.update(chunk)
        >>> stats = aggregator.compute()
    """

    def __init__(self, columns: Optional[List[str]] = None):
        """
        Initialize StreamingAggregator.

        Args:
            columns: Columns to track (None = all numeric)
        """
        self.columns = columns
        self.n = 0  # Total rows seen
        self.sum = {}
        self.sum_sq = {}  # For variance
        self.min = {}
        self.max = {}
        self.count_non_null = {}
        self.initialized = False

    def update(self, chunk: pd.DataFrame) -> None:
        """
        Update statistics with a new chunk.

        Args:
            chunk: DataFrame chunk to process
        """
        # Initialize columns on first chunk
        if not self.initialized:
            if self.columns is None:
                self.columns = chunk.select_dtypes(include=[np.number]).columns.tolist()

            for col in self.columns:
                self.sum[col] = 0.0
                self.sum_sq[col] = 0.0
                self.min[col] = np.inf
                self.max[col] = -np.inf
                self.count_non_null[col] = 0

            self.initialized = True

        # Update counts
        self.n += len(chunk)

        # Update statistics for each column
        for col in self.columns:
            if col not in chunk.columns:
                continue

            col_data = chunk[col].dropna()

            if len(col_data) > 0:
                self.sum[col] += col_data.sum()
                self.sum_sq[col] += (col_data ** 2).sum()
                self.min[col] = min(self.min[col], col_data.min())
                self.max[col] = max(self.max[col], col_data.max())
                self.count_non_null[col] += len(col_data)

    def compute(self) -> Dict[str, Dict[str, float]]:
        """
        Compute final statistics.

        Returns:
            Dictionary mapping column names to statistics

        Example:
            >>> stats = aggregator.compute()
            >>> print(stats['column_name']['mean'])
        """
        results = {}

        for col in self.columns:
            count = self.count_non_null[col]

            if count == 0:
                results[col] = {
                    'count': 0,
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan
                }
                continue

            mean = self.sum[col] / count
            variance = (self.sum_sq[col] / count) - (mean ** 2)
            std = np.sqrt(max(0, variance))  # Avoid negative due to numerical errors

            results[col] = {
                'count': int(count),
                'mean': float(mean),
                'std': float(std),
                'min': float(self.min[col]),
                'max': float(self.max[col]),
                'missing': int(self.n - count)
            }

        return results


def process_in_chunks(
    file_path: str,
    func: Callable[[pd.DataFrame], Any],
    chunksize: int = 10000,
    combine_func: Optional[Callable[[List[Any]], Any]] = None,
    **read_kwargs
) -> Any:
    """
    Process a large file in chunks and combine results.

    Args:
        file_path: Path to file
        func: Function to apply to each chunk
        chunksize: Chunk size
        combine_func: Function to combine chunk results (default: list)
        **read_kwargs: Arguments for pd.read_csv

    Returns:
        Combined result

    Example:
        >>> def count_rows(chunk):
        ...     return len(chunk)
        >>> total = process_in_chunks(
        ...     'large.csv',
        ...     count_rows,
        ...     combine_func=sum
        ... )
    """
    reader = ChunkedDataReader(file_path, chunksize=chunksize, **read_kwargs)

    results = []
    for chunk in reader:
        try:
            result = func(chunk)
            results.append(result)
        except Exception as e:
            logger.warning(f"Error processing chunk: {e}")

    if combine_func is not None:
        return combine_func(results)
    else:
        return results


def streaming_statistics(
    file_path: str,
    chunksize: int = 10000,
    columns: Optional[List[str]] = None,
    **read_kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics on large file without loading into memory.

    Args:
        file_path: Path to file
        chunksize: Chunk size
        columns: Columns to analyze (None = all numeric)
        **read_kwargs: Arguments for pd.read_csv

    Returns:
        Dictionary with statistics

    Example:
        >>> stats = streaming_statistics('huge_file.csv', chunksize=50000)
        >>> print(f"Mean: {stats['column_name']['mean']}")
    """
    logger.info(f"Computing streaming statistics for {file_path}")

    tracker = PerformanceTracker()
    tracker.start('streaming_stats')

    aggregator = StreamingAggregator(columns=columns)
    reader = ChunkedDataReader(file_path, chunksize=chunksize, **read_kwargs)

    n_chunks = 0
    for chunk in reader:
        aggregator.update(chunk)
        n_chunks += 1

    stats = aggregator.compute()

    elapsed = tracker.stop('streaming_stats')
    logger.info(
        f"Processed {n_chunks} chunks ({aggregator.n:,} rows) in {elapsed:.2f}s"
    )

    return stats


def sample_large_file(
    file_path: str,
    n_samples: int = 10000,
    method: str = 'random',
    chunksize: int = 10000,
    **read_kwargs
) -> pd.DataFrame:
    """
    Sample from a large file without loading it entirely.

    Args:
        file_path: Path to file
        n_samples: Number of samples to collect
        method: Sampling method ('random', 'systematic', 'first', 'last')
        chunksize: Chunk size for reading
        **read_kwargs: Arguments for pd.read_csv

    Returns:
        Sampled DataFrame

    Example:
        >>> sample = sample_large_file('huge.csv', n_samples=5000, method='random')
    """
    logger.info(f"Sampling {n_samples} rows from {file_path} using {method} method")

    if method == 'first':
        # Just read first n_samples rows
        return pd.read_csv(file_path, nrows=n_samples, **read_kwargs)

    elif method == 'last':
        # Read entire file and take last n_samples (not memory efficient)
        logger.warning("'last' method requires loading entire file")
        df = pd.read_csv(file_path, **read_kwargs)
        return df.tail(n_samples).reset_index(drop=True)

    elif method == 'systematic':
        # Systematic sampling: read every nth row
        # First, count total rows
        total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header

        if total_rows <= n_samples:
            return pd.read_csv(file_path, **read_kwargs)

        step = total_rows / n_samples
        skiprows = [int(i * step) + 1 for i in range(1, total_rows) if int(i * step) not in [int(j * step) for j in range(n_samples)]]

        return pd.read_csv(file_path, skiprows=skiprows, **read_kwargs).head(n_samples)

    elif method == 'random':
        # Reservoir sampling
        reservoir = []
        reader = ChunkedDataReader(file_path, chunksize=chunksize, **read_kwargs)

        row_index = 0
        np_random = np.random.RandomState(42)

        for chunk in reader:
            for _, row in chunk.iterrows():
                if row_index < n_samples:
                    reservoir.append(row)
                else:
                    # Randomly replace with decreasing probability
                    j = np_random.randint(0, row_index + 1)
                    if j < n_samples:
                        reservoir[j] = row
                row_index += 1

        return pd.DataFrame(reservoir).reset_index(drop=True)

    else:
        raise ValueError(f"Unknown sampling method: {method}")


def chunked_value_counts(
    file_path: str,
    column: str,
    chunksize: int = 10000,
    top_n: Optional[int] = None,
    **read_kwargs
) -> pd.Series:
    """
    Compute value counts for a column in large file.

    Args:
        file_path: Path to file
        column: Column name
        chunksize: Chunk size
        top_n: Return only top N values
        **read_kwargs: Arguments for pd.read_csv

    Returns:
        Value counts as Series

    Example:
        >>> counts = chunked_value_counts('large.csv', 'category', top_n=10)
    """
    logger.info(f"Computing value counts for '{column}' in {file_path}")

    counts = {}
    reader = ChunkedDataReader(file_path, chunksize=chunksize, **read_kwargs)

    for chunk in reader:
        if column not in chunk.columns:
            raise AnalysisError(
                f"Column '{column}' not found in file",
                details={'available_columns': list(chunk.columns)}
            )

        chunk_counts = chunk[column].value_counts()
        for value, count in chunk_counts.items():
            counts[value] = counts.get(value, 0) + count

    # Convert to Series and sort
    result = pd.Series(counts).sort_values(ascending=False)

    if top_n is not None:
        result = result.head(top_n)

    return result


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'ChunkedDataReader',
    'StreamingAggregator',
    'process_in_chunks',
    'streaming_statistics',
    'sample_large_file',
    'chunked_value_counts',
]
