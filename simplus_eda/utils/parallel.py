"""
Parallel processing utilities for EDA operations.

This module provides utilities for parallel processing of data analysis tasks,
including multiprocessing support for column-wise operations and batch processing.
"""

import pandas as pd
import numpy as np
from typing import Callable, List, Dict, Any, Optional, Tuple
from functools import partial
from multiprocessing import Pool, cpu_count
import warnings


class ParallelProcessor:
    """
    Handles parallel processing for EDA operations.

    Supports:
    - Column-wise parallel analysis
    - Batch processing of multiple datasets
    - Progress tracking for parallel operations
    - Automatic CPU count detection
    """

    def __init__(self, n_jobs: int = 1, verbose: bool = False):
        """
        Initialize ParallelProcessor.

        Args:
            n_jobs: Number of parallel jobs
                    -1 = use all CPUs
                    1 = sequential processing (no parallelization)
                    >1 = use specified number of CPUs
            verbose: Enable progress output
        """
        self.n_jobs = self._determine_n_jobs(n_jobs)
        self.verbose = verbose

    def _determine_n_jobs(self, n_jobs: int) -> int:
        """
        Determine the actual number of jobs to use.

        Args:
            n_jobs: Requested number of jobs

        Returns:
            Actual number of jobs (between 1 and cpu_count)
        """
        if n_jobs == -1:
            return cpu_count()
        elif n_jobs < -1:
            # n_jobs = -2 means use all but one CPU, etc.
            return max(1, cpu_count() + 1 + n_jobs)
        elif n_jobs == 0:
            return 1
        else:
            return min(n_jobs, cpu_count())

    def is_parallel(self) -> bool:
        """Check if parallel processing is enabled."""
        return self.n_jobs > 1

    def process_columns_parallel(
        self,
        data: pd.DataFrame,
        func: Callable,
        columns: Optional[List[str]] = None,
        **func_kwargs
    ) -> Dict[str, Any]:
        """
        Process DataFrame columns in parallel.

        Args:
            data: Input DataFrame
            func: Function to apply to each column (signature: func(column_data, column_name, **kwargs))
            columns: List of column names to process (None = all columns)
            **func_kwargs: Additional keyword arguments for func

        Returns:
            Dictionary mapping column names to results

        Example:
            >>> def analyze_column(col_data, col_name, threshold=0.5):
            ...     return {"mean": col_data.mean(), "std": col_data.std()}
            >>> processor = ParallelProcessor(n_jobs=-1)
            >>> results = processor.process_columns_parallel(df, analyze_column, threshold=0.6)
        """
        if columns is None:
            columns = list(data.columns)

        if self.verbose:
            print(f"Processing {len(columns)} columns with {self.n_jobs} workers...")

        if not self.is_parallel() or len(columns) == 1:
            # Sequential processing
            results = {}
            for col in columns:
                if self.verbose:
                    print(f"  Processing: {col}")
                results[col] = func(data[col], col, **func_kwargs)
            return results

        # Parallel processing
        with Pool(processes=self.n_jobs) as pool:
            # Create partial function with kwargs
            process_func = partial(
                _process_single_column,
                data=data,
                func=func,
                func_kwargs=func_kwargs
            )

            # Map columns to results
            column_results = pool.map(process_func, columns)

        # Combine results
        results = dict(zip(columns, column_results))

        if self.verbose:
            print(f"✓ Completed processing {len(columns)} columns")

        return results

    def process_numeric_columns_parallel(
        self,
        data: pd.DataFrame,
        func: Callable,
        **func_kwargs
    ) -> Dict[str, Any]:
        """
        Process only numeric columns in parallel.

        Args:
            data: Input DataFrame
            func: Function to apply to each numeric column
            **func_kwargs: Additional keyword arguments for func

        Returns:
            Dictionary mapping column names to results
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            if self.verbose:
                print("No numeric columns found")
            return {}

        return self.process_columns_parallel(
            data=data,
            func=func,
            columns=numeric_cols,
            **func_kwargs
        )

    def process_categorical_columns_parallel(
        self,
        data: pd.DataFrame,
        func: Callable,
        **func_kwargs
    ) -> Dict[str, Any]:
        """
        Process only categorical/object columns in parallel.

        Args:
            data: Input DataFrame
            func: Function to apply to each categorical column
            **func_kwargs: Additional keyword arguments for func

        Returns:
            Dictionary mapping column names to results
        """
        cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        if not cat_cols:
            if self.verbose:
                print("No categorical columns found")
            return {}

        return self.process_columns_parallel(
            data=data,
            func=func,
            columns=cat_cols,
            **func_kwargs
        )

    def process_pairwise_parallel(
        self,
        data: pd.DataFrame,
        func: Callable,
        columns: Optional[List[str]] = None,
        include_self: bool = False,
        **func_kwargs
    ) -> Dict[Tuple[str, str], Any]:
        """
        Process column pairs in parallel (useful for correlation, covariance, etc.).

        Args:
            data: Input DataFrame
            func: Function to apply to each column pair
                  Signature: func(col1_data, col2_data, col1_name, col2_name, **kwargs)
            columns: List of column names (None = all numeric columns)
            include_self: Include pairs like (col1, col1)
            **func_kwargs: Additional keyword arguments for func

        Returns:
            Dictionary mapping (col1, col2) tuples to results

        Example:
            >>> def compute_correlation(col1, col2, col1_name, col2_name):
            ...     return col1.corr(col2)
            >>> processor = ParallelProcessor(n_jobs=4)
            >>> corr_results = processor.process_pairwise_parallel(df, compute_correlation)
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Generate column pairs
        pairs = []
        for i, col1 in enumerate(columns):
            start_j = i if include_self else i + 1
            for j in range(start_j, len(columns)):
                col2 = columns[j]
                pairs.append((col1, col2))

        if self.verbose:
            print(f"Processing {len(pairs)} column pairs with {self.n_jobs} workers...")

        if not self.is_parallel() or len(pairs) == 1:
            # Sequential processing
            results = {}
            for col1, col2 in pairs:
                if self.verbose:
                    print(f"  Processing: ({col1}, {col2})")
                results[(col1, col2)] = func(
                    data[col1], data[col2], col1, col2, **func_kwargs
                )
            return results

        # Parallel processing
        with Pool(processes=self.n_jobs) as pool:
            process_func = partial(
                _process_column_pair,
                data=data,
                func=func,
                func_kwargs=func_kwargs
            )

            pair_results = pool.map(process_func, pairs)

        results = dict(zip(pairs, pair_results))

        if self.verbose:
            print(f"✓ Completed processing {len(pairs)} pairs")

        return results

    def process_batch_parallel(
        self,
        items: List[Any],
        func: Callable,
        **func_kwargs
    ) -> List[Any]:
        """
        Process a batch of items in parallel.

        Args:
            items: List of items to process
            func: Function to apply to each item
            **func_kwargs: Additional keyword arguments for func

        Returns:
            List of results in the same order as items

        Example:
            >>> def analyze_dataset(file_path):
            ...     df = pd.read_csv(file_path)
            ...     return df.describe()
            >>> processor = ParallelProcessor(n_jobs=4)
            >>> file_paths = ['data1.csv', 'data2.csv', 'data3.csv']
            >>> results = processor.process_batch_parallel(file_paths, analyze_dataset)
        """
        if self.verbose:
            print(f"Processing {len(items)} items with {self.n_jobs} workers...")

        if not self.is_parallel() or len(items) == 1:
            # Sequential processing
            results = []
            for i, item in enumerate(items):
                if self.verbose:
                    print(f"  Processing item {i+1}/{len(items)}")
                results.append(func(item, **func_kwargs))
            return results

        # Parallel processing
        with Pool(processes=self.n_jobs) as pool:
            if func_kwargs:
                process_func = partial(func, **func_kwargs)
                results = pool.map(process_func, items)
            else:
                results = pool.map(func, items)

        if self.verbose:
            print(f"✓ Completed processing {len(items)} items")

        return results

    def map_reduce_parallel(
        self,
        data: pd.DataFrame,
        map_func: Callable,
        reduce_func: Callable,
        columns: Optional[List[str]] = None,
        **func_kwargs
    ) -> Any:
        """
        Apply map-reduce pattern in parallel.

        Args:
            data: Input DataFrame
            map_func: Function to map over columns
            reduce_func: Function to reduce results
            columns: Columns to process (None = all)
            **func_kwargs: Additional arguments for map_func

        Returns:
            Reduced result

        Example:
            >>> # Calculate total variance across all columns
            >>> def map_variance(col_data, col_name):
            ...     return col_data.var()
            >>> def reduce_sum(results):
            ...     return sum(results.values())
            >>> processor = ParallelProcessor(n_jobs=4)
            >>> total_var = processor.map_reduce_parallel(df, map_variance, reduce_sum)
        """
        # Map phase
        map_results = self.process_columns_parallel(
            data=data,
            func=map_func,
            columns=columns,
            **func_kwargs
        )

        # Reduce phase
        if self.verbose:
            print("Reducing results...")

        reduced_result = reduce_func(map_results)

        return reduced_result


# Helper functions for multiprocessing (must be module-level for pickling)

def _process_single_column(
    column_name: str,
    data: pd.DataFrame,
    func: Callable,
    func_kwargs: Dict[str, Any]
) -> Any:
    """
    Process a single column (helper for multiprocessing).

    Args:
        column_name: Name of the column
        data: Full DataFrame
        func: Function to apply
        func_kwargs: Keyword arguments for func

    Returns:
        Result of func applied to the column
    """
    try:
        return func(data[column_name], column_name, **func_kwargs)
    except Exception as e:
        # Return error info instead of crashing the worker
        return {"error": str(e), "column": column_name}


def _process_column_pair(
    pair: Tuple[str, str],
    data: pd.DataFrame,
    func: Callable,
    func_kwargs: Dict[str, Any]
) -> Any:
    """
    Process a column pair (helper for multiprocessing).

    Args:
        pair: Tuple of (col1_name, col2_name)
        data: Full DataFrame
        func: Function to apply
        func_kwargs: Keyword arguments for func

    Returns:
        Result of func applied to the column pair
    """
    col1_name, col2_name = pair
    try:
        return func(
            data[col1_name],
            data[col2_name],
            col1_name,
            col2_name,
            **func_kwargs
        )
    except Exception as e:
        return {"error": str(e), "pair": pair}


def get_optimal_chunk_size(total_items: int, n_jobs: int) -> int:
    """
    Calculate optimal chunk size for parallel processing.

    Args:
        total_items: Total number of items to process
        n_jobs: Number of parallel jobs

    Returns:
        Optimal chunk size
    """
    if n_jobs <= 1:
        return total_items

    # Aim for at least 4 chunks per worker for load balancing
    min_chunks_per_worker = 4
    ideal_chunk_count = n_jobs * min_chunks_per_worker

    chunk_size = max(1, total_items // ideal_chunk_count)

    return chunk_size


def parallel_apply_with_progress(
    data: pd.DataFrame,
    func: Callable,
    n_jobs: int = 1,
    verbose: bool = False,
    desc: str = "Processing"
) -> pd.DataFrame:
    """
    Apply a function to DataFrame with optional parallel processing and progress.

    Args:
        data: Input DataFrame
        func: Function to apply row-wise or column-wise
        n_jobs: Number of parallel jobs
        verbose: Show progress
        desc: Description for progress output

    Returns:
        Transformed DataFrame
    """
    processor = ParallelProcessor(n_jobs=n_jobs, verbose=verbose)

    if processor.is_parallel():
        # Split data into chunks
        chunk_size = get_optimal_chunk_size(len(data), n_jobs)
        chunks = [data.iloc[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

        if verbose:
            print(f"{desc}: Processing {len(chunks)} chunks...")

        # Process chunks in parallel
        results = processor.process_batch_parallel(chunks, lambda chunk: chunk.apply(func))

        # Combine results
        return pd.concat(results, ignore_index=True)
    else:
        # Sequential processing
        if verbose:
            print(f"{desc}: Sequential processing...")
        return data.apply(func)
