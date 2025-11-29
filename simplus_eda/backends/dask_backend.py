"""
Dask backend for large-scale out-of-core computation.

This module provides Dask-based implementations for analyzing datasets
that don't fit in memory.

Features:
- Lazy evaluation for large datasets
- Distributed computing support
- Out-of-core algorithms
- Automatic pandas fallback
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union, List, TYPE_CHECKING
import warnings

from simplus_eda.logging_config import get_logger
from simplus_eda.exceptions import DependencyError, AnalysisError

logger = get_logger(__name__)

# Try to import Dask
try:
    import dask
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    dd = None

    # Create dummy dd module for type hints when dask not available
    if TYPE_CHECKING:
        import dask.dataframe as dd
    else:
        # Create a placeholder for runtime
        class _DummyDD:
            DataFrame = type('DataFrame', (), {})
            Series = type('Series', (), {})
        dd = _DummyDD()


def check_dask_available():
    """Check if Dask is available, raise error if not."""
    if not DASK_AVAILABLE:
        raise DependencyError(
            "Dask is required for large dataset support",
            missing_package="dask",
            install_command="pip install dask[complete]"
        )


class DaskBackend:
    """
    Dask backend for large-scale data analysis.

    Provides Dask-based implementations of common EDA operations
    for datasets that don't fit in memory.

    Example:
        >>> backend = DaskBackend(npartitions=10)
        >>> ddf = backend.from_pandas(large_df)
        >>> stats = backend.compute_statistics(ddf)
    """

    def __init__(
        self,
        npartitions: Optional[int] = None,
        scheduler: str = 'threads',
        enable_progress: bool = True
    ):
        """
        Initialize Dask backend.

        Args:
            npartitions: Number of partitions (None = auto)
            scheduler: Dask scheduler ('threads', 'processes', 'synchronous')
            enable_progress: Show progress bars
        """
        check_dask_available()

        self.npartitions = npartitions
        self.scheduler = scheduler
        self.enable_progress = enable_progress

    def from_pandas(
        self,
        df: pd.DataFrame,
        npartitions: Optional[int] = None
    ) -> dd.DataFrame:
        """
        Convert pandas DataFrame to Dask DataFrame.

        Args:
            df: Pandas DataFrame
            npartitions: Number of partitions (None = auto)

        Returns:
            Dask DataFrame
        """
        npartitions = npartitions or self.npartitions or self._auto_npartitions(len(df))

        logger.debug(f"Converting to Dask DataFrame with {npartitions} partitions")

        ddf = dd.from_pandas(df, npartitions=npartitions)

        return ddf

    def from_csv(
        self,
        path: str,
        **kwargs
    ) -> dd.DataFrame:
        """
        Read CSV file(s) as Dask DataFrame.

        Args:
            path: File path or glob pattern
            **kwargs: Additional arguments for dd.read_csv

        Returns:
            Dask DataFrame
        """
        logger.debug(f"Reading CSV with Dask: {path}")

        ddf = dd.read_csv(path, **kwargs)

        return ddf

    def to_pandas(
        self,
        ddf: dd.DataFrame,
        max_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Convert Dask DataFrame to pandas DataFrame.

        Args:
            ddf: Dask DataFrame
            max_rows: Maximum rows to convert (None = all)

        Returns:
            Pandas DataFrame
        """
        if max_rows is not None and len(ddf) > max_rows:
            logger.warning(
                f"DataFrame has {len(ddf):,} rows, sampling {max_rows:,} for conversion"
            )
            ddf = ddf.head(max_rows, npartitions=-1)

        logger.debug("Computing Dask DataFrame to pandas")

        with self._progress_context():
            df = ddf.compute(scheduler=self.scheduler)

        return df

    def compute_statistics(
        self,
        ddf: dd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute descriptive statistics using Dask.

        Args:
            ddf: Dask DataFrame
            columns: Columns to analyze (None = all numeric)

        Returns:
            Dictionary with statistics
        """
        if columns is None:
            columns = ddf.select_dtypes(include=[np.number]).columns.tolist()

        logger.debug(f"Computing statistics for {len(columns)} columns with Dask")

        stats = {}

        with self._progress_context():
            for col in columns:
                try:
                    col_data = ddf[col].dropna()

                    stats[col] = {
                        'count': int(col_data.count().compute()),
                        'mean': float(col_data.mean().compute()),
                        'std': float(col_data.std().compute()),
                        'min': float(col_data.min().compute()),
                        'max': float(col_data.max().compute()),
                        '25%': float(col_data.quantile(0.25).compute()),
                        '50%': float(col_data.quantile(0.50).compute()),
                        '75%': float(col_data.quantile(0.75).compute()),
                    }
                except Exception as e:
                    logger.warning(f"Failed to compute statistics for {col}: {e}")
                    stats[col] = {'error': str(e)}

        return stats

    def compute_correlation(
        self,
        ddf: dd.DataFrame,
        method: str = 'pearson',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute correlation matrix using Dask.

        Args:
            ddf: Dask DataFrame
            method: Correlation method ('pearson', 'spearman')
            columns: Columns to include (None = all numeric)

        Returns:
            Correlation matrix as pandas DataFrame
        """
        if columns is None:
            columns = ddf.select_dtypes(include=[np.number]).columns.tolist()

        logger.debug(f"Computing {method} correlation for {len(columns)} columns with Dask")

        # Select numeric columns
        ddf_numeric = ddf[columns]

        with self._progress_context():
            if method == 'pearson':
                # Compute correlation matrix
                corr_matrix = ddf_numeric.corr().compute()
            else:
                # For other methods, need to convert to pandas
                logger.warning(f"Method '{method}' requires pandas, converting subset")
                df_sample = ddf_numeric.head(100000, npartitions=-1).compute()
                corr_matrix = df_sample.corr(method=method)

        return corr_matrix

    def detect_outliers_iqr(
        self,
        ddf: dd.DataFrame,
        column: str,
        multiplier: float = 1.5
    ) -> dd.Series:
        """
        Detect outliers using IQR method with Dask.

        Args:
            ddf: Dask DataFrame
            column: Column name
            multiplier: IQR multiplier

        Returns:
            Dask Series with boolean outlier mask
        """
        logger.debug(f"Detecting outliers in '{column}' with IQR method")

        col_data = ddf[column].dropna()

        with self._progress_context():
            q1 = col_data.quantile(0.25).compute()
            q3 = col_data.quantile(0.75).compute()

        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        outlier_mask = (ddf[column] < lower_bound) | (ddf[column] > upper_bound)

        return outlier_mask

    def value_counts(
        self,
        ddf: dd.DataFrame,
        column: str,
        top_n: Optional[int] = None
    ) -> pd.Series:
        """
        Compute value counts using Dask.

        Args:
            ddf: Dask DataFrame
            column: Column name
            top_n: Return only top N values (None = all)

        Returns:
            Value counts as pandas Series
        """
        logger.debug(f"Computing value counts for '{column}'")

        with self._progress_context():
            counts = ddf[column].value_counts().compute()

        if top_n is not None:
            counts = counts.head(top_n)

        return counts

    def missing_values(
        self,
        ddf: dd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze missing values using Dask.

        Args:
            ddf: Dask DataFrame

        Returns:
            Dictionary with missing value statistics
        """
        logger.debug("Analyzing missing values with Dask")

        with self._progress_context():
            total_rows = len(ddf)
            missing_counts = ddf.isnull().sum().compute()

        missing_info = {}
        for col in ddf.columns:
            missing_count = int(missing_counts[col])
            missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0

            missing_info[col] = {
                'count': missing_count,
                'percentage': missing_pct
            }

        return missing_info

    def sample(
        self,
        ddf: dd.DataFrame,
        n: Optional[int] = None,
        frac: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Sample from Dask DataFrame.

        Args:
            ddf: Dask DataFrame
            n: Number of rows to sample
            frac: Fraction of rows to sample

        Returns:
            Sampled pandas DataFrame
        """
        logger.debug(f"Sampling from Dask DataFrame (n={n}, frac={frac})")

        if n is not None:
            # Sample specific number of rows
            sample = ddf.sample(frac=n/len(ddf), random_state=42)
        elif frac is not None:
            # Sample fraction
            sample = ddf.sample(frac=frac, random_state=42)
        else:
            # Default: sample 10%
            sample = ddf.sample(frac=0.1, random_state=42)

        with self._progress_context():
            df_sample = sample.compute(scheduler=self.scheduler)

        return df_sample

    def _auto_npartitions(self, n_rows: int) -> int:
        """
        Automatically determine number of partitions.

        Args:
            n_rows: Number of rows

        Returns:
            Recommended number of partitions
        """
        # Aim for 10,000 - 100,000 rows per partition
        if n_rows <= 100000:
            return 4
        elif n_rows <= 1000000:
            return 10
        elif n_rows <= 10000000:
            return 50
        else:
            return 100

    def _progress_context(self):
        """Context manager for progress bars."""
        if self.enable_progress:
            return ProgressBar()
        else:
            # No-op context manager
            class NoOpContext:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return NoOpContext()


def use_dask_if_large(
    df: pd.DataFrame,
    threshold: int = 100000,
    npartitions: Optional[int] = None
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Automatically convert to Dask DataFrame if data is large.

    Args:
        df: Pandas DataFrame
        threshold: Size threshold for using Dask
        npartitions: Number of partitions for Dask

    Returns:
        Pandas or Dask DataFrame

    Example:
        >>> data = use_dask_if_large(large_df, threshold=50000)
        >>> if isinstance(data, dd.DataFrame):
        ...     # Use Dask operations
        ...     result = data.mean().compute()
        ... else:
        ...     # Use pandas operations
        ...     result = data.mean()
    """
    if len(df) > threshold and DASK_AVAILABLE:
        logger.info(f"Dataset size ({len(df):,}) exceeds threshold ({threshold:,}), using Dask")
        backend = DaskBackend(npartitions=npartitions)
        return backend.from_pandas(df)
    else:
        return df


def compute_if_dask(data: Union[pd.DataFrame, dd.DataFrame]) -> pd.DataFrame:
    """
    Compute Dask DataFrame to pandas if needed.

    Args:
        data: Pandas or Dask DataFrame

    Returns:
        Pandas DataFrame

    Example:
        >>> result = compute_if_dask(data)  # Always returns pandas DataFrame
    """
    if DASK_AVAILABLE and isinstance(data, dd.DataFrame):
        logger.debug("Computing Dask DataFrame to pandas")
        return data.compute()
    else:
        return data


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'DaskBackend',
    'DASK_AVAILABLE',
    'check_dask_available',
    'use_dask_if_large',
    'compute_if_dask',
]
