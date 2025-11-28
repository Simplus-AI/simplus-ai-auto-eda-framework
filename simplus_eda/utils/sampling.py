"""
Data sampling strategies for large datasets.

This module provides intelligent sampling strategies to handle large datasets
efficiently while preserving statistical properties.

Features:
- Multiple sampling methods (random, stratified, reservoir, systematic)
- Adaptive sampling based on data size
- Statistical property preservation
- Memory-efficient implementations
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any, List
from enum import Enum
import warnings

from simplus_eda.logging_config import get_logger
from simplus_eda.exceptions import InsufficientDataError, DataError

logger = get_logger(__name__)


class SamplingMethod(Enum):
    """Available sampling methods."""
    RANDOM = "random"
    STRATIFIED = "stratified"
    RESERVOIR = "reservoir"
    SYSTEMATIC = "systematic"
    ADAPTIVE = "adaptive"


class DataSampler:
    """
    Intelligent data sampler for large datasets.

    Provides various sampling strategies to reduce data size while
    preserving statistical properties.

    Example:
        >>> sampler = DataSampler(method='stratified', target_size=10000)
        >>> sample = sampler.sample(large_df, stratify_column='category')
    """

    def __init__(
        self,
        method: Union[str, SamplingMethod] = SamplingMethod.ADAPTIVE,
        target_size: Optional[int] = None,
        random_state: int = 42,
        preserve_distribution: bool = True
    ):
        """
        Initialize DataSampler.

        Args:
            method: Sampling method to use
            target_size: Target sample size (None = auto-determine)
            random_state: Random seed for reproducibility
            preserve_distribution: Try to preserve data distribution
        """
        if isinstance(method, str):
            method = SamplingMethod(method.lower())

        self.method = method
        self.target_size = target_size
        self.random_state = random_state
        self.preserve_distribution = preserve_distribution
        self.np_random = np.random.RandomState(random_state)

    def sample(
        self,
        data: pd.DataFrame,
        stratify_column: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Sample data using configured method.

        Args:
            data: DataFrame to sample
            stratify_column: Column for stratified sampling
            **kwargs: Additional method-specific arguments

        Returns:
            Sampled DataFrame

        Example:
            >>> sampler = DataSampler(method='random', target_size=1000)
            >>> sample = sampler.sample(df)
        """
        if len(data) == 0:
            raise InsufficientDataError(
                "Cannot sample from empty DataFrame",
                n_rows=0
            )

        # Determine target size if not specified
        target_size = self.target_size or self._auto_target_size(len(data))

        # If data is already small enough, return as-is
        if len(data) <= target_size:
            logger.debug(f"Data size ({len(data)}) <= target ({target_size}), no sampling needed")
            return data.copy()

        # Apply appropriate sampling method
        if self.method == SamplingMethod.RANDOM:
            return self._random_sample(data, target_size)
        elif self.method == SamplingMethod.STRATIFIED:
            return self._stratified_sample(data, target_size, stratify_column)
        elif self.method == SamplingMethod.RESERVOIR:
            return self._reservoir_sample(data, target_size)
        elif self.method == SamplingMethod.SYSTEMATIC:
            return self._systematic_sample(data, target_size)
        elif self.method == SamplingMethod.ADAPTIVE:
            return self._adaptive_sample(data, target_size, stratify_column)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")

    def _auto_target_size(self, data_size: int) -> int:
        """
        Automatically determine target sample size based on data size.

        Args:
            data_size: Original data size

        Returns:
            Recommended sample size
        """
        if data_size <= 1000:
            return data_size
        elif data_size <= 10000:
            return min(5000, data_size)
        elif data_size <= 100000:
            return min(10000, data_size)
        elif data_size <= 1000000:
            return min(50000, data_size)
        else:
            # For very large datasets, use square root rule
            return min(int(np.sqrt(data_size) * 10), 100000)

    def _random_sample(
        self,
        data: pd.DataFrame,
        target_size: int
    ) -> pd.DataFrame:
        """
        Simple random sampling.

        Args:
            data: DataFrame to sample
            target_size: Target sample size

        Returns:
            Randomly sampled DataFrame
        """
        logger.debug(f"Random sampling: {len(data)} -> {target_size}")

        sample = data.sample(
            n=target_size,
            random_state=self.random_state,
            replace=False
        )

        return sample.reset_index(drop=True)

    def _stratified_sample(
        self,
        data: pd.DataFrame,
        target_size: int,
        stratify_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Stratified sampling to preserve class distributions.

        Args:
            data: DataFrame to sample
            target_size: Target sample size
            stratify_column: Column to stratify on

        Returns:
            Stratified sampled DataFrame
        """
        if stratify_column is None:
            # Try to find a suitable categorical column
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                stratify_column = categorical_cols[0]
                logger.debug(f"Auto-selected stratify column: {stratify_column}")
            else:
                logger.warning("No categorical column for stratification, using random sampling")
                return self._random_sample(data, target_size)

        if stratify_column not in data.columns:
            raise DataError(
                f"Stratify column '{stratify_column}' not found in DataFrame",
                details={'available_columns': list(data.columns)}
            )

        logger.debug(f"Stratified sampling on '{stratify_column}': {len(data)} -> {target_size}")

        # Get class counts
        value_counts = data[stratify_column].value_counts()
        n_classes = len(value_counts)

        # Calculate samples per class (proportional)
        samples_per_class = {}
        for class_value, count in value_counts.items():
            proportion = count / len(data)
            n_samples = max(1, int(target_size * proportion))
            samples_per_class[class_value] = min(n_samples, count)

        # Sample from each class
        sampled_dfs = []
        for class_value, n_samples in samples_per_class.items():
            class_data = data[data[stratify_column] == class_value]
            if len(class_data) >= n_samples:
                class_sample = class_data.sample(
                    n=n_samples,
                    random_state=self.random_state,
                    replace=False
                )
            else:
                class_sample = class_data

            sampled_dfs.append(class_sample)

        # Combine samples
        sample = pd.concat(sampled_dfs, ignore_index=True)

        # Shuffle
        sample = sample.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)

        logger.debug(f"Stratified sample: {len(sample)} rows from {n_classes} classes")

        return sample

    def _reservoir_sample(
        self,
        data: pd.DataFrame,
        target_size: int
    ) -> pd.DataFrame:
        """
        Reservoir sampling (useful for streaming data).

        Args:
            data: DataFrame to sample
            target_size: Target sample size

        Returns:
            Reservoir sampled DataFrame
        """
        logger.debug(f"Reservoir sampling: {len(data)} -> {target_size}")

        # Initialize reservoir
        reservoir = []

        for i, (idx, row) in enumerate(data.iterrows()):
            if i < target_size:
                reservoir.append(idx)
            else:
                # Randomly replace elements with decreasing probability
                j = self.np_random.randint(0, i + 1)
                if j < target_size:
                    reservoir[j] = idx

        sample = data.loc[reservoir].reset_index(drop=True)

        return sample

    def _systematic_sample(
        self,
        data: pd.DataFrame,
        target_size: int
    ) -> pd.DataFrame:
        """
        Systematic sampling (every nth row).

        Args:
            data: DataFrame to sample
            target_size: Target sample size

        Returns:
            Systematically sampled DataFrame
        """
        logger.debug(f"Systematic sampling: {len(data)} -> {target_size}")

        # Calculate step size
        step = len(data) / target_size

        # Select indices
        indices = [int(i * step) for i in range(target_size)]

        sample = data.iloc[indices].reset_index(drop=True)

        return sample

    def _adaptive_sample(
        self,
        data: pd.DataFrame,
        target_size: int,
        stratify_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Adaptive sampling - choose best method based on data characteristics.

        Args:
            data: DataFrame to sample
            target_size: Target sample size
            stratify_column: Optional column for stratification

        Returns:
            Adaptively sampled DataFrame
        """
        logger.debug("Adaptive sampling: analyzing data characteristics...")

        # Check for categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns

        # Check data size
        data_size = len(data)

        # Decision logic
        if self.preserve_distribution and len(categorical_cols) > 0:
            # Use stratified sampling if we want to preserve distribution
            logger.debug("Using stratified sampling (categorical columns found)")
            return self._stratified_sample(data, target_size, stratify_column)

        elif data_size > 1000000:
            # For very large data, use systematic sampling (faster)
            logger.debug("Using systematic sampling (very large dataset)")
            return self._systematic_sample(data, target_size)

        else:
            # Default to random sampling
            logger.debug("Using random sampling (default)")
            return self._random_sample(data, target_size)


def smart_sample(
    data: pd.DataFrame,
    target_size: Optional[int] = None,
    method: str = 'adaptive',
    stratify_column: Optional[str] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Intelligently sample data with automatic method selection.

    Args:
        data: DataFrame to sample
        target_size: Target sample size (None = auto)
        method: Sampling method ('random', 'stratified', 'adaptive', etc.)
        stratify_column: Column for stratified sampling
        random_state: Random seed

    Returns:
        Tuple of (sampled_df, metadata_dict)

    Example:
        >>> sample, info = smart_sample(large_df, target_size=10000)
        >>> print(f"Sampled {len(sample)} rows using {info['method']}")
    """
    sampler = DataSampler(
        method=method,
        target_size=target_size,
        random_state=random_state
    )

    original_size = len(data)
    sample = sampler.sample(data, stratify_column=stratify_column)
    sample_size = len(sample)

    metadata = {
        'original_size': original_size,
        'sample_size': sample_size,
        'sampling_rate': sample_size / original_size if original_size > 0 else 0,
        'method': sampler.method.value,
        'target_size': sampler.target_size,
        'sampled': sample_size < original_size
    }

    if sample_size < original_size:
        logger.info(
            f"Sampled {sample_size:,} rows from {original_size:,} "
            f"({metadata['sampling_rate']:.1%}) using {metadata['method']}"
        )

    return sample, metadata


def sample_for_visualization(
    data: pd.DataFrame,
    max_points: int = 10000,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Sample data specifically for visualization purposes.

    Args:
        data: DataFrame to sample
        max_points: Maximum number of points for visualization
        columns: Specific columns to consider (None = all)

    Returns:
        Sampled DataFrame suitable for visualization

    Example:
        >>> viz_data = sample_for_visualization(large_df, max_points=5000)
        >>> plt.scatter(viz_data['x'], viz_data['y'])
    """
    if columns is not None:
        data = data[columns]

    if len(data) <= max_points:
        return data

    # Use stratified sampling if there's a clear categorical variable
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns

    if len(categorical_cols) > 0:
        # Find column with reasonable number of categories
        for col in categorical_cols:
            n_unique = data[col].nunique()
            if 2 <= n_unique <= 20:
                sample, _ = smart_sample(
                    data,
                    target_size=max_points,
                    method='stratified',
                    stratify_column=col
                )
                return sample

    # Otherwise, use random sampling
    sample, _ = smart_sample(data, target_size=max_points, method='random')
    return sample


def adaptive_chunk_size(
    data_size: int,
    available_memory_mb: Optional[float] = None,
    row_size_bytes: Optional[int] = None
) -> int:
    """
    Calculate adaptive chunk size based on data and memory constraints.

    Args:
        data_size: Total number of rows
        available_memory_mb: Available memory in MB (None = auto-detect)
        row_size_bytes: Average row size in bytes (None = estimate)

    Returns:
        Recommended chunk size

    Example:
        >>> chunk_size = adaptive_chunk_size(1000000, available_memory_mb=1024)
        >>> for chunk in pd.read_csv('large.csv', chunksize=chunk_size):
        ...     process(chunk)
    """
    if available_memory_mb is None:
        # Try to detect available memory
        try:
            import psutil
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            # Conservative estimate: 1GB
            available_memory_mb = 1024

    # Estimate row size if not provided
    if row_size_bytes is None:
        # Conservative estimate: 1KB per row
        row_size_bytes = 1024

    # Use 10% of available memory for chunk
    target_chunk_memory_mb = available_memory_mb * 0.1
    target_chunk_memory_bytes = target_chunk_memory_mb * 1024 * 1024

    # Calculate chunk size
    chunk_size = int(target_chunk_memory_bytes / row_size_bytes)

    # Constraints
    min_chunk_size = 1000
    max_chunk_size = 100000

    chunk_size = max(min_chunk_size, min(chunk_size, max_chunk_size))

    logger.debug(
        f"Adaptive chunk size: {chunk_size:,} rows "
        f"(available memory: {available_memory_mb:.0f}MB)"
    )

    return chunk_size


def estimate_memory_usage(
    data: pd.DataFrame,
    deep: bool = True
) -> Dict[str, Any]:
    """
    Estimate memory usage of DataFrame.

    Args:
        data: DataFrame to analyze
        deep: Include deep memory usage (object types)

    Returns:
        Dictionary with memory usage information

    Example:
        >>> mem_info = estimate_memory_usage(df)
        >>> print(f"Total: {mem_info['total_mb']:.2f} MB")
    """
    memory_usage = data.memory_usage(deep=deep)

    total_bytes = memory_usage.sum()
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_mb / 1024

    per_column = {
        col: memory_usage[col] / (1024 * 1024)
        for col in data.columns
    }

    per_row_bytes = total_bytes / len(data) if len(data) > 0 else 0

    return {
        'total_bytes': int(total_bytes),
        'total_mb': total_mb,
        'total_gb': total_gb,
        'per_row_bytes': per_row_bytes,
        'per_column_mb': per_column,
        'n_rows': len(data),
        'n_columns': len(data.columns)
    }


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'DataSampler',
    'SamplingMethod',
    'smart_sample',
    'sample_for_visualization',
    'adaptive_chunk_size',
    'estimate_memory_usage',
]
