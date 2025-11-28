"""
Tests for large dataset support features.

This module tests data sampling, memory optimization, chunked processing,
and Dask backend functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from simplus_eda.utils.sampling import (
    DataSampler,
    SamplingMethod,
    smart_sample,
    estimate_sample_size,
    validate_sample_representativeness,
)
from simplus_eda.utils.memory import (
    MemoryProfiler,
    get_system_memory_info,
    optimize_dataframe_memory,
    check_memory_available,
    suggest_optimizations,
    estimate_dataframe_memory,
    force_garbage_collection,
    PSUTIL_AVAILABLE,
)
from simplus_eda.utils.chunked import (
    ChunkedDataReader,
    StreamingAggregator,
    process_in_chunks,
    streaming_statistics,
    sample_large_file,
    chunked_value_counts,
)

# Try to import Dask backend
try:
    from simplus_eda.backends.dask_backend import DaskBackend, check_dask_available
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(10000),
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 10000),
        'value': np.random.randn(10000),
        'price': np.random.uniform(10, 100, 10000),
        'quantity': np.random.randint(1, 100, 10000),
    })


@pytest.fixture
def imbalanced_dataframe():
    """Create an imbalanced DataFrame for stratified sampling tests."""
    np.random.seed(42)
    # 80% class A, 15% class B, 5% class C
    n_total = 10000
    categories = ['A'] * 8000 + ['B'] * 1500 + ['C'] * 500

    return pd.DataFrame({
        'id': range(n_total),
        'category': categories,
        'value': np.random.randn(n_total),
    })


@pytest.fixture
def temp_csv_file(sample_dataframe):
    """Create a temporary CSV file for chunked processing tests."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_dataframe.to_csv(f.name, index=False)
        yield f.name
    # Cleanup
    if os.path.exists(f.name):
        os.remove(f.name)


@pytest.fixture
def large_csv_file():
    """Create a large temporary CSV file."""
    np.random.seed(42)
    n_rows = 50000

    df = pd.DataFrame({
        'id': range(n_rows),
        'category': np.random.choice(['A', 'B', 'C'], n_rows),
        'value': np.random.randn(n_rows),
        'price': np.random.uniform(10, 100, n_rows),
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    # Cleanup
    if os.path.exists(f.name):
        os.remove(f.name)


# ============================================================================
# Data Sampling Tests
# ============================================================================

class TestDataSampler:
    """Tests for DataSampler class."""

    def test_random_sampling(self, sample_dataframe):
        """Test random sampling produces correct size."""
        sampler = DataSampler(method='random', target_size=1000, random_state=42)
        sample = sampler.sample(sample_dataframe)

        assert len(sample) == 1000
        assert set(sample.columns) == set(sample_dataframe.columns)

    def test_stratified_sampling(self, imbalanced_dataframe):
        """Test stratified sampling preserves distribution."""
        sampler = DataSampler(method='stratified', target_size=1000, random_state=42)
        sample = sampler.sample(imbalanced_dataframe, stratify_column='category')

        assert len(sample) == 1000

        # Check distribution is similar
        original_dist = imbalanced_dataframe['category'].value_counts(normalize=True)
        sample_dist = sample['category'].value_counts(normalize=True)

        # Allow 10% deviation
        for category in original_dist.index:
            assert abs(original_dist[category] - sample_dist[category]) < 0.1

    def test_reservoir_sampling(self, sample_dataframe):
        """Test reservoir sampling."""
        sampler = DataSampler(method='reservoir', target_size=1000, random_state=42)
        sample = sampler.sample(sample_dataframe)

        assert len(sample) == 1000
        assert set(sample.columns) == set(sample_dataframe.columns)

    def test_systematic_sampling(self, sample_dataframe):
        """Test systematic sampling."""
        sampler = DataSampler(method='systematic', target_size=1000, random_state=42)
        sample = sampler.sample(sample_dataframe)

        assert len(sample) == 1000

    def test_adaptive_sampling(self, sample_dataframe):
        """Test adaptive sampling chooses appropriate method."""
        sampler = DataSampler(method='adaptive', target_size=1000, random_state=42)
        sample = sampler.sample(sample_dataframe)

        assert len(sample) == 1000

    def test_sampling_larger_than_data(self, sample_dataframe):
        """Test sampling when target size > data size returns original data."""
        sampler = DataSampler(method='random', target_size=20000, random_state=42)
        sample = sampler.sample(sample_dataframe)

        assert len(sample) == len(sample_dataframe)

    def test_sampling_method_enum(self):
        """Test SamplingMethod enum."""
        assert SamplingMethod.RANDOM.value == 'random'
        assert SamplingMethod.STRATIFIED.value == 'stratified'
        assert SamplingMethod.ADAPTIVE.value == 'adaptive'


class TestSmartSample:
    """Tests for smart_sample function."""

    def test_smart_sample_basic(self, sample_dataframe):
        """Test smart_sample basic functionality."""
        sample, metadata = smart_sample(sample_dataframe, target_size=1000)

        assert len(sample) == 1000
        assert metadata['original_size'] == 10000
        assert metadata['sample_size'] == 1000
        assert 'method_used' in metadata
        assert 'sampling_ratio' in metadata

    def test_smart_sample_adaptive(self, imbalanced_dataframe):
        """Test smart_sample with adaptive method."""
        sample, metadata = smart_sample(
            imbalanced_dataframe,
            method='adaptive',
            target_size=1000
        )

        assert len(sample) == 1000
        assert 'recommendation_reason' in metadata

    def test_smart_sample_no_sampling_needed(self, sample_dataframe):
        """Test when no sampling is needed."""
        small_df = sample_dataframe.head(100)
        sample, metadata = smart_sample(small_df, target_size=1000)

        assert len(sample) == 100
        assert not metadata['was_sampled']


class TestSampleEstimation:
    """Tests for sample size estimation."""

    def test_estimate_sample_size(self):
        """Test sample size estimation."""
        # For large population, sample size should converge
        sample_size = estimate_sample_size(
            population_size=1000000,
            confidence_level=0.95,
            margin_of_error=0.05
        )

        # Should be around 384 for these parameters
        assert 300 < sample_size < 500

    def test_estimate_sample_size_small_population(self):
        """Test sample size for small population."""
        sample_size = estimate_sample_size(
            population_size=100,
            confidence_level=0.95,
            margin_of_error=0.05
        )

        # For small population, should recommend most of it
        assert sample_size <= 100

    def test_validate_sample_representativeness(self, sample_dataframe):
        """Test sample representativeness validation."""
        sampler = DataSampler(method='random', target_size=1000, random_state=42)
        sample = sampler.sample(sample_dataframe)

        is_representative, metrics = validate_sample_representativeness(
            original=sample_dataframe,
            sample=sample,
            columns=['value', 'price']
        )

        assert isinstance(is_representative, bool)
        assert 'mean_diff' in metrics
        assert 'std_diff' in metrics


# ============================================================================
# Memory Management Tests
# ============================================================================

class TestMemoryProfiler:
    """Tests for MemoryProfiler class."""

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_memory_profiler_basic(self, sample_dataframe):
        """Test basic memory profiling."""
        profiler = MemoryProfiler()

        profiler.start('test_operation')
        # Allocate some memory
        df = sample_dataframe.copy()
        df['new_column'] = np.random.randn(len(df))
        stats = profiler.stop('test_operation')

        assert 'start_memory_mb' in stats
        assert 'end_memory_mb' in stats
        assert 'memory_change_mb' in stats

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_memory_profiler_report(self):
        """Test memory profiler report generation."""
        profiler = MemoryProfiler()

        profiler.start('op1')
        x = [i for i in range(10000)]
        profiler.stop('op1')

        profiler.start('op2')
        y = {i: i**2 for i in range(10000)}
        profiler.stop('op2')

        # Should not raise
        profiler.report()

    def test_memory_profiler_without_psutil(self, monkeypatch):
        """Test memory profiler when psutil is not available."""
        # Temporarily disable psutil
        import simplus_eda.utils.memory as mem_module
        original_value = mem_module.PSUTIL_AVAILABLE
        mem_module.PSUTIL_AVAILABLE = False

        try:
            profiler = MemoryProfiler()
            profiler.start('test')
            profiler.stop('test')
            # Should not raise, just return 0.0 for memory
        finally:
            mem_module.PSUTIL_AVAILABLE = original_value


class TestMemoryOptimization:
    """Tests for memory optimization functions."""

    def test_optimize_dataframe_memory(self):
        """Test DataFrame memory optimization."""
        # Create DataFrame with suboptimal types
        df = pd.DataFrame({
            'category': ['A', 'B', 'C'] * 100,  # Only 3 unique values
            'small_int': [1, 2, 3] * 100,        # Fits in int8
            'large_int': range(300),
            'float_val': [1.0, 2.0, 3.0] * 100,
        })

        # Convert to suboptimal types
        df['category'] = df['category'].astype('object')
        df['small_int'] = df['small_int'].astype('int64')
        df['float_val'] = df['float_val'].astype('float64')

        original_memory = df.memory_usage(deep=True).sum()

        df_opt, report = optimize_dataframe_memory(df, verbose=False)

        assert report['memory_saved_mb'] > 0
        assert df_opt.memory_usage(deep=True).sum() < original_memory

        # Check specific optimizations
        assert df_opt['category'].dtype.name == 'category'
        assert df_opt['small_int'].dtype.name in ['int8', 'int16', 'int32']

    def test_suggest_optimizations(self):
        """Test optimization suggestions."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'C'] * 100,
            'large_int': range(300),
            'value': np.random.randn(300),
        })

        suggestions = suggest_optimizations(df)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        # Should suggest converting category to category type
        assert any('category' in s.lower() for s in suggestions)

    def test_estimate_dataframe_memory(self):
        """Test DataFrame memory estimation."""
        columns = {
            'id': 'int64',
            'name': 'object',
            'value': 'float64',
            'category': 'category',
        }

        memory_mb = estimate_dataframe_memory(
            n_rows=10000,
            columns=columns,
            categorical_cardinality={'category': 5}
        )

        assert memory_mb > 0
        assert isinstance(memory_mb, float)


class TestMemoryChecks:
    """Tests for memory availability checks."""

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_get_system_memory_info(self):
        """Test system memory information retrieval."""
        mem_info = get_system_memory_info()

        assert 'total_gb' in mem_info
        assert 'available_gb' in mem_info
        assert 'used_gb' in mem_info
        assert 'percent_used' in mem_info
        assert mem_info['psutil_available'] == True

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_check_memory_available(self):
        """Test memory availability check."""
        # Check for a small amount of memory (should be available)
        available, message = check_memory_available(required_mb=10)

        assert isinstance(available, bool)
        assert isinstance(message, str)

    def test_force_garbage_collection(self):
        """Test forced garbage collection."""
        stats = force_garbage_collection()

        assert 'collected' in stats
        assert 'objects_before' in stats
        assert 'objects_after' in stats


# ============================================================================
# Chunked Processing Tests
# ============================================================================

class TestChunkedDataReader:
    """Tests for ChunkedDataReader class."""

    def test_chunked_reader_iteration(self, temp_csv_file):
        """Test chunked reader iterates correctly."""
        reader = ChunkedDataReader(temp_csv_file, chunksize=1000)

        total_rows = 0
        n_chunks = 0

        for chunk in reader:
            assert isinstance(chunk, pd.DataFrame)
            total_rows += len(chunk)
            n_chunks += 1

        assert total_rows == 10000
        assert n_chunks == 10

    def test_chunked_reader_with_kwargs(self, temp_csv_file):
        """Test chunked reader with additional read kwargs."""
        reader = ChunkedDataReader(
            temp_csv_file,
            chunksize=1000,
            usecols=['id', 'category', 'value']
        )

        for chunk in reader:
            assert len(chunk.columns) == 3
            break  # Just test first chunk

    def test_chunked_reader_file_not_found(self):
        """Test chunked reader with non-existent file."""
        from simplus_eda.exceptions import DataLoadingError

        with pytest.raises(DataLoadingError):
            reader = ChunkedDataReader('nonexistent.csv', chunksize=1000)


class TestStreamingAggregator:
    """Tests for StreamingAggregator class."""

    def test_streaming_aggregator_basic(self, temp_csv_file):
        """Test streaming aggregator computes correct statistics."""
        # Compute with streaming
        aggregator = StreamingAggregator(columns=['value', 'price'])
        reader = ChunkedDataReader(temp_csv_file, chunksize=1000)

        for chunk in reader:
            aggregator.update(chunk)

        stream_stats = aggregator.compute()

        # Compute with pandas
        df = pd.read_csv(temp_csv_file)
        pandas_mean = df['value'].mean()
        pandas_std = df['value'].std()

        # Compare results (allow small numerical errors)
        assert abs(stream_stats['value']['mean'] - pandas_mean) < 0.01
        assert abs(stream_stats['value']['std'] - pandas_std) < 0.01

    def test_streaming_aggregator_with_missing(self):
        """Test streaming aggregator handles missing values."""
        # Create test data with missing values
        df = pd.DataFrame({
            'value': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            aggregator = StreamingAggregator(columns=['value'])
            reader = ChunkedDataReader(temp_file, chunksize=3)

            for chunk in reader:
                aggregator.update(chunk)

            stats = aggregator.compute()

            assert stats['value']['count'] == 8  # 2 missing values
            assert stats['value']['missing'] == 2
        finally:
            os.remove(temp_file)


class TestChunkedProcessing:
    """Tests for chunked processing functions."""

    def test_streaming_statistics(self, large_csv_file):
        """Test streaming statistics computation."""
        stats = streaming_statistics(
            large_csv_file,
            chunksize=5000,
            columns=['value', 'price']
        )

        assert 'value' in stats
        assert 'price' in stats
        assert 'mean' in stats['value']
        assert 'std' in stats['value']
        assert 'min' in stats['value']
        assert 'max' in stats['value']

    def test_process_in_chunks(self, temp_csv_file):
        """Test custom chunked processing."""
        def count_rows(chunk):
            return len(chunk)

        total = process_in_chunks(
            temp_csv_file,
            func=count_rows,
            chunksize=1000,
            combine_func=sum
        )

        assert total == 10000

    def test_process_in_chunks_no_combine(self, temp_csv_file):
        """Test chunked processing without combine function."""
        def count_rows(chunk):
            return len(chunk)

        results = process_in_chunks(
            temp_csv_file,
            func=count_rows,
            chunksize=1000
        )

        assert isinstance(results, list)
        assert len(results) == 10
        assert sum(results) == 10000

    def test_sample_large_file_first(self, large_csv_file):
        """Test sampling first N rows from large file."""
        sample = sample_large_file(
            large_csv_file,
            n_samples=1000,
            method='first'
        )

        assert len(sample) == 1000
        # First method should return first rows in order
        assert sample['id'].iloc[0] == 0

    def test_sample_large_file_random(self, large_csv_file):
        """Test random sampling from large file."""
        sample = sample_large_file(
            large_csv_file,
            n_samples=1000,
            method='random',
            chunksize=5000
        )

        assert len(sample) == 1000

    def test_chunked_value_counts(self, large_csv_file):
        """Test value counts on large file."""
        counts = chunked_value_counts(
            large_csv_file,
            column='category',
            chunksize=5000,
            top_n=3
        )

        assert isinstance(counts, pd.Series)
        assert len(counts) == 3
        assert counts.index.tolist() == ['A', 'B', 'C'] or \
               counts.index.tolist() == ['B', 'A', 'C'] or \
               counts.index.tolist() == ['C', 'A', 'B']  # Order may vary


# ============================================================================
# Dask Backend Tests
# ============================================================================

@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not installed")
class TestDaskBackend:
    """Tests for Dask backend."""

    def test_dask_available(self):
        """Test Dask availability check."""
        check_dask_available()  # Should not raise

    def test_dask_backend_init(self):
        """Test Dask backend initialization."""
        backend = DaskBackend(npartitions=4, scheduler='threads')

        assert backend.npartitions == 4
        assert backend.scheduler == 'threads'

    def test_from_pandas(self, sample_dataframe):
        """Test converting pandas to Dask DataFrame."""
        backend = DaskBackend()
        ddf = backend.from_pandas(sample_dataframe)

        # Verify it's a Dask DataFrame
        import dask.dataframe as dd
        assert isinstance(ddf, dd.DataFrame)

        # Verify data integrity
        result = ddf.compute()
        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_to_pandas(self, sample_dataframe):
        """Test converting Dask to pandas DataFrame."""
        backend = DaskBackend()
        ddf = backend.from_pandas(sample_dataframe)
        result = backend.to_pandas(ddf)

        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_read_csv(self, temp_csv_file):
        """Test reading CSV with Dask."""
        backend = DaskBackend()
        ddf = backend.read_csv(temp_csv_file)

        # Should be lazy (not computed yet)
        import dask.dataframe as dd
        assert isinstance(ddf, dd.DataFrame)

        # Compute and verify
        result = ddf.compute()
        assert len(result) == 10000

    def test_compute_statistics(self, sample_dataframe):
        """Test computing statistics with Dask."""
        backend = DaskBackend()
        ddf = backend.from_pandas(sample_dataframe)

        stats = backend.compute_statistics(ddf, columns=['value', 'price'])

        assert 'value' in stats
        assert 'price' in stats
        assert 'mean' in stats['value']
        assert 'std' in stats['value']

        # Verify against pandas
        pandas_mean = sample_dataframe['value'].mean()
        assert abs(stats['value']['mean'] - pandas_mean) < 0.01

    def test_compute_correlation(self, sample_dataframe):
        """Test computing correlation with Dask."""
        backend = DaskBackend()
        ddf = backend.from_pandas(sample_dataframe)

        corr = backend.compute_correlation(
            ddf,
            columns=['value', 'price', 'quantity']
        )

        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (3, 3)

    def test_compute_value_counts(self, sample_dataframe):
        """Test value counts with Dask."""
        backend = DaskBackend()
        ddf = backend.from_pandas(sample_dataframe)

        counts = backend.compute_value_counts(ddf, column='category', top_n=3)

        assert isinstance(counts, pd.Series)
        assert len(counts) <= 3

    def test_aggregate(self, sample_dataframe):
        """Test custom aggregation with Dask."""
        backend = DaskBackend()
        ddf = backend.from_pandas(sample_dataframe)

        agg_dict = {
            'value': ['mean', 'std'],
            'price': ['min', 'max'],
        }

        result = backend.aggregate(ddf, agg_dict)

        assert isinstance(result, pd.DataFrame)
        assert ('value', 'mean') in result.columns
        assert ('price', 'max') in result.columns

    def test_get_n_rows(self, sample_dataframe):
        """Test getting number of rows."""
        backend = DaskBackend()
        ddf = backend.from_pandas(sample_dataframe)

        n_rows = backend.get_n_rows(ddf)
        assert n_rows == len(sample_dataframe)


# ============================================================================
# Integration Tests
# ============================================================================

class TestLargeDatasetIntegration:
    """Integration tests for large dataset features."""

    def test_full_pipeline(self, sample_dataframe):
        """Test complete large dataset analysis pipeline."""
        # 1. Check memory
        if PSUTIL_AVAILABLE:
            mem_info = get_system_memory_info()
            assert mem_info['total_gb'] > 0

        # 2. Optimize memory
        df_opt, report = optimize_dataframe_memory(sample_dataframe)
        assert report['percent_saved'] >= 0

        # 3. Sample if needed
        if len(df_opt) > 1000:
            sample, metadata = smart_sample(df_opt, target_size=1000)
            assert metadata['was_sampled']
            df_final = sample
        else:
            df_final = df_opt

        # 4. Verify final DataFrame
        assert len(df_final) <= len(sample_dataframe)
        assert set(df_final.columns) == set(sample_dataframe.columns)

    def test_chunked_then_sample(self, large_csv_file):
        """Test chunked reading followed by sampling."""
        # Read in chunks and sample
        sample = sample_large_file(
            large_csv_file,
            n_samples=5000,
            method='random',
            chunksize=10000
        )

        assert len(sample) == 5000

        # Then optimize the sample
        sample_opt, report = optimize_dataframe_memory(sample)
        assert report['memory_saved_mb'] >= 0

    @pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not installed")
    def test_dask_then_sample(self, sample_dataframe):
        """Test Dask processing followed by sampling to pandas."""
        backend = DaskBackend()

        # Convert to Dask
        ddf = backend.from_pandas(sample_dataframe)

        # Compute statistics with Dask
        stats = backend.compute_statistics(ddf)
        assert 'value' in stats

        # Sample for visualization
        sample_ddf = ddf.sample(frac=0.1, random_state=42)
        sample_df = backend.to_pandas(sample_ddf)

        assert len(sample_df) < len(sample_dataframe)


# ============================================================================
# Performance Tests (Optional - run separately)
# ============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance tests for large dataset features."""

    def test_sampling_performance(self):
        """Test sampling performance on large dataset."""
        import time

        # Create large DataFrame
        n_rows = 1000000
        df = pd.DataFrame({
            'id': range(n_rows),
            'value': np.random.randn(n_rows),
            'category': np.random.choice(['A', 'B', 'C'], n_rows),
        })

        # Time sampling
        start = time.time()
        sample, metadata = smart_sample(df, target_size=10000, method='random')
        elapsed = time.time() - start

        assert len(sample) == 10000
        # Should complete in reasonable time
        assert elapsed < 5.0  # 5 seconds

    @pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not installed")
    def test_dask_performance(self):
        """Test Dask backend performance."""
        import time

        # Create large DataFrame
        n_rows = 1000000
        df = pd.DataFrame({
            'id': range(n_rows),
            'value': np.random.randn(n_rows),
            'price': np.random.uniform(10, 100, n_rows),
        })

        backend = DaskBackend()
        ddf = backend.from_pandas(df, npartitions=4)

        # Time statistics computation
        start = time.time()
        stats = backend.compute_statistics(ddf)
        elapsed = time.time() - start

        assert 'value' in stats
        # Should complete in reasonable time with parallelization
        assert elapsed < 10.0  # 10 seconds


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
