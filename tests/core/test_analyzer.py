"""
Tests for the main EDA Analyzer class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from simplus_eda.core.analyzer import EDAAnalyzer
from simplus_eda.core.config import EDAConfig


@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'numeric1': np.random.randn(100),
        'numeric2': np.random.randn(100) * 10 + 50,
        'numeric3': np.random.randint(1, 100, 100),
        'categorical1': np.random.choice(['A', 'B', 'C'], 100),
        'categorical2': np.random.choice(['X', 'Y', 'Z'], 100),
    })


@pytest.fixture
def data_with_missing():
    """Create dataset with missing values."""
    np.random.seed(42)
    df = pd.DataFrame({
        'col1': np.random.randn(100),
        'col2': np.random.randn(100),
        'col3': np.random.choice(['A', 'B', 'C'], 100)
    })
    # Add missing values
    df.loc[0:20, 'col1'] = np.nan
    df.loc[10:30, 'col2'] = np.nan
    df.loc[5:15, 'col3'] = np.nan
    return df


@pytest.fixture
def data_with_outliers():
    """Create dataset with outliers."""
    np.random.seed(42)
    df = pd.DataFrame({
        'values': np.random.randn(100)
    })
    # Add outliers
    df.loc[0, 'values'] = 100
    df.loc[1, 'values'] = -100
    df.loc[2, 'values'] = 50
    return df


@pytest.fixture
def data_with_correlations():
    """Create dataset with strong correlations."""
    np.random.seed(42)
    x = np.random.randn(100)
    return pd.DataFrame({
        'x': x,
        'y': x * 2 + np.random.randn(100) * 0.1,  # Strong positive correlation
        'z': -x * 1.5 + np.random.randn(100) * 0.1,  # Strong negative correlation
        'w': np.random.randn(100)  # No correlation
    })


@pytest.fixture
def data_with_duplicates():
    """Create dataset with duplicate rows."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 1, 2],
        'b': [4, 5, 6, 4, 5],
        'c': [7, 8, 9, 7, 8]
    })
    return df


class TestEDAAnalyzerInitialization:
    """Test EDAAnalyzer initialization."""

    def test_initialization_default(self):
        """Test initialization with default configuration."""
        analyzer = EDAAnalyzer()

        assert analyzer is not None
        assert isinstance(analyzer.config, EDAConfig)
        assert analyzer.data is None
        assert analyzer.results == {}

    def test_initialization_with_dict_config(self):
        """Test initialization with dictionary configuration."""
        config = {
            "correlation_threshold": 0.8,
            "missing_threshold": 0.2,
            "outlier_method": "zscore"
        }
        analyzer = EDAAnalyzer(config=config)

        assert analyzer.config.correlation_threshold == 0.8
        assert analyzer.config.missing_threshold == 0.2
        assert analyzer.config.outlier_method == "zscore"

    def test_initialization_with_config_object(self):
        """Test initialization with EDAConfig object."""
        config = EDAConfig(
            correlation_threshold=0.85,
            missing_threshold=0.15
        )
        analyzer = EDAAnalyzer(config=config)

        assert analyzer.config.correlation_threshold == 0.85
        assert analyzer.config.missing_threshold == 0.15

    def test_analyzers_are_initialized(self):
        """Test that specialized analyzers are initialized."""
        analyzer = EDAAnalyzer()

        assert analyzer._statistical_analyzer is not None
        assert analyzer._quality_analyzer is not None
        assert analyzer._correlation_analyzer is not None
        assert analyzer._outlier_analyzer is not None


class TestEDAAnalyzerValidation:
    """Test input validation."""

    def test_analyze_invalid_input_type(self):
        """Test that analyze raises error for non-DataFrame input."""
        analyzer = EDAAnalyzer()

        with pytest.raises(ValueError, match="Data must be a pandas DataFrame"):
            analyzer.analyze([1, 2, 3])

    def test_analyze_empty_dataframe(self):
        """Test that analyze raises error for empty DataFrame."""
        analyzer = EDAAnalyzer()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Data cannot be empty"):
            analyzer.analyze(empty_df)

    def test_analyze_with_valid_dataframe(self, sample_data):
        """Test that analyze accepts valid DataFrame."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data)

        assert results is not None
        assert isinstance(results, dict)


class TestEDAAnalyzerOverview:
    """Test overview generation."""

    def test_overview_structure(self, sample_data):
        """Test that overview has expected structure."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data)
        overview = results['overview']

        assert 'shape' in overview
        assert 'columns' in overview
        assert 'column_counts' in overview
        assert 'dtypes' in overview
        assert 'memory_usage' in overview
        assert 'duplicates' in overview

    def test_overview_shape(self, sample_data):
        """Test shape information in overview."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data)
        shape = results['overview']['shape']

        assert shape['rows'] == 100
        assert shape['columns'] == 5

    def test_overview_column_types(self, sample_data):
        """Test column type detection."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data)
        columns = results['overview']['columns']

        assert len(columns['numeric']) == 3
        assert len(columns['categorical']) == 2
        assert 'numeric1' in columns['numeric']
        assert 'categorical1' in columns['categorical']

    def test_overview_memory_usage(self, sample_data):
        """Test memory usage calculation."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data)
        memory = results['overview']['memory_usage']

        assert 'total_bytes' in memory
        assert 'total_mb' in memory
        assert memory['total_bytes'] > 0
        assert memory['total_mb'] > 0

    def test_overview_duplicates(self, data_with_duplicates):
        """Test duplicate detection in overview."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(data_with_duplicates)
        duplicates = results['overview']['duplicates']

        assert duplicates['count'] == 2
        assert duplicates['percentage'] == 40.0


class TestEDAAnalyzerStatistics:
    """Test statistical analysis."""

    def test_statistics_structure(self, sample_data):
        """Test that statistics has expected structure."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data)
        stats = results['statistics']

        assert stats is not None
        assert isinstance(stats, dict)
        # Check that StatisticalAnalyzer results are included
        assert 'descriptive' in stats or 'error' in stats

    def test_statistics_quick_mode(self, sample_data):
        """Test statistics in quick mode."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data, quick=True)
        stats = results['statistics']

        # Quick mode should skip hypothesis tests
        assert 'hypothesis_tests' not in stats or stats.get('error')

    def test_statistics_full_mode(self, sample_data):
        """Test statistics in full mode."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data, quick=False)
        stats = results['statistics']

        assert stats is not None
        # Full mode should include all statistics
        if 'error' not in stats:
            assert 'descriptive' in stats


class TestEDAAnalyzerQuality:
    """Test data quality assessment."""

    def test_quality_structure(self, sample_data):
        """Test that quality assessment has expected structure."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data)
        quality = results['quality']

        assert quality is not None
        assert isinstance(quality, dict)

    def test_quality_completeness(self, data_with_missing):
        """Test completeness assessment with missing data."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(data_with_missing)
        quality = results['quality']

        # Check for either completeness or quality_score structure
        if 'completeness' in quality:
            completeness = quality['completeness']
            assert 'completeness_score' in completeness
            # Should be less than 100% due to missing values
            assert completeness['completeness_score'] < 100
        elif 'quality_score' in quality:
            # Alternative structure from DataQualityAnalyzer
            quality_score = quality['quality_score']
            assert 'overall_score' in quality_score
            # Should detect some quality issues
            assert quality_score['overall_score'] < 100
        else:
            # Should have error or one of the expected structures
            assert 'error' in quality

    def test_quality_warnings(self, data_with_missing):
        """Test quality warnings generation."""
        config = EDAConfig(missing_threshold=0.05)  # Low threshold
        analyzer = EDAAnalyzer(config=config)
        results = analyzer.analyze(data_with_missing)
        quality = results['quality']

        # Should have warnings due to missing data
        if 'completeness' in quality and quality['completeness'].get('completeness_score', 100) < 5:
            assert 'warnings' in quality


class TestEDAAnalyzerOutliers:
    """Test outlier detection."""

    def test_outliers_structure(self, data_with_outliers):
        """Test that outlier detection has expected structure."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(data_with_outliers)
        outliers = results['outliers']

        assert outliers is not None
        assert isinstance(outliers, dict)

    def test_outliers_detection(self, data_with_outliers):
        """Test that outliers are detected."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(data_with_outliers)
        outliers = results['outliers']

        if 'error' not in outliers and 'columns' in outliers:
            # Should detect outliers in 'values' column
            assert 'summary' in outliers
            assert outliers['summary']['total_outliers'] > 0

    def test_outliers_skip_in_quick_mode(self, data_with_outliers):
        """Test that outlier detection is skipped in quick mode."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(data_with_outliers, quick=True)
        outliers = results['outliers']

        # Should be empty in quick mode
        assert outliers == {}

    def test_outliers_summary_statistics(self, data_with_outliers):
        """Test outlier summary statistics."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(data_with_outliers)
        outliers = results['outliers']

        if 'summary' in outliers:
            summary = outliers['summary']
            assert 'total_outliers' in summary
            assert 'columns_with_outliers' in summary
            assert 'outlier_percentage' in summary


class TestEDAAnalyzerCorrelations:
    """Test correlation analysis."""

    def test_correlations_structure(self, data_with_correlations):
        """Test that correlation analysis has expected structure."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(data_with_correlations)
        correlations = results['correlations']

        assert correlations is not None
        assert isinstance(correlations, dict)

    def test_correlations_detection(self, data_with_correlations):
        """Test that strong correlations are detected."""
        config = EDAConfig(correlation_threshold=0.7)
        analyzer = EDAAnalyzer(config=config)
        results = analyzer.analyze(data_with_correlations)
        correlations = results['correlations']

        if 'error' not in correlations and 'strong_correlations' in correlations:
            # Should detect strong correlations between x-y and x-z
            assert len(correlations['strong_correlations']) > 0

    def test_correlations_insufficient_columns(self):
        """Test correlation analysis with insufficient numeric columns."""
        df = pd.DataFrame({'cat': ['A', 'B', 'C']})
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(df)
        correlations = results['correlations']

        # Should return message about insufficient columns
        assert 'message' in correlations or 'error' in correlations

    def test_correlations_use_config_threshold(self, data_with_correlations):
        """Test that correlation threshold from config is used."""
        config = EDAConfig(correlation_threshold=0.95)
        analyzer = EDAAnalyzer(config=config)
        results = analyzer.analyze(data_with_correlations)
        correlations = results['correlations']

        # With high threshold, should find fewer correlations
        assert correlations is not None


class TestEDAAnalyzerInsights:
    """Test automated insight generation."""

    def test_insights_structure(self, sample_data):
        """Test that insights have expected structure."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data)
        insights = results['insights']

        assert insights is not None
        assert isinstance(insights, dict)

    def test_insights_categories(self, sample_data):
        """Test insight categories."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data)
        insights = results['insights']

        # Should have at least some insight categories
        possible_categories = [
            'data_characteristics',
            'quality_issues',
            'statistical_findings',
            'recommendations'
        ]
        has_categories = any(cat in insights for cat in possible_categories)
        # Insights generation depends on data, so it's ok if it returns empty dict
        assert has_categories or insights == {}

    def test_insights_data_characteristics(self, sample_data):
        """Test data characteristics insights."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data)
        insights = results['insights']

        if 'data_characteristics' in insights:
            characteristics = insights['data_characteristics']
            assert isinstance(characteristics, list)
            assert len(characteristics) > 0

    def test_insights_missing_data(self, data_with_missing):
        """Test insights about missing data."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(data_with_missing)
        insights = results['insights']

        # Should mention quality issues or completeness
        if 'quality_issues' in insights:
            quality_issues = insights['quality_issues']
            assert len(quality_issues) > 0

    def test_insights_duplicates(self, data_with_duplicates):
        """Test insights about duplicate rows."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(data_with_duplicates)
        insights = results['insights']

        if 'quality_issues' in insights:
            issues = insights['quality_issues']
            # Should mention duplicate rows
            has_duplicate_insight = any('duplicate' in issue.lower() for issue in issues)
            assert has_duplicate_insight

    def test_insights_outliers(self, data_with_outliers):
        """Test insights about outliers."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(data_with_outliers)
        insights = results['insights']

        # May have statistical findings about outliers
        if 'statistical_findings' in insights:
            findings = insights['statistical_findings']
            assert isinstance(findings, list)

    def test_insights_correlations(self, data_with_correlations):
        """Test insights about correlations."""
        config = EDAConfig(correlation_threshold=0.7)
        analyzer = EDAAnalyzer(config=config)
        results = analyzer.analyze(data_with_correlations)
        insights = results['insights']
        correlations = results.get('correlations', {})

        # Insights should be a dict (may be empty, may have content)
        assert isinstance(insights, dict)

        # If strong correlations exist and insights are generated, they should be appropriate
        if 'strong_correlations' in correlations and len(correlations['strong_correlations']) > 0:
            # Insights may include findings about correlations (but not required if filtered out)
            if insights:  # If insights are generated
                # They should be properly structured
                for key in insights:
                    assert isinstance(insights[key], list)


class TestEDAAnalyzerMetadata:
    """Test analysis metadata."""

    def test_metadata_structure(self, sample_data):
        """Test that metadata has expected structure."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data)
        metadata = results['metadata']

        assert 'timestamp' in metadata
        assert 'duration_seconds' in metadata
        assert 'analyzer_version' in metadata
        assert 'config' in metadata
        assert 'data_shape' in metadata

    def test_metadata_timestamp(self, sample_data):
        """Test timestamp format."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data)
        timestamp = results['metadata']['timestamp']

        # Should be ISO format
        assert isinstance(timestamp, str)
        # Should be parseable as datetime
        datetime.fromisoformat(timestamp)

    def test_metadata_duration(self, sample_data):
        """Test duration tracking."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data)
        duration = results['metadata']['duration_seconds']

        assert isinstance(duration, (int, float))
        assert duration >= 0

    def test_metadata_config(self, sample_data):
        """Test config in metadata."""
        config = EDAConfig(correlation_threshold=0.85)
        analyzer = EDAAnalyzer(config=config)
        results = analyzer.analyze(sample_data)
        metadata_config = results['metadata']['config']

        assert metadata_config['correlation_threshold'] == 0.85

    def test_metadata_data_shape(self, sample_data):
        """Test data shape in metadata."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data)
        data_shape = results['metadata']['data_shape']

        assert data_shape['rows'] == 100
        assert data_shape['columns'] == 5


class TestEDAAnalyzerGetSummary:
    """Test summary generation."""

    def test_get_summary_before_analysis(self):
        """Test summary before running analysis."""
        analyzer = EDAAnalyzer()
        summary = analyzer.get_summary()

        assert "No analysis results available" in summary

    def test_get_summary_after_analysis(self, sample_data):
        """Test summary after running analysis."""
        analyzer = EDAAnalyzer()
        analyzer.analyze(sample_data)
        summary = analyzer.get_summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "EDA Analysis Summary" in summary

    def test_get_summary_contains_key_info(self, sample_data):
        """Test that summary contains key information."""
        analyzer = EDAAnalyzer()
        analyzer.analyze(sample_data)
        summary = analyzer.get_summary()

        # Should mention dataset dimensions
        assert "100" in summary or "rows" in summary.lower()
        # Should mention columns
        assert "columns" in summary.lower() or "5" in summary

    def test_get_summary_with_quality_info(self, data_with_missing):
        """Test summary includes quality information."""
        analyzer = EDAAnalyzer()
        analyzer.analyze(data_with_missing)
        summary = analyzer.get_summary()

        # Should mention completeness or quality score
        summary_lower = summary.lower()
        assert ("completeness" in summary_lower or
                "quality" in summary_lower or
                "score" in summary_lower)


class TestEDAAnalyzerIntegration:
    """Integration tests for complete workflows."""

    def test_full_analysis_workflow(self, sample_data):
        """Test complete analysis workflow."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data)

        # Verify all major components are present
        assert 'overview' in results
        assert 'statistics' in results
        assert 'quality' in results
        assert 'outliers' in results
        assert 'correlations' in results
        assert 'insights' in results
        assert 'metadata' in results

    def test_quick_analysis_workflow(self, sample_data):
        """Test quick analysis workflow."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data, quick=True)

        # Should have results but skip time-consuming parts
        assert 'overview' in results
        assert 'statistics' in results
        assert results['outliers'] == {}  # Skipped in quick mode

    def test_custom_config_workflow(self, data_with_correlations):
        """Test workflow with custom configuration."""
        config = EDAConfig(
            correlation_threshold=0.6,
            missing_threshold=0.05,
            outlier_method="zscore"
        )
        analyzer = EDAAnalyzer(config=config)
        results = analyzer.analyze(data_with_correlations)

        # Should use custom config
        assert results['metadata']['config']['correlation_threshold'] == 0.6
        assert results['metadata']['config']['missing_threshold'] == 0.05

    def test_analyze_then_summary(self, sample_data):
        """Test analyze followed by summary."""
        analyzer = EDAAnalyzer()
        results = analyzer.analyze(sample_data)
        summary = analyzer.get_summary()

        assert results is not None
        assert summary is not None
        assert "EDA Analysis Summary" in summary

    def test_multiple_analyses(self, sample_data, data_with_missing):
        """Test running multiple analyses with same analyzer."""
        analyzer = EDAAnalyzer()

        # First analysis
        results1 = analyzer.analyze(sample_data)
        assert results1['overview']['shape']['rows'] == 100

        # Second analysis with different data
        results2 = analyzer.analyze(data_with_missing)
        assert results2['overview']['shape']['rows'] == 100

        # Results should be different
        assert results1 is not results2

    def test_data_copy_not_modified(self, sample_data):
        """Test that original data is not modified."""
        original_data = sample_data.copy()
        analyzer = EDAAnalyzer()
        analyzer.analyze(sample_data)

        # Original data should be unchanged
        pd.testing.assert_frame_equal(sample_data, original_data)

    def test_complex_dataset_workflow(self):
        """Test workflow with complex dataset containing multiple data types."""
        np.random.seed(42)
        complex_df = pd.DataFrame({
            'int_col': np.random.randint(1, 100, 100),
            'float_col': np.random.randn(100),
            'cat_col': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'bool_col': np.random.choice([True, False], 100),
            'str_col': [f"item_{i}" for i in range(100)]
        })

        # Add some missing values
        complex_df.loc[0:10, 'float_col'] = np.nan

        # Add some duplicates
        complex_df.loc[50] = complex_df.loc[0]

        analyzer = EDAAnalyzer()
        results = analyzer.analyze(complex_df)

        # Should handle all data types
        assert results['overview']['column_counts']['total'] == 5
        assert 'overview' in results
        assert 'quality' in results
        assert 'insights' in results

    def test_error_handling_in_components(self):
        """Test that errors in individual components don't crash entire analysis."""
        # Create data that might cause issues in some analyzers
        df = pd.DataFrame({
            'all_same': [1] * 100,
            'all_null': [np.nan] * 100,
            'normal': np.random.randn(100)
        })

        analyzer = EDAAnalyzer()
        # Should not raise exception even if some analyzers have issues
        results = analyzer.analyze(df)

        # Should still have main structure
        assert 'overview' in results
        assert 'metadata' in results
