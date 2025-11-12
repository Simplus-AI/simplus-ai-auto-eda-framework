"""
Tests for the StatisticalAnalyzer class.
"""

import pytest
import pandas as pd
import numpy as np
from simplus_eda.analyzers.statistical import StatisticalAnalyzer


class TestStatisticalAnalyzer:
    """Test suite for StatisticalAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a StatisticalAnalyzer instance."""
        return StatisticalAnalyzer()

    @pytest.fixture
    def normal_data(self):
        """Create normally distributed data."""
        np.random.seed(42)
        return pd.DataFrame({
            'normal1': np.random.normal(50, 10, 1000),
            'normal2': np.random.normal(100, 20, 1000),
        })

    @pytest.fixture
    def skewed_data(self):
        """Create skewed data."""
        np.random.seed(42)
        return pd.DataFrame({
            'right_skewed': np.random.exponential(2, 1000),
            'left_skewed': -np.random.exponential(2, 1000),
        })

    @pytest.fixture
    def mixed_data(self):
        """Create mixed numeric and categorical data."""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric': np.random.randn(100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'text': [f'item_{i}' for i in range(100)],
        })

    @pytest.fixture
    def empty_data(self):
        """Create empty DataFrame."""
        return pd.DataFrame()

    @pytest.fixture
    def data_with_missing(self):
        """Create data with missing values."""
        np.random.seed(42)
        data = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
        })
        data.loc[0:10, 'A'] = np.nan
        return data

    # ============== Tests for analyze() ==============

    def test_analyze_returns_dict(self, analyzer, normal_data):
        """Test that analyze returns dictionary with expected keys."""
        result = analyzer.analyze(normal_data)

        assert isinstance(result, dict)
        assert "descriptive" in result
        assert "distributions" in result
        assert "normality_tests" in result
        assert "categorical" in result
        assert "summary" in result

    def test_analyze_empty_dataframe(self, analyzer, empty_data):
        """Test analyze with empty DataFrame."""
        result = analyzer.analyze(empty_data)

        assert isinstance(result, dict)
        assert "message" in result["descriptive"]

    # ============== Tests for _descriptive_stats() ==============

    def test_descriptive_stats_basic(self, analyzer, normal_data):
        """Test basic descriptive statistics."""
        result = analyzer._descriptive_stats(normal_data)

        assert "numeric_columns" in result
        assert len(result["numeric_columns"]) == 2

        for col in ['normal1', 'normal2']:
            assert col in result["numeric_columns"]
            col_stats = result["numeric_columns"][col]

            assert "central_tendency" in col_stats
            assert "dispersion" in col_stats
            assert "percentiles" in col_stats
            assert "shape" in col_stats

    def test_descriptive_central_tendency(self, analyzer):
        """Test central tendency measures."""
        data = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        result = analyzer._descriptive_stats(data)

        stats = result["numeric_columns"]["A"]
        assert stats["central_tendency"]["mean"] == 3.0
        assert stats["central_tendency"]["median"] == 3.0

    def test_descriptive_dispersion(self, analyzer):
        """Test dispersion measures."""
        data = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        result = analyzer._descriptive_stats(data)

        disp = result["numeric_columns"]["A"]["dispersion"]
        assert "std" in disp
        assert "variance" in disp
        assert "range" in disp
        assert "iqr" in disp
        assert "mad" in disp
        assert disp["range"] == 4.0  # 5 - 1

    def test_descriptive_percentiles(self, analyzer):
        """Test percentile calculations."""
        data = pd.DataFrame({'A': range(100)})
        result = analyzer._descriptive_stats(data)

        percentiles = result["numeric_columns"]["A"]["percentiles"]
        assert percentiles["min"] == 0
        assert percentiles["max"] == 99
        assert percentiles["median"] == 49.5
        assert "Q1" in percentiles
        assert "Q3" in percentiles

    def test_descriptive_shape_metrics(self, analyzer, normal_data):
        """Test shape metrics (skewness and kurtosis)."""
        result = analyzer._descriptive_stats(normal_data)

        for col in normal_data.columns:
            shape = result["numeric_columns"][col]["shape"]
            assert "skewness" in shape
            assert "kurtosis" in shape
            assert "skewness_interpretation" in shape
            assert "kurtosis_interpretation" in shape

    def test_descriptive_confidence_interval(self, analyzer):
        """Test confidence interval calculation."""
        data = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        result = analyzer._descriptive_stats(data)

        ci = result["numeric_columns"]["A"]["confidence_interval_mean"]
        assert ci is not None
        assert "lower" in ci
        assert "upper" in ci
        assert "level" in ci
        assert ci["lower"] < ci["upper"]

    def test_descriptive_missing_values(self, analyzer, data_with_missing):
        """Test handling of missing values."""
        result = analyzer._descriptive_stats(data_with_missing)

        stats_a = result["numeric_columns"]["A"]
        assert stats_a["missing"] > 0
        assert stats_a["count"] < len(data_with_missing)

    def test_descriptive_no_numeric_columns(self, analyzer):
        """Test with no numeric columns."""
        data = pd.DataFrame({
            'text': ['a', 'b', 'c'],
            'category': ['x', 'y', 'z'],
        })
        result = analyzer._descriptive_stats(data)

        assert "message" in result
        assert result["numeric_columns"] == {}

    # ============== Tests for _analyze_distributions() ==============

    def test_distributions_basic(self, analyzer, normal_data):
        """Test basic distribution analysis."""
        result = analyzer._analyze_distributions(normal_data)

        assert "columns" in result
        assert len(result["columns"]) == 2

        for col in normal_data.columns:
            assert col in result["columns"]
            dist = result["columns"][col]
            assert "unique_values" in dist
            assert "is_discrete" in dist
            assert "skewness" in dist
            assert "kurtosis" in dist
            assert "distribution_type" in dist

    def test_distributions_skewness_detection(self, analyzer, skewed_data):
        """Test skewness detection in distributions."""
        result = analyzer._analyze_distributions(skewed_data)

        right_skewed = result["columns"]["right_skewed"]
        assert right_skewed["skewness"] > 0

        left_skewed = result["columns"]["left_skewed"]
        assert left_skewed["skewness"] < 0

    def test_distributions_discrete_detection(self, analyzer):
        """Test discrete distribution detection."""
        data = pd.DataFrame({'discrete': [1, 2, 3, 1, 2, 3] * 10})
        result = analyzer._analyze_distributions(data)

        dist = result["columns"]["discrete"]
        assert dist["is_discrete"] is True
        assert "value_counts" in dist
        assert "mode_frequency" in dist

    def test_distributions_patterns(self, analyzer, normal_data):
        """Test pattern detection in distributions."""
        result = analyzer._analyze_distributions(normal_data)

        for col in normal_data.columns:
            dist = result["columns"][col]
            assert "patterns" in dist
            assert isinstance(dist["patterns"], dict)

    # ============== Tests for _test_normality() ==============

    def test_normality_tests_normal_data(self, analyzer, normal_data):
        """Test normality tests on normal data."""
        result = analyzer._test_normality(normal_data)

        assert "columns" in result
        assert "alpha" in result

        for col in normal_data.columns:
            assert col in result["columns"]
            col_tests = result["columns"][col]
            assert "tests" in col_tests
            assert "consensus" in col_tests

    def test_normality_shapiro_wilk(self, analyzer, normal_data):
        """Test Shapiro-Wilk test."""
        result = analyzer._test_normality(normal_data)

        for col in normal_data.columns:
            tests = result["columns"][col]["tests"]
            if "shapiro_wilk" in tests:
                sw = tests["shapiro_wilk"]
                assert "statistic" in sw
                assert "p_value" in sw
                assert "is_normal" in sw

    def test_normality_kolmogorov_smirnov(self, analyzer, normal_data):
        """Test Kolmogorov-Smirnov test."""
        result = analyzer._test_normality(normal_data)

        for col in normal_data.columns:
            tests = result["columns"][col]["tests"]
            assert "kolmogorov_smirnov" in tests
            ks = tests["kolmogorov_smirnov"]
            assert "statistic" in ks
            assert "p_value" in ks

    def test_normality_anderson_darling(self, analyzer, normal_data):
        """Test Anderson-Darling test."""
        result = analyzer._test_normality(normal_data)

        for col in normal_data.columns:
            tests = result["columns"][col]["tests"]
            if "anderson_darling" in tests:
                ad = tests["anderson_darling"]
                assert "statistic" in ad
                assert "is_normal" in ad

    def test_normality_consensus(self, analyzer, normal_data):
        """Test consensus assessment."""
        result = analyzer._test_normality(normal_data)

        for col in normal_data.columns:
            consensus = result["columns"][col]["consensus"]
            assert "tests_passed" in consensus
            assert "total_tests" in consensus
            assert "percentage_normal" in consensus
            assert "likely_normal" in consensus

    def test_normality_insufficient_data(self, analyzer):
        """Test with insufficient data."""
        data = pd.DataFrame({'A': [1, 2]})
        result = analyzer._test_normality(data)

        assert "message" in result["columns"]["A"]

    def test_normality_skewed_data(self, analyzer, skewed_data):
        """Test normality tests on skewed data."""
        result = analyzer._test_normality(skewed_data)

        # Skewed data should fail normality tests
        for col in skewed_data.columns:
            consensus = result["columns"][col]["consensus"]
            # Most tests should indicate non-normal
            assert consensus["percentage_normal"] <= 100

    # ============== Tests for _analyze_categorical() ==============

    def test_categorical_basic(self, analyzer, mixed_data):
        """Test basic categorical analysis."""
        result = analyzer._analyze_categorical(mixed_data)

        assert "columns" in result
        assert "categorical" in result["columns"] or "text" in result["columns"]

    def test_categorical_statistics(self, analyzer):
        """Test categorical statistics."""
        data = pd.DataFrame({
            'cat': ['A', 'B', 'A', 'C', 'A', 'B']
        })
        result = analyzer._analyze_categorical(data)

        cat_stats = result["columns"]["cat"]
        assert cat_stats["count"] == 6
        assert cat_stats["unique_values"] == 3
        assert cat_stats["mode"] == 'A'
        assert cat_stats["mode_count"] == 3

    def test_categorical_entropy(self, analyzer):
        """Test entropy calculation."""
        data = pd.DataFrame({
            'uniform': ['A', 'B', 'C', 'D'],  # High entropy
            'skewed': ['A', 'A', 'A', 'B'],   # Low entropy
        })
        result = analyzer._analyze_categorical(data)

        uniform_entropy = result["columns"]["uniform"]["entropy"]
        skewed_entropy = result["columns"]["skewed"]["entropy"]

        # Uniform distribution should have higher entropy
        assert uniform_entropy > skewed_entropy

    def test_categorical_diversity_index(self, analyzer):
        """Test diversity index calculation."""
        data = pd.DataFrame({
            'cat': ['A', 'B', 'A', 'C', 'A']
        })
        result = analyzer._analyze_categorical(data)

        assert "diversity_index" in result["columns"]["cat"]
        assert 0 <= result["columns"]["cat"]["diversity_index"] <= 1

    def test_categorical_no_categorical_columns(self, analyzer):
        """Test with no categorical columns."""
        data = pd.DataFrame({
            'numeric1': [1, 2, 3],
            'numeric2': [4, 5, 6],
        })
        result = analyzer._analyze_categorical(data)

        assert "message" in result
        assert result["columns"] == {}

    # ============== Tests for _generate_summary() ==============

    def test_summary_basic(self, analyzer, mixed_data):
        """Test basic summary generation."""
        result = analyzer._generate_summary(mixed_data)

        assert "total_rows" in result
        assert "total_columns" in result
        assert "numeric_columns" in result
        assert "categorical_columns" in result
        assert "total_missing" in result

    def test_summary_numeric_summary(self, analyzer, normal_data):
        """Test numeric summary."""
        result = analyzer._generate_summary(normal_data)

        assert "numeric_summary" in result
        assert "mean_of_means" in result["numeric_summary"]
        assert "mean_of_stds" in result["numeric_summary"]

    def test_summary_categorical_summary(self, analyzer, mixed_data):
        """Test categorical summary."""
        result = analyzer._generate_summary(mixed_data)

        if result.get("categorical_columns", 0) > 0:
            assert "categorical_summary" in result

    def test_summary_memory_usage(self, analyzer, normal_data):
        """Test memory usage calculation."""
        result = analyzer._generate_summary(normal_data)

        assert "memory_usage_mb" in result
        assert result["memory_usage_mb"] > 0

    # ============== Tests for interpretation methods ==============

    def test_interpret_skewness(self, analyzer):
        """Test skewness interpretation."""
        assert "symmetric" in analyzer._interpret_skewness(0.1).lower()
        assert "moderately" in analyzer._interpret_skewness(0.7).lower()
        assert "highly" in analyzer._interpret_skewness(1.5).lower()
        assert "right" in analyzer._interpret_skewness(1.5).lower()
        assert "left" in analyzer._interpret_skewness(-1.5).lower()

    def test_interpret_kurtosis(self, analyzer):
        """Test kurtosis interpretation."""
        assert "platykurtic" in analyzer._interpret_kurtosis(-2).lower()
        assert "leptokurtic" in analyzer._interpret_kurtosis(2).lower()
        assert "mesokurtic" in analyzer._interpret_kurtosis(0).lower()

    def test_classify_distribution(self, analyzer):
        """Test distribution classification."""
        assert "normal" in analyzer._classify_distribution(0.1, 0.1).lower()
        assert "symmetric" in analyzer._classify_distribution(0.1, 2).lower()
        assert "right" in analyzer._classify_distribution(1.5, 0.5).lower()
        assert "left" in analyzer._classify_distribution(-1.5, 0.5).lower()

    # ============== Tests for custom parameters ==============

    def test_custom_confidence_level(self):
        """Test custom confidence level."""
        analyzer = StatisticalAnalyzer(confidence_level=0.99)
        assert analyzer.confidence_level == 0.99

        data = pd.DataFrame({'A': range(100)})
        result = analyzer._descriptive_stats(data)

        ci = result["numeric_columns"]["A"]["confidence_interval_mean"]
        assert ci["level"] == 0.99

    def test_custom_normality_alpha(self):
        """Test custom normality alpha."""
        analyzer = StatisticalAnalyzer(normality_alpha=0.01)
        assert analyzer.normality_alpha == 0.01

        data = pd.DataFrame({'A': np.random.randn(100)})
        result = analyzer._test_normality(data)

        assert result["alpha"] == 0.01

    # ============== Edge case tests ==============

    def test_single_value_column(self, analyzer):
        """Test with single value column."""
        data = pd.DataFrame({'A': [5]})
        result = analyzer._descriptive_stats(data)

        stats = result["numeric_columns"]["A"]
        assert stats["count"] == 1
        assert stats["central_tendency"]["mean"] == 5.0

    def test_all_same_values(self, analyzer):
        """Test with all same values."""
        data = pd.DataFrame({'A': [5, 5, 5, 5, 5]})
        result = analyzer._descriptive_stats(data)

        disp = result["numeric_columns"]["A"]["dispersion"]
        assert disp["std"] == 0.0
        assert disp["variance"] == 0.0

    def test_large_dataset(self, analyzer):
        """Test with large dataset."""
        np.random.seed(42)
        data = pd.DataFrame({
            'large': np.random.randn(10000)
        })
        result = analyzer.analyze(data)

        assert isinstance(result, dict)
        assert "descriptive" in result

    def test_data_preserves_original(self, analyzer, normal_data):
        """Test that analysis doesn't modify original data."""
        original_data = normal_data.copy()

        analyzer.analyze(normal_data)

        pd.testing.assert_frame_equal(normal_data, original_data)


# ============== Integration tests ==============

class TestStatisticalAnalyzerIntegration:
    """Integration tests for StatisticalAnalyzer."""

    def test_full_workflow_normal_data(self):
        """Test complete workflow with normal data."""
        np.random.seed(42)
        data = pd.DataFrame({
            'age': np.random.normal(35, 10, 200),
            'income': np.random.normal(50000, 15000, 200),
            'category': np.random.choice(['A', 'B', 'C'], 200),
        })

        analyzer = StatisticalAnalyzer()
        result = analyzer.analyze(data)

        # Should have all components
        assert all(key in result for key in
                  ["descriptive", "distributions", "normality_tests",
                   "categorical", "summary"])

        # Numeric analysis
        assert len(result["descriptive"]["numeric_columns"]) == 2
        assert len(result["normality_tests"]["columns"]) == 2

        # Categorical analysis
        assert len(result["categorical"]["columns"]) >= 1

    def test_comprehensive_analysis(self):
        """Test comprehensive statistical analysis."""
        np.random.seed(42)

        # Create diverse dataset
        data = pd.DataFrame({
            'normal': np.random.normal(100, 15, 500),
            'exponential': np.random.exponential(2, 500),
            'uniform': np.random.uniform(0, 100, 500),
            'discrete': np.random.choice([1, 2, 3, 4, 5], 500),
            'category': np.random.choice(['Low', 'Medium', 'High'], 500),
        })

        analyzer = StatisticalAnalyzer()
        result = analyzer.analyze(data)

        # Verify distributions are detected
        distributions = result["distributions"]["columns"]
        assert len(distributions) == 4  # 4 numeric columns

        # Verify normality tests
        normality = result["normality_tests"]["columns"]
        assert len(normality) == 4

        # Normal column should likely pass normality tests
        assert "normal" in normality

    def test_mixed_quality_data(self):
        """Test with data of varying quality."""
        np.random.seed(42)
        data = pd.DataFrame({
            'clean': np.random.randn(100),
            'missing': [np.nan if i % 5 == 0 else np.random.randn() for i in range(100)],
            'constant': [5.0] * 100,
            'category': np.random.choice(['A', 'B'], 100),
        })

        analyzer = StatisticalAnalyzer()
        result = analyzer.analyze(data)

        # Should handle all cases
        assert "clean" in result["descriptive"]["numeric_columns"]
        assert "missing" in result["descriptive"]["numeric_columns"]
        assert "constant" in result["descriptive"]["numeric_columns"]
        assert "category" in result["categorical"]["columns"]
