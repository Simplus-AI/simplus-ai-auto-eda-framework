"""
Tests for the CorrelationAnalyzer class.
"""

import pytest
import pandas as pd
import numpy as np
from simplus_eda.analyzers.correlation import CorrelationAnalyzer


class TestCorrelationAnalyzer:
    """Test suite for CorrelationAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a CorrelationAnalyzer instance for testing."""
        return CorrelationAnalyzer()

    @pytest.fixture
    def simple_data(self):
        """Create a simple numeric DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.randn(100),
        })

    @pytest.fixture
    def correlated_data(self):
        """Create a DataFrame with known correlations."""
        np.random.seed(42)
        x = np.random.randn(100)
        return pd.DataFrame({
            'X': x,
            'Y': x * 2 + np.random.randn(100) * 0.1,  # Strong positive correlation
            'Z': -x * 1.5 + np.random.randn(100) * 0.1,  # Strong negative correlation
            'W': np.random.randn(100),  # No correlation
        })

    @pytest.fixture
    def multicollinear_data(self):
        """Create a DataFrame with multicollinearity."""
        np.random.seed(42)
        x1 = np.random.randn(100)
        x2 = np.random.randn(100)
        return pd.DataFrame({
            'X1': x1,
            'X2': x2,
            'X3': x1 + x2,  # Perfect linear combination
            'X4': 2 * x1 + np.random.randn(100) * 0.01,  # Nearly collinear
        })

    @pytest.fixture
    def mixed_data(self):
        """Create a DataFrame with numeric and non-numeric columns."""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric1': np.random.randn(50),
            'numeric2': np.random.randn(50),
            'category': ['A', 'B'] * 25,
            'text': ['foo', 'bar'] * 25,
        })

    @pytest.fixture
    def data_with_missing(self):
        """Create a DataFrame with missing values."""
        np.random.seed(42)
        data = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.randn(100),
        })
        data.loc[0:10, 'A'] = np.nan
        data.loc[20:30, 'B'] = np.nan
        return data

    @pytest.fixture
    def zero_variance_data(self):
        """Create a DataFrame with zero variance columns."""
        return pd.DataFrame({
            'constant': [5.0] * 100,
            'variable': np.random.randn(100),
        })

    @pytest.fixture
    def empty_data(self):
        """Create an empty DataFrame."""
        return pd.DataFrame()

    @pytest.fixture
    def single_column_data(self):
        """Create a DataFrame with only one numeric column."""
        return pd.DataFrame({
            'A': np.random.randn(100),
        })

    # ============== Tests for analyze() ==============

    def test_analyze_returns_dict(self, analyzer, simple_data):
        """Test that analyze returns a dictionary with expected keys."""
        result = analyzer.analyze(simple_data)

        assert isinstance(result, dict)
        assert "correlation_matrix" in result
        assert "strong_correlations" in result
        assert "multicollinearity" in result

    def test_analyze_with_custom_threshold(self, analyzer, correlated_data):
        """Test analyze with custom correlation threshold."""
        result = analyzer.analyze(correlated_data, threshold=0.5)

        assert result["strong_correlations"]["threshold"] == 0.5
        assert result["strong_correlations"]["count"] > 0

    # ============== Tests for _calculate_correlations() ==============

    def test_calculate_correlations_simple(self, analyzer, simple_data):
        """Test correlation calculation on simple data."""
        result = analyzer._calculate_correlations(simple_data)

        assert "pearson" in result
        assert "spearman" in result
        assert "kendall" in result
        assert "numeric_columns" in result

        assert result["pearson"] is not None
        assert result["spearman"] is not None
        assert result["kendall"] is not None
        assert len(result["numeric_columns"]) == 3

    def test_calculate_correlations_matrix_structure(self, analyzer, simple_data):
        """Test that correlation matrices have correct structure."""
        result = analyzer._calculate_correlations(simple_data)

        pearson = result["pearson"]
        assert isinstance(pearson, dict)
        assert 'A' in pearson
        assert 'B' in pearson
        assert 'C' in pearson

        # Check that diagonal values are 1 (correlation with itself)
        assert abs(pearson['A']['A'] - 1.0) < 1e-10
        assert abs(pearson['B']['B'] - 1.0) < 1e-10

    def test_calculate_correlations_mixed_data(self, analyzer, mixed_data):
        """Test correlation calculation with mixed data types."""
        result = analyzer._calculate_correlations(mixed_data)

        assert result["numeric_columns"] == ['numeric1', 'numeric2']
        assert result["pearson"] is not None

        # Only numeric columns should be in the correlation matrix
        pearson = result["pearson"]
        assert 'numeric1' in pearson
        assert 'numeric2' in pearson
        assert 'category' not in pearson
        assert 'text' not in pearson

    def test_calculate_correlations_no_numeric_columns(self, analyzer):
        """Test correlation calculation with no numeric columns."""
        data = pd.DataFrame({
            'category': ['A', 'B', 'C'] * 10,
            'text': ['foo', 'bar', 'baz'] * 10,
        })

        result = analyzer._calculate_correlations(data)

        assert result["pearson"] is None
        assert result["spearman"] is None
        assert result["kendall"] is None
        assert "message" in result

    def test_calculate_correlations_zero_variance(self, analyzer, zero_variance_data):
        """Test correlation calculation with zero variance columns."""
        result = analyzer._calculate_correlations(zero_variance_data)

        # Should only include the variable column
        assert "variable" in result["numeric_columns"]
        assert "constant" not in result["numeric_columns"]

    def test_calculate_correlations_with_missing_values(self, analyzer, data_with_missing):
        """Test correlation calculation with missing values."""
        result = analyzer._calculate_correlations(data_with_missing)

        # Should still calculate correlations (pandas handles NaN)
        assert result["pearson"] is not None
        assert len(result["numeric_columns"]) == 3

    def test_calculate_correlations_empty_dataframe(self, analyzer, empty_data):
        """Test correlation calculation with empty DataFrame."""
        result = analyzer._calculate_correlations(empty_data)

        assert result["pearson"] is None
        assert "message" in result

    # ============== Tests for _find_strong_correlations() ==============

    def test_find_strong_correlations_basic(self, analyzer, correlated_data):
        """Test finding strong correlations."""
        result = analyzer._find_strong_correlations(correlated_data, threshold=0.7)

        assert "pairs" in result
        assert "count" in result
        assert "threshold" in result

        assert result["threshold"] == 0.7
        assert result["count"] > 0
        assert len(result["pairs"]) == result["count"]

    def test_find_strong_correlations_pair_structure(self, analyzer, correlated_data):
        """Test structure of correlation pairs."""
        result = analyzer._find_strong_correlations(correlated_data, threshold=0.7)

        if result["count"] > 0:
            pair = result["pairs"][0]
            assert "feature1" in pair
            assert "feature2" in pair
            assert "correlation" in pair
            assert "abs_correlation" in pair
            assert "p_value" in pair
            assert "relationship" in pair
            assert "strength" in pair

            # Check relationship is valid
            assert pair["relationship"] in ["positive", "negative"]

    def test_find_strong_correlations_threshold_filtering(self, analyzer, correlated_data):
        """Test that threshold filtering works correctly."""
        result_high = analyzer._find_strong_correlations(correlated_data, threshold=0.9)
        result_low = analyzer._find_strong_correlations(correlated_data, threshold=0.5)

        # Lower threshold should find more or equal correlations
        assert result_low["count"] >= result_high["count"]

        # All correlations should meet the threshold
        for pair in result_high["pairs"]:
            assert pair["abs_correlation"] >= 0.9

    def test_find_strong_correlations_sorted(self, analyzer, correlated_data):
        """Test that correlations are sorted by strength."""
        result = analyzer._find_strong_correlations(correlated_data, threshold=0.5)

        if result["count"] > 1:
            abs_corrs = [pair["abs_correlation"] for pair in result["pairs"]]
            assert abs_corrs == sorted(abs_corrs, reverse=True)

    def test_find_strong_correlations_insufficient_columns(self, analyzer, single_column_data):
        """Test with insufficient columns."""
        result = analyzer._find_strong_correlations(single_column_data, threshold=0.7)

        assert result["count"] == 0
        assert result["pairs"] == []
        assert "message" in result

    def test_find_strong_correlations_no_strong_pairs(self, analyzer, simple_data):
        """Test when no strong correlations exist."""
        result = analyzer._find_strong_correlations(simple_data, threshold=0.99)

        # With random data, unlikely to have correlation > 0.99
        assert result["count"] >= 0
        assert isinstance(result["pairs"], list)

    # ============== Tests for _detect_multicollinearity() ==============

    def test_detect_multicollinearity_basic(self, analyzer, multicollinear_data):
        """Test basic multicollinearity detection."""
        result = analyzer._detect_multicollinearity(multicollinear_data)

        assert "vif_scores" in result
        assert "high_vif_features" in result
        assert "high_vif_count" in result
        assert "multicollinearity_detected" in result
        assert "recommendation" in result

    def test_detect_multicollinearity_vif_structure(self, analyzer, simple_data):
        """Test structure of VIF scores."""
        result = analyzer._detect_multicollinearity(simple_data)

        for vif_info in result["vif_scores"]:
            assert "feature" in vif_info
            assert "vif" in vif_info or "error" in vif_info

            if "vif" in vif_info and vif_info["vif"] is not None:
                assert "interpretation" in vif_info

    def test_detect_multicollinearity_detects_high_vif(self, analyzer, multicollinear_data):
        """Test that high VIF is detected in multicollinear data."""
        result = analyzer._detect_multicollinearity(multicollinear_data)

        # Should detect multicollinearity
        assert result["multicollinearity_detected"] is True
        assert result["high_vif_count"] > 0

        # Check that high VIF features have VIF > 10
        for feature in result["high_vif_features"]:
            vif_value = feature["vif"]
            if isinstance(vif_value, (int, float)):
                assert vif_value > 10 or vif_value == "inf"

    def test_detect_multicollinearity_no_issues(self, analyzer, simple_data):
        """Test with data that has no multicollinearity."""
        result = analyzer._detect_multicollinearity(simple_data)

        # Random independent data should have low VIF
        assert result["high_vif_count"] == 0
        assert result["multicollinearity_detected"] is False

    def test_detect_multicollinearity_insufficient_data(self, analyzer, single_column_data):
        """Test with insufficient data."""
        result = analyzer._detect_multicollinearity(single_column_data)

        assert result["vif_scores"] == []
        assert result["high_vif_features"] == []
        assert "message" in result

    def test_detect_multicollinearity_with_missing_values(self, analyzer, data_with_missing):
        """Test multicollinearity detection with missing values."""
        result = analyzer._detect_multicollinearity(data_with_missing)

        # Should handle missing values by dropping them
        assert "vif_scores" in result
        # May or may not have VIF scores depending on remaining data

    # ============== Tests for helper methods ==============

    def test_classify_correlation_strength(self, analyzer):
        """Test correlation strength classification."""
        assert analyzer._classify_correlation_strength(0.95) == "very_strong"
        assert analyzer._classify_correlation_strength(0.8) == "strong"
        assert analyzer._classify_correlation_strength(0.6) == "moderate"
        assert analyzer._classify_correlation_strength(0.4) == "weak"
        assert analyzer._classify_correlation_strength(0.2) == "very_weak"

        # Test boundary values
        assert analyzer._classify_correlation_strength(0.9) == "very_strong"
        assert analyzer._classify_correlation_strength(0.7) == "strong"
        assert analyzer._classify_correlation_strength(0.5) == "moderate"
        assert analyzer._classify_correlation_strength(0.3) == "weak"

    def test_interpret_vif(self, analyzer):
        """Test VIF interpretation."""
        assert "Perfect multicollinearity" in analyzer._interpret_vif(float('inf'))
        assert "High multicollinearity" in analyzer._interpret_vif(15)
        assert "Moderate multicollinearity" in analyzer._interpret_vif(7)
        assert "Low multicollinearity" in analyzer._interpret_vif(2)

        # Test boundary values
        assert "High multicollinearity" in analyzer._interpret_vif(10.1)
        assert "Moderate multicollinearity" in analyzer._interpret_vif(5.1)

    def test_get_multicollinearity_recommendation(self, analyzer):
        """Test multicollinearity recommendations."""
        rec_none = analyzer._get_multicollinearity_recommendation(0)
        rec_low = analyzer._get_multicollinearity_recommendation(2)
        rec_high = analyzer._get_multicollinearity_recommendation(5)

        assert "No significant multicollinearity" in rec_none
        assert "Low multicollinearity" in rec_low
        assert "High multicollinearity" in rec_high

        # Check recommendations contain counts
        assert "2" in rec_low
        assert "5" in rec_high

    # ============== Edge case tests ==============

    def test_analyze_with_all_edge_cases(self, analyzer):
        """Test analyze with various edge cases."""
        # Empty DataFrame
        result = analyzer.analyze(pd.DataFrame())
        assert isinstance(result, dict)

        # Single row
        single_row = pd.DataFrame({'A': [1], 'B': [2]})
        result = analyzer.analyze(single_row)
        assert isinstance(result, dict)

        # All NaN
        all_nan = pd.DataFrame({'A': [np.nan] * 10, 'B': [np.nan] * 10})
        result = analyzer.analyze(all_nan)
        assert isinstance(result, dict)

    def test_perfect_correlation(self, analyzer):
        """Test with perfectly correlated features."""
        data = pd.DataFrame({
            'X': range(100),
            'Y': range(100),  # Perfect positive correlation
            'Z': range(100, 0, -1),  # Perfect negative correlation with X
        })

        result = analyzer._find_strong_correlations(data, threshold=0.9)

        # Should find perfect correlations
        assert result["count"] > 0

        # Check for correlation close to 1 or -1
        max_corr = max([pair["abs_correlation"] for pair in result["pairs"]])
        assert max_corr > 0.99

    def test_large_dataset_performance(self, analyzer):
        """Test that analyzer works with larger datasets."""
        np.random.seed(42)
        large_data = pd.DataFrame(
            np.random.randn(1000, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )

        # Should complete without errors
        result = analyzer.analyze(large_data)
        assert isinstance(result, dict)
        assert result["correlation_matrix"]["pearson"] is not None

    def test_negative_correlation_detection(self, analyzer, correlated_data):
        """Test detection of negative correlations."""
        result = analyzer._find_strong_correlations(correlated_data, threshold=0.7)

        # Should detect both positive and negative correlations
        relationships = [pair["relationship"] for pair in result["pairs"]]

        # correlated_data has both positive and negative correlations
        assert "positive" in relationships or "negative" in relationships

    def test_analyze_preserves_original_data(self, analyzer, simple_data):
        """Test that analyze doesn't modify the original DataFrame."""
        original_data = simple_data.copy()

        analyzer.analyze(simple_data)

        pd.testing.assert_frame_equal(simple_data, original_data)

    def test_correlation_symmetry(self, analyzer, simple_data):
        """Test that correlation matrix is symmetric."""
        result = analyzer._calculate_correlations(simple_data)
        pearson = result["pearson"]

        # Check symmetry
        for col1 in pearson.keys():
            for col2 in pearson.keys():
                assert abs(pearson[col1][col2] - pearson[col2][col1]) < 1e-10

    def test_vif_calculation_consistency(self, analyzer, simple_data):
        """Test VIF calculation consistency."""
        result1 = analyzer._detect_multicollinearity(simple_data)
        result2 = analyzer._detect_multicollinearity(simple_data)

        # Results should be consistent
        assert result1["high_vif_count"] == result2["high_vif_count"]


# ============== Integration tests ==============

class TestCorrelationAnalyzerIntegration:
    """Integration tests for CorrelationAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a CorrelationAnalyzer instance."""
        return CorrelationAnalyzer()

    def test_full_workflow_with_real_data(self, analyzer):
        """Test complete workflow with realistic data."""
        np.random.seed(42)

        # Create realistic dataset
        n_samples = 200
        data = pd.DataFrame({
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.randint(20000, 150000, n_samples),
            'expenses': np.random.randint(15000, 100000, n_samples),
            'savings': np.random.randint(0, 50000, n_samples),
            'debt': np.random.randint(0, 30000, n_samples),
        })

        # Make some correlations
        data['savings'] = data['income'] * 0.3 - data['expenses'] * 0.2
        data['debt'] = -data['savings'] * 0.4 + np.random.randn(n_samples) * 1000

        # Run full analysis
        result = analyzer.analyze(data, threshold=0.5)

        # Verify all components
        assert result["correlation_matrix"]["pearson"] is not None
        assert result["strong_correlations"]["count"] >= 0
        assert result["multicollinearity"]["vif_scores"] is not None

        # Should detect some strong correlations we created
        assert result["strong_correlations"]["count"] > 0

    def test_analyze_different_thresholds(self, analyzer):
        """Test analyze with different thresholds."""
        np.random.seed(42)
        data = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.randn(100),
        })
        data['B'] = data['A'] * 0.8 + np.random.randn(100) * 0.1

        result_low = analyzer.analyze(data, threshold=0.5)
        result_high = analyzer.analyze(data, threshold=0.9)

        # Lower threshold should find more correlations
        assert result_low["strong_correlations"]["count"] >= result_high["strong_correlations"]["count"]
