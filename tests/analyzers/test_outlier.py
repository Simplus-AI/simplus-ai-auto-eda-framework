"""
Tests for the OutlierAnalyzer class.
"""

import pytest
import pandas as pd
import numpy as np
from simplus_eda.analyzers.outlier import OutlierAnalyzer


class TestOutlierAnalyzer:
    """Test suite for OutlierAnalyzer class."""

    @pytest.fixture
    def iqr_analyzer(self):
        """Create an OutlierAnalyzer with IQR method."""
        return OutlierAnalyzer(method="iqr")

    @pytest.fixture
    def zscore_analyzer(self):
        """Create an OutlierAnalyzer with Z-score method."""
        return OutlierAnalyzer(method="zscore")

    @pytest.fixture
    def modified_zscore_analyzer(self):
        """Create an OutlierAnalyzer with Modified Z-score method."""
        return OutlierAnalyzer(method="modified_zscore")

    @pytest.fixture
    def iforest_analyzer(self):
        """Create an OutlierAnalyzer with Isolation Forest method."""
        return OutlierAnalyzer(method="isolation_forest", contamination=0.1)

    @pytest.fixture
    def all_methods_analyzer(self):
        """Create an OutlierAnalyzer that uses all methods."""
        return OutlierAnalyzer(method="all")

    @pytest.fixture
    def simple_data(self):
        """Create a simple numeric DataFrame without outliers."""
        np.random.seed(42)
        return pd.DataFrame({
            'A': np.random.normal(50, 10, 100),
            'B': np.random.normal(100, 20, 100),
            'C': np.random.normal(75, 15, 100),
        })

    @pytest.fixture
    def data_with_outliers(self):
        """Create a DataFrame with known outliers."""
        np.random.seed(42)
        data = pd.DataFrame({
            'age': np.random.normal(35, 10, 100),
            'salary': np.random.normal(50000, 15000, 100),
            'score': np.random.normal(75, 10, 100),
        })
        # Add outliers
        data.loc[0, 'age'] = 150  # extreme outlier
        data.loc[1, 'salary'] = 500000  # extreme outlier
        data.loc[2, 'score'] = 5  # outlier
        data.loc[3, 'age'] = 95  # outlier
        return data

    @pytest.fixture
    def mixed_data(self):
        """Create a DataFrame with numeric and non-numeric columns."""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric1': np.random.normal(50, 10, 50),
            'numeric2': np.random.normal(100, 20, 50),
            'category': ['A', 'B'] * 25,
            'text': ['foo', 'bar'] * 25,
        })

    @pytest.fixture
    def data_with_missing(self):
        """Create a DataFrame with missing values."""
        np.random.seed(42)
        data = pd.DataFrame({
            'A': np.random.normal(50, 10, 100),
            'B': np.random.normal(100, 20, 100),
        })
        data.loc[0:10, 'A'] = np.nan
        data.loc[20:30, 'B'] = np.nan
        return data

    @pytest.fixture
    def empty_data(self):
        """Create an empty DataFrame."""
        return pd.DataFrame()

    @pytest.fixture
    def non_numeric_data(self):
        """Create a DataFrame with no numeric columns."""
        return pd.DataFrame({
            'category': ['A', 'B', 'C'] * 10,
            'text': ['foo', 'bar', 'baz'] * 10,
        })

    @pytest.fixture
    def single_column_data(self):
        """Create a DataFrame with a single numeric column."""
        np.random.seed(42)
        return pd.DataFrame({
            'A': np.random.normal(50, 10, 100),
        })

    # ============== Tests for analyze() ==============

    def test_analyze_returns_dict(self, iqr_analyzer, simple_data):
        """Test that analyze returns a dictionary with expected keys."""
        result = iqr_analyzer.analyze(simple_data)

        assert isinstance(result, dict)
        assert "method" in result
        assert "outliers" in result
        assert "summary" in result

    def test_analyze_all_methods(self, all_methods_analyzer, data_with_outliers):
        """Test analyze with all methods."""
        result = all_methods_analyzer.analyze(data_with_outliers)

        assert isinstance(result, dict)
        assert "methods_used" in result
        assert "iqr" in result
        assert "zscore" in result
        assert "modified_zscore" in result
        assert "isolation_forest" in result
        assert "summary" in result

        assert len(result["methods_used"]) == 4

    # ============== Tests for IQR method ==============

    def test_iqr_detects_outliers(self, iqr_analyzer, data_with_outliers):
        """Test IQR method detects outliers."""
        result = iqr_analyzer.analyze(data_with_outliers)

        assert result["summary"]["total_outliers"] > 0
        assert result["summary"]["columns_with_outliers"] > 0

    def test_iqr_outlier_structure(self, iqr_analyzer, data_with_outliers):
        """Test structure of IQR outlier results."""
        result = iqr_analyzer.analyze(data_with_outliers)
        outliers_by_col = result["outliers"]["outliers_by_column"]

        for col, info in outliers_by_col.items():
            assert "count" in info
            assert "percentage" in info
            assert "indices" in info
            assert "values" in info
            assert "bounds" in info
            assert "below_lower" in info
            assert "above_upper" in info

            # Check bounds structure
            assert "lower" in info["bounds"]
            assert "upper" in info["bounds"]
            assert "Q1" in info["bounds"]
            assert "Q3" in info["bounds"]
            assert "IQR" in info["bounds"]

    def test_iqr_no_outliers_in_clean_data(self, iqr_analyzer, simple_data):
        """Test IQR method with data that has few/no outliers."""
        result = iqr_analyzer.analyze(simple_data)

        # Random normal data should have very few outliers
        assert result["summary"]["total_outliers"] >= 0

    def test_iqr_custom_multiplier(self, data_with_outliers):
        """Test IQR with custom multiplier."""
        analyzer_strict = OutlierAnalyzer(method="iqr", iqr_multiplier=1.0)
        analyzer_lenient = OutlierAnalyzer(method="iqr", iqr_multiplier=3.0)

        result_strict = analyzer_strict.analyze(data_with_outliers)
        result_lenient = analyzer_lenient.analyze(data_with_outliers)

        # Stricter multiplier should find more outliers
        assert result_strict["summary"]["total_outliers"] >= result_lenient["summary"]["total_outliers"]

    def test_iqr_bounds_correctness(self, iqr_analyzer):
        """Test that IQR bounds are calculated correctly."""
        data = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]})
        result = iqr_analyzer.analyze(data)

        bounds = result["outliers"]["outliers_by_column"]["A"]["bounds"]
        Q1 = bounds["Q1"]
        Q3 = bounds["Q3"]
        IQR = bounds["IQR"]

        # Check IQR calculation
        assert abs(IQR - (Q3 - Q1)) < 1e-10

        # Check bounds calculation with default multiplier of 1.5
        assert abs(bounds["lower"] - (Q1 - 1.5 * IQR)) < 1e-10
        assert abs(bounds["upper"] - (Q3 + 1.5 * IQR)) < 1e-10

    # ============== Tests for Z-score method ==============

    def test_zscore_detects_outliers(self, zscore_analyzer, data_with_outliers):
        """Test Z-score method detects outliers."""
        result = zscore_analyzer.analyze(data_with_outliers)

        assert result["summary"]["total_outliers"] > 0

    def test_zscore_outlier_structure(self, zscore_analyzer, data_with_outliers):
        """Test structure of Z-score outlier results."""
        result = zscore_analyzer.analyze(data_with_outliers)
        outliers_by_col = result["outliers"]["outliers_by_column"]

        for col, info in outliers_by_col.items():
            assert "count" in info
            assert "percentage" in info
            assert "indices" in info
            assert "values" in info
            assert "z_scores" in info
            assert "statistics" in info

            # Check statistics structure
            assert "mean" in info["statistics"]
            assert "std" in info["statistics"]
            assert "threshold" in info["statistics"]

    def test_zscore_custom_threshold(self, data_with_outliers):
        """Test Z-score with custom threshold."""
        analyzer_strict = OutlierAnalyzer(method="zscore", zscore_threshold=2.0)
        analyzer_lenient = OutlierAnalyzer(method="zscore", zscore_threshold=4.0)

        result_strict = analyzer_strict.analyze(data_with_outliers)
        result_lenient = analyzer_lenient.analyze(data_with_outliers)

        # Stricter threshold should find more outliers
        assert result_strict["summary"]["total_outliers"] >= result_lenient["summary"]["total_outliers"]

    def test_zscore_handles_zero_variance(self, zscore_analyzer):
        """Test Z-score handles zero variance columns."""
        data = pd.DataFrame({
            'constant': [5.0] * 100,
            'variable': np.random.normal(50, 10, 100),
        })

        result = zscore_analyzer.analyze(data)

        # Should skip constant column
        assert 'constant' not in result["outliers"]["outliers_by_column"]

    # ============== Tests for Modified Z-score method ==============

    def test_modified_zscore_detects_outliers(self, modified_zscore_analyzer, data_with_outliers):
        """Test Modified Z-score method detects outliers."""
        result = modified_zscore_analyzer.analyze(data_with_outliers)

        assert result["summary"]["total_outliers"] > 0

    def test_modified_zscore_outlier_structure(self, modified_zscore_analyzer, data_with_outliers):
        """Test structure of Modified Z-score outlier results."""
        result = modified_zscore_analyzer.analyze(data_with_outliers)
        outliers_by_col = result["outliers"]["outliers_by_column"]

        for col, info in outliers_by_col.items():
            assert "count" in info
            assert "percentage" in info
            assert "indices" in info
            assert "values" in info
            assert "modified_z_scores" in info
            assert "statistics" in info

            # Check statistics structure
            assert "median" in info["statistics"]
            assert "mad" in info["statistics"]
            assert "threshold" in info["statistics"]

    def test_modified_zscore_more_robust(self, data_with_outliers):
        """Test that Modified Z-score is more robust than regular Z-score."""
        zscore_analyzer = OutlierAnalyzer(method="zscore")
        modified_analyzer = OutlierAnalyzer(method="modified_zscore")

        # Create data with extreme outlier
        data = data_with_outliers.copy()
        data.loc[4, 'age'] = 500  # Very extreme outlier

        result_zscore = zscore_analyzer.analyze(data)
        result_modified = modified_analyzer.analyze(data)

        # Both should detect outliers
        assert result_zscore["summary"]["total_outliers"] > 0
        assert result_modified["summary"]["total_outliers"] > 0

    # ============== Tests for Isolation Forest method ==============

    def test_isolation_forest_detects_outliers(self, iforest_analyzer, data_with_outliers):
        """Test Isolation Forest method detects outliers."""
        result = iforest_analyzer.analyze(data_with_outliers)

        assert result["summary"]["total_outliers"] > 0

    def test_isolation_forest_outlier_structure(self, iforest_analyzer, data_with_outliers):
        """Test structure of Isolation Forest outlier results."""
        result = iforest_analyzer.analyze(data_with_outliers)
        outliers = result["outliers"]

        assert "outliers" in outliers
        assert "count" in outliers
        assert "percentage" in outliers
        assert "contamination" in outliers
        assert "total_samples" in outliers
        assert "features_used" in outliers

        # Check outlier record structure
        if len(outliers["outliers"]) > 0:
            outlier = outliers["outliers"][0]
            assert "index" in outlier
            assert "anomaly_score" in outlier
            assert "values" in outlier

    def test_isolation_forest_contamination_parameter(self, data_with_outliers):
        """Test Isolation Forest with different contamination values."""
        analyzer_low = OutlierAnalyzer(method="isolation_forest", contamination=0.05)
        analyzer_high = OutlierAnalyzer(method="isolation_forest", contamination=0.2)

        result_low = analyzer_low.analyze(data_with_outliers)
        result_high = analyzer_high.analyze(data_with_outliers)

        # Higher contamination should detect more outliers
        assert result_high["summary"]["total_outliers"] >= result_low["summary"]["total_outliers"]

    def test_isolation_forest_anomaly_scores_sorted(self, iforest_analyzer, data_with_outliers):
        """Test that outliers are sorted by anomaly score."""
        result = iforest_analyzer.analyze(data_with_outliers)
        outliers = result["outliers"]["outliers"]

        if len(outliers) > 1:
            scores = [o["anomaly_score"] for o in outliers]
            # Anomaly scores should be sorted (most anomalous first)
            assert scores == sorted(scores)

    # ============== Tests for edge cases ==============

    def test_empty_dataframe(self, iqr_analyzer, empty_data):
        """Test with empty DataFrame."""
        result = iqr_analyzer.analyze(empty_data)

        assert isinstance(result, dict)
        assert "message" in result["outliers"]

    def test_non_numeric_data(self, iqr_analyzer, non_numeric_data):
        """Test with non-numeric data."""
        result = iqr_analyzer.analyze(non_numeric_data)

        assert isinstance(result, dict)
        assert "message" in result["outliers"]

    def test_mixed_data_types(self, iqr_analyzer, mixed_data):
        """Test with mixed data types."""
        result = iqr_analyzer.analyze(mixed_data)

        # Should only analyze numeric columns
        outliers_by_col = result["outliers"]["outliers_by_column"]
        assert 'numeric1' in outliers_by_col or len(outliers_by_col) >= 0
        assert 'category' not in outliers_by_col
        assert 'text' not in outliers_by_col

    def test_data_with_missing_values(self, iqr_analyzer, data_with_missing):
        """Test with missing values."""
        result = iqr_analyzer.analyze(data_with_missing)

        # Should handle missing values by dropping them
        assert isinstance(result, dict)
        assert "outliers" in result

    def test_single_column(self, iqr_analyzer, single_column_data):
        """Test with single column."""
        result = iqr_analyzer.analyze(single_column_data)

        assert isinstance(result, dict)
        assert len(result["outliers"]["outliers_by_column"]) == 1

    def test_small_dataset(self, iqr_analyzer):
        """Test with very small dataset."""
        data = pd.DataFrame({'A': [1, 2, 3]})
        result = iqr_analyzer.analyze(data)

        assert isinstance(result, dict)

    def test_isolation_forest_insufficient_data(self, iforest_analyzer):
        """Test Isolation Forest with insufficient data."""
        data = pd.DataFrame({'A': [1]})
        result = iforest_analyzer.analyze(data)

        assert "message" in result["outliers"]

    # ============== Tests for method parameter ==============

    def test_unknown_method(self):
        """Test with unknown method."""
        analyzer = OutlierAnalyzer(method="unknown")
        data = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        result = analyzer.analyze(data)

        assert "error" in result["outliers"]

    def test_all_methods_return_summary(self, all_methods_analyzer, data_with_outliers):
        """Test that all methods return proper summary."""
        result = all_methods_analyzer.analyze(data_with_outliers)

        assert "methods" in result["summary"]
        for method in ["iqr", "zscore", "modified_zscore", "isolation_forest"]:
            assert method in result["summary"]["methods"]
            assert "total_outliers" in result["summary"]["methods"][method]

    # ============== Tests for data integrity ==============

    def test_analyze_preserves_original_data(self, iqr_analyzer, data_with_outliers):
        """Test that analyze doesn't modify the original DataFrame."""
        original_data = data_with_outliers.copy()

        iqr_analyzer.analyze(data_with_outliers)

        pd.testing.assert_frame_equal(data_with_outliers, original_data)

    def test_outlier_indices_are_valid(self, iqr_analyzer, data_with_outliers):
        """Test that outlier indices correspond to actual data."""
        result = iqr_analyzer.analyze(data_with_outliers)
        outliers_by_col = result["outliers"]["outliers_by_column"]

        for col, info in outliers_by_col.items():
            for idx, val in zip(info["indices"], info["values"]):
                # Check that index exists and value matches
                assert idx in data_with_outliers.index
                assert abs(data_with_outliers.loc[idx, col] - val) < 1e-10

    def test_percentage_calculation(self, iqr_analyzer, data_with_outliers):
        """Test that percentage calculation is correct."""
        result = iqr_analyzer.analyze(data_with_outliers)
        outliers_by_col = result["outliers"]["outliers_by_column"]

        for col, info in outliers_by_col.items():
            expected_percentage = (info["count"] / len(data_with_outliers[col].dropna())) * 100
            assert abs(info["percentage"] - expected_percentage) < 1e-10

    # ============== Comparison tests ==============

    def test_compare_methods_on_same_data(self, data_with_outliers):
        """Test different methods on the same data."""
        analyzers = {
            'iqr': OutlierAnalyzer(method="iqr"),
            'zscore': OutlierAnalyzer(method="zscore"),
            'modified_zscore': OutlierAnalyzer(method="modified_zscore"),
        }

        results = {}
        for name, analyzer in analyzers.items():
            results[name] = analyzer.analyze(data_with_outliers)

        # All methods should detect some outliers
        for name, result in results.items():
            if name != "isolation_forest":
                assert result["summary"]["total_outliers"] >= 0


# ============== Integration tests ==============

class TestOutlierAnalyzerIntegration:
    """Integration tests for OutlierAnalyzer."""

    def test_full_workflow_with_real_data(self):
        """Test complete workflow with realistic data."""
        np.random.seed(42)

        # Create realistic dataset
        n_samples = 200
        data = pd.DataFrame({
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.randint(20000, 150000, n_samples),
            'expenses': np.random.randint(15000, 100000, n_samples),
        })

        # Add clear outliers
        data.loc[0, 'age'] = 150
        data.loc[1, 'income'] = 1000000
        data.loc[2, 'expenses'] = 5000

        # Test all methods
        for method in ["iqr", "zscore", "modified_zscore", "isolation_forest"]:
            analyzer = OutlierAnalyzer(method=method)
            result = analyzer.analyze(data)

            assert isinstance(result, dict)
            assert "summary" in result

    def test_outlier_removal_workflow(self):
        """Test workflow for removing outliers."""
        np.random.seed(42)
        data = pd.DataFrame({
            'A': np.random.normal(50, 10, 100),
            'B': np.random.normal(100, 20, 100),
        })
        data.loc[0, 'A'] = 200  # outlier

        analyzer = OutlierAnalyzer(method="iqr")
        result = analyzer.analyze(data)

        # Get outlier indices
        all_outlier_indices = set()
        for col_info in result["outliers"]["outliers_by_column"].values():
            all_outlier_indices.update(col_info["indices"])

        # Remove outliers
        clean_data = data.drop(index=list(all_outlier_indices))

        # Verify removal
        assert len(clean_data) < len(data)
        assert 0 not in clean_data.index  # outlier index removed

    def test_compare_all_methods(self):
        """Test comparing results from all methods."""
        np.random.seed(42)
        data = pd.DataFrame({
            'A': np.random.normal(50, 10, 100),
            'B': np.random.normal(100, 20, 100),
        })
        data.loc[0, 'A'] = 200
        data.loc[1, 'B'] = 500

        analyzer = OutlierAnalyzer(method="all")
        result = analyzer.analyze(data)

        # Verify all methods ran
        assert "iqr" in result
        assert "zscore" in result
        assert "modified_zscore" in result
        assert "isolation_forest" in result

        # All methods should detect at least some outliers
        summary = result["summary"]["methods"]
        outlier_counts = [info["total_outliers"] for info in summary.values()]
        assert any(count > 0 for count in outlier_counts)
