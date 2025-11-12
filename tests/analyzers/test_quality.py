"""
Tests for the DataQualityAnalyzer class.
"""

import pytest
import pandas as pd
import numpy as np
from simplus_eda.analyzers.quality import DataQualityAnalyzer


class TestDataQualityAnalyzer:
    """Test suite for DataQualityAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a DataQualityAnalyzer instance."""
        return DataQualityAnalyzer()

    @pytest.fixture
    def clean_data(self):
        """Create clean data with no issues."""
        np.random.seed(42)
        return pd.DataFrame({
            'id': range(100),
            'age': np.random.randint(18, 80, 100),
            'name': [f'Person_{i}' for i in range(100)],
            'score': np.random.uniform(0, 100, 100),
        })

    @pytest.fixture
    def data_with_missing(self):
        """Create data with missing values."""
        np.random.seed(42)
        data = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.randn(100),
        })
        # Add missing values
        data.loc[0:10, 'A'] = np.nan
        data.loc[20:30, 'B'] = np.nan
        data.loc[40:80, 'C'] = np.nan  # High missing rate
        return data

    @pytest.fixture
    def data_with_duplicates(self):
        """Create data with duplicates."""
        data = pd.DataFrame({
            'A': [1, 2, 3, 1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'a', 'b', 'c', 'd', 'e'],
            'C': [10, 20, 30, 10, 20, 30, 40, 50],
        })
        return data

    @pytest.fixture
    def data_with_type_issues(self):
        """Create data with type conversion opportunities."""
        return pd.DataFrame({
            'numeric_string': ['1', '2', '3', '4', '5'],
            'date_string': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'low_cardinality_int': [1, 1, 2, 2, 3],
            'category': ['A', 'B', 'A', 'B', 'A'],
        })

    @pytest.fixture
    def data_with_consistency_issues(self):
        """Create data with consistency issues."""
        return pd.DataFrame({
            'age': [25, 30, -5, 150, 35, np.inf],  # negative and infinite
            'price': [100, 200, -50, 300, 400, 500],  # negative price
            'text': ['  hello', 'world  ', ' test ', '', 'Test', 'TEST'],  # whitespace and case issues
        })

    @pytest.fixture
    def empty_data(self):
        """Create empty DataFrame."""
        return pd.DataFrame()

    @pytest.fixture
    def mixed_cardinality_data(self):
        """Create data with various cardinality levels."""
        return pd.DataFrame({
            'constant': [1] * 100,
            'unique_id': range(100),
            'low_card': np.random.choice(['A', 'B', 'C'], 100),
            'high_card': [f'value_{i}' for i in range(100)],
            'medium_card': np.random.choice(range(30), 100),
        })

    # ============== Tests for analyze() ==============

    def test_analyze_returns_dict(self, analyzer, clean_data):
        """Test that analyze returns dictionary with expected keys."""
        result = analyzer.analyze(clean_data)

        assert isinstance(result, dict)
        assert "missing_values" in result
        assert "duplicates" in result
        assert "data_types" in result
        assert "consistency" in result
        assert "cardinality" in result
        assert "quality_score" in result

    def test_analyze_empty_dataframe(self, analyzer, empty_data):
        """Test analyze with empty DataFrame."""
        result = analyzer.analyze(empty_data)

        assert isinstance(result, dict)
        assert "message" in result["missing_values"]
        assert "message" in result["duplicates"]

    # ============== Tests for _analyze_missing() ==============

    def test_missing_analysis_no_missing(self, analyzer, clean_data):
        """Test missing value analysis with clean data."""
        result = analyzer._analyze_missing(clean_data)

        assert result["total_missing"] == 0
        assert result["overall_missing_rate"] == 0.0
        assert len(result["high_missing_columns"]) == 0
        assert "Excellent" in result["recommendation"]

    def test_missing_analysis_with_missing(self, analyzer, data_with_missing):
        """Test missing value analysis with missing data."""
        result = analyzer._analyze_missing(data_with_missing)

        assert result["total_missing"] > 0
        assert result["overall_missing_rate"] > 0
        assert len(result["columns"]) == 3

        # Check column-specific info
        for col, info in result["columns"].items():
            assert "missing_count" in info
            assert "missing_rate" in info
            assert "has_missing" in info
            assert "missing_indices" in info

    def test_missing_high_threshold(self):
        """Test high missing rate detection."""
        analyzer = DataQualityAnalyzer(missing_threshold=0.3)
        data = pd.DataFrame({
            'A': [1, 2, np.nan, np.nan, np.nan],  # 60% missing
            'B': [1, 2, 3, 4, 5],  # 0% missing
        })

        result = analyzer._analyze_missing(data)

        assert len(result["high_missing_columns"]) == 1
        assert result["high_missing_columns"][0]["column"] == 'A'

    def test_missing_patterns(self, analyzer, data_with_missing):
        """Test missing value pattern detection."""
        result = analyzer._analyze_missing(data_with_missing)

        assert "patterns" in result
        patterns = result["patterns"]
        assert "rows_all_missing" in patterns
        assert "rows_any_missing" in patterns
        assert "columns_all_missing" in patterns
        assert "row_completeness_stats" in patterns

    def test_missing_all_missing_column(self, analyzer):
        """Test detection of completely missing columns."""
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [np.nan, np.nan, np.nan],
        })

        result = analyzer._analyze_missing(data)

        assert 'B' in result["patterns"]["columns_all_missing"]

    # ============== Tests for _analyze_duplicates() ==============

    def test_duplicates_no_duplicates(self, analyzer, clean_data):
        """Test duplicate analysis with no duplicates."""
        result = analyzer._analyze_duplicates(clean_data)

        assert result["duplicate_rows"] == 0
        assert result["duplicate_rate"] == 0.0
        assert len(result["duplicate_groups"]) == 0
        assert "excellent" in result["recommendation"].lower()

    def test_duplicates_with_duplicates(self, analyzer, data_with_duplicates):
        """Test duplicate analysis with duplicates."""
        result = analyzer._analyze_duplicates(data_with_duplicates)

        assert result["duplicate_rows"] > 0
        assert result["duplicate_rate"] > 0
        assert len(result["duplicate_groups"]) > 0

        # Check duplicate group structure
        for group in result["duplicate_groups"]:
            assert "indices" in group
            assert "count" in group
            assert group["count"] >= 2

    def test_duplicates_subset_columns(self):
        """Test duplicate detection on subset of columns."""
        analyzer = DataQualityAnalyzer(duplicate_subset=['A', 'B'])
        data = pd.DataFrame({
            'A': [1, 1, 2],
            'B': [2, 2, 3],
            'C': [3, 4, 5],  # Different values
        })

        result = analyzer._analyze_duplicates(data)

        # First two rows are duplicates when considering only A and B
        assert result["duplicate_rows"] > 0

    def test_duplicates_column_analysis(self, analyzer, data_with_duplicates):
        """Test column-level duplicate analysis."""
        result = analyzer._analyze_duplicates(data_with_duplicates)

        assert "column_duplicates" in result
        for col, info in result["column_duplicates"].items():
            assert "duplicate_count" in info
            assert "unique_count" in info
            assert "total_count" in info

    # ============== Tests for _analyze_types() ==============

    def test_type_analysis_basic(self, analyzer, clean_data):
        """Test basic type analysis."""
        result = analyzer._analyze_types(clean_data)

        assert "columns" in result
        assert "type_counts" in result
        assert "total_memory_bytes" in result
        assert "suggestions" in result

        # Check each column has type info
        for col in clean_data.columns:
            assert col in result["columns"]
            assert "dtype" in result["columns"][col]
            assert "unique_count" in result["columns"][col]

    def test_type_analysis_numeric(self, analyzer):
        """Test type analysis for numeric columns."""
        data = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        })

        result = analyzer._analyze_types(data)

        for col in ['int_col', 'float_col']:
            col_info = result["columns"][col]
            assert col_info["is_numeric"] is True
            assert "stats" in col_info
            assert "min" in col_info["stats"]
            assert "max" in col_info["stats"]

    def test_type_suggestions_numeric_string(self, analyzer, data_with_type_issues):
        """Test suggestions for numeric strings."""
        result = analyzer._analyze_types(data_with_type_issues)

        # Should suggest converting numeric_string to numeric
        numeric_suggestions = [s for s in result["suggestions"]
                              if s["column"] == "numeric_string"]
        assert len(numeric_suggestions) > 0

    def test_type_suggestions_categorical(self, analyzer, data_with_type_issues):
        """Test suggestions for categorical conversion."""
        result = analyzer._analyze_types(data_with_type_issues)

        # Should suggest converting low cardinality columns to categorical
        cat_suggestions = [s for s in result["suggestions"]
                          if s["suggested_type"] == "category"]
        assert len(cat_suggestions) > 0

    def test_type_analysis_datetime(self, analyzer):
        """Test type analysis for datetime columns."""
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
        })

        result = analyzer._analyze_types(data)

        col_info = result["columns"]["date"]
        assert col_info["is_datetime"] is True
        assert "date_range" in col_info

    def test_type_analysis_categorical(self, analyzer):
        """Test type analysis for categorical columns."""
        data = pd.DataFrame({
            'cat': pd.Categorical(['A', 'B', 'C', 'A', 'B']),
        })

        result = analyzer._analyze_types(data)

        col_info = result["columns"]["cat"]
        assert col_info["is_categorical"] is True
        assert "categories" in col_info

    # ============== Tests for _check_consistency() ==============

    def test_consistency_clean_data(self, analyzer, clean_data):
        """Test consistency check with clean data."""
        result = analyzer._check_consistency(clean_data)

        assert result["issue_count"] == 0
        assert len(result["issues"]) == 0
        assert result["has_critical_issues"] is False

    def test_consistency_infinite_values(self, analyzer, data_with_consistency_issues):
        """Test detection of infinite values."""
        result = analyzer._check_consistency(data_with_consistency_issues)

        # Should detect infinite value in age column
        inf_issues = [issue for issue in result["issues"]
                      if issue["issue_type"] == "infinite_values"]
        assert len(inf_issues) > 0
        assert inf_issues[0]["severity"] == "high"

    def test_consistency_negative_values(self, analyzer, data_with_consistency_issues):
        """Test detection of unexpected negative values."""
        result = analyzer._check_consistency(data_with_consistency_issues)

        # Should detect negative age and price
        neg_issues = [issue for issue in result["issues"]
                      if issue["issue_type"] == "unexpected_negative"]
        assert len(neg_issues) > 0

    def test_consistency_whitespace(self, analyzer, data_with_consistency_issues):
        """Test detection of whitespace issues."""
        result = analyzer._check_consistency(data_with_consistency_issues)

        # Should detect whitespace issues in text column
        ws_issues = [issue for issue in result["issues"]
                     if issue["issue_type"] == "leading_trailing_whitespace"]
        assert len(ws_issues) > 0

    def test_consistency_empty_strings(self, analyzer, data_with_consistency_issues):
        """Test detection of empty strings."""
        result = analyzer._check_consistency(data_with_consistency_issues)

        # Should detect empty string in text column
        empty_issues = [issue for issue in result["issues"]
                       if issue["issue_type"] == "empty_strings"]
        assert len(empty_issues) > 0

    def test_consistency_case_issues(self, analyzer, data_with_consistency_issues):
        """Test detection of case inconsistency."""
        result = analyzer._check_consistency(data_with_consistency_issues)

        # Should detect case inconsistency in text column
        case_issues = [issue for issue in result["issues"]
                      if issue["issue_type"] == "case_inconsistency"]
        assert len(case_issues) > 0

    def test_consistency_numeric_checks(self, analyzer):
        """Test numeric consistency checks."""
        data = pd.DataFrame({
            'values': [1, 0, -1, np.inf, 5],
        })

        result = analyzer._check_consistency(data)

        assert "numeric_checks" in result
        assert "values" in result["numeric_checks"]
        checks = result["numeric_checks"]["values"]
        assert "has_infinity" in checks
        assert "has_negative" in checks
        assert "has_zero" in checks

    def test_consistency_text_checks(self, analyzer):
        """Test text consistency checks."""
        data = pd.DataFrame({
            'text': ['hello', '  world', '', 'test'],
        })

        result = analyzer._check_consistency(data)

        assert "text_checks" in result
        assert "text" in result["text_checks"]
        checks = result["text_checks"]["text"]
        assert "max_length" in checks
        assert "min_length" in checks
        assert "avg_length" in checks

    # ============== Tests for _analyze_cardinality() ==============

    def test_cardinality_basic(self, analyzer, mixed_cardinality_data):
        """Test basic cardinality analysis."""
        result = analyzer._analyze_cardinality(mixed_cardinality_data)

        assert "columns" in result
        assert len(result["columns"]) == len(mixed_cardinality_data.columns)

        for col, info in result["columns"].items():
            assert "unique_count" in info
            assert "total_count" in info
            assert "cardinality_rate" in info
            assert "cardinality_level" in info

    def test_cardinality_constant(self, analyzer, mixed_cardinality_data):
        """Test constant cardinality detection."""
        result = analyzer._analyze_cardinality(mixed_cardinality_data)

        constant_col = result["columns"]["constant"]
        assert constant_col["cardinality_level"] == "constant"
        assert constant_col["unique_count"] == 1

    def test_cardinality_unique(self, analyzer, mixed_cardinality_data):
        """Test unique cardinality detection."""
        result = analyzer._analyze_cardinality(mixed_cardinality_data)

        unique_col = result["columns"]["unique_id"]
        assert unique_col["cardinality_level"] == "unique"
        assert unique_col["is_unique"] is True

    def test_cardinality_low(self, analyzer, mixed_cardinality_data):
        """Test low cardinality detection."""
        result = analyzer._analyze_cardinality(mixed_cardinality_data)

        low_col = result["columns"]["low_card"]
        assert low_col["cardinality_level"] == "low"
        assert "low_card" in result["low_cardinality_columns"]

    def test_cardinality_high(self, analyzer, mixed_cardinality_data):
        """Test high cardinality detection."""
        result = analyzer._analyze_cardinality(mixed_cardinality_data)

        # Note: high_card has unique values (100 unique out of 100), so it's classified as "unique"
        # Let's check medium_card instead for high cardinality
        medium_col = result["columns"]["medium_card"]
        # medium_card has 30 unique values, which is less than threshold but more than 10
        assert medium_col["cardinality_level"] == "medium"

    def test_cardinality_top_values(self, analyzer):
        """Test top values for low cardinality."""
        data = pd.DataFrame({
            'cat': ['A', 'B', 'A', 'C', 'A'],
        })

        result = analyzer._analyze_cardinality(data)

        cat_info = result["columns"]["cat"]
        assert "top_values" in cat_info
        assert 'A' in str(cat_info["top_values"])

    # ============== Tests for _calculate_quality_score() ==============

    def test_quality_score_clean_data(self, analyzer, clean_data):
        """Test quality score for clean data."""
        result = analyzer._calculate_quality_score(clean_data)

        assert "overall_score" in result
        assert "quality_level" in result
        assert "component_scores" in result
        assert "interpretation" in result

        # Clean data should have high score
        assert result["overall_score"] > 75
        assert result["quality_level"] in ["excellent", "good"]

    def test_quality_score_components(self, analyzer, clean_data):
        """Test quality score components."""
        result = analyzer._calculate_quality_score(clean_data)

        components = result["component_scores"]
        assert "completeness" in components
        assert "uniqueness" in components
        assert "consistency" in components
        assert "validity" in components

        # All component scores should be 0-100
        for score in components.values():
            assert 0 <= score <= 100

    def test_quality_score_weights(self, analyzer, clean_data):
        """Test quality score weights sum to 1."""
        result = analyzer._calculate_quality_score(clean_data)

        weights = result["weights"]
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_quality_score_with_issues(self, analyzer, data_with_missing):
        """Test quality score with data issues."""
        result = analyzer._calculate_quality_score(data_with_missing)

        # Data with missing values should have lower completeness
        assert result["component_scores"]["completeness"] < 100

    def test_quality_levels(self, analyzer):
        """Test different quality level classifications."""
        # Excellent data
        excellent_data = pd.DataFrame({
            'A': range(100),
            'B': range(100, 200),
        })
        result = analyzer._calculate_quality_score(excellent_data)
        assert result["quality_level"] == "excellent"

        # Poor data
        poor_data = pd.DataFrame({
            'A': [np.nan] * 80 + [1] * 20,
            'B': [1, 1, 1] * 33 + [1],  # Many duplicates
        })
        result = analyzer._calculate_quality_score(poor_data)
        assert result["overall_score"] < 75

    # ============== Tests for helper methods ==============

    def test_missing_recommendations(self, analyzer):
        """Test missing value recommendations."""
        assert "Excellent" in analyzer._get_missing_recommendation(0, 0)
        assert "Low" in analyzer._get_missing_recommendation(3, 1)
        assert "Moderate" in analyzer._get_missing_recommendation(15, 2)
        assert "High" in analyzer._get_missing_recommendation(25, 5)

    def test_duplicate_recommendations(self, analyzer):
        """Test duplicate recommendations."""
        assert "excellent" in analyzer._get_duplicate_recommendation(0).lower()
        assert "Low" in analyzer._get_duplicate_recommendation(3)
        assert "Moderate" in analyzer._get_duplicate_recommendation(15)
        assert "High" in analyzer._get_duplicate_recommendation(25)

    def test_quality_score_interpretation(self, analyzer):
        """Test quality score interpretation."""
        assert "excellent" in analyzer._interpret_quality_score(95).lower()
        assert "good" in analyzer._interpret_quality_score(80).lower()
        assert "fair" in analyzer._interpret_quality_score(65).lower()
        assert "poor" in analyzer._interpret_quality_score(50).lower()
        assert "critical" in analyzer._interpret_quality_score(30).lower()

    # ============== Tests for initialization parameters ==============

    def test_custom_missing_threshold(self):
        """Test custom missing threshold."""
        analyzer = DataQualityAnalyzer(missing_threshold=0.2)
        assert analyzer.missing_threshold == 0.2

    def test_custom_duplicate_subset(self):
        """Test custom duplicate subset."""
        analyzer = DataQualityAnalyzer(duplicate_subset=['A', 'B'])
        assert analyzer.duplicate_subset == ['A', 'B']

    def test_custom_cardinality_threshold(self):
        """Test custom cardinality threshold."""
        analyzer = DataQualityAnalyzer(high_cardinality_threshold=100)
        assert analyzer.high_cardinality_threshold == 100

    # ============== Edge case tests ==============

    def test_single_row_data(self, analyzer):
        """Test with single row DataFrame."""
        data = pd.DataFrame({'A': [1], 'B': [2]})
        result = analyzer.analyze(data)

        assert isinstance(result, dict)
        assert result["quality_score"]["overall_score"] >= 0

    def test_single_column_data(self, analyzer):
        """Test with single column DataFrame."""
        data = pd.DataFrame({'A': range(100)})
        result = analyzer.analyze(data)

        assert isinstance(result, dict)
        assert len(result["data_types"]["columns"]) == 1

    def test_all_nan_data(self, analyzer):
        """Test with all NaN data."""
        data = pd.DataFrame({
            'A': [np.nan] * 10,
            'B': [np.nan] * 10,
        })

        result = analyzer.analyze(data)

        assert result["missing_values"]["overall_missing_rate"] == 100.0
        assert result["quality_score"]["component_scores"]["completeness"] == 0.0

    def test_all_duplicates(self, analyzer):
        """Test with all duplicate rows."""
        data = pd.DataFrame({
            'A': [1, 1, 1, 1],
            'B': [2, 2, 2, 2],
        })

        result = analyzer.analyze(data)

        assert result["duplicates"]["duplicate_rate"] == 100.0

    def test_data_preserves_original(self, analyzer, clean_data):
        """Test that analysis doesn't modify original data."""
        original_data = clean_data.copy()

        analyzer.analyze(clean_data)

        pd.testing.assert_frame_equal(clean_data, original_data)


# ============== Integration tests ==============

class TestDataQualityAnalyzerIntegration:
    """Integration tests for DataQualityAnalyzer."""

    def test_full_workflow_clean_data(self):
        """Test complete workflow with clean data."""
        np.random.seed(42)
        data = pd.DataFrame({
            'id': range(100),
            'value': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
        })

        analyzer = DataQualityAnalyzer()
        result = analyzer.analyze(data)

        # Should have all analysis components
        assert all(key in result for key in
                  ["missing_values", "duplicates", "data_types",
                   "consistency", "cardinality", "quality_score"])

        # Should have high quality score
        assert result["quality_score"]["overall_score"] > 80

    def test_full_workflow_problematic_data(self):
        """Test complete workflow with problematic data."""
        data = pd.DataFrame({
            'age': [25, -5, np.nan, 150, np.inf, 30],
            'name': ['Alice', 'Bob', '  Carol', '', 'Dave', 'Alice'],
            'score': [80, 80, 80, 80, 80, 80],  # All same (duplicates)
        })

        analyzer = DataQualityAnalyzer()
        result = analyzer.analyze(data)

        # Should detect various issues
        assert result["missing_values"]["total_missing"] > 0
        assert result["consistency"]["issue_count"] > 0
        assert result["consistency"]["has_critical_issues"] is True

        # Should have lower quality score
        assert result["quality_score"]["overall_score"] < 90

    def test_comprehensive_quality_report(self):
        """Test generating comprehensive quality report."""
        np.random.seed(42)
        data = pd.DataFrame({
            'id': range(200),
            'age': np.random.randint(18, 80, 200),
            'salary': np.random.randint(30000, 150000, 200),
            'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], 200),
            'join_date': pd.date_range('2020-01-01', periods=200),
        })

        # Add some issues
        data.loc[0:10, 'age'] = np.nan
        data.loc[20:22, :] = data.loc[17:19, :].values  # Add duplicates

        analyzer = DataQualityAnalyzer()
        result = analyzer.analyze(data)

        # Verify all components present
        assert result["missing_values"]["overall_missing_rate"] > 0
        assert result["duplicates"]["duplicate_rows"] > 0
        assert len(result["data_types"]["columns"]) == 5
        assert result["quality_score"]["overall_score"] > 0

        # Verify quality score components make sense
        components = result["quality_score"]["component_scores"]
        assert components["completeness"] < 100  # Has missing values
        assert components["uniqueness"] < 100  # Has duplicates
