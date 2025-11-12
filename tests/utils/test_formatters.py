"""
Tests for output formatting utilities.
"""

import pytest
import pandas as pd
import json
from datetime import datetime, timedelta
from simplus_eda.utils.formatters import OutputFormatter


class TestJSONFormatting:
    """Test JSON formatting methods."""

    def test_to_json_pretty(self):
        """Test pretty-printed JSON formatting."""
        data = {"name": "test", "value": 123, "nested": {"key": "value"}}
        result = OutputFormatter.to_json(data, pretty=True)

        assert isinstance(result, str)
        assert "  " in result  # Check for indentation
        assert "name" in result
        assert "123" in result

        # Verify valid JSON
        parsed = json.loads(result)
        assert parsed == data

    def test_to_json_compact(self):
        """Test compact JSON formatting."""
        data = {"name": "test", "value": 123}
        result = OutputFormatter.to_json(data, pretty=False)

        assert isinstance(result, str)
        assert "\n" not in result  # No newlines in compact mode
        assert "  " not in result  # No indentation

        # Verify valid JSON
        parsed = json.loads(result)
        assert parsed == data

    def test_to_json_sort_keys(self):
        """Test JSON with sorted keys."""
        data = {"z": 1, "a": 2, "m": 3}
        result = OutputFormatter.to_json(data, sort_keys=True)

        # Check key order
        keys_order = list(json.loads(result).keys())
        assert keys_order == ["a", "m", "z"]

    def test_to_json_with_datetime(self):
        """Test JSON formatting with datetime objects."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        data = {"timestamp": dt, "value": 100}

        result = OutputFormatter.to_json(data)

        # Should handle datetime conversion
        assert isinstance(result, str)
        assert "2024" in result


class TestMarkdownFormatting:
    """Test markdown formatting methods."""

    def test_to_markdown_simple_dict(self):
        """Test markdown formatting with simple dictionary."""
        data = {"metric_one": 100, "metric_two": 200}
        result = OutputFormatter.to_markdown(data)

        assert isinstance(result, str)
        assert "## Metric One" in result
        assert "## Metric Two" in result
        assert "100" in result
        assert "200" in result

    def test_to_markdown_with_title(self):
        """Test markdown formatting with title."""
        data = {"metric": 100}
        result = OutputFormatter.to_markdown(data, title="Test Report")

        assert "# Test Report" in result
        assert "## Metric" in result

    def test_to_markdown_nested_dict(self):
        """Test markdown formatting with nested dictionaries."""
        data = {
            "section_one": {
                "subsection_a": 10,
                "subsection_b": 20
            },
            "section_two": {
                "subsection_c": 30
            }
        }
        result = OutputFormatter.to_markdown(data)

        assert "## Section One" in result
        assert "### Subsection A" in result
        assert "### Subsection B" in result
        assert "## Section Two" in result
        assert "### Subsection C" in result

    def test_to_markdown_with_list(self):
        """Test markdown formatting with lists."""
        data = {
            "items": ["apple", "banana", "cherry"]
        }
        result = OutputFormatter.to_markdown(data)

        assert "## Items" in result
        assert "- apple" in result
        assert "- banana" in result
        assert "- cherry" in result

    def test_to_markdown_with_dict_list(self):
        """Test markdown formatting with list of dictionaries."""
        data = {
            "records": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ]
        }
        result = OutputFormatter.to_markdown(data)

        assert "## Records" in result
        assert "name: Alice" in result
        assert "age: 30" in result
        assert "name: Bob" in result

    def test_to_markdown_with_dataframe(self):
        """Test markdown formatting with DataFrame."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "age": [30, 25]
        })
        data = {"table": df}
        result = OutputFormatter.to_markdown(data)

        assert "## Table" in result
        assert "name" in result
        assert "age" in result
        assert "Alice" in result
        assert "Bob" in result

    def test_to_markdown_deep_nesting(self):
        """Test markdown formatting with deep nesting."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {
                                "level6": {
                                    "level7": "deep_value"
                                }
                            }
                        }
                    }
                }
            }
        }
        result = OutputFormatter.to_markdown(data)

        # Should handle deep nesting gracefully
        assert isinstance(result, str)
        assert "deep_value" in result


class TestHTMLFormatting:
    """Test HTML formatting methods."""

    def test_to_html_table_from_dataframe(self):
        """Test HTML table from DataFrame."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "age": [30, 25]
        })
        result = OutputFormatter.to_html_table(df)

        assert "<table" in result
        assert "</table>" in result
        assert "Alice" in result
        assert "Bob" in result
        assert "name" in result
        assert "age" in result

    def test_to_html_table_from_list(self):
        """Test HTML table from list of dictionaries."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
        result = OutputFormatter.to_html_table(data)

        assert "<table" in result
        assert "Alice" in result
        assert "30" in result

    def test_to_html_table_with_title(self):
        """Test HTML table with title."""
        df = pd.DataFrame({"col": [1, 2]})
        result = OutputFormatter.to_html_table(df, title="Test Table")

        assert "<h3>Test Table</h3>" in result
        assert "<table" in result

    def test_to_html_table_with_classes(self):
        """Test HTML table with custom CSS classes."""
        df = pd.DataFrame({"col": [1, 2]})
        result = OutputFormatter.to_html_table(df, classes="custom-class")

        # Pandas adds both dataframe and custom classes
        assert 'class="dataframe custom-class"' in result or 'class="custom-class"' in result


class TestCSVFormatting:
    """Test CSV formatting methods."""

    def test_to_csv_from_dataframe(self):
        """Test CSV export from DataFrame."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "age": [30, 25]
        })
        result = OutputFormatter.to_csv(df)

        assert "name,age" in result
        assert "Alice,30" in result
        assert "Bob,25" in result

    def test_to_csv_from_list(self):
        """Test CSV export from list of dictionaries."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
        result = OutputFormatter.to_csv(data)

        assert "name,age" in result
        assert "Alice,30" in result

    def test_to_csv_with_index(self):
        """Test CSV export with index."""
        df = pd.DataFrame({"col": [1, 2]})
        result = OutputFormatter.to_csv(df, include_index=True)

        # Should include index column
        lines = result.strip().split('\n')
        assert len(lines) == 3  # Header + 2 rows


class TestPercentageFormatting:
    """Test percentage formatting methods."""

    def test_format_percentage_decimal_input(self):
        """Test percentage formatting with decimal input (0-1)."""
        result = OutputFormatter.format_percentage(0.75)
        assert result == "75.00%"

    def test_format_percentage_whole_input(self):
        """Test percentage formatting with whole number input."""
        result = OutputFormatter.format_percentage(75)
        assert result == "75.00%"

    def test_format_percentage_custom_decimals(self):
        """Test percentage formatting with custom decimal places."""
        result = OutputFormatter.format_percentage(0.756, decimals=1)
        assert result == "75.6%"

    def test_format_percentage_no_sign(self):
        """Test percentage formatting without % sign."""
        result = OutputFormatter.format_percentage(0.75, include_sign=False)
        assert result == "75.00"

    def test_format_percentage_zero(self):
        """Test percentage formatting with zero."""
        result = OutputFormatter.format_percentage(0)
        assert result == "0.00%"

    def test_format_percentage_one(self):
        """Test percentage formatting with one."""
        result = OutputFormatter.format_percentage(1.0)
        assert result == "100.00%"


class TestNumberFormatting:
    """Test number formatting methods."""

    def test_format_number_basic(self):
        """Test basic number formatting."""
        result = OutputFormatter.format_number(1234.5678)
        assert result == "1,234.57"

    def test_format_number_without_separator(self):
        """Test number formatting without thousands separator."""
        result = OutputFormatter.format_number(1234.5678, thousands_separator=False)
        assert result == "1234.57"

    def test_format_number_custom_decimals(self):
        """Test number formatting with custom decimal places."""
        result = OutputFormatter.format_number(1234.5678, decimals=3)
        assert result == "1,234.568"

    def test_format_number_zero_decimals(self):
        """Test number formatting with zero decimal places."""
        result = OutputFormatter.format_number(1234.5678, decimals=0)
        assert result == "1,235"


class TestLargeNumberFormatting:
    """Test large number formatting methods."""

    def test_format_large_number_thousands(self):
        """Test formatting thousands."""
        result = OutputFormatter.format_large_number(1500)
        assert result == "1.5K"

    def test_format_large_number_millions(self):
        """Test formatting millions."""
        result = OutputFormatter.format_large_number(2500000)
        assert result == "2.5M"

    def test_format_large_number_billions(self):
        """Test formatting billions."""
        result = OutputFormatter.format_large_number(3500000000)
        assert result == "3.5B"

    def test_format_large_number_trillions(self):
        """Test formatting trillions."""
        result = OutputFormatter.format_large_number(4500000000000)
        assert result == "4.5T"

    def test_format_large_number_small(self):
        """Test formatting numbers less than 1000."""
        result = OutputFormatter.format_large_number(500)
        assert result == "500.0"

    def test_format_large_number_negative(self):
        """Test formatting negative numbers."""
        result = OutputFormatter.format_large_number(-1500)
        assert result == "-1.5K"

    def test_format_large_number_custom_decimals(self):
        """Test formatting with custom decimal places."""
        result = OutputFormatter.format_large_number(1567, decimals=2)
        assert result == "1.57K"


class TestCurrencyFormatting:
    """Test currency formatting methods."""

    def test_format_currency_positive(self):
        """Test currency formatting with positive value."""
        result = OutputFormatter.format_currency(1234.56)
        assert result == "$1,234.56"

    def test_format_currency_negative(self):
        """Test currency formatting with negative value."""
        result = OutputFormatter.format_currency(-1234.56)
        assert result == "-$1,234.56"

    def test_format_currency_custom_symbol(self):
        """Test currency formatting with custom symbol."""
        result = OutputFormatter.format_currency(1234.56, currency_symbol="€")
        assert result == "€1,234.56"

    def test_format_currency_custom_decimals(self):
        """Test currency formatting with custom decimal places."""
        result = OutputFormatter.format_currency(1234.567, decimals=3)
        assert result == "$1,234.567"

    def test_format_currency_zero(self):
        """Test currency formatting with zero."""
        result = OutputFormatter.format_currency(0)
        assert result == "$0.00"


class TestFileSizeFormatting:
    """Test file size formatting methods."""

    def test_format_file_size_bytes(self):
        """Test file size formatting for bytes."""
        result = OutputFormatter.format_file_size(500)
        assert result == "500.00 B"

    def test_format_file_size_kilobytes(self):
        """Test file size formatting for kilobytes."""
        result = OutputFormatter.format_file_size(1536)
        assert result == "1.50 KB"

    def test_format_file_size_megabytes(self):
        """Test file size formatting for megabytes."""
        result = OutputFormatter.format_file_size(1572864)
        assert result == "1.50 MB"

    def test_format_file_size_gigabytes(self):
        """Test file size formatting for gigabytes."""
        result = OutputFormatter.format_file_size(1610612736)
        assert result == "1.50 GB"

    def test_format_file_size_terabytes(self):
        """Test file size formatting for terabytes."""
        result = OutputFormatter.format_file_size(1649267441664)
        assert result == "1.50 TB"

    def test_format_file_size_petabytes(self):
        """Test file size formatting for petabytes."""
        result = OutputFormatter.format_file_size(1688849860263936)
        assert result == "1.50 PB"

    def test_format_file_size_zero(self):
        """Test file size formatting with zero."""
        result = OutputFormatter.format_file_size(0)
        assert result == "0.00 B"

    def test_format_file_size_custom_decimals(self):
        """Test file size formatting with custom decimals."""
        result = OutputFormatter.format_file_size(1536, decimals=1)
        assert result == "1.5 KB"


class TestDurationFormatting:
    """Test duration formatting methods."""

    def test_format_duration_seconds(self):
        """Test duration formatting for seconds."""
        result = OutputFormatter.format_duration(45)
        assert result == "45s"

    def test_format_duration_minutes(self):
        """Test duration formatting for minutes."""
        result = OutputFormatter.format_duration(150)
        assert result == "2m 30s"

    def test_format_duration_hours(self):
        """Test duration formatting for hours."""
        result = OutputFormatter.format_duration(5400)
        assert result == "1h 30m"

    def test_format_duration_days(self):
        """Test duration formatting for days."""
        result = OutputFormatter.format_duration(90000)
        assert result == "1d 1h"

    def test_format_duration_verbose(self):
        """Test verbose duration formatting."""
        result = OutputFormatter.format_duration(150, verbose=True)
        assert result == "2 minutes, 30 seconds"

    def test_format_duration_verbose_singular(self):
        """Test verbose duration formatting with singular units."""
        result = OutputFormatter.format_duration(61, verbose=True)
        assert result == "1 minute, 1 second"

    def test_format_duration_max_units(self):
        """Test duration formatting with max units limit."""
        result = OutputFormatter.format_duration(90061, max_units=1)
        assert result == "1d"

    def test_format_duration_zero(self):
        """Test duration formatting with zero."""
        result = OutputFormatter.format_duration(0)
        assert result == "0s"

    def test_format_duration_zero_verbose(self):
        """Test verbose duration formatting with zero."""
        result = OutputFormatter.format_duration(0, verbose=True)
        assert result == "0 seconds"


class TestTimestampFormatting:
    """Test timestamp formatting methods."""

    def test_format_timestamp_default(self):
        """Test timestamp formatting with default format."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = OutputFormatter.format_timestamp(dt)
        assert result == "2024-01-15 10:30:45"

    def test_format_timestamp_custom_format(self):
        """Test timestamp formatting with custom format."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = OutputFormatter.format_timestamp(dt, format_string="%Y/%m/%d")
        assert result == "2024/01/15"

    def test_format_timestamp_time_only(self):
        """Test timestamp formatting with time only."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = OutputFormatter.format_timestamp(dt, format_string="%H:%M:%S")
        assert result == "10:30:45"


class TestStatisticalSummaryFormatting:
    """Test statistical summary formatting methods."""

    def test_format_statistical_summary_simple(self):
        """Test statistical summary formatting with simple data."""
        data = {
            "count": 100,
            "mean": 50.5,
            "std": 28.9
        }
        result = OutputFormatter.format_statistical_summary(data)

        assert "Statistical Summary" in result
        assert "Count: 100" in result
        assert "Mean:" in result
        assert "Std:" in result

    def test_format_statistical_summary_nested(self):
        """Test statistical summary formatting with nested data."""
        data = {
            "basic_stats": {
                "count": 100,
                "mean": 50.5
            },
            "advanced_stats": {
                "skewness": 0.5,
                "kurtosis": 2.5
            }
        }
        result = OutputFormatter.format_statistical_summary(data)

        assert "Basic Stats:" in result
        assert "Count: 100" in result
        assert "Advanced Stats:" in result
        assert "Skewness:" in result

    def test_format_statistical_summary_with_list(self):
        """Test statistical summary formatting with lists."""
        data = {
            "outliers": [1, 2, 99, 100]
        }
        result = OutputFormatter.format_statistical_summary(data)

        assert "Outliers:" in result
        assert "- 1" in result
        assert "- 99" in result


class TestMarkdownTableCreation:
    """Test markdown table creation methods."""

    def test_create_markdown_table_basic(self):
        """Test basic markdown table creation."""
        headers = ["Name", "Age", "City"]
        rows = [
            ["Alice", 30, "NYC"],
            ["Bob", 25, "LA"]
        ]
        result = OutputFormatter.create_markdown_table(headers, rows)

        assert "| Name | Age | City |" in result
        assert "| Alice | 30 | NYC |" in result
        assert "| Bob | 25 | LA |" in result

    def test_create_markdown_table_alignment(self):
        """Test markdown table with alignment."""
        headers = ["Left", "Center", "Right"]
        rows = [["A", "B", "C"]]
        align = ["left", "center", "right"]
        result = OutputFormatter.create_markdown_table(headers, rows, align)

        assert "| :-- | :-: | --: |" in result

    def test_create_markdown_table_empty(self):
        """Test markdown table with empty data."""
        result = OutputFormatter.create_markdown_table([], [])
        assert result == ""

    def test_create_markdown_table_none_values(self):
        """Test markdown table with None values."""
        headers = ["A", "B"]
        rows = [[1, None], [None, 2]]
        result = OutputFormatter.create_markdown_table(headers, rows)

        assert "| 1 |  |" in result
        assert "|  | 2 |" in result

    def test_create_markdown_table_mismatched_rows(self):
        """Test markdown table with rows shorter than headers."""
        headers = ["A", "B", "C"]
        rows = [[1, 2], [3]]
        result = OutputFormatter.create_markdown_table(headers, rows)

        # Should pad with empty strings
        assert "| 1 | 2 |  |" in result
        assert "| 3 |  |  |" in result


class TestComparisonTableCreation:
    """Test comparison table creation methods."""

    def test_create_comparison_table_basic(self):
        """Test basic comparison table creation."""
        before = {"count": 100, "mean": 50.0}
        after = {"count": 150, "mean": 55.0}
        result = OutputFormatter.create_comparison_table(before, after)

        assert "## Comparison" in result
        assert "| Metric | Before | After | Change |" in result
        assert "Count" in result
        assert "Mean" in result

    def test_create_comparison_table_custom_title(self):
        """Test comparison table with custom title."""
        before = {"value": 10}
        after = {"value": 20}
        result = OutputFormatter.create_comparison_table(before, after, title="Test")

        assert "## Test" in result

    def test_create_comparison_table_numeric_change(self):
        """Test comparison table calculates numeric changes."""
        before = {"value": 10}
        after = {"value": 15}
        result = OutputFormatter.create_comparison_table(before, after)

        assert "+5" in result

    def test_create_comparison_table_negative_change(self):
        """Test comparison table with negative change."""
        before = {"value": 20}
        after = {"value": 15}
        result = OutputFormatter.create_comparison_table(before, after)

        assert "-5" in result

    def test_create_comparison_table_non_numeric(self):
        """Test comparison table with non-numeric values."""
        before = {"status": "active"}
        after = {"status": "inactive"}
        result = OutputFormatter.create_comparison_table(before, after)

        assert "active" in result
        assert "inactive" in result
        assert "| - |" in result or "| -  |" in result  # No change calculated

    def test_create_comparison_table_missing_keys(self):
        """Test comparison table with missing keys."""
        before = {"a": 10}
        after = {"b": 20}
        result = OutputFormatter.create_comparison_table(before, after)

        assert "N/A" in result


class TestIntegrationScenarios:
    """Test integration scenarios with multiple formatting operations."""

    def test_format_analysis_results(self):
        """Test formatting complete analysis results."""
        # Simulate complete analysis results
        data = {
            "summary": {
                "total_records": 1000,
                "missing_rate": 0.05,
                "duplicate_count": 10
            },
            "statistics": {
                "mean": 50.5,
                "median": 48.0,
                "std": 15.2
            }
        }

        # Test multiple formats
        json_result = OutputFormatter.to_json(data)
        md_result = OutputFormatter.to_markdown(data, title="Analysis Report")

        assert isinstance(json_result, str)
        assert isinstance(md_result, str)
        assert "# Analysis Report" in md_result
        assert "Summary" in md_result

    def test_format_dataframe_multiple_formats(self):
        """Test formatting DataFrame to multiple formats."""
        df = pd.DataFrame({
            "product": ["A", "B", "C"],
            "sales": [1000, 1500, 800],
            "profit": [200, 300, 150]
        })

        # Test multiple export formats
        html_result = OutputFormatter.to_html_table(df, title="Sales Report")
        csv_result = OutputFormatter.to_csv(df)
        md_data = {"sales_data": df}
        md_result = OutputFormatter.to_markdown(md_data)

        assert "<h3>Sales Report</h3>" in html_result
        assert "product,sales,profit" in csv_result
        assert isinstance(md_result, str)

    def test_comprehensive_number_formatting(self):
        """Test comprehensive number formatting scenario."""
        value = 1234567.89

        # Test multiple number formats
        basic = OutputFormatter.format_number(value)
        large = OutputFormatter.format_large_number(value)
        currency = OutputFormatter.format_currency(value)

        assert basic == "1,234,567.89"
        assert large == "1.2M"
        assert currency == "$1,234,567.89"
