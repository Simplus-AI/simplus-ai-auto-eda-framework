"""
Tests for data loader utilities.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
from simplus_eda.utils.data_loader import DataLoader


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10.5, 20.3, 30.1, 40.7, 50.2],
        'C': ['a', 'b', 'c', 'd', 'e'],
        'D': pd.date_range('2024-01-01', periods=5)
    })


# ============================================================================
# CSV Loading Tests
# ============================================================================

def test_load_csv_basic(temp_dir, sample_df):
    """Test basic CSV loading."""
    csv_path = temp_dir / "test.csv"
    sample_df.to_csv(csv_path, index=False)

    loaded = DataLoader.load(csv_path)

    assert len(loaded) == 5
    assert list(loaded.columns) == ['A', 'B', 'C', 'D']


def test_load_csv_with_encoding(temp_dir):
    """Test CSV loading with specific encoding."""
    csv_path = temp_dir / "test_utf8.csv"

    # Create CSV with UTF-8 content
    df = pd.DataFrame({
        'name': ['José', 'François', '北京'],
        'value': [1, 2, 3]
    })
    df.to_csv(csv_path, index=False, encoding='utf-8')

    loaded = DataLoader.load(csv_path, encoding='utf-8')

    assert len(loaded) == 3
    assert loaded['name'].tolist() == ['José', 'François', '北京']


def test_load_csv_with_auto_encoding(temp_dir):
    """Test CSV loading with automatic encoding detection."""
    csv_path = temp_dir / "test_auto.csv"

    df = pd.DataFrame({
        'name': ['José', 'François'],
        'value': [1, 2]
    })
    df.to_csv(csv_path, index=False, encoding='utf-8')

    # Should auto-detect UTF-8
    loaded = DataLoader.load(csv_path, detect_encoding=True)

    assert len(loaded) == 2


def test_load_tsv(temp_dir, sample_df):
    """Test TSV loading."""
    tsv_path = temp_dir / "test.tsv"
    sample_df.to_csv(tsv_path, index=False, sep='\t')

    loaded = DataLoader.load(tsv_path)

    assert len(loaded) == 5
    assert list(loaded.columns) == ['A', 'B', 'C', 'D']


def test_load_csv_with_kwargs(temp_dir):
    """Test CSV loading with additional pandas kwargs."""
    csv_path = temp_dir / "test_kwargs.csv"

    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    df.to_csv(csv_path, index=False)

    # Load only specific columns
    loaded = DataLoader.load(csv_path, usecols=['A'])

    assert list(loaded.columns) == ['A']


# ============================================================================
# Excel Loading Tests
# ============================================================================

def test_load_excel_basic(temp_dir, sample_df):
    """Test basic Excel loading."""
    excel_path = temp_dir / "test.xlsx"
    sample_df.to_excel(excel_path, index=False)

    loaded = DataLoader.load(excel_path)

    assert len(loaded) == 5
    assert list(loaded.columns) == ['A', 'B', 'C', 'D']


def test_load_excel_specific_sheet(temp_dir, sample_df):
    """Test loading specific Excel sheet."""
    excel_path = temp_dir / "test_sheets.xlsx"

    with pd.ExcelWriter(excel_path) as writer:
        sample_df.to_excel(writer, sheet_name='Sheet1', index=False)
        sample_df.to_excel(writer, sheet_name='Sheet2', index=False)

    loaded = DataLoader.load_excel(excel_path, sheet_name='Sheet2')

    assert len(loaded) == 5


def test_get_excel_sheets(temp_dir, sample_df):
    """Test getting Excel sheet names."""
    excel_path = temp_dir / "test_sheets.xlsx"

    with pd.ExcelWriter(excel_path) as writer:
        sample_df.to_excel(writer, sheet_name='Data', index=False)
        sample_df.to_excel(writer, sheet_name='Summary', index=False)

    sheets = DataLoader.get_excel_sheets(excel_path)

    assert sheets == ['Data', 'Summary']


# ============================================================================
# JSON Loading Tests
# ============================================================================

def test_load_json_basic(temp_dir, sample_df):
    """Test basic JSON loading."""
    json_path = temp_dir / "test.json"
    sample_df.to_json(json_path, orient='records')

    loaded = DataLoader.load(json_path)

    assert len(loaded) == 5


def test_load_json_with_orient(temp_dir):
    """Test JSON loading with specific orientation."""
    json_path = temp_dir / "test_orient.json"

    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    df.to_json(json_path, orient='split')

    loaded = DataLoader.load(json_path, orient='split')

    assert len(loaded) == 3


# ============================================================================
# Parquet Loading Tests
# ============================================================================

def test_load_parquet_basic(temp_dir, sample_df):
    """Test basic Parquet loading."""
    parquet_path = temp_dir / "test.parquet"
    sample_df.to_parquet(parquet_path, index=False)

    loaded = DataLoader.load(parquet_path)

    assert len(loaded) == 5
    assert list(loaded.columns) == ['A', 'B', 'C', 'D']


def test_load_parquet_convenience_method(temp_dir, sample_df):
    """Test Parquet loading via convenience method."""
    parquet_path = temp_dir / "test.parquet"
    sample_df.to_parquet(parquet_path, index=False)

    loaded = DataLoader.load_parquet(parquet_path)

    assert len(loaded) == 5


# ============================================================================
# Feather Loading Tests
# ============================================================================

def test_load_feather_basic(temp_dir, sample_df):
    """Test basic Feather loading."""
    feather_path = temp_dir / "test.feather"
    sample_df.to_feather(feather_path)

    loaded = DataLoader.load(feather_path)

    assert len(loaded) == 5


# ============================================================================
# HDF5 Loading Tests
# ============================================================================

def test_load_hdf5_basic(temp_dir, sample_df):
    """Test basic HDF5 loading."""
    hdf_path = temp_dir / "test.h5"
    sample_df.to_hdf(hdf_path, key='data', mode='w')

    loaded = DataLoader.load_hdf5(hdf_path, key='data')

    assert len(loaded) == 5


def test_load_hdf5_without_key_raises_error(temp_dir, sample_df):
    """Test that HDF5 loading without key raises error."""
    hdf_path = temp_dir / "test.h5"
    sample_df.to_hdf(hdf_path, key='data', mode='w')

    with pytest.raises(Exception, match="HDF5 files require a 'key' parameter"):
        DataLoader.load(hdf_path)


def test_get_hdf5_keys(temp_dir, sample_df):
    """Test getting HDF5 keys."""
    hdf_path = temp_dir / "test_multi.h5"

    sample_df.to_hdf(hdf_path, key='dataset1', mode='w')
    sample_df.to_hdf(hdf_path, key='dataset2', mode='a')

    keys = DataLoader.get_hdf5_keys(hdf_path)

    assert '/dataset1' in keys
    assert '/dataset2' in keys


# ============================================================================
# Pickle Loading Tests
# ============================================================================

def test_load_pickle_basic(temp_dir, sample_df):
    """Test basic Pickle loading."""
    pickle_path = temp_dir / "test.pkl"
    sample_df.to_pickle(pickle_path)

    loaded = DataLoader.load(pickle_path)

    assert len(loaded) == 5
    pd.testing.assert_frame_equal(loaded, sample_df)


# ============================================================================
# File Validation Tests
# ============================================================================

def test_load_nonexistent_file():
    """Test loading non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        DataLoader.load("nonexistent.csv")


def test_load_directory_raises_error(temp_dir):
    """Test loading directory raises error."""
    with pytest.raises(ValueError, match="Path is a directory"):
        DataLoader.load(temp_dir)


def test_load_empty_file(temp_dir):
    """Test loading empty file raises error."""
    empty_file = temp_dir / "empty.csv"
    empty_file.touch()

    with pytest.raises(ValueError, match="File is empty"):
        DataLoader.load(empty_file)


def test_load_unsupported_format(temp_dir):
    """Test loading unsupported format raises error."""
    unsupported = temp_dir / "test.xyz"
    unsupported.write_text("data")

    with pytest.raises(ValueError, match="Unsupported file format"):
        DataLoader.load(unsupported)


def test_load_with_validation_disabled(temp_dir):
    """Test loading with validation disabled."""
    csv_path = temp_dir / "test.csv"
    pd.DataFrame({'A': [1, 2, 3]}).to_csv(csv_path, index=False)

    # Should work
    loaded = DataLoader.load(csv_path, validate=False)
    assert len(loaded) == 3


# ============================================================================
# Preview and Info Tests
# ============================================================================

def test_preview_csv(temp_dir):
    """Test preview functionality for CSV."""
    csv_path = temp_dir / "test_preview.csv"
    df = pd.DataFrame({'A': range(100), 'B': range(100, 200)})
    df.to_csv(csv_path, index=False)

    preview = DataLoader.preview(csv_path, n_rows=5)

    assert len(preview) == 5
    assert preview['A'].tolist() == [0, 1, 2, 3, 4]


def test_preview_excel(temp_dir):
    """Test preview functionality for Excel."""
    excel_path = temp_dir / "test_preview.xlsx"
    df = pd.DataFrame({'A': range(100), 'B': range(100, 200)})
    df.to_excel(excel_path, index=False)

    preview = DataLoader.preview(excel_path, n_rows=3)

    assert len(preview) == 3


def test_get_info(temp_dir, sample_df):
    """Test getting file information."""
    csv_path = temp_dir / "test_info.csv"
    sample_df.to_csv(csv_path, index=False)

    info = DataLoader.get_info(csv_path)

    assert info['file_name'] == 'test_info.csv'
    assert info['format'] == 'CSV'
    assert info['extension'] == '.csv'
    assert info['file_size_bytes'] > 0
    assert info['file_size_mb'] > 0
    assert 'estimated_rows' in info


def test_get_info_nonexistent_file():
    """Test get_info on non-existent file."""
    with pytest.raises(FileNotFoundError):
        DataLoader.get_info("nonexistent.csv")


# ============================================================================
# Multiple Files Tests
# ============================================================================

def test_load_multiple_concat(temp_dir):
    """Test loading multiple files with concatenation."""
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

    path1 = temp_dir / "file1.csv"
    path2 = temp_dir / "file2.csv"

    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)

    loaded = DataLoader.load_multiple([path1, path2], concat=True)

    assert len(loaded) == 4
    assert loaded['A'].tolist() == [1, 2, 5, 6]


def test_load_multiple_no_concat(temp_dir):
    """Test loading multiple files without concatenation."""
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'A': [3, 4]})

    path1 = temp_dir / "file1.csv"
    path2 = temp_dir / "file2.csv"

    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)

    loaded = DataLoader.load_multiple([path1, path2], concat=False)

    assert len(loaded) == 2
    assert isinstance(loaded, list)
    assert len(loaded[0]) == 2
    assert len(loaded[1]) == 2


# ============================================================================
# Convenience Methods Tests
# ============================================================================

def test_load_csv_convenience(temp_dir, sample_df):
    """Test CSV convenience method."""
    csv_path = temp_dir / "test.csv"
    sample_df.to_csv(csv_path, index=False)

    loaded = DataLoader.load_csv(csv_path)

    assert len(loaded) == 5


def test_load_json_convenience(temp_dir, sample_df):
    """Test JSON convenience method."""
    json_path = temp_dir / "test.json"
    sample_df.to_json(json_path, orient='records')

    loaded = DataLoader.load_json(json_path)

    assert len(loaded) == 5


def test_load_feather_convenience(temp_dir, sample_df):
    """Test Feather convenience method."""
    feather_path = temp_dir / "test.feather"
    sample_df.to_feather(feather_path)

    loaded = DataLoader.load_feather(feather_path)

    assert len(loaded) == 5


def test_load_pickle_convenience(temp_dir, sample_df):
    """Test Pickle convenience method."""
    pickle_path = temp_dir / "test.pkl"
    sample_df.to_pickle(pickle_path)

    loaded = DataLoader.load_pickle(pickle_path)

    assert len(loaded) == 5


# ============================================================================
# Path Type Tests
# ============================================================================

def test_load_with_string_path(temp_dir, sample_df):
    """Test loading with string path."""
    csv_path = temp_dir / "test.csv"
    sample_df.to_csv(csv_path, index=False)

    loaded = DataLoader.load(str(csv_path))

    assert len(loaded) == 5


def test_load_with_path_object(temp_dir, sample_df):
    """Test loading with Path object."""
    csv_path = temp_dir / "test.csv"
    sample_df.to_csv(csv_path, index=False)

    loaded = DataLoader.load(csv_path)

    assert len(loaded) == 5


# ============================================================================
# Edge Cases Tests
# ============================================================================

def test_load_csv_with_missing_values(temp_dir):
    """Test loading CSV with missing values."""
    csv_path = temp_dir / "test_missing.csv"

    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, 2, 3, 4]
    })
    df.to_csv(csv_path, index=False)

    loaded = DataLoader.load(csv_path)

    assert loaded['A'].isna().sum() == 1
    assert loaded['B'].isna().sum() == 1


def test_load_csv_with_special_characters(temp_dir):
    """Test loading CSV with special characters in data."""
    csv_path = temp_dir / "test_special.csv"

    df = pd.DataFrame({
        'text': ['hello,world', 'foo"bar', 'test\nnewline'],
        'value': [1, 2, 3]
    })
    df.to_csv(csv_path, index=False)

    loaded = DataLoader.load(csv_path)

    assert len(loaded) == 3


def test_load_large_csv_preview(temp_dir):
    """Test preview of large CSV file."""
    csv_path = temp_dir / "large.csv"

    # Create a large DataFrame
    df = pd.DataFrame({
        'A': range(10000),
        'B': range(10000, 20000)
    })
    df.to_csv(csv_path, index=False)

    # Preview should only load 10 rows
    preview = DataLoader.preview(csv_path, n_rows=10)

    assert len(preview) == 10


# ============================================================================
# Error Message Tests
# ============================================================================

def test_helpful_error_message_for_excel_sheet(temp_dir, sample_df):
    """Test helpful error message when Excel sheet doesn't exist."""
    excel_path = temp_dir / "test.xlsx"
    sample_df.to_excel(excel_path, index=False, sheet_name='Data')

    with pytest.raises(Exception, match="Use get_excel_sheets"):
        DataLoader.load(excel_path, sheet_name='NonExistent')


def test_supported_formats_in_error_message(temp_dir):
    """Test that unsupported format error includes list of supported formats."""
    bad_file = temp_dir / "test.xyz"
    bad_file.write_text("data")

    with pytest.raises(ValueError) as exc_info:
        DataLoader.load(bad_file)

    assert "Supported formats:" in str(exc_info.value)


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_workflow_csv(temp_dir):
    """Test complete workflow with CSV file."""
    csv_path = temp_dir / "workflow.csv"

    # Create and save data
    original = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, 20.3, 30.1, 40.7, 50.2],
        'category': ['A', 'B', 'A', 'C', 'B']
    })
    original.to_csv(csv_path, index=False)

    # Get info
    info = DataLoader.get_info(csv_path)
    assert info['format'] == 'CSV'

    # Preview
    preview = DataLoader.preview(csv_path, n_rows=2)
    assert len(preview) == 2

    # Load full
    loaded = DataLoader.load(csv_path)
    assert len(loaded) == 5

    # Verify data integrity
    pd.testing.assert_frame_equal(loaded, original)


def test_full_workflow_excel(temp_dir):
    """Test complete workflow with Excel file."""
    excel_path = temp_dir / "workflow.xlsx"

    # Create multi-sheet Excel
    df1 = pd.DataFrame({'A': [1, 2, 3]})
    df2 = pd.DataFrame({'B': [4, 5, 6]})

    with pd.ExcelWriter(excel_path) as writer:
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet2', index=False)

    # Get sheets
    sheets = DataLoader.get_excel_sheets(excel_path)
    assert len(sheets) == 2

    # Load specific sheet
    loaded = DataLoader.load_excel(excel_path, sheet_name='Sheet2')
    assert 'B' in loaded.columns

    # Get info
    info = DataLoader.get_info(excel_path)
    assert info['format'] == 'Excel'


def test_different_formats_same_data(temp_dir):
    """Test that same data can be loaded from different formats."""
    original = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4.5, 5.5, 6.5]
    })

    # Save in different formats
    csv_path = temp_dir / "data.csv"
    json_path = temp_dir / "data.json"
    parquet_path = temp_dir / "data.parquet"

    original.to_csv(csv_path, index=False)
    original.to_json(json_path, orient='records')
    original.to_parquet(parquet_path, index=False)

    # Load from different formats
    from_csv = DataLoader.load(csv_path)
    from_json = DataLoader.load(json_path)
    from_parquet = DataLoader.load(parquet_path)

    # All should have same data
    assert len(from_csv) == len(from_json) == len(from_parquet) == 3
    assert list(from_csv.columns) == list(from_parquet.columns)
