"""
Data loading utilities for various file formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import chardet
import warnings


class DataLoader:
    """
    Load data from various file formats with robust error handling and validation.

    Supports the following formats:
    - CSV (.csv, .tsv, .txt)
    - Excel (.xlsx, .xls)
    - JSON (.json)
    - Parquet (.parquet)
    - Feather (.feather)
    - HDF5 (.h5, .hdf5)
    - Pickle (.pkl, .pickle)
    - SQL databases

    Features:
    - Automatic file format detection
    - Encoding detection for text files
    - File validation and existence checks
    - Data preview and information
    - URL support for remote files
    - Comprehensive error handling
    """

    SUPPORTED_FORMATS = {
        '.csv': 'CSV',
        '.tsv': 'TSV',
        '.txt': 'Text',
        '.xlsx': 'Excel',
        '.xls': 'Excel',
        '.json': 'JSON',
        '.parquet': 'Parquet',
        '.feather': 'Feather',
        '.h5': 'HDF5',
        '.hdf5': 'HDF5',
        '.pkl': 'Pickle',
        '.pickle': 'Pickle'
    }

    @staticmethod
    def load(file_path: Union[str, Path],
             validate: bool = True,
             encoding: Optional[str] = None,
             detect_encoding: bool = True,
             **kwargs) -> pd.DataFrame:
        """
        Load data from file with automatic format detection.

        Args:
            file_path: Path to the data file or URL
            validate: Whether to validate the file before loading
            encoding: Encoding to use for text files (auto-detected if None)
            detect_encoding: Whether to auto-detect encoding for text files
            **kwargs: Additional arguments passed to pandas read functions

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is not supported or file is invalid
            Exception: For other loading errors
        """
        # Handle string or Path
        if isinstance(file_path, str):
            # Check if it's a URL
            if file_path.startswith(('http://', 'https://', 'ftp://')):
                return DataLoader._load_from_url(file_path, **kwargs)
            file_path = Path(file_path)

        # Validate file
        if validate:
            DataLoader._validate_file(file_path)

        suffix = file_path.suffix.lower()

        # Check if format is supported
        if suffix not in DataLoader.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: {', '.join(DataLoader.SUPPORTED_FORMATS.keys())}"
            )

        try:
            # Load based on format
            if suffix in ['.csv', '.tsv', '.txt']:
                return DataLoader._load_csv_like(file_path, encoding, detect_encoding, **kwargs)
            elif suffix in ['.xlsx', '.xls']:
                return DataLoader._load_excel(file_path, **kwargs)
            elif suffix == '.json':
                return DataLoader._load_json(file_path, encoding, **kwargs)
            elif suffix == '.parquet':
                return pd.read_parquet(file_path, **kwargs)
            elif suffix == '.feather':
                return pd.read_feather(file_path, **kwargs)
            elif suffix in ['.h5', '.hdf5']:
                return DataLoader._load_hdf5(file_path, **kwargs)
            elif suffix in ['.pkl', '.pickle']:
                return pd.read_pickle(file_path, **kwargs)

        except Exception as e:
            raise Exception(f"Error loading file {file_path}: {str(e)}") from e

    @staticmethod
    def _validate_file(file_path: Path) -> None:
        """
        Validate that file exists and is readable.

        Args:
            file_path: Path to validate

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If path is a directory or file is empty
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.is_dir():
            raise ValueError(f"Path is a directory, not a file: {file_path}")

        if file_path.stat().st_size == 0:
            raise ValueError(f"File is empty: {file_path}")

    @staticmethod
    def _detect_encoding(file_path: Path, sample_size: int = 10000) -> str:
        """
        Detect file encoding using chardet.

        Args:
            file_path: Path to the file
            sample_size: Number of bytes to sample for detection

        Returns:
            Detected encoding name
        """
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(sample_size)
                result = chardet.detect(sample)
                encoding = result['encoding']
                confidence = result['confidence']

                if confidence < 0.7:
                    warnings.warn(
                        f"Low confidence ({confidence:.2f}) in encoding detection. "
                        f"Detected: {encoding}. Consider specifying encoding explicitly."
                    )

                return encoding if encoding else 'utf-8'
        except Exception:
            return 'utf-8'

    @staticmethod
    def _load_csv_like(file_path: Path,
                       encoding: Optional[str] = None,
                       detect_encoding: bool = True,
                       **kwargs) -> pd.DataFrame:
        """
        Load CSV-like files (CSV, TSV, TXT) with encoding detection.

        Args:
            file_path: Path to the file
            encoding: Encoding to use (auto-detected if None)
            detect_encoding: Whether to auto-detect encoding
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            Loaded DataFrame
        """
        # Detect encoding if not provided
        if encoding is None and detect_encoding:
            encoding = DataLoader._detect_encoding(file_path)

        # Set delimiter based on file extension
        if 'sep' not in kwargs and 'delimiter' not in kwargs:
            if file_path.suffix.lower() == '.tsv':
                kwargs['sep'] = '\t'

        return pd.read_csv(file_path, encoding=encoding, **kwargs)

    @staticmethod
    def _load_excel(file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Load Excel file with enhanced error handling.

        Args:
            file_path: Path to Excel file
            **kwargs: Additional arguments for pd.read_excel

        Returns:
            Loaded DataFrame
        """
        try:
            return pd.read_excel(file_path, **kwargs)
        except Exception as e:
            # Try to provide helpful error messages
            if 'sheet_name' in kwargs:
                raise Exception(
                    f"Error loading sheet '{kwargs['sheet_name']}' from {file_path}. "
                    f"Use get_excel_sheets() to see available sheets."
                ) from e
            raise

    @staticmethod
    def _load_json(file_path: Path, encoding: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load JSON file with encoding support.

        Args:
            file_path: Path to JSON file
            encoding: Encoding to use
            **kwargs: Additional arguments for pd.read_json

        Returns:
            Loaded DataFrame
        """
        if encoding:
            kwargs['encoding'] = encoding
        return pd.read_json(file_path, **kwargs)

    @staticmethod
    def _load_hdf5(file_path: Path, key: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load HDF5 file.

        Args:
            file_path: Path to HDF5 file
            key: Key to load (required for HDF5)
            **kwargs: Additional arguments for pd.read_hdf

        Returns:
            Loaded DataFrame

        Raises:
            ValueError: If key is not provided
        """
        if key is None and 'key' not in kwargs:
            raise ValueError("HDF5 files require a 'key' parameter. Use get_hdf5_keys() to see available keys.")

        if key:
            kwargs['key'] = key

        return pd.read_hdf(file_path, **kwargs)

    @staticmethod
    def _load_from_url(url: str, **kwargs) -> pd.DataFrame:
        """
        Load data from URL.

        Args:
            url: URL to load from
            **kwargs: Additional arguments for pandas read functions

        Returns:
            Loaded DataFrame
        """
        # Detect format from URL
        url_lower = url.lower()

        if '.csv' in url_lower or 'format=csv' in url_lower:
            return pd.read_csv(url, **kwargs)
        elif '.json' in url_lower or 'format=json' in url_lower:
            return pd.read_json(url, **kwargs)
        elif '.xlsx' in url_lower or '.xls' in url_lower:
            return pd.read_excel(url, **kwargs)
        elif '.parquet' in url_lower:
            return pd.read_parquet(url, **kwargs)
        else:
            # Default to CSV
            warnings.warn("Could not detect format from URL, trying CSV format")
            return pd.read_csv(url, **kwargs)

    @staticmethod
    def preview(file_path: Union[str, Path], n_rows: int = 5, **kwargs) -> pd.DataFrame:
        """
        Preview first n rows of a file without loading the entire dataset.

        Args:
            file_path: Path to the data file
            n_rows: Number of rows to preview
            **kwargs: Additional arguments for loading

        Returns:
            DataFrame with first n rows
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        suffix = file_path.suffix.lower()

        if suffix in ['.csv', '.tsv', '.txt']:
            kwargs['nrows'] = n_rows
            return DataLoader.load(file_path, **kwargs)
        else:
            # Load full file and return head
            df = DataLoader.load(file_path, **kwargs)
            return df.head(n_rows)

    @staticmethod
    def get_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a data file without loading it.

        Args:
            file_path: Path to the data file

        Returns:
            Dictionary containing file information
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        info = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size_bytes': file_path.stat().st_size,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'format': DataLoader.SUPPORTED_FORMATS.get(file_path.suffix.lower(), 'Unknown'),
            'extension': file_path.suffix.lower(),
        }

        # Try to get row count for CSV files
        if file_path.suffix.lower() in ['.csv', '.tsv', '.txt']:
            try:
                with open(file_path, 'rb') as f:
                    row_count = sum(1 for _ in f) - 1  # Subtract header
                info['estimated_rows'] = row_count
            except:
                pass

        return info

    @staticmethod
    def get_excel_sheets(file_path: Union[str, Path]) -> List[str]:
        """
        Get list of sheet names in an Excel file.

        Args:
            file_path: Path to Excel file

        Returns:
            List of sheet names
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        excel_file = pd.ExcelFile(file_path)
        return excel_file.sheet_names

    @staticmethod
    def get_hdf5_keys(file_path: Union[str, Path]) -> List[str]:
        """
        Get list of keys in an HDF5 file.

        Args:
            file_path: Path to HDF5 file

        Returns:
            List of keys
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        with pd.HDFStore(file_path, 'r') as store:
            return list(store.keys())

    @staticmethod
    def load_multiple(file_paths: List[Union[str, Path]],
                     concat: bool = True,
                     **kwargs) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Load multiple files at once.

        Args:
            file_paths: List of file paths to load
            concat: Whether to concatenate into single DataFrame
            **kwargs: Additional arguments passed to load()

        Returns:
            Single concatenated DataFrame if concat=True, otherwise list of DataFrames
        """
        dfs = [DataLoader.load(fp, **kwargs) for fp in file_paths]

        if concat:
            return pd.concat(dfs, ignore_index=True)
        return dfs

    # Convenience methods for specific formats
    @staticmethod
    def load_csv(file_path: Union[str, Path],
                 encoding: Optional[str] = None,
                 **kwargs) -> pd.DataFrame:
        """Load CSV file with encoding detection."""
        return DataLoader.load(file_path, encoding=encoding, **kwargs)

    @staticmethod
    def load_excel(file_path: Union[str, Path],
                   sheet_name: Optional[Union[str, int]] = 0,
                   **kwargs) -> pd.DataFrame:
        """Load Excel file with specific sheet."""
        return DataLoader.load(file_path, sheet_name=sheet_name, **kwargs)

    @staticmethod
    def load_json(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load JSON file."""
        return DataLoader.load(file_path, **kwargs)

    @staticmethod
    def load_parquet(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load Parquet file."""
        return DataLoader.load(file_path, **kwargs)

    @staticmethod
    def load_feather(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load Feather file."""
        return DataLoader.load(file_path, **kwargs)

    @staticmethod
    def load_hdf5(file_path: Union[str, Path], key: str, **kwargs) -> pd.DataFrame:
        """Load HDF5 file with specific key."""
        return DataLoader.load(file_path, key=key, **kwargs)

    @staticmethod
    def load_pickle(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load Pickle file."""
        return DataLoader.load(file_path, **kwargs)
