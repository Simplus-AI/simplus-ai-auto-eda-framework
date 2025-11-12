"""
Data validation utilities.

Provides comprehensive validation for DataFrames including:
- Structure validation (shape, columns, types)
- Data quality validation (missing values, duplicates, outliers)
- Value range validation (numeric bounds, string patterns)
- Schema validation (required columns, data types, constraints)
- Business rule validation (custom validation functions)
"""

import pandas as pd
from typing import List, Optional, Tuple, Dict, Any, Callable


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class DataValidator:
    """
    Comprehensive data validation for DataFrames.

    Provides validation methods for:
    - DataFrame structure and integrity
    - Column existence and types
    - Data quality checks
    - Value range and pattern validation
    - Schema validation with constraints
    - Custom business rule validation
    """

    @staticmethod
    def validate_dataframe(data: pd.DataFrame,
                          min_rows: Optional[int] = None,
                          min_cols: Optional[int] = None,
                          max_missing_rate: Optional[float] = None) -> Tuple[bool, List[str]]:
        """
        Validate basic DataFrame requirements.

        Args:
            data: DataFrame to validate
            min_rows: Minimum number of rows required
            min_cols: Minimum number of columns required
            max_missing_rate: Maximum allowed missing value rate (0-1)

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check if DataFrame exists
        if not isinstance(data, pd.DataFrame):
            errors.append("Input is not a pandas DataFrame")
            return False, errors

        # Check empty
        if data.empty:
            errors.append("DataFrame is empty")

        # Check rows
        if data.shape[0] == 0:
            errors.append("DataFrame has no rows")
        elif min_rows is not None and data.shape[0] < min_rows:
            errors.append(f"DataFrame has {data.shape[0]} rows, minimum required is {min_rows}")

        # Check columns
        if data.shape[1] == 0:
            errors.append("DataFrame has no columns")
        elif min_cols is not None and data.shape[1] < min_cols:
            errors.append(f"DataFrame has {data.shape[1]} columns, minimum required is {min_cols}")

        # Check missing values
        if max_missing_rate is not None and not data.empty:
            total_cells = data.shape[0] * data.shape[1]
            missing_cells = data.isna().sum().sum()
            missing_rate = missing_cells / total_cells

            if missing_rate > max_missing_rate:
                errors.append(
                    f"Missing value rate {missing_rate:.2%} exceeds maximum allowed {max_missing_rate:.2%}"
                )

        return len(errors) == 0, errors

    @staticmethod
    def check_column_exists(data: pd.DataFrame, columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if specified columns exist in DataFrame.

        Args:
            data: DataFrame to check
            columns: List of column names to verify

        Returns:
            Tuple of (all_exist, list of missing columns)
        """
        missing = [col for col in columns if col not in data.columns]
        return len(missing) == 0, missing

    @staticmethod
    def validate_numeric_columns(data: pd.DataFrame,
                                 columns: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Validate that specified columns are numeric.

        Args:
            data: DataFrame to check
            columns: List of columns to verify, or None for all

        Returns:
            Tuple of (all_numeric, list of non-numeric columns)
        """
        if columns is None:
            columns = data.columns.tolist()

        non_numeric = [col for col in columns if col in data.columns and
                      not pd.api.types.is_numeric_dtype(data[col])]
        return len(non_numeric) == 0, non_numeric

    @staticmethod
    def validate_categorical_columns(data: pd.DataFrame,
                                    columns: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Validate that specified columns are categorical/object type.

        Args:
            data: DataFrame to check
            columns: List of columns to verify, or None for all object columns

        Returns:
            Tuple of (all_categorical, list of non-categorical columns)
        """
        if columns is None:
            columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

        non_categorical = [
            col for col in columns if col in data.columns and
            not (pd.api.types.is_object_dtype(data[col]) or
                 pd.api.types.is_categorical_dtype(data[col]))
        ]
        return len(non_categorical) == 0, non_categorical

    @staticmethod
    def validate_datetime_columns(data: pd.DataFrame,
                                  columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that specified columns are datetime type.

        Args:
            data: DataFrame to check
            columns: List of columns to verify

        Returns:
            Tuple of (all_datetime, list of non-datetime columns)
        """
        non_datetime = [
            col for col in columns if col in data.columns and
            not pd.api.types.is_datetime64_any_dtype(data[col])
        ]
        return len(non_datetime) == 0, non_datetime

    @staticmethod
    def validate_value_range(data: pd.DataFrame,
                            column: str,
                            min_value: Optional[float] = None,
                            max_value: Optional[float] = None,
                            allow_null: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate that column values are within specified range.

        Args:
            data: DataFrame to check
            column: Column name to validate
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
            allow_null: Whether null values are allowed

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        if column not in data.columns:
            errors.append(f"Column '{column}' does not exist")
            return False, errors

        col_data = data[column]

        # Check null values
        if not allow_null and col_data.isna().any():
            null_count = col_data.isna().sum()
            errors.append(f"Column '{column}' contains {null_count} null values (not allowed)")

        # Check numeric type
        if not pd.api.types.is_numeric_dtype(col_data):
            errors.append(f"Column '{column}' is not numeric")
            return False, errors

        # Check range
        non_null_data = col_data.dropna()

        if min_value is not None and len(non_null_data) > 0:
            below_min = (non_null_data < min_value).sum()
            if below_min > 0:
                errors.append(
                    f"Column '{column}' has {below_min} values below minimum {min_value}"
                )

        if max_value is not None and len(non_null_data) > 0:
            above_max = (non_null_data > max_value).sum()
            if above_max > 0:
                errors.append(
                    f"Column '{column}' has {above_max} values above maximum {max_value}"
                )

        return len(errors) == 0, errors

    @staticmethod
    def validate_no_duplicates(data: pd.DataFrame,
                              subset: Optional[List[str]] = None,
                              keep: str = 'first') -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame has no duplicate rows.

        Args:
            data: DataFrame to check
            subset: Columns to check for duplicates (None means all columns)
            keep: Which duplicates to mark as valid ('first', 'last', False)

        Returns:
            Tuple of (no_duplicates, list of error messages)
        """
        errors = []

        duplicate_count = data.duplicated(subset=subset, keep=keep).sum()

        if duplicate_count > 0:
            errors.append(
                f"Found {duplicate_count} duplicate rows" +
                (f" based on columns {subset}" if subset else "")
            )

        return len(errors) == 0, errors

    @staticmethod
    def validate_no_missing(data: pd.DataFrame,
                           columns: Optional[List[str]] = None) -> Tuple[bool, Dict[str, int]]:
        """
        Validate that specified columns have no missing values.

        Args:
            data: DataFrame to check
            columns: Columns to check (None means all columns)

        Returns:
            Tuple of (no_missing, dict of column: missing_count)
        """
        if columns is None:
            columns = data.columns.tolist()

        missing_info = {}
        for col in columns:
            if col in data.columns:
                missing_count = data[col].isna().sum()
                if missing_count > 0:
                    missing_info[col] = missing_count

        return len(missing_info) == 0, missing_info

    @staticmethod
    def validate_unique_values(data: pd.DataFrame,
                              column: str) -> Tuple[bool, List[str]]:
        """
        Validate that column has all unique values.

        Args:
            data: DataFrame to check
            column: Column name to validate

        Returns:
            Tuple of (all_unique, list of error messages)
        """
        errors = []

        if column not in data.columns:
            errors.append(f"Column '{column}' does not exist")
            return False, errors

        duplicate_count = data[column].duplicated().sum()

        if duplicate_count > 0:
            errors.append(f"Column '{column}' has {duplicate_count} duplicate values")

        return len(errors) == 0, errors

    @staticmethod
    def validate_allowed_values(data: pd.DataFrame,
                               column: str,
                               allowed_values: List[Any],
                               allow_null: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate that column values are from allowed set.

        Args:
            data: DataFrame to check
            column: Column name to validate
            allowed_values: List of allowed values
            allow_null: Whether null values are allowed

        Returns:
            Tuple of (all_allowed, list of error messages)
        """
        errors = []

        if column not in data.columns:
            errors.append(f"Column '{column}' does not exist")
            return False, errors

        col_data = data[column]

        # Check null values
        if not allow_null and col_data.isna().any():
            null_count = col_data.isna().sum()
            errors.append(f"Column '{column}' contains {null_count} null values (not allowed)")

        # Check allowed values
        non_null_data = col_data.dropna()
        invalid_mask = ~non_null_data.isin(allowed_values)
        invalid_count = invalid_mask.sum()

        if invalid_count > 0:
            invalid_values = non_null_data[invalid_mask].unique()[:10]  # Show first 10
            errors.append(
                f"Column '{column}' has {invalid_count} invalid values. "
                f"Examples: {list(invalid_values)}"
            )

        return len(errors) == 0, errors

    @staticmethod
    def validate_string_pattern(data: pd.DataFrame,
                               column: str,
                               pattern: str,
                               allow_null: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate that string column values match regex pattern.

        Args:
            data: DataFrame to check
            column: Column name to validate
            pattern: Regex pattern to match
            allow_null: Whether null values are allowed

        Returns:
            Tuple of (all_match, list of error messages)
        """
        errors = []

        if column not in data.columns:
            errors.append(f"Column '{column}' does not exist")
            return False, errors

        col_data = data[column]

        # Check null values
        if not allow_null and col_data.isna().any():
            null_count = col_data.isna().sum()
            errors.append(f"Column '{column}' contains {null_count} null values (not allowed)")

        # Check pattern
        non_null_data = col_data.dropna().astype(str)

        if len(non_null_data) > 0:
            matches = non_null_data.str.match(pattern, na=False)
            non_matching_count = (~matches).sum()

            if non_matching_count > 0:
                non_matching_examples = non_null_data[~matches].head(5).tolist()
                errors.append(
                    f"Column '{column}' has {non_matching_count} values not matching pattern '{pattern}'. "
                    f"Examples: {non_matching_examples}"
                )

        return len(errors) == 0, errors

    @staticmethod
    def validate_string_length(data: pd.DataFrame,
                              column: str,
                              min_length: Optional[int] = None,
                              max_length: Optional[int] = None,
                              allow_null: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate string column length constraints.

        Args:
            data: DataFrame to check
            column: Column name to validate
            min_length: Minimum string length
            max_length: Maximum string length
            allow_null: Whether null values are allowed

        Returns:
            Tuple of (valid_lengths, list of error messages)
        """
        errors = []

        if column not in data.columns:
            errors.append(f"Column '{column}' does not exist")
            return False, errors

        col_data = data[column]

        # Check null values
        if not allow_null and col_data.isna().any():
            null_count = col_data.isna().sum()
            errors.append(f"Column '{column}' contains {null_count} null values (not allowed)")

        # Check lengths
        non_null_data = col_data.dropna().astype(str)
        lengths = non_null_data.str.len()

        if min_length is not None and len(lengths) > 0:
            too_short = (lengths < min_length).sum()
            if too_short > 0:
                errors.append(
                    f"Column '{column}' has {too_short} values shorter than {min_length} characters"
                )

        if max_length is not None and len(lengths) > 0:
            too_long = (lengths > max_length).sum()
            if too_long > 0:
                errors.append(
                    f"Column '{column}' has {too_long} values longer than {max_length} characters"
                )

        return len(errors) == 0, errors

    @staticmethod
    def validate_schema(data: pd.DataFrame,
                       schema: Dict[str, Dict[str, Any]],
                       strict: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame against a schema definition.

        Args:
            data: DataFrame to validate
            schema: Schema definition dictionary. Format:
                {
                    'column_name': {
                        'type': 'numeric'|'string'|'datetime'|'categorical',
                        'required': True|False,
                        'nullable': True|False,
                        'min_value': <number>,
                        'max_value': <number>,
                        'allowed_values': [<list>],
                        'pattern': '<regex>',
                        'min_length': <int>,
                        'max_length': <int>
                    }
                }
            strict: If True, no extra columns allowed in DataFrame

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check for required columns
        required_cols = [col for col, spec in schema.items() if spec.get('required', True)]
        is_valid, missing = DataValidator.check_column_exists(data, required_cols)
        if not is_valid:
            errors.append(f"Missing required columns: {missing}")

        # Check for extra columns in strict mode
        if strict:
            extra_cols = [col for col in data.columns if col not in schema]
            if extra_cols:
                errors.append(f"Extra columns not in schema: {extra_cols}")

        # Validate each column in schema
        for col, spec in schema.items():
            if col not in data.columns:
                if spec.get('required', True):
                    continue  # Already reported in missing columns
                else:
                    continue  # Optional column, skip

            # Type validation
            col_type = spec.get('type')
            if col_type == 'numeric':
                is_valid, type_errors = DataValidator.validate_numeric_columns(data, [col])
                if not is_valid:
                    errors.extend([f"Column '{col}': {err}" for err in type_errors])

            elif col_type == 'datetime':
                is_valid, type_errors = DataValidator.validate_datetime_columns(data, [col])
                if not is_valid:
                    errors.append(f"Column '{col}' must be datetime type")

            elif col_type in ['string', 'categorical']:
                if not (pd.api.types.is_object_dtype(data[col]) or
                       pd.api.types.is_categorical_dtype(data[col])):
                    errors.append(f"Column '{col}' must be string/categorical type")

            # Nullable validation
            if not spec.get('nullable', True):
                is_valid, missing_info = DataValidator.validate_no_missing(data, [col])
                if not is_valid:
                    errors.append(f"Column '{col}' has {missing_info[col]} null values (not allowed)")

            # Value range validation (numeric)
            if 'min_value' in spec or 'max_value' in spec:
                is_valid, range_errors = DataValidator.validate_value_range(
                    data, col,
                    min_value=spec.get('min_value'),
                    max_value=spec.get('max_value'),
                    allow_null=spec.get('nullable', True)
                )
                if not is_valid:
                    errors.extend(range_errors)

            # Allowed values validation
            if 'allowed_values' in spec:
                is_valid, allowed_errors = DataValidator.validate_allowed_values(
                    data, col,
                    allowed_values=spec['allowed_values'],
                    allow_null=spec.get('nullable', True)
                )
                if not is_valid:
                    errors.extend(allowed_errors)

            # Pattern validation (strings)
            if 'pattern' in spec:
                is_valid, pattern_errors = DataValidator.validate_string_pattern(
                    data, col,
                    pattern=spec['pattern'],
                    allow_null=spec.get('nullable', True)
                )
                if not is_valid:
                    errors.extend(pattern_errors)

            # String length validation
            if 'min_length' in spec or 'max_length' in spec:
                is_valid, length_errors = DataValidator.validate_string_length(
                    data, col,
                    min_length=spec.get('min_length'),
                    max_length=spec.get('max_length'),
                    allow_null=spec.get('nullable', True)
                )
                if not is_valid:
                    errors.extend(length_errors)

        return len(errors) == 0, errors

    @staticmethod
    def validate_custom(data: pd.DataFrame,
                       validation_func: Callable[[pd.DataFrame], Tuple[bool, List[str]]],
                       description: str = "Custom validation") -> Tuple[bool, List[str]]:
        """
        Apply custom validation function to DataFrame.

        Args:
            data: DataFrame to validate
            validation_func: Function that takes DataFrame and returns (is_valid, errors)
            description: Description of the validation

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        try:
            is_valid, errors = validation_func(data)
            return is_valid, errors
        except Exception as e:
            return False, [f"{description} failed with error: {str(e)}"]

    @staticmethod
    def validate_all(data: pd.DataFrame,
                    validations: List[Tuple[Callable, Dict[str, Any], str]]) -> Dict[str, Any]:
        """
        Run multiple validations and return comprehensive report.

        Args:
            data: DataFrame to validate
            validations: List of (validation_function, kwargs, description) tuples

        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'validations': [],
            'total_errors': 0,
            'error_messages': []
        }

        for i, (func, kwargs, description) in enumerate(validations):
            try:
                is_valid, errors = func(data, **kwargs)

                validation_result = {
                    'index': i,
                    'description': description,
                    'function': func.__name__,
                    'is_valid': is_valid,
                    'errors': errors
                }

                results['validations'].append(validation_result)

                if not is_valid:
                    results['is_valid'] = False
                    results['total_errors'] += len(errors)
                    results['error_messages'].extend([f"{description}: {err}" for err in errors])

            except Exception as e:
                results['is_valid'] = False
                results['total_errors'] += 1
                error_msg = f"{description} raised exception: {str(e)}"
                results['error_messages'].append(error_msg)
                results['validations'].append({
                    'index': i,
                    'description': description,
                    'function': func.__name__,
                    'is_valid': False,
                    'errors': [error_msg]
                })

        return results
