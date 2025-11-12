"""
Tests for data validation utilities.
"""

import pytest
import pandas as pd
import numpy as np
from simplus_eda.utils.validators import DataValidator, ValidationError


@pytest.fixture
def valid_df():
    """Create a valid DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, 45],
        'salary': [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        'department': ['IT', 'HR', 'IT', 'Finance', 'HR']
    })


@pytest.fixture
def df_with_issues():
    """Create DataFrame with various issues."""
    return pd.DataFrame({
        'id': [1, 2, 2, 4, 5],  # Duplicate
        'name': ['Alice', 'Bob', None, 'David', 'Eve'],  # Missing value
        'age': [25, 30, -5, 150, 45],  # Invalid range
        'email': ['a@b.com', 'invalid', 'c@d.com', 'e@f.com', 'g@h.com']  # Invalid pattern
    })


# ============================================================================
# Basic DataFrame Validation Tests
# ============================================================================

def test_validate_dataframe_valid(valid_df):
    """Test validation of valid DataFrame."""
    is_valid, errors = DataValidator.validate_dataframe(valid_df)

    assert is_valid
    assert len(errors) == 0


def test_validate_dataframe_empty():
    """Test validation of empty DataFrame."""
    df = pd.DataFrame()
    is_valid, errors = DataValidator.validate_dataframe(df)

    assert not is_valid
    assert "empty" in errors[0].lower()


def test_validate_dataframe_min_rows():
    """Test minimum row requirement."""
    df = pd.DataFrame({'A': [1, 2]})
    is_valid, errors = DataValidator.validate_dataframe(df, min_rows=5)

    assert not is_valid
    assert "minimum required is 5" in errors[0]


def test_validate_dataframe_min_cols():
    """Test minimum column requirement."""
    df = pd.DataFrame({'A': [1, 2, 3]})
    is_valid, errors = DataValidator.validate_dataframe(df, min_cols=3)

    assert not is_valid
    assert "minimum required is 3" in errors[0]


def test_validate_dataframe_max_missing_rate():
    """Test maximum missing value rate."""
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [np.nan, np.nan, 3],
        'C': [1, 2, 3]
    })
    # Missing rate: 3/9 = 0.333
    is_valid, errors = DataValidator.validate_dataframe(df, max_missing_rate=0.2)

    assert not is_valid
    assert "Missing value rate" in errors[0]


def test_validate_dataframe_not_dataframe():
    """Test validation with non-DataFrame input."""
    is_valid, errors = DataValidator.validate_dataframe([1, 2, 3])

    assert not is_valid
    assert "not a pandas DataFrame" in errors[0]


# ============================================================================
# Column Existence Tests
# ============================================================================

def test_check_column_exists_all_present(valid_df):
    """Test checking existing columns."""
    is_valid, missing = DataValidator.check_column_exists(valid_df, ['id', 'name', 'age'])

    assert is_valid
    assert len(missing) == 0


def test_check_column_exists_some_missing(valid_df):
    """Test checking with some missing columns."""
    is_valid, missing = DataValidator.check_column_exists(valid_df, ['id', 'missing1', 'missing2'])

    assert not is_valid
    assert set(missing) == {'missing1', 'missing2'}


def test_check_column_exists_all_missing(valid_df):
    """Test checking with all missing columns."""
    is_valid, missing = DataValidator.check_column_exists(valid_df, ['x', 'y', 'z'])

    assert not is_valid
    assert len(missing) == 3


# ============================================================================
# Numeric Column Validation Tests
# ============================================================================

def test_validate_numeric_columns_all_numeric():
    """Test validation of numeric columns."""
    df = pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.5, 2.5, 3.5]
    })
    is_valid, non_numeric = DataValidator.validate_numeric_columns(df)

    assert is_valid
    assert len(non_numeric) == 0


def test_validate_numeric_columns_mixed():
    """Test validation with mixed types."""
    df = pd.DataFrame({
        'numeric': [1, 2, 3],
        'string': ['a', 'b', 'c']
    })
    is_valid, non_numeric = DataValidator.validate_numeric_columns(df)

    assert not is_valid
    assert 'string' in non_numeric


def test_validate_numeric_columns_specific():
    """Test validation of specific columns."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c'],
        'C': [4, 5, 6]
    })
    is_valid, non_numeric = DataValidator.validate_numeric_columns(df, ['A', 'C'])

    assert is_valid
    assert len(non_numeric) == 0


# ============================================================================
# Categorical Column Validation Tests
# ============================================================================

def test_validate_categorical_columns():
    """Test validation of categorical columns."""
    df = pd.DataFrame({
        'cat': pd.Categorical(['A', 'B', 'C']),
        'obj': ['X', 'Y', 'Z'],
        'num': [1, 2, 3]
    })
    is_valid, non_cat = DataValidator.validate_categorical_columns(df, ['cat', 'obj'])

    assert is_valid
    assert len(non_cat) == 0


def test_validate_categorical_columns_with_numeric():
    """Test validation with numeric column."""
    df = pd.DataFrame({
        'cat': ['A', 'B', 'C'],
        'num': [1, 2, 3]
    })
    is_valid, non_cat = DataValidator.validate_categorical_columns(df, ['cat', 'num'])

    assert not is_valid
    assert 'num' in non_cat


# ============================================================================
# Datetime Column Validation Tests
# ============================================================================

def test_validate_datetime_columns():
    """Test validation of datetime columns."""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=3),
        'value': [1, 2, 3]
    })
    is_valid, non_datetime = DataValidator.validate_datetime_columns(df, ['date'])

    assert is_valid
    assert len(non_datetime) == 0


def test_validate_datetime_columns_non_datetime():
    """Test validation with non-datetime column."""
    df = pd.DataFrame({
        'date_str': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'value': [1, 2, 3]
    })
    is_valid, non_datetime = DataValidator.validate_datetime_columns(df, ['date_str'])

    assert not is_valid
    assert 'date_str' in non_datetime


# ============================================================================
# Value Range Validation Tests
# ============================================================================

def test_validate_value_range_valid():
    """Test value range validation with valid data."""
    df = pd.DataFrame({'age': [25, 30, 35, 40, 45]})
    is_valid, errors = DataValidator.validate_value_range(df, 'age', min_value=0, max_value=100)

    assert is_valid
    assert len(errors) == 0


def test_validate_value_range_below_min():
    """Test value range with values below minimum."""
    df = pd.DataFrame({'age': [25, 30, -5, 40, 45]})
    is_valid, errors = DataValidator.validate_value_range(df, 'age', min_value=0)

    assert not is_valid
    assert "below minimum" in errors[0]


def test_validate_value_range_above_max():
    """Test value range with values above maximum."""
    df = pd.DataFrame({'age': [25, 30, 150, 40, 45]})
    is_valid, errors = DataValidator.validate_value_range(df, 'age', max_value=100)

    assert not is_valid
    assert "above maximum" in errors[0]


def test_validate_value_range_with_nulls():
    """Test value range with null values."""
    df = pd.DataFrame({'age': [25, 30, np.nan, 40, 45]})
    is_valid, errors = DataValidator.validate_value_range(df, 'age', min_value=0, allow_null=True)

    assert is_valid
    assert len(errors) == 0


def test_validate_value_range_nulls_not_allowed():
    """Test value range with null values not allowed."""
    df = pd.DataFrame({'age': [25, 30, np.nan, 40, 45]})
    is_valid, errors = DataValidator.validate_value_range(df, 'age', allow_null=False)

    assert not is_valid
    assert "null values" in errors[0]


def test_validate_value_range_missing_column():
    """Test value range with missing column."""
    df = pd.DataFrame({'age': [25, 30, 35]})
    is_valid, errors = DataValidator.validate_value_range(df, 'missing_col')

    assert not is_valid
    assert "does not exist" in errors[0]


def test_validate_value_range_non_numeric():
    """Test value range with non-numeric column."""
    df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie']})
    is_valid, errors = DataValidator.validate_value_range(df, 'name', min_value=0)

    assert not is_valid
    assert "not numeric" in errors[0]


# ============================================================================
# Duplicate Validation Tests
# ============================================================================

def test_validate_no_duplicates_valid(valid_df):
    """Test no duplicates validation with unique data."""
    is_valid, errors = DataValidator.validate_no_duplicates(valid_df)

    assert is_valid
    assert len(errors) == 0


def test_validate_no_duplicates_with_duplicates():
    """Test no duplicates validation with duplicates."""
    df = pd.DataFrame({
        'A': [1, 2, 2, 3],
        'B': [4, 5, 5, 6]
    })
    is_valid, errors = DataValidator.validate_no_duplicates(df)

    assert not is_valid
    assert "duplicate rows" in errors[0]


def test_validate_no_duplicates_subset():
    """Test no duplicates validation on subset of columns."""
    df = pd.DataFrame({
        'id': [1, 2, 2, 3],
        'value': [10, 20, 30, 40]
    })
    is_valid, errors = DataValidator.validate_no_duplicates(df, subset=['id'])

    assert not is_valid
    assert "based on columns" in errors[0]


# ============================================================================
# Missing Value Validation Tests
# ============================================================================

def test_validate_no_missing_valid(valid_df):
    """Test no missing validation with complete data."""
    is_valid, missing_info = DataValidator.validate_no_missing(valid_df)

    assert is_valid
    assert len(missing_info) == 0


def test_validate_no_missing_with_missing():
    """Test no missing validation with missing values."""
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, 7, 8]
    })
    is_valid, missing_info = DataValidator.validate_no_missing(df)

    assert not is_valid
    assert 'A' in missing_info
    assert 'B' in missing_info
    assert missing_info['A'] == 1
    assert missing_info['B'] == 1


def test_validate_no_missing_specific_columns():
    """Test no missing validation on specific columns."""
    df = pd.DataFrame({
        'required': [1, 2, 3, 4],
        'optional': [5, np.nan, 7, 8]
    })
    is_valid, missing_info = DataValidator.validate_no_missing(df, ['required'])

    assert is_valid
    assert len(missing_info) == 0


# ============================================================================
# Unique Values Validation Tests
# ============================================================================

def test_validate_unique_values_valid():
    """Test unique values validation with unique data."""
    df = pd.DataFrame({'id': [1, 2, 3, 4, 5]})
    is_valid, errors = DataValidator.validate_unique_values(df, 'id')

    assert is_valid
    assert len(errors) == 0


def test_validate_unique_values_with_duplicates():
    """Test unique values validation with duplicates."""
    df = pd.DataFrame({'id': [1, 2, 2, 3, 3]})
    is_valid, errors = DataValidator.validate_unique_values(df, 'id')

    assert not is_valid
    assert "duplicate values" in errors[0]


def test_validate_unique_values_missing_column():
    """Test unique values validation with missing column."""
    df = pd.DataFrame({'id': [1, 2, 3]})
    is_valid, errors = DataValidator.validate_unique_values(df, 'missing')

    assert not is_valid
    assert "does not exist" in errors[0]


# ============================================================================
# Allowed Values Validation Tests
# ============================================================================

def test_validate_allowed_values_valid():
    """Test allowed values validation with valid data."""
    df = pd.DataFrame({'status': ['active', 'inactive', 'active', 'pending']})
    is_valid, errors = DataValidator.validate_allowed_values(
        df, 'status', ['active', 'inactive', 'pending']
    )

    assert is_valid
    assert len(errors) == 0


def test_validate_allowed_values_invalid():
    """Test allowed values validation with invalid values."""
    df = pd.DataFrame({'status': ['active', 'invalid', 'active', 'pending']})
    is_valid, errors = DataValidator.validate_allowed_values(
        df, 'status', ['active', 'pending']
    )

    assert not is_valid
    assert "invalid values" in errors[0]


def test_validate_allowed_values_with_null():
    """Test allowed values validation with null values."""
    df = pd.DataFrame({'status': ['active', None, 'active', 'pending']})
    is_valid, errors = DataValidator.validate_allowed_values(
        df, 'status', ['active', 'pending'], allow_null=True
    )

    assert is_valid
    assert len(errors) == 0


def test_validate_allowed_values_null_not_allowed():
    """Test allowed values validation with null not allowed."""
    df = pd.DataFrame({'status': ['active', None, 'active', 'pending']})
    is_valid, errors = DataValidator.validate_allowed_values(
        df, 'status', ['active', 'pending'], allow_null=False
    )

    assert not is_valid
    assert "null values" in errors[0]


# ============================================================================
# String Pattern Validation Tests
# ============================================================================

def test_validate_string_pattern_valid():
    """Test string pattern validation with valid data."""
    df = pd.DataFrame({'email': ['a@b.com', 'c@d.com', 'e@f.com']})
    is_valid, errors = DataValidator.validate_string_pattern(
        df, 'email', r'^[a-z]+@[a-z]+\.com$'
    )

    assert is_valid
    assert len(errors) == 0


def test_validate_string_pattern_invalid():
    """Test string pattern validation with invalid data."""
    df = pd.DataFrame({'email': ['a@b.com', 'invalid', 'c@d.com']})
    is_valid, errors = DataValidator.validate_string_pattern(
        df, 'email', r'^[a-z]+@[a-z]+\.com$'
    )

    assert not is_valid
    assert "not matching pattern" in errors[0]


def test_validate_string_pattern_with_null():
    """Test string pattern validation with null values."""
    df = pd.DataFrame({'email': ['a@b.com', None, 'c@d.com']})
    is_valid, errors = DataValidator.validate_string_pattern(
        df, 'email', r'^[a-z]+@[a-z]+\.com$', allow_null=True
    )

    assert is_valid
    assert len(errors) == 0


# ============================================================================
# String Length Validation Tests
# ============================================================================

def test_validate_string_length_valid():
    """Test string length validation with valid data."""
    df = pd.DataFrame({'code': ['ABC', 'DEF', 'GHI']})
    is_valid, errors = DataValidator.validate_string_length(
        df, 'code', min_length=3, max_length=3
    )

    assert is_valid
    assert len(errors) == 0


def test_validate_string_length_too_short():
    """Test string length validation with too short strings."""
    df = pd.DataFrame({'code': ['AB', 'DEF', 'GHI']})
    is_valid, errors = DataValidator.validate_string_length(
        df, 'code', min_length=3
    )

    assert not is_valid
    assert "shorter than" in errors[0]


def test_validate_string_length_too_long():
    """Test string length validation with too long strings."""
    df = pd.DataFrame({'code': ['ABC', 'DEFG', 'GHI']})
    is_valid, errors = DataValidator.validate_string_length(
        df, 'code', max_length=3
    )

    assert not is_valid
    assert "longer than" in errors[0]


# ============================================================================
# Schema Validation Tests
# ============================================================================

def test_validate_schema_valid():
    """Test schema validation with valid data."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35]
    })

    schema = {
        'id': {'type': 'numeric', 'required': True},
        'name': {'type': 'string', 'required': True},
        'age': {'type': 'numeric', 'min_value': 0, 'max_value': 100}
    }

    is_valid, errors = DataValidator.validate_schema(df, schema)

    assert is_valid
    assert len(errors) == 0


def test_validate_schema_missing_required():
    """Test schema validation with missing required column."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie']
    })

    schema = {
        'id': {'type': 'numeric', 'required': True},
        'age': {'type': 'numeric', 'required': True}
    }

    is_valid, errors = DataValidator.validate_schema(df, schema)

    assert not is_valid
    assert any("Missing required columns" in err for err in errors)


def test_validate_schema_extra_columns_strict():
    """Test schema validation with extra columns in strict mode."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'extra': ['X', 'Y', 'Z']
    })

    schema = {
        'id': {'type': 'numeric'},
        'name': {'type': 'string'}
    }

    is_valid, errors = DataValidator.validate_schema(df, schema, strict=True)

    assert not is_valid
    assert any("Extra columns" in err for err in errors)


def test_validate_schema_extra_columns_not_strict():
    """Test schema validation with extra columns in non-strict mode."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'extra': ['X', 'Y', 'Z']
    })

    schema = {
        'id': {'type': 'numeric'},
        'name': {'type': 'string'}
    }

    is_valid, errors = DataValidator.validate_schema(df, schema, strict=False)

    assert is_valid
    assert len(errors) == 0


def test_validate_schema_wrong_type():
    """Test schema validation with wrong data type."""
    df = pd.DataFrame({
        'id': ['1', '2', '3'],  # String instead of numeric
        'name': ['Alice', 'Bob', 'Charlie']
    })

    schema = {
        'id': {'type': 'numeric'},
        'name': {'type': 'string'}
    }

    is_valid, errors = DataValidator.validate_schema(df, schema)

    assert not is_valid


def test_validate_schema_nullable():
    """Test schema validation with nullable constraint."""
    df = pd.DataFrame({
        'id': [1, 2, np.nan],
        'name': ['Alice', 'Bob', 'Charlie']
    })

    schema = {
        'id': {'type': 'numeric', 'nullable': False}
    }

    is_valid, errors = DataValidator.validate_schema(df, schema, strict=False)

    assert not is_valid
    assert any("null values" in err for err in errors)


def test_validate_schema_allowed_values():
    """Test schema validation with allowed values constraint."""
    df = pd.DataFrame({
        'status': ['active', 'invalid', 'pending']
    })

    schema = {
        'status': {
            'type': 'string',
            'allowed_values': ['active', 'pending', 'inactive']
        }
    }

    is_valid, errors = DataValidator.validate_schema(df, schema, strict=False)

    assert not is_valid
    assert any("invalid values" in err for err in errors)


def test_validate_schema_pattern():
    """Test schema validation with pattern constraint."""
    df = pd.DataFrame({
        'email': ['a@b.com', 'invalid', 'c@d.com']
    })

    schema = {
        'email': {
            'type': 'string',
            'pattern': r'^[a-z]+@[a-z]+\.com$'
        }
    }

    is_valid, errors = DataValidator.validate_schema(df, schema, strict=False)

    assert not is_valid


def test_validate_schema_complex():
    """Test complex schema validation."""
    df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'age': [25, 30, 35, 40],
        'email': ['a@b.com', 'b@c.com', 'c@d.com', 'd@e.com'],
        'status': ['active', 'active', 'inactive', 'pending']
    })

    schema = {
        'id': {
            'type': 'numeric',
            'required': True,
            'nullable': False
        },
        'name': {
            'type': 'string',
            'required': True,
            'min_length': 2,
            'max_length': 50
        },
        'age': {
            'type': 'numeric',
            'min_value': 0,
            'max_value': 120
        },
        'email': {
            'type': 'string',
            'pattern': r'^[a-z]+@[a-z]+\.com$'
        },
        'status': {
            'type': 'string',
            'allowed_values': ['active', 'inactive', 'pending']
        }
    }

    is_valid, errors = DataValidator.validate_schema(df, schema)

    assert is_valid
    assert len(errors) == 0


# ============================================================================
# Custom Validation Tests
# ============================================================================

def test_validate_custom_valid():
    """Test custom validation with valid data."""
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    def custom_validator(data):
        if (data['A'] < data['B']).all():
            return True, []
        return False, ["A must be less than B"]

    is_valid, errors = DataValidator.validate_custom(df, custom_validator)

    assert is_valid
    assert len(errors) == 0


def test_validate_custom_invalid():
    """Test custom validation with invalid data."""
    df = pd.DataFrame({'A': [1, 5, 3], 'B': [4, 2, 6]})

    def custom_validator(data):
        if (data['A'] < data['B']).all():
            return True, []
        return False, ["A must be less than B"]

    is_valid, errors = DataValidator.validate_custom(df, custom_validator)

    assert not is_valid
    assert "A must be less than B" in errors


def test_validate_custom_exception():
    """Test custom validation with exception."""
    df = pd.DataFrame({'A': [1, 2, 3]})

    def failing_validator(data):
        raise ValueError("Something went wrong")

    is_valid, errors = DataValidator.validate_custom(
        df, failing_validator, description="Test validation"
    )

    assert not is_valid
    assert "failed with error" in errors[0]


# ============================================================================
# Validate All Tests
# ============================================================================

def test_validate_all_all_pass():
    """Test validate_all with all validations passing."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'age': [25, 30, 35]
    })

    validations = [
        (DataValidator.validate_dataframe, {}, "DataFrame structure"),
        (DataValidator.check_column_exists, {'columns': ['id', 'age']}, "Required columns"),
        (DataValidator.validate_numeric_columns, {'columns': ['age']}, "Numeric columns"),
    ]

    result = DataValidator.validate_all(df, validations)

    assert result['is_valid']
    assert result['total_errors'] == 0
    assert len(result['validations']) == 3


def test_validate_all_some_fail():
    """Test validate_all with some validations failing."""
    df = pd.DataFrame({
        'id': [1, 2, 2],  # Duplicate
        'name': ['Alice', 'Bob', 'Charlie']
    })

    validations = [
        (DataValidator.validate_dataframe, {}, "DataFrame structure"),
        (DataValidator.validate_no_duplicates, {}, "No duplicates"),
        (DataValidator.check_column_exists, {'columns': ['missing']}, "Required columns"),
    ]

    result = DataValidator.validate_all(df, validations)

    assert not result['is_valid']
    assert result['total_errors'] > 0
    assert len(result['error_messages']) > 0


def test_validate_all_with_exception():
    """Test validate_all with exception in validation."""
    df = pd.DataFrame({'A': [1, 2, 3]})

    def bad_validator(data, **kwargs):
        raise ValueError("Test error")

    validations = [
        (DataValidator.validate_dataframe, {}, "DataFrame structure"),
        (bad_validator, {}, "Bad validation"),
    ]

    result = DataValidator.validate_all(df, validations)

    assert not result['is_valid']
    assert result['total_errors'] > 0
    assert any("raised exception" in msg for msg in result['error_messages'])


# ============================================================================
# Integration Tests
# ============================================================================

def test_complete_data_validation_workflow():
    """Test complete validation workflow."""
    # Create test data
    df = pd.DataFrame({
        'employee_id': [1, 2, 3, 4, 5],
        'name': ['Alice Smith', 'Bob Jones', 'Charlie Brown', 'David Wilson', 'Eve Davis'],
        'age': [25, 30, 35, 40, 45],
        'email': ['alice@company.com', 'bob@company.com', 'charlie@company.com',
                 'david@company.com', 'eve@company.com'],
        'department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
        'salary': [50000, 60000, 70000, 80000, 90000]
    })

    # Define validation rules
    validations = [
        # Basic structure
        (DataValidator.validate_dataframe, {'min_rows': 1, 'min_cols': 5}, "Basic structure"),

        # Required columns
        (DataValidator.check_column_exists,
         {'columns': ['employee_id', 'name', 'age', 'email', 'department', 'salary']},
         "Required columns"),

        # No duplicates on employee_id
        (DataValidator.validate_no_duplicates, {'subset': ['employee_id']}, "Unique employee IDs"),

        # No missing values
        (DataValidator.validate_no_missing, {}, "No missing values"),

        # Age range
        (DataValidator.validate_value_range,
         {'column': 'age', 'min_value': 18, 'max_value': 65},
         "Valid age range"),

        # Salary range
        (DataValidator.validate_value_range,
         {'column': 'salary', 'min_value': 0},
         "Valid salary"),

        # Email pattern
        (DataValidator.validate_string_pattern,
         {'column': 'email', 'pattern': r'^[a-z]+@[a-z]+\.com$'},
         "Valid email format"),

        # Department allowed values
        (DataValidator.validate_allowed_values,
         {'column': 'department', 'allowed_values': ['IT', 'HR', 'Finance', 'Sales']},
         "Valid departments"),
    ]

    result = DataValidator.validate_all(df, validations)

    assert result['is_valid']
    assert result['total_errors'] == 0
    assert len(result['validations']) == 8
    assert all(v['is_valid'] for v in result['validations'])


def test_validation_error_exception():
    """Test ValidationError exception."""
    error = ValidationError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)
