"""
Custom exception hierarchy for Simplus EDA Framework.

This module provides a comprehensive exception hierarchy for better error
handling, diagnostics, and recovery mechanisms throughout the framework.

Exception Hierarchy:
    SimplusEDAError (base)
    ├── DataError
    │   ├── DataValidationError
    │   ├── DataLoadingError
    │   ├── EmptyDataError
    │   ├── InvalidDataTypeError
    │   └── DataQualityError
    ├── AnalysisError
    │   ├── StatisticalAnalysisError
    │   ├── CorrelationAnalysisError
    │   ├── OutlierAnalysisError
    │   └── InsufficientDataError
    ├── ConfigurationError (already exists)
    │   ├── InvalidConfigurationError
    │   └── MissingConfigurationError
    ├── ReportGenerationError
    │   ├── TemplateError
    │   ├── VisualizationError
    │   └── OutputFormatError
    └── ResourceError
        ├── MemoryError
        ├── DependencyError
        └── FileSystemError
"""

from typing import Optional, Dict, Any, List


class SimplusEDAError(Exception):
    """
    Base exception for all Simplus EDA errors.

    All custom exceptions in the framework inherit from this base class,
    making it easy to catch all framework-specific errors.

    Attributes:
        message: Error message
        details: Additional error details (dictionary)
        suggestions: List of suggested fixes
        original_error: Original exception if this wraps another error
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize SimplusEDAError.

        Args:
            message: Human-readable error message
            details: Additional context about the error
            suggestions: List of suggested fixes
            original_error: Original exception if wrapping another error
        """
        self.message = message
        self.details = details or {}
        self.suggestions = suggestions or []
        self.original_error = original_error

        # Build full error message
        full_message = self._build_message()
        super().__init__(full_message)

    def _build_message(self) -> str:
        """Build comprehensive error message with details and suggestions."""
        parts = [self.message]

        # Add details
        if self.details:
            parts.append("\nDetails:")
            for key, value in self.details.items():
                parts.append(f"  {key}: {value}")

        # Add suggestions
        if self.suggestions:
            parts.append("\nSuggested fixes:")
            for i, suggestion in enumerate(self.suggestions, 1):
                parts.append(f"  {i}. {suggestion}")

        # Add original error
        if self.original_error:
            parts.append(f"\nOriginal error: {type(self.original_error).__name__}: {self.original_error}")

        return "\n".join(parts)

    def __str__(self) -> str:
        """Return string representation."""
        return self._build_message()

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


# ============================================================================
# Data-related Exceptions
# ============================================================================

class DataError(SimplusEDAError):
    """Base exception for data-related errors."""
    pass


class DataValidationError(DataError):
    """
    Exception raised when data validation fails.

    Examples:
        - Missing required columns
        - Invalid data types
        - Schema mismatches
        - Value range violations
    """

    def __init__(
        self,
        message: str,
        invalid_columns: Optional[List[str]] = None,
        expected_type: Optional[str] = None,
        actual_type: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if invalid_columns:
            details['invalid_columns'] = invalid_columns
        if expected_type:
            details['expected_type'] = expected_type
        if actual_type:
            details['actual_type'] = actual_type

        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                "Check your DataFrame schema",
                "Verify column names and data types",
                "Use df.info() to inspect the data structure"
            ]

        super().__init__(message, details=details, suggestions=suggestions, **kwargs)


class DataLoadingError(DataError):
    """
    Exception raised when data loading fails.

    Examples:
        - File not found
        - Unsupported file format
        - Corrupt file
        - Encoding issues
    """

    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if file_path:
            details['file_path'] = file_path

        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                "Verify the file path exists",
                "Check file permissions",
                "Ensure the file format is supported (CSV, Excel, Parquet, HDF5, JSON)",
                "Try specifying encoding explicitly"
            ]

        super().__init__(message, details=details, suggestions=suggestions, **kwargs)


class EmptyDataError(DataError):
    """
    Exception raised when attempting to analyze empty data.

    Examples:
        - Empty DataFrame
        - No rows after filtering
        - All values are missing
    """

    def __init__(self, message: str = "Dataset is empty", **kwargs):
        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                "Check if data was loaded correctly",
                "Verify filters are not too restrictive",
                "Ensure the data source contains data"
            ]

        super().__init__(message, suggestions=suggestions, **kwargs)


class InvalidDataTypeError(DataError):
    """
    Exception raised when data has invalid or unexpected types.

    Examples:
        - Non-DataFrame input
        - Mixed types in column
        - Unsupported data types
    """

    def __init__(
        self,
        message: str,
        expected_type: Optional[str] = None,
        actual_type: Optional[str] = None,
        column: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if expected_type:
            details['expected_type'] = expected_type
        if actual_type:
            details['actual_type'] = actual_type
        if column:
            details['column'] = column

        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                f"Convert to {expected_type} if possible",
                "Check data type compatibility",
                "Use df.astype() for type conversion"
            ]

        super().__init__(message, details=details, suggestions=suggestions, **kwargs)


class DataQualityError(DataError):
    """
    Exception raised when data quality is too low for analysis.

    Examples:
        - Too many missing values
        - Too many duplicates
        - Data quality score below threshold
    """

    def __init__(
        self,
        message: str,
        quality_score: Optional[float] = None,
        threshold: Optional[float] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if quality_score is not None:
            details['quality_score'] = quality_score
        if threshold is not None:
            details['threshold'] = threshold

        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                "Clean the data before analysis",
                "Handle missing values appropriately",
                "Remove or investigate duplicates",
                "Consider data preprocessing steps"
            ]

        super().__init__(message, details=details, suggestions=suggestions, **kwargs)


# ============================================================================
# Analysis-related Exceptions
# ============================================================================

class AnalysisError(SimplusEDAError):
    """Base exception for analysis-related errors."""
    pass


class StatisticalAnalysisError(AnalysisError):
    """
    Exception raised during statistical analysis.

    Examples:
        - Normality test failures
        - Insufficient data for statistics
        - Numerical computation errors
    """

    def __init__(self, message: str, analysis_type: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if analysis_type:
            details['analysis_type'] = analysis_type

        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                "Check for sufficient data points",
                "Verify data is numeric where required",
                "Handle missing values before analysis",
                "Consider using robust statistical methods"
            ]

        super().__init__(message, details=details, suggestions=suggestions, **kwargs)


class CorrelationAnalysisError(AnalysisError):
    """
    Exception raised during correlation analysis.

    Examples:
        - Insufficient numeric columns
        - Zero variance columns
        - Multicollinearity issues
    """

    def __init__(
        self,
        message: str,
        n_numeric_columns: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if n_numeric_columns is not None:
            details['n_numeric_columns'] = n_numeric_columns

        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                "Ensure at least 2 numeric columns exist",
                "Remove zero-variance columns",
                "Check for constant columns",
                "Verify data types are numeric"
            ]

        super().__init__(message, details=details, suggestions=suggestions, **kwargs)


class OutlierAnalysisError(AnalysisError):
    """
    Exception raised during outlier detection.

    Examples:
        - Invalid outlier method
        - Insufficient data for method
        - Numerical issues in computation
    """

    def __init__(
        self,
        message: str,
        method: Optional[str] = None,
        column: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if method:
            details['method'] = method
        if column:
            details['column'] = column

        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                "Try a different outlier detection method",
                "Check for sufficient data points",
                "Verify data distribution",
                "Consider robust outlier methods (Modified Z-score, IQR)"
            ]

        super().__init__(message, details=details, suggestions=suggestions, **kwargs)


class InsufficientDataError(AnalysisError):
    """
    Exception raised when there's insufficient data for analysis.

    Examples:
        - Too few rows
        - Too few columns
        - Insufficient samples after filtering
    """

    def __init__(
        self,
        message: str,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        minimum_required: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if n_rows is not None:
            details['n_rows'] = n_rows
        if n_cols is not None:
            details['n_cols'] = n_cols
        if minimum_required is not None:
            details['minimum_required'] = minimum_required

        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                f"Provide at least {minimum_required} data points" if minimum_required else "Provide more data",
                "Check if data was filtered too aggressively",
                "Verify data loading was successful"
            ]

        super().__init__(message, details=details, suggestions=suggestions, **kwargs)


# ============================================================================
# Configuration-related Exceptions
# ============================================================================

class ConfigurationError(SimplusEDAError):
    """
    Base exception for configuration-related errors.

    This replaces the old ConfigurationError from config.py with enhanced functionality.
    """
    pass


class InvalidConfigurationError(ConfigurationError):
    """
    Exception raised when configuration parameters are invalid.

    Examples:
        - Out of range values
        - Invalid method names
        - Incompatible parameter combinations
    """

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Optional[Any] = None,
        valid_values: Optional[List[Any]] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if parameter:
            details['parameter'] = parameter
        if value is not None:
            details['provided_value'] = value
        if valid_values:
            details['valid_values'] = valid_values

        suggestions = kwargs.get('suggestions', [])
        if not suggestions and valid_values:
            suggestions = [f"Use one of the valid values: {', '.join(map(str, valid_values))}"]

        super().__init__(message, details=details, suggestions=suggestions, **kwargs)


class MissingConfigurationError(ConfigurationError):
    """
    Exception raised when required configuration is missing.

    Examples:
        - Required parameter not set
        - Missing configuration file
        - Incomplete configuration
    """

    def __init__(self, message: str, missing_parameters: Optional[List[str]] = None, **kwargs):
        details = kwargs.get('details', {})
        if missing_parameters:
            details['missing_parameters'] = missing_parameters

        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                "Provide required configuration parameters",
                "Check configuration file completeness",
                "Use EDAConfig.get_default() for a valid starting point"
            ]

        super().__init__(message, details=details, suggestions=suggestions, **kwargs)


# ============================================================================
# Report Generation Exceptions
# ============================================================================

class ReportGenerationError(SimplusEDAError):
    """Base exception for report generation errors."""
    pass


class TemplateError(ReportGenerationError):
    """
    Exception raised when template processing fails.

    Examples:
        - Missing template file
        - Template syntax errors
        - Missing template variables
    """

    def __init__(
        self,
        message: str,
        template_name: Optional[str] = None,
        missing_variables: Optional[List[str]] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if template_name:
            details['template_name'] = template_name
        if missing_variables:
            details['missing_variables'] = missing_variables

        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                "Check template file exists",
                "Verify template syntax",
                "Ensure all required variables are provided",
                "Check Jinja2 template documentation"
            ]

        super().__init__(message, details=details, suggestions=suggestions, **kwargs)


class VisualizationError(ReportGenerationError):
    """
    Exception raised when visualization generation fails.

    Examples:
        - Matplotlib errors
        - Plotly errors
        - Invalid plot parameters
        - Insufficient data for plot
    """

    def __init__(
        self,
        message: str,
        plot_type: Optional[str] = None,
        column: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if plot_type:
            details['plot_type'] = plot_type
        if column:
            details['column'] = column

        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                "Check data is suitable for this visualization",
                "Verify matplotlib/plotly is installed",
                "Reduce data size if memory issues",
                "Try a different visualization type"
            ]

        super().__init__(message, details=details, suggestions=suggestions, **kwargs)


class OutputFormatError(ReportGenerationError):
    """
    Exception raised when output format is unsupported or invalid.

    Examples:
        - Unknown format requested
        - Missing dependencies for format (e.g., weasyprint for PDF)
        - Invalid file extension
    """

    def __init__(
        self,
        message: str,
        requested_format: Optional[str] = None,
        supported_formats: Optional[List[str]] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if requested_format:
            details['requested_format'] = requested_format
        if supported_formats:
            details['supported_formats'] = supported_formats

        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            if supported_formats:
                suggestions = [f"Use one of: {', '.join(supported_formats)}"]
            suggestions.extend([
                "Check output format is supported",
                "Install required dependencies (e.g., pip install simplus-eda[pdf])",
                "Verify file extension matches format"
            ])

        super().__init__(message, details=details, suggestions=suggestions, **kwargs)


# ============================================================================
# Resource-related Exceptions
# ============================================================================

class ResourceError(SimplusEDAError):
    """Base exception for resource-related errors."""
    pass


class MemoryError(ResourceError):
    """
    Exception raised when memory is insufficient.

    Examples:
        - Out of memory
        - Dataset too large
        - Memory allocation failure
    """

    def __init__(
        self,
        message: str = "Insufficient memory for operation",
        required_memory_mb: Optional[float] = None,
        available_memory_mb: Optional[float] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if required_memory_mb:
            details['required_memory_mb'] = required_memory_mb
        if available_memory_mb:
            details['available_memory_mb'] = available_memory_mb

        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                "Reduce dataset size",
                "Use sampling (n_samples_viz parameter)",
                "Enable quick mode (skip statistical tests)",
                "Close other applications to free memory",
                "Process data in chunks"
            ]

        super().__init__(message, details=details, suggestions=suggestions, **kwargs)


class DependencyError(ResourceError):
    """
    Exception raised when required dependencies are missing.

    Examples:
        - Missing optional packages
        - Version incompatibility
        - Import errors
    """

    def __init__(
        self,
        message: str,
        missing_package: Optional[str] = None,
        install_command: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if missing_package:
            details['missing_package'] = missing_package

        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            if install_command:
                suggestions.append(f"Install with: {install_command}")
            elif missing_package:
                suggestions.append(f"Install with: pip install {missing_package}")
            suggestions.extend([
                "Check package is installed correctly",
                "Verify package version compatibility",
                "Try reinstalling the package"
            ])

        super().__init__(message, details=details, suggestions=suggestions, **kwargs)


class FileSystemError(ResourceError):
    """
    Exception raised for file system related errors.

    Examples:
        - Permission denied
        - Disk full
        - Path not found
        - Read-only filesystem
    """

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if path:
            details['path'] = path
        if operation:
            details['operation'] = operation

        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                "Check file/directory permissions",
                "Verify disk space is available",
                "Ensure parent directories exist",
                "Check path is valid and accessible"
            ]

        super().__init__(message, details=details, suggestions=suggestions, **kwargs)


# ============================================================================
# Utility Functions
# ============================================================================

def format_error_message(
    error: Exception,
    context: Optional[str] = None,
    show_traceback: bool = False
) -> str:
    """
    Format an error message for display.

    Args:
        error: The exception to format
        context: Optional context about where the error occurred
        show_traceback: Whether to include traceback information

    Returns:
        Formatted error message string
    """
    parts = []

    if context:
        parts.append(f"Error in {context}:")

    if isinstance(error, SimplusEDAError):
        parts.append(str(error))
    else:
        parts.append(f"{type(error).__name__}: {error}")

    if show_traceback:
        import traceback
        parts.append("\nTraceback:")
        parts.append(traceback.format_exc())

    return "\n".join(parts)


def wrap_error(
    original_error: Exception,
    error_class: type,
    message: str,
    **kwargs
) -> SimplusEDAError:
    """
    Wrap an existing exception in a SimplusEDA exception.

    Args:
        original_error: The original exception to wrap
        error_class: The SimplusEDA exception class to use
        message: Custom message for the new exception
        **kwargs: Additional arguments for the exception

    Returns:
        New SimplusEDA exception wrapping the original

    Example:
        >>> try:
        ...     df = pd.read_csv('missing.csv')
        ... except FileNotFoundError as e:
        ...     raise wrap_error(e, DataLoadingError,
        ...                     "Failed to load data file",
        ...                     file_path='missing.csv')
    """
    kwargs['original_error'] = original_error
    return error_class(message, **kwargs)
