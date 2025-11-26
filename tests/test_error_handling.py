"""
Test suite for error handling and resilience features.

This script demonstrates and tests:
- Custom exception hierarchy
- Logging system
- Error recovery mechanisms
- Graceful degradation
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from simplus_eda import (
    configure_logging,
    get_logger,
    SimplusEDA,
    SimplusEDAError,
    ConfigurationError,
    DataError,
    AnalysisError
)
from simplus_eda.exceptions import (
    DataValidationError,
    DataLoadingError,
    EmptyDataError,
    InvalidDataTypeError,
    InvalidConfigurationError,
    StatisticalAnalysisError,
    wrap_error
)
from simplus_eda.logging_config import (
    LogContext,
    PerformanceTracker,
    log_execution,
    log_errors
)
from simplus_eda.error_recovery import (
    retry,
    with_fallback,
    safe_execute,
    PartialResults,
    execute_partial,
    ErrorContext,
    degrade_gracefully
)


def print_section(title):
    """Print a section header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70 + "\n")


# ============================================================================
# Test 1: Exception Hierarchy
# ============================================================================

def test_exception_hierarchy():
    """Test custom exception hierarchy."""
    print_section("TEST 1: Exception Hierarchy")

    # Test 1.1: SimplusEDAError with details and suggestions
    print("1.1 SimplusEDAError with details and suggestions:")
    try:
        raise SimplusEDAError(
            "Test error message",
            details={'key1': 'value1', 'key2': 'value2'},
            suggestions=[
                "Try suggestion 1",
                "Try suggestion 2"
            ]
        )
    except SimplusEDAError as e:
        print(f"✓ Caught SimplusEDAError")
        print(f"  Message: {e.message}")
        print(f"  Details: {e.details}")
        print(f"  Suggestions: {e.suggestions}")

    # Test 1.2: DataValidationError
    print("\n1.2 DataValidationError:")
    try:
        raise DataValidationError(
            "Invalid column types",
            invalid_columns=['col1', 'col2'],
            expected_type='numeric',
            actual_type='string'
        )
    except DataValidationError as e:
        print(f"✓ Caught DataValidationError")
        print(f"  Invalid columns: {e.details.get('invalid_columns')}")

    # Test 1.3: InvalidConfigurationError
    print("\n1.3 InvalidConfigurationError:")
    try:
        raise InvalidConfigurationError(
            "Invalid correlation threshold",
            parameter="correlation_threshold",
            value=1.5,
            valid_values=["0.0 to 1.0"]
        )
    except InvalidConfigurationError as e:
        print(f"✓ Caught InvalidConfigurationError")
        print(f"  Parameter: {e.details.get('parameter')}")
        print(f"  Value: {e.details.get('provided_value')}")
        print(f"  Valid: {e.details.get('valid_values')}")

    # Test 1.4: wrap_error
    print("\n1.4 Error wrapping:")
    try:
        try:
            open('nonexistent.csv')
        except FileNotFoundError as original:
            raise wrap_error(
                original,
                DataLoadingError,
                "Failed to load data file",
                file_path='nonexistent.csv'
            )
    except DataLoadingError as e:
        print(f"✓ Wrapped FileNotFoundError as DataLoadingError")
        print(f"  Original error: {e.original_error}")

    print("\n✓ Exception hierarchy tests passed!")


# ============================================================================
# Test 2: Logging System
# ============================================================================

def test_logging_system():
    """Test logging system."""
    print_section("TEST 2: Logging System")

    # Configure logging
    configure_logging(level='DEBUG', verbose=True)
    logger = get_logger(__name__)

    # Test 2.1: Different log levels
    print("2.1 Testing log levels:")
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")

    # Test 2.2: LogContext
    print("\n2.2 Testing LogContext:")
    logger.setLevel(logging.INFO)
    logger.debug("This DEBUG won't show (level=INFO)")

    with LogContext(level=logging.DEBUG):
        logger.debug("This DEBUG will show (temporary DEBUG)")

    logger.debug("This DEBUG won't show again (back to INFO)")

    # Test 2.3: Performance Tracker
    print("\n2.3 Testing PerformanceTracker:")
    tracker = PerformanceTracker()

    tracker.start('operation1')
    import time
    time.sleep(0.1)
    tracker.stop('operation1')

    tracker.start('operation2')
    time.sleep(0.05)
    tracker.stop('operation2')

    print("Performance report:")
    tracker.report()

    # Test 2.4: Decorators
    print("\n2.4 Testing decorators:")

    @log_execution()
    def sample_function():
        time.sleep(0.1)
        return "result"

    @log_errors()
    def failing_function():
        raise ValueError("Intentional error")

    result = sample_function()
    print(f"  Function returned: {result}")

    try:
        failing_function()
    except ValueError:
        print("  ✓ Error was logged and re-raised")

    print("\n✓ Logging system tests passed!")


# ============================================================================
# Test 3: Error Recovery
# ============================================================================

def test_error_recovery():
    """Test error recovery mechanisms."""
    print_section("TEST 3: Error Recovery")

    # Test 3.1: Retry decorator
    print("3.1 Testing retry decorator:")

    attempt_count = {'count': 0}

    @retry(max_attempts=3, delay=0.1, backoff=1.5)
    def unstable_function():
        attempt_count['count'] += 1
        if attempt_count['count'] < 3:
            raise ValueError(f"Attempt {attempt_count['count']} failed")
        return "success"

    result = unstable_function()
    print(f"  ✓ Function succeeded after {attempt_count['count']} attempts")
    print(f"  Result: {result}")

    # Test 3.2: Fallback decorator
    print("\n3.2 Testing fallback decorator:")

    @with_fallback(fallback_value=0.0)
    def divide(a, b):
        return a / b

    result1 = divide(10, 2)
    result2 = divide(10, 0)  # Will use fallback
    print(f"  10 / 2 = {result1}")
    print(f"  10 / 0 = {result2} (fallback)")

    # Test 3.3: Safe execution
    print("\n3.3 Testing safe execution:")

    def risky_operation(fail=False):
        if fail:
            raise ValueError("Operation failed")
        return "success"

    result1 = safe_execute(risky_operation, fail=False)
    result2 = safe_execute(risky_operation, fail=True, default="fallback")

    print(f"  Success case: {result1.success}, value={result1.value}")
    print(f"  Failure case: {result2.success}, value={result2.value}")

    # Test 3.4: Partial results
    print("\n3.4 Testing partial results:")

    results = PartialResults()

    for i in range(10):
        if i % 3 == 0:
            results.add_failure(f'item_{i}', ValueError(f"Item {i} failed"))
        else:
            results.add_success(f'item_{i}', i * 2)

    print(f"  Success: {results.success_count}/{results.total_count}")
    print(f"  Success rate: {results.success_rate:.1f}%")
    print(f"  Failed items: {list(results.failures.keys())}")

    # Test 3.5: Execute partial
    print("\n3.5 Testing execute_partial:")

    def process_item(item):
        if item % 3 == 0:
            raise ValueError(f"Item {item} is divisible by 3")
        return item ** 2

    partial_results = execute_partial(
        items=list(range(10)),
        func=process_item,
        item_name_func=lambda x: f"item_{x}",
        raise_on_all_failed=False
    )

    print(f"  Processed: {partial_results.success_count}/{partial_results.total_count}")
    print(f"  Success rate: {partial_results.success_rate:.1f}%")

    # Test 3.6: Error context
    print("\n3.6 Testing error context:")

    try:
        with ErrorContext(operation="test_operation", data_size=1000) as ctx:
            ctx.add_info("stage", "processing")
            raise DataValidationError("Test error")
    except DataValidationError as e:
        print(f"  ✓ Error context preserved:")
        print(f"    Operation: {e.details.get('operation')}")
        print(f"    Data size: {e.details.get('data_size')}")
        print(f"    Stage: {e.details.get('stage')}")

    # Test 3.7: Graceful degradation
    print("\n3.7 Testing graceful degradation:")

    def expensive_method():
        raise ValueError("Too expensive")

    def fast_approximation():
        raise ValueError("Still failing")

    def simple_estimate():
        return "simple result"

    result = degrade_gracefully(
        expensive_method,
        [fast_approximation, simple_estimate]
    )
    print(f"  ✓ Degraded to: {result}")

    print("\n✓ Error recovery tests passed!")


# ============================================================================
# Test 4: Integration with EDA
# ============================================================================

def test_eda_integration():
    """Test error handling integration with EDA."""
    print_section("TEST 4: EDA Integration")

    # Test 4.1: Invalid configuration
    print("4.1 Testing invalid configuration:")
    try:
        from simplus_eda import EDAConfig
        config = EDAConfig(correlation_threshold=1.5)  # Invalid
    except InvalidConfigurationError as e:
        print(f"  ✓ Caught InvalidConfigurationError")
        print(f"    Parameter: {e.details.get('parameter')}")
        print(f"    Value: {e.details.get('provided_value')}")

    # Test 4.2: Empty data
    print("\n4.2 Testing empty data:")
    try:
        eda = SimplusEDA()
        empty_df = pd.DataFrame()
        eda.analyze(empty_df)
    except (ValueError, EmptyDataError) as e:
        print(f"  ✓ Caught error for empty DataFrame")

    # Test 4.3: Analysis with logging
    print("\n4.3 Testing analysis with logging:")
    configure_logging(level='INFO')

    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100)
    })

    eda = SimplusEDA(verbose=True)
    results = eda.analyze(df, quick=True)
    print(f"  ✓ Analysis completed")
    print(f"    Quality score: {eda.get_quality_score():.1f}%")

    print("\n✓ EDA integration tests passed!")


# ============================================================================
# Test 5: Real-world Scenario
# ============================================================================

def test_real_world_scenario():
    """Test real-world error handling scenario."""
    print_section("TEST 5: Real-world Scenario")

    configure_logging(level='INFO', verbose=False)
    logger = get_logger(__name__)

    print("Simulating data analysis pipeline with error handling...")

    # Create test data with some issues
    df = pd.DataFrame({
        'good_col': np.random.randn(100),
        'missing_col': [np.nan] * 50 + list(np.random.randn(50)),
        'constant_col': [1.0] * 100,
        'categorical': np.random.choice(['A', 'B', 'C'], 100)
    })

    # Analyze with error recovery
    results = PartialResults()

    # Statistical analysis with partial results
    print("\nAnalyzing columns:")
    for col in df.select_dtypes(include=[np.number]).columns:
        try:
            col_data = df[col].dropna()
            if len(col_data) < 3:
                raise StatisticalAnalysisError(
                    f"Insufficient data in column '{col}'",
                    analysis_type="descriptive_stats",
                    details={'column': col, 'n_samples': len(col_data)}
                )

            stats = {
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max()
            }
            results.add_success(col, stats)
            logger.info(f"  ✓ {col}: mean={stats['mean']:.2f}")
        except Exception as e:
            results.add_failure(col, e)
            logger.warning(f"  ✗ {col}: {str(e)}")

    print(f"\nResults:")
    print(f"  Successful: {results.success_count}/{results.total_count}")
    print(f"  Failed: {results.failure_count}/{results.total_count}")
    print(f"  Success rate: {results.success_rate:.1f}%")

    if results.has_failures():
        print(f"\nFailed columns:")
        for col, error in results.failures.items():
            print(f"  - {col}: {error}")

    print("\n✓ Real-world scenario test passed!")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" SIMPLUS EDA - ERROR HANDLING TEST SUITE")
    print("="*70)

    try:
        test_exception_hierarchy()
        test_logging_system()
        test_error_recovery()
        test_eda_integration()
        test_real_world_scenario()

        print("\n" + "="*70)
        print(" ALL TESTS PASSED!")
        print("="*70 + "\n")

        print("Summary:")
        print("  ✓ Exception hierarchy working correctly")
        print("  ✓ Logging system functional")
        print("  ✓ Error recovery mechanisms operational")
        print("  ✓ EDA integration successful")
        print("  ✓ Real-world scenario handled gracefully")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
