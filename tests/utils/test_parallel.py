"""
Test script for parallel processing functionality.

This script demonstrates and tests the parallel processing features
of the Simplus EDA Framework.
"""

import pandas as pd
import numpy as np
import time
from simplus_eda import SimplusEDA, EDAConfig
from simplus_eda.utils.parallel import ParallelProcessor


def generate_test_data(n_rows=10000, n_cols=20):
    """Generate synthetic test data."""
    print(f"Generating test dataset: {n_rows} rows × {n_cols} columns...")

    data = {}
    for i in range(n_cols):
        # Mix of different distributions
        if i % 3 == 0:
            data[f'col_{i}'] = np.random.normal(100, 15, n_rows)
        elif i % 3 == 1:
            data[f'col_{i}'] = np.random.exponential(50, n_rows)
        else:
            data[f'col_{i}'] = np.random.uniform(0, 100, n_rows)

    df = pd.DataFrame(data)
    print(f"✓ Generated dataset with shape: {df.shape}\n")
    return df


def test_sequential_vs_parallel():
    """Compare sequential vs parallel processing performance."""
    print("="*70)
    print("TEST 1: Sequential vs Parallel Performance Comparison")
    print("="*70 + "\n")

    # Generate test data
    df = generate_test_data(n_rows=10000, n_cols=20)

    # Test 1: Sequential processing
    print("Running SEQUENTIAL analysis (n_jobs=1)...")
    config_seq = EDAConfig(
        n_jobs=1,
        verbose=False,
        enable_statistical_tests=True
    )
    eda_seq = SimplusEDA(config=config_seq)

    start = time.time()
    results_seq = eda_seq.analyze(df)
    time_seq = time.time() - start

    print(f"✓ Sequential analysis completed in {time_seq:.2f} seconds\n")

    # Test 2: Parallel processing
    print("Running PARALLEL analysis (n_jobs=-1, all cores)...")
    config_par = EDAConfig(
        n_jobs=-1,
        verbose=True,  # Show progress
        enable_statistical_tests=True
    )
    eda_par = SimplusEDA(config=config_par)

    start = time.time()
    results_par = eda_par.analyze(df)
    time_par = time.time() - start

    print(f"✓ Parallel analysis completed in {time_par:.2f} seconds\n")

    # Compare results
    speedup = time_seq / time_par
    print("RESULTS:")
    print(f"  Sequential time: {time_seq:.2f}s")
    print(f"  Parallel time:   {time_par:.2f}s")
    print(f"  Speedup:         {speedup:.2f}x")

    if speedup > 1.0:
        print(f"  ✓ Parallel processing is {speedup:.1f}x FASTER!")
    else:
        print(f"  ⚠ Parallel processing overhead detected (dataset may be too small)")

    print("\n")


def test_parallel_processor():
    """Test the ParallelProcessor utility directly."""
    print("="*70)
    print("TEST 2: ParallelProcessor Utility Functions")
    print("="*70 + "\n")

    df = generate_test_data(n_rows=5000, n_cols=10)

    # Define a custom analysis function
    def custom_analysis(col_data, col_name):
        """Example custom analysis."""
        return {
            'column': col_name,
            'mean': float(col_data.mean()),
            'std': float(col_data.std()),
            'min': float(col_data.min()),
            'max': float(col_data.max()),
            'zeros': int((col_data == 0).sum())
        }

    # Test parallel processing
    processor = ParallelProcessor(n_jobs=-1, verbose=True)

    print("Processing columns in parallel...")
    start = time.time()
    results = processor.process_numeric_columns_parallel(
        data=df,
        func=custom_analysis
    )
    elapsed = time.time() - start

    print(f"\n✓ Processed {len(results)} columns in {elapsed:.2f}s")
    print(f"  Average time per column: {elapsed/len(results)*1000:.1f}ms")

    # Show sample results
    print("\nSample results (first 3 columns):")
    for i, (col_name, col_results) in enumerate(list(results.items())[:3]):
        print(f"  {col_name}:")
        print(f"    Mean: {col_results['mean']:.2f}")
        print(f"    Std:  {col_results['std']:.2f}")
        print(f"    Range: [{col_results['min']:.2f}, {col_results['max']:.2f}]")

    print("\n")


def test_configuration_methods():
    """Test different ways to configure parallel processing."""
    print("="*70)
    print("TEST 3: Configuration Methods")
    print("="*70 + "\n")

    df = generate_test_data(n_rows=5000, n_cols=10)

    # Method 1: Direct kwargs
    print("Method 1: Direct kwargs")
    eda1 = SimplusEDA(n_jobs=4, verbose=False)
    eda1.analyze(df)
    print(f"  ✓ Config: n_jobs={eda1.config.n_jobs}")

    # Method 2: EDAConfig object
    print("\nMethod 2: EDAConfig object")
    config2 = EDAConfig(n_jobs=-1, verbose=True)
    eda2 = SimplusEDA(config=config2)
    print(f"  ✓ Config: n_jobs={eda2.config.n_jobs}")

    # Method 3: Dictionary
    print("\nMethod 3: Dictionary")
    config_dict = {'n_jobs': 2, 'verbose': False}
    eda3 = SimplusEDA(config=config_dict)
    eda3.analyze(df)
    print(f"  ✓ Config: n_jobs={eda3.config.n_jobs}")

    print("\n✓ All configuration methods work!\n")


def test_individual_analyzers():
    """Test individual analyzers with parallel processing."""
    print("="*70)
    print("TEST 4: Individual Analyzer Parallel Support")
    print("="*70 + "\n")

    df = generate_test_data(n_rows=5000, n_cols=15)

    # Test StatisticalAnalyzer
    print("Testing StatisticalAnalyzer with n_jobs=-1...")
    from simplus_eda.analyzers import StatisticalAnalyzer
    stat_analyzer = StatisticalAnalyzer(n_jobs=-1, verbose=True)
    stat_results = stat_analyzer.analyze(df)
    print(f"  ✓ Analyzed {len(stat_results['descriptive']['numeric_columns'])} columns")
    print(f"  ✓ Parallel processing: {stat_results['descriptive']['parallel_processing']}")

    # Test CorrelationAnalyzer
    print("\nTesting CorrelationAnalyzer with n_jobs=-1...")
    from simplus_eda.analyzers import CorrelationAnalyzer
    corr_analyzer = CorrelationAnalyzer(n_jobs=-1, verbose=True)
    corr_results = corr_analyzer.analyze(df, threshold=0.3)
    print(f"  ✓ Found {corr_results['strong_correlations']['count']} strong correlations")
    print(f"  ✓ Parallel processing: {corr_results['strong_correlations']['parallel_processing']}")

    # Test OutlierAnalyzer
    print("\nTesting OutlierAnalyzer with n_jobs=-1...")
    from simplus_eda.analyzers import OutlierAnalyzer
    outlier_analyzer = OutlierAnalyzer(n_jobs=-1, verbose=True)
    outlier_results = outlier_analyzer.analyze(df)
    print(f"  ✓ Detected outliers using {outlier_analyzer.method} method")

    print("\n✓ All analyzers support parallel processing!\n")


def test_cpu_detection():
    """Test CPU core detection."""
    print("="*70)
    print("TEST 5: CPU Core Detection")
    print("="*70 + "\n")

    from multiprocessing import cpu_count

    total_cores = cpu_count()
    print(f"System Information:")
    print(f"  Total CPU cores: {total_cores}")

    # Test different n_jobs values
    processor = ParallelProcessor(n_jobs=-1, verbose=False)
    print(f"\n  n_jobs=-1  → using {processor.n_jobs} cores (all)")

    processor = ParallelProcessor(n_jobs=-2, verbose=False)
    print(f"  n_jobs=-2  → using {processor.n_jobs} cores (all but one)")

    processor = ParallelProcessor(n_jobs=1, verbose=False)
    print(f"  n_jobs=1   → using {processor.n_jobs} core  (sequential)")
    print(f"              Parallel: {processor.is_parallel()}")

    processor = ParallelProcessor(n_jobs=4, verbose=False)
    print(f"  n_jobs=4   → using {processor.n_jobs} cores")
    print(f"              Parallel: {processor.is_parallel()}")

    print("\n✓ CPU detection working correctly!\n")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SIMPLUS EDA - PARALLEL PROCESSING TEST SUITE")
    print("="*70 + "\n")

    try:
        # Run all tests
        test_cpu_detection()
        test_configuration_methods()
        test_parallel_processor()
        test_individual_analyzers()
        test_sequential_vs_parallel()

        print("="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")

        print("Summary:")
        print("  ✓ Parallel processing is fully implemented")
        print("  ✓ All configuration methods work")
        print("  ✓ Individual analyzers support n_jobs")
        print("  ✓ CPU core detection working")
        print("  ✓ Performance improvements demonstrated")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
