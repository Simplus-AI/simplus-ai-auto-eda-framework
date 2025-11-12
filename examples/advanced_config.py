"""
Advanced configuration example for Simplus EDA Framework.

This example demonstrates how to customize the EDA analysis
using configuration options.
"""

import pandas as pd
import numpy as np
from simplus_eda import EDAAnalyzer, ReportGenerator
from simplus_eda.core.config import EDAConfig

# Create sample dataset with more complexity
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'feature_1': np.random.normal(100, 15, n_samples),
    'feature_2': np.random.exponential(50, n_samples),
    'feature_3': np.random.uniform(0, 100, n_samples),
    'feature_4': np.random.poisson(5, n_samples),
    'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
    'target': np.random.choice([0, 1], n_samples)
})

# Add some missing values
data.loc[np.random.choice(data.index, 50), 'feature_1'] = np.nan
data.loc[np.random.choice(data.index, 30), 'feature_2'] = np.nan

# Add some outliers
data.loc[np.random.choice(data.index, 10), 'feature_3'] = 500


def main():
    """Run advanced EDA analysis with custom configuration."""
    print("Simplus EDA Framework - Advanced Configuration Example")
    print("=" * 60)

    # Create custom configuration
    config = EDAConfig(
        enable_statistical_tests=True,
        enable_visualizations=True,
        correlation_threshold=0.5,  # Lower threshold for correlations
        missing_threshold=0.05,     # 5% missing data threshold
        outlier_method="zscore",    # Use Z-score method for outliers
        n_samples_viz=5000,         # Max samples for visualizations
        random_state=42,            # For reproducibility
        custom_params={
            "generate_plots": True,
            "verbose": True
        }
    )

    print("\nConfiguration:")
    print(f"  Statistical tests: {config.enable_statistical_tests}")
    print(f"  Visualizations: {config.enable_visualizations}")
    print(f"  Correlation threshold: {config.correlation_threshold}")
    print(f"  Missing data threshold: {config.missing_threshold}")
    print(f"  Outlier method: {config.outlier_method}")
    print(f"  Samples for viz: {config.n_samples_viz}")

    # Initialize analyzer with custom config
    analyzer = EDAAnalyzer(config=config.to_dict())

    # Perform EDA
    print("\nPerforming customized EDA analysis...")
    results = analyzer.analyze(data)

    # Display results
    print("\nAnalysis Summary:")
    print(f"  Dataset shape: {data.shape}")
    print(f"  Missing values found: {data.isnull().sum().sum()}")
    print(f"  Numeric features: {data.select_dtypes(include=[np.number]).shape[1]}")
    print(f"  Categorical features: {data.select_dtypes(include=['object']).shape[1]}")

    # Generate comprehensive reports
    print("\nGenerating reports with custom settings...")
    report_gen = ReportGenerator(config={
        "include_visualizations": True,
        "format": "detailed"
    })

    # Generate reports
    report_gen.generate_json(results, "output/advanced_eda_report.json")
    print("✓ JSON report: output/advanced_eda_report.json")

    report_gen.generate_html(results, "output/advanced_eda_report.html")
    print("✓ HTML report: output/advanced_eda_report.html")

    print("\n" + "=" * 60)
    print("Advanced analysis complete!")


if __name__ == "__main__":
    main()
