"""
Examples demonstrating the unified SimplusEDA API.

This file shows how the config, analyzer, and report modules work seamlessly
together through the SimplusEDA unified interface.
"""

import pandas as pd
import numpy as np
from simplus_eda import SimplusEDA, quick_analysis, analyze_and_report, EDAConfig


def create_sample_data():
    """Create sample dataset for demonstration."""
    np.random.seed(42)
    n = 500

    data = {
        'age': np.random.normal(35, 10, n).astype(int),
        'income': np.random.lognormal(10.5, 0.5, n),
        'credit_score': np.random.normal(700, 50, n),
        'loan_amount': np.random.uniform(5000, 50000, n),
        'debt_ratio': np.random.beta(2, 5, n),
        'employment_years': np.random.poisson(7, n),
        'num_accounts': np.random.randint(1, 10, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'approved': np.random.choice([True, False], n, p=[0.7, 0.3])
    }

    df = pd.DataFrame(data)

    # Add some missing values
    df.loc[df.sample(frac=0.05).index, 'credit_score'] = np.nan
    df.loc[df.sample(frac=0.03).index, 'income'] = np.nan

    # Create correlation
    df['loan_amount'] = df['income'] * 0.3 + np.random.normal(0, 5000, n)

    return df


def example_1_simplest_usage():
    """Example 1: The simplest possible usage - one function call."""
    print("=" * 70)
    print("Example 1: Simplest Usage - One Function Call")
    print("=" * 70)

    df = create_sample_data()

    # Single function call does everything!
    report_path = quick_analysis(df, 'outputs/example1_quick_report.html')

    print(f"\n✓ Analysis complete! Report saved to: {report_path}")
    print("  This one function call performed:")
    print("  - Comprehensive data analysis")
    print("  - Statistical computations")
    print("  - Data quality assessment")
    print("  - Visualization generation")
    print("  - Professional HTML report creation")


def example_2_basic_workflow():
    """Example 2: Basic workflow with default configuration."""
    print("\n" + "=" * 70)
    print("Example 2: Basic Workflow with Default Configuration")
    print("=" * 70)

    df = create_sample_data()

    # Step 1: Create EDA instance
    eda = SimplusEDA()
    print("\n1. Created SimplusEDA instance with default configuration")

    # Step 2: Analyze data
    results = eda.analyze(df)
    print(f"2. Analyzed {df.shape[0]} rows × {df.shape[1]} columns")

    # Step 3: View summary
    print("\n3. Analysis Summary:")
    print(eda.summary())

    # Step 4: Generate report
    report_path = eda.generate_report('outputs/example2_basic_report.html')
    print(f"\n4. Generated report: {report_path}")

    # Step 5: Access insights
    insights = eda.get_insights()
    print(f"\n5. Key Insights:")
    for category, items in insights.items():
        if items:
            print(f"\n   {category.replace('_', ' ').title()}:")
            for item in items[:3]:  # Show first 3
                print(f"   • {item}")


def example_3_custom_configuration():
    """Example 3: Custom configuration using kwargs."""
    print("\n" + "=" * 70)
    print("Example 3: Custom Configuration Using Kwargs")
    print("=" * 70)

    df = create_sample_data()

    # Configure using kwargs (most convenient way)
    eda = SimplusEDA(
        correlation_threshold=0.6,
        outlier_method='zscore',
        missing_threshold=0.05,
        verbose=True
    )
    print("\n1. Created EDA with custom configuration:")
    print(f"   - Correlation threshold: 0.6")
    print(f"   - Outlier method: zscore")
    print(f"   - Missing threshold: 5%")

    # Analyze
    results = eda.analyze(df)
    print(f"\n2. Analysis completed")

    # Get quality score
    quality_score = eda.get_quality_score()
    print(f"\n3. Data Quality Score: {quality_score:.1f}%")

    # Get correlations
    correlations = eda.get_correlations()
    print(f"\n4. Found {len(correlations)} strong correlations")
    for corr in correlations[:3]:
        print(f"   {corr.get('feature1')} <-> {corr.get('feature2')}: "
              f"{corr.get('correlation', 0):.3f}")

    # Generate customized report
    report_path = eda.generate_report(
        'outputs/example3_custom_report.html',
        title='Custom Configuration Example',
        author='Data Science Team',
        company='Example Corp'
    )
    print(f"\n5. Generated customized report: {report_path}")


def example_4_config_object():
    """Example 4: Using EDAConfig object for full control."""
    print("\n" + "=" * 70)
    print("Example 4: Using EDAConfig Object for Full Control")
    print("=" * 70)

    df = create_sample_data()

    # Create detailed configuration
    config = EDAConfig(
        # Analysis settings
        enable_statistical_tests=True,
        enable_visualizations=True,

        # Thresholds
        correlation_threshold=0.7,
        missing_threshold=0.1,
        significance_level=0.05,

        # Methods
        outlier_method='isolation_forest',
        distribution_test_method='shapiro',

        # Performance
        n_samples_viz=10000,
        n_jobs=1,
        verbose=True
    )

    print("\n1. Created detailed EDAConfig:")
    print(config.get_summary())

    # Create EDA with config
    eda = SimplusEDA(config=config)

    # Analyze
    results = eda.analyze(df)
    print(f"\n2. Analysis completed with custom configuration")

    # Generate report
    report_path = eda.generate_report(
        'outputs/example4_config_report.html',
        title='Advanced Configuration Example'
    )
    print(f"\n3. Generated report: {report_path}")

    # Save configuration for reuse
    config_path = 'outputs/my_config.json'
    eda.save_config(config_path)
    print(f"\n4. Saved configuration to: {config_path}")


def example_5_config_from_file():
    """Example 5: Loading configuration from file."""
    print("\n" + "=" * 70)
    print("Example 5: Loading Configuration from File")
    print("=" * 70)

    df = create_sample_data()

    # Load configuration from previous example
    config_path = 'outputs/my_config.json'

    try:
        eda = SimplusEDA.from_config_file(config_path)
        print(f"\n1. Loaded configuration from: {config_path}")

        # Analyze
        results = eda.analyze(df)
        print(f"\n2. Analysis completed using loaded configuration")

        # Generate report
        report_path = eda.generate_report('outputs/example5_from_file_report.html')
        print(f"\n3. Generated report: {report_path}")

    except FileNotFoundError:
        print(f"\n⚠️  Config file not found. Run Example 4 first to create it.")


def example_6_update_config():
    """Example 6: Updating configuration dynamically."""
    print("\n" + "=" * 70)
    print("Example 6: Updating Configuration Dynamically")
    print("=" * 70)

    df = create_sample_data()

    # Start with default config
    eda = SimplusEDA()
    print("\n1. Started with default configuration")

    # Analyze with defaults
    results1 = eda.analyze(df)
    quality1 = eda.get_quality_score()
    print(f"2. First analysis - Quality score: {quality1:.1f}%")

    # Update configuration
    eda.update_config(
        correlation_threshold=0.9,
        outlier_method='zscore'
    )
    print("\n3. Updated configuration:")
    print("   - Correlation threshold: 0.9")
    print("   - Outlier method: zscore")

    # Re-analyze with new config
    results2 = eda.analyze(df)
    correlations = eda.get_correlations(threshold=0.9)
    print(f"\n4. Re-analyzed with new config")
    print(f"   Found {len(correlations)} correlations above 0.9 threshold")

    # Generate report
    report_path = eda.generate_report('outputs/example6_updated_config_report.html')
    print(f"\n5. Generated report: {report_path}")


def example_7_multiple_formats():
    """Example 7: Generating reports in multiple formats."""
    print("\n" + "=" * 70)
    print("Example 7: Generating Reports in Multiple Formats")
    print("=" * 70)

    df = create_sample_data()

    # Create and analyze
    eda = SimplusEDA(correlation_threshold=0.7)
    results = eda.analyze(df)
    print("\n1. Analysis completed")

    # Generate HTML report
    html_path = eda.generate_report(
        'outputs/example7_report.html',
        format='html',
        title='Multi-Format Example'
    )
    print(f"\n2. Generated HTML report: {html_path}")

    # Generate JSON report
    json_path = eda.generate_report(
        'outputs/example7_report.json',
        format='json'
    )
    print(f"3. Generated JSON report: {json_path}")

    # Export results (convenience method)
    export_path = eda.export_results('outputs/example7_results.json')
    print(f"4. Exported results: {export_path}")

    # Try PDF (will warn if weasyprint not installed)
    try:
        pdf_path = eda.generate_report(
            'outputs/example7_report.pdf',
            format='pdf',
            title='Multi-Format Example'
        )
        print(f"5. Generated PDF report: {pdf_path}")
    except ImportError as e:
        print(f"5. PDF generation skipped: {e}")


def example_8_convenience_function():
    """Example 8: Using the analyze_and_report convenience function."""
    print("\n" + "=" * 70)
    print("Example 8: Using Convenience Function")
    print("=" * 70)

    df = create_sample_data()

    # One function call that returns both results and report path
    results, report_path = analyze_and_report(
        df,
        'outputs/example8_convenience_report.html',
        title='Convenience Function Example',
        author='Automated System'
    )

    print(f"\n1. Analysis and report completed in one call")
    print(f"   Report: {report_path}")

    # Access results directly
    quality_score = results['quality']['quality_score']['overall_score']
    print(f"\n2. Quality Score: {quality_score:.1f}%")

    insights = results.get('insights', {})
    recommendations = insights.get('recommendations', [])
    print(f"\n3. Recommendations ({len(recommendations)}):")
    for rec in recommendations[:3]:
        print(f"   • {rec}")


def example_9_method_chaining():
    """Example 9: Method chaining for fluent API."""
    print("\n" + "=" * 70)
    print("Example 9: Method Chaining")
    print("=" * 70)

    df = create_sample_data()

    # Fluent API with method chaining
    eda = (SimplusEDA()
           .update_config(correlation_threshold=0.8, verbose=True))

    print("\n1. Created and configured EDA with method chaining")

    # Analyze
    eda.analyze(df)
    print(f"2. Analysis completed")
    print(f"   Quality Score: {eda.get_quality_score():.1f}%")

    # Generate report
    report_path = eda.generate_report('outputs/example9_chaining_report.html')
    print(f"\n3. Generated report: {report_path}")


def example_10_accessing_components():
    """Example 10: Accessing underlying components for advanced usage."""
    print("\n" + "=" * 70)
    print("Example 10: Accessing Underlying Components")
    print("=" * 70)

    df = create_sample_data()

    # Create EDA
    eda = SimplusEDA()

    # Access the underlying analyzer
    analyzer = eda.analyzer
    print(f"\n1. Accessed underlying EDAAnalyzer: {type(analyzer).__name__}")

    # Access the config
    config = eda.config
    print(f"\n2. Current correlation threshold: {config.correlation_threshold}")

    # Perform analysis
    results = eda.analyze(df)

    # Access the report generator
    report_gen = eda.report_generator
    print(f"\n3. Accessed ReportGenerator: {type(report_gen).__name__}")

    # You can use components directly if needed
    # For example, generate a custom visualization
    print(f"\n4. Components are accessible for advanced customization")
    print(f"   - EDAAnalyzer for custom analysis")
    print(f"   - ReportGenerator for custom reports")
    print(f"   - EDAConfig for configuration management")


def main():
    """Run all examples."""
    import os
    os.makedirs('outputs', exist_ok=True)

    print("\n" + "=" * 70)
    print("Simplus EDA - Unified API Examples")
    print("=" * 70)
    print("\nThese examples demonstrate how config, analyzer, and report")
    print("modules work seamlessly together through the SimplusEDA API.\n")

    examples = [
        example_1_simplest_usage,
        example_2_basic_workflow,
        example_3_custom_configuration,
        example_4_config_object,
        example_5_config_from_file,
        example_6_update_config,
        example_7_multiple_formats,
        example_8_convenience_function,
        example_9_method_chaining,
        example_10_accessing_components,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n⚠️  Error in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All Examples Completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. SimplusEDA provides a unified, easy-to-use API")
    print("2. Config, Analyzer, and Report work seamlessly together")
    print("3. Multiple usage patterns from simple to advanced")
    print("4. Flexible configuration management")
    print("5. Consistent behavior across all components")
    print("\nGenerated reports are in the 'outputs/' directory.")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
