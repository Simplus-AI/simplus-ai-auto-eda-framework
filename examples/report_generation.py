"""
Example: Comprehensive Report Generation

This example demonstrates how to generate comprehensive EDA reports
in multiple formats (HTML, JSON, PDF) using the ReportGenerator.
"""

import pandas as pd
import numpy as np
from simplus_eda.core.analyzer import EDAAnalyzer
from simplus_eda.core.report import ReportGenerator
from simplus_eda.core.config import EDAConfig


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    np.random.seed(42)

    n_samples = 1000

    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(35, 12, n_samples).astype(int),
        'income': np.random.exponential(50000, n_samples),
        'credit_score': np.random.normal(680, 80, n_samples),
        'loan_amount': np.random.uniform(5000, 100000, n_samples),
        'employment_years': np.random.poisson(8, n_samples),
        'num_dependents': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.2, 0.3, 0.25, 0.15, 0.1]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'loan_status': np.random.choice(['Approved', 'Rejected', 'Pending'], n_samples, p=[0.6, 0.25, 0.15]),
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H')
    }

    df = pd.DataFrame(data)

    # Add some missing values
    df.loc[df.sample(frac=0.05).index, 'credit_score'] = np.nan
    df.loc[df.sample(frac=0.03).index, 'income'] = np.nan

    # Add some outliers
    df.loc[df.sample(n=20).index, 'income'] = df['income'].quantile(0.95) * 3

    # Create correlation (loan_amount depends on income)
    df['loan_amount'] = df['income'] * 0.5 + np.random.normal(0, 10000, n_samples)
    df.loc[df['loan_amount'] < 5000, 'loan_amount'] = 5000

    return df


def example_basic_html_report():
    """Example 1: Generate a basic HTML report."""
    print("=" * 70)
    print("Example 1: Basic HTML Report")
    print("=" * 70)

    # Create sample data
    df = create_sample_dataset()
    print(f"\nDataset created: {df.shape[0]} rows, {df.shape[1]} columns")

    # Perform EDA analysis
    print("\nPerforming EDA analysis...")
    analyzer = EDAAnalyzer()
    results = analyzer.analyze(df)

    # Generate report
    print("\nGenerating HTML report...")
    report_gen = ReportGenerator()
    output_path = report_gen.generate_report(
        results=results,
        data=df,
        output_path='outputs/basic_report.html',
        format='html',
        include_visualizations=True,
        include_data_preview=True
    )

    print(f"\n✓ Report generated successfully!")
    print(f"  Location: {output_path}")
    print(f"  Open this file in a web browser to view the report.")


def example_customized_report():
    """Example 2: Generate a customized report with configuration."""
    print("\n" + "=" * 70)
    print("Example 2: Customized Report with Configuration")
    print("=" * 70)

    # Create sample data
    df = create_sample_dataset()

    # Configure analysis
    config = EDAConfig(
        correlation_threshold=0.6,
        outlier_method='zscore',
        missing_threshold=0.05,
        verbose=True
    )

    # Perform analysis with custom config
    print("\nPerforming EDA analysis with custom configuration...")
    analyzer = EDAAnalyzer(config=config)
    results = analyzer.analyze(df)

    # Configure report
    report_config = {
        'title': 'Loan Application Analysis Report',
        'author': 'Data Science Team',
        'company': 'Financial Services Inc.',
        'include_toc': True
    }

    # Generate customized report
    print("\nGenerating customized HTML report...")
    report_gen = ReportGenerator(config=report_config)
    output_path = report_gen.generate_report(
        results=results,
        data=df,
        output_path='outputs/customized_report.html',
        format='html',
        include_visualizations=True
    )

    print(f"\n✓ Customized report generated successfully!")
    print(f"  Location: {output_path}")


def example_json_report():
    """Example 3: Generate a JSON report for programmatic access."""
    print("\n" + "=" * 70)
    print("Example 3: JSON Report for Programmatic Access")
    print("=" * 70)

    # Create sample data
    df = create_sample_dataset()

    # Perform analysis
    print("\nPerforming EDA analysis...")
    analyzer = EDAAnalyzer()
    results = analyzer.analyze(df)

    # Generate JSON report
    print("\nGenerating JSON report...")
    report_gen = ReportGenerator()
    output_path = report_gen.generate_json(
        results=results,
        output_path='outputs/analysis_results.json'
    )

    print(f"\n✓ JSON report generated successfully!")
    print(f"  Location: {output_path}")
    print(f"  This format is ideal for:")
    print(f"  - API responses")
    print(f"  - Integration with other tools")
    print(f"  - Version control and diffing")


def example_multiple_formats():
    """Example 4: Generate reports in multiple formats."""
    print("\n" + "=" * 70)
    print("Example 4: Generate Reports in Multiple Formats")
    print("=" * 70)

    # Create sample data
    df = create_sample_dataset()

    # Perform analysis once
    print("\nPerforming EDA analysis...")
    analyzer = EDAAnalyzer()
    results = analyzer.analyze(df)

    # Initialize report generator
    report_gen = ReportGenerator(config={
        'title': 'Multi-Format EDA Report',
        'author': 'Automated Analysis System'
    })

    # Generate HTML report
    print("\n1. Generating HTML report...")
    html_path = report_gen.generate_report(
        results=results,
        data=df,
        output_path='outputs/multi_format_report.html',
        format='html',
        include_visualizations=True
    )
    print(f"   ✓ HTML: {html_path}")

    # Generate JSON report
    print("2. Generating JSON report...")
    json_path = report_gen.generate_report(
        results=results,
        data=df,
        output_path='outputs/multi_format_report.json',
        format='json'
    )
    print(f"   ✓ JSON: {json_path}")

    # Generate PDF report (requires weasyprint)
    print("3. Attempting to generate PDF report...")
    try:
        pdf_path = report_gen.generate_report(
            results=results,
            data=df,
            output_path='outputs/multi_format_report.pdf',
            format='pdf',
            include_visualizations=True
        )
        print(f"   ✓ PDF: {pdf_path}")
    except ImportError as e:
        print(f"   ⚠ PDF generation skipped: {e}")
        print(f"   Install weasyprint to enable PDF reports: pip install weasyprint")

    print(f"\n✓ All reports generated successfully!")


def example_summary_insights():
    """Example 5: Display analysis summary and insights."""
    print("\n" + "=" * 70)
    print("Example 5: Analysis Summary and Key Insights")
    print("=" * 70)

    # Create sample data
    df = create_sample_dataset()

    # Perform analysis
    print("\nPerforming EDA analysis...")
    analyzer = EDAAnalyzer()
    results = analyzer.analyze(df)

    # Display summary
    print("\n" + "-" * 70)
    print(analyzer.get_summary())
    print("-" * 70)

    # Display insights
    insights = results.get('insights', {})

    if insights.get('data_characteristics'):
        print("\n📊 Data Characteristics:")
        for insight in insights['data_characteristics']:
            print(f"  • {insight}")

    if insights.get('quality_issues'):
        print("\n⚠️  Quality Issues:")
        for insight in insights['quality_issues']:
            print(f"  • {insight}")

    if insights.get('statistical_findings'):
        print("\n📈 Statistical Findings:")
        for insight in insights['statistical_findings']:
            print(f"  • {insight}")

    if insights.get('recommendations'):
        print("\n💡 Recommendations:")
        for insight in insights['recommendations']:
            print(f"  • {insight}")

    # Generate report
    print("\n\nGenerating comprehensive report...")
    report_gen = ReportGenerator()
    output_path = report_gen.generate_report(
        results=results,
        data=df,
        output_path='outputs/insights_report.html',
        format='html',
        include_visualizations=True
    )

    print(f"\n✓ Report with insights generated: {output_path}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Simplus EDA Framework - Report Generation Examples")
    print("=" * 70)

    # Create outputs directory
    import os
    os.makedirs('outputs', exist_ok=True)

    # Run examples
    example_basic_html_report()
    example_customized_report()
    example_json_report()
    example_multiple_formats()
    example_summary_insights()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nGenerated files can be found in the 'outputs/' directory.")
    print("Open the HTML files in a web browser to view the reports.")
    print("\nNext steps:")
    print("  1. Open the HTML reports in your browser")
    print("  2. Review the JSON output for API integration")
    print("  3. Customize the reports with your own configuration")
    print("  4. Integrate with your data pipeline")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
