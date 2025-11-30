"""
Example 1: Quick Start Examples
================================

This script demonstrates the simplest ways to use the Simplus EDA framework,
from one-line analysis to basic customization.

Run this example:
    python examples/01_quick_start_examples.py
"""

import pandas as pd
from simplus_eda import SimplusEDA, quick_analysis
import sys
import os

# Add examples directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from data_generators import generate_sales_data


def example_1_one_liner():
    """Example 1: Absolute simplest - one line of code"""
    print("\n" + "="*80)
    print("Example 1: One-Line Analysis")
    print("="*80)

    # Generate sample data
    df = generate_sales_data(n_samples=500)
    print(f"\nGenerated sales dataset: {len(df)} rows, {len(df.columns)} columns")

    # ONE LINE - that's it!
    report_path = quick_analysis(df, 'outputs/quick_analysis_report.html')

    print(f"\n✓ Analysis complete! Report saved to: {report_path}")
    print("  Open this file in your browser to view the comprehensive report.")


def example_2_basic_workflow():
    """Example 2: Basic workflow with results access"""
    print("\n" + "="*80)
    print("Example 2: Basic Workflow")
    print("="*80)

    # Generate data
    df = generate_sales_data(n_samples=500)

    # Create EDA instance
    eda = SimplusEDA()

    # Analyze the data
    print("\nAnalyzing dataset...")
    results = eda.analyze(df)

    # Access key results
    print("\n--- Analysis Results ---")
    print(f"Data Quality Score: {eda.get_quality_score():.1f}%")

    # Get quality insights
    insights = eda.get_insights()
    if 'quality_issues' in insights and insights['quality_issues']:
        print("\nQuality Issues Found:")
        for issue in insights['quality_issues'][:3]:
            print(f"  • {issue}")

    if 'recommendations' in insights and insights['recommendations']:
        print("\nRecommendations:")
        for rec in insights['recommendations'][:3]:
            print(f"  • {rec}")

    # Generate report
    print("\nGenerating report...")
    eda.generate_report('outputs/basic_workflow_report.html')
    print("✓ Report saved to: outputs/basic_workflow_report.html")


def example_3_view_summary():
    """Example 3: Quick summary without generating full report"""
    print("\n" + "="*80)
    print("Example 3: Quick Summary")
    print("="*80)

    df = generate_sales_data(n_samples=500)

    eda = SimplusEDA()
    eda.analyze(df)

    # Print text summary
    print("\n" + eda.summary())


def example_4_basic_customization():
    """Example 4: Basic configuration options"""
    print("\n" + "="*80)
    print("Example 4: Basic Customization")
    print("="*80)

    df = generate_sales_data(n_samples=500)

    # Customize the analysis
    eda = SimplusEDA(
        correlation_threshold=0.7,  # Lower threshold to detect more correlations
        outlier_method='isolation_forest',  # Use advanced outlier detection
        verbose=True  # Show progress messages
    )

    print("\nAnalyzing with custom configuration...")
    eda.analyze(df)

    # View correlations
    correlations = eda.get_correlations()
    if correlations:
        print(f"\n--- Strong Correlations Found (>{0.7}) ---")
        for corr in correlations[:5]:
            print(f"  {corr['feature1']:20s} <-> {corr['feature2']:20s}: {corr['correlation']:6.3f}")

    # Generate customized report
    eda.generate_report(
        'outputs/customized_report.html',
        title='Sales Data Analysis - Customized',
        author='Data Science Team',
        include_data_preview=True
    )
    print("\n✓ Customized report saved!")


def example_5_export_json():
    """Example 5: Export results as JSON for API/programmatic use"""
    print("\n" + "="*80)
    print("Example 5: JSON Export")
    print("="*80)

    df = generate_sales_data(n_samples=500)

    eda = SimplusEDA()
    eda.analyze(df)

    # Export as JSON
    json_path = eda.export_results('outputs/analysis_results.json')
    print(f"\n✓ Results exported to JSON: {json_path}")
    print("  This JSON file contains all analysis results in a machine-readable format.")

    # You can also generate JSON report directly
    eda.generate_report('outputs/results.json', format='json')
    print("✓ JSON report generated: outputs/results.json")


def example_6_multiple_formats():
    """Example 6: Generate reports in multiple formats"""
    print("\n" + "="*80)
    print("Example 6: Multiple Report Formats")
    print("="*80)

    df = generate_sales_data(n_samples=500)

    eda = SimplusEDA()
    eda.analyze(df)

    # Generate multiple formats
    print("\nGenerating reports in multiple formats...")

    # HTML report (interactive, best for viewing)
    eda.generate_report('outputs/report.html', format='html')
    print("✓ HTML report: outputs/report.html")

    # JSON report (best for APIs)
    eda.generate_report('outputs/report.json', format='json')
    print("✓ JSON report: outputs/report.json")

    # Note: PDF requires weasyprint: pip install weasyprint
    # eda.generate_report('outputs/report.pdf', format='pdf')


def run_all_examples():
    """Run all quick start examples"""
    print("\n" + "="*80)
    print("SIMPLUS EDA FRAMEWORK - QUICK START EXAMPLES")
    print("="*80)

    # Create output directory
    os.makedirs('outputs', exist_ok=True)

    try:
        example_1_one_liner()
        example_2_basic_workflow()
        example_3_view_summary()
        example_4_basic_customization()
        example_5_export_json()
        example_6_multiple_formats()

        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Check the outputs/ folder for generated reports")
        print("  2. Open the HTML reports in your browser")
        print("  3. Try the other example scripts for advanced features")

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_examples()
