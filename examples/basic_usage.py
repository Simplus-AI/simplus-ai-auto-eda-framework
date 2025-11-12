"""
Basic usage example for Simplus EDA Framework.

This example demonstrates the simplest way to use the framework
for automated exploratory data analysis.
"""

import pandas as pd
from simplus_eda import EDAAnalyzer, ReportGenerator

# Create sample data
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    'income': [30000, 45000, 55000, 65000, 75000, 85000, 95000, 105000, 115000, 125000],
    'years_experience': [2, 5, 8, 12, 15, 20, 25, 30, 35, 40],
    'satisfaction': [7, 8, 7, 9, 8, 9, 10, 9, 8, 10],
    'department': ['Sales', 'IT', 'Sales', 'IT', 'HR', 'Sales', 'IT', 'HR', 'Sales', 'IT']
})

def main():
    """Run basic EDA analysis."""
    print("Starting Simplus EDA Framework - Basic Usage Example")
    print("=" * 60)

    # Initialize the analyzer
    analyzer = EDAAnalyzer()

    # Perform automated EDA
    print("\nPerforming automated EDA...")
    results = analyzer.analyze(data)

    # Display summary
    print("\nAnalysis completed!")
    print(f"Dataset shape: {data.shape}")
    print(f"Number of features: {data.shape[1]}")
    print(f"Number of records: {data.shape[0]}")

    # Generate reports
    print("\nGenerating reports...")
    report_gen = ReportGenerator()

    # Generate JSON report
    report_gen.generate_json(results, "output/basic_eda_report.json")
    print("✓ JSON report generated: output/basic_eda_report.json")

    # Generate HTML report
    report_gen.generate_html(results, "output/basic_eda_report.html")
    print("✓ HTML report generated: output/basic_eda_report.html")

    print("\n" + "=" * 60)
    print("Analysis complete! Check the output directory for reports.")


if __name__ == "__main__":
    main()
