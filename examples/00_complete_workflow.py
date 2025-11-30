"""
Example 0: Complete End-to-End Workflow
========================================

This example demonstrates a complete data analysis workflow from data generation
through analysis, insights extraction, and reporting.

This is the perfect starting point to understand how all pieces fit together.

Run this example:
    python examples/00_complete_workflow.py
"""

import pandas as pd
import numpy as np
from simplus_eda import SimplusEDA, EDAConfig
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from data_generators import generate_sales_data


def complete_workflow():
    """Complete end-to-end EDA workflow"""

    print("\n" + "="*80)
    print("COMPLETE END-TO-END DATA ANALYSIS WORKFLOW")
    print("="*80)

    # =========================================================================
    # STEP 1: DATA GENERATION
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 1: GENERATE SAMPLE DATA")
    print("-"*80)

    print("\nGenerating realistic e-commerce sales dataset...")
    df = generate_sales_data(n_samples=2000, random_state=42)

    print(f"\n✓ Generated dataset with {len(df)} rows and {len(df.columns)} columns")
    print("\nFirst few rows:")
    print(df.head())

    print("\nDataset info:")
    print(f"  • Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  • Categories: {df['category'].nunique()} unique")
    print(f"  • Regions: {df['region'].nunique()} unique")
    print(f"  • Total revenue: ${df['revenue'].sum():,.2f}")

    # =========================================================================
    # STEP 2: INITIAL EXPLORATION
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 2: INITIAL DATA EXPLORATION")
    print("-"*80)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])

    print("\nBasic statistics:")
    print(df.describe())

    # =========================================================================
    # STEP 3: CONFIGURE EDA
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 3: CONFIGURE EDA ANALYSIS")
    print("-"*80)

    print("\nSetting up comprehensive analysis configuration...")

    # Create custom configuration
    config = EDAConfig(
        # Analysis settings
        enable_statistical_tests=True,
        enable_visualizations=True,

        # Thresholds
        correlation_threshold=0.6,      # Detect correlations > 0.6
        missing_threshold=0.1,          # Flag if > 10% missing
        significance_level=0.05,        # 95% confidence level

        # Methods
        outlier_method='isolation_forest',  # Advanced outlier detection
        distribution_test_method='shapiro',

        # Performance
        n_jobs=-1,                      # Use all CPU cores
        verbose=True,                   # Show progress

        # Reproducibility
        random_state=42
    )

    print("\n✓ Configuration created:")
    print(f"  • Correlation threshold: {config.correlation_threshold}")
    print(f"  • Outlier method: {config.outlier_method}")
    print(f"  • Missing threshold: {config.missing_threshold}")
    print(f"  • Parallel processing: {config.n_jobs} cores")

    # =========================================================================
    # STEP 4: RUN ANALYSIS
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 4: RUN COMPREHENSIVE ANALYSIS")
    print("-"*80)

    # Create EDA instance
    eda = SimplusEDA(config=config)

    print("\nRunning analysis...")
    results = eda.analyze(df)

    print("\n✓ Analysis complete!")

    # =========================================================================
    # STEP 5: EXAMINE RESULTS
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 5: EXAMINE ANALYSIS RESULTS")
    print("-"*80)

    # Data quality score
    quality_score = eda.get_quality_score()
    print(f"\n📊 DATA QUALITY SCORE: {quality_score:.1f}/100")

    if quality_score >= 90:
        print("   Status: Excellent ✓")
    elif quality_score >= 75:
        print("   Status: Good ✓")
    elif quality_score >= 60:
        print("   Status: Fair ⚠️")
    else:
        print("   Status: Poor ✗")

    # View summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(eda.summary())

    # Correlations
    print("\n" + "="*80)
    print("STRONG CORRELATIONS")
    print("="*80)

    correlations = eda.get_correlations(threshold=0.6)
    if correlations:
        print(f"\nFound {len(correlations)} strong correlations:")
        for i, corr in enumerate(correlations[:10], 1):
            print(f"{i:2d}. {corr['feature1']:20s} <-> {corr['feature2']:20s}: "
                  f"{corr['correlation']:6.3f} ({corr['correlation_type']})")
    else:
        print("No strong correlations found above threshold.")

    # Insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    insights = eda.get_insights()

    for category, items in insights.items():
        if items:
            print(f"\n{category.replace('_', ' ').title()}:")
            for item in items[:5]:  # Show top 5 per category
                print(f"  • {item}")

    # =========================================================================
    # STEP 6: BUSINESS ANALYSIS
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 6: BUSINESS-SPECIFIC ANALYSIS")
    print("-"*80)

    print("\n📈 Revenue Analysis by Category:")
    category_analysis = df.groupby('category').agg({
        'revenue': ['sum', 'mean', 'count'],
        'quantity': 'sum',
        'discount': 'mean'
    }).round(2)
    category_analysis.columns = ['Total Revenue', 'Avg Revenue', 'Transactions',
                                  'Total Units', 'Avg Discount']
    category_analysis = category_analysis.sort_values('Total Revenue', ascending=False)
    print(category_analysis)

    print("\n📍 Revenue Analysis by Region:")
    region_analysis = df.groupby('region').agg({
        'revenue': ['sum', 'mean', 'count']
    }).round(2)
    region_analysis.columns = ['Total Revenue', 'Avg Revenue', 'Transactions']
    region_analysis = region_analysis.sort_values('Total Revenue', ascending=False)
    print(region_analysis)

    print("\n👥 Customer Segment Analysis:")
    segment_analysis = df.groupby('customer_segment').agg({
        'revenue': ['sum', 'mean'],
        'quantity': 'mean',
        'customer_age': 'mean'
    }).round(2)
    segment_analysis.columns = ['Total Revenue', 'Avg Revenue', 'Avg Quantity', 'Avg Age']
    print(segment_analysis)

    # =========================================================================
    # STEP 7: GENERATE REPORTS
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 7: GENERATE REPORTS")
    print("-"*80)

    # Create output directory
    os.makedirs('outputs', exist_ok=True)

    print("\nGenerating comprehensive HTML report...")
    html_report = eda.generate_report(
        'outputs/complete_workflow_report.html',
        format='html',
        title='E-Commerce Sales Analysis - Complete Workflow',
        author='Data Science Team',
        company='Your Company',
        include_visualizations=True,
        include_data_preview=True,
        max_preview_rows=20
    )
    print(f"✓ HTML Report: {html_report}")

    print("\nGenerating JSON export for API/pipeline...")
    json_export = eda.export_results('outputs/complete_workflow_results.json')
    print(f"✓ JSON Export: {json_export}")

    # =========================================================================
    # STEP 8: SAVE CONFIGURATION
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 8: SAVE CONFIGURATION FOR REUSE")
    print("-"*80)

    config_path = 'outputs/analysis_config.json'
    eda.save_config(config_path)
    print(f"\n✓ Configuration saved to: {config_path}")
    print("  This configuration can be reused for consistent analysis:")
    print("  >>> eda = SimplusEDA.from_config_file('outputs/analysis_config.json')")

    # =========================================================================
    # STEP 9: ACTIONABLE RECOMMENDATIONS
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 9: ACTIONABLE RECOMMENDATIONS")
    print("="*80)

    print("\n🎯 Based on the analysis, here are actionable recommendations:")

    # Data quality recommendations
    if quality_score < 85:
        print("\n📋 Data Quality Improvements:")
        print("  1. Address missing values in discount column (15%)")
        print("  2. Address missing values in customer_age column (10%)")
        print("  3. Investigate and handle duplicate records (2%)")
        print("  4. Review outlier transactions for fraud detection")

    # Business recommendations
    print("\n💼 Business Insights:")
    top_category = category_analysis.index[0]
    top_revenue = category_analysis['Total Revenue'].iloc[0]
    print(f"  1. Focus inventory on top category: {top_category} (${top_revenue:,.2f})")

    top_region = region_analysis.index[0]
    print(f"  2. Expand operations in top region: {top_region}")

    print("  3. Implement targeted discounting strategy based on segment performance")

    premium_avg = segment_analysis.loc['Premium', 'Avg Revenue']
    standard_avg = segment_analysis.loc['Standard', 'Avg Revenue']
    print(f"  4. Premium customers spend {premium_avg/standard_avg:.1f}x more - create loyalty program")

    # Technical recommendations
    print("\n🔧 Technical Next Steps:")
    print("  1. Build predictive model for revenue forecasting")
    print("  2. Implement real-time anomaly detection for fraud")
    print("  3. Create automated daily data quality monitoring")
    print("  4. Set up A/B testing for discount optimization")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE!")
    print("="*80)

    print("\n📁 Generated Files:")
    print("  • outputs/complete_workflow_report.html - Interactive HTML report")
    print("  • outputs/complete_workflow_results.json - Machine-readable results")
    print("  • outputs/analysis_config.json - Reusable configuration")

    print("\n📖 What We Learned:")
    print("  ✓ How to generate realistic test data")
    print("  ✓ How to configure comprehensive analysis")
    print("  ✓ How to interpret data quality scores")
    print("  ✓ How to extract business insights")
    print("  ✓ How to generate professional reports")
    print("  ✓ How to save configurations for reuse")

    print("\n🚀 Next Steps:")
    print("  1. Open outputs/complete_workflow_report.html in your browser")
    print("  2. Review the visualizations and detailed findings")
    print("  3. Try the other example scripts:")
    print("     - 01_quick_start_examples.py - Quick start patterns")
    print("     - 02_advanced_analysis_examples.py - Advanced features")
    print("     - 03_real_world_use_cases.py - Industry-specific examples")
    print("  4. Apply this workflow to your own datasets!")

    print("\n" + "="*80)
    print("Thank you for trying Simplus EDA Framework!")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        complete_workflow()
    except Exception as e:
        print(f"\n✗ Error during workflow: {e}")
        import traceback
        traceback.print_exc()
