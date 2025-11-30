"""
Example 2: Advanced Analysis Features
======================================

This script demonstrates the advanced analytical capabilities of the framework:
- Statistical tests and hypothesis testing
- Time series analysis
- Feature engineering suggestions
- Anomaly detection

Run this example:
    python examples/02_advanced_analysis_examples.py
"""

import pandas as pd
import numpy as np
from simplus_eda import (
    SimplusEDA,
    StatisticalTestsManager,
    TimeSeriesAnalyzer,
    FeatureEngineeringManager,
    AnomalyDetectionManager
)
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from data_generators import (
    generate_customer_churn_data,
    generate_time_series_sales,
    generate_financial_data,
    generate_sensor_data
)


def example_1_statistical_tests():
    """Example 1: Advanced statistical hypothesis testing"""
    print("\n" + "="*80)
    print("Example 1: Statistical Hypothesis Testing")
    print("="*80)

    # Generate data with groups to compare
    df = generate_customer_churn_data(n_samples=1000)

    # Create test manager
    test_mgr = StatisticalTestsManager()

    print("\n--- Comparing Groups ---")
    print("Question: Do monthly charges differ across contract types?")

    # Compare monthly charges across contract types
    result = test_mgr.compare_groups(
        df,
        value_col='monthly_charges',
        group_col='contract_type',
        parametric=None,  # Auto-detect if data is parametric
        post_hoc=True  # Run post-hoc tests if significant
    )

    print(f"\nTest Used: {result['test_used']}")
    print(f"Test Statistic: {result['test_result']['statistic']:.4f}")
    print(f"P-value: {result['test_result']['p_value']:.6f}")
    print(f"Significant: {'Yes' if result['significant'] else 'No'}")

    if result.get('post_hoc_results'):
        print("\nPost-hoc Test Results:")
        for comparison in result['post_hoc_results']['pairwise_comparisons'][:3]:
            print(f"  {comparison['group1']} vs {comparison['group2']}: "
                  f"p={comparison['p_value']:.4f} ({'significant' if comparison['significant'] else 'not significant'})")

    # Test categorical association
    print("\n--- Testing Categorical Association ---")
    print("Question: Is there an association between contract type and churn?")

    chi_result = test_mgr.test_categorical_association(
        df,
        var1='contract_type',
        var2='churned',
        method='auto'
    )

    print(f"Test Used: {chi_result['test_used']}")
    print(f"Chi-square Statistic: {chi_result['test_result']['statistic']:.4f}")
    print(f"P-value: {chi_result['test_result']['p_value']:.6f}")
    print(f"Effect Size (Cramér's V): {chi_result['effect_size']['cramers_v']:.4f}")
    print(f"Association: {'Significant' if chi_result['significant'] else 'Not significant'}")


def example_2_time_series_analysis():
    """Example 2: Comprehensive time series analysis"""
    print("\n" + "="*80)
    print("Example 2: Time Series Analysis")
    print("="*80)

    # Generate time series data
    df = generate_time_series_sales(n_days=365)

    # Filter for one category
    electronics = df[df['category'] == 'Electronics'].copy()
    series = electronics.set_index('date')['sales']

    print(f"\nAnalyzing time series: {len(series)} daily observations")

    # Create analyzer with full capabilities
    ts_analyzer = TimeSeriesAnalyzer(
        enable_forecasting=True,
        enable_decomposition=True,
        enable_acf_pacf=True,
        forecast_steps=30  # Forecast next 30 days
    )

    print("\nRunning time series analysis...")
    results = ts_analyzer.analyze(series, seasonal_period=7)  # Weekly seasonality

    # Display results
    print("\n--- Stationarity Tests ---")
    print(f"ADF Test p-value: {results['stationarity']['adf']['p_value']:.6f}")
    print(f"Series is {'stationary' if results['stationarity']['adf']['is_stationary'] else 'non-stationary'}")

    print("\n--- Trend Analysis ---")
    trend_info = results['trend']['mann_kendall']
    print(f"Trend: {trend_info['trend']}")
    print(f"p-value: {trend_info['p_value']:.6f}")

    if 'seasonal' in results:
        print("\n--- Seasonality ---")
        print(f"Seasonal Strength: {results['seasonality']['seasonal_strength']:.3f}")
        print(f"Trend Strength: {results['seasonality']['trend_strength']:.3f}")

    if 'forecasting' in results and results['forecasting']['forecast']:
        print("\n--- Forecast ---")
        forecast = results['forecasting']['forecast']['forecast']
        print(f"Next 30 days forecast (first 7 days):")
        for i, value in enumerate(forecast[:7], 1):
            print(f"  Day {i}: ${value:,.2f}")

    print("\n✓ Time series analysis complete!")
    print("  Use SimplusEDA to generate a full report with visualizations")


def example_3_feature_engineering():
    """Example 3: Automated feature engineering suggestions"""
    print("\n" + "="*80)
    print("Example 3: Feature Engineering Suggestions")
    print("="*80)

    # Generate data
    df = generate_customer_churn_data(n_samples=1000)

    # Create feature engineering manager
    fe_mgr = FeatureEngineeringManager(
        detect_interactions=True,
        detect_polynomials=True,
        suggest_binning=True,
        suggest_encoding=True,
        suggest_scaling=True
    )

    print("\nAnalyzing features and generating suggestions...")
    results = fe_mgr.analyze(df, target_col='churned')

    # Display interaction suggestions
    if results['interactions']:
        print("\n--- Top Feature Interactions ---")
        print("(Features that may work well when combined)")
        for interaction in results['interactions'][:5]:
            print(f"  {interaction['feature1']} × {interaction['feature2']}")
            print(f"    Score: {interaction['score']:.4f}")
            print(f"    Type: {interaction['type']}")

    # Display polynomial suggestions
    if results['polynomials']:
        print("\n--- Polynomial Feature Suggestions ---")
        for poly in results['polynomials'][:5]:
            print(f"  {poly['feature']}^{poly['degree']}")
            print(f"    Score: {poly['score']:.4f}")

    # Display binning recommendations
    if results['binning']:
        print("\n--- Binning Recommendations ---")
        for binning in results['binning'][:5]:
            print(f"  {binning['feature']}: {binning['strategy']} ({binning['n_bins']} bins)")

    # Display encoding recommendations
    if results['encoding']:
        print("\n--- Encoding Recommendations ---")
        for encoding in results['encoding'][:5]:
            print(f"  {encoding['feature']}: {encoding['method']}")
            print(f"    Reason: {encoding['reason']}")

    # Display scaling recommendations
    if results['scaling']:
        print("\n--- Scaling Recommendations ---")
        for scaling in results['scaling'][:5]:
            print(f"  {scaling['feature']}: {scaling['method']}")
            print(f"    Reason: {scaling['reason']}")

    # Generate code
    print("\n--- Generated Python Code ---")
    code = fe_mgr.generate_code(results, n_top=3)
    print(code[:500] + "..." if len(code) > 500 else code)


def example_4_anomaly_detection():
    """Example 4: Multi-method anomaly detection"""
    print("\n" + "="*80)
    print("Example 4: Anomaly Detection")
    print("="*80)

    # Generate sensor data with anomalies
    df = generate_sensor_data(n_samples=5000)

    # Focus on numeric columns
    numeric_cols = ['temperature', 'humidity', 'pressure', 'vibration']
    df_numeric = df[numeric_cols].dropna()

    print(f"\nAnalyzing {len(df_numeric)} sensor readings across {len(numeric_cols)} metrics")

    # Create anomaly detection manager
    anomaly_mgr = AnomalyDetectionManager()

    # 1. Univariate anomaly detection
    print("\n--- Univariate Anomaly Detection ---")
    print("(Detecting outliers in each variable independently)")

    univariate_results = anomaly_mgr.detect_univariate(
        df_numeric,
        methods=['iqr', 'zscore', 'mad']
    )

    for method, result in univariate_results.items():
        print(f"\n{method.upper()} Method:")
        print(f"  Total anomalies: {result.n_anomalies}")
        print(f"  Percentage: {result.anomaly_percentage:.2f}%")
        if result.feature_anomaly_counts:
            print("  Per feature:")
            for feature, count in list(result.feature_anomaly_counts.items())[:3]:
                print(f"    {feature}: {count}")

    # 2. Multivariate anomaly detection
    print("\n--- Multivariate Anomaly Detection ---")
    print("(Detecting unusual combinations of values)")

    multivariate_results = anomaly_mgr.detect_multivariate(
        df_numeric,
        methods=['isolation_forest', 'lof']
    )

    for method, result in multivariate_results.items():
        print(f"\n{method.upper().replace('_', ' ').title()}:")
        print(f"  Anomalies detected: {result.n_anomalies}")
        print(f"  Percentage: {result.anomaly_percentage:.2f}%")

    # 3. Ensemble detection (combine methods)
    print("\n--- Ensemble Anomaly Detection ---")
    print("(Combining multiple methods for robust detection)")

    ensemble_result = anomaly_mgr.ensemble_detection(
        df_numeric,
        voting_threshold=0.5  # Flag as anomaly if 50%+ methods agree
    )

    print(f"Ensemble anomalies: {ensemble_result.n_anomalies}")
    print(f"Percentage: {ensemble_result.anomaly_percentage:.2f}%")
    print(f"Voting threshold: {0.5}")

    # 4. Explain anomalies
    if ensemble_result.n_anomalies > 0:
        print("\n--- Anomaly Explanations ---")
        print("(Which features contributed most to the anomaly?)")

        explanations = anomaly_mgr.explain_anomalies(
            df_numeric,
            ensemble_result,
            top_n=3
        )

        # Show explanation for first few anomalies
        for idx, contrib in list(explanations.items())[:3]:
            print(f"\nAnomaly at index {idx}:")
            for feature, contribution in list(contrib.items())[:3]:
                print(f"  {feature}: {contribution:.3f}")


def example_5_comprehensive_analysis():
    """Example 5: Comprehensive analysis combining all features"""
    print("\n" + "="*80)
    print("Example 5: Comprehensive Analysis with All Features")
    print("="*80)

    # Generate financial data
    df = generate_financial_data(n_samples=500)

    print(f"\nAnalyzing financial dataset: {len(df)} companies")

    # Run comprehensive EDA with all features enabled
    eda = SimplusEDA(
        enable_statistical_tests=True,
        correlation_threshold=0.6,
        outlier_method='isolation_forest',
        verbose=True
    )

    print("\nPerforming comprehensive analysis...")
    results = eda.analyze(df)

    # Display summary
    print("\n" + eda.summary())

    # Get and display insights
    insights = eda.get_insights()

    if insights.get('quality_issues'):
        print("\n--- Data Quality Issues ---")
        for issue in insights['quality_issues'][:5]:
            print(f"  • {issue}")

    if insights.get('statistical_findings'):
        print("\n--- Statistical Findings ---")
        for finding in insights['statistical_findings'][:5]:
            print(f"  • {finding}")

    if insights.get('correlations'):
        print("\n--- Key Correlations ---")
        for corr in insights['correlations'][:5]:
            print(f"  • {corr}")

    if insights.get('recommendations'):
        print("\n--- Recommendations ---")
        for rec in insights['recommendations'][:5]:
            print(f"  • {rec}")

    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    eda.generate_report(
        'outputs/comprehensive_financial_analysis.html',
        title='Financial Data - Comprehensive Analysis',
        author='Data Science Team',
        include_visualizations=True,
        include_data_preview=True
    )

    print("\n✓ Comprehensive analysis complete!")
    print("  Report: outputs/comprehensive_financial_analysis.html")


def run_all_examples():
    """Run all advanced analysis examples"""
    print("\n" + "="*80)
    print("SIMPLUS EDA FRAMEWORK - ADVANCED ANALYSIS EXAMPLES")
    print("="*80)

    # Create output directory
    os.makedirs('outputs', exist_ok=True)

    try:
        example_1_statistical_tests()
        example_2_time_series_analysis()
        example_3_feature_engineering()
        example_4_anomaly_detection()
        example_5_comprehensive_analysis()

        print("\n" + "="*80)
        print("All advanced examples completed successfully!")
        print("="*80)
        print("\nKey takeaways:")
        print("  • Statistical tests help validate hypotheses about your data")
        print("  • Time series analysis reveals trends and seasonality")
        print("  • Feature engineering suggestions can improve ML models")
        print("  • Anomaly detection helps identify unusual patterns")
        print("  • Combine all features for comprehensive data understanding")

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_examples()
