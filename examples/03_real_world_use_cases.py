"""
Example 3: Real-World Use Cases
================================

This script demonstrates how to use the Simplus EDA framework in real-world scenarios:
- E-commerce sales analysis
- Customer churn prediction
- Healthcare data analysis
- IoT sensor monitoring
- Marketing campaign optimization

Run this example:
    python examples/03_real_world_use_cases.py
"""

import pandas as pd
import numpy as np
from simplus_eda import SimplusEDA, EDAConfig
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from data_generators import (
    generate_sales_data,
    generate_customer_churn_data,
    generate_healthcare_data,
    generate_sensor_data,
    generate_marketing_data
)


def use_case_1_ecommerce_sales():
    """Use Case 1: E-commerce Sales Analysis"""
    print("\n" + "="*80)
    print("USE CASE 1: E-Commerce Sales Analysis")
    print("="*80)
    print("\nScenario: Analyze sales data to identify revenue drivers and data quality issues")

    # Generate realistic e-commerce data
    df = generate_sales_data(n_samples=2000)

    print(f"\nDataset: {len(df)} transactions across {df['category'].nunique()} categories")

    # Configure for business analysis
    eda = SimplusEDA(
        correlation_threshold=0.6,
        outlier_method='isolation_forest',
        missing_threshold=0.05,
        verbose=True
    )

    print("\nAnalyzing sales patterns...")
    eda.analyze(df)

    # Business insights
    print("\n--- Business Insights ---")
    quality = eda.get_quality_score()
    print(f"Data Quality Score: {quality:.1f}%")

    if quality < 85:
        print("⚠️  Data quality concerns detected - review before making decisions")

    # Key metrics by category
    print("\n--- Revenue by Category ---")
    category_revenue = df.groupby('category')['revenue'].agg(['sum', 'mean', 'count'])
    category_revenue = category_revenue.sort_values('sum', ascending=False)
    print(category_revenue)

    # Correlation insights
    correlations = eda.get_correlations(threshold=0.6)
    if correlations:
        print("\n--- Key Correlations (Revenue Drivers) ---")
        revenue_corrs = [c for c in correlations if 'revenue' in [c['feature1'], c['feature2']]]
        for corr in revenue_corrs[:5]:
            print(f"  {corr['feature1']} <-> {corr['feature2']}: {corr['correlation']:.3f}")

    # Generate business report
    eda.generate_report(
        'outputs/ecommerce_sales_analysis.html',
        title='E-Commerce Sales Analysis - Q4 2024',
        author='Business Analytics Team',
        company='Your E-Commerce Co.',
        include_visualizations=True
    )

    print("\n✓ Analysis complete: outputs/ecommerce_sales_analysis.html")
    print("\nAction items:")
    print("  1. Review high-revenue categories for inventory planning")
    print("  2. Investigate missing discount data (15% missing)")
    print("  3. Analyze outlier transactions for fraud detection")


def use_case_2_customer_churn():
    """Use Case 2: Customer Churn Prediction - Data Preparation"""
    print("\n" + "="*80)
    print("USE CASE 2: Customer Churn Analysis")
    print("="*80)
    print("\nScenario: Prepare churn prediction data and identify key risk factors")

    # Generate customer data
    df = generate_customer_churn_data(n_samples=3000)

    churn_rate = df['churned'].mean() * 100
    print(f"\nDataset: {len(df)} customers, {churn_rate:.1f}% churn rate")

    # Configure for ML preparation
    config = EDAConfig(
        enable_statistical_tests=True,
        correlation_threshold=0.5,
        missing_threshold=0.1,
        outlier_method='iqr',
        verbose=True
    )

    eda = SimplusEDA(config=config)

    print("\nAnalyzing customer data for ML model preparation...")
    eda.analyze(df)

    # Churn analysis
    print("\n--- Churn Analysis by Segment ---")

    # By contract type
    churn_by_contract = df.groupby('contract_type')['churned'].agg(['mean', 'count'])
    churn_by_contract['mean'] *= 100
    churn_by_contract.columns = ['Churn Rate %', 'Customers']
    print("\nBy Contract Type:")
    print(churn_by_contract)

    # By tenure
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 100],
                                 labels=['0-6mo', '6-12mo', '12-24mo', '24mo+'])
    churn_by_tenure = df.groupby('tenure_group')['churned'].agg(['mean', 'count'])
    churn_by_tenure['mean'] *= 100
    churn_by_tenure.columns = ['Churn Rate %', 'Customers']
    print("\nBy Tenure:")
    print(churn_by_tenure)

    # Feature importance for churn (correlation with target)
    print("\n--- Features Correlated with Churn ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    churn_correlations = df[numeric_cols].corrwith(df['churned']).abs().sort_values(ascending=False)
    print(churn_correlations.head(10))

    # Generate report
    eda.generate_report(
        'outputs/customer_churn_analysis.html',
        title='Customer Churn Analysis & ML Preparation',
        author='Data Science Team',
        include_visualizations=True
    )

    # Export for ML pipeline
    eda.export_results('outputs/churn_eda_results.json')

    print("\n✓ Analysis complete!")
    print("\nNext steps for ML model:")
    print("  1. Handle missing values in internet_service (5%) and tech_support (8%)")
    print("  2. Encode categorical variables: contract_type, payment_method")
    print("  3. Consider feature engineering: tenure × monthly_charges")
    print("  4. Address class imbalance (25% churn rate)")


def use_case_3_healthcare_monitoring():
    """Use Case 3: Healthcare Patient Data Analysis"""
    print("\n" + "="*80)
    print("USE CASE 3: Healthcare Patient Monitoring")
    print("="*80)
    print("\nScenario: Analyze patient health records to identify risk factors")

    # Generate patient data
    df = generate_healthcare_data(n_samples=2000)

    print(f"\nDataset: {len(df)} patient records")

    # Configure for healthcare analysis
    eda = SimplusEDA(
        correlation_threshold=0.4,
        outlier_method='iqr',  # More conservative for medical data
        missing_threshold=0.15,
        verbose=True
    )

    print("\nAnalyzing patient health records...")
    eda.analyze(df)

    # Risk stratification
    print("\n--- Patient Risk Stratification ---")
    df['risk_category'] = pd.cut(df['risk_score'],
                                  bins=[0, 25, 50, 75, 100],
                                  labels=['Low', 'Moderate', 'High', 'Very High'])

    risk_distribution = df['risk_category'].value_counts().sort_index()
    print("\nRisk Distribution:")
    for category, count in risk_distribution.items():
        pct = count / len(df) * 100
        print(f"  {category:12s}: {count:4d} ({pct:5.1f}%)")

    # Readmission analysis
    readmit_rate = df['readmitted_30_days'].mean() * 100
    print(f"\n30-day Readmission Rate: {readmit_rate:.1f}%")

    high_risk_readmit = df[df['risk_score'] > 75]['readmitted_30_days'].mean() * 100
    low_risk_readmit = df[df['risk_score'] <= 25]['readmitted_30_days'].mean() * 100

    print(f"  High risk patients: {high_risk_readmit:.1f}%")
    print(f"  Low risk patients: {low_risk_readmit:.1f}%")

    # Health indicators
    print("\n--- Key Health Indicators ---")
    print(f"Average Age: {df['age'].mean():.1f} years")
    print(f"Average BMI: {df['bmi'].mean():.1f}")
    print(f"Hypertension Prevalence: {df['hypertension'].mean()*100:.1f}%")
    print(f"Diabetes Prevalence: {df['diabetes'].mean()*100:.1f}%")

    # Generate clinical report
    eda.generate_report(
        'outputs/healthcare_patient_analysis.html',
        title='Patient Health Records Analysis',
        author='Clinical Analytics Team',
        company='Healthcare System',
        include_visualizations=True
    )

    print("\n✓ Analysis complete: outputs/healthcare_patient_analysis.html")
    print("\nClinical insights:")
    print("  1. 15% missing cholesterol data - follow up on lab results")
    print("  2. Strong correlation between age and blood pressure")
    print("  3. High-risk patients have 3x higher readmission rate")


def use_case_4_iot_sensor_monitoring():
    """Use Case 4: IoT Sensor Data Monitoring"""
    print("\n" + "="*80)
    print("USE CASE 4: IoT Sensor Monitoring")
    print("="*80)
    print("\nScenario: Monitor sensor data quality and detect anomalies")

    # Generate sensor data
    df = generate_sensor_data(n_samples=10000)

    print(f"\nDataset: {len(df)} sensor readings from {df['sensor_id'].nunique()} sensors")

    # Configure for time-series monitoring
    eda = SimplusEDA(
        correlation_threshold=0.7,
        outlier_method='isolation_forest',
        missing_threshold=0.05,
        enable_auto_sampling=True,  # Handle large dataset
        n_samples_viz=5000,
        verbose=True
    )

    print("\nAnalyzing sensor data...")
    eda.analyze(df)

    # Data quality by sensor
    print("\n--- Data Quality by Sensor ---")
    for sensor in df['sensor_id'].unique():
        sensor_data = df[df['sensor_id'] == sensor]
        missing_pct = sensor_data[['temperature', 'humidity', 'pressure']].isnull().mean().mean() * 100
        warning_count = (sensor_data['status'] == 'WARNING').sum()

        print(f"\n{sensor}:")
        print(f"  Readings: {len(sensor_data)}")
        print(f"  Missing data: {missing_pct:.1f}%")
        print(f"  Warnings: {warning_count}")
        print(f"  Avg battery: {sensor_data['battery_level'].mean():.1f}%")

    # Anomaly detection
    print("\n--- Sensor Anomalies ---")
    temp_outliers = df[
        (df['temperature'] < -10) |
        (df['temperature'] > 50)
    ]
    print(f"Temperature anomalies: {len(temp_outliers)} ({len(temp_outliers)/len(df)*100:.2f}%)")

    # Battery alerts
    low_battery = df[df['battery_level'] < 20]
    print(f"Low battery alerts: {len(low_battery)} readings")

    # Generate monitoring report
    eda.generate_report(
        'outputs/iot_sensor_monitoring.html',
        title='IoT Sensor Data Quality Report',
        author='Operations Team',
        include_visualizations=True
    )

    print("\n✓ Analysis complete: outputs/iot_sensor_monitoring.html")
    print("\nMaintenance recommendations:")
    print("  1. Schedule battery replacement for sensors below 20%")
    print("  2. Investigate connectivity issues causing 5% data loss")
    print("  3. Calibrate sensors showing temperature anomalies")


def use_case_5_marketing_optimization():
    """Use Case 5: Marketing Campaign Optimization"""
    print("\n" + "="*80)
    print("USE CASE 5: Marketing Campaign Optimization")
    print("="*80)
    print("\nScenario: Analyze campaign performance to optimize marketing spend")

    # Generate marketing data
    df = generate_marketing_data(n_samples=3000)

    print(f"\nDataset: {len(df)} campaigns across {df['channel'].nunique()} channels")

    # Configure for marketing analysis
    eda = SimplusEDA(
        correlation_threshold=0.5,
        outlier_method='isolation_forest',
        missing_threshold=0.1,
        verbose=True
    )

    print("\nAnalyzing campaign performance...")
    eda.analyze(df)

    # Performance by channel
    print("\n--- Performance by Channel ---")
    channel_metrics = df.groupby('channel').agg({
        'budget': 'sum',
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum',
        'revenue': 'sum'
    })

    channel_metrics['CTR'] = (channel_metrics['clicks'] / channel_metrics['impressions'] * 100).round(2)
    channel_metrics['CVR'] = (channel_metrics['conversions'] / channel_metrics['clicks'] * 100).round(2)
    channel_metrics['ROI'] = ((channel_metrics['revenue'] - channel_metrics['budget']) /
                              channel_metrics['budget'] * 100).round(2)

    print("\n", channel_metrics[['budget', 'CTR', 'CVR', 'ROI']])

    # Best performing campaigns
    print("\n--- Top 5 Campaigns by ROI ---")
    top_campaigns = df.nlargest(5, 'roi')[['channel', 'campaign_type', 'budget', 'roi']]
    print(top_campaigns)

    # Budget efficiency
    print("\n--- Budget Efficiency ---")
    total_budget = df['budget'].sum()
    total_revenue = df['revenue'].sum()
    overall_roi = (total_revenue - total_budget) / total_budget * 100

    print(f"Total Budget: ${total_budget:,.2f}")
    print(f"Total Revenue: ${total_revenue:,.2f}")
    print(f"Overall ROI: {overall_roi:.2f}%")

    # Generate marketing report
    eda.generate_report(
        'outputs/marketing_campaign_analysis.html',
        title='Marketing Campaign Performance Analysis',
        author='Marketing Analytics Team',
        company='Marketing Department',
        include_visualizations=True
    )

    print("\n✓ Analysis complete: outputs/marketing_campaign_analysis.html")
    print("\nOptimization recommendations:")
    print("  1. Increase budget allocation to high-ROI channels")
    print("  2. Review campaigns with negative ROI")
    print("  3. Investigate viral campaigns (outliers) to replicate success")
    print("  4. Address 10% missing revenue data for accurate ROI calculation")


def run_all_use_cases():
    """Run all real-world use case examples"""
    print("\n" + "="*80)
    print("SIMPLUS EDA FRAMEWORK - REAL-WORLD USE CASES")
    print("="*80)

    # Create output directory
    os.makedirs('outputs', exist_ok=True)

    try:
        use_case_1_ecommerce_sales()
        use_case_2_customer_churn()
        use_case_3_healthcare_monitoring()
        use_case_4_iot_sensor_monitoring()
        use_case_5_marketing_optimization()

        print("\n" + "="*80)
        print("All use cases completed successfully!")
        print("="*80)
        print("\nGenerated Reports:")
        print("  1. outputs/ecommerce_sales_analysis.html")
        print("  2. outputs/customer_churn_analysis.html")
        print("  3. outputs/healthcare_patient_analysis.html")
        print("  4. outputs/iot_sensor_monitoring.html")
        print("  5. outputs/marketing_campaign_analysis.html")
        print("\nThese examples demonstrate how Simplus EDA can be applied to")
        print("various industries and use cases for data-driven decision making.")

    except Exception as e:
        print(f"\n✗ Error running use cases: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_use_cases()
