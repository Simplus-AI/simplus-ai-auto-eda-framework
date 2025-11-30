"""
Data Generators for Testing Simplus AI Auto EDA Framework

This module provides realistic data generators for testing and demonstrating
the capabilities of the Simplus EDA framework across different use cases.

Available Generators:
- generate_sales_data: E-commerce sales dataset
- generate_customer_churn_data: Customer churn prediction dataset
- generate_time_series_sales: Time series sales data with seasonality
- generate_financial_data: Financial metrics with anomalies
- generate_sensor_data: IoT sensor readings with outliers
- generate_healthcare_data: Patient health records
- generate_marketing_data: Marketing campaign performance data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional


def generate_sales_data(
    n_samples: int = 1000,
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate realistic e-commerce sales dataset.

    Features:
    - Strong correlations (price vs revenue, quantity vs revenue)
    - Outliers in revenue and quantity
    - Missing values in discount and customer_age
    - Categorical features (category, region, customer_segment)
    - Data quality issues (duplicates, inconsistencies)

    Args:
        n_samples: Number of records to generate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with sales data
    """
    np.random.seed(random_state)

    # Generate base features
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
    regions = ['North', 'South', 'East', 'West', 'Central']
    segments = ['Premium', 'Standard', 'Budget']

    # Generate numeric features with correlations
    base_price = np.random.gamma(shape=2, scale=50, size=n_samples)
    prices = np.round(base_price, 2)

    # Segments
    segments_arr = np.random.choice(segments, n_samples, p=[0.2, 0.5, 0.3])

    # Quantity correlated with segment
    segment_multiplier = pd.Series(segments_arr).map({
        'Premium': 1.5,
        'Standard': 1.0,
        'Budget': 0.7
    }).values
    quantities = np.maximum(1, np.random.poisson(lam=3 * segment_multiplier, size=n_samples))

    # Discount (with missing values)
    discount = np.random.beta(a=2, b=5, size=n_samples) * 0.5
    # Introduce 15% missing values
    missing_idx = np.random.choice(n_samples, int(n_samples * 0.15), replace=False)
    discount[missing_idx] = np.nan

    # Revenue (highly correlated with price and quantity)
    revenue = prices * quantities * (1 - np.nan_to_num(discount))

    # Customer age (with missing values) - use float to allow NaN
    customer_age = np.random.normal(loc=40, scale=15, size=n_samples)
    customer_age = np.clip(customer_age, 18, 80)
    # Introduce 10% missing values
    missing_idx = np.random.choice(n_samples, int(n_samples * 0.1), replace=False)
    customer_age[missing_idx] = np.nan

    # Shipping cost
    shipping_cost = np.random.uniform(5, 25, n_samples).round(2)

    # Add outliers (5% of revenue values)
    outlier_idx = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    revenue[outlier_idx] *= np.random.uniform(5, 10, len(outlier_idx))

    data = {
        'transaction_id': [f'TXN{str(i).zfill(6)}' for i in range(n_samples)],
        'date': pd.date_range(start='2023-01-01', periods=n_samples, freq='4h'),
        'category': np.random.choice(categories, n_samples),
        'region': np.random.choice(regions, n_samples),
        'customer_segment': segments_arr,
        'price': prices,
        'quantity': quantities,
        'discount': discount,
        'revenue': revenue,
        'customer_age': customer_age,
        'shipping_cost': shipping_cost
    }

    df = pd.DataFrame(data)

    # Add some duplicate rows (2%)
    n_duplicates = int(n_samples * 0.02)
    duplicate_idx = np.random.choice(n_samples, n_duplicates, replace=False)
    duplicates = df.iloc[duplicate_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


def generate_customer_churn_data(
    n_samples: int = 2000,
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate customer churn prediction dataset.

    Features:
    - Binary target variable (churned)
    - Mix of numerical and categorical features
    - Feature interactions (tenure * monthly_charges)
    - Imbalanced classes (25% churn rate)
    - Missing values in internet_service and tech_support

    Args:
        n_samples: Number of customers to generate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with customer churn data
    """
    np.random.seed(random_state)

    data = {
        'customer_id': [f'CUST{str(i).zfill(6)}' for i in range(n_samples)],
    }

    # Tenure in months
    data['tenure'] = np.random.gamma(shape=2, scale=12, size=n_samples).astype(int)
    data['tenure'] = np.clip(data['tenure'], 1, 72)

    # Contract type
    contract_types = ['Month-to-month', 'One year', 'Two year']
    data['contract_type'] = np.random.choice(
        contract_types,
        n_samples,
        p=[0.5, 0.3, 0.2]
    )

    # Monthly charges (correlated with contract type)
    base_charges = np.random.normal(loc=50, scale=20, size=n_samples)
    contract_multiplier = pd.Series(data['contract_type']).map({
        'Month-to-month': 1.3,
        'One year': 1.0,
        'Two year': 0.8
    }).values
    data['monthly_charges'] = np.maximum(20, base_charges * contract_multiplier).round(2)

    # Total charges (highly correlated with tenure and monthly charges)
    data['total_charges'] = (data['monthly_charges'] * data['tenure']).round(2)

    # Payment method
    payment_methods = ['Electronic check', 'Mailed check', 'Credit card', 'Bank transfer']
    data['payment_method'] = np.random.choice(payment_methods, n_samples)

    # Internet service (with missing values)
    internet_services = ['DSL', 'Fiber optic', 'No']
    data['internet_service'] = np.random.choice(internet_services, n_samples, p=[0.35, 0.4, 0.25])
    missing_idx = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    data['internet_service'] = pd.Series(data['internet_service'])
    data['internet_service'].iloc[missing_idx] = np.nan

    # Tech support
    data['tech_support'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    missing_idx = np.random.choice(n_samples, int(n_samples * 0.08), replace=False)
    data['tech_support'] = pd.Series(data['tech_support'])
    data['tech_support'].iloc[missing_idx] = np.nan

    # Number of services
    data['num_services'] = np.random.poisson(lam=2.5, size=n_samples)
    data['num_services'] = np.clip(data['num_services'], 0, 8)

    # Generate churn target (25% churn rate)
    # Higher churn probability for: short tenure, high charges, month-to-month contracts
    churn_probability = (
        0.1 +  # Base rate
        0.3 * (data['tenure'] < 6) +  # New customers
        0.2 * (data['monthly_charges'] > 80) +  # High charges
        0.2 * (pd.Series(data['contract_type']) == 'Month-to-month') +  # Flexible contracts
        0.1 * (pd.Series(data['payment_method']) == 'Electronic check')  # Payment method
    )
    churn_probability = np.clip(churn_probability, 0, 0.8)
    data['churned'] = (np.random.random(n_samples) < churn_probability).astype(int)

    return pd.DataFrame(data)


def generate_time_series_sales(
    n_days: int = 365,
    start_date: str = '2023-01-01',
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate time series sales data with trend, seasonality, and noise.

    Features:
    - Daily sales with clear trend
    - Weekly seasonality (lower on weekends)
    - Monthly seasonality (peaks mid-month)
    - Random noise and occasional anomalies
    - Multiple product categories

    Args:
        n_days: Number of days to generate
        start_date: Start date for the time series
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with time series sales data
    """
    np.random.seed(random_state)

    dates = pd.date_range(start=start_date, periods=n_days, freq='D')

    # Generate base trend (increasing over time)
    trend = np.linspace(1000, 2000, n_days)

    # Weekly seasonality (day of week effect)
    day_of_week = pd.Series(dates).dt.dayofweek.values
    weekly_seasonality = 1 + 0.2 * np.sin(2 * np.pi * day_of_week / 7)
    weekly_seasonality[day_of_week >= 5] *= 0.7  # Weekend penalty

    # Monthly seasonality (peak mid-month)
    day_of_month = pd.Series(dates).dt.day.values
    monthly_seasonality = 1 + 0.15 * np.sin(2 * np.pi * (day_of_month - 7) / 30)

    # Combine components
    base_sales = trend * weekly_seasonality * monthly_seasonality

    # Add random noise
    noise = np.random.normal(0, 50, n_days)

    # Generate sales for multiple categories
    categories = ['Electronics', 'Clothing', 'Home']

    data = []
    for category in categories:
        category_multiplier = np.random.uniform(0.8, 1.2)
        category_sales = base_sales * category_multiplier + noise

        # Add random anomalies (5%)
        anomaly_idx = np.random.choice(n_days, int(n_days * 0.05), replace=False)
        category_sales[anomaly_idx] *= np.random.uniform(1.5, 2.5, len(anomaly_idx))

        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'category': category,
                'sales': max(0, category_sales[i]),
                'day_of_week': day_of_week[i],
                'day_of_month': day_of_month[i],
                'is_weekend': day_of_week[i] >= 5,
                'month': date.month,
                'quarter': (date.month - 1) // 3 + 1
            })

    return pd.DataFrame(data)


def generate_financial_data(
    n_samples: int = 500,
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate financial metrics dataset with anomalies.

    Features:
    - Financial ratios and metrics
    - Outliers in profitability metrics
    - Missing values in optional fields
    - Correlated financial indicators
    - Anomalous companies (fraud indicators)

    Args:
        n_samples: Number of companies to generate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with financial data
    """
    np.random.seed(random_state)

    data = {
        'company_id': [f'COMP{str(i).zfill(5)}' for i in range(n_samples)],
        'industry': np.random.choice(
            ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing'],
            n_samples
        ),
    }

    # Revenue (in millions)
    data['revenue'] = np.random.lognormal(mean=4, sigma=1.5, size=n_samples).round(2)

    # Operating margin (correlated with industry)
    base_margin = np.random.normal(0.15, 0.08, n_samples)
    industry_adjustment = pd.Series(data['industry']).map({
        'Technology': 0.05,
        'Finance': 0.03,
        'Healthcare': 0.02,
        'Retail': -0.03,
        'Manufacturing': 0.0
    }).values
    data['operating_margin'] = np.clip(base_margin + industry_adjustment, -0.2, 0.5).round(4)

    # Net income (correlated with revenue and margin)
    data['net_income'] = (data['revenue'] * data['operating_margin'] *
                          np.random.normal(0.7, 0.1, n_samples)).round(2)

    # Total assets
    data['total_assets'] = (data['revenue'] * np.random.uniform(1.5, 3, n_samples)).round(2)

    # Total liabilities
    data['total_liabilities'] = (data['total_assets'] *
                                 np.random.uniform(0.4, 0.7, n_samples)).round(2)

    # Equity (assets - liabilities)
    data['equity'] = (data['total_assets'] - data['total_liabilities']).round(2)

    # ROE (Return on Equity)
    data['roe'] = np.where(
        data['equity'] > 0,
        (data['net_income'] / data['equity']).round(4),
        np.nan
    )

    # Current ratio (with missing values)
    current_ratio = np.random.uniform(0.8, 2.5, n_samples).round(2)
    missing_idx = np.random.choice(n_samples, int(n_samples * 0.1), replace=False)
    current_ratio[missing_idx] = np.nan
    data['current_ratio'] = current_ratio

    # Debt to equity ratio
    data['debt_to_equity'] = (data['total_liabilities'] /
                              np.maximum(data['equity'], 1)).round(2)

    # Employee count
    data['employees'] = (data['revenue'] * np.random.uniform(5, 15, n_samples)).astype(int)

    # Market cap (with some correlation to revenue and profitability)
    data['market_cap'] = (data['revenue'] * np.random.uniform(2, 8, n_samples) *
                          (1 + data['operating_margin'])).round(2)

    # Add anomalies (potential fraud cases - 3%)
    operating_margin = data['operating_margin'].copy()
    roe = data['roe'].copy()
    anomaly_idx = np.random.choice(n_samples, int(n_samples * 0.03), replace=False)
    operating_margin[anomaly_idx] *= np.random.uniform(3, 5, len(anomaly_idx))
    roe[anomaly_idx] = np.where(
        ~np.isnan(roe[anomaly_idx]),
        roe[anomaly_idx] * np.random.uniform(5, 10, len(anomaly_idx)),
        roe[anomaly_idx]
    )
    data['operating_margin'] = operating_margin
    data['roe'] = roe

    return pd.DataFrame(data)


def generate_sensor_data(
    n_samples: int = 10000,
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate IoT sensor readings with outliers and anomalies.

    Features:
    - Multiple sensor types (temperature, humidity, pressure)
    - Time-based patterns (daily cycles)
    - Sensor drift over time
    - Measurement outliers (sensor failures)
    - Missing values (connectivity issues)

    Args:
        n_samples: Number of readings to generate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with sensor data
    """
    np.random.seed(random_state)

    # Generate timestamps (1 reading per 5 minutes)
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=5*i) for i in range(n_samples)]

    data = {
        'timestamp': timestamps,
        'sensor_id': np.random.choice(['SENSOR_A', 'SENSOR_B', 'SENSOR_C'], n_samples),
    }

    # Hours since start (for trends)
    hours = np.array([(ts - start_time).total_seconds() / 3600 for ts in timestamps])
    hour_of_day = hours % 24

    # Temperature (with daily cycle and drift)
    daily_cycle = 20 + 5 * np.sin(2 * np.pi * hour_of_day / 24)
    drift = 0.0002 * hours  # Gradual drift
    noise = np.random.normal(0, 0.5, n_samples)
    data['temperature'] = (daily_cycle + drift + noise).round(2)

    # Humidity (inversely correlated with temperature)
    data['humidity'] = (70 - 2 * (data['temperature'] - 20) +
                       np.random.normal(0, 3, n_samples)).round(2)
    data['humidity'] = np.clip(data['humidity'], 20, 95)

    # Pressure (with slight variations)
    data['pressure'] = (1013 + np.random.normal(0, 2, n_samples)).round(2)

    # Vibration (mostly low, occasional spikes)
    base_vibration = np.random.exponential(scale=0.5, size=n_samples)
    data['vibration'] = base_vibration.round(3)

    # Add outliers (sensor malfunctions - 2%)
    temperature = data['temperature'].copy()
    outlier_idx = np.random.choice(n_samples, int(n_samples * 0.02), replace=False)
    temperature[outlier_idx] += np.random.uniform(-50, 50, len(outlier_idx))
    data['temperature'] = temperature

    # Add missing values (connectivity issues - 5%)
    for col in ['temperature', 'humidity', 'pressure']:
        col_data = data[col].copy()
        missing_idx = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
        col_data[missing_idx] = np.nan
        data[col] = col_data

    # Battery level (decreasing over time)
    data['battery_level'] = np.maximum(
        10,
        100 - 0.01 * hours + np.random.normal(0, 2, n_samples)
    ).round(1)

    # Status flag
    data['status'] = np.where(
        (pd.Series(data['battery_level']) < 20) |
        (pd.Series(data['temperature']).isna()),
        'WARNING',
        'NORMAL'
    )

    return pd.DataFrame(data)


def generate_healthcare_data(
    n_samples: int = 1500,
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate patient health records dataset.

    Features:
    - Patient demographics
    - Vital signs and lab results
    - Medical history indicators
    - Risk scores and outcomes
    - Missing values (incomplete records)

    Args:
        n_samples: Number of patients to generate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with healthcare data
    """
    np.random.seed(random_state)

    data = {
        'patient_id': [f'PAT{str(i).zfill(6)}' for i in range(n_samples)],
        'age': np.random.normal(55, 18, n_samples).astype(int),
        'gender': np.random.choice(['M', 'F'], n_samples),
    }
    data['age'] = np.clip(data['age'], 18, 95)

    # BMI (Body Mass Index)
    data['bmi'] = np.random.normal(27, 5, n_samples).round(1)
    data['bmi'] = np.clip(data['bmi'], 15, 50)

    # Blood pressure (systolic)
    age_factor = (data['age'] - 18) / 77 * 30  # Increases with age
    data['blood_pressure_systolic'] = (110 + age_factor +
                                       np.random.normal(0, 10, n_samples)).astype(int)

    # Blood pressure (diastolic)
    data['blood_pressure_diastolic'] = (data['blood_pressure_systolic'] * 0.65 +
                                        np.random.normal(0, 5, n_samples)).astype(int)

    # Cholesterol (with missing values)
    cholesterol = np.random.normal(200, 40, n_samples)
    missing_idx = np.random.choice(n_samples, int(n_samples * 0.15), replace=False)
    cholesterol[missing_idx] = np.nan
    data['cholesterol'] = cholesterol

    # Blood glucose (with missing values)
    glucose = np.random.normal(100, 25, n_samples)
    glucose = np.maximum(70, glucose)
    missing_idx = np.random.choice(n_samples, int(n_samples * 0.12), replace=False)
    glucose[missing_idx] = np.nan
    data['glucose'] = glucose

    # Heart rate
    data['heart_rate'] = np.random.normal(72, 12, n_samples).astype(int)
    data['heart_rate'] = np.clip(data['heart_rate'], 50, 120)

    # Medical history (binary indicators)
    data['diabetes'] = (np.random.random(n_samples) < 0.15).astype(int)
    data['hypertension'] = (np.random.random(n_samples) < 0.25).astype(int)
    data['heart_disease'] = (np.random.random(n_samples) < 0.10).astype(int)

    # Smoking status
    data['smoking_status'] = np.random.choice(
        ['Never', 'Former', 'Current'],
        n_samples,
        p=[0.6, 0.25, 0.15]
    )

    # Exercise frequency (hours per week)
    data['exercise_hours_per_week'] = np.random.gamma(shape=2, scale=2, size=n_samples).round(1)
    data['exercise_hours_per_week'] = np.clip(data['exercise_hours_per_week'], 0, 20)

    # Risk score (composite of various factors)
    risk_score = (
        0.2 * (data['age'] - 18) / 77 +
        0.15 * data['diabetes'] +
        0.15 * data['hypertension'] +
        0.2 * data['heart_disease'] +
        0.1 * (pd.Series(data['smoking_status']) == 'Current').astype(int) +
        0.1 * (data['bmi'] > 30).astype(int) +
        0.1 * ((data['blood_pressure_systolic'] > 140).astype(int))
    )
    data['risk_score'] = np.clip(risk_score * 100, 0, 100).round(1)

    # Hospital readmission within 30 days (outcome)
    readmit_prob = risk_score * 0.3
    data['readmitted_30_days'] = (np.random.random(n_samples) < readmit_prob).astype(int)

    return pd.DataFrame(data)


def generate_marketing_data(
    n_samples: int = 5000,
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate marketing campaign performance dataset.

    Features:
    - Campaign attributes (channel, type, budget)
    - Performance metrics (impressions, clicks, conversions)
    - ROI and cost metrics
    - Temporal patterns
    - Missing values in optional metrics

    Args:
        n_samples: Number of campaigns to generate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with marketing campaign data
    """
    np.random.seed(random_state)

    data = {
        'campaign_id': [f'CAMP{str(i).zfill(6)}' for i in range(n_samples)],
        'date': pd.date_range(start='2023-01-01', periods=n_samples, freq='h')[:n_samples],
    }

    # Campaign attributes
    channels = ['Social Media', 'Email', 'Search', 'Display', 'Video']
    data['channel'] = np.random.choice(channels, n_samples)

    campaign_types = ['Brand Awareness', 'Lead Generation', 'Conversion', 'Retention']
    data['campaign_type'] = np.random.choice(campaign_types, n_samples)

    # Budget (in dollars)
    data['budget'] = np.random.lognormal(mean=7, sigma=1, size=n_samples).round(2)

    # Channel-specific performance multipliers
    channel_ctr = pd.Series(data['channel']).map({
        'Social Media': 0.02,
        'Email': 0.03,
        'Search': 0.05,
        'Display': 0.015,
        'Video': 0.025
    }).values

    # Impressions (correlated with budget)
    data['impressions'] = (data['budget'] * np.random.uniform(50, 150, n_samples)).astype(int)

    # Clicks (based on impressions and CTR)
    base_ctr = channel_ctr * np.random.uniform(0.8, 1.2, n_samples)
    data['clicks'] = (data['impressions'] * base_ctr).astype(int)

    # Cost per click
    data['cpc'] = (data['budget'] / np.maximum(data['clicks'], 1)).round(2)

    # Conversions (based on clicks)
    conversion_rate = np.random.beta(a=2, b=50, size=n_samples)
    data['conversions'] = (data['clicks'] * conversion_rate).astype(int)

    # Revenue (with missing values for some campaigns)
    avg_order_value = np.random.uniform(50, 200, n_samples)
    revenue = (data['conversions'] * avg_order_value).round(2)
    missing_idx = np.random.choice(n_samples, int(n_samples * 0.1), replace=False)
    revenue[missing_idx] = np.nan
    data['revenue'] = revenue

    # Cost per acquisition
    data['cpa'] = np.where(
        data['conversions'] > 0,
        (data['budget'] / data['conversions']).round(2),
        np.nan
    )

    # ROI (Return on Investment)
    data['roi'] = np.where(
        data['budget'] > 0,
        ((np.nan_to_num(revenue, 0) - data['budget']) / data['budget'] * 100).round(2),
        np.nan
    )

    # Engagement score (with missing values)
    engagement_score = np.random.beta(a=5, b=2, size=n_samples) * 100
    missing_idx = np.random.choice(n_samples, int(n_samples * 0.08), replace=False)
    engagement_score[missing_idx] = np.nan
    data['engagement_score'] = engagement_score

    # Target audience size
    data['target_audience'] = (data['impressions'] *
                               np.random.uniform(1.5, 3, n_samples)).astype(int)

    # Add some outliers (viral campaigns - 1%)
    impressions = data['impressions'].copy()
    clicks = data['clicks'].copy()
    outlier_idx = np.random.choice(n_samples, int(n_samples * 0.01), replace=False)
    impressions[outlier_idx] = (impressions[outlier_idx] * np.random.uniform(10, 50, len(outlier_idx))).astype(int)
    clicks[outlier_idx] = (clicks[outlier_idx] * np.random.uniform(10, 50, len(outlier_idx))).astype(int)
    data['impressions'] = impressions
    data['clicks'] = clicks

    return pd.DataFrame(data)


# Utility function to save all datasets
def generate_all_datasets(output_dir: str = './sample_data', random_state: int = 42):
    """
    Generate all sample datasets and save to CSV files.

    Args:
        output_dir: Directory to save the CSV files
        random_state: Random seed for reproducibility
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        'sales_data.csv': generate_sales_data(1000, random_state),
        'customer_churn.csv': generate_customer_churn_data(2000, random_state),
        'time_series_sales.csv': generate_time_series_sales(365, random_state=random_state),
        'financial_data.csv': generate_financial_data(500, random_state),
        'sensor_data.csv': generate_sensor_data(10000, random_state),
        'healthcare_data.csv': generate_healthcare_data(1500, random_state),
        'marketing_data.csv': generate_marketing_data(5000, random_state),
    }

    for filename, df in datasets.items():
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"✓ Generated {filename}: {len(df)} rows, {len(df.columns)} columns")

    print(f"\n✓ All datasets saved to {output_dir}/")


if __name__ == '__main__':
    # Generate all sample datasets
    generate_all_datasets()
