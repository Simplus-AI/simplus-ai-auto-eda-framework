# Simplus AI Auto EDA Framework - Usage Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Basic Usage](#basic-usage)
6. [Advanced Features](#advanced-features)
7. [Command-Line Interface](#command-line-interface)
8. [Configuration](#configuration)
9. [Output Formats](#output-formats)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

## Introduction

The Simplus AI Auto EDA Framework is a comprehensive Python package for automated exploratory data analysis. It provides a unified, intuitive API for analyzing datasets, generating insights, and creating professional reports.

### Key Features

- **Automated Analysis**: Comprehensive statistical analysis with one function call
- **Professional Reports**: Generate HTML, JSON, and PDF reports
- **Advanced Analytics**: Statistical tests, time series analysis, feature engineering, anomaly detection
- **Large Dataset Support**: Handle datasets with millions of rows through sampling and parallel processing
- **Easy Integration**: Simple API for embedding in pipelines and services
- **Extensible**: Plugin architecture for custom analyzers and visualizations

## Installation

### Basic Installation

```bash
pip install -e .
```

### Optional Dependencies

For PDF reports:
```bash
pip install weasyprint
```

For large dataset support with Dask:
```bash
pip install dask[complete]
```

For time series analysis:
```bash
pip install statsmodels>=0.13.0
```

### Verify Installation

```python
from simplus_eda import SimplusEDA
print("SimplusEDA installed successfully!")
```

## Quick Start

### The Simplest Way (One-Liner)

```python
from simplus_eda import quick_analysis
import pandas as pd

df = pd.read_csv('your_data.csv')
quick_analysis(df, 'report.html')
```

That's it! This will analyze your data and generate a comprehensive HTML report.

### Basic Workflow

```python
from simplus_eda import SimplusEDA
import pandas as pd

# 1. Load data
df = pd.read_csv('your_data.csv')

# 2. Create EDA instance
eda = SimplusEDA()

# 3. Analyze
eda.analyze(df)

# 4. View summary
print(eda.summary())

# 5. Generate report
eda.generate_report('report.html')
```

### With Custom Configuration

```python
from simplus_eda import SimplusEDA

# Configure for your needs
eda = SimplusEDA(
    correlation_threshold=0.8,
    outlier_method='isolation_forest',
    missing_threshold=0.05,
    verbose=True
)

eda.analyze(df)
eda.generate_report('custom_report.html',
                    title='My Analysis',
                    author='Data Team')
```

## Core Concepts

### The SimplusEDA Class

`SimplusEDA` is the main interface that provides:

- **Automatic Component Management**: Creates and configures all analyzers
- **State Persistence**: Stores results for reuse across operations
- **Unified Configuration**: Single source of truth for all settings
- **Flexible API**: Multiple ways to configure and use

### Analysis Results

When you call `analyze()`, you get a comprehensive dictionary containing:

```python
results = eda.analyze(df)

# Access different sections
quality = results['quality']
statistics = results['statistics']
correlations = results['correlations']
insights = results['insights']
```

### Configuration System

Configuration flows through the entire framework:

```python
# Method 1: Keyword arguments (simplest)
eda = SimplusEDA(correlation_threshold=0.8)

# Method 2: Configuration object (most control)
from simplus_eda import EDAConfig
config = EDAConfig(correlation_threshold=0.8, outlier_method='zscore')
eda = SimplusEDA(config=config)

# Method 3: Load from file (reusable)
eda = SimplusEDA.from_config_file('my_config.json')
```

## Basic Usage

### Loading Data

SimplusEDA works with pandas DataFrames:

```python
import pandas as pd

# From CSV
df = pd.read_csv('data.csv')

# From Excel
df = pd.read_excel('data.xlsx')

# From Parquet
df = pd.read_parquet('data.parquet')

# From database
import sqlalchemy
engine = sqlalchemy.create_engine('postgresql://...')
df = pd.read_sql('SELECT * FROM table', engine)
```

### Performing Analysis

```python
eda = SimplusEDA()

# Basic analysis
results = eda.analyze(df)

# Quick analysis (skip expensive tests)
results = eda.analyze(df, quick=True)

# Analyze without storing internally
results = eda.analyze(df, inplace=False)
```

### Accessing Results

```python
# Get quality score
quality_score = eda.get_quality_score()
print(f"Data Quality: {quality_score:.1f}%")

# Get correlations
correlations = eda.get_correlations(threshold=0.8)
for corr in correlations:
    print(f"{corr['feature1']} <-> {corr['feature2']}: {corr['correlation']:.3f}")

# Get insights
insights = eda.get_insights()
for category, items in insights.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  - {item}")

# Get summary
print(eda.summary())
```

### Generating Reports

```python
# HTML report (default)
eda.generate_report('report.html')

# JSON report
eda.generate_report('results.json', format='json')

# PDF report (requires weasyprint)
eda.generate_report('report.pdf', format='pdf')

# Customized report
eda.generate_report(
    'report.html',
    title='Q4 2024 Sales Analysis',
    author='Data Science Team',
    company='Your Company',
    include_visualizations=True,
    include_data_preview=True,
    max_preview_rows=20
)

# Multiple formats
eda.generate_report('report.html', format='html')
eda.export_results('results.json')
```

## Advanced Features

### Statistical Tests

The framework includes comprehensive statistical testing:

```python
from simplus_eda.analyzers.statistical_tests import StatisticalTestsManager

# Create test manager
test_mgr = StatisticalTestsManager()

# Compare groups
result = test_mgr.compare_groups(
    df,
    value_col='sales',
    group_col='region',
    parametric=True,  # or None for auto-detect
    post_hoc=True
)

print(f"Test used: {result['test_used']}")
print(f"P-value: {result['test_result']['p_value']:.4f}")
print(f"Significant: {result['significant']}")

# Test categorical association
chi_result = test_mgr.test_categorical_association(
    df,
    var1='category',
    var2='segment',
    method='auto'  # 'chi_square', 'fisher', or 'auto'
)
```

Available tests:
- **ANOVA**: One-way and two-way with interaction
- **Post-hoc tests**: Tukey HSD, Bonferroni, Holm-Bonferroni
- **Non-parametric**: Mann-Whitney U, Kruskal-Wallis, Wilcoxon
- **Categorical**: Chi-square, Fisher's exact, McNemar's
- **Variance**: Levene's, Bartlett's
- **Effect sizes**: Cohen's d, eta-squared, Cramér's V
- **Time series**: Granger causality

### Time Series Analysis

Comprehensive time series capabilities:

```python
from simplus_eda.analyzers.timeseries import TimeSeriesAnalyzer

# Create analyzer
ts_analyzer = TimeSeriesAnalyzer(
    enable_forecasting=True,
    enable_decomposition=True,
    enable_acf_pacf=True,
    forecast_steps=12
)

# Analyze time series
series = df.set_index('date')['sales']
results = ts_analyzer.analyze(series, seasonal_period=12)

# Access results
print(f"Trend: {results['trend']['mann_kendall']['trend']}")
print(f"Seasonality: {results['seasonality']['seasonal_strength']:.2f}")
print(f"Forecast:\n{results['forecasting']['forecast']['forecast']}")
```

Features:
- **Stationarity tests**: ADF, KPSS
- **Trend detection**: Mann-Kendall, linear trend
- **Decomposition**: STL, classical
- **Seasonality analysis**: Strength metrics, testing
- **Forecasting**: Auto-ARIMA, SARIMA
- **ACF/PACF**: Autocorrelation analysis

### Feature Engineering

Automatic feature engineering suggestions:

```python
from simplus_eda.analyzers.feature_engineering import FeatureEngineeringManager

# Create manager
fe_mgr = FeatureEngineeringManager(
    detect_interactions=True,
    detect_polynomials=True,
    suggest_binning=True,
    suggest_encoding=True,
    suggest_scaling=True
)

# Analyze
results = fe_mgr.analyze(df, target_col='target')

# View suggestions
print("\nFeature Interactions:")
for interaction in results['interactions'][:5]:
    print(f"  {interaction['feature1']} × {interaction['feature2']}: {interaction['score']:.3f}")

print("\nPolynomial Features:")
for poly in results['polynomials'][:5]:
    print(f"  {poly['feature']}^{poly['degree']}: {poly['score']:.3f}")

print("\nBinning Recommendations:")
for binning in results['binning'][:5]:
    print(f"  {binning['feature']}: {binning['strategy']} ({binning['n_bins']} bins)")

# Generate Python code
code = fe_mgr.generate_code(results)
print("\nGenerated Code:")
print(code)
```

Features:
- **Interaction detection**: Multiplicative, additive, ratio, difference
- **Polynomial features**: Automatic degree selection
- **Binning strategies**: Quantile, uniform, k-means
- **Encoding recommendations**: One-hot, label, target, frequency
- **Scaling recommendations**: Standard, minmax, robust, log

### Anomaly Detection

Advanced anomaly detection methods:

```python
from simplus_eda.analyzers.anomaly_detection import AnomalyDetectionManager

# Create manager
anomaly_mgr = AnomalyDetectionManager()

# Detect univariate anomalies
univariate_results = anomaly_mgr.detect_univariate(
    df,
    methods=['iqr', 'zscore', 'mad']
)

for method, result in univariate_results.items():
    print(f"\n{method.upper()}:")
    print(f"  Anomalies found: {result.n_anomalies}")
    print(f"  Percentage: {result.anomaly_percentage:.2f}%")

# Detect multivariate anomalies
multivariate_results = anomaly_mgr.detect_multivariate(
    df[numeric_cols],
    methods=['isolation_forest', 'lof']
)

# Ensemble detection (combine multiple methods)
ensemble_result = anomaly_mgr.ensemble_detection(
    df[numeric_cols],
    voting_threshold=0.5  # Anomaly if 50% of methods agree
)

# Explain anomalies
explanations = anomaly_mgr.explain_anomalies(
    df[numeric_cols],
    ensemble_result,
    top_n=3
)

for idx, contrib in explanations.items():
    print(f"\nAnomaly at index {idx}:")
    for feature, value in contrib.items():
        print(f"  {feature}: {value:.3f}")
```

Methods available:
- **Univariate**: IQR, Z-score, MAD
- **Multivariate**: Isolation Forest, LOF, Mahalanobis, Elliptic Envelope
- **Clustering**: DBSCAN
- **Time series**: STL decomposition
- **Ensemble**: Voting-based combination

### Large Dataset Support

Handle datasets with millions of rows:

```python
from simplus_eda import SimplusEDA, EDAConfig

# Use preset configuration
config = EDAConfig.get_large_dataset()
eda = SimplusEDA(config=config)

# Or customize
config = EDAConfig(
    # Auto-sampling
    enable_auto_sampling=True,
    auto_sample_threshold=100000,
    sampling_method='adaptive',

    # Memory optimization
    optimize_memory=True,
    memory_check_enabled=True,

    # Parallel processing
    n_jobs=-1,  # Use all cores

    # Reduce visualization samples
    n_samples_viz=5000,

    verbose=True
)

eda = SimplusEDA(config=config)
eda.analyze(large_df)
```

Features:
- **Automatic sampling**: Intelligent sampling strategies
- **Memory optimization**: Reduce DataFrame memory by 50-90%
- **Chunked processing**: Process files larger than memory
- **Dask backend**: Distributed computing
- **Parallel processing**: Multi-core analysis

### Parallel Processing

Speed up analysis with parallel processing:

```python
from simplus_eda import SimplusEDA, EDAConfig

# Use all CPU cores
config = EDAConfig(
    n_jobs=-1,      # -1 = all cores, -2 = all but one
    verbose=True    # Show progress
)

eda = SimplusEDA(config=config)
eda.analyze(df)
```

Performance benefits:
- **50+ columns**: 5-6x speedup with 8 cores
- **Statistical tests**: Parallelized column-wise
- **Correlation analysis**: Parallelized pair-wise
- **Outlier detection**: Parallelized per column

## Command-Line Interface

### Basic Commands

```bash
# Analyze a dataset
simplus-eda analyze data.csv

# Specify output path
simplus-eda analyze data.csv -o report.html

# Generate JSON output
simplus-eda analyze data.csv -o results.json -f json

# Multiple formats
simplus-eda analyze data.csv -o report --formats html --formats json
```

### Common Options

```bash
# Quick analysis (skip tests)
simplus-eda analyze data.csv --quick -v

# Custom configuration
simplus-eda analyze data.csv \
  --correlation-threshold 0.8 \
  --outlier-method isolation_forest \
  --n-jobs -1 \
  --verbose

# Use configuration file
simplus-eda analyze data.csv --config my_config.json

# Custom report metadata
simplus-eda analyze data.csv \
  --title "Sales Analysis Q4 2024" \
  --author "Data Team" \
  --company "Your Company"
```

### Configuration Management

```bash
# Generate configuration template
simplus-eda init-config -o my_config.json

# Use preset profiles
simplus-eda init-config --profile thorough -o config.json

# Validate configuration
simplus-eda validate-config my_config.json

# Show framework information
simplus-eda info
```

### Batch Processing

```bash
# Process multiple files
for file in data/*.csv; do
  simplus-eda analyze "$file" -o "reports/$(basename "$file" .csv)_report.html"
done
```

## Configuration

### Configuration Parameters

```python
from simplus_eda import EDAConfig

config = EDAConfig(
    # Analysis settings
    enable_statistical_tests=True,
    enable_visualizations=True,
    verbose=False,

    # Thresholds
    correlation_threshold=0.7,      # Correlation detection threshold
    missing_threshold=0.5,          # Missing value warning threshold
    significance_level=0.05,        # Statistical significance level

    # Methods
    outlier_method='iqr',          # 'iqr', 'zscore', 'isolation_forest'
    distribution_test_method='shapiro',  # 'shapiro', 'ks', 'anderson'

    # Performance
    n_samples_viz=10000,           # Max samples for visualization
    n_jobs=1,                      # Number of parallel jobs (-1 for all)
    random_state=42,               # Random seed

    # Large datasets
    enable_auto_sampling=False,
    auto_sample_threshold=100000,
    sampling_method='random',

    # Caching
    enable_cache=True,
    cache_backend='memory',        # 'memory' or 'disk'
    cache_ttl=None,               # Time-to-live (None = never expires)

    # Progress tracking
    enable_progress=True,
    use_tqdm=True
)
```

### Preset Configurations

```python
# Default (balanced)
config = EDAConfig.get_default()

# Quick analysis (fast, minimal tests)
config = EDAConfig.get_quick()

# Thorough analysis (comprehensive)
config = EDAConfig.get_thorough()

# Large dataset optimized
config = EDAConfig.get_large_dataset()
```

### Saving and Loading

```python
# Save configuration
config = EDAConfig(correlation_threshold=0.8, outlier_method='zscore')
config.to_json('my_config.json')

# Load configuration
config = EDAConfig.from_json('my_config.json')
eda = SimplusEDA(config=config)

# Or load directly
eda = SimplusEDA.from_config_file('my_config.json')
```

### Updating Configuration

```python
# Update existing configuration
eda = SimplusEDA()
eda.update_config(correlation_threshold=0.9, verbose=True)
eda.analyze(df)  # Uses updated config
```

## Output Formats

### HTML Report

Professional, interactive HTML report with:
- Executive summary
- Data quality score
- Statistical analysis
- Correlation matrices
- Distribution plots
- Outlier detection
- Key insights and recommendations

```python
eda.generate_report('report.html', format='html')
```

### JSON Report

Machine-readable output with all analysis results:

```python
eda.generate_report('results.json', format='json')

# Or use convenience method
eda.export_results('results.json')
```

JSON structure:
```json
{
  "metadata": {...},
  "quality": {...},
  "statistics": {...},
  "correlations": {...},
  "outliers": {...},
  "insights": {...}
}
```

### PDF Report

Print-friendly version of HTML report (requires weasyprint):

```python
eda.generate_report('report.pdf', format='pdf')
```

## Best Practices

### 1. Start Simple, Then Customize

```python
# First run: Use defaults
eda = SimplusEDA()
eda.analyze(df)

# Review results, then customize
eda = SimplusEDA(
    correlation_threshold=0.8,  # Adjust based on first run
    outlier_method='isolation_forest',
    verbose=True
)
```

### 2. Check Data Quality First

```python
eda = SimplusEDA()
eda.analyze(df)

quality = eda.get_quality_score()
if quality < 80:
    print("Data quality issues detected!")
    insights = eda.get_insights()
    for issue in insights.get('quality_issues', []):
        print(f"  - {issue}")
```

### 3. Use Appropriate Configuration

```python
# Small dataset (< 10K rows)
eda = SimplusEDA(config=EDAConfig.get_default())

# Medium dataset (10K - 100K rows)
eda = SimplusEDA(n_jobs=-1, verbose=True)

# Large dataset (> 100K rows)
eda = SimplusEDA(config=EDAConfig.get_large_dataset())
```

### 4. Save Configurations for Reuse

```python
# Team standard configuration
team_config = EDAConfig(
    correlation_threshold=0.8,
    outlier_method='isolation_forest',
    missing_threshold=0.05,
    n_jobs=-1
)
team_config.to_json('team_config.json')

# Everyone uses the same config
eda = SimplusEDA.from_config_file('team_config.json')
```

### 5. Document Analysis with Reports

```python
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

eda.generate_report(
    f'reports/analysis_{timestamp}.html',
    title=f'Analysis - {timestamp}',
    author='Data Pipeline',
    include_visualizations=True
)
```

### 6. Handle Large Datasets Properly

```python
# Check memory first
from simplus_eda.utils import get_system_memory_info

mem_info = get_system_memory_info()
print(f"Available: {mem_info['available_gb']:.2f} GB")

# Configure appropriately
if len(df) > 100000:
    config = EDAConfig.get_large_dataset()
else:
    config = EDAConfig.get_default()

eda = SimplusEDA(config=config)
```

## Troubleshooting

### Issue: Analysis Takes Too Long

**Solution**: Enable parallel processing and sampling

```python
config = EDAConfig(
    n_jobs=-1,                  # Use all cores
    enable_auto_sampling=True,
    auto_sample_threshold=50000,
    n_samples_viz=5000
)
```

### Issue: Out of Memory Errors

**Solution**: Enable memory optimization and sampling

```python
from simplus_eda.utils import optimize_dataframe_memory, smart_sample

# Optimize memory
df, report = optimize_dataframe_memory(df, verbose=True)
print(f"Memory saved: {report['memory_saved_mb']:.2f} MB")

# Sample if still too large
if len(df) > 100000:
    df, metadata = smart_sample(df, target_size=50000, method='adaptive')

# Use quick mode
eda = SimplusEDA()
eda.analyze(df, quick=True)
```

### Issue: Configuration Not Taking Effect

**Solution**: Call `analyze()` after configuration changes

```python
# Wrong
eda = SimplusEDA(correlation_threshold=0.9)
correlations = eda.get_correlations()  # Still uses default!

# Correct
eda = SimplusEDA(correlation_threshold=0.9)
eda.analyze(df)  # Configuration applied here
correlations = eda.get_correlations()  # Now uses 0.9
```

### Issue: No Results Available

**Solution**: Always call `analyze()` before accessing results

```python
# Wrong
eda = SimplusEDA()
eda.generate_report('report.html')  # Error!

# Correct
eda = SimplusEDA()
eda.analyze(df)  # Required!
eda.generate_report('report.html')
```

### Issue: PDF Generation Fails

**Solution**: Install weasyprint

```bash
pip install weasyprint
```

### Issue: Progress Bars Not Showing

**Solution**: Install tqdm and enable progress

```bash
pip install tqdm
```

```python
config = EDAConfig(enable_progress=True, use_tqdm=True)
eda = SimplusEDA(config=config)
```

### Issue: Import Errors for Advanced Features

**Solution**: Install optional dependencies

```bash
# For time series
pip install statsmodels>=0.13.0

# For large datasets
pip install dask[complete]

# For PDF reports
pip install weasyprint
```

## Next Steps

- **Explore Examples**: Check the [examples/](../examples/) directory for comprehensive examples
- **Read Architecture Guide**: Understand the framework design in [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- **API Reference**: Detailed API documentation in module docstrings
- **Contributing**: See CONTRIBUTING.md for contribution guidelines

## Support

- **GitHub Issues**: https://github.com/simplus-ai/simplus-eda-framework/issues
- **Documentation**: Full documentation in the repository
- **Email**: contact@simplusai.com

---

**Ready to get started?**

```python
from simplus_eda import SimplusEDA
import pandas as pd

df = pd.read_csv('your_data.csv')
eda = SimplusEDA()
eda.analyze(df)
eda.generate_report('report.html')
```

That's all you need! 🎉
