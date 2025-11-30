# Simplus EDA Framework - Quick Reference

## 🚀 Quick Start

### One-Line Analysis
```python
from simplus_eda import quick_analysis
quick_analysis(df, 'report.html')
```

### Basic Workflow
```python
from simplus_eda import SimplusEDA

eda = SimplusEDA()
eda.analyze(df)
print(eda.summary())
eda.generate_report('report.html')
```

## 📊 Data Generators

### Generate All Sample Datasets
```python
from examples.data_generators import generate_all_datasets
generate_all_datasets('./sample_data')
```

### Individual Generators
```python
from examples.data_generators import (
    generate_sales_data,              # E-commerce sales
    generate_customer_churn_data,     # Customer churn
    generate_time_series_sales,       # Time series with seasonality
    generate_financial_data,          # Financial metrics
    generate_sensor_data,             # IoT sensor readings
    generate_healthcare_data,         # Patient records
    generate_marketing_data           # Campaign performance
)

df = generate_sales_data(n_samples=1000, random_state=42)
```

## ⚙️ Configuration

### Simple Configuration
```python
eda = SimplusEDA(
    correlation_threshold=0.7,
    outlier_method='isolation_forest',
    verbose=True
)
```

### Advanced Configuration
```python
from simplus_eda import EDAConfig

config = EDAConfig(
    enable_statistical_tests=True,
    correlation_threshold=0.6,
    outlier_method='isolation_forest',
    n_jobs=-1,  # Use all cores
    verbose=True
)
eda = SimplusEDA(config=config)
```

### Preset Configurations
```python
config = EDAConfig.get_quick()          # Fast analysis
config = EDAConfig.get_thorough()       # Comprehensive
config = EDAConfig.get_large_dataset()  # Large data optimized
```

## 📈 Analysis Features

### Get Quality Score
```python
quality = eda.get_quality_score()
print(f"Quality: {quality:.1f}%")
```

### Get Correlations
```python
correlations = eda.get_correlations(threshold=0.7)
for corr in correlations:
    print(f"{corr['feature1']} <-> {corr['feature2']}: {corr['correlation']:.3f}")
```

### Get Insights
```python
insights = eda.get_insights()
for category, items in insights.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  • {item}")
```

## 🧪 Statistical Tests

### Compare Groups
```python
from simplus_eda import StatisticalTestsManager

test_mgr = StatisticalTestsManager()
result = test_mgr.compare_groups(
    df,
    value_col='sales',
    group_col='region',
    post_hoc=True
)
print(f"P-value: {result['test_result']['p_value']:.6f}")
```

### Test Categorical Association
```python
result = test_mgr.test_categorical_association(
    df,
    var1='category',
    var2='outcome'
)
print(f"Cramér's V: {result['effect_size']['cramers_v']:.4f}")
```

## 📉 Time Series Analysis

### Analyze Time Series
```python
from simplus_eda import TimeSeriesAnalyzer

ts_analyzer = TimeSeriesAnalyzer(
    enable_forecasting=True,
    forecast_steps=30
)

series = df.set_index('date')['value']
results = ts_analyzer.analyze(series, seasonal_period=7)

# Access results
print(f"Trend: {results['trend']['mann_kendall']['trend']}")
forecast = results['forecasting']['forecast']['forecast']
```

## 🔍 Anomaly Detection

### Univariate Anomalies
```python
from simplus_eda import AnomalyDetectionManager

anomaly_mgr = AnomalyDetectionManager()
results = anomaly_mgr.detect_univariate(
    df,
    methods=['iqr', 'zscore', 'mad']
)
```

### Multivariate Anomalies
```python
results = anomaly_mgr.detect_multivariate(
    df[numeric_cols],
    methods=['isolation_forest', 'lof']
)
```

### Ensemble Detection
```python
result = anomaly_mgr.ensemble_detection(
    df[numeric_cols],
    voting_threshold=0.5
)
print(f"Anomalies: {result.n_anomalies} ({result.anomaly_percentage:.2f}%)")
```

### Explain Anomalies
```python
explanations = anomaly_mgr.explain_anomalies(
    df[numeric_cols],
    result,
    top_n=3
)
```

## 🛠️ Feature Engineering

### Get Suggestions
```python
from simplus_eda import FeatureEngineeringManager

fe_mgr = FeatureEngineeringManager()
results = fe_mgr.analyze(df, target_col='target')

# View interactions
for interaction in results['interactions'][:5]:
    print(f"{interaction['feature1']} × {interaction['feature2']}")

# View polynomials
for poly in results['polynomials'][:5]:
    print(f"{poly['feature']}^{poly['degree']}")
```

### Generate Code
```python
code = fe_mgr.generate_code(results, n_top=5)
print(code)
```

## 📄 Report Generation

### HTML Report
```python
eda.generate_report(
    'report.html',
    title='My Analysis',
    author='Data Team',
    include_visualizations=True
)
```

### JSON Export
```python
eda.export_results('results.json')
# or
eda.generate_report('results.json', format='json')
```

### PDF Report (requires weasyprint)
```python
eda.generate_report('report.pdf', format='pdf')
```

### Multiple Formats
```python
eda.generate_report('report.html', format='html')
eda.generate_report('report.json', format='json')
eda.generate_report('report.pdf', format='pdf')
```

## 💾 Configuration Management

### Save Configuration
```python
eda.save_config('my_config.json')
```

### Load Configuration
```python
eda = SimplusEDA.from_config_file('my_config.json')
```

### Update Configuration
```python
eda.update_config(correlation_threshold=0.9)
eda.analyze(df)  # Re-analyze with new settings
```

## 🎯 Common Patterns

### Quick Exploration
```python
from simplus_eda import quick_analysis
quick_analysis(df, 'report.html')
```

### Data Quality Check
```python
eda = SimplusEDA()
eda.analyze(df)
if eda.get_quality_score() < 80:
    print("Quality issues detected!")
    for issue in eda.get_insights()['quality_issues']:
        print(f"  • {issue}")
```

### ML Preparation
```python
eda = SimplusEDA(
    correlation_threshold=0.8,
    outlier_method='isolation_forest'
)
eda.analyze(df)

# Check for multicollinearity
high_corr = eda.get_correlations(threshold=0.8)

# Get feature engineering suggestions
fe_mgr = FeatureEngineeringManager()
suggestions = fe_mgr.analyze(df, target_col='target')

# Export for pipeline
eda.export_results('ml_prep_results.json')
```

### Production Pipeline
```python
from simplus_eda import SimplusEDA, EDAConfig

config = EDAConfig.from_json('production_config.json')
eda = SimplusEDA(config=config)
eda.analyze(df)

quality = eda.get_quality_score()
if quality < 85:
    send_alert(f"Data quality below threshold: {quality:.1f}%")

eda.generate_report(f'reports/daily_report_{date}.html')
```

## 🏃 Running Examples

### Run Complete Workflow
```bash
python examples/00_complete_workflow.py
```

### Run All Examples
```bash
python examples/run_all_examples.py
```

### Individual Examples
```bash
python examples/01_quick_start_examples.py
python examples/02_advanced_analysis_examples.py
python examples/03_real_world_use_cases.py
```

## 📚 Getting Help

### View Summary
```python
print(eda.summary())
```

### Check Configuration
```python
print(eda.config)
```

### Access Raw Results
```python
results = eda.results
print(results.keys())
```

## ⚡ Performance Tips

### Large Datasets
```python
config = EDAConfig.get_large_dataset()
eda = SimplusEDA(config=config)
```

### Parallel Processing
```python
eda = SimplusEDA(n_jobs=-1)  # Use all cores
```

### Quick Mode
```python
eda.analyze(df, quick=True)  # Skip expensive tests
```

### Sampling
```python
from simplus_eda.utils import smart_sample
df_sample, metadata = smart_sample(df, target_size=10000)
```

## 🔗 Useful Links

- [Full Documentation](../docs/USAGE_GUIDE.md)
- [Examples README](README.md)
- [Main README](../README.md)
- [Data Generators](data_generators.py)

---

**💡 Pro Tips:**
- Start with `quick_analysis()` for exploration
- Use `verbose=True` to see progress
- Save configurations with `save_config()` for reuse
- Check quality score before making decisions
- Generate multiple report formats for different audiences
- Use parallel processing (`n_jobs=-1`) for large datasets
