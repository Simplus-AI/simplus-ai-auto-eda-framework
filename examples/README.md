# Simplus EDA Framework - Examples

This directory contains comprehensive examples demonstrating how to use the Simplus AI Auto EDA Framework for various data analysis tasks.

## 📁 Directory Contents

### Data Generators
- **[data_generators.py](data_generators.py)** - Realistic data generators for testing and demonstration
  - E-commerce sales data
  - Customer churn data
  - Time series sales data
  - Financial metrics data
  - IoT sensor data
  - Healthcare patient records
  - Marketing campaign data

### Example Scripts

#### 1. Quick Start Examples
**[01_quick_start_examples.py](01_quick_start_examples.py)**

Learn the basics of the framework:
- One-line analysis
- Basic workflow
- Viewing summaries
- Basic customization
- JSON export
- Multiple report formats

```bash
python examples/01_quick_start_examples.py
```

#### 2. Advanced Analysis Examples
**[02_advanced_analysis_examples.py](02_advanced_analysis_examples.py)**

Explore advanced analytical features:
- Statistical hypothesis testing
- Time series analysis with forecasting
- Feature engineering suggestions
- Multi-method anomaly detection
- Comprehensive analysis combining all features

```bash
python examples/02_advanced_analysis_examples.py
```

#### 3. Real-World Use Cases
**[03_real_world_use_cases.py](03_real_world_use_cases.py)**

See how to apply the framework to real-world scenarios:
- E-commerce sales analysis
- Customer churn prediction preparation
- Healthcare patient monitoring
- IoT sensor monitoring
- Marketing campaign optimization

```bash
python examples/03_real_world_use_cases.py
```

## 🚀 Getting Started

### Prerequisites

Ensure you have the framework installed:

```bash
cd /path/to/simplus-ai-auto-eda-framework-1
pip install -e .
```

### Running the Examples

#### Run All Examples at Once

Each script is self-contained and runs multiple examples:

```bash
# Quick start examples
python examples/01_quick_start_examples.py

# Advanced analysis
python examples/02_advanced_analysis_examples.py

# Real-world use cases
python examples/03_real_world_use_cases.py
```

#### Generate Sample Datasets

Generate all sample datasets for manual exploration:

```bash
python -c "from examples.data_generators import generate_all_datasets; generate_all_datasets('./sample_data')"
```

This creates:
- `sample_data/sales_data.csv`
- `sample_data/customer_churn.csv`
- `sample_data/time_series_sales.csv`
- `sample_data/financial_data.csv`
- `sample_data/sensor_data.csv`
- `sample_data/healthcare_data.csv`
- `sample_data/marketing_data.csv`

## 📊 Example Outputs

All examples generate reports in the `outputs/` directory:

- **HTML Reports**: Interactive, visual reports for human consumption
- **JSON Reports**: Machine-readable data for APIs and pipelines
- **Analysis Summaries**: Console output showing key findings

## 💡 Quick Examples

### One-Line Analysis

```python
from simplus_eda import quick_analysis
import pandas as pd

df = pd.read_csv('data.csv')
quick_analysis(df, 'report.html')
```

### Basic Workflow

```python
from simplus_eda import SimplusEDA
import pandas as pd

df = pd.read_csv('data.csv')
eda = SimplusEDA()
eda.analyze(df)
print(eda.summary())
eda.generate_report('report.html')
```

### Statistical Testing

```python
from simplus_eda import StatisticalTestsManager
import pandas as pd

df = pd.read_csv('data.csv')
test_mgr = StatisticalTestsManager()

result = test_mgr.compare_groups(
    df,
    value_col='sales',
    group_col='region'
)

print(f"P-value: {result['test_result']['p_value']:.6f}")
print(f"Significant: {result['significant']}")
```

### Time Series Analysis

```python
from simplus_eda import TimeSeriesAnalyzer
import pandas as pd

df = pd.read_csv('time_series.csv')
series = df.set_index('date')['value']

ts_analyzer = TimeSeriesAnalyzer(
    enable_forecasting=True,
    forecast_steps=30
)

results = ts_analyzer.analyze(series, seasonal_period=7)
forecast = results['forecasting']['forecast']['forecast']
print(f"30-day forecast: {forecast}")
```

### Anomaly Detection

```python
from simplus_eda import AnomalyDetectionManager
import pandas as pd

df = pd.read_csv('sensor_data.csv')
anomaly_mgr = AnomalyDetectionManager()

# Detect anomalies using multiple methods
results = anomaly_mgr.detect_multivariate(
    df[['temp', 'humidity', 'pressure']],
    methods=['isolation_forest', 'lof']
)

for method, result in results.items():
    print(f"{method}: {result.n_anomalies} anomalies")
```

### Feature Engineering

```python
from simplus_eda import FeatureEngineeringManager
import pandas as pd

df = pd.read_csv('ml_data.csv')
fe_mgr = FeatureEngineeringManager()

results = fe_mgr.analyze(df, target_col='target')

# View feature interaction suggestions
for interaction in results['interactions'][:5]:
    print(f"{interaction['feature1']} × {interaction['feature2']}")

# Generate code for top features
code = fe_mgr.generate_code(results, n_top=5)
print(code)
```

## 📖 Data Generator Details

### E-Commerce Sales Data
**Function**: `generate_sales_data(n_samples=1000)`

Features:
- Transaction details (ID, date, category, region)
- Customer information (age, segment)
- Product metrics (price, quantity, discount)
- Revenue calculations
- Missing values (15% in discount, 10% in customer_age)
- Outliers (5% in revenue)
- Duplicate records (2%)

**Use Cases**: Revenue analysis, sales forecasting, data quality assessment

### Customer Churn Data
**Function**: `generate_customer_churn_data(n_samples=2000)`

Features:
- Customer demographics
- Service details (contract type, internet service)
- Usage metrics (tenure, monthly charges, total charges)
- Binary churn indicator (25% churn rate)
- Missing values in service fields

**Use Cases**: Churn prediction, customer segmentation, ML model preparation

### Time Series Sales Data
**Function**: `generate_time_series_sales(n_days=365)`

Features:
- Daily sales data with trend
- Weekly seasonality (weekend effects)
- Monthly seasonality
- Multiple product categories
- Random noise and anomalies

**Use Cases**: Trend analysis, seasonality detection, forecasting, time series decomposition

### Financial Data
**Function**: `generate_financial_data(n_samples=500)`

Features:
- Financial ratios (ROE, debt-to-equity)
- Income statement items (revenue, net income)
- Balance sheet items (assets, liabilities, equity)
- Industry categorization
- Anomalies (potential fraud indicators)

**Use Cases**: Financial analysis, anomaly detection, industry comparison

### IoT Sensor Data
**Function**: `generate_sensor_data(n_samples=10000)`

Features:
- Multi-sensor readings (temperature, humidity, pressure, vibration)
- Timestamp data
- Daily cycles and sensor drift
- Battery levels
- Status flags
- Missing values (5% due to connectivity)
- Outliers (2% sensor malfunctions)

**Use Cases**: Sensor monitoring, predictive maintenance, anomaly detection

### Healthcare Patient Data
**Function**: `generate_healthcare_data(n_samples=1500)`

Features:
- Patient demographics (age, gender)
- Vital signs (blood pressure, heart rate, BMI)
- Lab results (cholesterol, glucose)
- Medical history (diabetes, hypertension, heart disease)
- Lifestyle factors (smoking, exercise)
- Risk scores and outcomes
- Missing values in lab results

**Use Cases**: Patient risk stratification, readmission prediction, health monitoring

### Marketing Campaign Data
**Function**: `generate_marketing_data(n_samples=5000)`

Features:
- Campaign attributes (channel, type, budget)
- Performance metrics (impressions, clicks, conversions)
- Financial metrics (CPC, CPA, ROI)
- Revenue tracking
- Engagement scores
- Viral campaigns (outliers)

**Use Cases**: Campaign optimization, ROI analysis, channel performance comparison

## 🎯 Learning Path

### Beginner
1. Start with `01_quick_start_examples.py` to understand basic usage
2. Generate sample datasets using `data_generators.py`
3. Try modifying the quick start examples with your own data

### Intermediate
1. Explore `02_advanced_analysis_examples.py` for specialized analyses
2. Learn about statistical testing and hypothesis validation
3. Experiment with feature engineering suggestions

### Advanced
1. Study `03_real_world_use_cases.py` for production patterns
2. Combine multiple analysis techniques
3. Integrate the framework into your data pipelines

## 🔧 Customization

All examples can be customized by modifying:

- **Sample sizes**: Change `n_samples` parameter in generators
- **Configuration**: Adjust `EDAConfig` parameters
- **Output paths**: Modify report output locations
- **Data characteristics**: Adjust generator parameters (missing rates, outlier percentages, etc.)

### Example Customization

```python
from examples.data_generators import generate_sales_data
from simplus_eda import SimplusEDA, EDAConfig

# Generate larger dataset
df = generate_sales_data(n_samples=10000, random_state=42)

# Custom configuration
config = EDAConfig(
    correlation_threshold=0.8,
    outlier_method='isolation_forest',
    missing_threshold=0.1,
    n_jobs=-1,  # Use all CPU cores
    verbose=True
)

eda = SimplusEDA(config=config)
eda.analyze(df)
eda.generate_report('custom_report.html')
```

## 📚 Additional Resources

- **[Usage Guide](../docs/USAGE_GUIDE.md)** - Comprehensive framework documentation
- **[README](../README.md)** - Framework overview and features
- **[API Documentation](../docs/)** - Detailed API reference

## 🤝 Contributing Examples

Have a great use case or example? Contributions are welcome!

1. Create a new example script following the existing pattern
2. Add realistic data generators if needed
3. Include clear documentation and comments
4. Submit a pull request

## 💬 Questions?

If you have questions about the examples:
- Check the main [Usage Guide](../docs/USAGE_GUIDE.md)
- Review the inline comments in example scripts
- Open an issue on GitHub

---

**Happy Analyzing! 🚀**
