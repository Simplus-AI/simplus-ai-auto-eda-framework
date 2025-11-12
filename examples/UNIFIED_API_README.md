# SimplusEDA - Unified API Documentation

## 🎯 Quick Start

The simplest way to perform EDA on your data:

```python
from simplus_eda import quick_analysis
import pandas as pd

df = pd.read_csv('your_data.csv')
quick_analysis(df, 'report.html')
```

That's it! One function call analyzes your data and generates a comprehensive HTML report.

## 📦 Installation

```bash
pip install -e .
```

Optional dependencies for PDF reports:
```bash
pip install weasyprint
```

## 🚀 Why Use the Unified API?

### Before (Manual Integration)

```python
from simplus_eda.core.config import EDAConfig
from simplus_eda.core.analyzer import EDAAnalyzer
from simplus_eda.core.report import ReportGenerator

# Lots of manual wiring...
config = EDAConfig(correlation_threshold=0.8)
analyzer = EDAAnalyzer(config=config)
results = analyzer.analyze(df)
report_gen = ReportGenerator(config={'title': 'Report'})
report_gen.generate_report(results=results, data=df, output_path='report.html')
```

### After (Unified API)

```python
from simplus_eda import SimplusEDA

eda = SimplusEDA(correlation_threshold=0.8)
eda.analyze(df)
eda.generate_report('report.html', title='Report')
```

**Benefits:**
- ✅ 70% less code
- ✅ Automatic state management
- ✅ Cleaner, more Pythonic syntax
- ✅ Still fully customizable
- ✅ Access to all features

## 📖 Usage Examples

### Example 1: Basic Workflow

```python
from simplus_eda import SimplusEDA

# Create instance
eda = SimplusEDA()

# Analyze data
eda.analyze(df)

# View summary
print(eda.summary())

# Generate report
eda.generate_report('report.html')
```

### Example 2: Custom Configuration

```python
from simplus_eda import SimplusEDA

eda = SimplusEDA(
    correlation_threshold=0.8,
    outlier_method='zscore',
    missing_threshold=0.05,
    verbose=True
)

eda.analyze(df)

# Access specific results
quality_score = eda.get_quality_score()
correlations = eda.get_correlations()
insights = eda.get_insights()

# Generate customized report
eda.generate_report(
    'report.html',
    title='Sales Analysis Q4 2024',
    author='Data Team',
    company='Your Company'
)
```

### Example 3: Multiple Formats

```python
from simplus_eda import SimplusEDA

eda = SimplusEDA()
eda.analyze(df)

# Generate different formats
eda.generate_report('report.html', format='html')
eda.generate_report('report.json', format='json')
eda.generate_report('report.pdf', format='pdf')  # Requires weasyprint

# Or use convenience method for JSON
eda.export_results('results.json')
```

### Example 4: Advanced Configuration

```python
from simplus_eda import SimplusEDA, EDAConfig

# Create detailed configuration
config = EDAConfig(
    # Analysis settings
    enable_statistical_tests=True,
    enable_visualizations=True,

    # Thresholds
    correlation_threshold=0.7,
    missing_threshold=0.1,
    significance_level=0.05,

    # Methods
    outlier_method='isolation_forest',
    distribution_test_method='shapiro',

    # Performance
    n_samples_viz=10000,
    n_jobs=-1,
    verbose=True
)

# Use configuration
eda = SimplusEDA(config=config)
eda.analyze(df)

# Save configuration for reuse
eda.save_config('my_config.json')
```

### Example 5: Load Configuration from File

```python
from simplus_eda import SimplusEDA

# Load saved configuration
eda = SimplusEDA.from_config_file('my_config.json')
eda.analyze(df)
eda.generate_report('report.html')
```

### Example 6: Dynamic Configuration Updates

```python
from simplus_eda import SimplusEDA

# Start with defaults
eda = SimplusEDA()
eda.analyze(df)
print(f"Correlations (default): {len(eda.get_correlations())}")

# Update configuration
eda.update_config(correlation_threshold=0.9)
eda.analyze(df)
print(f"Correlations (strict): {len(eda.get_correlations())}")
```

### Example 7: Convenience Functions

```python
from simplus_eda import quick_analysis, analyze_and_report

# One-liner
quick_analysis(df, 'quick_report.html')

# With results
results, report_path = analyze_and_report(
    df,
    'report.html',
    title='My Analysis'
)

# Access results
quality = results['quality']['quality_score']['overall_score']
print(f"Quality: {quality:.1f}%")
```

## 🔧 API Reference

### SimplusEDA Class

#### Constructor
```python
SimplusEDA(
    config: Optional[Union[Dict, EDAConfig]] = None,
    **kwargs  # Any configuration parameter
)
```

#### Core Methods

**analyze(data, quick=False, inplace=True)**
- Perform comprehensive EDA analysis
- Returns: Analysis results dictionary
- Parameters:
  - `data`: pandas DataFrame
  - `quick`: Skip time-consuming analyses
  - `inplace`: Store results internally

**generate_report(output_path, format='html', **kwargs)**
- Generate professional report
- Returns: Path to generated report
- Parameters:
  - `output_path`: Where to save the report
  - `format`: 'html', 'json', or 'pdf'
  - `title`: Custom report title
  - `author`: Report author
  - `company`: Organization name
  - `include_visualizations`: Include charts
  - `include_data_preview`: Include data preview
  - `max_preview_rows`: Preview rows (default: 10)

**export_results(output_path)**
- Export results to JSON
- Returns: Path to JSON file

**summary()**
- Get text summary of analysis
- Returns: Formatted string

**get_quality_score()**
- Get overall data quality score
- Returns: Float (0-100)

**get_correlations(threshold=None)**
- Get strong correlations
- Returns: List of correlation dictionaries

**get_insights()**
- Get auto-generated insights
- Returns: Dictionary with categorized insights

**update_config(**kwargs)**
- Update configuration parameters
- Returns: Self (for chaining)

**save_config(path)**
- Save configuration to JSON file

#### Class Methods

**from_config_file(path)**
- Create instance from config file
- Returns: New SimplusEDA instance

#### Properties

- `analyzer`: Access EDAAnalyzer
- `report_generator`: Access ReportGenerator
- `config`: Access EDAConfig
- `results`: Access analysis results
- `data`: Access analyzed DataFrame

### Convenience Functions

**quick_analysis(data, output_path='eda_report.html', **config_kwargs)**
- One-line analysis and report generation
- Returns: Report path

**analyze_and_report(data, output_path, config=None, format='html', **report_kwargs)**
- Analyze and report in one call
- Returns: Tuple of (results, report_path)

## 📊 Configuration Options

### Analysis Settings
- `enable_statistical_tests`: bool (default: True)
- `enable_visualizations`: bool (default: True)
- `verbose`: bool (default: False)

### Thresholds
- `correlation_threshold`: float 0-1 (default: 0.7)
- `missing_threshold`: float 0-1 (default: 0.5)
- `significance_level`: float 0-1 (default: 0.05)

### Methods
- `outlier_method`: 'iqr', 'zscore', 'isolation_forest' (default: 'iqr')
- `distribution_test_method`: 'shapiro', 'ks', 'anderson' (default: 'shapiro')

### Performance
- `n_samples_viz`: int (default: 10000)
- `max_categorical_values`: int (default: 50)
- `n_jobs`: int (default: 1, -1 for all CPUs)
- `random_state`: int (default: 42)

## 🎓 Learning Path

1. **Start with one-liner:**
   ```python
   from simplus_eda import quick_analysis
   quick_analysis(df, 'report.html')
   ```

2. **Learn basic workflow:**
   ```python
   eda = SimplusEDA()
   eda.analyze(df)
   eda.generate_report('report.html')
   ```

3. **Add configuration:**
   ```python
   eda = SimplusEDA(correlation_threshold=0.8, verbose=True)
   ```

4. **Access results:**
   ```python
   print(eda.summary())
   quality = eda.get_quality_score()
   insights = eda.get_insights()
   ```

5. **Use advanced features:**
   ```python
   config = EDAConfig(...)
   eda = SimplusEDA(config=config)
   eda.save_config('config.json')
   ```

## 📚 Documentation

- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Complete integration guide with examples
- **[REPORT_GENERATOR_README.md](REPORT_GENERATOR_README.md)** - Report generation documentation
- **[examples/unified_api_examples.py](examples/unified_api_examples.py)** - 10 comprehensive examples
- **[examples/report_generation.py](examples/report_generation.py)** - Report generation examples

## 🔍 How It Works

The SimplusEDA class provides a unified interface by:

1. **Automatic Component Management**
   - Creates and configures all components automatically
   - Manages component lifecycle and dependencies

2. **State Persistence**
   - Stores analysis results internally
   - Reuses results across multiple operations
   - Eliminates manual result passing

3. **Configuration Flow**
   - Configuration flows seamlessly through all components
   - Single source of truth for all settings
   - Consistent behavior across the framework

4. **Smart Initialization**
   - Lazy initialization of expensive components
   - Only creates what you actually use
   - Optimizes memory and performance

## 🎯 Use Cases

### Data Science Projects
```python
eda = SimplusEDA(correlation_threshold=0.8)
eda.analyze(train_df)
eda.generate_report('train_analysis.html')

# Check quality before modeling
if eda.get_quality_score() < 80:
    print("Data quality issues detected!")
    print(eda.get_insights()['quality_issues'])
```

### Automated Pipelines
```python
def analyze_pipeline(data_path):
    df = pd.read_csv(data_path)
    eda = SimplusEDA(verbose=False)
    eda.analyze(df)

    return {
        'quality': eda.get_quality_score(),
        'report': eda.generate_report(f'{data_path}.html')
    }
```

### Interactive Analysis
```python
eda = SimplusEDA()
eda.analyze(df)

# Explore interactively
print(eda.summary())
print(f"Quality: {eda.get_quality_score():.1f}%")

# Adjust and re-analyze
eda.update_config(correlation_threshold=0.9)
eda.analyze(df)
```

### Report Generation
```python
eda = SimplusEDA()
eda.analyze(df)

# Generate multiple formats
eda.generate_report('report.html', format='html')
eda.export_results('results.json')

# Customized reports
eda.generate_report(
    'executive_report.html',
    title='Q4 2024 Data Analysis',
    author='Data Science Team',
    company='Your Company',
    include_visualizations=True
)
```

## 🤝 Backward Compatibility

The old API still works:

```python
# Old API (still supported)
from simplus_eda.core.config import EDAConfig
from simplus_eda.core.analyzer import EDAAnalyzer
from simplus_eda.core.report import ReportGenerator

config = EDAConfig()
analyzer = EDAAnalyzer(config=config)
results = analyzer.analyze(df)
```

But we recommend using the new unified API for:
- Cleaner code
- Better maintainability
- Improved user experience
- Future-proof applications

## 💡 Best Practices

1. **Use SimplusEDA for all new projects**
2. **Save configurations for team standardization**
3. **Validate data quality before modeling**
4. **Generate reports for documentation**
5. **Use appropriate configuration for your domain**

## 🐛 Troubleshooting

**Problem: Configuration not applied**
```python
# Wrong
eda = SimplusEDA(correlation_threshold=0.9)
correlations = eda.get_correlations()  # analyze() not called yet!

# Correct
eda = SimplusEDA(correlation_threshold=0.9)
eda.analyze(df)  # Configuration applied here
correlations = eda.get_correlations()
```

**Problem: No results available**
```python
# Wrong
eda = SimplusEDA()
eda.generate_report('report.html')  # Error!

# Correct
eda = SimplusEDA()
eda.analyze(df)  # Must analyze first!
eda.generate_report('report.html')
```

**Problem: Memory issues**
```python
# Solution 1: Sample for visualization
sample = df.sample(n=10000)
eda.generate_report('report.html', data=sample)

# Solution 2: Disable visualizations
eda.generate_report('report.html', include_visualizations=False)
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/simplus-ai/simplus-eda/issues)
- **Documentation**: See guides in the repository
- **Examples**: Check [examples/](examples/) directory

## 📝 License

See LICENSE file for details.

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
