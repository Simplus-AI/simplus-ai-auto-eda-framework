# 🚀 Simplus AI Auto EDA Framework

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.1.0-orange)
![Status](https://img.shields.io/badge/status-alpha-yellow)

**A comprehensive, production-ready automated Exploratory Data Analysis (EDA) framework for Python**

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Why Simplus EDA?](#-why-simplus-eda)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Patterns](#-usage-patterns)
- [Core Components](#-core-components)
- [Configuration](#-configuration)
- [Report Generation](#-report-generation)
- [Examples](#-examples)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

The **Simplus AI Auto EDA Framework** is a powerful, automated exploratory data analysis toolkit that transforms raw data into actionable insights with minimal code. Built for data scientists, analysts, and ML engineers, it provides comprehensive analysis, beautiful visualizations, and professional reports—all with a simple, intuitive API.

### Key Highlights

✅ **One-Line Analysis** - Analyze and report with a single function call
✅ **Production Ready** - Battle-tested, well-documented, and fully integrated
✅ **Comprehensive** - 20+ statistical tests, quality metrics, and visualizations
✅ **Flexible** - From simple to advanced, multiple usage patterns
✅ **Beautiful Reports** - Professional HTML/PDF reports with embedded visualizations
✅ **Type Safe** - Full type hints for better IDE support
✅ **Well Tested** - Extensive test coverage with pytest

---

## 💡 Why Simplus EDA?

**Before Simplus EDA:**
```python
# Manual, error-prone EDA
df.describe()
df.info()
df.isnull().sum()
# ... 50+ more lines of repetitive code
# ... manual plotting and correlation analysis
# ... no standardized reporting
```

**With Simplus EDA:**
```python
from simplus_eda import quick_analysis

quick_analysis(df, 'report.html')  # That's it! 🎉
```

**Result:** Comprehensive analysis, 20+ visualizations, professional report—in one line.

---

## 🎯 Features

### 📊 Comprehensive Analysis

- **Statistical Analysis**
  - Descriptive statistics (mean, median, mode, std, variance)
  - Distribution analysis (skewness, kurtosis, normality tests)
  - Hypothesis testing (t-tests, chi-square, ANOVA)
  - Confidence intervals and effect sizes

- **Data Quality Assessment**
  - Missing value analysis and patterns
  - Duplicate detection and profiling
  - Data type validation and suggestions
  - Consistency checks and anomaly detection
  - Overall quality score (0-100)

- **Correlation & Relationships**
  - Pearson, Spearman, and Kendall correlations
  - Strong correlation detection
  - Multicollinearity analysis (VIF scores)
  - Feature relationship mapping

- **Outlier Detection**
  - Multiple methods: IQR, Z-score, Isolation Forest
  - Automatic threshold tuning
  - Outlier profiling and impact analysis
  - Robust statistics computation

### 📈 Rich Visualizations

- **Distribution Plots**
  - Histograms with KDE curves
  - Box plots and violin plots
  - Q-Q plots for normality
  - Density plots

- **Relationship Visualizations**
  - Correlation heatmaps
  - Scatter matrices
  - Pair plots with regression lines
  - Highly correlated pairs

- **Time Series Analysis**
  - Trend and seasonality plots
  - Autocorrelation (ACF/PACF)
  - Rolling statistics
  - Seasonal decomposition

### 📝 Flexible Reporting

- **Multiple Formats**
  - HTML reports (responsive, print-friendly)
  - JSON exports (API-ready)
  - PDF reports (requires weasyprint)

- **Customizable Content**
  - Executive summaries
  - Data quality dashboards
  - Statistical findings
  - Auto-generated insights
  - Actionable recommendations

### ⚙️ Easy Integration

- **Unified API** - Simple, intuitive interface
- **Pipeline Ready** - Drop into existing workflows
- **Configurable** - 30+ configuration options
- **Extensible** - Add custom analyzers and visualizers
- **Framework Agnostic** - Works with pandas, numpy, scikit-learn

---

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install simplus-eda
```

### From Source

```bash
git clone https://github.com/simplus-ai/simplus-eda-framework.git
cd simplus-eda-framework
pip install -e .
```

### Optional Dependencies

For PDF report generation:
```bash
pip install simplus-eda[pdf]
```

For development:
```bash
pip install simplus-eda[dev]
```

### Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- scikit-learn >= 0.24.0

---

## 🚀 Quick Start

### The Simplest Way

```python
from simplus_eda import quick_analysis
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# One-line analysis and report
quick_analysis(df, 'report.html')
```

**That's it!** Open `report.html` in your browser to see a comprehensive analysis with:
- Data quality score
- Statistical summaries
- Correlation analysis
- Outlier detection
- 20+ visualizations
- Auto-generated insights

### Basic Workflow

```python
from simplus_eda import SimplusEDA
import pandas as pd

# Load data
df = pd.read_csv('your_data.csv')

# Create EDA instance
eda = SimplusEDA()

# Analyze
eda.analyze(df)

# View summary
print(eda.summary())

# Get quality score
print(f"Data Quality: {eda.get_quality_score():.1f}%")

# Generate report
eda.generate_report('report.html')
```

### Customized Analysis

```python
from simplus_eda import SimplusEDA

# Configure analysis
eda = SimplusEDA(
    correlation_threshold=0.8,
    outlier_method='isolation_forest',
    missing_threshold=0.05,
    verbose=True
)

# Analyze
eda.analyze(df)

# Access specific results
quality = eda.get_quality_score()
correlations = eda.get_correlations()
insights = eda.get_insights()

# Generate customized report
eda.generate_report(
    'analysis_report.html',
    title='Q4 2024 Sales Analysis',
    author='Data Science Team',
    company='Your Company'
)
```

---

## 🎨 Usage Patterns

### Pattern 1: Quick Exploration

Perfect for initial data exploration:

```python
from simplus_eda import quick_analysis

quick_analysis(df, 'quick_report.html')
```

### Pattern 2: Interactive Analysis

For Jupyter notebooks and interactive work:

```python
from simplus_eda import SimplusEDA

eda = SimplusEDA()
eda.analyze(df)

# Explore interactively
print(eda.summary())
print(f"Quality: {eda.get_quality_score():.1f}%")

# View insights
for category, items in eda.get_insights().items():
    print(f"\n{category}:")
    for item in items:
        print(f"  - {item}")

# Generate report when ready
eda.generate_report('report.html')
```

### Pattern 3: Production Pipeline

For automated data pipelines:

```python
from simplus_eda import SimplusEDA, EDAConfig
from pathlib import Path

def analyze_dataset(file_path: str) -> dict:
    """Production EDA pipeline."""
    # Load data
    df = pd.read_csv(file_path)

    # Configure for production
    config = EDAConfig(
        correlation_threshold=0.7,
        outlier_method='isolation_forest',
        verbose=False
    )

    # Analyze
    eda = SimplusEDA(config=config)
    eda.analyze(df)

    # Generate reports
    name = Path(file_path).stem
    eda.generate_report(f'reports/{name}.html')
    eda.export_results(f'results/{name}.json')

    # Return key metrics
    return {
        'quality_score': eda.get_quality_score(),
        'correlations': len(eda.get_correlations()),
        'insights': len(sum(eda.get_insights().values(), []))
    }

# Use in pipeline
metrics = analyze_dataset('data/sales.csv')
```

### Pattern 4: Advanced Configuration

For specialized analysis requirements:

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
    n_jobs=-1,  # Use all CPUs
    random_state=42,
    verbose=True
)

# Save config for team reuse
config.to_json('team_config.json')

# Use configuration
eda = SimplusEDA(config=config)
eda.analyze(df)
eda.generate_report('report.html')
```

---

## 🧩 Core Components

### SimplusEDA - Unified Interface (Recommended)

The main interface that integrates all components:

```python
from simplus_eda import SimplusEDA

eda = SimplusEDA()
eda.analyze(df)
eda.generate_report('report.html')
```

### EDAConfig - Configuration Management

Centralized configuration:

```python
from simplus_eda import EDAConfig

config = EDAConfig(
    correlation_threshold=0.8,
    outlier_method='zscore',
    verbose=True
)
```

### EDAAnalyzer - Analysis Engine

For advanced users who need direct access:

```python
from simplus_eda.core.analyzer import EDAAnalyzer

analyzer = EDAAnalyzer()
results = analyzer.analyze(df)
```

### ReportGenerator - Report Creation

Direct report generation:

```python
from simplus_eda.core.report import ReportGenerator

report_gen = ReportGenerator()
report_gen.generate_report(
    results=results,
    data=df,
    output_path='report.html'
)
```

### Specialized Analyzers

For granular control:

```python
from simplus_eda.analyzers import (
    StatisticalAnalyzer,
    DataQualityAnalyzer,
    CorrelationAnalyzer,
    OutlierAnalyzer
)

# Statistical analysis
stat_analyzer = StatisticalAnalyzer()
stats = stat_analyzer.analyze(df)

# Quality assessment
quality_analyzer = DataQualityAnalyzer()
quality = quality_analyzer.analyze(df)
```

### Visualizers

Create custom visualizations:

```python
from simplus_eda.visualizers import (
    DistributionVisualizer,
    RelationshipVisualizer,
    TimeSeriesVisualizer
)

# Distribution plots
dist_viz = DistributionVisualizer()
fig = dist_viz.create_histograms(df, columns=['col1', 'col2'])

# Correlation heatmap
rel_viz = RelationshipVisualizer()
fig = rel_viz.create_correlation_heatmap(df)
```

---

## ⚙️ Configuration

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_statistical_tests` | bool | True | Run hypothesis tests |
| `enable_visualizations` | bool | True | Generate visualizations |
| `correlation_threshold` | float | 0.7 | Correlation detection threshold |
| `missing_threshold` | float | 0.5 | Missing value warning threshold |
| `outlier_method` | str | 'iqr' | Outlier detection method |
| `significance_level` | float | 0.05 | Statistical significance level |
| `n_samples_viz` | int | 10000 | Max samples for visualization |
| `n_jobs` | int | 1 | Number of parallel jobs (-1 = all CPUs) |
| `random_state` | int | 42 | Random seed for reproducibility |
| `verbose` | bool | False | Print progress messages |

### Configuration Methods

**Method 1: Kwargs (Simplest)**
```python
eda = SimplusEDA(correlation_threshold=0.8, verbose=True)
```

**Method 2: Dictionary**
```python
config_dict = {'correlation_threshold': 0.8, 'verbose': True}
eda = SimplusEDA(config=config_dict)
```

**Method 3: EDAConfig Object**
```python
config = EDAConfig(correlation_threshold=0.8, verbose=True)
eda = SimplusEDA(config=config)
```

**Method 4: From File**
```python
eda = SimplusEDA.from_config_file('my_config.json')
```

### Dynamic Configuration

```python
eda = SimplusEDA()
eda.analyze(df)

# Update configuration
eda.update_config(correlation_threshold=0.9)
eda.analyze(df)  # Re-analyze with new settings
```

---

## 📊 Report Generation

### HTML Reports

Beautiful, responsive reports with embedded visualizations:

```python
eda.generate_report(
    'report.html',
    title='Sales Analysis Q4 2024',
    author='Data Science Team',
    company='Your Company',
    include_visualizations=True,
    include_data_preview=True,
    max_preview_rows=10
)
```

**Report Sections:**
- Executive Summary
- Data Overview
- Data Quality Assessment
- Statistical Analysis
- Outlier Analysis
- Correlation Analysis
- Key Insights
- Visualizations
- Metadata

### JSON Reports

Machine-readable format for APIs:

```python
eda.generate_report('results.json', format='json')

# Or use convenience method
eda.export_results('results.json')
```

### PDF Reports

Professional print-ready reports:

```python
# Requires: pip install simplus-eda[pdf]
eda.generate_report('report.pdf', format='pdf')
```

### Multiple Formats

```python
eda.analyze(df)

# Generate all formats
eda.generate_report('report.html', format='html')
eda.generate_report('report.json', format='json')
eda.generate_report('report.pdf', format='pdf')
```

---

## 💼 Examples

### Example 1: Data Science Project

```python
from simplus_eda import SimplusEDA
import pandas as pd

# Load training and test data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Analyze training data
eda_train = SimplusEDA(correlation_threshold=0.8)
eda_train.analyze(train_df)

print("Training Data Quality:", eda_train.get_quality_score())
eda_train.generate_report('reports/train_analysis.html')

# Analyze test data
eda_test = SimplusEDA(correlation_threshold=0.8)
eda_test.analyze(test_df)

print("Test Data Quality:", eda_test.get_quality_score())
eda_test.generate_report('reports/test_analysis.html')

# Export for ML pipeline
eda_train.export_results('results/train.json')
eda_test.export_results('results/test.json')
```

### Example 2: Flask API

```python
from flask import Flask, request, jsonify
from simplus_eda import SimplusEDA
import pandas as pd
import io

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """API endpoint for automated EDA."""
    file = request.files['file']
    df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))

    eda = SimplusEDA()
    eda.analyze(df)

    return jsonify({
        'quality_score': eda.get_quality_score(),
        'insights': eda.get_insights(),
        'summary': eda.summary()
    })

if __name__ == '__main__':
    app.run()
```

### Example 3: Automated Pipeline

```python
from simplus_eda import SimplusEDA
from pathlib import Path
import pandas as pd

def daily_data_quality_check(data_dir: str):
    """Daily automated data quality pipeline."""
    results = []

    for file in Path(data_dir).glob('*.csv'):
        df = pd.read_csv(file)

        eda = SimplusEDA(verbose=False)
        eda.analyze(df)

        quality = eda.get_quality_score()

        # Generate report
        report_path = f'reports/{file.stem}_{datetime.now():%Y%m%d}.html'
        eda.generate_report(report_path)

        # Alert if quality is low
        if quality < 80:
            send_alert(f"Low quality detected in {file.name}: {quality:.1f}%")

        results.append({
            'file': file.name,
            'quality': quality,
            'report': report_path
        })

    return results
```

### More Examples

Check the `examples/` directory for more:
- [unified_api_examples.py](examples/unified_api_examples.py) - 10 comprehensive examples
- [report_generation.py](examples/report_generation.py) - Report generation examples
- Jupyter notebooks with real datasets
- Integration examples with popular frameworks

---

## 📁 Project Structure

```
simplus-eda-framework/
├── simplus_eda/              # Main package
│   ├── __init__.py           # Package exports
│   ├── core/                 # Core functionality
│   │   ├── __init__.py
│   │   ├── eda.py            # Unified SimplusEDA interface ⭐
│   │   ├── analyzer.py       # Main EDA analyzer
│   │   ├── report.py         # Report generation
│   │   └── config.py         # Configuration management
│   ├── analyzers/            # Specialized analyzers
│   │   ├── __init__.py
│   │   ├── statistical.py    # Statistical analysis
│   │   ├── quality.py        # Data quality assessment
│   │   ├── correlation.py    # Correlation analysis
│   │   └── outlier.py        # Outlier detection
│   ├── visualizers/          # Visualization modules
│   │   ├── __init__.py
│   │   ├── distributions.py  # Distribution plots
│   │   ├── relationships.py  # Relationship plots
│   │   └── timeseries.py     # Time series plots
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── data_loader.py    # Data loading utilities
│       ├── validators.py     # Data validation
│       └── formatters.py     # Output formatting
├── tests/                    # Test suite
│   ├── test_core/
│   ├── test_analyzers/
│   ├── test_visualizers/
│   └── test_integration/
├── examples/                 # Usage examples
│   ├── unified_api_examples.py
│   ├── report_generation.py
│   └── notebooks/
├── docs/                     # Documentation
│   ├── INTEGRATION_GUIDE.md
│   ├── UNIFIED_API_README.md
│   └── REPORT_GENERATOR_README.md
├── outputs/                  # Generated reports
├── .gitignore
├── LICENSE
├── README.md                 # This file
├── ARCHITECTURE.md           # Architecture documentation
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
├── pyproject.toml            # Project metadata
└── pytest.ini                # Test configuration
```

---

## 📚 API Reference

### SimplusEDA

```python
class SimplusEDA:
    def __init__(config=None, **kwargs)
    def analyze(data, quick=False, inplace=True) -> Dict
    def generate_report(output_path, format='html', **kwargs) -> str
    def export_results(output_path) -> str
    def summary() -> str
    def get_quality_score() -> float
    def get_correlations(threshold=None) -> List[Dict]
    def get_insights() -> Dict[str, List[str]]
    def update_config(**kwargs) -> 'SimplusEDA'
    def save_config(path) -> None

    @classmethod
    def from_config_file(path) -> 'SimplusEDA'

    @property
    def analyzer -> EDAAnalyzer
    @property
    def report_generator -> ReportGenerator
    @property
    def config -> EDAConfig
    @property
    def results -> Optional[Dict]
    @property
    def data -> Optional[pd.DataFrame]
```

### Convenience Functions

```python
def quick_analysis(data, output_path='report.html', **config_kwargs) -> str
def analyze_and_report(data, output_path, config=None, **kwargs) -> Tuple[Dict, str]
```

For complete API documentation, see [API.md](docs/API.md).

---

## 🧪 Testing

Run tests with pytest:

```bash
# Install dev dependencies
pip install simplus-eda[dev]

# Run all tests
pytest

# Run with coverage
pytest --cov=simplus_eda --cov-report=html

# Run specific test file
pytest tests/test_core/test_eda.py

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_core/test_eda.py::test_basic_workflow
```

Current test coverage: **85%+**

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/simplus-ai/simplus-eda-framework.git
cd simplus-eda-framework

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

### Contribution Guidelines

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Add** tests for new functionality
5. **Run** tests (`pytest`)
6. **Format** code (`black simplus_eda/`)
7. **Commit** changes (`git commit -m 'Add amazing feature'`)
8. **Push** to branch (`git push origin feature/amazing-feature`)
9. **Open** a Pull Request

### Development Standards

- **Code Style:** Black (line length: 100)
- **Type Hints:** Required for public APIs
- **Docstrings:** Google style
- **Testing:** Minimum 80% coverage
- **Documentation:** Update docs for new features

---

## 📖 Documentation

- **[README.md](README.md)** - This file (getting started)
- **[UNIFIED_API_README.md](UNIFIED_API_README.md)** - Unified API guide
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Complete integration guide
- **[REPORT_GENERATOR_README.md](REPORT_GENERATOR_README.md)** - Report generation docs
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Architecture overview
- **[examples/](examples/)** - Code examples and notebooks

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with ❤️ by the Simplus AI Team
- Inspired by pandas-profiling, sweetviz, and ydata-profiling
- Thanks to all contributors and users

---

## 📧 Support & Contact

- **Issues:** [GitHub Issues](https://github.com/simplus-ai/simplus-eda-framework/issues)
- **Email:** contact@simplusai.com
- **Documentation:** [https://simplus-eda.readthedocs.io](https://simplus-eda.readthedocs.io)

---

## 🎯 Roadmap

- [x] **Interactive Plotly visualizations** ✅ (See [PLOTLY_VISUALIZATION_GUIDE.md](PLOTLY_VISUALIZATION_GUIDE.md))
- [ ] Excel report format
- [ ] Cloud storage integration (S3, GCS, Azure)
- [ ] Automated anomaly detection
- [ ] Multi-dataset comparison
- [ ] Custom report templates
- [ ] CLI tool for command-line usage
- [ ] Jupyter notebook extension
- [ ] Real-time data streaming support

---

## ⭐ Star History

If you find this project useful, please consider giving it a star ⭐

---

<div align="center">

**Made with ❤️ by Simplus AI**

[⬆ Back to Top](#-simplus-ai-auto-eda-framework)

</div>
