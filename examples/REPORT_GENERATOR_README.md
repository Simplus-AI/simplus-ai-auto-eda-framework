# Report Generator - Comprehensive Documentation

## Overview

The `ReportGenerator` class is the core module that integrates all analysis and visualization components of the Simplus EDA Framework to generate comprehensive, professional reports in multiple formats.

## Features

### 📊 Multiple Output Formats
- **HTML Reports**: Beautiful, interactive reports with embedded visualizations
- **JSON Reports**: Machine-readable format for API integration and programmatic access
- **PDF Reports**: Print-ready reports (requires weasyprint)

### 🎨 Comprehensive Content
- **Executive Summary**: High-level overview with key metrics
- **Data Overview**: Dataset dimensions, column types, and data preview
- **Data Quality Assessment**: Quality scores, missing values, duplicates
- **Statistical Analysis**: Descriptive statistics for all numeric columns
- **Outlier Analysis**: Detection and summary of outliers
- **Correlation Analysis**: Strong correlations and multicollinearity detection
- **Key Insights**: Auto-generated insights and recommendations
- **Visualizations**: Embedded charts (histograms, boxplots, heatmaps, time series)

### ⚙️ Customizable
- Custom report titles, authors, and organization names
- Configurable table of contents
- Control over visualization inclusion
- Adjustable data preview rows

## Installation

The report generator is included in the core Simplus EDA package:

```bash
pip install -e .
```

For PDF generation support:
```bash
pip install weasyprint
```

## Quick Start

### Basic Usage

```python
from simplus_eda.core.analyzer import EDAAnalyzer
from simplus_eda.core.report import ReportGenerator
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Perform analysis
analyzer = EDAAnalyzer()
results = analyzer.analyze(df)

# Generate report
report_gen = ReportGenerator()
report_gen.generate_report(
    results=results,
    data=df,
    output_path='my_report.html',
    format='html',
    include_visualizations=True
)
```

### Customized Report

```python
# Configure report customization
config = {
    'title': 'My Custom Analysis Report',
    'author': 'Data Science Team',
    'company': 'My Organization',
    'include_toc': True
}

report_gen = ReportGenerator(config=config)
report_gen.generate_report(
    results=results,
    data=df,
    output_path='custom_report.html',
    format='html',
    include_visualizations=True,
    include_data_preview=True,
    max_preview_rows=20
)
```

### Multiple Formats

```python
report_gen = ReportGenerator()

# Generate HTML
report_gen.generate_report(
    results=results,
    data=df,
    output_path='report.html',
    format='html'
)

# Generate JSON
report_gen.generate_report(
    results=results,
    data=df,
    output_path='report.json',
    format='json'
)

# Generate PDF (requires weasyprint)
report_gen.generate_report(
    results=results,
    data=df,
    output_path='report.pdf',
    format='pdf'
)
```

## Report Sections

### 1. Executive Summary
High-level overview with key metrics:
- Total rows and columns
- Data quality score with badge
- Number of key insights generated

### 2. Data Overview
- Dataset dimensions (rows, columns, memory usage)
- Column type breakdown (numeric, categorical, datetime)
- Duplicate row analysis
- Data preview (first N rows)

### 3. Data Quality Assessment
- Quality score components (completeness, uniqueness, consistency, validity)
- Missing value analysis and recommendations
- Data type suggestions

### 4. Statistical Analysis
- Descriptive statistics for numeric columns
- Central tendency (mean, median, mode)
- Dispersion (std dev, variance, range, IQR)
- Distribution shape (skewness, kurtosis)

### 5. Outlier Analysis
- Total outlier count and percentage
- Columns affected by outliers
- Outlier detection method used

### 6. Correlation Analysis
- Strong correlation pairs
- Correlation coefficients
- Relationship type (positive/negative)

### 7. Key Insights
Auto-generated insights categorized as:
- Data characteristics
- Quality issues
- Statistical findings
- Recommendations

### 8. Visualizations
Embedded charts include:
- Distribution histograms with KDE
- Boxplots for outlier visualization
- Correlation heatmaps
- Highly correlated variable pairs
- Time series plots (if datetime columns exist)

### 9. Metadata
- Analysis timestamp
- Analysis duration
- Analyzer version
- Configuration used

## Configuration Options

### ReportGenerator Configuration

```python
config = {
    # Report metadata
    'title': 'Custom Report Title',
    'author': 'Your Name',
    'company': 'Organization Name',

    # Report options
    'include_toc': True,  # Table of contents
    'max_figures': 10,     # Maximum visualizations
}
```

### generate_report() Parameters

```python
report_gen.generate_report(
    results=analysis_results,        # Required: Analysis results from EDAAnalyzer
    data=dataframe,                   # Required: Original DataFrame
    output_path='report.html',        # Required: Output file path
    format='html',                    # Format: 'html', 'json', or 'pdf'
    include_visualizations=True,      # Include embedded charts
    include_data_preview=True,        # Include data preview table
    max_preview_rows=10               # Number of preview rows
)
```

## Integration with EDA Pipeline

The Report Generator seamlessly integrates with all other components:

```python
from simplus_eda.core.analyzer import EDAAnalyzer
from simplus_eda.core.config import EDAConfig
from simplus_eda.core.report import ReportGenerator

# Configure analysis
config = EDAConfig(
    correlation_threshold=0.7,
    outlier_method='zscore',
    missing_threshold=0.1
)

# Run analysis
analyzer = EDAAnalyzer(config=config)
results = analyzer.analyze(df)

# Generate report
report_config = {
    'title': 'Automated EDA Report',
    'author': 'ML Pipeline'
}

report_gen = ReportGenerator(config=report_config)
report_path = report_gen.generate_report(
    results=results,
    data=df,
    output_path='pipeline_report.html',
    format='html'
)

print(f"Report generated: {report_path}")
```

## Output Examples

### HTML Report Features
- **Professional styling** with modern CSS
- **Responsive design** that works on all devices
- **Print-friendly** layout
- **Interactive elements** with hover effects
- **Color-coded badges** for quality scores
- **Organized sections** with navigation links

### JSON Report Structure
```json
{
  "metadata": {
    "generated_at": "2024-11-11T21:10:00",
    "generator_version": "1.0.0",
    "config": {...}
  },
  "results": {
    "overview": {...},
    "statistics": {...},
    "quality": {...},
    "outliers": {...},
    "correlations": {...},
    "insights": {...}
  }
}
```

## Visualization Integration

The report generator automatically integrates with all visualizer modules:

- **DistributionVisualizer**: Histograms, boxplots, density plots, Q-Q plots
- **RelationshipVisualizer**: Correlation heatmaps, scatter matrices, pair plots
- **TimeSeriesVisualizer**: Time plots, seasonal decomposition, autocorrelation

All visualizations are:
- Embedded as base64-encoded images in HTML reports
- Automatically generated based on data types
- Properly sized and styled
- Cleaned up after generation to free memory

## Best Practices

### 1. Memory Management
```python
# The report generator automatically closes matplotlib figures
# But for very large datasets, consider:
report_gen.generate_report(
    results=results,
    data=df.head(10000),  # Use sample for visualization
    output_path='report.html'
)
```

### 2. Custom Analysis Configuration
```python
# Tailor analysis to your data
config = EDAConfig(
    correlation_threshold=0.6,      # Adjust for your domain
    outlier_method='isolation_forest',  # Choose appropriate method
    missing_threshold=0.05          # Set quality standards
)
```

### 3. Report Organization
```python
# Organize reports by date/project
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'reports/{project_name}_{timestamp}.html'
```

### 4. Batch Report Generation
```python
# Generate reports for multiple datasets
datasets = {
    'train': train_df,
    'test': test_df,
    'validation': val_df
}

for name, df in datasets.items():
    results = analyzer.analyze(df)
    report_gen.generate_report(
        results=results,
        data=df,
        output_path=f'reports/{name}_report.html'
    )
```

## API Reference

### ReportGenerator Class

#### `__init__(config: Optional[Dict[str, Any]] = None)`
Initialize the report generator with optional configuration.

#### `generate_report(...) -> str`
Main method to generate reports. Returns path to generated report.

#### `generate_html(...) -> str`
Generate HTML report specifically.

#### `generate_json(...) -> str`
Generate JSON report specifically.

#### `generate_pdf(...) -> str`
Generate PDF report specifically (requires weasyprint).

## Examples

See `/examples/report_generation.py` for comprehensive examples including:
1. Basic HTML report
2. Customized report with configuration
3. JSON report for API integration
4. Multiple format generation
5. Analysis summary and insights

Run examples:
```bash
python examples/report_generation.py
```

## Troubleshooting

### PDF Generation Issues
**Problem**: ImportError when generating PDFs
**Solution**: Install weasyprint:
```bash
pip install weasyprint
```

### Large File Sizes
**Problem**: HTML reports are too large
**Solution**: Reduce visualizations or use lower resolution:
```python
report_gen.generate_report(
    results=results,
    data=df,
    output_path='report.html',
    include_visualizations=False  # Skip visualizations
)
```

### Memory Issues
**Problem**: Out of memory with large datasets
**Solution**: Sample your data for visualization:
```python
viz_sample = df.sample(n=10000) if len(df) > 10000 else df
report_gen.generate_report(
    results=results,
    data=viz_sample,
    output_path='report.html'
)
```

## Future Enhancements

Planned features for future releases:
- [ ] Interactive HTML reports with Plotly
- [ ] Excel report format
- [ ] Markdown report format
- [ ] Custom CSS themes
- [ ] Report templates
- [ ] Multi-dataset comparison reports
- [ ] Email integration
- [ ] Cloud storage integration

## Contributing

To contribute improvements to the report generator:
1. Test with various dataset types
2. Ensure backward compatibility
3. Update documentation
4. Add unit tests
5. Submit pull request

## License

Part of the Simplus EDA Framework - See LICENSE file for details.
