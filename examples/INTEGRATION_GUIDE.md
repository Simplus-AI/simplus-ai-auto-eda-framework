# Simplus EDA - Complete Integration Guide

## Overview

This guide demonstrates how the **Config**, **Analyzer**, and **Report** modules work seamlessly together through the unified `SimplusEDA` API.

## The Problem (Before Integration)

Previously, users had to manually manage three separate components:

```python
# Before - Manual integration required
from simplus_eda.core.config import EDAConfig
from simplus_eda.core.analyzer import EDAAnalyzer
from simplus_eda.core.report import ReportGenerator

# Step 1: Create config
config = EDAConfig(correlation_threshold=0.8)

# Step 2: Create analyzer with config
analyzer = EDAAnalyzer(config=config)

# Step 3: Analyze data
results = analyzer.analyze(df)

# Step 4: Create report generator
report_gen = ReportGenerator(config={'title': 'My Report'})

# Step 5: Generate report
report_gen.generate_report(results=results, data=df, output_path='report.html')
```

**Issues:**
- Too many steps for simple tasks
- Config had to be passed manually
- Results had to be managed manually
- No state persistence between steps

## The Solution (Unified API)

The `SimplusEDA` class provides a **unified, intuitive interface** that automatically manages all components:

```python
# After - Unified and simple
from simplus_eda import SimplusEDA

eda = SimplusEDA(correlation_threshold=0.8)
eda.analyze(df)
eda.generate_report('report.html', title='My Report')
```

## Key Features of the Integration

### 1. **Automatic Component Management**

SimplusEDA automatically creates and manages all components:

```python
eda = SimplusEDA()

# Internally, this creates:
# - EDAConfig with your settings
# - EDAAnalyzer configured with your config
# - ReportGenerator (lazy initialization when needed)
```

### 2. **State Persistence**

Analysis results are automatically stored and reused:

```python
eda = SimplusEDA()
eda.analyze(df)  # Results stored internally

# All these use the stored results automatically:
eda.generate_report('report.html')
eda.export_results('results.json')
print(eda.summary())
quality = eda.get_quality_score()
```

### 3. **Flexible Configuration**

Multiple ways to configure, all fully integrated:

```python
# Method 1: kwargs (simplest)
eda = SimplusEDA(correlation_threshold=0.8, verbose=True)

# Method 2: Config dictionary
eda = SimplusEDA(config={'correlation_threshold': 0.8})

# Method 3: EDAConfig object (most control)
config = EDAConfig(correlation_threshold=0.8, outlier_method='zscore')
eda = SimplusEDA(config=config)

# Method 4: Load from file
eda = SimplusEDA.from_config_file('my_config.json')
```

### 4. **Consistent Behavior**

Configuration flows seamlessly through all components:

```python
# Configuration is automatically shared
eda = SimplusEDA(
    correlation_threshold=0.7,
    missing_threshold=0.1,
    outlier_method='zscore'
)

# Analyzer uses these settings
eda.analyze(df)

# Report respects these settings
eda.generate_report('report.html')

# Results reflect these settings
correlations = eda.get_correlations()  # Uses 0.7 threshold
```

## Usage Patterns

### Pattern 1: One-Liner (Quickest)

For rapid exploratory analysis:

```python
from simplus_eda import quick_analysis

quick_analysis(df, 'report.html')
```

**Best for:**
- Quick data checks
- Initial exploration
- Prototyping

### Pattern 2: Basic Workflow

For typical EDA tasks:

```python
from simplus_eda import SimplusEDA

eda = SimplusEDA()
eda.analyze(df)
print(eda.summary())
eda.generate_report('report.html')
```

**Best for:**
- Standard EDA workflows
- Balanced control and simplicity
- Most common use cases

### Pattern 3: Customized Analysis

For specific requirements:

```python
from simplus_eda import SimplusEDA

eda = SimplusEDA(
    correlation_threshold=0.8,
    outlier_method='isolation_forest',
    missing_threshold=0.05,
    verbose=True
)

results = eda.analyze(df)

# Access specific results
quality = eda.get_quality_score()
correlations = eda.get_correlations(threshold=0.9)
insights = eda.get_insights()

# Generate customized report
eda.generate_report(
    'report.html',
    title='Custom Analysis',
    author='Data Team',
    include_visualizations=True
)
```

**Best for:**
- Domain-specific analysis
- Custom thresholds
- Detailed control

### Pattern 4: Advanced Configuration

For maximum control:

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
    verbose=True
)

# Save config for reuse
config.to_json('production_config.json')

# Use config
eda = SimplusEDA(config=config)
eda.analyze(df)
eda.generate_report('report.html')
```

**Best for:**
- Production pipelines
- Reusable configurations
- Team standardization

### Pattern 5: Dynamic Configuration

For iterative analysis:

```python
from simplus_eda import SimplusEDA

# Start with defaults
eda = SimplusEDA()
eda.analyze(df)

# Try different thresholds
eda.update_config(correlation_threshold=0.9)
eda.analyze(df)
correlations_strict = eda.get_correlations()

eda.update_config(correlation_threshold=0.6)
eda.analyze(df)
correlations_relaxed = eda.get_correlations()

# Compare results
print(f"Strict (0.9): {len(correlations_strict)} correlations")
print(f"Relaxed (0.6): {len(correlations_relaxed)} correlations")
```

**Best for:**
- Iterative exploration
- Sensitivity analysis
- Parameter tuning

## Complete Workflow Examples

### Example 1: Data Science Project

```python
from simplus_eda import SimplusEDA, EDAConfig
import pandas as pd

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Configure for data science
config = EDAConfig(
    correlation_threshold=0.8,
    outlier_method='isolation_forest',
    missing_threshold=0.05,
    enable_statistical_tests=True
)

# Analyze training data
eda_train = SimplusEDA(config=config)
eda_train.analyze(train_df)

print("Training Data Summary:")
print(eda_train.summary())
print(f"\nQuality Score: {eda_train.get_quality_score():.1f}%")

# Generate detailed report
eda_train.generate_report(
    'reports/train_analysis.html',
    title='Training Data Analysis',
    author='ML Team',
    include_visualizations=True
)

# Analyze test data with same config
eda_test = SimplusEDA(config=config)
eda_test.analyze(test_df)

# Compare quality
train_quality = eda_train.get_quality_score()
test_quality = eda_test.get_quality_score()
print(f"\nQuality Comparison:")
print(f"Train: {train_quality:.1f}%")
print(f"Test: {test_quality:.1f}%")

# Export results for ML pipeline
eda_train.export_results('results/train_results.json')
eda_test.export_results('results/test_results.json')
```

### Example 2: Automated Pipeline

```python
from simplus_eda import SimplusEDA
from pathlib import Path
import pandas as pd

def analyze_dataset(file_path, output_dir='reports'):
    """Automated EDA pipeline function."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Load data
    df = pd.read_csv(file_path)

    # Analyze
    eda = SimplusEDA(
        correlation_threshold=0.7,
        outlier_method='iqr',
        verbose=False
    )
    eda.analyze(df)

    # Get file name without extension
    file_name = Path(file_path).stem

    # Generate reports
    eda.generate_report(
        f'{output_dir}/{file_name}_report.html',
        title=f'Analysis: {file_name}',
        format='html'
    )

    eda.export_results(f'{output_dir}/{file_name}_results.json')

    # Return key metrics
    return {
        'file': file_name,
        'rows': df.shape[0],
        'columns': df.shape[1],
        'quality_score': eda.get_quality_score(),
        'correlations': len(eda.get_correlations())
    }

# Process multiple files
files = ['data1.csv', 'data2.csv', 'data3.csv']
results = [analyze_dataset(f) for f in files]

# Summary
for r in results:
    print(f"{r['file']}: Quality={r['quality_score']:.1f}%, "
          f"Correlations={r['correlations']}")
```

### Example 3: Interactive Analysis

```python
from simplus_eda import SimplusEDA
import pandas as pd

# Load data
df = pd.read_csv('sales_data.csv')

# Create EDA instance
eda = SimplusEDA()

# Perform analysis
print("Analyzing dataset...")
eda.analyze(df)

# Interactive exploration
while True:
    print("\n" + "="*50)
    print("1. Show summary")
    print("2. Show quality score")
    print("3. Show correlations")
    print("4. Show insights")
    print("5. Generate report")
    print("6. Change correlation threshold")
    print("7. Exit")

    choice = input("\nSelect option: ")

    if choice == '1':
        print(eda.summary())

    elif choice == '2':
        score = eda.get_quality_score()
        print(f"\nQuality Score: {score:.1f}%")

    elif choice == '3':
        correlations = eda.get_correlations()
        print(f"\nFound {len(correlations)} correlations:")
        for c in correlations[:5]:
            print(f"  {c['feature1']} <-> {c['feature2']}: {c['correlation']:.3f}")

    elif choice == '4':
        insights = eda.get_insights()
        for category, items in insights.items():
            if items:
                print(f"\n{category.replace('_', ' ').title()}:")
                for item in items:
                    print(f"  - {item}")

    elif choice == '5':
        path = input("Output path: ")
        eda.generate_report(path)
        print(f"Report generated: {path}")

    elif choice == '6':
        threshold = float(input("New threshold (0-1): "))
        eda.update_config(correlation_threshold=threshold)
        eda.analyze(df)
        print(f"Re-analyzed with threshold {threshold}")

    elif choice == '7':
        break
```

## API Reference

### SimplusEDA Class

#### Initialization

```python
SimplusEDA(
    config: Optional[Union[Dict, EDAConfig]] = None,
    **kwargs  # Any EDAConfig parameter
)
```

#### Core Methods

- `analyze(data, quick=False, inplace=True)` - Perform EDA
- `generate_report(output_path, format='html', **kwargs)` - Generate report
- `export_results(output_path)` - Export JSON results
- `summary()` - Get text summary
- `get_quality_score()` - Get quality score
- `get_correlations(threshold=None)` - Get correlations
- `get_insights()` - Get auto-generated insights
- `update_config(**kwargs)` - Update configuration
- `save_config(path)` - Save configuration to file

#### Class Methods

- `from_config_file(path)` - Load config from file

#### Properties

- `analyzer` - Access underlying EDAAnalyzer
- `report_generator` - Access ReportGenerator
- `config` - Access EDAConfig
- `results` - Access analysis results
- `data` - Access analyzed DataFrame

### Convenience Functions

```python
# Quick analysis
quick_analysis(data, output_path='report.html', **config_kwargs)

# Analyze and report
analyze_and_report(data, output_path, config=None, format='html', **report_kwargs)
```

## Migration Guide

### From Separate Components

**Before:**
```python
from simplus_eda.core.config import EDAConfig
from simplus_eda.core.analyzer import EDAAnalyzer
from simplus_eda.core.report import ReportGenerator

config = EDAConfig(correlation_threshold=0.8)
analyzer = EDAAnalyzer(config=config)
results = analyzer.analyze(df)

report_gen = ReportGenerator()
report_gen.generate_report(
    results=results,
    data=df,
    output_path='report.html'
)
```

**After:**
```python
from simplus_eda import SimplusEDA

eda = SimplusEDA(correlation_threshold=0.8)
eda.analyze(df)
eda.generate_report('report.html')
```

### Advantages of New API

1. **Fewer imports** - One import vs. three
2. **Less code** - ~70% reduction in lines
3. **Better state management** - Automatic result storage
4. **Cleaner syntax** - More Pythonic
5. **Backward compatible** - Old API still works

## Best Practices

### 1. Use SimplusEDA for New Projects

```python
# Recommended
from simplus_eda import SimplusEDA
eda = SimplusEDA()
```

### 2. Save Configurations for Reuse

```python
# Create once
config = EDAConfig(correlation_threshold=0.8, outlier_method='zscore')
config.to_json('team_config.json')

# Reuse everywhere
eda = SimplusEDA.from_config_file('team_config.json')
```

### 3. Use Context Managers for Large Datasets

```python
import pandas as pd
from simplus_eda import SimplusEDA

# Process in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    eda = SimplusEDA()
    eda.analyze(chunk)
    # Process results
    del eda  # Free memory
```

### 4. Validate Data Quality Before Modeling

```python
eda = SimplusEDA(missing_threshold=0.05)
eda.analyze(df)

quality = eda.get_quality_score()
if quality < 80:
    insights = eda.get_insights()
    print("Data quality issues:")
    for issue in insights.get('quality_issues', []):
        print(f"  - {issue}")
    raise ValueError("Data quality too low for modeling")
```

### 5. Document Analysis with Reports

```python
eda = SimplusEDA()
eda.analyze(df)

# Generate timestamped reports
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

eda.generate_report(
    f'reports/analysis_{timestamp}.html',
    title=f'Analysis - {timestamp}',
    author='Data Pipeline'
)
```

## Troubleshooting

### Issue: Configuration Not Taking Effect

**Problem:**
```python
eda = SimplusEDA(correlation_threshold=0.9)
correlations = eda.get_correlations()  # Still shows 0.7 threshold
```

**Solution:** Call `analyze()` after configuration changes:
```python
eda = SimplusEDA(correlation_threshold=0.9)
eda.analyze(df)  # Configuration applied here
correlations = eda.get_correlations()  # Now uses 0.9
```

### Issue: Results Not Available

**Problem:**
```python
eda = SimplusEDA()
eda.generate_report('report.html')  # Error: No results available
```

**Solution:** Always call `analyze()` first:
```python
eda = SimplusEDA()
eda.analyze(df)  # Required!
eda.generate_report('report.html')
```

### Issue: Memory Usage with Large Datasets

**Problem:** Running out of memory with large datasets

**Solution:** Use sampling or disable visualizations:
```python
# Option 1: Sample data for visualization
sample = df.sample(n=10000) if len(df) > 10000 else df
eda = SimplusEDA()
eda.analyze(df)  # Full analysis
eda.generate_report('report.html', data=sample)  # Visualize sample

# Option 2: Disable visualizations
eda.generate_report('report.html', include_visualizations=False)
```

## Summary

The unified `SimplusEDA` API provides:

✅ **Seamless integration** of Config, Analyzer, and Report modules
✅ **Intuitive interface** that's easy to learn and use
✅ **Flexible configuration** with multiple approaches
✅ **State management** that reduces boilerplate code
✅ **Backward compatibility** with existing code
✅ **Production-ready** for real-world applications

Use `SimplusEDA` for all new projects and consider migrating existing code for better maintainability and user experience.
