# Simplus EDA Framework - Examples

This directory contains example scripts demonstrating different use cases and integration patterns for the Simplus AI Auto EDA Framework.

## Examples Overview

### 1. Basic Usage ([basic_usage.py](basic_usage.py))
The simplest way to get started with the framework.

**What it demonstrates:**
- Loading data
- Performing basic EDA analysis
- Generating reports

**Run it:**
```bash
python examples/basic_usage.py
```

### 2. Advanced Configuration ([advanced_config.py](advanced_config.py))
Shows how to customize the analysis using configuration options.

**What it demonstrates:**
- Creating custom `EDAConfig`
- Setting analysis parameters
- Customizing report generation
- Working with larger datasets

**Run it:**
```bash
python examples/advanced_config.py
```

### 3. Pipeline Integration ([pipeline_integration.py](pipeline_integration.py))
Integration into a data processing pipeline.

**What it demonstrates:**
- Building a data pipeline class
- Data loading and validation
- EDA as part of pipeline stages
- Error handling and logging

**Run it:**
```bash
python examples/pipeline_integration.py
```

### 4. API Service ([api_service.py](api_service.py))
Exposing EDA capabilities as a REST API using Flask.

**What it demonstrates:**
- REST API endpoints for EDA
- File upload handling
- Batch processing
- Report generation via API

**Requirements:**
```bash
pip install flask
```

**Run it:**
```bash
python examples/api_service.py
```

**Test the API:**
```bash
# Health check
curl http://localhost:5000/health

# Analyze a file
curl -X POST -F "file=@data.csv" http://localhost:5000/analyze

# Generate HTML report
curl -X POST -F "file=@data.csv" -F "format=html" \
  http://localhost:5000/analyze/report -o report.html
```

## Sample Data

Each example generates its own sample data or provides instructions for creating test datasets. You can also use your own data files.

### Supported File Formats
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- JSON (`.json`)
- Parquet (`.parquet`)
- Feather (`.feather`)

## Output

By default, all examples create an `output/` directory in the project root for storing:
- Generated reports (HTML, JSON, PDF)
- Temporary data files
- API response data

## Integration Patterns

### Pattern 1: Simple Script Integration
```python
from simplus_eda import EDAAnalyzer
import pandas as pd

data = pd.read_csv("data.csv")
analyzer = EDAAnalyzer()
results = analyzer.analyze(data)
```

### Pattern 2: Configuration-Based
```python
from simplus_eda import EDAAnalyzer
from simplus_eda.core.config import EDAConfig

config = EDAConfig(
    correlation_threshold=0.5,
    outlier_method="zscore"
)
analyzer = EDAAnalyzer(config=config.to_dict())
results = analyzer.analyze(data)
```

### Pattern 3: Pipeline Integration
```python
class MyPipeline:
    def __init__(self):
        self.eda = EDAAnalyzer()

    def process(self, data):
        # ... preprocessing ...
        eda_results = self.eda.analyze(data)
        # ... post-processing ...
        return eda_results
```

### Pattern 4: Microservice
```python
@app.route('/analyze', methods=['POST'])
def analyze():
    data = load_from_request()
    analyzer = EDAAnalyzer()
    return jsonify(analyzer.analyze(data))
```

## Next Steps

After running these examples:

1. **Customize for your data**: Modify the examples to work with your specific datasets
2. **Extend functionality**: Add custom analyzers or visualizers
3. **Deploy**: Integrate into your production pipelines or services
4. **Contribute**: Share your use cases and improvements

## Troubleshooting

### Common Issues

**Import Error:**
```bash
# Make sure the package is installed
pip install -e .
```

**File Not Found:**
```bash
# Create output directory
mkdir -p output
```

**API Port Already in Use:**
```python
# Change the port in api_service.py
app.run(port=5001)  # Use a different port
```

## Additional Resources

- [Main README](../README.md) - Full documentation
- [Architecture](../ARCHITECTURE.md) - System design details
- [API Reference](../docs/api.md) - Detailed API documentation

## Contributing Examples

Have a useful example? We'd love to include it! Please:
1. Follow the existing code style
2. Add comments explaining key concepts
3. Include a description in this README
4. Test your example before submitting
