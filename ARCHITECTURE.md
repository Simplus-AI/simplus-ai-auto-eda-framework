# Simplus AI Auto EDA Framework - Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [System Architecture](#system-architecture)
4. [Component Design](#component-design)
5. [Data Flow](#data-flow)
6. [Integration Patterns](#integration-patterns)
7. [Extension Points](#extension-points)
8. [Performance Considerations](#performance-considerations)

## Overview

The Simplus AI Auto EDA Framework is designed as a modular, extensible Python package that can be easily integrated into data science workflows, pipelines, and services. The architecture follows object-oriented design principles with clear separation of concerns.

### Key Design Goals
- **Modularity**: Independent, reusable components
- **Extensibility**: Easy to add new analyzers and visualizers
- **Configurability**: Flexible configuration system
- **Performance**: Efficient processing of large datasets
- **Integration**: Simple API for embedding in other services

## Design Principles

### 1. Separation of Concerns
Each module has a single, well-defined responsibility:
- **Analyzers**: Data analysis logic
- **Visualizers**: Visualization generation
- **Core**: Orchestration and coordination
- **Utils**: Supporting utilities

### 2. Dependency Injection
Components accept configuration through constructors, making them testable and flexible.

### 3. Plugin Architecture
New analyzers and visualizers can be added without modifying core code.

### 4. Data Pipeline Pattern
Analysis flows through a series of transformations, each producing structured output.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Application                       │
│              (Data Science Service/Pipeline)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Simplus EDA Package                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Core Layer (Orchestration)                │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │  │
│  │  │ EDAAnalyzer │  │ ReportGen    │  │  EDAConfig  │  │  │
│  │  └─────────────┘  └──────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────┴─────────────────────────────┐    │
│  │              Analysis Layer                         │    │
│  │  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │ Statistical  │  │ Data Quality │               │    │
│  │  │   Analyzer   │  │   Analyzer   │               │    │
│  │  └──────────────┘  └──────────────┘               │    │
│  │  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │ Correlation  │  │   Outlier    │               │    │
│  │  │   Analyzer   │  │   Analyzer   │               │    │
│  │  └──────────────┘  └──────────────┘               │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                    │
│  ┌──────────────────────┴─────────────────────────────┐    │
│  │           Visualization Layer                       │    │
│  │  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │ Distribution │  │ Relationship │               │    │
│  │  │  Visualizer  │  │  Visualizer  │               │    │
│  │  └──────────────┘  └──────────────┘               │    │
│  │  ┌──────────────┐                                  │    │
│  │  │  TimeSeries  │                                  │    │
│  │  │  Visualizer  │                                  │    │
│  │  └──────────────┘                                  │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                    │
│  ┌──────────────────────┴─────────────────────────────┐    │
│  │              Utility Layer                          │    │
│  │  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │ DataLoader   │  │  Validator   │               │    │
│  │  └──────────────┘  └──────────────┘               │    │
│  │  ┌──────────────┐                                  │    │
│  │  │  Formatter   │                                  │    │
│  │  └──────────────┘                                  │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Component Design

### Core Layer

#### EDAAnalyzer
**Purpose**: Main orchestrator for EDA process

**Responsibilities**:
- Coordinate analysis workflow
- Manage configuration
- Aggregate results from specialized analyzers
- Provide unified API

**Key Methods**:
```python
__init__(config: Optional[Dict[str, Any]] = None)
analyze(data: pd.DataFrame) -> Dict[str, Any]
```

#### ReportGenerator
**Purpose**: Generate reports in various formats

**Key Methods**:
```python
generate_html(results: Dict[str, Any], output_path: str) -> str
generate_json(results: Dict[str, Any], output_path: str) -> str
generate_pdf(results: Dict[str, Any], output_path: str) -> str
```

#### EDAConfig
**Purpose**: Configuration management using dataclass pattern

### Analysis Layer

Specialized analyzers for different aspects of EDA:
- **StatisticalAnalyzer**: Descriptive statistics, distributions, normality tests
- **DataQualityAnalyzer**: Missing values, duplicates, consistency checks
- **CorrelationAnalyzer**: Feature relationships and multicollinearity
- **OutlierAnalyzer**: Multiple outlier detection methods

### Visualization Layer

- **DistributionVisualizer**: Histograms, box plots, density plots
- **RelationshipVisualizer**: Correlation heatmaps, scatter matrices
- **TimeSeriesVisualizer**: Time series plots, seasonal decomposition

### Utility Layer

- **DataLoader**: Multi-format data loading
- **DataValidator**: Data validation and checks
- **OutputFormatter**: Result formatting utilities

## Data Flow

```
Input DataFrame
    │
    ▼
EDAAnalyzer.analyze()
    │
    ├──► StatisticalAnalyzer
    ├──► DataQualityAnalyzer
    ├──► CorrelationAnalyzer
    └──► OutlierAnalyzer
    │
    ▼
Results Dictionary
    │
    ▼
ReportGenerator
    │
    ├──► HTML Report
    ├──► JSON Report
    └──► PDF Report
```

## Integration Patterns

### 1. Direct Integration
```python
from simplus_eda import EDAAnalyzer
analyzer = EDAAnalyzer()
results = analyzer.analyze(data)
```

### 2. Pipeline Integration
```python
class DataPipeline:
    def __init__(self):
        self.eda_analyzer = EDAAnalyzer()

    def process(self, data):
        eda_results = self.eda_analyzer.analyze(data)
        return eda_results
```

### 3. API Service Integration
```python
@app.route('/analyze', methods=['POST'])
def perform_eda():
    data = load_data_from_request()
    analyzer = EDAAnalyzer()
    return analyzer.analyze(data)
```

## Extension Points

### Adding New Analyzers
Create a new analyzer class with an `analyze()` method and integrate it into the main EDAAnalyzer.

### Adding New Visualizers
Implement visualization methods and register them in the visualizers module.

### Custom Report Formats
Extend ReportGenerator with new format methods.

## Performance Considerations

1. **Large Dataset Handling**: Sampling for visualizations via configuration
2. **Memory Optimization**: Efficient data types and chunked processing
3. **Parallel Processing**: Future support for concurrent analysis
4. **Caching**: Results caching for repeated analyses

## Security Considerations

- Input validation for file paths and data
- Data size limits to prevent resource exhaustion
- Output path sanitization
- Secure handling of sensitive data

## Future Enhancements

- Advanced statistical tests
- Machine learning integration
- Interactive visualizations (Plotly)
- Streaming data support
- GPU acceleration
- AI-powered insight generation
- Plugin system for extensions
- Distributed computing (Dask/Spark)
