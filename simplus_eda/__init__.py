"""
Simplus AI Auto EDA Framework

A comprehensive automated exploratory data analysis framework for Python.

Quick Start:
    >>> import pandas as pd
    >>> from simplus_eda import SimplusEDA
    >>>
    >>> # Load your data
    >>> df = pd.read_csv('data.csv')
    >>>
    >>> # Perform EDA and generate report
    >>> eda = SimplusEDA()
    >>> eda.analyze(df)
    >>> eda.generate_report('report.html')

One-liner:
    >>> from simplus_eda import quick_analysis
    >>> quick_analysis(df, 'report.html')
"""

__version__ = "0.1.0"
__author__ = "Simplus AI"

# Import unified API (recommended for most users)
from simplus_eda.core.eda import (
    SimplusEDA,
    quick_analysis,
    analyze_and_report
)

# Import core components (for advanced users)
from simplus_eda.core.config import EDAConfig
from simplus_eda.core.analyzer import EDAAnalyzer
from simplus_eda.core.report import ReportGenerator

# Import exceptions
from simplus_eda.exceptions import (
    SimplusEDAError,
    ConfigurationError,
    DataError,
    AnalysisError,
    ReportGenerationError
)

# Import logging
from simplus_eda.logging_config import get_logger, configure_logging

# Import caching
from simplus_eda.cache import (
    ResultCache,
    compute_dataframe_hash,
    compute_cache_key,
)

# Import progress tracking
from simplus_eda.progress import (
    ProgressTracker,
    ProgressCallback,
    progress_bar,
    progress_context,
    TQDM_AVAILABLE,
)

# Import statistical tests
from simplus_eda.analyzers.statistical_tests import (
    StatisticalTestsManager,
    ANOVATests,
    PostHocTests,
    NonParametricTests,
    CategoricalTests,
    VarianceTests,
    GrangerCausalityTest,
    EffectSize,
)

# Import time series analysis
from simplus_eda.analyzers.timeseries import (
    TimeSeriesAnalyzer,
    TimeSeriesValidator,
    StationarityTests,
    TrendAnalyzer,
    SeasonalityAnalyzer,
    AutocorrelationAnalyzer,
    TimeSeriesForecaster,
    TimeSeriesComparator,
)

# Import feature engineering
from simplus_eda.analyzers.feature_engineering import (
    FeatureEngineeringManager,
    InteractionDetector,
    PolynomialDetector,
    BinningRecommender,
    EncodingRecommender,
    ScalingRecommender,
    ScalingMethod,
    EncodingMethod,
    BinningStrategy,
)

# Import backends (optional)
try:
    from simplus_eda.backends.dask_backend import DaskBackend
    _DASK_BACKEND_AVAILABLE = True
except ImportError:
    _DASK_BACKEND_AVAILABLE = False
    DaskBackend = None

__all__ = [
    # Unified API (recommended)
    "SimplusEDA",
    "quick_analysis",
    "analyze_and_report",

    # Configuration
    "EDAConfig",

    # Core components (advanced usage)
    "EDAAnalyzer",
    "ReportGenerator",

    # Exceptions
    "SimplusEDAError",
    "ConfigurationError",
    "DataError",
    "AnalysisError",
    "ReportGenerationError",

    # Logging
    "get_logger",
    "configure_logging",

    # Caching
    "ResultCache",
    "compute_dataframe_hash",
    "compute_cache_key",

    # Progress tracking
    "ProgressTracker",
    "ProgressCallback",
    "progress_bar",
    "progress_context",
    "TQDM_AVAILABLE",

    # Statistical tests
    "StatisticalTestsManager",
    "ANOVATests",
    "PostHocTests",
    "NonParametricTests",
    "CategoricalTests",
    "VarianceTests",
    "GrangerCausalityTest",
    "EffectSize",

    # Time series analysis
    "TimeSeriesAnalyzer",
    "TimeSeriesValidator",
    "StationarityTests",
    "TrendAnalyzer",
    "SeasonalityAnalyzer",
    "AutocorrelationAnalyzer",
    "TimeSeriesForecaster",
    "TimeSeriesComparator",

    # Feature engineering
    "FeatureEngineeringManager",
    "InteractionDetector",
    "PolynomialDetector",
    "BinningRecommender",
    "EncodingRecommender",
    "ScalingRecommender",
    "ScalingMethod",
    "EncodingMethod",
    "BinningStrategy",

    # Backends (optional)
    "DaskBackend",
]
