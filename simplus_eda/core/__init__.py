"""
Core modules for the Simplus EDA framework.

This package provides the core functionality for automated exploratory data analysis,
including configuration management, data analysis, report generation, and a unified API.
"""

from simplus_eda.core.config import EDAConfig, ConfigurationError
from simplus_eda.core.analyzer import EDAAnalyzer
from simplus_eda.core.report import ReportGenerator
from simplus_eda.core.eda import SimplusEDA, quick_analysis, analyze_and_report

__all__ = [
    # Configuration
    "EDAConfig",
    "ConfigurationError",

    # Core components
    "EDAAnalyzer",
    "ReportGenerator",

    # Unified API (recommended)
    "SimplusEDA",

    # Convenience functions
    "quick_analysis",
    "analyze_and_report",
]
