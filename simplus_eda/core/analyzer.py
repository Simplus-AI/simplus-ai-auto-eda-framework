"""
Main EDA Analyzer class for automated exploratory data analysis.
"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from datetime import datetime

from simplus_eda.analyzers import (
    StatisticalAnalyzer,
    DataQualityAnalyzer,
    CorrelationAnalyzer,
    OutlierAnalyzer
)
from simplus_eda.core.config import EDAConfig


class EDAAnalyzer:
    """
    Main analyzer class for automated exploratory data analysis.

    This class orchestrates the entire EDA process including statistical analysis,
    data quality assessment, outlier detection, and correlation analysis.

    Example:
        >>> analyzer = EDAAnalyzer()
        >>> results = analyzer.analyze(df)
        >>> print(results['overview']['shape'])
        >>> print(results['quality']['completeness_score'])
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], EDAConfig]] = None):
        """
        Initialize the EDA Analyzer.

        Args:
            config: Optional configuration (dictionary or EDAConfig object)
        """
        if isinstance(config, EDAConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = EDAConfig.from_dict(config)
        else:
            self.config = EDAConfig()

        self.data = None
        self.results = {}

        # Initialize specialized analyzers
        self._statistical_analyzer = StatisticalAnalyzer()
        self._quality_analyzer = DataQualityAnalyzer()
        self._correlation_analyzer = CorrelationAnalyzer()
        self._outlier_analyzer = OutlierAnalyzer()

    def analyze(self, data: pd.DataFrame, quick: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive automated EDA on the provided dataset.

        Args:
            data: Pandas DataFrame to analyze
            quick: If True, skip time-consuming analyses (default: False)

        Returns:
            Dictionary containing comprehensive analysis results with keys:
            - overview: Basic dataset information
            - statistics: Statistical analysis results
            - quality: Data quality assessment
            - outliers: Outlier detection results
            - correlations: Correlation analysis results
            - insights: Auto-generated insights
            - metadata: Analysis metadata (timestamp, config, etc.)

        Raises:
            ValueError: If data is not a pandas DataFrame
            ValueError: If data is empty
        """
        # Validate input
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("Data cannot be empty")

        self.data = data.copy()  # Work with a copy to avoid modifying original

        # Record analysis metadata
        analysis_start = datetime.now()

        # Perform all analyses
        self.results = {
            "overview": self._get_overview(),
            "statistics": self._get_statistics(quick=quick),
            "quality": self._assess_quality(),
            "outliers": self._detect_outliers() if not quick else {},
            "correlations": self._analyze_correlations(),
            "insights": self._generate_insights(),
            "metadata": self._get_metadata(analysis_start)
        }

        return self.results

    def _get_overview(self) -> Dict[str, Any]:
        """
        Get basic dataset overview including shape, columns, memory usage.

        Returns:
            Dictionary with dataset overview information
        """
        if self.data is None:
            return {}

        # Get column types
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.data.select_dtypes(include=['datetime64']).columns.tolist()

        # Calculate memory usage
        memory_bytes = self.data.memory_usage(deep=True).sum()
        memory_mb = memory_bytes / (1024 * 1024)

        return {
            "shape": {
                "rows": len(self.data),
                "columns": len(self.data.columns)
            },
            "columns": {
                "all": self.data.columns.tolist(),
                "numeric": numeric_cols,
                "categorical": categorical_cols,
                "datetime": datetime_cols
            },
            "column_counts": {
                "total": len(self.data.columns),
                "numeric": len(numeric_cols),
                "categorical": len(categorical_cols),
                "datetime": len(datetime_cols)
            },
            "dtypes": self.data.dtypes.astype(str).to_dict(),
            "memory_usage": {
                "total_bytes": int(memory_bytes),
                "total_mb": round(memory_mb, 2),
                "per_column": self.data.memory_usage(deep=True).to_dict()
            },
            "duplicates": {
                "count": int(self.data.duplicated().sum()),
                "percentage": round(self.data.duplicated().sum() / len(self.data) * 100, 2)
            }
        }

    def _get_statistics(self, quick: bool = False) -> Dict[str, Any]:
        """
        Calculate comprehensive descriptive statistics using StatisticalAnalyzer.

        Args:
            quick: If True, skip time-consuming statistical tests

        Returns:
            Dictionary with statistical analysis results
        """
        if self.data is None:
            return {}

        try:
            # Use StatisticalAnalyzer for comprehensive analysis
            stats_results = self._statistical_analyzer.analyze(self.data)

            # Optionally skip hypothesis tests if quick mode
            if quick and 'hypothesis_tests' in stats_results:
                stats_results.pop('hypothesis_tests', None)

            return stats_results
        except Exception as e:
            return {"error": f"Statistical analysis failed: {str(e)}"}

    def _assess_quality(self) -> Dict[str, Any]:
        """
        Assess data quality using DataQualityAnalyzer.

        Returns:
            Dictionary with data quality assessment results
        """
        if self.data is None:
            return {}

        try:
            # Use DataQualityAnalyzer for comprehensive quality assessment
            quality_results = self._quality_analyzer.analyze(self.data)

            # Add quality warnings based on configured thresholds
            warnings = self._generate_quality_warnings(quality_results)
            if warnings:
                quality_results['warnings'] = warnings

            return quality_results
        except Exception as e:
            return {"error": f"Quality analysis failed: {str(e)}"}

    def _detect_outliers(self) -> Dict[str, Any]:
        """
        Detect outliers using OutlierAnalyzer.

        Returns:
            Dictionary with outlier detection results
        """
        if self.data is None:
            return {}

        try:
            # Use OutlierAnalyzer for outlier detection
            outlier_results = self._outlier_analyzer.analyze(self.data)

            # Add summary statistics if not already present
            if 'summary' not in outlier_results and 'columns' in outlier_results:
                total_outliers = sum(
                    col_data.get('outlier_count', 0)
                    for col_data in outlier_results['columns'].values()
                )
                num_cols = len(self.data.select_dtypes(include=[np.number]).columns)
                total_data_points = len(self.data) * num_cols if num_cols > 0 else 1

                outlier_results['summary'] = {
                    "total_outliers": total_outliers,
                    "columns_with_outliers": sum(
                        1 for col_data in outlier_results['columns'].values()
                        if col_data.get('outlier_count', 0) > 0
                    ),
                    "outlier_percentage": round(
                        total_outliers / total_data_points * 100, 2
                    ) if total_data_points > 0 else 0.0
                }
            elif 'summary' in outlier_results and 'outlier_percentage' not in outlier_results['summary']:
                # Add outlier_percentage if missing
                total_outliers = outlier_results['summary'].get('total_outliers', 0)
                num_cols = len(self.data.select_dtypes(include=[np.number]).columns)
                total_data_points = len(self.data) * num_cols if num_cols > 0 else 1
                outlier_results['summary']['outlier_percentage'] = round(
                    total_outliers / total_data_points * 100, 2
                ) if total_data_points > 0 else 0.0

            return outlier_results
        except Exception as e:
            return {"error": f"Outlier detection failed: {str(e)}"}

    def _analyze_correlations(self) -> Dict[str, Any]:
        """
        Analyze feature correlations using CorrelationAnalyzer.

        Returns:
            Dictionary with correlation analysis results
        """
        if self.data is None:
            return {}

        # Skip if no numeric columns
        if len(self.data.select_dtypes(include=[np.number]).columns) < 2:
            return {
                "message": "Insufficient numeric columns for correlation analysis",
                "numeric_columns_count": len(self.data.select_dtypes(include=[np.number]).columns)
            }

        try:
            # Use CorrelationAnalyzer with configured threshold
            correlation_results = self._correlation_analyzer.analyze(
                self.data,
                threshold=self.config.correlation_threshold
            )

            return correlation_results
        except Exception as e:
            return {"error": f"Correlation analysis failed: {str(e)}"}

    def _generate_insights(self) -> Dict[str, Any]:
        """
        Generate automated insights from analysis results.

        Returns:
            Dictionary with auto-generated insights
        """
        if self.data is None or not self.results:
            return {}

        insights = {
            "data_characteristics": [],
            "quality_issues": [],
            "statistical_findings": [],
            "recommendations": []
        }

        # Data characteristics insights
        overview = self.results.get('overview', {})
        if overview:
            shape = overview.get('shape', {})
            col_counts = overview.get('column_counts', {})

            insights["data_characteristics"].append(
                f"Dataset contains {shape.get('rows', 0):,} rows and {shape.get('columns', 0)} columns"
            )

            if col_counts.get('numeric', 0) > 0:
                insights["data_characteristics"].append(
                    f"Found {col_counts['numeric']} numeric columns suitable for statistical analysis"
                )

            if col_counts.get('categorical', 0) > 0:
                insights["data_characteristics"].append(
                    f"Found {col_counts['categorical']} categorical columns"
                )

            # Duplicate insights
            duplicates = overview.get('duplicates', {})
            if duplicates.get('count', 0) > 0:
                insights["quality_issues"].append(
                    f"Found {duplicates['count']:,} duplicate rows ({duplicates.get('percentage', 0)}%)"
                )

        # Quality insights
        quality = self.results.get('quality', {})
        if quality and 'completeness' in quality:
            completeness = quality['completeness']
            completeness_score = completeness.get('completeness_score', 100)

            if completeness_score < 80:
                insights["quality_issues"].append(
                    f"Low data completeness: {completeness_score:.1f}% (significant missing data)"
                )
            elif completeness_score < 95:
                insights["quality_issues"].append(
                    f"Moderate data completeness: {completeness_score:.1f}%"
                )

            # Missing data per column
            missing_info = completeness.get('missing_by_column', {})
            high_missing_cols = [
                col for col, info in missing_info.items()
                if isinstance(info, dict) and info.get('percentage', 0) > self.config.missing_threshold * 100
            ]

            if high_missing_cols:
                insights["quality_issues"].append(
                    f"Columns with high missing rates (>{self.config.missing_threshold * 100}%): {', '.join(high_missing_cols[:5])}"
                )

        # Statistical insights
        stats = self.results.get('statistics', {})
        if stats and 'descriptive' in stats:
            descriptive = stats['descriptive']

            # High variance insights
            high_variance_cols = []
            for col, col_stats in descriptive.items():
                if isinstance(col_stats, dict):
                    std = col_stats.get('std', 0)
                    mean = col_stats.get('mean', 0)
                    if mean != 0 and abs(std / mean) > 1:  # Coefficient of variation > 1
                        high_variance_cols.append(col)

            if high_variance_cols:
                insights["statistical_findings"].append(
                    f"High variance detected in columns: {', '.join(high_variance_cols[:5])}"
                )

        # Outlier insights
        outliers = self.results.get('outliers', {})
        if outliers and 'summary' in outliers:
            summary = outliers['summary']
            outlier_pct = summary.get('outlier_percentage', 0)

            if outlier_pct > 5:
                insights["statistical_findings"].append(
                    f"Significant outliers detected: {outlier_pct:.1f}% of numeric data points"
                )
                insights["recommendations"].append(
                    "Consider outlier treatment: removal, transformation, or robust methods"
                )

        # Correlation insights
        correlations = self.results.get('correlations', {})
        if correlations and 'strong_correlations' in correlations:
            strong_corr = correlations['strong_correlations']
            if strong_corr:
                insights["statistical_findings"].append(
                    f"Found {len(strong_corr)} strong correlations (|r| > {self.config.correlation_threshold})"
                )
                insights["recommendations"].append(
                    "Strong correlations detected - consider feature selection or dimensionality reduction"
                )

        # General recommendations
        if not insights["quality_issues"]:
            insights["recommendations"].append(
                "Data quality is good - proceed with modeling"
            )
        else:
            insights["recommendations"].append(
                "Address data quality issues before modeling"
            )

        # Remove empty categories
        insights = {k: v for k, v in insights.items() if v}

        return insights

    def _generate_quality_warnings(self, quality_results: Dict[str, Any]) -> List[str]:
        """
        Generate quality warnings based on configured thresholds.

        Args:
            quality_results: Results from quality analysis

        Returns:
            List of warning messages
        """
        warnings = []

        if 'completeness' in quality_results:
            completeness_score = quality_results['completeness'].get('completeness_score', 100)
            if completeness_score < self.config.missing_threshold * 100:
                warnings.append(
                    f"Overall completeness ({completeness_score:.1f}%) below threshold "
                    f"({self.config.missing_threshold * 100}%)"
                )

        return warnings

    def _get_metadata(self, analysis_start: datetime) -> Dict[str, Any]:
        """
        Get analysis metadata including timestamp and configuration.

        Args:
            analysis_start: Timestamp when analysis started

        Returns:
            Dictionary with metadata
        """
        analysis_end = datetime.now()
        duration = (analysis_end - analysis_start).total_seconds()

        return {
            "timestamp": analysis_start.isoformat(),
            "duration_seconds": round(duration, 2),
            "analyzer_version": "1.0.0",
            "config": self.config.to_dict(),
            "data_shape": {
                "rows": len(self.data) if self.data is not None else 0,
                "columns": len(self.data.columns) if self.data is not None else 0
            }
        }

    def get_summary(self) -> str:
        """
        Get a human-readable summary of the analysis results.

        Returns:
            String with formatted summary
        """
        if not self.results:
            return "No analysis results available. Run analyze() first."

        summary_lines = ["=== EDA Analysis Summary ===\n"]

        # Overview
        if 'overview' in self.results:
            overview = self.results['overview']
            shape = overview.get('shape', {})
            summary_lines.append(f"Dataset: {shape.get('rows', 0):,} rows × {shape.get('columns', 0)} columns")

            col_counts = overview.get('column_counts', {})
            summary_lines.append(
                f"Columns: {col_counts.get('numeric', 0)} numeric, "
                f"{col_counts.get('categorical', 0)} categorical, "
                f"{col_counts.get('datetime', 0)} datetime"
            )

        # Quality
        if 'quality' in self.results:
            quality = self.results['quality']
            if 'completeness' in quality:
                completeness_score = quality['completeness'].get('completeness_score', 0)
                summary_lines.append(f"Data Completeness: {completeness_score:.1f}%")
            elif 'quality_score' in quality:
                overall_score = quality['quality_score'].get('overall_score', 0)
                summary_lines.append(f"Data Quality Score: {overall_score:.1f}%")

        # Outliers
        if 'outliers' in self.results and 'summary' in self.results['outliers']:
            outlier_summary = self.results['outliers']['summary']
            summary_lines.append(
                f"Outliers: {outlier_summary.get('total_outliers', 0)} detected "
                f"({outlier_summary.get('outlier_percentage', 0):.1f}%)"
            )

        # Insights
        if 'insights' in self.results:
            insights = self.results['insights']
            total_insights = sum(len(v) for v in insights.values() if isinstance(v, list))
            summary_lines.append(f"Insights Generated: {total_insights}")

        return "\n".join(summary_lines)
