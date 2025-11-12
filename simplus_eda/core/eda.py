"""
Unified EDA interface that seamlessly integrates config, analyzer, and report modules.

This module provides a high-level, user-friendly API that binds together all core
components of the Simplus EDA Framework.
"""

from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import pandas as pd

from simplus_eda.core.config import EDAConfig, ConfigurationError
from simplus_eda.core.analyzer import EDAAnalyzer
from simplus_eda.core.report import ReportGenerator


class SimplusEDA:
    """
    Unified interface for the Simplus EDA Framework.

    This class provides a streamlined API that integrates configuration,
    analysis, and report generation into a single, cohesive workflow.

    The SimplusEDA class follows a builder pattern, allowing you to:
    1. Configure analysis parameters
    2. Perform comprehensive EDA
    3. Generate professional reports
    4. Access individual components as needed

    Examples:
        Basic usage with defaults:
        >>> eda = SimplusEDA()
        >>> eda.analyze(df)
        >>> eda.generate_report('report.html')

        Customized analysis:
        >>> eda = SimplusEDA(
        ...     correlation_threshold=0.8,
        ...     outlier_method='zscore',
        ...     verbose=True
        ... )
        >>> eda.analyze(df)
        >>> eda.generate_report('report.html', title='My Analysis')

        Advanced workflow:
        >>> config = EDAConfig(
        ...     correlation_threshold=0.7,
        ...     missing_threshold=0.1,
        ...     enable_visualizations=True
        ... )
        >>> eda = SimplusEDA(config=config)
        >>> results = eda.analyze(df)
        >>> print(eda.summary())
        >>> eda.generate_report('report.html')
        >>> eda.export_results('results.json')
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], EDAConfig]] = None,
        **kwargs
    ):
        """
        Initialize the Simplus EDA interface.

        Args:
            config: Optional EDAConfig object or configuration dictionary.
                   If not provided, default configuration will be used.
            **kwargs: Additional configuration parameters that will override
                     settings in the config object. Supported parameters:
                     - correlation_threshold: float (0-1)
                     - missing_threshold: float (0-1)
                     - outlier_method: str ('iqr', 'zscore', etc.)
                     - enable_statistical_tests: bool
                     - enable_visualizations: bool
                     - verbose: bool
                     - And all other EDAConfig parameters

        Example:
            >>> # Using default configuration
            >>> eda = SimplusEDA()

            >>> # Using configuration dictionary
            >>> eda = SimplusEDA(config={'correlation_threshold': 0.8})

            >>> # Using EDAConfig object
            >>> config = EDAConfig(correlation_threshold=0.8, verbose=True)
            >>> eda = SimplusEDA(config=config)

            >>> # Using kwargs (most convenient)
            >>> eda = SimplusEDA(
            ...     correlation_threshold=0.8,
            ...     outlier_method='zscore',
            ...     verbose=True
            ... )
        """
        # Initialize configuration
        if config is None:
            self.config = EDAConfig(**kwargs) if kwargs else EDAConfig()
        elif isinstance(config, EDAConfig):
            self.config = config.update(**kwargs) if kwargs else config
        elif isinstance(config, dict):
            combined_config = {**config, **kwargs}
            self.config = EDAConfig.from_dict(combined_config)
        else:
            raise ConfigurationError(
                f"config must be EDAConfig, dict, or None. Got {type(config)}"
            )

        # Initialize core components
        self._analyzer = EDAAnalyzer(config=self.config)
        self._report_generator = None  # Lazy initialization

        # State management
        self._data = None
        self._results = None
        self._analyzed = False

    @property
    def analyzer(self) -> EDAAnalyzer:
        """Get the underlying EDAAnalyzer instance."""
        return self._analyzer

    @property
    def report_generator(self) -> ReportGenerator:
        """Get the underlying ReportGenerator instance (lazy initialization)."""
        if self._report_generator is None:
            self._report_generator = ReportGenerator()
        return self._report_generator

    @property
    def results(self) -> Optional[Dict[str, Any]]:
        """Get the analysis results. Returns None if analyze() hasn't been called."""
        return self._results

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get the analyzed DataFrame. Returns None if analyze() hasn't been called."""
        return self._data

    def analyze(
        self,
        data: pd.DataFrame,
        quick: bool = False,
        inplace: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive EDA on the provided dataset.

        This method runs all configured analyses including statistical analysis,
        data quality assessment, outlier detection, and correlation analysis.

        Args:
            data: Pandas DataFrame to analyze
            quick: If True, skip time-consuming analyses (default: False)
            inplace: If True, store results internally for later use (default: True)

        Returns:
            Dictionary containing comprehensive analysis results

        Raises:
            ValueError: If data is not a pandas DataFrame or is empty

        Example:
            >>> eda = SimplusEDA()
            >>> results = eda.analyze(df)
            >>> print(f"Quality score: {results['quality']['quality_score']['overall_score']}")
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("Data cannot be empty")

        # Perform analysis
        results = self._analyzer.analyze(data, quick=quick)

        # Store results if inplace
        if inplace:
            self._data = data.copy()
            self._results = results
            self._analyzed = True

        return results

    def generate_report(
        self,
        output_path: Union[str, Path],
        format: str = 'html',
        data: Optional[pd.DataFrame] = None,
        results: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        author: Optional[str] = None,
        company: Optional[str] = None,
        include_visualizations: bool = True,
        include_data_preview: bool = True,
        max_preview_rows: int = 10,
        include_toc: bool = True
    ) -> str:
        """
        Generate a comprehensive EDA report.

        This method creates professional reports in various formats (HTML, JSON, PDF)
        using the analysis results. If analyze() was called previously, the stored
        results will be used automatically.

        Args:
            output_path: Path where the report will be saved
            format: Report format ('html', 'json', 'pdf')
            data: Optional DataFrame to use. If None, uses stored data from analyze()
            results: Optional analysis results. If None, uses stored results
            title: Custom report title
            author: Report author name
            company: Company/organization name
            include_visualizations: Whether to include embedded visualizations
            include_data_preview: Whether to include data preview table
            max_preview_rows: Maximum rows to show in data preview
            include_toc: Whether to include table of contents

        Returns:
            Path to the generated report

        Raises:
            RuntimeError: If no data/results available and none provided
            ValueError: If format is not supported

        Example:
            >>> eda = SimplusEDA()
            >>> eda.analyze(df)
            >>>
            >>> # Simple report generation
            >>> eda.generate_report('report.html')
            >>>
            >>> # Customized report
            >>> eda.generate_report(
            ...     'report.html',
            ...     title='Sales Analysis',
            ...     author='Data Team',
            ...     include_visualizations=True
            ... )
        """
        # Validate that we have data and results
        use_data = data if data is not None else self._data
        use_results = results if results is not None else self._results

        if use_data is None or use_results is None:
            raise RuntimeError(
                "No data or results available. Either call analyze() first "
                "or provide data and results parameters."
            )

        # Build report configuration
        report_config = {
            'include_toc': include_toc
        }

        if title:
            report_config['title'] = title
        if author:
            report_config['author'] = author
        if company:
            report_config['company'] = company

        # Initialize report generator with config
        if self._report_generator is None or report_config:
            self._report_generator = ReportGenerator(config=report_config)

        # Generate report
        report_path = self._report_generator.generate_report(
            results=use_results,
            data=use_data,
            output_path=output_path,
            format=format,
            include_visualizations=include_visualizations,
            include_data_preview=include_data_preview,
            max_preview_rows=max_preview_rows
        )

        return report_path

    def export_results(
        self,
        output_path: Union[str, Path],
        results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export analysis results to JSON format.

        This is a convenience method for generating JSON reports. It's equivalent
        to calling generate_report() with format='json' but with a simpler API.

        Args:
            output_path: Path where the JSON file will be saved
            results: Optional analysis results. If None, uses stored results

        Returns:
            Path to the exported JSON file

        Raises:
            RuntimeError: If no results available and none provided

        Example:
            >>> eda = SimplusEDA()
            >>> eda.analyze(df)
            >>> eda.export_results('results.json')
        """
        use_results = results if results is not None else self._results

        if use_results is None:
            raise RuntimeError(
                "No results available. Call analyze() first or provide results parameter."
            )

        return self._report_generator.generate_json(
            results=use_results,
            output_path=str(output_path)
        )

    def summary(self) -> str:
        """
        Get a human-readable summary of the analysis results.

        This method provides a quick overview of the key findings from the analysis,
        including dataset dimensions, quality scores, and insights.

        Returns:
            Formatted string with analysis summary

        Raises:
            RuntimeError: If analyze() hasn't been called yet

        Example:
            >>> eda = SimplusEDA()
            >>> eda.analyze(df)
            >>> print(eda.summary())
            === EDA Analysis Summary ===
            Dataset: 1,000 rows × 10 columns
            ...
        """
        if not self._analyzed:
            raise RuntimeError("No analysis results available. Call analyze() first.")

        return self._analyzer.get_summary()

    def quick_report(
        self,
        data: pd.DataFrame,
        output_path: Union[str, Path],
        title: Optional[str] = None,
        format: str = 'html'
    ) -> str:
        """
        Perform quick analysis and generate report in one step.

        This is a convenience method that combines analyze() and generate_report()
        with quick=True for faster execution. Ideal for rapid exploratory analysis.

        Args:
            data: Pandas DataFrame to analyze
            output_path: Path where the report will be saved
            title: Optional custom report title
            format: Report format ('html', 'json', 'pdf')

        Returns:
            Path to the generated report

        Example:
            >>> eda = SimplusEDA()
            >>> report_path = eda.quick_report(df, 'quick_report.html')
            >>> print(f"Report generated: {report_path}")
        """
        # Perform quick analysis
        self.analyze(data, quick=True, inplace=True)

        # Generate report
        return self.generate_report(
            output_path=output_path,
            format=format,
            title=title or "Quick EDA Report",
            include_visualizations=True
        )

    def get_insights(self) -> Dict[str, List[str]]:
        """
        Get auto-generated insights from the analysis.

        Returns:
            Dictionary with categorized insights:
            - data_characteristics: Basic data properties
            - quality_issues: Data quality problems detected
            - statistical_findings: Statistical patterns found
            - recommendations: Suggested next steps

        Raises:
            RuntimeError: If analyze() hasn't been called yet

        Example:
            >>> eda = SimplusEDA()
            >>> eda.analyze(df)
            >>> insights = eda.get_insights()
            >>> for issue in insights['quality_issues']:
            ...     print(f"⚠️  {issue}")
        """
        if not self._analyzed:
            raise RuntimeError("No analysis results available. Call analyze() first.")

        return self._results.get('insights', {})

    def get_quality_score(self) -> float:
        """
        Get the overall data quality score.

        Returns:
            Quality score as a percentage (0-100)

        Raises:
            RuntimeError: If analyze() hasn't been called yet

        Example:
            >>> eda = SimplusEDA()
            >>> eda.analyze(df)
            >>> score = eda.get_quality_score()
            >>> print(f"Data quality: {score:.1f}%")
        """
        if not self._analyzed:
            raise RuntimeError("No analysis results available. Call analyze() first.")

        quality = self._results.get('quality', {})
        quality_score = quality.get('quality_score', {})
        return quality_score.get('overall_score', 0.0)

    def get_correlations(
        self,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get strong correlations from the analysis.

        Args:
            threshold: Optional correlation threshold to filter results.
                      If None, uses the configured threshold.

        Returns:
            List of correlation pairs with their coefficients

        Raises:
            RuntimeError: If analyze() hasn't been called yet

        Example:
            >>> eda = SimplusEDA()
            >>> eda.analyze(df)
            >>> correlations = eda.get_correlations(threshold=0.8)
            >>> for corr in correlations:
            ...     print(f"{corr['feature1']} <-> {corr['feature2']}: {corr['correlation']:.3f}")
        """
        if not self._analyzed:
            raise RuntimeError("No analysis results available. Call analyze() first.")

        correlations = self._results.get('correlations', {})
        strong_corr = correlations.get('strong_correlations', {})

        # Handle both dict and list formats
        if isinstance(strong_corr, dict):
            pairs = strong_corr.get('pairs', [])
        else:
            pairs = strong_corr if isinstance(strong_corr, list) else []

        # Filter by threshold if provided
        if threshold is not None:
            pairs = [
                p for p in pairs
                if abs(p.get('correlation', 0)) >= threshold
            ]

        return pairs

    def update_config(self, **kwargs) -> 'SimplusEDA':
        """
        Update configuration parameters.

        This creates a new configuration with updated parameters while preserving
        existing settings. The analyzer is re-initialized with the new configuration.

        Args:
            **kwargs: Configuration parameters to update

        Returns:
            Self for method chaining

        Example:
            >>> eda = SimplusEDA()
            >>> eda.update_config(
            ...     correlation_threshold=0.9,
            ...     outlier_method='isolation_forest'
            ... )
            >>> eda.analyze(df)
        """
        self.config = self.config.update(**kwargs)
        self._analyzer = EDAAnalyzer(config=self.config)
        return self

    def save_config(self, output_path: Union[str, Path]) -> None:
        """
        Save the current configuration to a JSON file.

        Args:
            output_path: Path where the configuration will be saved

        Example:
            >>> eda = SimplusEDA(correlation_threshold=0.8)
            >>> eda.save_config('my_config.json')
        """
        self.config.to_json(str(output_path))

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> 'SimplusEDA':
        """
        Create a SimplusEDA instance from a configuration file.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            New SimplusEDA instance with loaded configuration

        Example:
            >>> eda = SimplusEDA.from_config_file('my_config.json')
            >>> eda.analyze(df)
        """
        config = EDAConfig.from_json(str(config_path))
        return cls(config=config)

    def __repr__(self) -> str:
        """String representation of the SimplusEDA instance."""
        status = "analyzed" if self._analyzed else "not analyzed"
        data_info = ""

        if self._data is not None:
            data_info = f", data: {self._data.shape[0]} rows × {self._data.shape[1]} cols"

        return f"SimplusEDA(status={status}{data_info})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self._analyzed:
            return self.summary()
        else:
            return "SimplusEDA instance - No analysis performed yet. Call analyze() to begin."


# Convenience functions for one-liners

def quick_analysis(
    data: pd.DataFrame,
    output_path: str = 'eda_report.html',
    **config_kwargs
) -> str:
    """
    Perform quick EDA analysis and generate report in one function call.

    This is the simplest way to use the Simplus EDA Framework. It performs
    a quick analysis and generates an HTML report with a single function call.

    Args:
        data: Pandas DataFrame to analyze
        output_path: Path where the report will be saved (default: 'eda_report.html')
        **config_kwargs: Optional configuration parameters

    Returns:
        Path to the generated report

    Example:
        >>> import pandas as pd
        >>> from simplus_eda import quick_analysis
        >>>
        >>> df = pd.read_csv('data.csv')
        >>> quick_analysis(df, 'my_report.html', correlation_threshold=0.8)
        'my_report.html'
    """
    eda = SimplusEDA(**config_kwargs)
    return eda.quick_report(data, output_path)


def analyze_and_report(
    data: pd.DataFrame,
    output_path: str,
    config: Optional[Union[Dict, EDAConfig]] = None,
    format: str = 'html',
    **report_kwargs
) -> tuple[Dict[str, Any], str]:
    """
    Perform full analysis and generate report, returning both results and report path.

    Args:
        data: Pandas DataFrame to analyze
        output_path: Path where the report will be saved
        config: Optional configuration
        format: Report format ('html', 'json', 'pdf')
        **report_kwargs: Additional report configuration (title, author, etc.)

    Returns:
        Tuple of (analysis_results, report_path)

    Example:
        >>> results, report_path = analyze_and_report(
        ...     df,
        ...     'report.html',
        ...     title='Sales Analysis'
        ... )
        >>> print(f"Quality score: {results['quality']['quality_score']['overall_score']}")
    """
    eda = SimplusEDA(config=config)
    results = eda.analyze(data)
    report_path = eda.generate_report(output_path, format=format, **report_kwargs)
    return results, report_path
