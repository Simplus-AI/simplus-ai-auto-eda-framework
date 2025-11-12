"""
Report generation module for creating comprehensive EDA reports.

This module integrates all analysis and visualization components to generate
master reports in multiple formats (HTML, JSON, PDF).
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
import json
import base64
from io import BytesIO
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from simplus_eda.core.analyzer import EDAAnalyzer
from simplus_eda.visualizers.distributions import DistributionVisualizer
from simplus_eda.visualizers.relationships import RelationshipVisualizer
from simplus_eda.visualizers.timeseries import TimeSeriesVisualizer


class ReportGenerator:
    """
    Generate comprehensive EDA reports in various formats.

    This class integrates analysis results from EDAAnalyzer and visualizations
    from all visualizer modules to create comprehensive, professional reports.

    Supports:
    - HTML reports with embedded visualizations
    - JSON reports for programmatic access
    - PDF reports (requires additional dependencies)

    Example:
        >>> analyzer = EDAAnalyzer()
        >>> results = analyzer.analyze(df)
        >>>
        >>> report_gen = ReportGenerator()
        >>> report_gen.generate_report(
        ...     results=results,
        ...     data=df,
        ...     output_path='report.html',
        ...     format='html',
        ...     include_visualizations=True
        ... )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Report Generator.

        Args:
            config: Optional configuration for report customization
                - title: Custom report title
                - author: Report author name
                - company: Company/organization name
                - theme: Color theme ('light', 'dark')
                - logo_path: Path to logo image
                - include_toc: Include table of contents (default: True)
                - max_figures: Maximum number of figures to include
        """
        self.config = config or {}

        # Initialize visualizers
        self.dist_viz = DistributionVisualizer()
        self.rel_viz = RelationshipVisualizer()
        self.ts_viz = TimeSeriesVisualizer()

        # Report metadata
        self.report_metadata = {
            'generated_at': datetime.now().isoformat(),
            'generator_version': '1.0.0',
            'config': self.config
        }

    def generate_report(
        self,
        results: Dict[str, Any],
        data: pd.DataFrame,
        output_path: Union[str, Path],
        format: str = 'html',
        include_visualizations: bool = True,
        include_data_preview: bool = True,
        max_preview_rows: int = 10
    ) -> str:
        """
        Generate a comprehensive master report.

        Args:
            results: Analysis results from EDAAnalyzer.analyze()
            data: Original DataFrame that was analyzed
            output_path: Path to save the report
            format: Output format ('html', 'json', 'pdf')
            include_visualizations: Whether to include visualizations
            include_data_preview: Whether to include data preview
            max_preview_rows: Maximum rows to show in preview

        Returns:
            Path to generated report

        Raises:
            ValueError: If format is not supported
            IOError: If unable to write report

        Example:
            >>> report_gen = ReportGenerator()
            >>> path = report_gen.generate_report(
            ...     results=analysis_results,
            ...     data=df,
            ...     output_path='my_report.html'
            ... )
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        format = format.lower()

        if format == 'html':
            return self.generate_html(
                results=results,
                data=data,
                output_path=str(output_path),
                include_visualizations=include_visualizations,
                include_data_preview=include_data_preview,
                max_preview_rows=max_preview_rows
            )
        elif format == 'json':
            return self.generate_json(
                results=results,
                output_path=str(output_path)
            )
        elif format == 'pdf':
            return self.generate_pdf(
                results=results,
                data=data,
                output_path=str(output_path),
                include_visualizations=include_visualizations
            )
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'html', 'json', or 'pdf'")

    def generate_html(
        self,
        results: Dict[str, Any],
        data: pd.DataFrame,
        output_path: str,
        include_visualizations: bool = True,
        include_data_preview: bool = True,
        max_preview_rows: int = 10
    ) -> str:
        """
        Generate comprehensive HTML report with embedded visualizations.

        Args:
            results: Analysis results dictionary from EDAAnalyzer
            data: Original DataFrame
            output_path: Path to save the HTML report
            include_visualizations: Whether to embed visualizations
            include_data_preview: Whether to include data preview
            max_preview_rows: Maximum rows in preview

        Returns:
            Path to generated report
        """
        # Generate visualizations if requested
        figures = {}
        if include_visualizations:
            figures = self._generate_visualizations(data, results)

        # Build HTML content
        html_content = self._build_html_structure(
            results=results,
            data=data,
            figures=figures,
            include_data_preview=include_data_preview,
            max_preview_rows=max_preview_rows
        )

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Clean up matplotlib figures
        plt.close('all')

        return output_path

    def generate_json(self, results: Dict[str, Any], output_path: str) -> str:
        """
        Generate JSON report from analysis results.

        This format is useful for:
        - Programmatic access to results
        - Integration with other tools
        - Version control and diffing
        - API responses

        Args:
            results: Analysis results dictionary
            output_path: Path to save the JSON report

        Returns:
            Path to generated report
        """
        # Prepare JSON-serializable results
        json_data = {
            'metadata': self.report_metadata,
            'results': self._prepare_json_serializable(results)
        }

        # Write to file with pretty formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

        return output_path

    def generate_pdf(
        self,
        results: Dict[str, Any],
        data: pd.DataFrame,
        output_path: str,
        include_visualizations: bool = True
    ) -> str:
        """
        Generate PDF report from analysis results.

        Note: Requires additional dependencies (reportlab or weasyprint).
        This is a basic implementation that converts HTML to PDF.

        Args:
            results: Analysis results dictionary
            data: Original DataFrame
            output_path: Path to save the PDF report
            include_visualizations: Whether to include visualizations

        Returns:
            Path to generated report

        Raises:
            ImportError: If required PDF generation libraries are not installed
        """
        try:
            from weasyprint import HTML
        except ImportError:
            raise ImportError(
                "PDF generation requires 'weasyprint'. "
                "Install with: pip install weasyprint"
            )

        # Generate HTML first
        html_path = output_path.replace('.pdf', '_temp.html')
        self.generate_html(
            results=results,
            data=data,
            output_path=html_path,
            include_visualizations=include_visualizations
        )

        # Convert HTML to PDF
        HTML(filename=html_path).write_pdf(output_path)

        # Clean up temporary HTML
        Path(html_path).unlink(missing_ok=True)

        return output_path

    def _generate_visualizations(
        self,
        data: pd.DataFrame,
        results: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate all visualizations and return as base64 encoded strings.

        Args:
            data: DataFrame to visualize
            results: Analysis results

        Returns:
            Dictionary mapping visualization names to base64 encoded images
        """
        figures = {}

        # Get numeric and categorical columns
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()

        try:
            # 1. Distribution visualizations (histograms)
            if len(numeric_cols) > 0:
                fig = self.dist_viz.create_histograms(
                    data,
                    columns=numeric_cols[:6],  # Limit to first 6 columns
                    kde=True
                )
                figures['distributions_histogram'] = self._fig_to_base64(fig)

            # 2. Boxplots for outlier visualization
            if len(numeric_cols) > 0:
                fig = self.dist_viz.create_boxplots(
                    data,
                    columns=numeric_cols[:6]
                )
                figures['distributions_boxplot'] = self._fig_to_base64(fig)

            # 3. Correlation heatmap
            if len(numeric_cols) >= 2:
                fig = self.rel_viz.create_correlation_heatmap(
                    data,
                    method='pearson',
                    mask_upper=True
                )
                figures['correlation_heatmap'] = self._fig_to_base64(fig)

            # 4. Highly correlated pairs
            if len(numeric_cols) >= 2:
                try:
                    # Get correlation threshold from results or use default
                    corr_threshold = results.get('metadata', {}).get('config', {}).get('correlation_threshold', 0.7)
                    fig = self.rel_viz.create_correlation_pairs(
                        data,
                        threshold=corr_threshold,
                        max_pairs=6
                    )
                    figures['correlation_pairs'] = self._fig_to_base64(fig)
                except ValueError:
                    # No strong correlations found
                    pass

            # 5. Time series plots (if datetime columns exist)
            if len(datetime_cols) > 0 and len(numeric_cols) > 0:
                time_col = datetime_cols[0]
                value_cols = numeric_cols[:3]  # Plot first 3 numeric columns

                fig = self.ts_viz.create_time_plot(
                    data,
                    time_col=time_col,
                    value_cols=value_cols,
                    show_trend=True
                )
                figures['timeseries_plot'] = self._fig_to_base64(fig)

        except Exception as e:
            warnings.warn(f"Error generating visualizations: {str(e)}")

        return figures

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64 encoded string.

        Args:
            fig: Matplotlib figure

        Returns:
            Base64 encoded string of the figure
        """
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close(fig)
        return image_base64

    def _build_html_structure(
        self,
        results: Dict[str, Any],
        data: pd.DataFrame,
        figures: Dict[str, str],
        include_data_preview: bool,
        max_preview_rows: int
    ) -> str:
        """
        Build complete HTML structure for the report.

        Args:
            results: Analysis results
            data: DataFrame
            figures: Dictionary of base64 encoded figures
            include_data_preview: Whether to include data preview
            max_preview_rows: Maximum preview rows

        Returns:
            Complete HTML string
        """
        # Extract key information
        overview = results.get('overview', {})
        statistics = results.get('statistics', {})
        quality = results.get('quality', {})
        outliers = results.get('outliers', {})
        correlations = results.get('correlations', {})
        insights = results.get('insights', {})
        metadata = results.get('metadata', {})

        # Build HTML
        html_parts = [
            self._html_header(),
            self._html_title_section(),
            self._html_toc(),
            self._html_executive_summary(overview, quality, insights),
            self._html_data_overview(overview, data, include_data_preview, max_preview_rows),
            self._html_data_quality(quality),
            self._html_statistical_analysis(statistics),
            self._html_outlier_analysis(outliers),
            self._html_correlation_analysis(correlations),
            self._html_insights_section(insights),
            self._html_visualizations(figures),
            self._html_metadata(metadata),
            self._html_footer()
        ]

        return '\n'.join(html_parts)

    def _html_header(self) -> str:
        """Generate HTML header with CSS styling."""
        title = self.config.get('title', 'EDA Analysis Report')

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}

        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}

        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}

        h3 {{
            color: #555;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}

        .executive-summary {{
            background-color: #e8f4f8;
            padding: 25px;
            border-radius: 8px;
            margin: 30px 0;
            border-left: 5px solid #3498db;
        }}

        .metric-card {{
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border-left: 4px solid #3498db;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .metric-item {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}

        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
            margin: 10px 0;
        }}

        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}

        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}

        tr:hover {{
            background-color: #f5f5f5;
        }}

        .visualization {{
            margin: 30px 0;
            text-align: center;
        }}

        .visualization img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }}

        .insight-list {{
            list-style: none;
            padding: 0;
        }}

        .insight-item {{
            padding: 12px 15px;
            margin: 8px 0;
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            border-radius: 4px;
        }}

        .insight-item.quality {{
            background-color: #f8d7da;
            border-left-color: #dc3545;
        }}

        .insight-item.statistical {{
            background-color: #d1ecf1;
            border-left-color: #17a2b8;
        }}

        .insight-item.recommendation {{
            background-color: #d4edda;
            border-left-color: #28a745;
        }}

        .toc {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 30px 0;
        }}

        .toc ul {{
            list-style: none;
        }}

        .toc li {{
            padding: 8px 0;
        }}

        .toc a {{
            color: #3498db;
            text-decoration: none;
            transition: color 0.3s;
        }}

        .toc a:hover {{
            color: #2980b9;
            text-decoration: underline;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }}

        .badge-success {{
            background-color: #d4edda;
            color: #155724;
        }}

        .badge-warning {{
            background-color: #fff3cd;
            color: #856404;
        }}

        .badge-danger {{
            background-color: #f8d7da;
            color: #721c24;
        }}

        .badge-info {{
            background-color: #d1ecf1;
            color: #0c5460;
        }}

        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}

        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border: 1px solid #ddd;
        }}

        code {{
            font-family: 'Courier New', monospace;
            color: #e83e8c;
        }}

        @media print {{
            body {{
                background-color: white;
            }}
            .container {{
                box-shadow: none;
                padding: 20px;
            }}
            .visualization img {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">"""

    def _html_title_section(self) -> str:
        """Generate title section."""
        title = self.config.get('title', 'EDA Analysis Report')
        author = self.config.get('author', '')
        company = self.config.get('company', '')

        author_info = f"<p><strong>Author:</strong> {author}</p>" if author else ""
        company_info = f"<p><strong>Organization:</strong> {company}</p>" if company else ""

        return f"""
        <h1>{title}</h1>
        <div style="margin-bottom: 30px; color: #666;">
            <p><strong>Generated:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            {author_info}
            {company_info}
        </div>"""

    def _html_toc(self) -> str:
        """Generate table of contents."""
        if not self.config.get('include_toc', True):
            return ""

        return """
        <div class="toc">
            <h2>Table of Contents</h2>
            <ul>
                <li><a href="#executive-summary">1. Executive Summary</a></li>
                <li><a href="#data-overview">2. Data Overview</a></li>
                <li><a href="#data-quality">3. Data Quality Assessment</a></li>
                <li><a href="#statistical-analysis">4. Statistical Analysis</a></li>
                <li><a href="#outlier-analysis">5. Outlier Analysis</a></li>
                <li><a href="#correlation-analysis">6. Correlation Analysis</a></li>
                <li><a href="#insights">7. Key Insights</a></li>
                <li><a href="#visualizations">8. Visualizations</a></li>
                <li><a href="#metadata">9. Analysis Metadata</a></li>
            </ul>
        </div>"""

    def _html_executive_summary(
        self,
        overview: Dict[str, Any],
        quality: Dict[str, Any],
        insights: Dict[str, Any]
    ) -> str:
        """Generate executive summary section."""
        shape = overview.get('shape', {})
        rows = shape.get('rows', 0)
        cols = shape.get('columns', 0)

        # Get quality score
        quality_score_data = quality.get('quality_score', {})
        quality_score = quality_score_data.get('overall_score', 0)
        quality_level = quality_score_data.get('quality_level', 'unknown')

        # Quality badge
        if quality_score >= 90:
            badge_class = 'badge-success'
        elif quality_score >= 75:
            badge_class = 'badge-info'
        elif quality_score >= 60:
            badge_class = 'badge-warning'
        else:
            badge_class = 'badge-danger'

        # Count insights
        total_insights = sum(len(v) for v in insights.values() if isinstance(v, list))

        return f"""
        <div id="executive-summary" class="executive-summary">
            <h2>Executive Summary</h2>
            <p>This report provides a comprehensive exploratory data analysis of the dataset.</p>

            <div class="metric-grid">
                <div class="metric-item">
                    <div class="metric-label">Total Rows</div>
                    <div class="metric-value">{rows:,}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Total Columns</div>
                    <div class="metric-value">{cols}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Quality Score</div>
                    <div class="metric-value">{quality_score:.1f}%</div>
                    <span class="badge {badge_class}">{quality_level.upper()}</span>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Key Insights</div>
                    <div class="metric-value">{total_insights}</div>
                </div>
            </div>
        </div>"""

    def _html_data_overview(
        self,
        overview: Dict[str, Any],
        data: pd.DataFrame,
        include_preview: bool,
        max_rows: int
    ) -> str:
        """Generate data overview section."""
        shape = overview.get('shape', {})
        column_counts = overview.get('column_counts', {})
        duplicates = overview.get('duplicates', {})
        memory = overview.get('memory_usage', {})

        html = f"""
        <h2 id="data-overview">Data Overview</h2>

        <div class="metric-card">
            <h3>Dataset Dimensions</h3>
            <p><strong>Rows:</strong> {shape.get('rows', 0):,}</p>
            <p><strong>Columns:</strong> {shape.get('columns', 0)}</p>
            <p><strong>Memory Usage:</strong> {memory.get('total_mb', 0):.2f} MB</p>
        </div>

        <div class="metric-card">
            <h3>Column Types</h3>
            <table>
                <tr>
                    <th>Type</th>
                    <th>Count</th>
                </tr>
                <tr>
                    <td>Numeric</td>
                    <td>{column_counts.get('numeric', 0)}</td>
                </tr>
                <tr>
                    <td>Categorical</td>
                    <td>{column_counts.get('categorical', 0)}</td>
                </tr>
                <tr>
                    <td>DateTime</td>
                    <td>{column_counts.get('datetime', 0)}</td>
                </tr>
            </table>
        </div>

        <div class="metric-card">
            <h3>Duplicate Rows</h3>
            <p><strong>Count:</strong> {duplicates.get('count', 0):,} ({duplicates.get('percentage', 0):.2f}%)</p>
        </div>"""

        if include_preview:
            preview_html = data.head(max_rows).to_html(classes='', border=0)
            html += f"""
        <div class="metric-card">
            <h3>Data Preview (First {max_rows} Rows)</h3>
            {preview_html}
        </div>"""

        return html

    def _html_data_quality(self, quality: Dict[str, Any]) -> str:
        """Generate data quality section."""
        missing = quality.get('missing_values', {})
        quality_score_data = quality.get('quality_score', {})
        component_scores = quality_score_data.get('component_scores', {})

        html = f"""
        <h2 id="data-quality">Data Quality Assessment</h2>

        <div class="metric-card">
            <h3>Quality Score Components</h3>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Score</th>
                </tr>"""

        for component, score in component_scores.items():
            html += f"""
                <tr>
                    <td>{component.replace('_', ' ').title()}</td>
                    <td>{score:.1f}%</td>
                </tr>"""

        html += """
            </table>
        </div>

        <div class="metric-card">
            <h3>Missing Values</h3>"""

        total_missing = missing.get('total_missing', 0)
        overall_rate = missing.get('overall_missing_rate', 0)

        html += f"""
            <p><strong>Total Missing:</strong> {total_missing:,} cells ({overall_rate:.2f}%)</p>
            <p><strong>Recommendation:</strong> {missing.get('recommendation', 'N/A')}</p>
        </div>"""

        return html

    def _html_statistical_analysis(self, statistics: Dict[str, Any]) -> str:
        """Generate statistical analysis section."""
        descriptive = statistics.get('descriptive', {})

        if not descriptive or 'numeric_columns' not in descriptive:
            return """
        <h2 id="statistical-analysis">Statistical Analysis</h2>
        <p>No numeric columns found for statistical analysis.</p>"""

        numeric_cols = descriptive.get('numeric_columns', {})

        html = f"""
        <h2 id="statistical-analysis">Statistical Analysis</h2>
        <p>Statistical summary for {len(numeric_cols)} numeric columns.</p>"""

        # Show statistics for first few columns
        for col_name, col_stats in list(numeric_cols.items())[:5]:
            if isinstance(col_stats, dict) and 'central_tendency' in col_stats:
                ct = col_stats['central_tendency']
                disp = col_stats['dispersion']

                html += f"""
        <div class="metric-card">
            <h3>{col_name}</h3>
            <div class="metric-grid">
                <div class="metric-item">
                    <div class="metric-label">Mean</div>
                    <div class="metric-value" style="font-size: 1.5em;">{ct.get('mean', 0):.2f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Median</div>
                    <div class="metric-value" style="font-size: 1.5em;">{ct.get('median', 0):.2f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Std Dev</div>
                    <div class="metric-value" style="font-size: 1.5em;">{disp.get('std', 0):.2f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Range</div>
                    <div class="metric-value" style="font-size: 1.5em;">{disp.get('range', 0):.2f}</div>
                </div>
            </div>
        </div>"""

        return html

    def _html_outlier_analysis(self, outliers: Dict[str, Any]) -> str:
        """Generate outlier analysis section."""
        summary = outliers.get('summary', {})

        if not summary:
            return """
        <h2 id="outlier-analysis">Outlier Analysis</h2>
        <p>No outlier analysis available.</p>"""

        total_outliers = summary.get('total_outliers', 0)
        outlier_pct = summary.get('outlier_percentage', 0)
        cols_with_outliers = summary.get('columns_with_outliers', 0)

        return f"""
        <h2 id="outlier-analysis">Outlier Analysis</h2>

        <div class="metric-card">
            <h3>Outlier Summary</h3>
            <div class="metric-grid">
                <div class="metric-item">
                    <div class="metric-label">Total Outliers</div>
                    <div class="metric-value">{total_outliers:,}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Percentage</div>
                    <div class="metric-value">{outlier_pct:.2f}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Affected Columns</div>
                    <div class="metric-value">{cols_with_outliers}</div>
                </div>
            </div>
        </div>"""

    def _html_correlation_analysis(self, correlations: Dict[str, Any]) -> str:
        """Generate correlation analysis section."""
        strong_corr_data = correlations.get('strong_correlations', {})

        # Handle both dict and list formats
        if isinstance(strong_corr_data, dict):
            strong_corr = strong_corr_data.get('pairs', [])
        else:
            strong_corr = strong_corr_data if isinstance(strong_corr_data, list) else []

        if not strong_corr:
            return """
        <h2 id="correlation-analysis">Correlation Analysis</h2>
        <p>No strong correlations found.</p>"""

        html = f"""
        <h2 id="correlation-analysis">Correlation Analysis</h2>

        <div class="metric-card">
            <h3>Strong Correlations ({len(strong_corr)} found)</h3>
            <table>
                <tr>
                    <th>Variable 1</th>
                    <th>Variable 2</th>
                    <th>Correlation</th>
                </tr>"""

        for corr in strong_corr[:10]:  # Show first 10
            # Handle both 'var1'/'var2' and 'feature1'/'feature2' keys
            var1 = corr.get('var1') or corr.get('feature1', 'N/A')
            var2 = corr.get('var2') or corr.get('feature2', 'N/A')
            correlation = corr.get('correlation', 0)

            html += f"""
                <tr>
                    <td>{var1}</td>
                    <td>{var2}</td>
                    <td>{correlation:.3f}</td>
                </tr>"""

        html += """
            </table>
        </div>"""

        return html

    def _html_insights_section(self, insights: Dict[str, Any]) -> str:
        """Generate insights section."""
        html = """
        <h2 id="insights">Key Insights</h2>"""

        # Data characteristics
        if 'data_characteristics' in insights and insights['data_characteristics']:
            html += """
        <div class="metric-card">
            <h3>Data Characteristics</h3>
            <ul class="insight-list">"""
            for insight in insights['data_characteristics']:
                html += f"""
                <li class="insight-item">{insight}</li>"""
            html += """
            </ul>
        </div>"""

        # Quality issues
        if 'quality_issues' in insights and insights['quality_issues']:
            html += """
        <div class="metric-card">
            <h3>Quality Issues</h3>
            <ul class="insight-list">"""
            for insight in insights['quality_issues']:
                html += f"""
                <li class="insight-item quality">{insight}</li>"""
            html += """
            </ul>
        </div>"""

        # Statistical findings
        if 'statistical_findings' in insights and insights['statistical_findings']:
            html += """
        <div class="metric-card">
            <h3>Statistical Findings</h3>
            <ul class="insight-list">"""
            for insight in insights['statistical_findings']:
                html += f"""
                <li class="insight-item statistical">{insight}</li>"""
            html += """
            </ul>
        </div>"""

        # Recommendations
        if 'recommendations' in insights and insights['recommendations']:
            html += """
        <div class="metric-card">
            <h3>Recommendations</h3>
            <ul class="insight-list">"""
            for insight in insights['recommendations']:
                html += f"""
                <li class="insight-item recommendation">{insight}</li>"""
            html += """
            </ul>
        </div>"""

        return html

    def _html_visualizations(self, figures: Dict[str, str]) -> str:
        """Generate visualizations section."""
        if not figures:
            return ""

        html = """
        <h2 id="visualizations">Visualizations</h2>"""

        viz_titles = {
            'distributions_histogram': 'Distribution Analysis - Histograms',
            'distributions_boxplot': 'Distribution Analysis - Boxplots',
            'correlation_heatmap': 'Correlation Heatmap',
            'correlation_pairs': 'Highly Correlated Variable Pairs',
            'timeseries_plot': 'Time Series Analysis'
        }

        for viz_key, viz_base64 in figures.items():
            title = viz_titles.get(viz_key, viz_key.replace('_', ' ').title())
            html += f"""
        <div class="visualization">
            <h3>{title}</h3>
            <img src="data:image/png;base64,{viz_base64}" alt="{title}">
        </div>"""

        return html

    def _html_metadata(self, metadata: Dict[str, Any]) -> str:
        """Generate metadata section."""
        timestamp = metadata.get('timestamp', 'N/A')
        duration = metadata.get('duration_seconds', 0)
        version = metadata.get('analyzer_version', 'N/A')

        return f"""
        <h2 id="metadata">Analysis Metadata</h2>

        <div class="metric-card">
            <p><strong>Analysis Timestamp:</strong> {timestamp}</p>
            <p><strong>Analysis Duration:</strong> {duration:.2f} seconds</p>
            <p><strong>Analyzer Version:</strong> {version}</p>
        </div>"""

    def _html_footer(self) -> str:
        """Generate HTML footer."""
        return """
        <div class="footer">
            <p>Generated by Simplus EDA Framework</p>
            <p>&copy; 2024 - Automated Exploratory Data Analysis</p>
        </div>
    </div>
</body>
</html>"""

    def _prepare_json_serializable(self, obj: Any) -> Any:
        """
        Convert analysis results to JSON-serializable format.

        Handles numpy types, pandas objects, and nested structures.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version of the object
        """
        import numpy as np

        if isinstance(obj, dict):
            return {key: self._prepare_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif pd.isna(obj):
            return None
        else:
            return obj
