"""
Interactive Plotly visualizations for EDA.

This module provides interactive, web-based visualizations using Plotly,
offering a more engaging and exploratory experience compared to static plots.
"""

from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings


class PlotlyVisualizer:
    """
    Create interactive visualizations using Plotly.

    This visualizer generates interactive charts that allow users to:
    - Hover for detailed information
    - Zoom and pan
    - Toggle series visibility
    - Export to PNG/SVG
    - Interact with data points

    All visualizations are returned as Plotly figure objects that can be:
    - Displayed in Jupyter notebooks
    - Saved as HTML
    - Embedded in reports
    - Converted to static images

    Example:
        >>> viz = PlotlyVisualizer()
        >>> fig = viz.create_interactive_histogram(df, 'age')
        >>> fig.show()  # Display in notebook
        >>> fig.write_html('histogram.html')  # Save to file
    """

    def __init__(
        self,
        template: str = 'plotly_white',
        color_palette: Optional[List[str]] = None,
        default_height: int = 500,
        default_width: Optional[int] = None
    ):
        """
        Initialize the Plotly visualizer.

        Args:
            template: Plotly template ('plotly', 'plotly_white', 'plotly_dark', etc.)
            color_palette: Custom color palette for charts
            default_height: Default figure height in pixels
            default_width: Default figure width in pixels (None = responsive)
        """
        self.template = template
        self.color_palette = color_palette or px.colors.qualitative.Plotly
        self.default_height = default_height
        self.default_width = default_width

    def create_interactive_histogram(
        self,
        data: pd.DataFrame,
        column: str,
        bins: int = 30,
        show_kde: bool = True,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive histogram with optional KDE overlay.

        Args:
            data: DataFrame containing the data
            column: Column name to visualize
            bins: Number of bins for histogram
            show_kde: Whether to show KDE curve
            title: Custom title for the plot

        Returns:
            Plotly figure object

        Example:
            >>> fig = viz.create_interactive_histogram(df, 'income', bins=50)
            >>> fig.show()
        """
        fig = go.Figure()

        # Add histogram
        fig.add_trace(go.Histogram(
            x=data[column].dropna(),
            nbinsx=bins,
            name='Distribution',
            marker_color=self.color_palette[0],
            opacity=0.7,
            hovertemplate='<b>Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
        ))

        # Add KDE if requested
        if show_kde:
            from scipy import stats
            values = data[column].dropna()
            kde = stats.gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 100)
            kde_values = kde(x_range)

            # Scale KDE to match histogram
            hist_values, bin_edges = np.histogram(values, bins=bins)
            bin_width = bin_edges[1] - bin_edges[0]
            kde_scaled = kde_values * len(values) * bin_width

            fig.add_trace(go.Scatter(
                x=x_range,
                y=kde_scaled,
                mode='lines',
                name='KDE',
                line=dict(color=self.color_palette[1], width=2),
                hovertemplate='<b>Value:</b> %{x:.2f}<br><b>Density:</b> %{y:.2f}<extra></extra>'
            ))

        fig.update_layout(
            title=title or f'Distribution of {column}',
            xaxis_title=column,
            yaxis_title='Count',
            template=self.template,
            height=self.default_height,
            width=self.default_width,
            hovermode='x unified',
            showlegend=True
        )

        return fig

    def create_interactive_boxplot(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        orientation: str = 'v',
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive box plots for numerical columns.

        Args:
            data: DataFrame containing the data
            columns: List of columns to plot (None = all numeric)
            orientation: 'v' for vertical, 'h' for horizontal
            title: Custom title for the plot

        Returns:
            Plotly figure object

        Example:
            >>> fig = viz.create_interactive_boxplot(df, ['age', 'income', 'score'])
            >>> fig.show()
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        fig = go.Figure()

        for i, col in enumerate(columns[:10]):  # Limit to 10 columns
            if orientation == 'v':
                fig.add_trace(go.Box(
                    y=data[col].dropna(),
                    name=col,
                    marker_color=self.color_palette[i % len(self.color_palette)],
                    boxmean='sd',
                    hovertemplate=(
                        '<b>Column:</b> ' + col + '<br>'
                        '<b>Value:</b> %{y}<br>'
                        '<extra></extra>'
                    )
                ))
            else:
                fig.add_trace(go.Box(
                    x=data[col].dropna(),
                    name=col,
                    marker_color=self.color_palette[i % len(self.color_palette)],
                    boxmean='sd',
                    hovertemplate=(
                        '<b>Column:</b> ' + col + '<br>'
                        '<b>Value:</b> %{x}<br>'
                        '<extra></extra>'
                    )
                ))

        fig.update_layout(
            title=title or 'Box Plots - Numerical Columns',
            template=self.template,
            height=self.default_height,
            width=self.default_width,
            showlegend=True
        )

        return fig

    def create_correlation_heatmap(
        self,
        data: pd.DataFrame,
        method: str = 'pearson',
        columns: Optional[List[str]] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive correlation heatmap.

        Args:
            data: DataFrame containing the data
            method: Correlation method ('pearson', 'spearman', 'kendall')
            columns: Columns to include (None = all numeric)
            title: Custom title for the plot

        Returns:
            Plotly figure object

        Example:
            >>> fig = viz.create_correlation_heatmap(df, method='spearman')
            >>> fig.show()
        """
        if columns is None:
            numeric_data = data.select_dtypes(include=[np.number])
        else:
            numeric_data = data[columns]

        # Calculate correlation
        corr_matrix = numeric_data.corr(method=method)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
            hovertemplate=(
                '<b>%{y}</b> vs <b>%{x}</b><br>'
                'Correlation: %{z:.3f}<extra></extra>'
            )
        ))

        fig.update_layout(
            title=title or f'Correlation Matrix ({method.capitalize()})',
            template=self.template,
            height=max(self.default_height, len(corr_matrix) * 30),
            width=max(self.default_height, len(corr_matrix) * 30),
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'}
        )

        return fig

    def create_scatter_matrix(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        color_by: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive scatter matrix (pair plot).

        Args:
            data: DataFrame containing the data
            columns: Columns to include in the matrix
            color_by: Column to use for color coding
            title: Custom title for the plot

        Returns:
            Plotly figure object

        Example:
            >>> fig = viz.create_scatter_matrix(df, ['age', 'income', 'score'], color_by='category')
            >>> fig.show()
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()[:5]

        # Limit to 5 columns for readability
        columns = columns[:5]

        if color_by and color_by in data.columns:
            fig = px.scatter_matrix(
                data,
                dimensions=columns,
                color=color_by,
                title=title or 'Scatter Matrix',
                template=self.template,
                height=max(800, len(columns) * 200),
                color_discrete_sequence=self.color_palette
            )
        else:
            fig = px.scatter_matrix(
                data,
                dimensions=columns,
                title=title or 'Scatter Matrix',
                template=self.template,
                height=max(800, len(columns) * 200)
            )

        fig.update_traces(diagonal_visible=False, showupperhalf=False)

        return fig

    def create_interactive_scatter(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color_by: Optional[str] = None,
        size_by: Optional[str] = None,
        add_trendline: bool = True,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive scatter plot with optional trendline.

        Args:
            data: DataFrame containing the data
            x: Column for x-axis
            y: Column for y-axis
            color_by: Column to use for color coding
            size_by: Column to use for point sizing
            add_trendline: Whether to add a trendline
            title: Custom title for the plot

        Returns:
            Plotly figure object

        Example:
            >>> fig = viz.create_interactive_scatter(df, 'age', 'income', color_by='category')
            >>> fig.show()
        """
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color_by,
            size=size_by,
            trendline='ols' if add_trendline else None,
            title=title or f'{y} vs {x}',
            template=self.template,
            height=self.default_height,
            color_discrete_sequence=self.color_palette
        )

        fig.update_traces(
            marker=dict(line=dict(width=0.5, color='DarkSlateGrey')),
            hovertemplate=(
                f'<b>{x}:</b> %{{x}}<br>'
                f'<b>{y}:</b> %{{y}}<br>'
                '<extra></extra>'
            )
        )

        return fig

    def create_time_series_plot(
        self,
        data: pd.DataFrame,
        time_col: str,
        value_cols: Union[str, List[str]],
        show_range_slider: bool = True,
        show_buttons: bool = True,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive time series plot with range slider.

        Args:
            data: DataFrame containing the data
            time_col: Column containing datetime values
            value_cols: Column(s) to plot
            show_range_slider: Show range slider for zooming
            show_buttons: Show time period buttons (1M, 6M, YTD, etc.)
            title: Custom title for the plot

        Returns:
            Plotly figure object

        Example:
            >>> fig = viz.create_time_series_plot(df, 'date', ['sales', 'profit'])
            >>> fig.show()
        """
        if isinstance(value_cols, str):
            value_cols = [value_cols]

        fig = go.Figure()

        for i, col in enumerate(value_cols):
            fig.add_trace(go.Scatter(
                x=data[time_col],
                y=data[col],
                mode='lines',
                name=col,
                line=dict(color=self.color_palette[i % len(self.color_palette)]),
                hovertemplate=(
                    '<b>Date:</b> %{x}<br>'
                    f'<b>{col}:</b> %{{y:.2f}}<br>'
                    '<extra></extra>'
                )
            ))

        # Add range slider
        if show_range_slider:
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ) if show_buttons else None
            )

        fig.update_layout(
            title=title or 'Time Series Analysis',
            xaxis_title=time_col,
            yaxis_title='Value',
            template=self.template,
            height=self.default_height,
            width=self.default_width,
            hovermode='x unified'
        )

        return fig

    def create_distribution_comparison(
        self,
        data: pd.DataFrame,
        column: str,
        group_by: str,
        plot_type: str = 'violin',
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive violin or box plots grouped by category.

        Args:
            data: DataFrame containing the data
            column: Numerical column to plot
            group_by: Categorical column to group by
            plot_type: 'violin' or 'box'
            title: Custom title for the plot

        Returns:
            Plotly figure object

        Example:
            >>> fig = viz.create_distribution_comparison(df, 'income', 'region', 'violin')
            >>> fig.show()
        """
        fig = go.Figure()

        groups = data[group_by].unique()

        for i, group in enumerate(groups):
            group_data = data[data[group_by] == group][column].dropna()

            if plot_type == 'violin':
                fig.add_trace(go.Violin(
                    y=group_data,
                    name=str(group),
                    marker_color=self.color_palette[i % len(self.color_palette)],
                    box_visible=True,
                    meanline_visible=True,
                    hovertemplate=(
                        f'<b>Group:</b> {group}<br>'
                        '<b>Value:</b> %{y}<br>'
                        '<extra></extra>'
                    )
                ))
            else:  # box
                fig.add_trace(go.Box(
                    y=group_data,
                    name=str(group),
                    marker_color=self.color_palette[i % len(self.color_palette)],
                    boxmean='sd',
                    hovertemplate=(
                        f'<b>Group:</b> {group}<br>'
                        '<b>Value:</b> %{y}<br>'
                        '<extra></extra>'
                    )
                ))

        fig.update_layout(
            title=title or f'{column} Distribution by {group_by}',
            yaxis_title=column,
            xaxis_title=group_by,
            template=self.template,
            height=self.default_height,
            width=self.default_width,
            showlegend=True
        )

        return fig

    def create_bar_chart(
        self,
        data: pd.DataFrame,
        x: str,
        y: Optional[str] = None,
        orientation: str = 'v',
        sort_by: Optional[str] = None,
        top_n: Optional[int] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive bar chart.

        Args:
            data: DataFrame containing the data
            x: Column for x-axis (categories)
            y: Column for y-axis (values). If None, counts are used
            orientation: 'v' for vertical, 'h' for horizontal
            sort_by: Column to sort by
            top_n: Show only top N categories
            title: Custom title for the plot

        Returns:
            Plotly figure object

        Example:
            >>> fig = viz.create_bar_chart(df, 'category', 'sales', top_n=10)
            >>> fig.show()
        """
        if y is None:
            # Count occurrences
            plot_data = data[x].value_counts().reset_index()
            plot_data.columns = [x, 'count']
            y = 'count'
        else:
            plot_data = data[[x, y]].copy()

        # Sort if requested
        if sort_by:
            plot_data = plot_data.sort_values(sort_by, ascending=False)

        # Limit to top N
        if top_n:
            plot_data = plot_data.head(top_n)

        if orientation == 'h':
            fig = px.bar(
                plot_data,
                x=y,
                y=x,
                orientation='h',
                title=title or f'{y} by {x}',
                template=self.template,
                height=max(self.default_height, len(plot_data) * 30),
                color=y,
                color_continuous_scale='Blues'
            )
        else:
            fig = px.bar(
                plot_data,
                x=x,
                y=y,
                title=title or f'{y} by {x}',
                template=self.template,
                height=self.default_height,
                color=y,
                color_continuous_scale='Blues'
            )

        fig.update_traces(
            hovertemplate=(
                f'<b>{x}:</b> %{{{"y" if orientation == "h" else "x"}}}<br>'
                f'<b>{y}:</b> %{{{"x" if orientation == "h" else "y"}}}<br>'
                '<extra></extra>'
            )
        )

        return fig

    def create_sunburst(
        self,
        data: pd.DataFrame,
        path: List[str],
        values: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive sunburst chart for hierarchical data.

        Args:
            data: DataFrame containing the data
            path: List of columns representing the hierarchy
            values: Column for sizing (None = count)
            title: Custom title for the plot

        Returns:
            Plotly figure object

        Example:
            >>> fig = viz.create_sunburst(df, ['country', 'city', 'district'], 'population')
            >>> fig.show()
        """
        fig = px.sunburst(
            data,
            path=path,
            values=values,
            title=title or 'Hierarchical Distribution',
            template=self.template,
            height=600,
            color_discrete_sequence=self.color_palette
        )

        fig.update_traces(
            textinfo='label+percent parent',
            hovertemplate=(
                '<b>%{label}</b><br>'
                'Value: %{value}<br>'
                'Percent: %{percentParent}<br>'
                '<extra></extra>'
            )
        )

        return fig

    def create_3d_scatter(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        z: str,
        color_by: Optional[str] = None,
        size_by: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive 3D scatter plot.

        Args:
            data: DataFrame containing the data
            x: Column for x-axis
            y: Column for y-axis
            z: Column for z-axis
            color_by: Column to use for color coding
            size_by: Column to use for point sizing
            title: Custom title for the plot

        Returns:
            Plotly figure object

        Example:
            >>> fig = viz.create_3d_scatter(df, 'x', 'y', 'z', color_by='category')
            >>> fig.show()
        """
        fig = px.scatter_3d(
            data,
            x=x,
            y=y,
            z=z,
            color=color_by,
            size=size_by,
            title=title or f'3D Scatter: {x}, {y}, {z}',
            template=self.template,
            height=700,
            color_discrete_sequence=self.color_palette
        )

        fig.update_traces(
            marker=dict(line=dict(width=0.5, color='DarkSlateGrey'))
        )

        return fig

    def create_dashboard(
        self,
        figures: List[go.Figure],
        titles: Optional[List[str]] = None,
        rows: Optional[int] = None,
        cols: Optional[int] = None
    ) -> go.Figure:
        """
        Combine multiple figures into a dashboard layout.

        Args:
            figures: List of Plotly figures to combine
            titles: List of titles for each subplot
            rows: Number of rows (auto-calculated if None)
            cols: Number of columns (defaults to 2)

        Returns:
            Combined Plotly figure

        Example:
            >>> fig1 = viz.create_interactive_histogram(df, 'age')
            >>> fig2 = viz.create_interactive_boxplot(df, ['income'])
            >>> dashboard = viz.create_dashboard([fig1, fig2])
            >>> dashboard.show()
        """
        n_figures = len(figures)

        if cols is None:
            cols = 2

        if rows is None:
            rows = (n_figures + cols - 1) // cols

        if titles is None:
            titles = [f'Chart {i+1}' for i in range(n_figures)]

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=titles[:n_figures]
        )

        for i, figure in enumerate(figures):
            row = i // cols + 1
            col = i % cols + 1

            for trace in figure.data:
                fig.add_trace(trace, row=row, col=col)

        fig.update_layout(
            height=rows * 400,
            template=self.template,
            showlegend=False
        )

        return fig

    def save_figure(
        self,
        fig: go.Figure,
        path: str,
        format: str = 'html',
        **kwargs
    ) -> None:
        """
        Save a Plotly figure to file.

        Args:
            fig: Plotly figure to save
            path: Output file path
            format: Output format ('html', 'png', 'svg', 'pdf', 'json')
            **kwargs: Additional arguments for the save method

        Example:
            >>> fig = viz.create_interactive_histogram(df, 'age')
            >>> viz.save_figure(fig, 'histogram.html')
            >>> viz.save_figure(fig, 'histogram.png', format='png', width=800, height=600)
        """
        if format == 'html':
            fig.write_html(path, **kwargs)
        elif format == 'png':
            fig.write_image(path, format='png', **kwargs)
        elif format == 'svg':
            fig.write_image(path, format='svg', **kwargs)
        elif format == 'pdf':
            fig.write_image(path, format='pdf', **kwargs)
        elif format == 'json':
            fig.write_json(path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
