"""
Time series visualization module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import warnings


class TimeSeriesVisualizer:
    """
    Create visualizations for time series data.

    This class provides methods to visualize time series data including
    trend plots, seasonal decomposition, autocorrelation, and more.

    Attributes:
        default_figsize: Default figure size for plots (width, height)
        default_style: Default matplotlib style
        default_palette: Default seaborn color palette
        default_dpi: Default DPI for saved figures

    Example:
        >>> visualizer = TimeSeriesVisualizer(style='whitegrid')
        >>> fig = visualizer.create_time_plot(df, time_col='date', value_cols=['sales'])
        >>> visualizer.save_plot(fig, 'timeseries.png')
    """

    def __init__(
        self,
        style: str = 'whitegrid',
        palette: str = 'husl',
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 100
    ):
        """
        Initialize the TimeSeriesVisualizer.

        Args:
            style: Seaborn style (whitegrid, darkgrid, white, dark, ticks)
            palette: Seaborn color palette
            figsize: Default figure size (width, height)
            dpi: Default DPI for saved figures
        """
        self.default_figsize = figsize
        self.default_style = style
        self.default_palette = palette
        self.default_dpi = dpi

        sns.set_style(style)
        sns.set_palette(palette)

    def create_time_plot(
        self,
        data: pd.DataFrame,
        time_col: str,
        value_cols: Union[str, List[str]],
        figsize: Optional[Tuple[int, int]] = None,
        show_trend: bool = False,
        show_points: bool = False,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Create time series line plots.

        Args:
            data: Input DataFrame
            time_col: Name of the time column
            value_cols: Column name(s) to plot (string or list of strings)
            figsize: Figure size (width, height), uses default if None
            show_trend: Whether to show trend line
            show_points: Whether to show individual data points
            title: Custom title for the plot

        Returns:
            matplotlib Figure object

        Example:
            >>> fig = visualizer.create_time_plot(df, 'date', ['sales', 'profit'])
        """
        # Validate inputs
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not found in DataFrame")

        # Convert value_cols to list if string
        if isinstance(value_cols, str):
            value_cols = [value_cols]

        # Validate value columns
        for col in value_cols:
            if col not in data.columns:
                raise ValueError(f"Value column '{col}' not found in DataFrame")

        # Create a copy and sort by time
        plot_data = data[[time_col] + value_cols].copy()
        plot_data = plot_data.sort_values(time_col)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize or self.default_figsize)

        # Plot each value column
        for col in value_cols:
            ax.plot(plot_data[time_col], plot_data[col],
                   label=col, linewidth=2, alpha=0.8)

            if show_points:
                ax.scatter(plot_data[time_col], plot_data[col],
                          alpha=0.5, s=20)

            # Add trend line if requested
            if show_trend:
                # Convert datetime to numeric for polyfit
                if pd.api.types.is_datetime64_any_dtype(plot_data[time_col]):
                    x_numeric = (plot_data[time_col] - plot_data[time_col].min()).dt.total_seconds()
                else:
                    x_numeric = pd.to_numeric(plot_data[time_col])

                valid_mask = ~(plot_data[col].isna() | x_numeric.isna())
                if valid_mask.sum() > 1:
                    z = np.polyfit(x_numeric[valid_mask], plot_data[col][valid_mask], 1)
                    p = np.poly1d(z)
                    ax.plot(plot_data[time_col], p(x_numeric),
                           '--', alpha=0.5, label=f'{col} trend')

        ax.set_xlabel(time_col, fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(title or 'Time Series Plot', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()

        return fig

    def create_seasonal_decomposition(
        self,
        data: pd.DataFrame,
        time_col: str,
        value_col: str,
        period: Optional[int] = None,
        model: str = 'additive',
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create seasonal decomposition visualization.

        Args:
            data: Input DataFrame
            time_col: Name of the time column
            value_col: Name of the value column
            period: Period for seasonal decomposition (auto-detected if None)
            model: 'additive' or 'multiplicative'
            figsize: Figure size (width, height), uses default if None

        Returns:
            matplotlib Figure object

        Example:
            >>> fig = visualizer.create_seasonal_decomposition(df, 'date', 'sales', period=12)
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
        except ImportError:
            raise ImportError("statsmodels is required for seasonal decomposition. "
                            "Install it with: pip install statsmodels")

        # Validate inputs
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not found in DataFrame")
        if value_col not in data.columns:
            raise ValueError(f"Value column '{value_col}' not found in DataFrame")

        # Prepare data
        plot_data = data[[time_col, value_col]].copy()
        plot_data = plot_data.sort_values(time_col)
        plot_data = plot_data.set_index(time_col)

        # Remove NaN values
        plot_data = plot_data.dropna()

        if len(plot_data) < 4:
            raise ValueError("Need at least 4 data points for seasonal decomposition")

        # Perform decomposition
        if period is None:
            # Try to infer period from datetime frequency
            if isinstance(plot_data.index, pd.DatetimeIndex):
                freq = pd.infer_freq(plot_data.index)
                if freq:
                    period = {'D': 7, 'W': 52, 'M': 12, 'Q': 4, 'Y': 1}.get(freq[0], None)

        if period is None:
            period = min(len(plot_data) // 2, 12)  # Default to 12 or half the data

        decomposition = seasonal_decompose(
            plot_data[value_col],
            model=model,
            period=period,
            extrapolate_trend='freq'
        )

        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=figsize or (12, 10))

        # Original
        axes[0].plot(decomposition.observed, color='blue', linewidth=1.5)
        axes[0].set_ylabel('Observed', fontsize=10)
        axes[0].set_title(f'Seasonal Decomposition ({model.capitalize()})',
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Trend
        axes[1].plot(decomposition.trend, color='red', linewidth=1.5)
        axes[1].set_ylabel('Trend', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # Seasonal
        axes[2].plot(decomposition.seasonal, color='green', linewidth=1.5)
        axes[2].set_ylabel('Seasonal', fontsize=10)
        axes[2].grid(True, alpha=0.3)

        # Residual
        axes[3].plot(decomposition.resid, color='purple', linewidth=1.5)
        axes[3].set_ylabel('Residual', fontsize=10)
        axes[3].set_xlabel(time_col, fontsize=10)
        axes[3].grid(True, alpha=0.3)

        # Rotate x-axis labels
        for ax in axes:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        return fig

    def create_autocorrelation_plot(
        self,
        data: pd.DataFrame,
        column: str,
        lags: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        alpha: float = 0.05
    ) -> plt.Figure:
        """
        Create autocorrelation (ACF) and partial autocorrelation (PACF) plots.

        Args:
            data: Input DataFrame
            column: Column to analyze
            lags: Number of lags to show (default: min(len(data)//2, 40))
            figsize: Figure size (width, height), uses default if None
            alpha: Significance level for confidence intervals

        Returns:
            matplotlib Figure object

        Example:
            >>> fig = visualizer.create_autocorrelation_plot(df, 'sales', lags=30)
        """
        try:
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        except ImportError:
            raise ImportError("statsmodels is required for autocorrelation plots. "
                            "Install it with: pip install statsmodels")

        # Validate input
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        # Prepare data
        series = data[column].dropna()

        if len(series) < 3:
            raise ValueError("Need at least 3 data points for autocorrelation plot")

        # Default lags
        if lags is None:
            lags = min(len(series) // 2, 40)

        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize or (12, 8))

        # ACF plot
        plot_acf(series, lags=lags, ax=axes[0], alpha=alpha)
        axes[0].set_title('Autocorrelation Function (ACF)',
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Lag', fontsize=10)
        axes[0].set_ylabel('ACF', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # PACF plot
        plot_pacf(series, lags=lags, ax=axes[1], alpha=alpha)
        axes[1].set_title('Partial Autocorrelation Function (PACF)',
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Lag', fontsize=10)
        axes[1].set_ylabel('PACF', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'Autocorrelation Analysis: {column}',
                    fontsize=14, fontweight='bold', y=1.0)
        plt.tight_layout()

        return fig

    def create_rolling_statistics(
        self,
        data: pd.DataFrame,
        time_col: str,
        value_col: str,
        window: int = 7,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create rolling mean and standard deviation plot.

        Args:
            data: Input DataFrame
            time_col: Name of the time column
            value_col: Name of the value column
            window: Rolling window size
            figsize: Figure size (width, height), uses default if None

        Returns:
            matplotlib Figure object

        Example:
            >>> fig = visualizer.create_rolling_statistics(df, 'date', 'sales', window=7)
        """
        # Validate inputs
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not found in DataFrame")
        if value_col not in data.columns:
            raise ValueError(f"Value column '{value_col}' not found in DataFrame")

        # Prepare data
        plot_data = data[[time_col, value_col]].copy()
        plot_data = plot_data.sort_values(time_col)

        # Calculate rolling statistics
        rolling_mean = plot_data[value_col].rolling(window=window, center=True).mean()
        rolling_std = plot_data[value_col].rolling(window=window, center=True).std()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize or self.default_figsize)

        # Plot original data
        ax.plot(plot_data[time_col], plot_data[value_col],
               label='Original', alpha=0.5, linewidth=1)

        # Plot rolling mean
        ax.plot(plot_data[time_col], rolling_mean,
               label=f'Rolling Mean (window={window})',
               color='red', linewidth=2)

        # Plot confidence band (mean ± std)
        ax.fill_between(
            plot_data[time_col],
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.2,
            color='red',
            label=f'Rolling Std (±1σ)'
        )

        ax.set_xlabel(time_col, fontsize=11)
        ax.set_ylabel(value_col, fontsize=11)
        ax.set_title(f'Rolling Statistics: {value_col}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()

        return fig

    def create_lag_plot(
        self,
        data: pd.DataFrame,
        column: str,
        lag: int = 1,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create lag plot to check for autocorrelation.

        Args:
            data: Input DataFrame
            column: Column to analyze
            lag: Lag value
            figsize: Figure size (width, height), uses default if None

        Returns:
            matplotlib Figure object

        Example:
            >>> fig = visualizer.create_lag_plot(df, 'sales', lag=1)
        """
        # Validate input
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        # Prepare data
        series = data[column].dropna()

        if len(series) <= lag:
            raise ValueError(f"Need at least {lag + 1} data points for lag plot with lag={lag}")

        # Create lagged series
        y = series.iloc[lag:]
        x = series.iloc[:-lag]

        # Reset indices to align
        y = y.reset_index(drop=True)
        x = x.reset_index(drop=True)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize or (8, 8))

        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=50)

        # Add diagonal line (perfect correlation)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.5, zorder=0, label='Perfect correlation')

        # Calculate and display correlation
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(f'{column}(t)', fontsize=11)
        ax.set_ylabel(f'{column}(t+{lag})', fontsize=11)
        ax.set_title(f'Lag Plot: {column} (lag={lag})',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        return fig

    def create_time_series_heatmap(
        self,
        data: pd.DataFrame,
        time_col: str,
        value_col: str,
        freq: str = 'D',
        agg_func: str = 'mean',
        figsize: Optional[Tuple[int, int]] = None,
        cmap: str = 'YlOrRd'
    ) -> plt.Figure:
        """
        Create a calendar heatmap for time series data.

        Args:
            data: Input DataFrame
            time_col: Name of the time column (must be datetime)
            value_col: Name of the value column
            freq: Frequency for aggregation ('D', 'W', 'M')
            agg_func: Aggregation function ('mean', 'sum', 'count', 'median')
            figsize: Figure size (width, height), uses default if None
            cmap: Colormap for heatmap

        Returns:
            matplotlib Figure object

        Example:
            >>> fig = visualizer.create_time_series_heatmap(df, 'date', 'sales', freq='D')
        """
        # Validate inputs
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not found in DataFrame")
        if value_col not in data.columns:
            raise ValueError(f"Value column '{value_col}' not found in DataFrame")

        # Prepare data
        plot_data = data[[time_col, value_col]].copy()

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(plot_data[time_col]):
            plot_data[time_col] = pd.to_datetime(plot_data[time_col])

        plot_data = plot_data.sort_values(time_col)

        # Aggregate by frequency
        plot_data = plot_data.set_index(time_col)
        aggregated = plot_data.resample(freq).agg(agg_func)

        # Create pivot for heatmap (year vs week/month/day)
        if freq == 'D':
            aggregated['year'] = aggregated.index.year
            aggregated['week'] = aggregated.index.isocalendar().week
            pivot = aggregated.pivot_table(
                values=value_col,
                index='week',
                columns='year',
                aggfunc=agg_func
            )
            y_label = 'Week of Year'
        elif freq == 'W':
            aggregated['year'] = aggregated.index.year
            aggregated['month'] = aggregated.index.month
            pivot = aggregated.pivot_table(
                values=value_col,
                index='month',
                columns='year',
                aggfunc=agg_func
            )
            y_label = 'Month'
        else:  # Monthly or other
            aggregated['year'] = aggregated.index.year
            aggregated['month'] = aggregated.index.month
            pivot = aggregated.pivot_table(
                values=value_col,
                index='month',
                columns='year',
                aggfunc=agg_func
            )
            y_label = 'Month'

        # Create figure
        fig, ax = plt.subplots(figsize=figsize or (12, 8))

        # Create heatmap
        sns.heatmap(
            pivot,
            cmap=cmap,
            annot=True,
            fmt='.1f',
            linewidths=0.5,
            cbar_kws={'label': value_col},
            ax=ax
        )

        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(f'Time Series Heatmap: {value_col} ({agg_func})',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    @staticmethod
    def save_plot(
        fig: plt.Figure,
        filepath: Union[str, Path],
        dpi: int = 300,
        bbox_inches: str = 'tight',
        **kwargs
    ) -> None:
        """
        Save a matplotlib figure to file.

        Args:
            fig: matplotlib Figure object
            filepath: Path to save the figure
            dpi: DPI for saved figure
            bbox_inches: Bounding box setting
            **kwargs: Additional arguments passed to fig.savefig()

        Example:
            >>> fig = visualizer.create_time_plot(df, 'date', 'sales')
            >>> TimeSeriesVisualizer.save_plot(fig, 'timeseries.png', dpi=300)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, **kwargs)

    @staticmethod
    def close_all() -> None:
        """
        Close all matplotlib figures to free memory.

        Example:
            >>> TimeSeriesVisualizer.close_all()
        """
        plt.close('all')
