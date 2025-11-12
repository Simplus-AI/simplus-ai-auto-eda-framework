"""
Tests for the TimeSeriesVisualizer class.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from simplus_eda.visualizers.timeseries import TimeSeriesVisualizer


@pytest.fixture
def time_series_data():
    """Create sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n = len(dates)

    # Create time series with trend and seasonality
    trend = np.linspace(100, 150, n)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 5, n)
    values = trend + seasonal + noise

    return pd.DataFrame({
        'date': dates,
        'sales': values,
        'profit': values * 0.2 + np.random.normal(0, 2, n)
    })


@pytest.fixture
def numeric_time_series():
    """Create time series with numeric time column."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'time': np.arange(n),
        'value': np.random.randn(n).cumsum() + 50
    })


@pytest.fixture
def visualizer():
    """Create TimeSeriesVisualizer instance."""
    return TimeSeriesVisualizer()


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Cleanup plots after each test."""
    yield
    plt.close('all')


# Initialization Tests

def test_init_default():
    """Test initialization with default parameters."""
    vis = TimeSeriesVisualizer()
    assert vis.default_figsize == (12, 6)
    assert vis.default_style == 'whitegrid'
    assert vis.default_palette == 'husl'
    assert vis.default_dpi == 100


def test_init_custom():
    """Test initialization with custom parameters."""
    vis = TimeSeriesVisualizer(
        style='darkgrid',
        palette='Set2',
        figsize=(15, 8),
        dpi=150
    )
    assert vis.default_figsize == (15, 8)
    assert vis.default_style == 'darkgrid'
    assert vis.default_palette == 'Set2'
    assert vis.default_dpi == 150


# Time Plot Tests

def test_create_time_plot_basic(visualizer, time_series_data):
    """Test basic time plot creation."""
    fig = visualizer.create_time_plot(time_series_data, 'date', 'sales')
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1


def test_create_time_plot_multiple_columns(visualizer, time_series_data):
    """Test time plot with multiple value columns."""
    fig = visualizer.create_time_plot(
        time_series_data,
        'date',
        ['sales', 'profit']
    )
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    # Should have multiple lines
    assert len(ax.lines) >= 2


def test_create_time_plot_string_column(visualizer, time_series_data):
    """Test time plot with string value column."""
    fig = visualizer.create_time_plot(time_series_data, 'date', 'sales')
    assert isinstance(fig, plt.Figure)


def test_create_time_plot_with_trend(visualizer, time_series_data):
    """Test time plot with trend line."""
    fig = visualizer.create_time_plot(
        time_series_data,
        'date',
        'sales',
        show_trend=True
    )
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    # Should have original line + trend line
    assert len(ax.lines) >= 2


def test_create_time_plot_with_points(visualizer, time_series_data):
    """Test time plot with data points."""
    fig = visualizer.create_time_plot(
        time_series_data,
        'date',
        'sales',
        show_points=True
    )
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    # Should have scatter plot collections
    assert len(ax.collections) > 0


def test_create_time_plot_custom_title(visualizer, time_series_data):
    """Test time plot with custom title."""
    custom_title = "My Custom Time Series"
    fig = visualizer.create_time_plot(
        time_series_data,
        'date',
        'sales',
        title=custom_title
    )
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert custom_title in ax.get_title()


def test_create_time_plot_custom_figsize(visualizer, time_series_data):
    """Test time plot with custom figure size."""
    fig = visualizer.create_time_plot(
        time_series_data,
        'date',
        'sales',
        figsize=(15, 10)
    )
    assert isinstance(fig, plt.Figure)
    assert fig.get_figwidth() == 15
    assert fig.get_figheight() == 10


def test_create_time_plot_time_col_not_found(visualizer, time_series_data):
    """Test error when time column not found."""
    with pytest.raises(ValueError, match="not found in DataFrame"):
        visualizer.create_time_plot(time_series_data, 'nonexistent', 'sales')


def test_create_time_plot_value_col_not_found(visualizer, time_series_data):
    """Test error when value column not found."""
    with pytest.raises(ValueError, match="not found in DataFrame"):
        visualizer.create_time_plot(time_series_data, 'date', 'nonexistent')


def test_create_time_plot_numeric_time(visualizer, numeric_time_series):
    """Test time plot with numeric time column."""
    fig = visualizer.create_time_plot(
        numeric_time_series,
        'time',
        'value',
        show_trend=True
    )
    assert isinstance(fig, plt.Figure)


# Seasonal Decomposition Tests

def test_create_seasonal_decomposition_basic(visualizer, time_series_data):
    """Test basic seasonal decomposition."""
    fig = visualizer.create_seasonal_decomposition(
        time_series_data,
        'date',
        'sales',
        period=7
    )
    assert isinstance(fig, plt.Figure)
    # Should have 4 subplots (observed, trend, seasonal, residual)
    assert len(fig.axes) == 4


def test_create_seasonal_decomposition_additive(visualizer, time_series_data):
    """Test seasonal decomposition with additive model."""
    fig = visualizer.create_seasonal_decomposition(
        time_series_data,
        'date',
        'sales',
        period=7,
        model='additive'
    )
    assert isinstance(fig, plt.Figure)


def test_create_seasonal_decomposition_multiplicative(visualizer, time_series_data):
    """Test seasonal decomposition with multiplicative model."""
    # Use positive values for multiplicative model
    data = time_series_data.copy()
    data['sales'] = data['sales'] + 100  # Make all values positive

    fig = visualizer.create_seasonal_decomposition(
        data,
        'date',
        'sales',
        period=7,
        model='multiplicative'
    )
    assert isinstance(fig, plt.Figure)


def test_create_seasonal_decomposition_auto_period(visualizer, time_series_data):
    """Test seasonal decomposition with auto-detected period."""
    fig = visualizer.create_seasonal_decomposition(
        time_series_data,
        'date',
        'sales'
    )
    assert isinstance(fig, plt.Figure)


def test_create_seasonal_decomposition_insufficient_data(visualizer):
    """Test error with insufficient data points."""
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=3),
        'value': [1, 2, 3]
    })
    with pytest.raises(ValueError, match="at least 4 data points"):
        visualizer.create_seasonal_decomposition(data, 'date', 'value', period=2)


def test_create_seasonal_decomposition_column_not_found(visualizer, time_series_data):
    """Test error when column not found."""
    with pytest.raises(ValueError, match="not found in DataFrame"):
        visualizer.create_seasonal_decomposition(
            time_series_data,
            'date',
            'nonexistent'
        )


# Autocorrelation Plot Tests

def test_create_autocorrelation_plot_basic(visualizer, time_series_data):
    """Test basic autocorrelation plot."""
    fig = visualizer.create_autocorrelation_plot(time_series_data, 'sales')
    assert isinstance(fig, plt.Figure)
    # Should have 2 subplots (ACF and PACF)
    assert len(fig.axes) == 2


def test_create_autocorrelation_plot_custom_lags(visualizer, time_series_data):
    """Test autocorrelation plot with custom lags."""
    fig = visualizer.create_autocorrelation_plot(
        time_series_data,
        'sales',
        lags=20
    )
    assert isinstance(fig, plt.Figure)


def test_create_autocorrelation_plot_custom_alpha(visualizer, time_series_data):
    """Test autocorrelation plot with custom significance level."""
    fig = visualizer.create_autocorrelation_plot(
        time_series_data,
        'sales',
        alpha=0.01
    )
    assert isinstance(fig, plt.Figure)


def test_create_autocorrelation_plot_column_not_found(visualizer, time_series_data):
    """Test error when column not found."""
    with pytest.raises(ValueError, match="not found in DataFrame"):
        visualizer.create_autocorrelation_plot(time_series_data, 'nonexistent')


def test_create_autocorrelation_plot_insufficient_data(visualizer):
    """Test error with insufficient data."""
    data = pd.DataFrame({'value': [1, 2]})
    with pytest.raises(ValueError, match="at least 3 data points"):
        visualizer.create_autocorrelation_plot(data, 'value')


# Rolling Statistics Tests

def test_create_rolling_statistics_basic(visualizer, time_series_data):
    """Test basic rolling statistics plot."""
    fig = visualizer.create_rolling_statistics(
        time_series_data,
        'date',
        'sales',
        window=7
    )
    assert isinstance(fig, plt.Figure)


def test_create_rolling_statistics_custom_window(visualizer, time_series_data):
    """Test rolling statistics with custom window."""
    fig = visualizer.create_rolling_statistics(
        time_series_data,
        'date',
        'sales',
        window=14
    )
    assert isinstance(fig, plt.Figure)


def test_create_rolling_statistics_column_not_found(visualizer, time_series_data):
    """Test error when column not found."""
    with pytest.raises(ValueError, match="not found in DataFrame"):
        visualizer.create_rolling_statistics(
            time_series_data,
            'date',
            'nonexistent'
        )


# Lag Plot Tests

def test_create_lag_plot_basic(visualizer, time_series_data):
    """Test basic lag plot."""
    fig = visualizer.create_lag_plot(time_series_data, 'sales', lag=1)
    assert isinstance(fig, plt.Figure)


def test_create_lag_plot_custom_lag(visualizer, time_series_data):
    """Test lag plot with custom lag."""
    fig = visualizer.create_lag_plot(time_series_data, 'sales', lag=7)
    assert isinstance(fig, plt.Figure)


def test_create_lag_plot_column_not_found(visualizer, time_series_data):
    """Test error when column not found."""
    with pytest.raises(ValueError, match="not found in DataFrame"):
        visualizer.create_lag_plot(time_series_data, 'nonexistent')


def test_create_lag_plot_insufficient_data(visualizer):
    """Test error with insufficient data."""
    data = pd.DataFrame({'value': [1, 2, 3]})
    with pytest.raises(ValueError, match="at least"):
        visualizer.create_lag_plot(data, 'value', lag=5)


# Time Series Heatmap Tests

def test_create_time_series_heatmap_daily(visualizer, time_series_data):
    """Test time series heatmap with daily frequency."""
    fig = visualizer.create_time_series_heatmap(
        time_series_data,
        'date',
        'sales',
        freq='D'
    )
    assert isinstance(fig, plt.Figure)


def test_create_time_series_heatmap_weekly(visualizer, time_series_data):
    """Test time series heatmap with weekly frequency."""
    fig = visualizer.create_time_series_heatmap(
        time_series_data,
        'date',
        'sales',
        freq='W'
    )
    assert isinstance(fig, plt.Figure)


def test_create_time_series_heatmap_monthly(visualizer, time_series_data):
    """Test time series heatmap with monthly frequency."""
    fig = visualizer.create_time_series_heatmap(
        time_series_data,
        'date',
        'sales',
        freq='M'
    )
    assert isinstance(fig, plt.Figure)


def test_create_time_series_heatmap_custom_agg(visualizer, time_series_data):
    """Test time series heatmap with custom aggregation."""
    fig = visualizer.create_time_series_heatmap(
        time_series_data,
        'date',
        'sales',
        freq='W',
        agg_func='sum'
    )
    assert isinstance(fig, plt.Figure)


def test_create_time_series_heatmap_custom_cmap(visualizer, time_series_data):
    """Test time series heatmap with custom colormap."""
    fig = visualizer.create_time_series_heatmap(
        time_series_data,
        'date',
        'sales',
        freq='M',
        cmap='Blues'
    )
    assert isinstance(fig, plt.Figure)


def test_create_time_series_heatmap_column_not_found(visualizer, time_series_data):
    """Test error when column not found."""
    with pytest.raises(ValueError, match="not found in DataFrame"):
        visualizer.create_time_series_heatmap(
            time_series_data,
            'date',
            'nonexistent'
        )


# Utility Tests

def test_save_plot(visualizer, time_series_data, tmp_path):
    """Test saving plot to file."""
    fig = visualizer.create_time_plot(time_series_data, 'date', 'sales')
    filepath = tmp_path / "test_plot.png"

    TimeSeriesVisualizer.save_plot(fig, filepath)

    assert filepath.exists()


def test_save_plot_creates_directory(visualizer, time_series_data, tmp_path):
    """Test that save_plot creates parent directories."""
    fig = visualizer.create_time_plot(time_series_data, 'date', 'sales')
    filepath = tmp_path / "subdir" / "test_plot.png"

    TimeSeriesVisualizer.save_plot(fig, filepath)

    assert filepath.exists()


def test_save_plot_custom_dpi(visualizer, time_series_data, tmp_path):
    """Test saving plot with custom DPI."""
    fig = visualizer.create_time_plot(time_series_data, 'date', 'sales')
    filepath = tmp_path / "test_plot.png"

    TimeSeriesVisualizer.save_plot(fig, filepath, dpi=150)

    assert filepath.exists()


def test_close_all(visualizer, time_series_data):
    """Test closing all figures."""
    visualizer.create_time_plot(time_series_data, 'date', 'sales')
    visualizer.create_rolling_statistics(time_series_data, 'date', 'sales')

    # Should have open figures
    assert len(plt.get_fignums()) > 0

    TimeSeriesVisualizer.close_all()

    # All figures should be closed
    assert len(plt.get_fignums()) == 0


# Edge Case Tests

def test_time_plot_with_nans(visualizer):
    """Test time plot with NaN values."""
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=10),
        'value': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
    })
    fig = visualizer.create_time_plot(data, 'date', 'value')
    assert isinstance(fig, plt.Figure)


def test_very_short_time_series(visualizer):
    """Test with very short time series."""
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=5),
        'value': [1, 2, 3, 4, 5]
    })
    fig = visualizer.create_time_plot(data, 'date', 'value')
    assert isinstance(fig, plt.Figure)


def test_long_time_series(visualizer):
    """Test with long time series."""
    np.random.seed(42)
    data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=1000),
        'value': np.random.randn(1000).cumsum()
    })
    fig = visualizer.create_time_plot(data, 'date', 'value')
    assert isinstance(fig, plt.Figure)


def test_constant_values(visualizer):
    """Test with constant values."""
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100),
        'value': [5.0] * 100
    })
    fig = visualizer.create_time_plot(data, 'date', 'value')
    assert isinstance(fig, plt.Figure)


# Integration Tests

def test_multiple_visualizations_same_data(visualizer, time_series_data):
    """Test creating multiple visualization types on same data."""
    fig1 = visualizer.create_time_plot(time_series_data, 'date', 'sales')
    fig2 = visualizer.create_rolling_statistics(time_series_data, 'date', 'sales')
    fig3 = visualizer.create_lag_plot(time_series_data, 'sales', lag=1)

    assert all(isinstance(fig, plt.Figure) for fig in [fig1, fig2, fig3])


def test_workflow_with_save(visualizer, time_series_data, tmp_path):
    """Test complete workflow: create, save, cleanup."""
    # Create visualization
    fig = visualizer.create_time_plot(time_series_data, 'date', 'sales')
    assert isinstance(fig, plt.Figure)

    # Save to file
    filepath = tmp_path / "timeseries.png"
    TimeSeriesVisualizer.save_plot(fig, filepath)
    assert filepath.exists()

    # Cleanup
    TimeSeriesVisualizer.close_all()
    assert len(plt.get_fignums()) == 0


def test_different_styles(time_series_data):
    """Test different visualization styles."""
    styles = ['whitegrid', 'darkgrid', 'white', 'dark', 'ticks']

    for style in styles:
        vis = TimeSeriesVisualizer(style=style)
        fig = vis.create_time_plot(time_series_data, 'date', 'sales')
        assert isinstance(fig, plt.Figure)
        plt.close('all')


def test_comprehensive_time_series_analysis(visualizer, time_series_data):
    """Test comprehensive time series analysis workflow."""
    # Create time plot
    fig1 = visualizer.create_time_plot(
        time_series_data,
        'date',
        ['sales', 'profit'],
        show_trend=True
    )
    assert isinstance(fig1, plt.Figure)

    # Create rolling statistics
    fig2 = visualizer.create_rolling_statistics(
        time_series_data,
        'date',
        'sales',
        window=7
    )
    assert isinstance(fig2, plt.Figure)

    # Create seasonal decomposition
    fig3 = visualizer.create_seasonal_decomposition(
        time_series_data,
        'date',
        'sales',
        period=7
    )
    assert isinstance(fig3, plt.Figure)

    # Create autocorrelation plot
    fig4 = visualizer.create_autocorrelation_plot(time_series_data, 'sales')
    assert isinstance(fig4, plt.Figure)

    # All should be successful
    assert all(isinstance(fig, plt.Figure) for fig in [fig1, fig2, fig3, fig4])


def test_unsorted_time_series(visualizer):
    """Test with unsorted time series data."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=50)
    shuffled_dates = dates.to_series().sample(frac=1, random_state=42)

    data = pd.DataFrame({
        'date': shuffled_dates,
        'value': np.random.randn(50).cumsum()
    })

    # Should sort automatically
    fig = visualizer.create_time_plot(data, 'date', 'value')
    assert isinstance(fig, plt.Figure)
