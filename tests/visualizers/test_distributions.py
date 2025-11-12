"""
Tests for the DistributionVisualizer class.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from simplus_eda.visualizers.distributions import DistributionVisualizer


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'normal': np.random.normal(100, 15, 1000),
        'uniform': np.random.uniform(0, 100, 1000),
        'exponential': np.random.exponential(2, 1000),
        'bimodal': np.concatenate([
            np.random.normal(30, 5, 500),
            np.random.normal(70, 5, 500)
        ]),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })


@pytest.fixture
def visualizer():
    """Create DistributionVisualizer instance."""
    return DistributionVisualizer()


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Cleanup plots after each test."""
    yield
    plt.close('all')


# Initialization Tests

def test_init_default():
    """Test initialization with default parameters."""
    vis = DistributionVisualizer()
    assert vis.default_figsize == (12, 8)
    assert vis.style == 'whitegrid'
    assert vis.palette == 'husl'
    assert vis.dpi == 100


def test_init_custom():
    """Test initialization with custom parameters."""
    vis = DistributionVisualizer(
        style='darkgrid',
        palette='Set2',
        figsize=(15, 10),
        dpi=150
    )
    assert vis.default_figsize == (15, 10)
    assert vis.style == 'darkgrid'
    assert vis.palette == 'Set2'
    assert vis.dpi == 150


# Histogram Tests

def test_create_histograms_basic(visualizer, sample_data):
    """Test basic histogram creation."""
    fig = visualizer.create_histograms(sample_data)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0


def test_create_histograms_single_column(visualizer, sample_data):
    """Test histogram for single column."""
    fig = visualizer.create_histograms(sample_data, columns=['normal'])
    assert isinstance(fig, plt.Figure)
    # May have 2 axes if KDE uses secondary axis
    assert len(fig.axes) >= 1


def test_create_histograms_multiple_columns(visualizer, sample_data):
    """Test histograms for multiple columns."""
    fig = visualizer.create_histograms(
        sample_data,
        columns=['normal', 'uniform', 'exponential']
    )
    assert isinstance(fig, plt.Figure)
    # May have more axes due to KDE secondary axes
    assert len(fig.axes) >= 3


def test_create_histograms_with_kde(visualizer, sample_data):
    """Test histograms with KDE overlay."""
    fig = visualizer.create_histograms(sample_data, kde=True)
    assert isinstance(fig, plt.Figure)


def test_create_histograms_without_kde(visualizer, sample_data):
    """Test histograms without KDE overlay."""
    fig = visualizer.create_histograms(sample_data, kde=False)
    assert isinstance(fig, plt.Figure)


def test_create_histograms_custom_bins(visualizer, sample_data):
    """Test histograms with custom bins."""
    fig = visualizer.create_histograms(sample_data, bins=30)
    assert isinstance(fig, plt.Figure)


def test_create_histograms_custom_figsize(visualizer, sample_data):
    """Test histograms with custom figure size."""
    fig = visualizer.create_histograms(sample_data, figsize=(20, 15))
    assert isinstance(fig, plt.Figure)
    assert fig.get_figwidth() == 20
    assert fig.get_figheight() == 15


def test_create_histograms_no_numeric_columns(visualizer):
    """Test error when no numeric columns available."""
    data = pd.DataFrame({'category': ['A', 'B', 'C']})
    with pytest.raises(ValueError, match="No numeric columns found"):
        visualizer.create_histograms(data)


# Boxplot Tests

def test_create_boxplots_basic(visualizer, sample_data):
    """Test basic boxplot creation."""
    fig = visualizer.create_boxplots(sample_data)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0


def test_create_boxplots_single_column(visualizer, sample_data):
    """Test boxplot for single column."""
    fig = visualizer.create_boxplots(sample_data, columns=['normal'])
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1


def test_create_boxplots_vertical(visualizer, sample_data):
    """Test vertical boxplots."""
    fig = visualizer.create_boxplots(sample_data, orient='v')
    assert isinstance(fig, plt.Figure)


def test_create_boxplots_horizontal(visualizer, sample_data):
    """Test horizontal boxplots."""
    fig = visualizer.create_boxplots(sample_data, orient='h')
    assert isinstance(fig, plt.Figure)


def test_create_boxplots_with_outliers(visualizer, sample_data):
    """Test boxplots with outliers shown."""
    fig = visualizer.create_boxplots(sample_data, showfliers=True)
    assert isinstance(fig, plt.Figure)


def test_create_boxplots_without_outliers(visualizer, sample_data):
    """Test boxplots without outliers."""
    fig = visualizer.create_boxplots(sample_data, showfliers=False)
    assert isinstance(fig, plt.Figure)


def test_create_boxplots_no_numeric_columns(visualizer):
    """Test error when no numeric columns available."""
    data = pd.DataFrame({'category': ['A', 'B', 'C']})
    with pytest.raises(ValueError, match="No numeric columns found"):
        visualizer.create_boxplots(data)


# Density Plot Tests

def test_create_density_plots_basic(visualizer, sample_data):
    """Test basic density plot creation."""
    fig = visualizer.create_density_plots(sample_data)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0


def test_create_density_plots_single_column(visualizer, sample_data):
    """Test density plot for single column."""
    fig = visualizer.create_density_plots(sample_data, columns=['normal'])
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1


def test_create_density_plots_with_fill(visualizer, sample_data):
    """Test density plots with fill."""
    fig = visualizer.create_density_plots(sample_data, fill=True)
    assert isinstance(fig, plt.Figure)


def test_create_density_plots_without_fill(visualizer, sample_data):
    """Test density plots without fill."""
    fig = visualizer.create_density_plots(sample_data, fill=False)
    assert isinstance(fig, plt.Figure)


def test_create_density_plots_no_numeric_columns(visualizer):
    """Test error when no numeric columns available."""
    data = pd.DataFrame({'category': ['A', 'B', 'C']})
    with pytest.raises(ValueError, match="No numeric columns found"):
        visualizer.create_density_plots(data)


# Violin Plot Tests

def test_create_violin_plots_basic(visualizer, sample_data):
    """Test basic violin plot creation."""
    fig = visualizer.create_violin_plots(sample_data)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0


def test_create_violin_plots_single_column(visualizer, sample_data):
    """Test violin plot for single column."""
    fig = visualizer.create_violin_plots(sample_data, columns=['normal'])
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1


def test_create_violin_plots_no_numeric_columns(visualizer):
    """Test error when no numeric columns available."""
    data = pd.DataFrame({'category': ['A', 'B', 'C']})
    with pytest.raises(ValueError, match="No numeric columns found"):
        visualizer.create_violin_plots(data)


# Q-Q Plot Tests

def test_create_qq_plots_basic(visualizer, sample_data):
    """Test basic Q-Q plot creation."""
    fig = visualizer.create_qq_plots(sample_data)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0


def test_create_qq_plots_single_column(visualizer, sample_data):
    """Test Q-Q plot for single column."""
    fig = visualizer.create_qq_plots(sample_data, columns=['normal'])
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1


def test_create_qq_plots_no_numeric_columns(visualizer):
    """Test error when no numeric columns available."""
    data = pd.DataFrame({'category': ['A', 'B', 'C']})
    with pytest.raises(ValueError, match="No numeric columns found"):
        visualizer.create_qq_plots(data)


# Distribution Summary Tests

def test_create_distribution_summary_basic(visualizer, sample_data):
    """Test basic distribution summary creation."""
    fig = visualizer.create_distribution_summary(sample_data, column='normal')
    assert isinstance(fig, plt.Figure)
    # Should have 6 subplots (histogram, boxplot, violin, Q-Q, stats table, description)
    assert len(fig.axes) == 6


def test_create_distribution_summary_column_not_found(visualizer, sample_data):
    """Test error when column not found."""
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        visualizer.create_distribution_summary(sample_data, column='nonexistent')


def test_create_distribution_summary_non_numeric_column(visualizer, sample_data):
    """Test error for non-numeric column."""
    with pytest.raises((ValueError, TypeError)):
        visualizer.create_distribution_summary(sample_data, column='category')


def test_create_distribution_summary_custom_figsize(visualizer, sample_data):
    """Test distribution summary with custom figure size."""
    fig = visualizer.create_distribution_summary(
        sample_data,
        column='normal',
        figsize=(20, 15)
    )
    assert isinstance(fig, plt.Figure)
    assert fig.get_figwidth() == 20
    assert fig.get_figheight() == 15


# Utility Tests

def test_save_plot(visualizer, sample_data, tmp_path):
    """Test saving plot to file."""
    fig = visualizer.create_histograms(sample_data, columns=['normal'])
    filepath = tmp_path / "test_plot.png"

    DistributionVisualizer.save_plot(fig, filepath)

    assert filepath.exists()


def test_save_plot_creates_directory(visualizer, sample_data, tmp_path):
    """Test that save_plot creates parent directories."""
    fig = visualizer.create_histograms(sample_data, columns=['normal'])
    filepath = tmp_path / "subdir" / "test_plot.png"

    DistributionVisualizer.save_plot(fig, filepath)

    assert filepath.exists()


def test_save_plot_custom_dpi(visualizer, sample_data, tmp_path):
    """Test saving plot with custom DPI."""
    fig = visualizer.create_histograms(sample_data, columns=['normal'])
    filepath = tmp_path / "test_plot.png"

    DistributionVisualizer.save_plot(fig, filepath, dpi=150)

    assert filepath.exists()


def test_close_all(visualizer, sample_data):
    """Test closing all figures."""
    visualizer.create_histograms(sample_data)
    visualizer.create_boxplots(sample_data)

    # Should have open figures
    assert len(plt.get_fignums()) > 0

    DistributionVisualizer.close_all()

    # All figures should be closed
    assert len(plt.get_fignums()) == 0


# Edge Case Tests

def test_empty_dataframe(visualizer):
    """Test with empty DataFrame."""
    data = pd.DataFrame()
    with pytest.raises(ValueError, match="No numeric columns found"):
        visualizer.create_histograms(data)


def test_single_value_column(visualizer):
    """Test with column containing single unique value."""
    data = pd.DataFrame({'constant': [5.0] * 100})
    fig = visualizer.create_histograms(data)
    assert isinstance(fig, plt.Figure)


def test_column_with_nans(visualizer):
    """Test with column containing NaN values."""
    data = pd.DataFrame({
        'with_nans': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
    })
    fig = visualizer.create_histograms(data)
    assert isinstance(fig, plt.Figure)


def test_all_nans_column(visualizer):
    """Test with column containing all NaN values."""
    data = pd.DataFrame({'all_nans': [np.nan] * 10})
    # Should handle gracefully, possibly with warning
    fig = visualizer.create_histograms(data)
    assert isinstance(fig, plt.Figure)


def test_very_small_dataset(visualizer):
    """Test with very small dataset."""
    data = pd.DataFrame({'small': [1.0, 2.0, 3.0]})
    fig = visualizer.create_histograms(data)
    assert isinstance(fig, plt.Figure)


def test_large_number_of_columns(visualizer):
    """Test with many columns (grid layout)."""
    np.random.seed(42)
    data = pd.DataFrame({
        f'col_{i}': np.random.normal(0, 1, 100)
        for i in range(10)
    })
    fig = visualizer.create_histograms(data)
    assert isinstance(fig, plt.Figure)
    # Should have at least 10 subplots (may have more due to KDE secondary axes and extra grid spaces)
    assert len(fig.axes) >= 10


# Integration Tests

def test_multiple_visualizations_same_data(visualizer, sample_data):
    """Test creating multiple visualization types on same data."""
    fig1 = visualizer.create_histograms(sample_data)
    fig2 = visualizer.create_boxplots(sample_data)
    fig3 = visualizer.create_density_plots(sample_data)
    fig4 = visualizer.create_violin_plots(sample_data)

    assert all(isinstance(fig, plt.Figure) for fig in [fig1, fig2, fig3, fig4])


def test_workflow_with_save(visualizer, sample_data, tmp_path):
    """Test complete workflow: create, save, cleanup."""
    # Create visualization
    fig = visualizer.create_distribution_summary(sample_data, column='normal')
    assert isinstance(fig, plt.Figure)

    # Save to file
    filepath = tmp_path / "distribution_summary.png"
    DistributionVisualizer.save_plot(fig, filepath)
    assert filepath.exists()

    # Cleanup
    DistributionVisualizer.close_all()
    assert len(plt.get_fignums()) == 0


def test_different_styles(sample_data):
    """Test different visualization styles."""
    styles = ['whitegrid', 'darkgrid', 'white', 'dark', 'ticks']

    for style in styles:
        vis = DistributionVisualizer(style=style)
        fig = vis.create_histograms(sample_data, columns=['normal'])
        assert isinstance(fig, plt.Figure)
        plt.close('all')
