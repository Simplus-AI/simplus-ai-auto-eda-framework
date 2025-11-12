"""
Tests for the RelationshipVisualizer class.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from simplus_eda.visualizers.relationships import RelationshipVisualizer


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 200

    # Create correlated data
    x1 = np.random.normal(100, 15, n)
    x2 = x1 + np.random.normal(0, 5, n)  # Strongly correlated with x1
    x3 = np.random.uniform(0, 100, n)    # Independent
    x4 = -x1 + np.random.normal(0, 10, n)  # Negatively correlated with x1

    return pd.DataFrame({
        'feature1': x1,
        'feature2': x2,
        'feature3': x3,
        'feature4': x4,
        'feature5': np.random.exponential(2, n),
        'category': np.random.choice(['A', 'B', 'C'], n)
    })


@pytest.fixture
def visualizer():
    """Create RelationshipVisualizer instance."""
    return RelationshipVisualizer()


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Cleanup plots after each test."""
    yield
    plt.close('all')


# Initialization Tests

def test_init_default():
    """Test initialization with default parameters."""
    vis = RelationshipVisualizer()
    assert vis.default_figsize == (10, 8)
    assert vis.default_style == 'whitegrid'
    assert vis.default_palette == 'husl'
    assert vis.default_dpi == 100


def test_init_custom():
    """Test initialization with custom parameters."""
    vis = RelationshipVisualizer(
        style='darkgrid',
        palette='Set2',
        figsize=(15, 10),
        dpi=150
    )
    assert vis.default_figsize == (15, 10)
    assert vis.default_style == 'darkgrid'
    assert vis.default_palette == 'Set2'
    assert vis.default_dpi == 150


# Correlation Heatmap Tests

def test_create_correlation_heatmap_basic(visualizer, sample_data):
    """Test basic correlation heatmap creation."""
    fig = visualizer.create_correlation_heatmap(sample_data)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # heatmap + colorbar


def test_create_correlation_heatmap_pearson(visualizer, sample_data):
    """Test correlation heatmap with Pearson method."""
    fig = visualizer.create_correlation_heatmap(sample_data, method='pearson')
    assert isinstance(fig, plt.Figure)


def test_create_correlation_heatmap_spearman(visualizer, sample_data):
    """Test correlation heatmap with Spearman method."""
    fig = visualizer.create_correlation_heatmap(sample_data, method='spearman')
    assert isinstance(fig, plt.Figure)


def test_create_correlation_heatmap_kendall(visualizer, sample_data):
    """Test correlation heatmap with Kendall method."""
    fig = visualizer.create_correlation_heatmap(sample_data, method='kendall')
    assert isinstance(fig, plt.Figure)


def test_create_correlation_heatmap_no_annot(visualizer, sample_data):
    """Test correlation heatmap without annotations."""
    fig = visualizer.create_correlation_heatmap(sample_data, annot=False)
    assert isinstance(fig, plt.Figure)


def test_create_correlation_heatmap_custom_cmap(visualizer, sample_data):
    """Test correlation heatmap with custom colormap."""
    fig = visualizer.create_correlation_heatmap(sample_data, cmap='viridis')
    assert isinstance(fig, plt.Figure)


def test_create_correlation_heatmap_mask_diagonal(visualizer, sample_data):
    """Test correlation heatmap with diagonal masked."""
    fig = visualizer.create_correlation_heatmap(sample_data, mask_diagonal=True)
    assert isinstance(fig, plt.Figure)


def test_create_correlation_heatmap_mask_upper(visualizer, sample_data):
    """Test correlation heatmap with upper triangle masked."""
    fig = visualizer.create_correlation_heatmap(sample_data, mask_upper=True)
    assert isinstance(fig, plt.Figure)


def test_create_correlation_heatmap_both_masks(visualizer, sample_data):
    """Test correlation heatmap with both masks."""
    fig = visualizer.create_correlation_heatmap(
        sample_data,
        mask_diagonal=True,
        mask_upper=True
    )
    assert isinstance(fig, plt.Figure)


def test_create_correlation_heatmap_custom_figsize(visualizer, sample_data):
    """Test correlation heatmap with custom figure size."""
    fig = visualizer.create_correlation_heatmap(sample_data, figsize=(15, 12))
    assert isinstance(fig, plt.Figure)
    assert fig.get_figwidth() == 15
    assert fig.get_figheight() == 12


def test_create_correlation_heatmap_insufficient_columns(visualizer):
    """Test error with insufficient numeric columns."""
    data = pd.DataFrame({'single': [1, 2, 3, 4, 5]})
    with pytest.raises(ValueError, match="at least 2 numerical columns"):
        visualizer.create_correlation_heatmap(data)


# Scatter Matrix Tests

def test_create_scatter_matrix_basic(visualizer, sample_data):
    """Test basic scatter matrix creation."""
    fig = visualizer.create_scatter_matrix(sample_data)
    assert isinstance(fig, plt.Figure)


def test_create_scatter_matrix_selected_columns(visualizer, sample_data):
    """Test scatter matrix with selected columns."""
    fig = visualizer.create_scatter_matrix(
        sample_data,
        columns=['feature1', 'feature2', 'feature3']
    )
    assert isinstance(fig, plt.Figure)
    # Should have 3x3 grid
    assert len(fig.axes) == 9


def test_create_scatter_matrix_diagonal_hist(visualizer, sample_data):
    """Test scatter matrix with histogram diagonal."""
    fig = visualizer.create_scatter_matrix(
        sample_data,
        columns=['feature1', 'feature2'],
        diagonal='hist'
    )
    assert isinstance(fig, plt.Figure)


def test_create_scatter_matrix_diagonal_kde(visualizer, sample_data):
    """Test scatter matrix with KDE diagonal."""
    fig = visualizer.create_scatter_matrix(
        sample_data,
        columns=['feature1', 'feature2'],
        diagonal='kde'
    )
    assert isinstance(fig, plt.Figure)


def test_create_scatter_matrix_diagonal_none(visualizer, sample_data):
    """Test scatter matrix with no diagonal plots."""
    fig = visualizer.create_scatter_matrix(
        sample_data,
        columns=['feature1', 'feature2'],
        diagonal=None
    )
    assert isinstance(fig, plt.Figure)


def test_create_scatter_matrix_custom_alpha(visualizer, sample_data):
    """Test scatter matrix with custom alpha."""
    fig = visualizer.create_scatter_matrix(
        sample_data,
        columns=['feature1', 'feature2'],
        alpha=0.3
    )
    assert isinstance(fig, plt.Figure)


def test_create_scatter_matrix_column_not_found(visualizer, sample_data):
    """Test error when column not found."""
    with pytest.raises(ValueError, match="not found in DataFrame"):
        visualizer.create_scatter_matrix(sample_data, columns=['nonexistent'])


def test_create_scatter_matrix_non_numeric_column(visualizer, sample_data):
    """Test error with non-numeric column."""
    with pytest.raises(ValueError, match="not numerical"):
        visualizer.create_scatter_matrix(sample_data, columns=['category'])


def test_create_scatter_matrix_insufficient_columns(visualizer, sample_data):
    """Test error with insufficient columns."""
    with pytest.raises(ValueError, match="at least 2 columns"):
        visualizer.create_scatter_matrix(sample_data, columns=['feature1'])


# Pairplot Tests

def test_create_pairplot_basic(visualizer, sample_data):
    """Test basic pairplot creation."""
    fig = visualizer.create_pairplot(sample_data)
    assert isinstance(fig, plt.Figure)


def test_create_pairplot_selected_columns(visualizer, sample_data):
    """Test pairplot with selected columns."""
    fig = visualizer.create_pairplot(
        sample_data,
        columns=['feature1', 'feature2', 'feature3']
    )
    assert isinstance(fig, plt.Figure)


def test_create_pairplot_with_hue(visualizer, sample_data):
    """Test pairplot with hue."""
    fig = visualizer.create_pairplot(sample_data, hue='category')
    assert isinstance(fig, plt.Figure)


def test_create_pairplot_kind_scatter(visualizer, sample_data):
    """Test pairplot with scatter kind."""
    fig = visualizer.create_pairplot(
        sample_data,
        columns=['feature1', 'feature2'],
        kind='scatter'
    )
    assert isinstance(fig, plt.Figure)


def test_create_pairplot_kind_reg(visualizer, sample_data):
    """Test pairplot with regression kind."""
    fig = visualizer.create_pairplot(
        sample_data,
        columns=['feature1', 'feature2'],
        kind='reg'
    )
    assert isinstance(fig, plt.Figure)


def test_create_pairplot_diag_hist(visualizer, sample_data):
    """Test pairplot with histogram diagonal."""
    fig = visualizer.create_pairplot(
        sample_data,
        columns=['feature1', 'feature2'],
        diag_kind='hist'
    )
    assert isinstance(fig, plt.Figure)


def test_create_pairplot_diag_kde(visualizer, sample_data):
    """Test pairplot with KDE diagonal."""
    fig = visualizer.create_pairplot(
        sample_data,
        columns=['feature1', 'feature2'],
        diag_kind='kde'
    )
    assert isinstance(fig, plt.Figure)


def test_create_pairplot_corner(visualizer, sample_data):
    """Test pairplot with corner mode."""
    fig = visualizer.create_pairplot(
        sample_data,
        columns=['feature1', 'feature2', 'feature3'],
        corner=True
    )
    assert isinstance(fig, plt.Figure)


def test_create_pairplot_hue_not_found(visualizer, sample_data):
    """Test error when hue column not found."""
    with pytest.raises(ValueError, match="not found in DataFrame"):
        visualizer.create_pairplot(sample_data, hue='nonexistent')


def test_create_pairplot_insufficient_columns(visualizer, sample_data):
    """Test error with insufficient columns."""
    with pytest.raises(ValueError, match="at least 2 numerical columns"):
        visualizer.create_pairplot(sample_data, columns=['feature1'])


# Scatter Plot Tests

def test_create_scatter_plot_basic(visualizer, sample_data):
    """Test basic scatter plot creation."""
    fig = visualizer.create_scatter_plot(sample_data, 'feature1', 'feature2')
    assert isinstance(fig, plt.Figure)


def test_create_scatter_plot_with_regression(visualizer, sample_data):
    """Test scatter plot with regression line."""
    fig = visualizer.create_scatter_plot(
        sample_data,
        'feature1',
        'feature2',
        add_regression=True
    )
    assert isinstance(fig, plt.Figure)


def test_create_scatter_plot_without_regression(visualizer, sample_data):
    """Test scatter plot without regression line."""
    fig = visualizer.create_scatter_plot(
        sample_data,
        'feature1',
        'feature2',
        add_regression=False
    )
    assert isinstance(fig, plt.Figure)


def test_create_scatter_plot_with_hue(visualizer, sample_data):
    """Test scatter plot with hue."""
    fig = visualizer.create_scatter_plot(
        sample_data,
        'feature1',
        'feature2',
        hue='category'
    )
    assert isinstance(fig, plt.Figure)


def test_create_scatter_plot_with_size(visualizer, sample_data):
    """Test scatter plot with size."""
    fig = visualizer.create_scatter_plot(
        sample_data,
        'feature1',
        'feature2',
        size='feature3'
    )
    assert isinstance(fig, plt.Figure)


def test_create_scatter_plot_custom_figsize(visualizer, sample_data):
    """Test scatter plot with custom figure size."""
    fig = visualizer.create_scatter_plot(
        sample_data,
        'feature1',
        'feature2',
        figsize=(12, 10)
    )
    assert isinstance(fig, plt.Figure)
    assert fig.get_figwidth() == 12
    assert fig.get_figheight() == 10


def test_create_scatter_plot_x_not_found(visualizer, sample_data):
    """Test error when x column not found."""
    with pytest.raises(ValueError, match="not found in DataFrame"):
        visualizer.create_scatter_plot(sample_data, 'nonexistent', 'feature2')


def test_create_scatter_plot_y_not_found(visualizer, sample_data):
    """Test error when y column not found."""
    with pytest.raises(ValueError, match="not found in DataFrame"):
        visualizer.create_scatter_plot(sample_data, 'feature1', 'nonexistent')


def test_create_scatter_plot_hue_not_found(visualizer, sample_data):
    """Test error when hue column not found."""
    with pytest.raises(ValueError, match="not found in DataFrame"):
        visualizer.create_scatter_plot(
            sample_data,
            'feature1',
            'feature2',
            hue='nonexistent'
        )


def test_create_scatter_plot_size_not_found(visualizer, sample_data):
    """Test error when size column not found."""
    with pytest.raises(ValueError, match="not found in DataFrame"):
        visualizer.create_scatter_plot(
            sample_data,
            'feature1',
            'feature2',
            size='nonexistent'
        )


# Correlation Pairs Tests

def test_create_correlation_pairs_basic(visualizer, sample_data):
    """Test basic correlation pairs creation."""
    # Lower threshold to ensure we find pairs
    fig = visualizer.create_correlation_pairs(sample_data, threshold=0.5)
    assert isinstance(fig, plt.Figure)


def test_create_correlation_pairs_custom_threshold(visualizer, sample_data):
    """Test correlation pairs with custom threshold."""
    fig = visualizer.create_correlation_pairs(sample_data, threshold=0.6)
    assert isinstance(fig, plt.Figure)


def test_create_correlation_pairs_max_pairs(visualizer, sample_data):
    """Test correlation pairs with max pairs limit."""
    fig = visualizer.create_correlation_pairs(
        sample_data,
        threshold=0.5,
        max_pairs=3
    )
    assert isinstance(fig, plt.Figure)


def test_create_correlation_pairs_spearman(visualizer, sample_data):
    """Test correlation pairs with Spearman method."""
    fig = visualizer.create_correlation_pairs(
        sample_data,
        threshold=0.5,
        method='spearman'
    )
    assert isinstance(fig, plt.Figure)


def test_create_correlation_pairs_no_pairs_found(visualizer, sample_data):
    """Test error when no pairs found above threshold."""
    with pytest.raises(ValueError, match="No correlation pairs found"):
        visualizer.create_correlation_pairs(sample_data, threshold=0.99)


def test_create_correlation_pairs_insufficient_columns(visualizer):
    """Test error with insufficient numeric columns."""
    data = pd.DataFrame({'single': [1, 2, 3, 4, 5]})
    with pytest.raises(ValueError, match="at least 2 numerical columns"):
        visualizer.create_correlation_pairs(data)


# Utility Tests

def test_save_plot(visualizer, sample_data, tmp_path):
    """Test saving plot to file."""
    fig = visualizer.create_correlation_heatmap(sample_data)
    filepath = tmp_path / "test_plot.png"

    RelationshipVisualizer.save_plot(fig, filepath)

    assert filepath.exists()


def test_save_plot_creates_directory(visualizer, sample_data, tmp_path):
    """Test that save_plot creates parent directories."""
    fig = visualizer.create_correlation_heatmap(sample_data)
    filepath = tmp_path / "subdir" / "test_plot.png"

    RelationshipVisualizer.save_plot(fig, filepath)

    assert filepath.exists()


def test_save_plot_custom_dpi(visualizer, sample_data, tmp_path):
    """Test saving plot with custom DPI."""
    fig = visualizer.create_correlation_heatmap(sample_data)
    filepath = tmp_path / "test_plot.png"

    RelationshipVisualizer.save_plot(fig, filepath, dpi=150)

    assert filepath.exists()


def test_close_all(visualizer, sample_data):
    """Test closing all figures."""
    visualizer.create_correlation_heatmap(sample_data)
    visualizer.create_scatter_plot(sample_data, 'feature1', 'feature2')

    # Should have open figures
    assert len(plt.get_fignums()) > 0

    RelationshipVisualizer.close_all()

    # All figures should be closed
    assert len(plt.get_fignums()) == 0


# Edge Case Tests

def test_data_with_nans(visualizer):
    """Test with data containing NaN values."""
    data = pd.DataFrame({
        'x': [1, 2, np.nan, 4, 5],
        'y': [2, np.nan, 6, 8, 10],
        'z': [1, 2, 3, 4, 5]
    })
    fig = visualizer.create_correlation_heatmap(data)
    assert isinstance(fig, plt.Figure)


def test_perfect_correlation(visualizer):
    """Test with perfectly correlated data."""
    x = np.arange(100)
    data = pd.DataFrame({
        'x': x,
        'y': x * 2  # Perfect positive correlation
    })
    fig = visualizer.create_scatter_plot(data, 'x', 'y')
    assert isinstance(fig, plt.Figure)


def test_no_correlation(visualizer):
    """Test with uncorrelated data."""
    np.random.seed(42)
    data = pd.DataFrame({
        'x': np.random.normal(0, 1, 100),
        'y': np.random.normal(0, 1, 100)
    })
    fig = visualizer.create_scatter_plot(data, 'x', 'y')
    assert isinstance(fig, plt.Figure)


def test_constant_column(visualizer):
    """Test with constant column (zero variance)."""
    data = pd.DataFrame({
        'constant': [5.0] * 100,
        'variable': np.random.normal(0, 1, 100)
    })
    # Should handle gracefully
    fig = visualizer.create_correlation_heatmap(data)
    assert isinstance(fig, plt.Figure)


def test_very_small_dataset(visualizer):
    """Test with very small dataset."""
    data = pd.DataFrame({
        'x': [1.0, 2.0, 3.0],
        'y': [2.0, 4.0, 6.0]
    })
    fig = visualizer.create_scatter_plot(data, 'x', 'y')
    assert isinstance(fig, plt.Figure)


# Integration Tests

def test_multiple_visualizations_same_data(visualizer, sample_data):
    """Test creating multiple visualization types on same data."""
    fig1 = visualizer.create_correlation_heatmap(sample_data)
    fig2 = visualizer.create_scatter_matrix(
        sample_data,
        columns=['feature1', 'feature2', 'feature3']
    )
    fig3 = visualizer.create_scatter_plot(sample_data, 'feature1', 'feature2')

    assert all(isinstance(fig, plt.Figure) for fig in [fig1, fig2, fig3])


def test_workflow_with_save(visualizer, sample_data, tmp_path):
    """Test complete workflow: create, save, cleanup."""
    # Create visualization
    fig = visualizer.create_correlation_heatmap(sample_data)
    assert isinstance(fig, plt.Figure)

    # Save to file
    filepath = tmp_path / "correlation_heatmap.png"
    RelationshipVisualizer.save_plot(fig, filepath)
    assert filepath.exists()

    # Cleanup
    RelationshipVisualizer.close_all()
    assert len(plt.get_fignums()) == 0


def test_different_styles(sample_data):
    """Test different visualization styles."""
    styles = ['whitegrid', 'darkgrid', 'white', 'dark', 'ticks']

    for style in styles:
        vis = RelationshipVisualizer(style=style)
        fig = vis.create_scatter_plot(sample_data, 'feature1', 'feature2')
        assert isinstance(fig, plt.Figure)
        plt.close('all')


def test_comprehensive_analysis(visualizer, sample_data):
    """Test comprehensive relationship analysis workflow."""
    # Create correlation heatmap
    fig1 = visualizer.create_correlation_heatmap(sample_data)
    assert isinstance(fig1, plt.Figure)

    # Find and visualize highly correlated pairs
    fig2 = visualizer.create_correlation_pairs(sample_data, threshold=0.5)
    assert isinstance(fig2, plt.Figure)

    # Create pairplot for detailed view
    fig3 = visualizer.create_pairplot(
        sample_data,
        columns=['feature1', 'feature2', 'feature3'],
        hue='category'
    )
    assert isinstance(fig3, plt.Figure)

    # All should be successful
    assert all(isinstance(fig, plt.Figure) for fig in [fig1, fig2, fig3])
