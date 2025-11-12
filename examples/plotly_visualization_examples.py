"""
Examples demonstrating interactive Plotly visualizations.

This file shows how to create rich, interactive visualizations using the
PlotlyVisualizer class for enhanced data exploration.
"""

import pandas as pd
import numpy as np
from simplus_eda.visualizers import PlotlyVisualizer


def create_sample_data():
    """Create sample dataset for visualization examples."""
    np.random.seed(42)
    n = 1000

    data = {
        'date': pd.date_range('2023-01-01', periods=n, freq='D'),
        'sales': np.random.normal(10000, 2000, n) + np.sin(np.arange(n) / 30) * 3000,
        'profit': np.random.normal(2000, 500, n) + np.sin(np.arange(n) / 30) * 600,
        'customers': np.random.poisson(50, n),
        'temperature': np.random.normal(20, 5, n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'category': np.random.choice(['A', 'B', 'C'], n, p=[0.5, 0.3, 0.2]),
        'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], n),
        'satisfaction': np.random.uniform(1, 5, n),
        'orders': np.random.randint(1, 20, n)
    }

    df = pd.DataFrame(data)

    # Add some correlations
    df['profit'] = df['sales'] * 0.2 + np.random.normal(0, 200, n)
    df['satisfaction'] = 3 + (df['sales'] - df['sales'].mean()) / df['sales'].std() * 0.5 + np.random.normal(0, 0.3, n)
    df['satisfaction'] = df['satisfaction'].clip(1, 5)

    return df


def example_1_interactive_histogram():
    """Example 1: Interactive histogram with KDE."""
    print("=" * 70)
    print("Example 1: Interactive Histogram with KDE")
    print("=" * 70)

    df = create_sample_data()

    # Create Plotly visualizer
    viz = PlotlyVisualizer(template='plotly_white')

    # Create interactive histogram
    fig = viz.create_interactive_histogram(
        df,
        'sales',
        bins=50,
        show_kde=True,
        title='Distribution of Daily Sales (Interactive)'
    )

    # Save to HTML
    output_path = 'outputs/plotly_histogram.html'
    viz.save_figure(fig, output_path)

    print(f"\n✓ Interactive histogram saved to: {output_path}")
    print("  Features:")
    print("  - Hover for exact values")
    print("  - Zoom in/out")
    print("  - Pan across the distribution")
    print("  - Toggle KDE on/off")
    print("  - Download as PNG")


def example_2_interactive_boxplots():
    """Example 2: Interactive box plots."""
    print("\n" + "=" * 70)
    print("Example 2: Interactive Box Plots")
    print("=" * 70)

    df = create_sample_data()

    viz = PlotlyVisualizer()

    # Create box plots
    fig = viz.create_interactive_boxplot(
        df,
        columns=['sales', 'profit', 'customers', 'temperature'],
        orientation='v',
        title='Distribution Comparison (Hover for Details)'
    )

    output_path = 'outputs/plotly_boxplots.html'
    viz.save_figure(fig, output_path)

    print(f"\n✓ Interactive box plots saved to: {output_path}")
    print("  Features:")
    print("  - Hover for quartiles, outliers, mean, std")
    print("  - Click legend to show/hide series")
    print("  - Zoom to focus on specific ranges")


def example_3_correlation_heatmap():
    """Example 3: Interactive correlation heatmap."""
    print("\n" + "=" * 70)
    print("Example 3: Interactive Correlation Heatmap")
    print("=" * 70)

    df = create_sample_data()

    viz = PlotlyVisualizer(template='plotly_white')

    # Create correlation heatmap
    fig = viz.create_correlation_heatmap(
        df,
        method='pearson',
        title='Interactive Correlation Matrix'
    )

    output_path = 'outputs/plotly_heatmap.html'
    viz.save_figure(fig, output_path)

    print(f"\n✓ Interactive heatmap saved to: {output_path}")
    print("  Features:")
    print("  - Hover for exact correlation values")
    print("  - Color scale: Blue (positive) to Red (negative)")
    print("  - Correlation coefficients displayed")


def example_4_scatter_matrix():
    """Example 4: Interactive scatter matrix (pair plot)."""
    print("\n" + "=" * 70)
    print("Example 4: Interactive Scatter Matrix")
    print("=" * 70)

    df = create_sample_data()

    viz = PlotlyVisualizer()

    # Create scatter matrix
    fig = viz.create_scatter_matrix(
        df,
        columns=['sales', 'profit', 'customers', 'satisfaction'],
        color_by='category',
        title='Relationships Between Key Metrics'
    )

    output_path = 'outputs/plotly_scatter_matrix.html'
    viz.save_figure(fig, output_path)

    print(f"\n✓ Interactive scatter matrix saved to: {output_path}")
    print("  Features:")
    print("  - Explore relationships between all variable pairs")
    print("  - Color-coded by category")
    print("  - Zoom into specific regions")
    print("  - Click legend to filter categories")


def example_5_time_series():
    """Example 5: Interactive time series with range slider."""
    print("\n" + "=" * 70)
    print("Example 5: Interactive Time Series Analysis")
    print("=" * 70)

    df = create_sample_data()

    viz = PlotlyVisualizer(template='plotly_white')

    # Create time series plot
    fig = viz.create_time_series_plot(
        df,
        time_col='date',
        value_cols=['sales', 'profit'],
        show_range_slider=True,
        show_buttons=True,
        title='Sales and Profit Over Time'
    )

    output_path = 'outputs/plotly_timeseries.html'
    viz.save_figure(fig, output_path)

    print(f"\n✓ Interactive time series saved to: {output_path}")
    print("  Features:")
    print("  - Range slider for zooming")
    print("  - Quick buttons (1M, 6M, YTD, 1Y, All)")
    print("  - Hover for exact values on both series")
    print("  - Toggle series visibility")


def example_6_scatter_with_trendline():
    """Example 6: Scatter plot with trendline."""
    print("\n" + "=" * 70)
    print("Example 6: Scatter Plot with Trendline")
    print("=" * 70)

    df = create_sample_data()

    viz = PlotlyVisualizer()

    # Create scatter plot
    fig = viz.create_interactive_scatter(
        df,
        x='sales',
        y='profit',
        color_by='region',
        size_by='customers',
        add_trendline=True,
        title='Profit vs Sales by Region (with Trendline)'
    )

    output_path = 'outputs/plotly_scatter.html'
    viz.save_figure(fig, output_path)

    print(f"\n✓ Interactive scatter plot saved to: {output_path}")
    print("  Features:")
    print("  - Point size represents customer count")
    print("  - Color-coded by region")
    print("  - Trendline shows relationship")
    print("  - Hover for detailed point information")


def example_7_distribution_comparison():
    """Example 7: Violin plots for distribution comparison."""
    print("\n" + "=" * 70)
    print("Example 7: Distribution Comparison (Violin Plots)")
    print("=" * 70)

    df = create_sample_data()

    viz = PlotlyVisualizer(template='plotly_white')

    # Create violin plots
    fig = viz.create_distribution_comparison(
        df,
        column='satisfaction',
        group_by='age_group',
        plot_type='violin',
        title='Satisfaction Distribution by Age Group'
    )

    output_path = 'outputs/plotly_violin.html'
    viz.save_figure(fig, output_path)

    print(f"\n✓ Interactive violin plots saved to: {output_path}")
    print("  Features:")
    print("  - Compare distributions across groups")
    print("  - Box plot embedded in violin")
    print("  - Mean line visible")
    print("  - Click legend to isolate groups")


def example_8_bar_chart():
    """Example 8: Interactive bar chart."""
    print("\n" + "=" * 70)
    print("Example 8: Interactive Bar Chart")
    print("=" * 70)

    df = create_sample_data()

    # Aggregate data
    region_sales = df.groupby('region')['sales'].sum().reset_index()

    viz = PlotlyVisualizer()

    # Create bar chart
    fig = viz.create_bar_chart(
        region_sales,
        x='region',
        y='sales',
        title='Total Sales by Region'
    )

    output_path = 'outputs/plotly_bar.html'
    viz.save_figure(fig, output_path)

    print(f"\n✓ Interactive bar chart saved to: {output_path}")
    print("  Features:")
    print("  - Hover for exact values")
    print("  - Color gradient by value")
    print("  - Zoom and pan capabilities")


def example_9_sunburst():
    """Example 9: Hierarchical sunburst chart."""
    print("\n" + "=" * 70)
    print("Example 9: Sunburst Chart (Hierarchical)")
    print("=" * 70)

    df = create_sample_data()

    # Aggregate data for sunburst
    hierarchy = df.groupby(['region', 'category', 'age_group']).size().reset_index(name='count')

    viz = PlotlyVisualizer(template='plotly_white')

    # Create sunburst
    fig = viz.create_sunburst(
        hierarchy,
        path=['region', 'category', 'age_group'],
        values='count',
        title='Customer Distribution Hierarchy'
    )

    output_path = 'outputs/plotly_sunburst.html'
    viz.save_figure(fig, output_path)

    print(f"\n✓ Interactive sunburst saved to: {output_path}")
    print("  Features:")
    print("  - Click segments to drill down")
    print("  - Hover for percentages and values")
    print("  - Hierarchical exploration")
    print("  - Click center to zoom out")


def example_10_3d_scatter():
    """Example 10: 3D scatter plot."""
    print("\n" + "=" * 70)
    print("Example 10: 3D Scatter Plot")
    print("=" * 70)

    df = create_sample_data()

    viz = PlotlyVisualizer()

    # Create 3D scatter
    fig = viz.create_3d_scatter(
        df.sample(n=500),  # Sample for performance
        x='sales',
        y='profit',
        z='satisfaction',
        color_by='region',
        size_by='customers',
        title='3D Exploration: Sales, Profit, and Satisfaction'
    )

    output_path = 'outputs/plotly_3d.html'
    viz.save_figure(fig, output_path)

    print(f"\n✓ Interactive 3D scatter saved to: {output_path}")
    print("  Features:")
    print("  - Rotate the plot to explore from any angle")
    print("  - Zoom in/out")
    print("  - Color-coded by region")
    print("  - Point size represents customer count")


def example_11_dashboard():
    """Example 11: Combined dashboard."""
    print("\n" + "=" * 70)
    print("Example 11: Combined Dashboard")
    print("=" * 70)

    df = create_sample_data()

    viz = PlotlyVisualizer(template='plotly_white')

    # Create multiple visualizations
    fig1 = viz.create_interactive_histogram(df, 'sales', bins=30)
    fig2 = viz.create_interactive_boxplot(df, ['profit', 'satisfaction'])
    fig3 = viz.create_bar_chart(
        df.groupby('region')['sales'].sum().reset_index(),
        'region',
        'sales'
    )
    fig4 = viz.create_interactive_scatter(df.sample(200), 'sales', 'profit')

    # Combine into dashboard
    dashboard = viz.create_dashboard(
        figures=[fig1, fig2, fig3, fig4],
        titles=['Sales Distribution', 'Box Plots', 'Sales by Region', 'Profit vs Sales'],
        rows=2,
        cols=2
    )

    output_path = 'outputs/plotly_dashboard.html'
    viz.save_figure(dashboard, output_path)

    print(f"\n✓ Interactive dashboard saved to: {output_path}")
    print("  Features:")
    print("  - Multiple visualizations in one view")
    print("  - All interactions work independently")
    print("  - Perfect for comprehensive reports")


def example_12_custom_styling():
    """Example 12: Custom styling and themes."""
    print("\n" + "=" * 70)
    print("Example 12: Custom Styling and Themes")
    print("=" * 70)

    df = create_sample_data()

    # Create visualizer with dark theme
    viz_dark = PlotlyVisualizer(
        template='plotly_dark',
        color_palette=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
        default_height=600
    )

    fig = viz_dark.create_interactive_histogram(
        df,
        'sales',
        bins=40,
        show_kde=True,
        title='Sales Distribution (Dark Theme)'
    )

    output_path = 'outputs/plotly_dark_theme.html'
    viz_dark.save_figure(fig, output_path)

    print(f"\n✓ Dark theme visualization saved to: {output_path}")
    print("  Customization options:")
    print("  - Template: plotly, plotly_white, plotly_dark, ggplot2, etc.")
    print("  - Custom color palettes")
    print("  - Adjustable dimensions")


def example_13_save_formats():
    """Example 13: Saving in different formats."""
    print("\n" + "=" * 70)
    print("Example 13: Exporting to Multiple Formats")
    print("=" * 70)

    df = create_sample_data()

    viz = PlotlyVisualizer()

    fig = viz.create_correlation_heatmap(df, method='pearson')

    # Save as HTML (interactive)
    viz.save_figure(fig, 'outputs/correlation.html', format='html')
    print("\n✓ Saved as HTML (interactive)")

    # Save as static images (requires kaleido)
    try:
        viz.save_figure(fig, 'outputs/correlation.png', format='png', width=1200, height=800)
        print("✓ Saved as PNG (static)")
    except Exception as e:
        print(f"⚠️  PNG export failed: {e}")
        print("   Install kaleido: pip install kaleido")

    try:
        viz.save_figure(fig, 'outputs/correlation.svg', format='svg')
        print("✓ Saved as SVG (vector)")
    except Exception as e:
        print(f"⚠️  SVG export failed: {e}")

    # Save as JSON (for programmatic use)
    viz.save_figure(fig, 'outputs/correlation.json', format='json')
    print("✓ Saved as JSON (data format)")

    print("\nSupported formats:")
    print("  - HTML: Interactive, web-ready")
    print("  - PNG: Static image (requires kaleido)")
    print("  - SVG: Vector format (requires kaleido)")
    print("  - PDF: Print-ready (requires kaleido)")
    print("  - JSON: Programmatic access")


def main():
    """Run all Plotly visualization examples."""
    import os
    os.makedirs('outputs', exist_ok=True)

    print("\n" + "=" * 70)
    print("Simplus EDA - Interactive Plotly Visualizations")
    print("=" * 70)
    print("\nThese examples demonstrate the power of interactive visualizations")
    print("using Plotly for enhanced data exploration.\n")

    examples = [
        example_1_interactive_histogram,
        example_2_interactive_boxplots,
        example_3_correlation_heatmap,
        example_4_scatter_matrix,
        example_5_time_series,
        example_6_scatter_with_trendline,
        example_7_distribution_comparison,
        example_8_bar_chart,
        example_9_sunburst,
        example_10_3d_scatter,
        example_11_dashboard,
        example_12_custom_styling,
        example_13_save_formats,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n⚠️  Error in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All Examples Completed!")
    print("=" * 70)
    print("\nKey Features of Plotly Visualizations:")
    print("✓ Interactive - Hover, zoom, pan, and explore")
    print("✓ Web-ready - Embed in dashboards and reports")
    print("✓ Professional - Publication-quality output")
    print("✓ Responsive - Works on all devices")
    print("✓ Customizable - Full control over styling")
    print("✓ Exportable - HTML, PNG, SVG, PDF formats")
    print("\nOpen any .html file in your browser to explore!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
