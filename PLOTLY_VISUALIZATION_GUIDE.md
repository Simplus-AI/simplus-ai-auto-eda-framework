# Interactive Plotly Visualizations Guide

## Overview

The **PlotlyVisualizer** class provides rich, interactive visualizations using Plotly, offering a superior exploration experience compared to static plots. These visualizations are web-ready, responsive, and provide interactivity features like hover details, zooming, panning, and data filtering.

## Why Use Plotly Visualizations?

### Static Plots (Matplotlib/Seaborn)
```python
# Static - Limited interaction
import matplotlib.pyplot as plt
plt.hist(df['sales'])
plt.show()  # Just a static image
```

### Interactive Plots (Plotly)
```python
# Interactive - Rich exploration
from simplus_eda.visualizers import PlotlyVisualizer

viz = PlotlyVisualizer()
fig = viz.create_interactive_histogram(df, 'sales')
fig.show()  # Interactive: hover, zoom, pan, download!
```

### Key Advantages

✅ **Interactive** - Hover for details, zoom, pan, rotate (3D)
✅ **Web-Ready** - Embed in dashboards, reports, and websites
✅ **Professional** - Publication-quality output
✅ **Responsive** - Works on desktop, tablet, and mobile
✅ **Exportable** - HTML, PNG, SVG, PDF, JSON formats
✅ **Customizable** - Full control over styling and themes

## Installation

Plotly is already included in the core dependencies:

```bash
pip install simplus-eda
```

For static image export (PNG, SVG, PDF):
```bash
pip install kaleido
```

## Quick Start

### Basic Usage

```python
from simplus_eda.visualizers import PlotlyVisualizer
import pandas as pd

# Create visualizer
viz = PlotlyVisualizer()

# Load data
df = pd.read_csv('your_data.csv')

# Create interactive histogram
fig = viz.create_interactive_histogram(df, 'sales', bins=30)

# Display in Jupyter
fig.show()

# Or save to HTML
viz.save_figure(fig, 'sales_histogram.html')
```

## Available Visualizations

### 1. Interactive Histogram

Create histograms with optional KDE overlay:

```python
fig = viz.create_interactive_histogram(
    data=df,
    column='sales',
    bins=50,
    show_kde=True,
    title='Sales Distribution'
)
```

**Features:**
- Hover for bin ranges and counts
- Toggle KDE curve on/off
- Zoom into specific ranges
- Download as PNG

**Use Cases:**
- Understand data distribution
- Identify skewness and outliers
- Compare to normal distribution

### 2. Interactive Box Plots

Multiple box plots for comparison:

```python
fig = viz.create_interactive_boxplot(
    data=df,
    columns=['sales', 'profit', 'revenue'],
    orientation='v',  # or 'h' for horizontal
    title='Metrics Comparison'
)
```

**Features:**
- Hover for quartiles, outliers, mean, std dev
- Click legend to show/hide series
- Zoom to focus on specific ranges
- Box plot with standard deviation

**Use Cases:**
- Compare distributions across variables
- Identify outliers
- Understand data spread

### 3. Correlation Heatmap

Interactive correlation matrix:

```python
fig = viz.create_correlation_heatmap(
    data=df,
    method='pearson',  # or 'spearman', 'kendall'
    columns=None,  # None = all numeric
    title='Correlation Matrix'
)
```

**Features:**
- Hover for exact correlation values
- Color scale: Blue (positive) to Red (negative)
- Coefficient values displayed
- Zoom into specific regions

**Use Cases:**
- Identify correlated features
- Feature selection for ML
- Understand relationships

### 4. Scatter Matrix (Pair Plot)

Explore all variable relationships:

```python
fig = viz.create_scatter_matrix(
    data=df,
    columns=['age', 'income', 'score'],
    color_by='category',  # Optional categorical column
    title='Variable Relationships'
)
```

**Features:**
- All pairwise scatter plots
- Color-coded by category
- Zoom into specific regions
- Click legend to filter

**Use Cases:**
- Explore multivariate relationships
- Identify patterns and clusters
- Compare groups

### 5. Interactive Scatter Plot

Scatter plot with trendline:

```python
fig = viz.create_interactive_scatter(
    data=df,
    x='age',
    y='income',
    color_by='region',  # Optional
    size_by='score',     # Optional
    add_trendline=True,
    title='Income vs Age'
)
```

**Features:**
- Point size represents additional variable
- Color-coded by category
- OLS trendline
- Hover for all values

**Use Cases:**
- Explore relationships
- Identify trends
- Spot outliers and clusters

### 6. Time Series Plot

Time series with range slider:

```python
fig = viz.create_time_series_plot(
    data=df,
    time_col='date',
    value_cols=['sales', 'profit'],  # Single or multiple
    show_range_slider=True,
    show_buttons=True,
    title='Sales Over Time'
)
```

**Features:**
- Range slider for zooming
- Quick period buttons (1M, 6M, YTD, 1Y, All)
- Multiple series support
- Unified hover mode

**Use Cases:**
- Analyze trends over time
- Seasonal pattern detection
- Compare multiple metrics

### 7. Distribution Comparison

Violin or box plots by group:

```python
fig = viz.create_distribution_comparison(
    data=df,
    column='satisfaction',
    group_by='region',
    plot_type='violin',  # or 'box'
    title='Satisfaction by Region'
)
```

**Features:**
- Violin plots show distribution shape
- Box plot embedded in violin
- Mean line visible
- Click legend to isolate groups

**Use Cases:**
- Compare distributions across categories
- A/B testing visualization
- Segment analysis

### 8. Interactive Bar Chart

Bar charts with sorting and filtering:

```python
fig = viz.create_bar_chart(
    data=df,
    x='category',
    y='sales',  # None = count occurrences
    orientation='v',  # or 'h'
    sort_by='sales',  # Optional
    top_n=10,  # Optional
    title='Top 10 Categories'
)
```

**Features:**
- Hover for exact values
- Color gradient by value
- Sort by any column
- Limit to top N

**Use Cases:**
- Category comparison
- Ranking visualization
- Frequency analysis

### 9. Sunburst Chart

Hierarchical data visualization:

```python
fig = viz.create_sunburst(
    data=df,
    path=['country', 'city', 'district'],
    values='population',  # None = count
    title='Geographic Distribution'
)
```

**Features:**
- Click segments to drill down
- Hover for percentages and values
- Hierarchical exploration
- Click center to zoom out

**Use Cases:**
- Hierarchical data exploration
- Budget breakdowns
- Organization structure

### 10. 3D Scatter Plot

Three-dimensional exploration:

```python
fig = viz.create_3d_scatter(
    data=df,
    x='age',
    y='income',
    z='satisfaction',
    color_by='region',  # Optional
    size_by='score',    # Optional
    title='3D Exploration'
)
```

**Features:**
- Rotate plot to any angle
- Zoom in/out
- Color and size encoding
- Interactive legend

**Use Cases:**
- Explore three variables simultaneously
- Identify 3D patterns
- Advanced multivariate analysis

### 11. Dashboard Layout

Combine multiple visualizations:

```python
# Create individual figures
fig1 = viz.create_interactive_histogram(df, 'sales')
fig2 = viz.create_interactive_boxplot(df, ['profit'])
fig3 = viz.create_bar_chart(df, 'region', 'sales')

# Combine into dashboard
dashboard = viz.create_dashboard(
    figures=[fig1, fig2, fig3],
    titles=['Sales Dist', 'Profit', 'By Region'],
    rows=2,
    cols=2
)

viz.save_figure(dashboard, 'dashboard.html')
```

**Features:**
- Multiple charts in one view
- Independent interactions
- Custom grid layout
- Responsive design

**Use Cases:**
- Comprehensive reports
- Executive dashboards
- Multi-metric monitoring

## Customization

### Templates and Themes

```python
# Built-in templates
viz = PlotlyVisualizer(template='plotly')        # Default
viz = PlotlyVisualizer(template='plotly_white')  # Clean white
viz = PlotlyVisualizer(template='plotly_dark')   # Dark mode
viz = PlotlyVisualizer(template='ggplot2')       # ggplot2 style
viz = PlotlyVisualizer(template='seaborn')       # Seaborn style
viz = PlotlyVisualizer(template='simple_white')  # Minimal
```

### Custom Color Palettes

```python
# Use custom colors
viz = PlotlyVisualizer(
    template='plotly_white',
    color_palette=[
        '#FF6B6B',  # Red
        '#4ECDC4',  # Teal
        '#45B7D1',  # Blue
        '#FFA07A',  # Salmon
        '#98D8C8'   # Mint
    ]
)
```

### Default Dimensions

```python
# Set default figure size
viz = PlotlyVisualizer(
    default_height=600,
    default_width=1000  # None = responsive
)
```

## Saving Visualizations

### HTML (Interactive)

```python
# Save as interactive HTML
viz.save_figure(fig, 'chart.html', format='html')

# Include Plotly.js in file (larger but standalone)
viz.save_figure(
    fig,
    'chart.html',
    include_plotlyjs='cdn'  # or True for embedded
)
```

### Static Images (requires kaleido)

```python
# Install kaleido first: pip install kaleido

# PNG
viz.save_figure(fig, 'chart.png', format='png', width=1200, height=800)

# SVG (vector format)
viz.save_figure(fig, 'chart.svg', format='svg')

# PDF (print-ready)
viz.save_figure(fig, 'chart.pdf', format='pdf')
```

### JSON (Data Format)

```python
# Save figure as JSON for programmatic use
viz.save_figure(fig, 'chart.json', format='json')

# Load and modify later
import plotly.io as pio
loaded_fig = pio.read_json('chart.json')
```

## Integration with SimplusEDA

### Using in Analysis

```python
from simplus_eda import SimplusEDA
from simplus_eda.visualizers import PlotlyVisualizer

# Perform analysis
eda = SimplusEDA()
eda.analyze(df)

# Create custom visualizations
viz = PlotlyVisualizer()

# Get correlations
correlations = eda.get_correlations()

# Visualize correlations
fig = viz.create_correlation_heatmap(df, method='pearson')
viz.save_figure(fig, 'correlations.html')
```

### Adding to Reports

```python
# Generate report with static plots
eda.generate_report('report.html')

# Create interactive Plotly versions separately
viz = PlotlyVisualizer()
fig = viz.create_time_series_plot(df, 'date', ['sales', 'profit'])
viz.save_figure(fig, 'interactive_timeseries.html')
```

## Best Practices

### 1. Sample Large Datasets

```python
# For performance, sample large datasets
if len(df) > 10000:
    sample_df = df.sample(n=10000, random_state=42)
    fig = viz.create_interactive_scatter(sample_df, 'x', 'y')
else:
    fig = viz.create_interactive_scatter(df, 'x', 'y')
```

### 2. Choose Appropriate Chart Types

- **Histogram**: Single variable distribution
- **Box Plot**: Multiple variable comparison
- **Scatter**: Two variable relationship
- **Heatmap**: Many variable correlations
- **Time Series**: Temporal patterns
- **Bar Chart**: Category comparison
- **Violin**: Distribution comparison across groups

### 3. Use Consistent Themes

```python
# Create visualizer once with theme
viz = PlotlyVisualizer(template='plotly_white')

# Use for all visualizations
fig1 = viz.create_interactive_histogram(df, 'sales')
fig2 = viz.create_bar_chart(df, 'region', 'sales')
fig3 = viz.create_correlation_heatmap(df)
# All will have consistent styling
```

### 4. Add Meaningful Titles

```python
# Good - descriptive titles
fig = viz.create_interactive_histogram(
    df,
    'revenue',
    title='Daily Revenue Distribution (2024 Q1-Q3)'
)

# Bad - generic titles
fig = viz.create_interactive_histogram(df, 'revenue')
```

### 5. Optimize for Web

```python
# For web embedding, use CDN for Plotly.js
viz.save_figure(
    fig,
    'chart.html',
    include_plotlyjs='cdn'  # Smaller file size
)
```

## Examples

### Example 1: Sales Dashboard

```python
from simplus_eda.visualizers import PlotlyVisualizer
import pandas as pd

# Load sales data
df = pd.read_csv('sales_data.csv')

# Create visualizer
viz = PlotlyVisualizer(template='plotly_white')

# 1. Time series of sales
ts_fig = viz.create_time_series_plot(
    df,
    'date',
    ['sales', 'profit'],
    title='Sales and Profit Over Time'
)

# 2. Distribution by region
region_fig = viz.create_distribution_comparison(
    df,
    'sales',
    'region',
    plot_type='violin',
    title='Sales Distribution by Region'
)

# 3. Correlation analysis
corr_fig = viz.create_correlation_heatmap(
    df,
    title='Metric Correlations'
)

# 4. Top products
product_fig = viz.create_bar_chart(
    df.groupby('product')['sales'].sum().reset_index(),
    'product',
    'sales',
    top_n=10,
    title='Top 10 Products by Sales'
)

# Save all
viz.save_figure(ts_fig, 'sales_timeseries.html')
viz.save_figure(region_fig, 'sales_by_region.html')
viz.save_figure(corr_fig, 'correlations.html')
viz.save_figure(product_fig, 'top_products.html')

# Or combine into dashboard
dashboard = viz.create_dashboard(
    [ts_fig, region_fig, corr_fig, product_fig],
    rows=2,
    cols=2
)
viz.save_figure(dashboard, 'sales_dashboard.html')
```

### Example 2: Customer Segmentation

```python
# Scatter matrix for customer attributes
fig = viz.create_scatter_matrix(
    df,
    columns=['age', 'income', 'lifetime_value', 'satisfaction'],
    color_by='segment',
    title='Customer Segmentation Analysis'
)
viz.save_figure(fig, 'customer_segments.html')

# 3D visualization
fig_3d = viz.create_3d_scatter(
    df,
    'age',
    'income',
    'lifetime_value',
    color_by='segment',
    size_by='satisfaction',
    title='3D Customer Analysis'
)
viz.save_figure(fig_3d, 'customer_3d.html')
```

### Example 3: A/B Test Results

```python
# Compare distributions
fig = viz.create_distribution_comparison(
    df,
    column='conversion_rate',
    group_by='test_group',
    plot_type='violin',
    title='A/B Test: Conversion Rate Comparison'
)
viz.save_figure(fig, 'ab_test_results.html')
```

## Jupyter Notebook Integration

```python
# Display inline in Jupyter
from simplus_eda.visualizers import PlotlyVisualizer

viz = PlotlyVisualizer()
fig = viz.create_interactive_histogram(df, 'sales')
fig.show()  # Displays inline with full interactivity
```

## Performance Tips

1. **Sample large datasets** - Use `.sample()` for > 10K points in scatter plots
2. **Limit columns** - Don't put 50 columns in a scatter matrix
3. **Use CDN** - Include Plotly.js via CDN for smaller files
4. **Batch save** - Create all figures, then save together
5. **Close figures** - Not needed with Plotly (automatically managed)

## Troubleshooting

### Issue: Static export not working

**Problem:** PNG/SVG/PDF export fails
**Solution:** Install kaleido:
```bash
pip install kaleido
```

### Issue: Slow rendering

**Problem:** Chart takes long to render
**Solution:** Sample your data:
```python
sample = df.sample(n=5000)
fig = viz.create_interactive_scatter(sample, 'x', 'y')
```

### Issue: Large HTML files

**Problem:** HTML files are huge
**Solution:** Use CDN for Plotly.js:
```python
viz.save_figure(fig, 'chart.html', include_plotlyjs='cdn')
```

## API Reference

See [plotly_viz.py](simplus_eda/visualizers/plotly_viz.py) for complete API documentation.

### PlotlyVisualizer

```python
class PlotlyVisualizer:
    def __init__(template='plotly_white', color_palette=None,
                 default_height=500, default_width=None)

    def create_interactive_histogram(data, column, bins=30, show_kde=True, title=None)
    def create_interactive_boxplot(data, columns=None, orientation='v', title=None)
    def create_correlation_heatmap(data, method='pearson', columns=None, title=None)
    def create_scatter_matrix(data, columns=None, color_by=None, title=None)
    def create_interactive_scatter(data, x, y, color_by=None, size_by=None,
                                   add_trendline=True, title=None)
    def create_time_series_plot(data, time_col, value_cols, show_range_slider=True,
                               show_buttons=True, title=None)
    def create_distribution_comparison(data, column, group_by, plot_type='violin',
                                      title=None)
    def create_bar_chart(data, x, y=None, orientation='v', sort_by=None,
                        top_n=None, title=None)
    def create_sunburst(data, path, values=None, title=None)
    def create_3d_scatter(data, x, y, z, color_by=None, size_by=None, title=None)
    def create_dashboard(figures, titles=None, rows=None, cols=None)
    def save_figure(fig, path, format='html', **kwargs)
```

## Resources

- **Examples**: [plotly_visualization_examples.py](examples/plotly_visualization_examples.py)
- **Plotly Documentation**: https://plotly.com/python/
- **Gallery**: https://plotly.com/python/plotly-express/

## Summary

The PlotlyVisualizer provides:

✅ **13+ interactive chart types**
✅ **Full customization** - themes, colors, dimensions
✅ **Multiple export formats** - HTML, PNG, SVG, PDF, JSON
✅ **Web-ready** - embed anywhere
✅ **Professional** - publication-quality
✅ **Easy to use** - simple API, powerful results

Start exploring your data with interactive visualizations today!
