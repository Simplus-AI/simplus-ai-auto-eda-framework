"""
Relationship and correlation visualization module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path


class RelationshipVisualizer:
    """
    Create visualizations for relationships between features.

    This class provides methods to visualize relationships and correlations
    between variables in a dataset, including correlation heatmaps, scatter
    matrices, pairplots, and individual scatter plots.

    Attributes:
        default_figsize: Default figure size for plots (width, height)
        default_style: Default matplotlib style
        default_palette: Default seaborn color palette
        default_dpi: Default DPI for saved figures

    Example:
        >>> visualizer = RelationshipVisualizer(style='whitegrid', palette='husl')
        >>> fig = visualizer.create_correlation_heatmap(df)
        >>> visualizer.save_plot(fig, 'correlation.png')
    """

    def __init__(
        self,
        style: str = 'whitegrid',
        palette: str = 'husl',
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 100
    ):
        """
        Initialize the RelationshipVisualizer.

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

    def create_correlation_heatmap(
        self,
        data: pd.DataFrame,
        method: str = 'pearson',
        annot: bool = True,
        fmt: str = '.2f',
        cmap: str = 'coolwarm',
        figsize: Optional[Tuple[int, int]] = None,
        mask_diagonal: bool = False,
        mask_upper: bool = False
    ) -> plt.Figure:
        """
        Create correlation heatmap for numerical columns.

        Args:
            data: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            annot: Whether to annotate cells with correlation values
            fmt: Format string for annotations
            cmap: Colormap for heatmap
            figsize: Figure size (width, height), uses default if None
            mask_diagonal: Whether to mask diagonal (self-correlation)
            mask_upper: Whether to mask upper triangle

        Returns:
            matplotlib Figure object

        Example:
            >>> fig = visualizer.create_correlation_heatmap(df, method='spearman')
        """
        # Get numerical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numerical columns for correlation heatmap")

        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr(method=method)

        # Create mask if requested
        mask = None
        if mask_diagonal or mask_upper:
            mask = np.zeros_like(corr_matrix, dtype=bool)
            if mask_diagonal:
                np.fill_diagonal(mask, True)
            if mask_upper:
                mask[np.triu_indices_from(mask, k=1)] = True

        # Create figure
        fig, ax = plt.subplots(figsize=figsize or self.default_figsize)

        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            mask=mask,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title(f'Correlation Heatmap ({method.capitalize()})',
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig

    def create_scatter_matrix(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        alpha: float = 0.5,
        diagonal: str = 'hist'
    ) -> plt.Figure:
        """
        Create scatter plot matrix for numerical columns.

        Args:
            data: Input DataFrame
            columns: Optional list of columns to include (uses all numeric if None)
            figsize: Figure size (width, height), uses default if None
            alpha: Transparency of scatter points (0-1)
            diagonal: What to plot on diagonal ('hist', 'kde', or None)

        Returns:
            matplotlib Figure object

        Example:
            >>> fig = visualizer.create_scatter_matrix(df, columns=['age', 'income', 'score'])
        """
        # Get columns to plot
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Validate columns are numeric
            for col in columns:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
                if not np.issubdtype(data[col].dtype, np.number):
                    raise ValueError(f"Column '{col}' is not numerical")

        if len(columns) < 2:
            raise ValueError("Need at least 2 columns for scatter matrix")

        n_cols = len(columns)

        # Create figure
        fig_size = figsize or (min(3 * n_cols, 15), min(3 * n_cols, 15))
        fig, axes = plt.subplots(n_cols, n_cols, figsize=fig_size)

        # Ensure axes is 2D array
        if n_cols == 1:
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape(-1, 1)

        # Plot each combination
        for i, col_y in enumerate(columns):
            for j, col_x in enumerate(columns):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: histogram or KDE
                    if diagonal == 'hist':
                        ax.hist(data[col_x].dropna(), bins=20, alpha=0.7,
                               color='steelblue', edgecolor='black')
                    elif diagonal == 'kde':
                        data[col_x].dropna().plot(kind='kde', ax=ax, color='steelblue')
                    else:
                        ax.text(0.5, 0.5, col_x, ha='center', va='center',
                               fontsize=10, transform=ax.transAxes)
                else:
                    # Off-diagonal: scatter plot
                    ax.scatter(data[col_x], data[col_y], alpha=alpha, s=10,
                              color='steelblue')

                # Set labels only on edges
                if i == n_cols - 1:
                    ax.set_xlabel(col_x, fontsize=9)
                else:
                    ax.set_xlabel('')
                    ax.set_xticklabels([])

                if j == 0:
                    ax.set_ylabel(col_y, fontsize=9)
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])

                ax.tick_params(labelsize=7)

        plt.suptitle('Scatter Plot Matrix', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig

    def create_pairplot(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        kind: str = 'scatter',
        diag_kind: str = 'hist',
        corner: bool = False
    ) -> plt.Figure:
        """
        Create pairplot visualization using seaborn.

        Args:
            data: Input DataFrame
            columns: Optional list of columns to include
            hue: Optional column for color coding
            kind: Kind of plot for off-diagonal ('scatter', 'kde', 'hist', 'reg')
            diag_kind: Kind of plot for diagonal ('hist', 'kde')
            corner: Whether to plot only lower triangle

        Returns:
            matplotlib Figure object (from seaborn PairGrid)

        Example:
            >>> fig = visualizer.create_pairplot(df, hue='category', kind='reg')
        """
        # Get columns to plot
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Validate hue column
        if hue is not None:
            if hue not in data.columns:
                raise ValueError(f"Hue column '{hue}' not found in DataFrame")
            # Include hue in data but not in plot columns if it's not numeric
            if hue not in columns and hue in data.columns:
                columns_to_plot = columns
            else:
                columns_to_plot = columns
        else:
            columns_to_plot = columns

        if len(columns_to_plot) < 2:
            raise ValueError("Need at least 2 numerical columns for pairplot")

        # Create subset of data
        plot_data = data[columns_to_plot + ([hue] if hue and hue not in columns_to_plot else [])]

        # Create pairplot with appropriate kwargs based on kind
        plot_kws = {'s': 20} if kind in ['scatter', 'kde', 'hist'] else {}
        if kind in ['scatter', 'kde', 'hist']:
            plot_kws['alpha'] = 0.6

        diag_kws = {}
        if diag_kind == 'hist':
            diag_kws = {'alpha': 0.7, 'bins': 20}
        elif diag_kind == 'kde':
            diag_kws = {'alpha': 0.7}

        pairplot = sns.pairplot(
            plot_data,
            vars=columns_to_plot,
            hue=hue,
            kind=kind,
            diag_kind=diag_kind,
            corner=corner,
            plot_kws=plot_kws,
            diag_kws=diag_kws
        )

        pairplot.fig.suptitle('Pairplot Visualization',
                             fontsize=14, fontweight='bold', y=1.0)

        return pairplot.fig

    def create_scatter_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        hue: Optional[str] = None,
        size: Optional[str] = None,
        add_regression: bool = True,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create individual scatter plot with optional regression line.

        Args:
            data: Input DataFrame
            x: Column name for x-axis
            y: Column name for y-axis
            hue: Optional column for color coding
            size: Optional column for size coding
            add_regression: Whether to add regression line
            figsize: Figure size (width, height), uses default if None

        Returns:
            matplotlib Figure object

        Example:
            >>> fig = visualizer.create_scatter_plot(df, 'age', 'income', hue='gender')
        """
        # Validate columns
        for col in [x, y]:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

        if hue and hue not in data.columns:
            raise ValueError(f"Hue column '{hue}' not found in DataFrame")

        if size and size not in data.columns:
            raise ValueError(f"Size column '{size}' not found in DataFrame")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize or self.default_figsize)

        # Create scatter plot
        if hue:
            # Use seaborn for hue support
            sns.scatterplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                size=size,
                alpha=0.6,
                ax=ax
            )
        else:
            # Simple matplotlib scatter
            sizes = data[size] * 20 if size else 50
            ax.scatter(data[x], data[y], alpha=0.6, s=sizes, color='steelblue')

        # Add regression line
        if add_regression:
            # Remove NaN values
            valid_data = data[[x, y]].dropna()
            if len(valid_data) > 1:
                z = np.polyfit(valid_data[x], valid_data[y], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid_data[x].min(), valid_data[x].max(), 100)
                ax.plot(x_line, p(x_line), "r--", linewidth=2,
                       label=f'y = {z[0]:.2f}x + {z[1]:.2f}', alpha=0.8)

                # Calculate and display R²
                from scipy import stats
                _, _, r_value, _, _ = stats.linregress(valid_data[x], valid_data[y])
                ax.text(0.05, 0.95, f'R² = {r_value**2:.3f}',
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)
        ax.set_title(f'{y} vs {x}', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)

        if add_regression or hue:
            ax.legend()

        plt.tight_layout()
        return fig

    def create_correlation_pairs(
        self,
        data: pd.DataFrame,
        threshold: float = 0.7,
        method: str = 'pearson',
        max_pairs: int = 6,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create scatter plots for highly correlated variable pairs.

        Args:
            data: Input DataFrame
            threshold: Minimum absolute correlation to include (0-1)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            max_pairs: Maximum number of pairs to plot
            figsize: Figure size (width, height), auto-calculated if None

        Returns:
            matplotlib Figure object

        Example:
            >>> fig = visualizer.create_correlation_pairs(df, threshold=0.8)
        """
        # Get numerical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numerical columns for correlation pairs")

        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr(method=method)

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })

        if not high_corr_pairs:
            raise ValueError(f"No correlation pairs found with |correlation| >= {threshold}")

        # Sort by absolute correlation
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        high_corr_pairs = high_corr_pairs[:max_pairs]

        # Calculate grid dimensions
        n_pairs = len(high_corr_pairs)
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols

        # Create figure
        fig_width = n_cols * 5
        fig_height = n_rows * 4
        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=figsize or (fig_width, fig_height))

        # Ensure axes is iterable
        if n_pairs == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_pairs > 1 else [axes]

        # Plot each pair
        for idx, pair in enumerate(high_corr_pairs):
            ax = axes[idx]

            x_col = pair['var1']
            y_col = pair['var2']
            corr = pair['correlation']

            # Scatter plot
            ax.scatter(data[x_col], data[y_col], alpha=0.5, s=20, color='steelblue')

            # Add regression line
            valid_data = data[[x_col, y_col]].dropna()
            if len(valid_data) > 1:
                z = np.polyfit(valid_data[x_col], valid_data[y_col], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid_data[x_col].min(),
                                    valid_data[x_col].max(), 100)
                ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)

            ax.set_xlabel(x_col, fontsize=10)
            ax.set_ylabel(y_col, fontsize=10)
            ax.set_title(f'r = {corr:.3f}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Remove extra subplots
        for idx in range(n_pairs, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(f'Highly Correlated Pairs (|r| ≥ {threshold})',
                    fontsize=14, fontweight='bold', y=1.0)
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
            >>> fig = visualizer.create_correlation_heatmap(df)
            >>> RelationshipVisualizer.save_plot(fig, 'correlation.png', dpi=300)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, **kwargs)

    @staticmethod
    def close_all() -> None:
        """
        Close all matplotlib figures to free memory.

        Example:
            >>> RelationshipVisualizer.close_all()
        """
        plt.close('all')
