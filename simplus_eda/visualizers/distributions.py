"""
Distribution visualization module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import warnings


class DistributionVisualizer:
    """
    Create visualizations for data distributions.

    This class provides methods to visualize the distribution of data using
    various plot types including histograms, boxplots, violin plots, and KDE plots.

    Example:
        >>> visualizer = DistributionVisualizer()
        >>> fig = visualizer.create_histograms(df, columns=['age', 'income'])
        >>> visualizer.save_plot(fig, 'histograms.png')
    """

    def __init__(self, style: str = "whitegrid", palette: str = "husl",
                 figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize the distribution visualizer.

        Args:
            style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
            palette: Color palette ('husl', 'Set2', 'pastel', etc.)
            figsize: Default figure size (width, height)
            dpi: Resolution for saved figures
        """
        self.style = style
        self.palette = palette
        self.default_figsize = figsize
        self.dpi = dpi

        # Set default style
        sns.set_style(style)
        sns.set_palette(palette)

    def create_histograms(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                         bins: Union[int, str] = 'auto', kde: bool = True,
                         figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create histogram plots for numerical columns.

        Args:
            data: Input DataFrame
            columns: Optional list of columns to visualize (default: all numeric)
            bins: Number of bins or binning strategy ('auto', 'sturges', 'scott', etc.)
            kde: Whether to overlay KDE plot
            figsize: Figure size override

        Returns:
            Matplotlib Figure object

        Example:
            >>> fig = visualizer.create_histograms(df, columns=['age', 'salary'])
            >>> plt.show()
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            raise ValueError("No numeric columns found for histogram")

        # Calculate grid dimensions
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=figsize or self.default_figsize)
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, col in enumerate(columns):
            ax = axes[idx]
            values = data[col].dropna()

            if len(values) > 0:
                ax.hist(values, bins=bins, alpha=0.7, color='skyblue',
                       edgecolor='black', linewidth=0.5)

                if kde and len(values) > 1:
                    try:
                        values.plot.kde(ax=ax, color='red', linewidth=2,
                                      secondary_y=True, legend=False)
                    except Exception:
                        pass  # Skip KDE if it fails

                ax.set_title(f'Distribution of {col}', fontsize=10, fontweight='bold')
                ax.set_xlabel(col, fontsize=9)
                ax.set_ylabel('Frequency', fontsize=9)
                ax.grid(True, alpha=0.3)

                # Add statistics
                mean_val = values.mean()
                median_val = values.median()
                ax.axvline(mean_val, color='red', linestyle='--',
                          linewidth=1.5, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--',
                          linewidth=1.5, label=f'Median: {median_val:.2f}')
                ax.legend(fontsize=8)

        # Remove extra subplots
        for idx in range(len(columns), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        return fig

    def create_boxplots(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                       orient: str = 'v', figsize: Optional[Tuple[int, int]] = None,
                       showfliers: bool = True) -> plt.Figure:
        """
        Create boxplot visualizations.

        Args:
            data: Input DataFrame
            columns: Optional list of columns to visualize (default: all numeric)
            orient: Orientation ('v' for vertical, 'h' for horizontal)
            figsize: Figure size override
            showfliers: Whether to show outliers

        Returns:
            Matplotlib Figure object

        Example:
            >>> fig = visualizer.create_boxplots(df, columns=['price', 'quantity'])
            >>> plt.show()
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            raise ValueError("No numeric columns found for boxplot")

        # Calculate grid dimensions
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=figsize or self.default_figsize)
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, col in enumerate(columns):
            ax = axes[idx]
            values = data[col].dropna()

            if len(values) > 0:
                if orient == 'v':
                    ax.boxplot(values, showfliers=showfliers, patch_artist=True,
                             boxprops=dict(facecolor='lightblue', alpha=0.7),
                             medianprops=dict(color='red', linewidth=2))
                    ax.set_ylabel(col, fontsize=9)
                else:
                    ax.boxplot(values, vert=False, showfliers=showfliers,
                             patch_artist=True,
                             boxprops=dict(facecolor='lightblue', alpha=0.7),
                             medianprops=dict(color='red', linewidth=2))
                    ax.set_xlabel(col, fontsize=9)

                ax.set_title(f'Boxplot of {col}', fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y' if orient == 'v' else 'x')

        # Remove extra subplots
        for idx in range(len(columns), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        return fig

    def create_density_plots(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                            figsize: Optional[Tuple[int, int]] = None,
                            fill: bool = True) -> plt.Figure:
        """
        Create density plots (KDE).

        Args:
            data: Input DataFrame
            columns: Optional list of columns to visualize (default: all numeric)
            figsize: Figure size override
            fill: Whether to fill under the curve

        Returns:
            Matplotlib Figure object

        Example:
            >>> fig = visualizer.create_density_plots(df, columns=['age'])
            >>> plt.show()
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            raise ValueError("No numeric columns found for density plot")

        # Calculate grid dimensions
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=figsize or self.default_figsize)
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, col in enumerate(columns):
            ax = axes[idx]
            values = data[col].dropna()

            if len(values) > 1:
                try:
                    sns.kdeplot(data=values, ax=ax, fill=fill, alpha=0.6,
                              linewidth=2, color='steelblue')
                    ax.set_title(f'Density Plot of {col}', fontsize=10, fontweight='bold')
                    ax.set_xlabel(col, fontsize=9)
                    ax.set_ylabel('Density', fontsize=9)
                    ax.grid(True, alpha=0.3)

                    # Add mean and median lines
                    mean_val = values.mean()
                    median_val = values.median()
                    ax.axvline(mean_val, color='red', linestyle='--',
                             linewidth=1.5, label=f'Mean: {mean_val:.2f}')
                    ax.axvline(median_val, color='green', linestyle='--',
                             linewidth=1.5, label=f'Median: {median_val:.2f}')
                    ax.legend(fontsize=8)
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error creating KDE:\n{str(e)}',
                           ha='center', va='center', transform=ax.transAxes)

        # Remove extra subplots
        for idx in range(len(columns), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        return fig

    def create_violin_plots(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                           figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create violin plots for numerical columns.

        Args:
            data: Input DataFrame
            columns: Optional list of columns to visualize (default: all numeric)
            figsize: Figure size override

        Returns:
            Matplotlib Figure object

        Example:
            >>> fig = visualizer.create_violin_plots(df, columns=['score'])
            >>> plt.show()
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            raise ValueError("No numeric columns found for violin plot")

        # Calculate grid dimensions
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=figsize or self.default_figsize)
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, col in enumerate(columns):
            ax = axes[idx]
            values = data[col].dropna()

            if len(values) > 1:
                try:
                    parts = ax.violinplot([values], positions=[0], showmeans=True,
                                        showmedians=True, showextrema=True)

                    # Customize colors
                    for pc in parts['bodies']:
                        pc.set_facecolor('lightblue')
                        pc.set_alpha(0.7)

                    ax.set_title(f'Violin Plot of {col}', fontsize=10, fontweight='bold')
                    ax.set_ylabel(col, fontsize=9)
                    ax.set_xticks([])
                    ax.grid(True, alpha=0.3, axis='y')
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error creating violin plot:\n{str(e)}',
                           ha='center', va='center', transform=ax.transAxes)

        # Remove extra subplots
        for idx in range(len(columns), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        return fig

    def create_qq_plots(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                       figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create Q-Q plots to assess normality.

        Args:
            data: Input DataFrame
            columns: Optional list of columns to visualize (default: all numeric)
            figsize: Figure size override

        Returns:
            Matplotlib Figure object

        Example:
            >>> fig = visualizer.create_qq_plots(df, columns=['values'])
            >>> plt.show()
        """
        from scipy import stats

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            raise ValueError("No numeric columns found for Q-Q plot")

        # Calculate grid dimensions
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=figsize or self.default_figsize)
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, col in enumerate(columns):
            ax = axes[idx]
            values = data[col].dropna()

            if len(values) > 2:
                try:
                    stats.probplot(values, dist="norm", plot=ax)
                    ax.set_title(f'Q-Q Plot of {col}', fontsize=10, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error creating Q-Q plot:\n{str(e)}',
                           ha='center', va='center', transform=ax.transAxes)

        # Remove extra subplots
        for idx in range(len(columns), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        return fig

    def create_distribution_summary(self, data: pd.DataFrame,
                                   column: str,
                                   figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create a comprehensive distribution summary for a single column.

        Includes histogram, boxplot, violin plot, and Q-Q plot.

        Args:
            data: Input DataFrame
            column: Column name to visualize
            figsize: Figure size override

        Returns:
            Matplotlib Figure object

        Example:
            >>> fig = visualizer.create_distribution_summary(df, 'age')
            >>> plt.show()
        """
        from scipy import stats

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        values = data[column].dropna()

        if len(values) == 0:
            raise ValueError(f"Column '{column}' has no valid values")

        fig = plt.figure(figsize=figsize or (14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Histogram with KDE
        ax1 = fig.add_subplot(gs[0, :])
        ax1.hist(values, bins='auto', alpha=0.7, color='skyblue',
                edgecolor='black', linewidth=0.5)
        if len(values) > 1:
            try:
                values.plot.kde(ax=ax1, color='red', linewidth=2,
                              secondary_y=True, legend=False)
            except Exception:
                pass

        mean_val = values.mean()
        median_val = values.median()
        ax1.axvline(mean_val, color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax1.axvline(median_val, color='green', linestyle='--',
                   linewidth=2, label=f'Median: {median_val:.2f}')
        ax1.set_title(f'Distribution Summary: {column}', fontsize=14, fontweight='bold')
        ax1.set_xlabel(column, fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Boxplot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.boxplot(values, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax2.set_title('Boxplot', fontsize=11, fontweight='bold')
        ax2.set_ylabel(column, fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        # Violin plot
        ax3 = fig.add_subplot(gs[1, 1])
        parts = ax3.violinplot([values], positions=[0], showmeans=True,
                              showmedians=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightgreen')
            pc.set_alpha(0.7)
        ax3.set_title('Violin Plot', fontsize=11, fontweight='bold')
        ax3.set_ylabel(column, fontsize=10)
        ax3.set_xticks([])
        ax3.grid(True, alpha=0.3, axis='y')

        # Q-Q plot
        ax4 = fig.add_subplot(gs[2, 0])
        if len(values) > 2:
            stats.probplot(values, dist="norm", plot=ax4)
            ax4.set_title('Q-Q Plot', fontsize=11, fontweight='bold')
            ax4.grid(True, alpha=0.3)

        # Statistics table
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        stats_text = [
            f"Count: {len(values):,}",
            f"Mean: {mean_val:.4f}",
            f"Median: {median_val:.4f}",
            f"Std Dev: {values.std():.4f}",
            f"Min: {values.min():.4f}",
            f"25%: {values.quantile(0.25):.4f}",
            f"75%: {values.quantile(0.75):.4f}",
            f"Max: {values.max():.4f}",
            f"Skewness: {values.skew():.4f}",
            f"Kurtosis: {values.kurtosis():.4f}"
        ]

        y_pos = 0.9
        for stat in stats_text:
            ax5.text(0.1, y_pos, stat, fontsize=10, family='monospace')
            y_pos -= 0.09

        ax5.set_title('Statistics', fontsize=11, fontweight='bold')

        return fig

    @staticmethod
    def save_plot(fig: plt.Figure, filepath: Union[str, Path],
                 dpi: Optional[int] = None, bbox_inches: str = 'tight') -> None:
        """
        Save a plot to file.

        Args:
            fig: Matplotlib Figure object
            filepath: Path to save the figure
            dpi: Resolution (default: use instance default)
            bbox_inches: Bounding box setting

        Example:
            >>> visualizer.save_plot(fig, 'output/distribution.png', dpi=150)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)

    @staticmethod
    def close_all():
        """Close all open matplotlib figures."""
        plt.close('all')
