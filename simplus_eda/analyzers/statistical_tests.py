"""
Advanced statistical tests for the EDA framework.

This module provides comprehensive statistical testing capabilities including:
- ANOVA and post-hoc tests (Tukey HSD, Bonferroni, etc.)
- Non-parametric tests (Kruskal-Wallis, Mann-Whitney U)
- Categorical relationship tests (Chi-square, Fisher's exact)
- Variance homogeneity tests (Levene's, Bartlett's)
- Granger causality for time series
- Multiple comparison corrections

Features:
- Automatic assumption checking
- Effect size calculations
- Multiple comparison corrections
- Detailed interpretation and recommendations
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

from simplus_eda.logging_config import get_logger
from simplus_eda.exceptions import AnalysisError
from simplus_eda.progress import ProgressTracker

logger = get_logger(__name__)

# Try to import optional dependencies
try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.contingency_tables import Table2x2
    from statsmodels.tsa.stattools import grangercausalitytests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available. Install with: pip install statsmodels")

try:
    from scikit_posthocs import posthoc_dunn, posthoc_conover
    POSTHOCS_AVAILABLE = True
except ImportError:
    POSTHOCS_AVAILABLE = False


# ============================================================================
# Effect Size Calculators
# ============================================================================

class EffectSize:
    """Calculate various effect size measures."""

    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.

        Args:
            group1: First group data
            group2: Second group data

        Returns:
            Cohen's d value
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    @staticmethod
    def eta_squared(groups: List[np.ndarray]) -> float:
        """
        Calculate eta-squared (η²) effect size for ANOVA.

        Args:
            groups: List of group data arrays

        Returns:
            Eta-squared value (0-1)
        """
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)

        # Between-group sum of squares
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)

        # Total sum of squares
        ss_total = np.sum((all_data - grand_mean) ** 2)

        if ss_total == 0:
            return 0.0

        return ss_between / ss_total

    @staticmethod
    def cramers_v(chi2: float, n: int, min_dim: int) -> float:
        """
        Calculate Cramér's V effect size for chi-square test.

        Args:
            chi2: Chi-square statistic
            n: Sample size
            min_dim: min(rows-1, cols-1)

        Returns:
            Cramér's V value (0-1)
        """
        if n == 0 or min_dim == 0:
            return 0.0

        return np.sqrt(chi2 / (n * min_dim))

    @staticmethod
    def interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d value."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"

    @staticmethod
    def interpret_eta_squared(eta2: float) -> str:
        """Interpret eta-squared value."""
        if eta2 < 0.01:
            return "Negligible"
        elif eta2 < 0.06:
            return "Small"
        elif eta2 < 0.14:
            return "Medium"
        else:
            return "Large"

    @staticmethod
    def interpret_cramers_v(v: float) -> str:
        """Interpret Cramér's V value."""
        if v < 0.1:
            return "Negligible"
        elif v < 0.3:
            return "Small"
        elif v < 0.5:
            return "Medium"
        else:
            return "Large"


# ============================================================================
# ANOVA Tests
# ============================================================================

class ANOVATests:
    """Analysis of Variance tests."""

    @staticmethod
    def one_way_anova(
        data: pd.DataFrame,
        value_col: str,
        group_col: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        One-way ANOVA test.

        Args:
            data: DataFrame with data
            value_col: Name of value column
            group_col: Name of grouping column
            alpha: Significance level

        Returns:
            ANOVA results with interpretation
        """
        # Extract groups
        groups = [group[value_col].dropna().values
                  for name, group in data.groupby(group_col)]

        # Remove empty groups
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            raise AnalysisError(
                "Need at least 2 groups for ANOVA",
                details={'n_groups': len(groups)}
            )

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        # Calculate effect size
        eta2 = EffectSize.eta_squared(groups)

        # Check assumptions
        assumptions = ANOVATests._check_anova_assumptions(groups)

        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'effect_size': {
                'eta_squared': eta2,
                'interpretation': EffectSize.interpret_eta_squared(eta2)
            },
            'n_groups': len(groups),
            'group_sizes': [len(g) for g in groups],
            'assumptions': assumptions,
            'conclusion': ANOVATests._interpret_anova(p_value, alpha, eta2),
            'recommendations': ANOVATests._get_anova_recommendations(
                p_value, alpha, assumptions
            )
        }

    @staticmethod
    def two_way_anova(
        data: pd.DataFrame,
        value_col: str,
        factor1_col: str,
        factor2_col: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Two-way ANOVA test.

        Args:
            data: DataFrame with data
            value_col: Name of value column
            factor1_col: First factor column
            factor2_col: Second factor column
            alpha: Significance level

        Returns:
            Two-way ANOVA results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for two-way ANOVA")

        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm

        # Create formula
        formula = f'{value_col} ~ C({factor1_col}) + C({factor2_col}) + C({factor1_col}):C({factor2_col})'

        # Fit model
        model = ols(formula, data=data).fit()

        # Perform ANOVA
        anova_table = anova_lm(model, typ=2)

        return {
            'anova_table': anova_table,
            'factor1_effect': {
                'f_statistic': anova_table.loc[f'C({factor1_col})', 'F'],
                'p_value': anova_table.loc[f'C({factor1_col})', 'PR(>F)'],
                'significant': anova_table.loc[f'C({factor1_col})', 'PR(>F)'] < alpha
            },
            'factor2_effect': {
                'f_statistic': anova_table.loc[f'C({factor2_col})', 'F'],
                'p_value': anova_table.loc[f'C({factor2_col})', 'PR(>F)'],
                'significant': anova_table.loc[f'C({factor2_col})', 'PR(>F)'] < alpha
            },
            'interaction_effect': {
                'f_statistic': anova_table.loc[f'C({factor1_col}):C({factor2_col})', 'F'],
                'p_value': anova_table.loc[f'C({factor1_col}):C({factor2_col})', 'PR(>F)'],
                'significant': anova_table.loc[f'C({factor1_col}):C({factor2_col})', 'PR(>F)'] < alpha
            }
        }

    @staticmethod
    def _check_anova_assumptions(groups: List[np.ndarray]) -> Dict[str, Any]:
        """Check ANOVA assumptions."""
        # 1. Normality (Shapiro-Wilk for each group)
        normality_tests = []
        for i, group in enumerate(groups):
            if len(group) >= 3:
                _, p = stats.shapiro(group)
                normality_tests.append({
                    'group': i,
                    'p_value': p,
                    'normal': p > 0.05
                })

        # 2. Homogeneity of variance (Levene's test)
        levene_stat, levene_p = stats.levene(*groups)

        return {
            'normality': {
                'tests': normality_tests,
                'all_normal': all(t['normal'] for t in normality_tests)
            },
            'homogeneity_of_variance': {
                'levene_statistic': levene_stat,
                'p_value': levene_p,
                'homogeneous': levene_p > 0.05
            }
        }

    @staticmethod
    def _interpret_anova(p_value: float, alpha: float, eta2: float) -> str:
        """Interpret ANOVA results."""
        if p_value < alpha:
            effect = EffectSize.interpret_eta_squared(eta2)
            return f"Significant difference between groups (p={p_value:.4f}). Effect size: {effect} (η²={eta2:.4f})"
        else:
            return f"No significant difference between groups (p={p_value:.4f})"

    @staticmethod
    def _get_anova_recommendations(
        p_value: float,
        alpha: float,
        assumptions: Dict[str, Any]
    ) -> List[str]:
        """Get recommendations based on ANOVA results."""
        recommendations = []

        if p_value < alpha:
            recommendations.append("Significant result - perform post-hoc tests (e.g., Tukey HSD)")

        if not assumptions['normality']['all_normal']:
            recommendations.append(
                "Normality assumption violated - consider Kruskal-Wallis test instead"
            )

        if not assumptions['homogeneity_of_variance']['homogeneous']:
            recommendations.append(
                "Homogeneity of variance violated - consider Welch's ANOVA or Kruskal-Wallis"
            )

        return recommendations


# ============================================================================
# Post-hoc Tests
# ============================================================================

class PostHocTests:
    """Post-hoc pairwise comparison tests."""

    @staticmethod
    def tukey_hsd(
        data: pd.DataFrame,
        value_col: str,
        group_col: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Tukey's Honestly Significant Difference test.

        Args:
            data: DataFrame with data
            value_col: Name of value column
            group_col: Name of grouping column
            alpha: Significance level

        Returns:
            Tukey HSD results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for Tukey HSD")

        # Perform Tukey HSD
        tukey = pairwise_tukeyhsd(
            endog=data[value_col],
            groups=data[group_col],
            alpha=alpha
        )

        # Parse results
        results = []
        for i in range(len(tukey.summary().data) - 1):  # Skip header
            row = tukey.summary().data[i + 1]
            results.append({
                'group1': row[0],
                'group2': row[1],
                'mean_diff': float(row[2]),
                'lower_ci': float(row[3]),
                'upper_ci': float(row[4]),
                'p_adj': float(row[5]),
                'reject': row[6]
            })

        # Count significant pairs
        n_significant = sum(1 for r in results if r['reject'])

        return {
            'pairwise_comparisons': results,
            'n_comparisons': len(results),
            'n_significant': n_significant,
            'alpha': alpha,
            'summary': tukey.summary()
        }

    @staticmethod
    def bonferroni_correction(
        p_values: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Bonferroni correction for multiple comparisons.

        Args:
            p_values: List of p-values
            alpha: Family-wise error rate

        Returns:
            Corrected results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for Bonferroni correction")

        reject, p_corrected, _, _ = multipletests(
            p_values,
            alpha=alpha,
            method='bonferroni'
        )

        return {
            'original_p_values': p_values,
            'corrected_p_values': p_corrected.tolist(),
            'reject': reject.tolist(),
            'n_tests': len(p_values),
            'n_significant': sum(reject),
            'alpha': alpha,
            'corrected_alpha': alpha / len(p_values)
        }

    @staticmethod
    def holm_correction(
        p_values: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Holm-Bonferroni correction (more powerful than Bonferroni).

        Args:
            p_values: List of p-values
            alpha: Family-wise error rate

        Returns:
            Corrected results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for Holm correction")

        reject, p_corrected, _, _ = multipletests(
            p_values,
            alpha=alpha,
            method='holm'
        )

        return {
            'original_p_values': p_values,
            'corrected_p_values': p_corrected.tolist(),
            'reject': reject.tolist(),
            'n_tests': len(p_values),
            'n_significant': sum(reject),
            'alpha': alpha
        }


# ============================================================================
# Non-Parametric Tests
# ============================================================================

class NonParametricTests:
    """Non-parametric statistical tests."""

    @staticmethod
    def mann_whitney_u(
        group1: Union[pd.Series, np.ndarray],
        group2: Union[pd.Series, np.ndarray],
        alternative: str = 'two-sided',
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Mann-Whitney U test (Wilcoxon rank-sum test).

        Non-parametric alternative to independent t-test.

        Args:
            group1: First group data
            group2: Second group data
            alternative: 'two-sided', 'less', or 'greater'
            alpha: Significance level

        Returns:
            Mann-Whitney U test results
        """
        g1 = np.asarray(group1)
        g2 = np.asarray(group2)

        # Perform test
        statistic, p_value = stats.mannwhitneyu(
            g1, g2,
            alternative=alternative
        )

        # Calculate effect size (rank-biserial correlation)
        r = 1 - (2 * statistic) / (len(g1) * len(g2))

        return {
            'u_statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'effect_size': {
                'r': r,
                'interpretation': NonParametricTests._interpret_rank_biserial(abs(r))
            },
            'alternative': alternative,
            'conclusion': NonParametricTests._interpret_mann_whitney(
                p_value, alpha, g1, g2, alternative
            )
        }

    @staticmethod
    def kruskal_wallis(
        data: pd.DataFrame,
        value_col: str,
        group_col: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Kruskal-Wallis H test.

        Non-parametric alternative to one-way ANOVA.

        Args:
            data: DataFrame with data
            value_col: Name of value column
            group_col: Name of grouping column
            alpha: Significance level

        Returns:
            Kruskal-Wallis test results
        """
        # Extract groups
        groups = [group[value_col].dropna().values
                  for name, group in data.groupby(group_col)]

        # Remove empty groups
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            raise AnalysisError(
                "Need at least 2 groups for Kruskal-Wallis",
                details={'n_groups': len(groups)}
            )

        # Perform test
        h_stat, p_value = stats.kruskal(*groups)

        # Calculate effect size (epsilon-squared)
        n_total = sum(len(g) for g in groups)
        k = len(groups)
        epsilon2 = (h_stat - k + 1) / (n_total - k)

        return {
            'h_statistic': h_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'effect_size': {
                'epsilon_squared': epsilon2,
                'interpretation': NonParametricTests._interpret_epsilon_squared(epsilon2)
            },
            'n_groups': k,
            'group_sizes': [len(g) for g in groups],
            'conclusion': NonParametricTests._interpret_kruskal(p_value, alpha),
            'recommendations': NonParametricTests._get_kruskal_recommendations(p_value, alpha)
        }

    @staticmethod
    def wilcoxon_signed_rank(
        x: Union[pd.Series, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        alternative: str = 'two-sided',
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Wilcoxon signed-rank test.

        Non-parametric alternative to paired t-test.

        Args:
            x: First sample (or differences if y is None)
            y: Second sample (optional)
            alternative: 'two-sided', 'less', or 'greater'
            alpha: Significance level

        Returns:
            Wilcoxon test results
        """
        if y is not None:
            x = np.asarray(x)
            y = np.asarray(y)
        else:
            x = np.asarray(x)
            y = None

        # Perform test
        statistic, p_value = stats.wilcoxon(x, y, alternative=alternative)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'alternative': alternative,
            'conclusion': f"{'Significant' if p_value < alpha else 'No significant'} difference (p={p_value:.4f})"
        }

    @staticmethod
    def _interpret_rank_biserial(r: float) -> str:
        """Interpret rank-biserial correlation."""
        if r < 0.1:
            return "Negligible"
        elif r < 0.3:
            return "Small"
        elif r < 0.5:
            return "Medium"
        else:
            return "Large"

    @staticmethod
    def _interpret_epsilon_squared(eps2: float) -> str:
        """Interpret epsilon-squared effect size."""
        if eps2 < 0.01:
            return "Negligible"
        elif eps2 < 0.08:
            return "Small"
        elif eps2 < 0.26:
            return "Medium"
        else:
            return "Large"

    @staticmethod
    def _interpret_mann_whitney(
        p_value: float,
        alpha: float,
        g1: np.ndarray,
        g2: np.ndarray,
        alternative: str
    ) -> str:
        """Interpret Mann-Whitney U test results."""
        if p_value >= alpha:
            return f"No significant difference between groups (p={p_value:.4f})"

        median1 = np.median(g1)
        median2 = np.median(g2)

        if alternative == 'two-sided':
            if median1 > median2:
                return f"Group 1 significantly higher than Group 2 (p={p_value:.4f})"
            else:
                return f"Group 2 significantly higher than Group 1 (p={p_value:.4f})"
        elif alternative == 'less':
            return f"Group 1 significantly less than Group 2 (p={p_value:.4f})"
        else:  # greater
            return f"Group 1 significantly greater than Group 2 (p={p_value:.4f})"

    @staticmethod
    def _interpret_kruskal(p_value: float, alpha: float) -> str:
        """Interpret Kruskal-Wallis results."""
        if p_value < alpha:
            return f"Significant difference between groups (p={p_value:.4f})"
        else:
            return f"No significant difference between groups (p={p_value:.4f})"

    @staticmethod
    def _get_kruskal_recommendations(p_value: float, alpha: float) -> List[str]:
        """Get recommendations for Kruskal-Wallis results."""
        if p_value < alpha:
            return [
                "Significant result found",
                "Perform post-hoc pairwise comparisons (e.g., Dunn's test)",
                "Consider effect size (epsilon-squared) for practical significance"
            ]
        else:
            return [
                "No significant differences found",
                "Groups appear to have similar distributions"
            ]


# ============================================================================
# Categorical Tests
# ============================================================================

class CategoricalTests:
    """Tests for categorical data relationships."""

    @staticmethod
    def chi_square(
        data: pd.DataFrame,
        var1: str,
        var2: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Chi-square test of independence.

        Args:
            data: DataFrame with data
            var1: First categorical variable
            var2: Second categorical variable
            alpha: Significance level

        Returns:
            Chi-square test results
        """
        # Create contingency table
        contingency = pd.crosstab(data[var1], data[var2])

        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        # Calculate effect size (Cramér's V)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = EffectSize.cramers_v(chi2, n, min_dim)

        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < alpha,
            'contingency_table': contingency,
            'expected_frequencies': expected,
            'effect_size': {
                'cramers_v': cramers_v,
                'interpretation': EffectSize.interpret_cramers_v(cramers_v)
            },
            'conclusion': CategoricalTests._interpret_chi_square(p_value, alpha, cramers_v),
            'assumptions': CategoricalTests._check_chi_square_assumptions(expected)
        }

    @staticmethod
    def fishers_exact(
        data: pd.DataFrame,
        var1: str,
        var2: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Fisher's exact test for 2x2 contingency tables.

        Args:
            data: DataFrame with data
            var1: First categorical variable
            var2: Second categorical variable
            alpha: Significance level

        Returns:
            Fisher's exact test results
        """
        # Create contingency table
        contingency = pd.crosstab(data[var1], data[var2])

        if contingency.shape != (2, 2):
            raise AnalysisError(
                "Fisher's exact test requires 2x2 table",
                details={'shape': contingency.shape}
            )

        # Perform test
        odds_ratio, p_value = stats.fisher_exact(contingency)

        return {
            'odds_ratio': odds_ratio,
            'p_value': p_value,
            'significant': p_value < alpha,
            'contingency_table': contingency,
            'conclusion': f"{'Significant' if p_value < alpha else 'No significant'} association (p={p_value:.4f}, OR={odds_ratio:.2f})"
        }

    @staticmethod
    def mcnemar_test(
        data: pd.DataFrame,
        var1: str,
        var2: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        McNemar's test for paired nominal data.

        Args:
            data: DataFrame with paired observations
            var1: First variable
            var2: Second variable
            alpha: Significance level

        Returns:
            McNemar's test results
        """
        # Create 2x2 table
        contingency = pd.crosstab(data[var1], data[var2])

        if contingency.shape != (2, 2):
            raise AnalysisError(
                "McNemar's test requires 2x2 table",
                details={'shape': contingency.shape}
            )

        # Get discordant pairs
        b = contingency.iloc[0, 1]  # Changed from cat1 to cat2
        c = contingency.iloc[1, 0]  # Changed from cat2 to cat1

        # Perform test
        statistic, p_value = stats.mcnemar([[contingency.iloc[0, 0], b],
                                            [c, contingency.iloc[1, 1]]])

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'b': b,  # Discordant pair 1
            'c': c,  # Discordant pair 2
            'conclusion': f"{'Significant' if p_value < alpha else 'No significant'} change (p={p_value:.4f})"
        }

    @staticmethod
    def _interpret_chi_square(p_value: float, alpha: float, cramers_v: float) -> str:
        """Interpret chi-square test results."""
        if p_value < alpha:
            effect = EffectSize.interpret_cramers_v(cramers_v)
            return f"Significant association (p={p_value:.4f}). Effect size: {effect} (V={cramers_v:.4f})"
        else:
            return f"No significant association (p={p_value:.4f})"

    @staticmethod
    def _check_chi_square_assumptions(expected: np.ndarray) -> Dict[str, Any]:
        """Check chi-square test assumptions."""
        min_expected = expected.min()
        pct_below_5 = (expected < 5).sum() / expected.size * 100

        return {
            'min_expected_frequency': min_expected,
            'percent_below_5': pct_below_5,
            'assumption_met': min_expected >= 5 and pct_below_5 < 20,
            'recommendation': "Use Fisher's exact test" if min_expected < 5 else "Chi-square is appropriate"
        }


# ============================================================================
# Variance Tests
# ============================================================================

class VarianceTests:
    """Tests for variance homogeneity."""

    @staticmethod
    def levene_test(
        data: pd.DataFrame,
        value_col: str,
        group_col: str,
        center: str = 'median',
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Levene's test for homogeneity of variance.

        Args:
            data: DataFrame with data
            value_col: Name of value column
            group_col: Name of grouping column
            center: 'mean', 'median', or 'trimmed'
            alpha: Significance level

        Returns:
            Levene's test results
        """
        # Extract groups
        groups = [group[value_col].dropna().values
                  for name, group in data.groupby(group_col)]

        # Remove empty groups
        groups = [g for g in groups if len(g) > 0]

        # Perform test
        statistic, p_value = stats.levene(*groups, center=center)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'homogeneous': p_value > alpha,
            'center': center,
            'conclusion': VarianceTests._interpret_levene(p_value, alpha),
            'recommendations': VarianceTests._get_levene_recommendations(p_value, alpha)
        }

    @staticmethod
    def bartlett_test(
        data: pd.DataFrame,
        value_col: str,
        group_col: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Bartlett's test for homogeneity of variance.

        More sensitive to normality than Levene's test.

        Args:
            data: DataFrame with data
            value_col: Name of value column
            group_col: Name of grouping column
            alpha: Significance level

        Returns:
            Bartlett's test results
        """
        # Extract groups
        groups = [group[value_col].dropna().values
                  for name, group in data.groupby(group_col)]

        # Remove empty groups
        groups = [g for g in groups if len(g) > 0]

        # Perform test
        statistic, p_value = stats.bartlett(*groups)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'homogeneous': p_value > alpha,
            'conclusion': VarianceTests._interpret_bartlett(p_value, alpha),
            'note': "Bartlett's test is sensitive to non-normality. Use Levene's test if data is not normal."
        }

    @staticmethod
    def _interpret_levene(p_value: float, alpha: float) -> str:
        """Interpret Levene's test results."""
        if p_value > alpha:
            return f"Variances are homogeneous (p={p_value:.4f})"
        else:
            return f"Variances are not homogeneous (p={p_value:.4f})"

    @staticmethod
    def _interpret_bartlett(p_value: float, alpha: float) -> str:
        """Interpret Bartlett's test results."""
        if p_value > alpha:
            return f"Variances are homogeneous (p={p_value:.4f})"
        else:
            return f"Variances are not homogeneous (p={p_value:.4f})"

    @staticmethod
    def _get_levene_recommendations(p_value: float, alpha: float) -> List[str]:
        """Get recommendations based on Levene's test."""
        if p_value > alpha:
            return [
                "Variances are homogeneous",
                "ANOVA assumptions met (for this aspect)",
                "Can proceed with standard ANOVA"
            ]
        else:
            return [
                "Variances are not homogeneous",
                "Consider Welch's ANOVA (doesn't assume equal variances)",
                "Or use non-parametric Kruskal-Wallis test"
            ]


# ============================================================================
# Granger Causality
# ============================================================================

class GrangerCausalityTest:
    """Granger causality test for time series."""

    @staticmethod
    def test_granger_causality(
        data: pd.DataFrame,
        caused: str,
        causing: str,
        max_lag: int = 5,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Test if one time series Granger-causes another.

        Args:
            data: DataFrame with time series data
            caused: Name of caused variable (dependent)
            causing: Name of causing variable (independent)
            max_lag: Maximum number of lags to test
            alpha: Significance level

        Returns:
            Granger causality test results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for Granger causality")

        # Prepare data
        ts_data = data[[caused, causing]].dropna()

        if len(ts_data) < max_lag + 10:
            raise AnalysisError(
                f"Need at least {max_lag + 10} observations",
                details={'length': len(ts_data)}
            )

        # Perform test
        try:
            results = grangercausalitytests(
                ts_data,
                max_lag,
                verbose=False
            )
        except Exception as e:
            raise AnalysisError(
                f"Granger causality test failed: {e}",
                details={'caused': caused, 'causing': causing}
            )

        # Extract results for each lag
        lag_results = []
        for lag in range(1, max_lag + 1):
            test_stats = results[lag][0]

            lag_results.append({
                'lag': lag,
                'ssr_ftest': {
                    'f_statistic': test_stats['ssr_ftest'][0],
                    'p_value': test_stats['ssr_ftest'][1],
                    'significant': test_stats['ssr_ftest'][1] < alpha
                },
                'ssr_chi2test': {
                    'chi2_statistic': test_stats['ssr_chi2test'][0],
                    'p_value': test_stats['ssr_chi2test'][1],
                    'significant': test_stats['ssr_chi2test'][1] < alpha
                },
                'lrtest': {
                    'lr_statistic': test_stats['lrtest'][0],
                    'p_value': test_stats['lrtest'][1],
                    'significant': test_stats['lrtest'][1] < alpha
                }
            })

        # Determine if Granger causality exists
        granger_causes = any(
            r['ssr_ftest']['significant'] for r in lag_results
        )

        # Find best lag
        if granger_causes:
            best_lag = min(
                (r for r in lag_results if r['ssr_ftest']['significant']),
                key=lambda r: r['ssr_ftest']['p_value']
            )['lag']
        else:
            best_lag = None

        return {
            'granger_causes': granger_causes,
            'best_lag': best_lag,
            'lag_results': lag_results,
            'conclusion': GrangerCausalityTest._interpret_granger(
                granger_causes, causing, caused, best_lag
            ),
            'recommendations': GrangerCausalityTest._get_granger_recommendations(
                granger_causes
            )
        }

    @staticmethod
    def _interpret_granger(
        causes: bool,
        causing: str,
        caused: str,
        best_lag: Optional[int]
    ) -> str:
        """Interpret Granger causality results."""
        if causes:
            return f"'{causing}' Granger-causes '{caused}' (best lag: {best_lag})"
        else:
            return f"'{causing}' does NOT Granger-cause '{caused}'"

    @staticmethod
    def _get_granger_recommendations(causes: bool) -> List[str]:
        """Get recommendations for Granger causality."""
        if causes:
            return [
                "Granger causality detected",
                "Past values of causing variable help predict caused variable",
                "Consider using causing variable in forecasting models",
                "Note: Granger causality ≠ true causality"
            ]
        else:
            return [
                "No Granger causality detected",
                "Past values of causing variable don't help predict caused variable",
                "Variables may still be correlated contemporaneously"
            ]


# ============================================================================
# Main Statistical Tests Manager
# ============================================================================

class StatisticalTestsManager:
    """
    Comprehensive statistical testing manager.

    Example:
        >>> manager = StatisticalTestsManager()
        >>> results = manager.compare_groups(data, value_col='score', group_col='treatment')
    """

    def __init__(self, alpha: float = 0.05, verbose: bool = False):
        """
        Initialize test manager.

        Args:
            alpha: Default significance level
            verbose: Verbose output
        """
        self.alpha = alpha
        self.verbose = verbose

    def compare_groups(
        self,
        data: pd.DataFrame,
        value_col: str,
        group_col: str,
        parametric: Optional[bool] = None,
        post_hoc: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive group comparison with automatic test selection.

        Args:
            data: DataFrame with data
            value_col: Value column name
            group_col: Grouping column name
            parametric: Force parametric (True) or non-parametric (False), None=auto
            post_hoc: Perform post-hoc tests if significant

        Returns:
            Complete comparison results
        """
        n_groups = data[group_col].nunique()

        results = {
            'n_groups': n_groups,
            'group_sizes': data.groupby(group_col)[value_col].count().to_dict()
        }

        # Decide on test type
        if parametric is None:
            # Check normality and homogeneity
            groups = [group[value_col].dropna().values
                     for name, group in data.groupby(group_col)]
            normality_ok = all(
                stats.shapiro(g)[1] > 0.05 if len(g) >= 3 else True
                for g in groups
            )
            homogeneity_ok = stats.levene(*groups)[1] > 0.05

            parametric = normality_ok and homogeneity_ok

        # Perform appropriate test
        if n_groups == 2:
            if parametric:
                # Independent t-test
                groups = [group[value_col].dropna().values
                         for name, group in data.groupby(group_col)]
                t_stat, p_val = stats.ttest_ind(*groups)

                results['test_used'] = 'Independent t-test'
                results['test_result'] = {
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'significant': p_val < self.alpha,
                    'effect_size': {
                        'cohens_d': EffectSize.cohens_d(*groups),
                        'interpretation': EffectSize.interpret_cohens_d(
                            EffectSize.cohens_d(*groups)
                        )
                    }
                }
            else:
                # Mann-Whitney U
                groups = [group[value_col].dropna().values
                         for name, group in data.groupby(group_col)]
                results['test_used'] = 'Mann-Whitney U'
                results['test_result'] = NonParametricTests.mann_whitney_u(
                    *groups, alpha=self.alpha
                )

        else:  # > 2 groups
            if parametric:
                # One-way ANOVA
                results['test_used'] = 'One-way ANOVA'
                results['test_result'] = ANOVATests.one_way_anova(
                    data, value_col, group_col, alpha=self.alpha
                )

                # Post-hoc if significant
                if post_hoc and results['test_result']['significant']:
                    results['post_hoc'] = PostHocTests.tukey_hsd(
                        data, value_col, group_col, alpha=self.alpha
                    )
            else:
                # Kruskal-Wallis
                results['test_used'] = 'Kruskal-Wallis'
                results['test_result'] = NonParametricTests.kruskal_wallis(
                    data, value_col, group_col, alpha=self.alpha
                )

        return results

    def test_categorical_association(
        self,
        data: pd.DataFrame,
        var1: str,
        var2: str,
        method: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Test association between categorical variables.

        Args:
            data: DataFrame
            var1: First variable
            var2: Second variable
            method: 'auto', 'chi_square', or 'fisher'

        Returns:
            Test results
        """
        contingency = pd.crosstab(data[var1], data[var2])

        if method == 'auto':
            # Use Fisher's for 2x2, Chi-square otherwise
            if contingency.shape == (2, 2):
                expected = stats.chi2_contingency(contingency)[3]
                if expected.min() < 5:
                    method = 'fisher'
                else:
                    method = 'chi_square'
            else:
                method = 'chi_square'

        if method == 'fisher':
            return CategoricalTests.fishers_exact(data, var1, var2, alpha=self.alpha)
        else:
            return CategoricalTests.chi_square(data, var1, var2, alpha=self.alpha)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'StatisticalTestsManager',
    'ANOVATests',
    'PostHocTests',
    'NonParametricTests',
    'CategoricalTests',
    'VarianceTests',
    'GrangerCausalityTest',
    'EffectSize',
    'STATSMODELS_AVAILABLE',
]
