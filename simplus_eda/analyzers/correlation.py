"""
Correlation and relationship analysis module.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from scipy import stats


class CorrelationAnalyzer:
    """
    Analyze correlations and relationships between features.
    """

    def analyze(self, data: pd.DataFrame, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Perform correlation analysis.

        Args:
            data: Input DataFrame
            threshold: Correlation threshold for flagging strong relationships

        Returns:
            Dictionary containing correlation analysis results
        """
        results = {
            "correlation_matrix": self._calculate_correlations(data),
            "strong_correlations": self._find_strong_correlations(data, threshold),
            "multicollinearity": self._detect_multicollinearity(data),
        }
        return results

    def _calculate_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate correlation matrices using different methods.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing different correlation matrices
        """
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            return {
                "pearson": None,
                "spearman": None,
                "kendall": None,
                "message": "No numeric columns found for correlation analysis"
            }

        # Remove columns with zero variance
        numeric_data = numeric_data.loc[:, numeric_data.std() > 0]

        if numeric_data.empty:
            return {
                "pearson": None,
                "spearman": None,
                "kendall": None,
                "message": "All numeric columns have zero variance"
            }

        results = {}

        # Pearson correlation (linear relationships)
        try:
            pearson_corr = numeric_data.corr(method='pearson')
            results["pearson"] = pearson_corr.to_dict()
        except Exception as e:
            results["pearson"] = None
            results["pearson_error"] = str(e)

        # Spearman correlation (monotonic relationships)
        try:
            spearman_corr = numeric_data.corr(method='spearman')
            results["spearman"] = spearman_corr.to_dict()
        except Exception as e:
            results["spearman"] = None
            results["spearman_error"] = str(e)

        # Kendall correlation (rank-based, more robust)
        try:
            kendall_corr = numeric_data.corr(method='kendall')
            results["kendall"] = kendall_corr.to_dict()
        except Exception as e:
            results["kendall"] = None
            results["kendall_error"] = str(e)

        results["numeric_columns"] = list(numeric_data.columns)

        return results

    def _find_strong_correlations(
        self, data: pd.DataFrame, threshold: float
    ) -> Dict[str, Any]:
        """
        Find strongly correlated feature pairs.

        Args:
            data: Input DataFrame
            threshold: Correlation threshold (absolute value)

        Returns:
            Dictionary containing strongly correlated pairs
        """
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty or len(numeric_data.columns) < 2:
            return {
                "pairs": [],
                "count": 0,
                "message": "Not enough numeric columns for correlation analysis"
            }

        # Remove columns with zero variance
        numeric_data = numeric_data.loc[:, numeric_data.std() > 0]

        if len(numeric_data.columns) < 2:
            return {
                "pairs": [],
                "count": 0,
                "message": "Not enough columns with variance for correlation analysis"
            }

        # Calculate Pearson correlation
        corr_matrix = numeric_data.corr(method='pearson')

        # Find strong correlations
        strong_pairs = []
        columns = corr_matrix.columns

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                corr_value = corr_matrix.loc[col1, col2]

                if not np.isnan(corr_value) and abs(corr_value) >= threshold:
                    # Calculate p-value for statistical significance
                    try:
                        _, p_value = stats.pearsonr(
                            numeric_data[col1].dropna(),
                            numeric_data[col2].dropna()
                        )
                    except:
                        p_value = None

                    strong_pairs.append({
                        "feature1": col1,
                        "feature2": col2,
                        "correlation": float(corr_value),
                        "abs_correlation": float(abs(corr_value)),
                        "p_value": float(p_value) if p_value is not None else None,
                        "relationship": "positive" if corr_value > 0 else "negative",
                        "strength": self._classify_correlation_strength(abs(corr_value))
                    })

        # Sort by absolute correlation value
        strong_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

        return {
            "pairs": strong_pairs,
            "count": len(strong_pairs),
            "threshold": threshold
        }

    def _detect_multicollinearity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect multicollinearity issues using VIF (Variance Inflation Factor).

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing multicollinearity analysis
        """
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty or len(numeric_data.columns) < 2:
            return {
                "vif_scores": [],
                "high_vif_features": [],
                "message": "Not enough numeric columns for multicollinearity analysis"
            }

        # Remove columns with zero variance
        numeric_data = numeric_data.loc[:, numeric_data.std() > 0]

        # Remove rows with any NaN values for VIF calculation
        numeric_data_clean = numeric_data.dropna()

        if numeric_data_clean.empty or len(numeric_data_clean.columns) < 2:
            return {
                "vif_scores": [],
                "high_vif_features": [],
                "message": "Not enough data for VIF calculation after removing missing values"
            }

        vif_scores = []
        high_vif_features = []

        # Calculate VIF for each feature
        for i, col in enumerate(numeric_data_clean.columns):
            try:
                # Get all other columns as features
                X = numeric_data_clean.drop(columns=[col])
                y = numeric_data_clean[col]

                # Calculate R-squared
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X, y)
                r_squared = model.score(X, y)

                # Calculate VIF: VIF = 1 / (1 - R²)
                if r_squared < 0.9999:  # Avoid division by zero
                    vif = 1 / (1 - r_squared)
                else:
                    vif = float('inf')

                vif_info = {
                    "feature": col,
                    "vif": float(vif) if not np.isinf(vif) else "inf",
                    "interpretation": self._interpret_vif(vif)
                }

                vif_scores.append(vif_info)

                # Flag high VIF (> 10 is commonly used threshold)
                if vif > 10:
                    high_vif_features.append(vif_info)

            except Exception as e:
                vif_scores.append({
                    "feature": col,
                    "vif": None,
                    "error": str(e)
                })

        # Sort by VIF value
        vif_scores.sort(
            key=lambda x: x["vif"] if isinstance(x["vif"], (int, float)) and not np.isinf(x["vif"]) else 0,
            reverse=True
        )

        return {
            "vif_scores": vif_scores,
            "high_vif_features": high_vif_features,
            "high_vif_count": len(high_vif_features),
            "multicollinearity_detected": len(high_vif_features) > 0,
            "recommendation": self._get_multicollinearity_recommendation(len(high_vif_features))
        }

    def _classify_correlation_strength(self, abs_corr: float) -> str:
        """
        Classify correlation strength based on absolute value.

        Args:
            abs_corr: Absolute correlation value

        Returns:
            String describing correlation strength
        """
        if abs_corr >= 0.9:
            return "very_strong"
        elif abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.5:
            return "moderate"
        elif abs_corr >= 0.3:
            return "weak"
        else:
            return "very_weak"

    def _interpret_vif(self, vif: float) -> str:
        """
        Interpret VIF value.

        Args:
            vif: Variance Inflation Factor

        Returns:
            String interpretation of VIF
        """
        if np.isinf(vif):
            return "Perfect multicollinearity detected"
        elif vif > 10:
            return "High multicollinearity - consider removing"
        elif vif > 5:
            return "Moderate multicollinearity - investigate further"
        else:
            return "Low multicollinearity - acceptable"

    def _get_multicollinearity_recommendation(self, high_vif_count: int) -> str:
        """
        Get recommendation based on multicollinearity detection.

        Args:
            high_vif_count: Number of features with high VIF

        Returns:
            Recommendation string
        """
        if high_vif_count == 0:
            return "No significant multicollinearity detected. Features are suitable for modeling."
        elif high_vif_count <= 2:
            return f"Low multicollinearity detected ({high_vif_count} features). Consider feature selection or regularization."
        else:
            return f"High multicollinearity detected ({high_vif_count} features). Recommend removing redundant features or using dimensionality reduction."
