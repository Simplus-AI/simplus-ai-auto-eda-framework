"""
Feature Engineering Suggestions

Automatically detect and suggest feature engineering opportunities including:
- Feature interactions
- Polynomial features
- Binning strategies
- Encoding methods
- Scaling recommendations
- Derived features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from simplus_eda.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class ScalingMethod(Enum):
    """Recommended scaling methods."""
    STANDARD = "standard"  # StandardScaler (z-score)
    MINMAX = "minmax"  # MinMaxScaler (0-1)
    ROBUST = "robust"  # RobustScaler (median/IQR)
    LOG = "log"  # Log transformation
    SQRT = "sqrt"  # Square root transformation
    BOXCOX = "boxcox"  # Box-Cox transformation
    NONE = "none"  # No scaling needed


class EncodingMethod(Enum):
    """Recommended encoding methods."""
    ONE_HOT = "one_hot"  # One-hot encoding
    LABEL = "label"  # Label encoding (ordinal)
    TARGET = "target"  # Target encoding
    FREQUENCY = "frequency"  # Frequency encoding
    BINARY = "binary"  # Binary encoding
    NONE = "none"  # Already numeric


class BinningStrategy(Enum):
    """Binning strategies."""
    QUANTILE = "quantile"  # Equal frequency
    UNIFORM = "uniform"  # Equal width
    KMEANS = "kmeans"  # K-means clustering
    CUSTOM = "custom"  # Custom thresholds


@dataclass
class InteractionCandidate:
    """Feature interaction candidate."""
    feature1: str
    feature2: str
    interaction_type: str  # 'multiply', 'add', 'divide', 'subtract'
    importance_score: float
    reason: str


@dataclass
class PolynomialCandidate:
    """Polynomial feature candidate."""
    feature: str
    degree: int
    importance_score: float
    reason: str


@dataclass
class BinningRecommendation:
    """Binning recommendation."""
    feature: str
    strategy: BinningStrategy
    n_bins: int
    bin_edges: Optional[List[float]]
    bin_labels: Optional[List[str]]
    reason: str


@dataclass
class EncodingRecommendation:
    """Encoding recommendation."""
    feature: str
    method: EncodingMethod
    cardinality: int
    reason: str
    alternative_methods: List[EncodingMethod]


@dataclass
class ScalingRecommendation:
    """Scaling recommendation."""
    feature: str
    method: ScalingMethod
    reason: str
    alternative_methods: List[ScalingMethod]


# ============================================================================
# Feature Interaction Detector
# ============================================================================

class InteractionDetector:
    """Detect candidate feature interactions."""

    @staticmethod
    def detect_interactions(
        data: pd.DataFrame,
        target_col: Optional[str] = None,
        max_interactions: int = 10,
        correlation_threshold: float = 0.3,
        mutual_info_threshold: float = 0.1
    ) -> List[InteractionCandidate]:
        """
        Detect candidate feature interactions.

        Args:
            data: Input DataFrame
            target_col: Target column for supervised interaction detection
            max_interactions: Maximum number of interactions to return
            correlation_threshold: Min correlation for interaction candidates
            mutual_info_threshold: Min mutual information threshold

        Returns:
            List of interaction candidates
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)

        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric features for interaction detection")
            return []

        candidates = []

        # 1. Correlation-based interactions
        corr_matrix = data[numeric_cols].corr().abs()

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Check if features are correlated
                if corr_matrix.loc[col1, col2] > correlation_threshold:
                    # Suggest multiplication for correlated features
                    score = corr_matrix.loc[col1, col2]
                    candidates.append(InteractionCandidate(
                        feature1=col1,
                        feature2=col2,
                        interaction_type='multiply',
                        importance_score=float(score),
                        reason=f"High correlation ({score:.3f}) suggests multiplicative interaction"
                    ))

                # Check if ratio might be meaningful (different scales)
                if (data[col1].std() > 0 and data[col2].std() > 0 and
                    not np.any(data[col2] == 0)):
                    ratio_var = (data[col1] / data[col2]).var()
                    if ratio_var > data[col1].var() * 0.5:
                        candidates.append(InteractionCandidate(
                            feature1=col1,
                            feature2=col2,
                            interaction_type='divide',
                            importance_score=float(ratio_var / data[col1].var()),
                            reason=f"Ratio shows high variance, may capture important relationship"
                        ))

        # 2. Target-based interactions (if supervised)
        if target_col and target_col in data.columns:
            y = data[target_col].values

            # Check if classification or regression
            is_classification = data[target_col].dtype == 'object' or data[target_col].nunique() < 20

            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    # Create interaction features
                    interactions = {
                        'multiply': data[col1] * data[col2],
                        'add': data[col1] + data[col2],
                        'subtract': data[col1] - data[col2]
                    }

                    for interaction_type, interaction_feature in interactions.items():
                        # Calculate mutual information
                        try:
                            if is_classification:
                                mi = mutual_info_classif(
                                    interaction_feature.values.reshape(-1, 1),
                                    y,
                                    random_state=42
                                )[0]
                            else:
                                mi = mutual_info_regression(
                                    interaction_feature.values.reshape(-1, 1),
                                    y,
                                    random_state=42
                                )[0]

                            if mi > mutual_info_threshold:
                                candidates.append(InteractionCandidate(
                                    feature1=col1,
                                    feature2=col2,
                                    interaction_type=interaction_type,
                                    importance_score=float(mi),
                                    reason=f"High mutual information with target ({mi:.3f})"
                                ))
                        except Exception as e:
                            logger.debug(f"Error calculating MI for {col1} {interaction_type} {col2}: {e}")

        # Sort by importance and return top candidates
        candidates.sort(key=lambda x: x.importance_score, reverse=True)
        return candidates[:max_interactions]

    @staticmethod
    def create_interaction_features(
        data: pd.DataFrame,
        interactions: List[InteractionCandidate]
    ) -> pd.DataFrame:
        """
        Create interaction features from candidates.

        Args:
            data: Input DataFrame
            interactions: List of interaction candidates

        Returns:
            DataFrame with new interaction features
        """
        result = data.copy()

        for interaction in interactions:
            feature_name = f"{interaction.feature1}_{interaction.interaction_type}_{interaction.feature2}"

            try:
                if interaction.interaction_type == 'multiply':
                    result[feature_name] = data[interaction.feature1] * data[interaction.feature2]
                elif interaction.interaction_type == 'add':
                    result[feature_name] = data[interaction.feature1] + data[interaction.feature2]
                elif interaction.interaction_type == 'subtract':
                    result[feature_name] = data[interaction.feature1] - data[interaction.feature2]
                elif interaction.interaction_type == 'divide':
                    # Avoid division by zero
                    denominator = data[interaction.feature2].replace(0, np.nan)
                    result[feature_name] = data[interaction.feature1] / denominator
            except Exception as e:
                logger.warning(f"Failed to create interaction {feature_name}: {e}")

        return result


# ============================================================================
# Polynomial Feature Detector
# ============================================================================

class PolynomialDetector:
    """Detect features that benefit from polynomial transformations."""

    @staticmethod
    def detect_polynomial_features(
        data: pd.DataFrame,
        target_col: Optional[str] = None,
        max_degree: int = 3,
        max_features: int = 10,
        linearity_threshold: float = 0.95
    ) -> List[PolynomialCandidate]:
        """
        Detect features that benefit from polynomial transformations.

        Args:
            data: Input DataFrame
            target_col: Target column for supervised detection
            max_degree: Maximum polynomial degree to consider
            max_features: Maximum features to return
            linearity_threshold: R² threshold below which to suggest polynomials

        Returns:
            List of polynomial candidates
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)

        if not numeric_cols:
            return []

        candidates = []

        for col in numeric_cols:
            x = data[col].dropna().values

            # Check for non-linearity patterns
            # 1. Test for quadratic pattern
            if target_col and target_col in data.columns:
                y = data[target_col].loc[data[col].notna()].values

                # Fit linear model
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()

                try:
                    # Linear fit
                    X_linear = x.reshape(-1, 1)
                    lr.fit(X_linear, y)
                    r2_linear = lr.score(X_linear, y)

                    # Polynomial fit (degree 2)
                    X_poly2 = np.column_stack([x, x**2])
                    lr.fit(X_poly2, y)
                    r2_poly2 = lr.score(X_poly2, y)

                    # Polynomial fit (degree 3)
                    if max_degree >= 3:
                        X_poly3 = np.column_stack([x, x**2, x**3])
                        lr.fit(X_poly3, y)
                        r2_poly3 = lr.score(X_poly3, y)
                    else:
                        r2_poly3 = r2_poly2

                    # Suggest polynomial if it significantly improves fit
                    if r2_linear < linearity_threshold and r2_poly2 > r2_linear + 0.05:
                        degree = 2 if r2_poly3 <= r2_poly2 + 0.03 else 3
                        improvement = max(r2_poly2, r2_poly3) - r2_linear

                        candidates.append(PolynomialCandidate(
                            feature=col,
                            degree=degree,
                            importance_score=float(improvement),
                            reason=f"Polynomial degree {degree} improves R² by {improvement:.3f} "
                                   f"(linear: {r2_linear:.3f} → poly: {max(r2_poly2, r2_poly3):.3f})"
                        ))
                except Exception as e:
                    logger.debug(f"Error testing polynomial for {col}: {e}")

            # 2. Check distribution shape (skewness, kurtosis)
            else:
                skewness = stats.skew(x)
                kurtosis = stats.kurtosis(x)

                # High skewness or kurtosis suggests non-linear patterns
                if abs(skewness) > 1.0 or abs(kurtosis) > 3.0:
                    score = abs(skewness) + abs(kurtosis) / 3
                    candidates.append(PolynomialCandidate(
                        feature=col,
                        degree=2,
                        importance_score=float(score),
                        reason=f"High skewness ({skewness:.2f}) or kurtosis ({kurtosis:.2f}) "
                               f"suggests non-linear relationship"
                    ))

        # Sort by importance
        candidates.sort(key=lambda x: x.importance_score, reverse=True)
        return candidates[:max_features]

    @staticmethod
    def create_polynomial_features(
        data: pd.DataFrame,
        polynomials: List[PolynomialCandidate]
    ) -> pd.DataFrame:
        """
        Create polynomial features from candidates.

        Args:
            data: Input DataFrame
            polynomials: List of polynomial candidates

        Returns:
            DataFrame with polynomial features
        """
        result = data.copy()

        for poly in polynomials:
            for degree in range(2, poly.degree + 1):
                feature_name = f"{poly.feature}_pow{degree}"
                try:
                    result[feature_name] = data[poly.feature] ** degree
                except Exception as e:
                    logger.warning(f"Failed to create polynomial {feature_name}: {e}")

        return result


# ============================================================================
# Binning Recommender
# ============================================================================

class BinningRecommender:
    """Recommend binning strategies for continuous variables."""

    @staticmethod
    def recommend_binning(
        data: pd.DataFrame,
        target_col: Optional[str] = None,
        max_features: int = 10,
        min_unique_values: int = 20
    ) -> List[BinningRecommendation]:
        """
        Recommend binning strategies for continuous variables.

        Args:
            data: Input DataFrame
            target_col: Target column for supervised binning
            max_features: Maximum features to return
            min_unique_values: Minimum unique values to consider binning

        Returns:
            List of binning recommendations
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)

        recommendations = []

        for col in numeric_cols:
            n_unique = data[col].nunique()

            # Only bin if enough unique values
            if n_unique < min_unique_values:
                continue

            # Determine optimal number of bins
            n_bins = BinningRecommender._suggest_n_bins(data[col])

            # Determine strategy
            strategy, reason = BinningRecommender._determine_strategy(
                data[col], target_col, data
            )

            # Calculate bin edges
            bin_edges, bin_labels = BinningRecommender._calculate_bins(
                data[col], strategy, n_bins
            )

            recommendations.append(BinningRecommendation(
                feature=col,
                strategy=strategy,
                n_bins=n_bins,
                bin_edges=bin_edges,
                bin_labels=bin_labels,
                reason=reason
            ))

        return recommendations[:max_features]

    @staticmethod
    def _suggest_n_bins(series: pd.Series) -> int:
        """Suggest optimal number of bins using multiple rules."""
        n = len(series.dropna())

        # Sturges' rule
        sturges = int(np.ceil(np.log2(n) + 1))

        # Freedman-Diaconis rule
        iqr = series.quantile(0.75) - series.quantile(0.25)
        if iqr > 0:
            bin_width = 2 * iqr / (n ** (1/3))
            fd = int(np.ceil((series.max() - series.min()) / bin_width))
        else:
            fd = sturges

        # Scott's rule
        std = series.std()
        if std > 0:
            bin_width = 3.5 * std / (n ** (1/3))
            scott = int(np.ceil((series.max() - series.min()) / bin_width))
        else:
            scott = sturges

        # Return median of the three rules, bounded
        n_bins = int(np.median([sturges, fd, scott]))
        return max(3, min(n_bins, 20))  # Between 3 and 20 bins

    @staticmethod
    def _determine_strategy(
        series: pd.Series,
        target_col: Optional[str],
        data: pd.DataFrame
    ) -> Tuple[BinningStrategy, str]:
        """Determine optimal binning strategy."""
        # Check distribution
        skewness = stats.skew(series.dropna())

        # If highly skewed, use quantile binning
        if abs(skewness) > 1.0:
            return (
                BinningStrategy.QUANTILE,
                f"Quantile binning recommended due to skewed distribution (skew={skewness:.2f})"
            )

        # If uniform distribution, use uniform binning
        if abs(skewness) < 0.5:
            return (
                BinningStrategy.UNIFORM,
                f"Uniform binning recommended for approximately uniform distribution (skew={skewness:.2f})"
            )

        # Default to quantile for balanced bin sizes
        return (
            BinningStrategy.QUANTILE,
            "Quantile binning recommended for balanced bin sizes"
        )

    @staticmethod
    def _calculate_bins(
        series: pd.Series,
        strategy: BinningStrategy,
        n_bins: int
    ) -> Tuple[Optional[List[float]], Optional[List[str]]]:
        """Calculate bin edges and labels."""
        try:
            if strategy == BinningStrategy.QUANTILE:
                # Quantile-based bins
                bin_edges = series.quantile(np.linspace(0, 1, n_bins + 1)).tolist()
            elif strategy == BinningStrategy.UNIFORM:
                # Uniform-width bins
                bin_edges = np.linspace(series.min(), series.max(), n_bins + 1).tolist()
            else:
                bin_edges = None

            # Generate labels
            if bin_edges:
                bin_labels = [f"bin_{i}" for i in range(n_bins)]
            else:
                bin_labels = None

            return bin_edges, bin_labels
        except Exception as e:
            logger.debug(f"Error calculating bins: {e}")
            return None, None

    @staticmethod
    def apply_binning(
        data: pd.DataFrame,
        recommendations: List[BinningRecommendation]
    ) -> pd.DataFrame:
        """
        Apply binning recommendations to data.

        Args:
            data: Input DataFrame
            recommendations: List of binning recommendations

        Returns:
            DataFrame with binned features
        """
        result = data.copy()

        for rec in recommendations:
            if rec.bin_edges is None:
                continue

            feature_name = f"{rec.feature}_binned"

            try:
                result[feature_name] = pd.cut(
                    data[rec.feature],
                    bins=rec.bin_edges,
                    labels=rec.bin_labels,
                    include_lowest=True,
                    duplicates='drop'
                )
            except Exception as e:
                logger.warning(f"Failed to apply binning to {rec.feature}: {e}")

        return result


# ============================================================================
# Encoding Recommender
# ============================================================================

class EncodingRecommender:
    """Recommend encoding methods for categorical features."""

    @staticmethod
    def recommend_encoding(
        data: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> List[EncodingRecommendation]:
        """
        Recommend encoding methods for categorical features.

        Args:
            data: Input DataFrame
            target_col: Target column for supervised encoding

        Returns:
            List of encoding recommendations
        """
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        if target_col and target_col in categorical_cols:
            categorical_cols.remove(target_col)

        recommendations = []

        for col in categorical_cols:
            cardinality = data[col].nunique()

            # Determine primary method
            method, reason, alternatives = EncodingRecommender._determine_encoding(
                data[col], cardinality, target_col, data
            )

            recommendations.append(EncodingRecommendation(
                feature=col,
                method=method,
                cardinality=cardinality,
                reason=reason,
                alternative_methods=alternatives
            ))

        return recommendations

    @staticmethod
    def _determine_encoding(
        series: pd.Series,
        cardinality: int,
        target_col: Optional[str],
        data: pd.DataFrame
    ) -> Tuple[EncodingMethod, str, List[EncodingMethod]]:
        """Determine optimal encoding method."""
        alternatives = []

        # Binary feature (2 unique values)
        if cardinality == 2:
            return (
                EncodingMethod.LABEL,
                f"Binary feature with 2 categories - label encoding is sufficient",
                [EncodingMethod.ONE_HOT, EncodingMethod.BINARY]
            )

        # Low cardinality (3-10 categories)
        elif cardinality <= 10:
            alternatives = [EncodingMethod.TARGET, EncodingMethod.FREQUENCY]
            return (
                EncodingMethod.ONE_HOT,
                f"Low cardinality ({cardinality} categories) - one-hot encoding recommended",
                alternatives
            )

        # Medium cardinality (11-50 categories)
        elif cardinality <= 50:
            alternatives = [EncodingMethod.BINARY, EncodingMethod.ONE_HOT]

            # If target available, suggest target encoding
            if target_col:
                return (
                    EncodingMethod.TARGET,
                    f"Medium cardinality ({cardinality} categories) with target - "
                    f"target encoding recommended to avoid high dimensionality",
                    alternatives
                )
            else:
                return (
                    EncodingMethod.FREQUENCY,
                    f"Medium cardinality ({cardinality} categories) - "
                    f"frequency encoding recommended",
                    alternatives
                )

        # High cardinality (>50 categories)
        else:
            alternatives = [EncodingMethod.BINARY, EncodingMethod.LABEL]

            if target_col:
                return (
                    EncodingMethod.TARGET,
                    f"High cardinality ({cardinality} categories) - "
                    f"target encoding recommended to avoid curse of dimensionality",
                    alternatives
                )
            else:
                return (
                    EncodingMethod.FREQUENCY,
                    f"High cardinality ({cardinality} categories) - "
                    f"frequency encoding or consider feature reduction",
                    alternatives
                )


# ============================================================================
# Scaling Recommender
# ============================================================================

class ScalingRecommender:
    """Recommend scaling methods for numeric features."""

    @staticmethod
    def recommend_scaling(
        data: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> List[ScalingRecommendation]:
        """
        Recommend scaling methods for numeric features.

        Args:
            data: Input DataFrame
            target_col: Target column (excluded from scaling)

        Returns:
            List of scaling recommendations
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)

        recommendations = []

        for col in numeric_cols:
            method, reason, alternatives = ScalingRecommender._determine_scaling(
                data[col]
            )

            recommendations.append(ScalingRecommendation(
                feature=col,
                method=method,
                reason=reason,
                alternative_methods=alternatives
            ))

        return recommendations

    @staticmethod
    def _determine_scaling(series: pd.Series) -> Tuple[ScalingMethod, str, List[ScalingMethod]]:
        """Determine optimal scaling method."""
        # Calculate statistics
        data_clean = series.dropna()

        if len(data_clean) == 0:
            return ScalingMethod.NONE, "All values are missing", []

        mean = data_clean.mean()
        std = data_clean.std()
        median = data_clean.median()
        min_val = data_clean.min()
        max_val = data_clean.max()
        skewness = stats.skew(data_clean)

        # Count outliers
        q1, q3 = data_clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((data_clean < q1 - 1.5 * iqr) | (data_clean > q3 + 1.5 * iqr)).sum()
        outlier_pct = outliers / len(data_clean)

        # Already scaled (mean ~0, std ~1)
        if abs(mean) < 0.1 and abs(std - 1) < 0.1:
            return (
                ScalingMethod.NONE,
                f"Already standardized (mean={mean:.3f}, std={std:.3f})",
                []
            )

        # Already in [0, 1] range
        if min_val >= 0 and max_val <= 1:
            return (
                ScalingMethod.NONE,
                f"Already in [0, 1] range",
                []
            )

        # Highly skewed - suggest transformation
        if abs(skewness) > 2.0:
            if min_val > 0:
                return (
                    ScalingMethod.LOG,
                    f"Highly skewed ({skewness:.2f}) with positive values - log transformation recommended",
                    [ScalingMethod.SQRT, ScalingMethod.BOXCOX, ScalingMethod.ROBUST]
                )
            else:
                return (
                    ScalingMethod.ROBUST,
                    f"Highly skewed ({skewness:.2f}) with non-positive values - robust scaling recommended",
                    [ScalingMethod.STANDARD]
                )

        # Many outliers - use robust scaling
        if outlier_pct > 0.05:
            return (
                ScalingMethod.ROBUST,
                f"Contains {outlier_pct:.1%} outliers - robust scaling recommended",
                [ScalingMethod.STANDARD, ScalingMethod.MINMAX]
            )

        # Normal distribution - standard scaling
        if abs(skewness) < 0.5:
            return (
                ScalingMethod.STANDARD,
                f"Approximately normal distribution (skew={skewness:.2f}) - standard scaling recommended",
                [ScalingMethod.MINMAX, ScalingMethod.ROBUST]
            )

        # Default: standard scaling
        return (
            ScalingMethod.STANDARD,
            "Standard scaling recommended for general use",
            [ScalingMethod.MINMAX, ScalingMethod.ROBUST]
        )


# ============================================================================
# Feature Engineering Manager
# ============================================================================

class FeatureEngineeringManager:
    """
    Main feature engineering manager.

    Coordinates all feature engineering suggestions.
    """

    def __init__(
        self,
        detect_interactions: bool = True,
        detect_polynomials: bool = True,
        suggest_binning: bool = True,
        suggest_encoding: bool = True,
        suggest_scaling: bool = True
    ):
        """
        Initialize feature engineering manager.

        Args:
            detect_interactions: Enable interaction detection
            detect_polynomials: Enable polynomial detection
            suggest_binning: Enable binning suggestions
            suggest_encoding: Enable encoding suggestions
            suggest_scaling: Enable scaling suggestions
        """
        self.detect_interactions = detect_interactions
        self.detect_polynomials = detect_polynomials
        self.suggest_binning = suggest_binning
        self.suggest_encoding = suggest_encoding
        self.suggest_scaling = suggest_scaling

    def analyze(
        self,
        data: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive feature engineering analysis.

        Args:
            data: Input DataFrame
            target_col: Optional target column for supervised suggestions

        Returns:
            Dictionary with all feature engineering suggestions
        """
        results = {}

        # Interaction detection
        if self.detect_interactions:
            logger.info("Detecting feature interactions...")
            interactions = InteractionDetector.detect_interactions(
                data, target_col=target_col
            )
            results['interactions'] = [
                {
                    'feature1': i.feature1,
                    'feature2': i.feature2,
                    'type': i.interaction_type,
                    'score': i.importance_score,
                    'reason': i.reason
                }
                for i in interactions
            ]

        # Polynomial detection
        if self.detect_polynomials:
            logger.info("Detecting polynomial features...")
            polynomials = PolynomialDetector.detect_polynomial_features(
                data, target_col=target_col
            )
            results['polynomials'] = [
                {
                    'feature': p.feature,
                    'degree': p.degree,
                    'score': p.importance_score,
                    'reason': p.reason
                }
                for p in polynomials
            ]

        # Binning suggestions
        if self.suggest_binning:
            logger.info("Generating binning suggestions...")
            binning = BinningRecommender.recommend_binning(
                data, target_col=target_col
            )
            results['binning'] = [
                {
                    'feature': b.feature,
                    'strategy': b.strategy.value,
                    'n_bins': b.n_bins,
                    'bin_edges': b.bin_edges,
                    'bin_labels': b.bin_labels,
                    'reason': b.reason
                }
                for b in binning
            ]

        # Encoding suggestions
        if self.suggest_encoding:
            logger.info("Generating encoding suggestions...")
            encoding = EncodingRecommender.recommend_encoding(
                data, target_col=target_col
            )
            results['encoding'] = [
                {
                    'feature': e.feature,
                    'method': e.method.value,
                    'cardinality': e.cardinality,
                    'reason': e.reason,
                    'alternatives': [m.value for m in e.alternative_methods]
                }
                for e in encoding
            ]

        # Scaling suggestions
        if self.suggest_scaling:
            logger.info("Generating scaling suggestions...")
            scaling = ScalingRecommender.recommend_scaling(
                data, target_col=target_col
            )
            results['scaling'] = [
                {
                    'feature': s.feature,
                    'method': s.method.value,
                    'reason': s.reason,
                    'alternatives': [m.value for m in s.alternative_methods]
                }
                for s in scaling
            ]

        return results

    def generate_code(self, results: Dict[str, Any]) -> str:
        """
        Generate Python code to apply suggestions.

        Args:
            results: Results from analyze()

        Returns:
            Python code string
        """
        code_lines = [
            "# Feature Engineering Code",
            "# Generated by Simplus EDA",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler",
            "",
            "def engineer_features(df):",
            "    \"\"\"Apply feature engineering transformations.\"\"\"",
            "    df = df.copy()",
            ""
        ]

        # Interactions
        if 'interactions' in results and results['interactions']:
            code_lines.append("    # Feature Interactions")
            for interaction in results['interactions'][:5]:
                f1, f2 = interaction['feature1'], interaction['feature2']
                itype = interaction['type']
                name = f"{f1}_{itype}_{f2}"

                if itype == 'multiply':
                    code_lines.append(f"    df['{name}'] = df['{f1}'] * df['{f2}']")
                elif itype == 'add':
                    code_lines.append(f"    df['{name}'] = df['{f1}'] + df['{f2}']")
                elif itype == 'subtract':
                    code_lines.append(f"    df['{name}'] = df['{f1}'] - df['{f2}']")
                elif itype == 'divide':
                    code_lines.append(f"    df['{name}'] = df['{f1}'] / df['{f2}'].replace(0, np.nan)")

            code_lines.append("")

        # Polynomials
        if 'polynomials' in results and results['polynomials']:
            code_lines.append("    # Polynomial Features")
            for poly in results['polynomials'][:5]:
                feat = poly['feature']
                deg = poly['degree']
                for d in range(2, deg + 1):
                    code_lines.append(f"    df['{feat}_pow{d}'] = df['{feat}'] ** {d}")
            code_lines.append("")

        # Binning
        if 'binning' in results and results['binning']:
            code_lines.append("    # Binning")
            for binning in results['binning'][:5]:
                feat = binning['feature']
                if binning['bin_edges']:
                    edges = binning['bin_edges']
                    code_lines.append(f"    df['{feat}_binned'] = pd.cut(df['{feat}'], bins={edges}, labels=False)")
            code_lines.append("")

        # Encoding
        if 'encoding' in results and results['encoding']:
            code_lines.append("    # Encoding")
            for enc in results['encoding']:
                feat = enc['feature']
                method = enc['method']

                if method == 'one_hot':
                    code_lines.append(f"    # One-hot encode {feat}")
                    code_lines.append(f"    df = pd.get_dummies(df, columns=['{feat}'], prefix='{feat}')")
                elif method == 'label':
                    code_lines.append(f"    # Label encode {feat}")
                    code_lines.append(f"    df['{feat}_encoded'] = df['{feat}'].astype('category').cat.codes")
            code_lines.append("")

        # Scaling
        if 'scaling' in results and results['scaling']:
            code_lines.append("    # Scaling")
            features_to_scale = [s['feature'] for s in results['scaling'] if s['method'] != 'none']
            if features_to_scale:
                code_lines.append(f"    scaler = StandardScaler()")
                code_lines.append(f"    df[{features_to_scale}] = scaler.fit_transform(df[{features_to_scale}])")
            code_lines.append("")

        code_lines.extend([
            "    return df",
            "",
            "# Apply transformations",
            "# df_engineered = engineer_features(df)"
        ])

        return "\n".join(code_lines)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'FeatureEngineeringManager',
    'InteractionDetector',
    'PolynomialDetector',
    'BinningRecommender',
    'EncodingRecommender',
    'ScalingRecommender',
    'ScalingMethod',
    'EncodingMethod',
    'BinningStrategy',
    'InteractionCandidate',
    'PolynomialCandidate',
    'BinningRecommendation',
    'EncodingRecommendation',
    'ScalingRecommendation',
]
