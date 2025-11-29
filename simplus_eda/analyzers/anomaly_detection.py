"""
Advanced Anomaly Detection

Comprehensive anomaly detection methods including:
- Univariate outlier detection (IQR, Z-score, MAD)
- Multivariate outlier detection (Mahalanobis, Isolation Forest, LOF)
- Time-series anomaly detection (STL decomposition, ARIMA residuals)
- Clustering-based detection (DBSCAN)
- Anomaly explanation (feature importance, SHAP values)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope

from simplus_eda.logging_config import get_logger

logger = get_logger(__name__)

# Optional dependencies
try:
    from statsmodels.tsa.seasonal import STL
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available - time series anomaly detection limited")


# ============================================================================
# Enums and Data Classes
# ============================================================================

class AnomalyMethod(Enum):
    """Anomaly detection methods."""
    IQR = "iqr"  # Interquartile Range
    ZSCORE = "zscore"  # Z-score
    MAD = "mad"  # Median Absolute Deviation
    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"  # Local Outlier Factor
    DBSCAN = "dbscan"  # Density-based clustering
    MAHALANOBIS = "mahalanobis"  # Mahalanobis distance
    ELLIPTIC_ENVELOPE = "elliptic_envelope"  # Robust covariance
    STL_DECOMPOSITION = "stl"  # Time series decomposition
    ARIMA_RESIDUALS = "arima"  # ARIMA model residuals


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    method: str
    is_anomaly: np.ndarray  # Boolean array
    anomaly_scores: np.ndarray  # Continuous scores
    threshold: Optional[float]
    n_anomalies: int
    anomaly_percentage: float
    anomaly_indices: List[int]
    metadata: Dict[str, Any]


# ============================================================================
# Univariate Anomaly Detection
# ============================================================================

class UnivariateAnomalyDetector:
    """Detect anomalies in single variables."""

    @staticmethod
    def iqr_method(
        data: Union[pd.Series, np.ndarray],
        multiplier: float = 1.5
    ) -> AnomalyResult:
        """
        IQR-based outlier detection.

        Args:
            data: Input data
            multiplier: IQR multiplier (1.5 for outliers, 3.0 for extreme outliers)

        Returns:
            AnomalyResult
        """
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.asarray(data)

        # Remove NaN
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]

        if len(valid_values) == 0:
            return UnivariateAnomalyDetector._empty_result("iqr", len(values))

        # Calculate quartiles
        q1, q3 = np.percentile(valid_values, [25, 75])
        iqr = q3 - q1

        # Define bounds
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        # Detect anomalies
        is_anomaly = np.zeros(len(values), dtype=bool)
        is_anomaly[valid_mask] = (valid_values < lower_bound) | (valid_values > upper_bound)

        # Calculate scores (distance from nearest bound)
        scores = np.zeros(len(values))
        scores[valid_mask] = np.maximum(
            lower_bound - valid_values,
            valid_values - upper_bound
        )
        scores = np.maximum(scores, 0)

        return AnomalyResult(
            method="iqr",
            is_anomaly=is_anomaly,
            anomaly_scores=scores,
            threshold=None,
            n_anomalies=int(is_anomaly.sum()),
            anomaly_percentage=float(is_anomaly.sum() / len(is_anomaly) * 100),
            anomaly_indices=np.where(is_anomaly)[0].tolist(),
            metadata={
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(iqr),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'multiplier': multiplier
            }
        )

    @staticmethod
    def zscore_method(
        data: Union[pd.Series, np.ndarray],
        threshold: float = 3.0
    ) -> AnomalyResult:
        """
        Z-score based outlier detection.

        Args:
            data: Input data
            threshold: Z-score threshold (typically 2.5-3.0)

        Returns:
            AnomalyResult
        """
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.asarray(data)

        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]

        if len(valid_values) == 0:
            return UnivariateAnomalyDetector._empty_result("zscore", len(values))

        # Calculate z-scores
        mean = np.mean(valid_values)
        std = np.std(valid_values)

        if std == 0:
            return UnivariateAnomalyDetector._empty_result("zscore", len(values))

        z_scores = np.zeros(len(values))
        z_scores[valid_mask] = np.abs((valid_values - mean) / std)

        is_anomaly = z_scores > threshold

        return AnomalyResult(
            method="zscore",
            is_anomaly=is_anomaly,
            anomaly_scores=z_scores,
            threshold=float(threshold),
            n_anomalies=int(is_anomaly.sum()),
            anomaly_percentage=float(is_anomaly.sum() / len(is_anomaly) * 100),
            anomaly_indices=np.where(is_anomaly)[0].tolist(),
            metadata={
                'mean': float(mean),
                'std': float(std),
                'threshold': float(threshold)
            }
        )

    @staticmethod
    def mad_method(
        data: Union[pd.Series, np.ndarray],
        threshold: float = 3.5
    ) -> AnomalyResult:
        """
        Modified Z-score using Median Absolute Deviation (MAD).
        More robust to outliers than standard z-score.

        Args:
            data: Input data
            threshold: MAD threshold (typically 3.5)

        Returns:
            AnomalyResult
        """
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.asarray(data)

        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]

        if len(valid_values) == 0:
            return UnivariateAnomalyDetector._empty_result("mad", len(values))

        # Calculate MAD
        median = np.median(valid_values)
        mad = np.median(np.abs(valid_values - median))

        if mad == 0:
            return UnivariateAnomalyDetector._empty_result("mad", len(values))

        # Modified z-score
        modified_z_scores = np.zeros(len(values))
        modified_z_scores[valid_mask] = 0.6745 * np.abs(valid_values - median) / mad

        is_anomaly = modified_z_scores > threshold

        return AnomalyResult(
            method="mad",
            is_anomaly=is_anomaly,
            anomaly_scores=modified_z_scores,
            threshold=float(threshold),
            n_anomalies=int(is_anomaly.sum()),
            anomaly_percentage=float(is_anomaly.sum() / len(is_anomaly) * 100),
            anomaly_indices=np.where(is_anomaly)[0].tolist(),
            metadata={
                'median': float(median),
                'mad': float(mad),
                'threshold': float(threshold)
            }
        )

    @staticmethod
    def _empty_result(method: str, n_samples: int) -> AnomalyResult:
        """Create empty result for edge cases."""
        return AnomalyResult(
            method=method,
            is_anomaly=np.zeros(n_samples, dtype=bool),
            anomaly_scores=np.zeros(n_samples),
            threshold=None,
            n_anomalies=0,
            anomaly_percentage=0.0,
            anomaly_indices=[],
            metadata={}
        )


# ============================================================================
# Multivariate Anomaly Detection
# ============================================================================

class MultivariateAnomalyDetector:
    """Detect anomalies in multiple variables simultaneously."""

    @staticmethod
    def isolation_forest(
        data: pd.DataFrame,
        contamination: float = 0.1,
        random_state: int = 42
    ) -> AnomalyResult:
        """
        Isolation Forest for anomaly detection.

        Args:
            data: Input DataFrame (numeric columns only)
            contamination: Expected proportion of outliers
            random_state: Random seed

        Returns:
            AnomalyResult
        """
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        X = data[numeric_cols].dropna()

        if len(X) == 0:
            return MultivariateAnomalyDetector._empty_result("isolation_forest", len(data))

        # Fit model
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )

        # Predict (-1 for anomalies, 1 for normal)
        predictions = iso_forest.fit_predict(X)

        # Get anomaly scores (lower = more anomalous)
        scores = iso_forest.score_samples(X)

        # Map back to original indices
        is_anomaly = np.zeros(len(data), dtype=bool)
        anomaly_scores = np.zeros(len(data))

        is_anomaly[X.index] = predictions == -1
        anomaly_scores[X.index] = -scores  # Negate so higher = more anomalous

        return AnomalyResult(
            method="isolation_forest",
            is_anomaly=is_anomaly,
            anomaly_scores=anomaly_scores,
            threshold=None,
            n_anomalies=int(is_anomaly.sum()),
            anomaly_percentage=float(is_anomaly.sum() / len(is_anomaly) * 100),
            anomaly_indices=np.where(is_anomaly)[0].tolist(),
            metadata={
                'contamination': contamination,
                'n_estimators': 100,
                'features_used': numeric_cols.tolist()
            }
        )

    @staticmethod
    def local_outlier_factor(
        data: pd.DataFrame,
        contamination: float = 0.1,
        n_neighbors: int = 20
    ) -> AnomalyResult:
        """
        Local Outlier Factor (LOF) for anomaly detection.

        Args:
            data: Input DataFrame (numeric columns only)
            contamination: Expected proportion of outliers
            n_neighbors: Number of neighbors to consider

        Returns:
            AnomalyResult
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        X = data[numeric_cols].dropna()

        if len(X) == 0:
            return MultivariateAnomalyDetector._empty_result("lof", len(data))

        # Fit model
        lof = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=min(n_neighbors, len(X) - 1)
        )

        # Predict
        predictions = lof.fit_predict(X)

        # Get negative outlier factor (lower = more anomalous)
        scores = -lof.negative_outlier_factor_

        # Map back
        is_anomaly = np.zeros(len(data), dtype=bool)
        anomaly_scores = np.zeros(len(data))

        is_anomaly[X.index] = predictions == -1
        anomaly_scores[X.index] = scores

        return AnomalyResult(
            method="lof",
            is_anomaly=is_anomaly,
            anomaly_scores=anomaly_scores,
            threshold=None,
            n_anomalies=int(is_anomaly.sum()),
            anomaly_percentage=float(is_anomaly.sum() / len(is_anomaly) * 100),
            anomaly_indices=np.where(is_anomaly)[0].tolist(),
            metadata={
                'contamination': contamination,
                'n_neighbors': n_neighbors,
                'features_used': numeric_cols.tolist()
            }
        )

    @staticmethod
    def mahalanobis_distance(
        data: pd.DataFrame,
        threshold_percentile: float = 97.5
    ) -> AnomalyResult:
        """
        Mahalanobis distance for multivariate outlier detection.

        Args:
            data: Input DataFrame (numeric columns only)
            threshold_percentile: Percentile for threshold (e.g., 97.5 for top 2.5%)

        Returns:
            AnomalyResult
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        X = data[numeric_cols].dropna()

        if len(X) == 0 or X.shape[1] < 2:
            return MultivariateAnomalyDetector._empty_result("mahalanobis", len(data))

        # Calculate covariance matrix
        try:
            mean = X.mean(axis=0)
            cov = np.cov(X.T)
            inv_cov = np.linalg.inv(cov)

            # Calculate Mahalanobis distance for each point
            distances = np.zeros(len(X))
            for i, (idx, row) in enumerate(X.iterrows()):
                distances[i] = mahalanobis(row, mean, inv_cov)

            # Determine threshold
            threshold = np.percentile(distances, threshold_percentile)

            # Map back
            is_anomaly_local = distances > threshold

            is_anomaly = np.zeros(len(data), dtype=bool)
            anomaly_scores = np.zeros(len(data))

            is_anomaly[X.index] = is_anomaly_local
            anomaly_scores[X.index] = distances

            return AnomalyResult(
                method="mahalanobis",
                is_anomaly=is_anomaly,
                anomaly_scores=anomaly_scores,
                threshold=float(threshold),
                n_anomalies=int(is_anomaly.sum()),
                anomaly_percentage=float(is_anomaly.sum() / len(is_anomaly) * 100),
                anomaly_indices=np.where(is_anomaly)[0].tolist(),
                metadata={
                    'threshold_percentile': threshold_percentile,
                    'threshold_value': float(threshold),
                    'features_used': numeric_cols.tolist()
                }
            )
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix, cannot compute Mahalanobis distance")
            return MultivariateAnomalyDetector._empty_result("mahalanobis", len(data))

    @staticmethod
    def elliptic_envelope(
        data: pd.DataFrame,
        contamination: float = 0.1
    ) -> AnomalyResult:
        """
        Robust covariance estimation (Elliptic Envelope).

        Args:
            data: Input DataFrame (numeric columns only)
            contamination: Expected proportion of outliers

        Returns:
            AnomalyResult
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        X = data[numeric_cols].dropna()

        if len(X) == 0 or X.shape[1] < 2:
            return MultivariateAnomalyDetector._empty_result("elliptic_envelope", len(data))

        try:
            # Fit model
            envelope = EllipticEnvelope(contamination=contamination, random_state=42)
            predictions = envelope.fit_predict(X)

            # Get Mahalanobis distances
            scores = envelope.mahalanobis(X)

            # Map back
            is_anomaly = np.zeros(len(data), dtype=bool)
            anomaly_scores = np.zeros(len(data))

            is_anomaly[X.index] = predictions == -1
            anomaly_scores[X.index] = scores

            return AnomalyResult(
                method="elliptic_envelope",
                is_anomaly=is_anomaly,
                anomaly_scores=anomaly_scores,
                threshold=None,
                n_anomalies=int(is_anomaly.sum()),
                anomaly_percentage=float(is_anomaly.sum() / len(is_anomaly) * 100),
                anomaly_indices=np.where(is_anomaly)[0].tolist(),
                metadata={
                    'contamination': contamination,
                    'features_used': numeric_cols.tolist()
                }
            )
        except Exception as e:
            logger.warning(f"Elliptic envelope failed: {e}")
            return MultivariateAnomalyDetector._empty_result("elliptic_envelope", len(data))

    @staticmethod
    def _empty_result(method: str, n_samples: int) -> AnomalyResult:
        """Create empty result."""
        return AnomalyResult(
            method=method,
            is_anomaly=np.zeros(n_samples, dtype=bool),
            anomaly_scores=np.zeros(n_samples),
            threshold=None,
            n_anomalies=0,
            anomaly_percentage=0.0,
            anomaly_indices=[],
            metadata={}
        )


# ============================================================================
# Clustering-Based Anomaly Detection
# ============================================================================

class ClusteringAnomalyDetector:
    """Detect anomalies using clustering methods."""

    @staticmethod
    def dbscan_method(
        data: pd.DataFrame,
        eps: Optional[float] = None,
        min_samples: int = 5,
        metric: str = 'euclidean'
    ) -> AnomalyResult:
        """
        DBSCAN-based anomaly detection.
        Points labeled as noise (-1) are considered anomalies.

        Args:
            data: Input DataFrame (numeric columns only)
            eps: Maximum distance between samples (auto if None)
            min_samples: Minimum samples in a neighborhood
            metric: Distance metric

        Returns:
            AnomalyResult
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        X = data[numeric_cols].dropna()

        if len(X) == 0:
            return ClusteringAnomalyDetector._empty_result("dbscan", len(data))

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Auto-determine eps if not provided
        if eps is None:
            # Use k-distance graph heuristic
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=min_samples)
            neighbors.fit(X_scaled)
            distances, _ = neighbors.kneighbors(X_scaled)
            distances = np.sort(distances[:, -1])
            eps = np.percentile(distances, 90)  # Use 90th percentile

        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = dbscan.fit_predict(X_scaled)

        # -1 label indicates noise/anomaly
        is_anomaly_local = labels == -1

        # Calculate scores (distance to nearest cluster center)
        scores = np.zeros(len(X))
        if len(np.unique(labels)) > 1:
            # Calculate distance to nearest non-noise cluster
            from sklearn.neighbors import NearestNeighbors
            non_noise_mask = labels != -1

            if non_noise_mask.sum() > 0:
                neighbors = NearestNeighbors(n_neighbors=1)
                neighbors.fit(X_scaled[non_noise_mask])
                distances, _ = neighbors.kneighbors(X_scaled)
                scores = distances.flatten()

        # Map back
        is_anomaly = np.zeros(len(data), dtype=bool)
        anomaly_scores = np.zeros(len(data))

        is_anomaly[X.index] = is_anomaly_local
        anomaly_scores[X.index] = scores

        return AnomalyResult(
            method="dbscan",
            is_anomaly=is_anomaly,
            anomaly_scores=anomaly_scores,
            threshold=float(eps),
            n_anomalies=int(is_anomaly.sum()),
            anomaly_percentage=float(is_anomaly.sum() / len(is_anomaly) * 100),
            anomaly_indices=np.where(is_anomaly)[0].tolist(),
            metadata={
                'eps': float(eps),
                'min_samples': min_samples,
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'features_used': numeric_cols.tolist()
            }
        )

    @staticmethod
    def _empty_result(method: str, n_samples: int) -> AnomalyResult:
        """Create empty result."""
        return AnomalyResult(
            method=method,
            is_anomaly=np.zeros(n_samples, dtype=bool),
            anomaly_scores=np.zeros(n_samples),
            threshold=None,
            n_anomalies=0,
            anomaly_percentage=0.0,
            anomaly_indices=[],
            metadata={}
        )


# ============================================================================
# Time Series Anomaly Detection
# ============================================================================

class TimeSeriesAnomalyDetector:
    """Detect anomalies in time series data."""

    @staticmethod
    def stl_decomposition(
        series: Union[pd.Series, np.ndarray],
        period: Optional[int] = None,
        threshold: float = 3.0
    ) -> AnomalyResult:
        """
        STL decomposition-based anomaly detection.

        Args:
            series: Time series data
            period: Seasonal period (auto-detect if None)
            threshold: Z-score threshold for residuals

        Returns:
            AnomalyResult
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for STL decomposition")

        if isinstance(series, np.ndarray):
            series = pd.Series(series)

        # Remove NaN
        series_clean = series.dropna()

        if len(series_clean) < 2:
            return TimeSeriesAnomalyDetector._empty_result("stl", len(series))

        # Auto-detect period
        if period is None:
            from statsmodels.tsa.stattools import acf
            try:
                acf_vals = acf(series_clean, nlags=min(len(series_clean) // 2, 100))
                # Find first significant peak after lag 1
                period = np.argmax(acf_vals[1:]) + 1
                period = max(period, 2)  # At least 2
            except:
                period = 2

        try:
            # Perform STL decomposition
            stl = STL(series_clean, seasonal=period)
            result = stl.fit()

            # Use residuals for anomaly detection
            residuals = result.resid
            residual_std = residuals.std()

            if residual_std == 0:
                return TimeSeriesAnomalyDetector._empty_result("stl", len(series))

            # Z-score of residuals
            z_scores = np.abs(residuals / residual_std)
            is_anomaly_local = z_scores > threshold

            # Map back
            is_anomaly = np.zeros(len(series), dtype=bool)
            anomaly_scores = np.zeros(len(series))

            is_anomaly[series_clean.index] = is_anomaly_local
            anomaly_scores[series_clean.index] = z_scores

            return AnomalyResult(
                method="stl",
                is_anomaly=is_anomaly,
                anomaly_scores=anomaly_scores,
                threshold=float(threshold),
                n_anomalies=int(is_anomaly.sum()),
                anomaly_percentage=float(is_anomaly.sum() / len(is_anomaly) * 100),
                anomaly_indices=np.where(is_anomaly)[0].tolist(),
                metadata={
                    'period': int(period),
                    'threshold': float(threshold),
                    'residual_std': float(residual_std)
                }
            )
        except Exception as e:
            logger.warning(f"STL decomposition failed: {e}")
            return TimeSeriesAnomalyDetector._empty_result("stl", len(series))

    @staticmethod
    def _empty_result(method: str, n_samples: int) -> AnomalyResult:
        """Create empty result."""
        return AnomalyResult(
            method=method,
            is_anomaly=np.zeros(n_samples, dtype=bool),
            anomaly_scores=np.zeros(n_samples),
            threshold=None,
            n_anomalies=0,
            anomaly_percentage=0.0,
            anomaly_indices=[],
            metadata={}
        )


# ============================================================================
# Anomaly Explanation
# ============================================================================

class AnomalyExplainer:
    """Explain why points are flagged as anomalies."""

    @staticmethod
    def feature_contributions(
        data: pd.DataFrame,
        anomaly_indices: List[int],
        method: str = 'isolation_forest'
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculate feature contributions to anomaly scores.

        Args:
            data: Original data
            anomaly_indices: Indices of anomalies
            method: Detection method used

        Returns:
            Dictionary mapping anomaly index to feature contributions
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        X = data[numeric_cols]

        explanations = {}

        for idx in anomaly_indices:
            if idx >= len(data):
                continue

            point = X.iloc[idx]

            # Calculate how much each feature deviates from normal
            contributions = {}

            for col in numeric_cols:
                col_data = X[col].dropna()

                if len(col_data) == 0:
                    continue

                # Z-score deviation
                mean = col_data.mean()
                std = col_data.std()

                if std > 0:
                    z_score = abs((point[col] - mean) / std)
                    contributions[col] = float(z_score)
                else:
                    contributions[col] = 0.0

            # Sort by contribution
            explanations[idx] = dict(sorted(
                contributions.items(),
                key=lambda x: x[1],
                reverse=True
            ))

        return explanations

    @staticmethod
    def get_top_contributing_features(
        explanations: Dict[int, Dict[str, float]],
        top_n: int = 3
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top contributing features for each anomaly.

        Args:
            explanations: Output from feature_contributions
            top_n: Number of top features to return

        Returns:
            Dictionary mapping anomaly index to top features
        """
        result = {}

        for idx, contributions in explanations.items():
            top_features = list(contributions.items())[:top_n]
            result[idx] = top_features

        return result


# ============================================================================
# Main Anomaly Detection Manager
# ============================================================================

class AnomalyDetectionManager:
    """
    Main anomaly detection manager.

    Coordinates all anomaly detection methods.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        methods: Optional[List[str]] = None
    ):
        """
        Initialize anomaly detection manager.

        Args:
            contamination: Expected proportion of outliers (0.0 to 0.5)
            methods: List of methods to use (None = all available)
        """
        self.contamination = contamination

        if methods is None:
            methods = ['iqr', 'isolation_forest', 'lof']

        self.methods = methods

    def detect_univariate(
        self,
        data: Union[pd.Series, pd.DataFrame],
        methods: Optional[List[str]] = None
    ) -> Dict[str, AnomalyResult]:
        """
        Detect univariate anomalies.

        Args:
            data: Input data (Series or DataFrame with one column per variable)
            methods: Methods to use (overrides instance methods)

        Returns:
            Dictionary of results per method
        """
        if methods is None:
            methods = [m for m in self.methods if m in ['iqr', 'zscore', 'mad']]

        if isinstance(data, pd.Series):
            data = data.to_frame()

        results = {}

        for col in data.columns:
            col_results = {}

            if 'iqr' in methods:
                col_results['iqr'] = UnivariateAnomalyDetector.iqr_method(data[col])

            if 'zscore' in methods:
                col_results['zscore'] = UnivariateAnomalyDetector.zscore_method(data[col])

            if 'mad' in methods:
                col_results['mad'] = UnivariateAnomalyDetector.mad_method(data[col])

            results[col] = col_results

        return results

    def detect_multivariate(
        self,
        data: pd.DataFrame,
        methods: Optional[List[str]] = None
    ) -> Dict[str, AnomalyResult]:
        """
        Detect multivariate anomalies.

        Args:
            data: Input DataFrame
            methods: Methods to use

        Returns:
            Dictionary of results per method
        """
        if methods is None:
            methods = [m for m in self.methods if m in [
                'isolation_forest', 'lof', 'mahalanobis', 'elliptic_envelope', 'dbscan'
            ]]

        results = {}

        if 'isolation_forest' in methods:
            results['isolation_forest'] = MultivariateAnomalyDetector.isolation_forest(
                data, contamination=self.contamination
            )

        if 'lof' in methods:
            results['lof'] = MultivariateAnomalyDetector.local_outlier_factor(
                data, contamination=self.contamination
            )

        if 'mahalanobis' in methods:
            threshold_percentile = 100 * (1 - self.contamination)
            results['mahalanobis'] = MultivariateAnomalyDetector.mahalanobis_distance(
                data, threshold_percentile=threshold_percentile
            )

        if 'elliptic_envelope' in methods:
            results['elliptic_envelope'] = MultivariateAnomalyDetector.elliptic_envelope(
                data, contamination=self.contamination
            )

        if 'dbscan' in methods:
            results['dbscan'] = ClusteringAnomalyDetector.dbscan_method(data)

        return results

    def detect_timeseries(
        self,
        series: Union[pd.Series, np.ndarray],
        period: Optional[int] = None
    ) -> AnomalyResult:
        """
        Detect time series anomalies.

        Args:
            series: Time series data
            period: Seasonal period

        Returns:
            AnomalyResult
        """
        return TimeSeriesAnomalyDetector.stl_decomposition(series, period=period)

    def ensemble_detection(
        self,
        data: pd.DataFrame,
        voting_threshold: float = 0.5
    ) -> AnomalyResult:
        """
        Ensemble method: combine multiple detectors.

        Args:
            data: Input DataFrame
            voting_threshold: Proportion of methods that must agree (0.0-1.0)

        Returns:
            Combined AnomalyResult
        """
        # Run multiple methods
        results = self.detect_multivariate(data)

        if not results:
            return MultivariateAnomalyDetector._empty_result("ensemble", len(data))

        # Voting
        votes = np.zeros(len(data))

        for result in results.values():
            votes += result.is_anomaly.astype(int)

        # Require threshold proportion of methods to agree
        n_methods = len(results)
        is_anomaly = votes >= (voting_threshold * n_methods)

        # Average anomaly scores
        avg_scores = np.zeros(len(data))
        for result in results.values():
            # Normalize scores to [0, 1]
            if result.anomaly_scores.max() > 0:
                normalized = result.anomaly_scores / result.anomaly_scores.max()
            else:
                normalized = result.anomaly_scores
            avg_scores += normalized

        avg_scores /= n_methods

        return AnomalyResult(
            method="ensemble",
            is_anomaly=is_anomaly,
            anomaly_scores=avg_scores,
            threshold=float(voting_threshold),
            n_anomalies=int(is_anomaly.sum()),
            anomaly_percentage=float(is_anomaly.sum() / len(is_anomaly) * 100),
            anomaly_indices=np.where(is_anomaly)[0].tolist(),
            metadata={
                'methods_used': list(results.keys()),
                'voting_threshold': voting_threshold,
                'n_methods': n_methods
            }
        )

    def explain_anomalies(
        self,
        data: pd.DataFrame,
        result: AnomalyResult,
        top_n: int = 3
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Explain detected anomalies.

        Args:
            data: Original data
            result: Anomaly detection result
            top_n: Number of top contributing features

        Returns:
            Dictionary mapping anomaly index to top features
        """
        explanations = AnomalyExplainer.feature_contributions(
            data,
            result.anomaly_indices,
            method=result.method
        )

        return AnomalyExplainer.get_top_contributing_features(explanations, top_n)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'AnomalyDetectionManager',
    'UnivariateAnomalyDetector',
    'MultivariateAnomalyDetector',
    'ClusteringAnomalyDetector',
    'TimeSeriesAnomalyDetector',
    'AnomalyExplainer',
    'AnomalyResult',
    'AnomalyMethod',
]
