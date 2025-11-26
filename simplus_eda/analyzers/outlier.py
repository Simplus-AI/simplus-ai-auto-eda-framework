"""
Outlier detection and analysis module.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Literal, List, Optional
from scipy import stats
from ..utils.parallel import ParallelProcessor


class OutlierAnalyzer:
    """
    Detect and analyze outliers using various methods.

    Supported methods:
    - IQR (Interquartile Range): Classical method using Q1 and Q3
    - Z-score: Statistical method based on standard deviations
    - Isolation Forest: Machine learning-based anomaly detection
    - Modified Z-score: Robust alternative using median absolute deviation

    Supports parallel processing for analyzing multiple columns simultaneously.
    """

    def __init__(
        self,
        method: Literal["iqr", "zscore", "isolation_forest", "modified_zscore", "all"] = "iqr",
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        contamination: float = 0.1,
        n_jobs: int = 1,
        verbose: bool = False
    ):
        """
        Initialize OutlierAnalyzer.

        Args:
            method: Method for outlier detection
            iqr_multiplier: Multiplier for IQR method (default 1.5)
            zscore_threshold: Threshold for z-score method (default 3.0)
            contamination: Expected proportion of outliers for Isolation Forest (default 0.1)
            n_jobs: Number of parallel jobs (-1 for all CPUs, default 1)
            verbose: Enable verbose output (default False)
        """
        self.method = method
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.contamination = contamination
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.parallel_processor = ParallelProcessor(n_jobs=n_jobs, verbose=verbose)

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform outlier detection and analysis.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing outlier analysis results
        """
        if self.method == "all":
            results = {
                "methods_used": ["iqr", "zscore", "modified_zscore", "isolation_forest"],
                "iqr": self._detect_iqr_outliers(data),
                "zscore": self._detect_zscore_outliers(data),
                "modified_zscore": self._detect_modified_zscore_outliers(data),
                "isolation_forest": self._detect_isolation_forest_outliers(data),
                "summary": self._summarize_all_methods(data),
            }
        else:
            results = {
                "method": self.method,
                "outliers": self._detect_outliers(data),
                "summary": self._summarize_outliers(data),
            }
        return results

    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers based on configured method."""
        if self.method == "iqr":
            return self._detect_iqr_outliers(data)
        elif self.method == "zscore":
            return self._detect_zscore_outliers(data)
        elif self.method == "modified_zscore":
            return self._detect_modified_zscore_outliers(data)
        elif self.method == "isolation_forest":
            return self._detect_isolation_forest_outliers(data)
        else:
            return {"error": f"Unknown method: {self.method}"}

    def _detect_iqr_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers using IQR (Interquartile Range) method.

        Outliers are values that fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing outlier information per column
        """
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            return {
                "outliers_by_column": {},
                "message": "No numeric columns found for outlier detection"
            }

        outliers_by_column = {}

        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()

            if len(col_data) == 0:
                continue

            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR

            # Find outlier indices and values
            outlier_mask = (numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)
            outlier_indices = numeric_data[outlier_mask].index.tolist()
            outlier_values = numeric_data.loc[outlier_mask, col].tolist()

            outliers_by_column[col] = {
                "count": len(outlier_indices),
                "percentage": (len(outlier_indices) / len(col_data)) * 100,
                "indices": outlier_indices,
                "values": outlier_values,
                "bounds": {
                    "lower": float(lower_bound),
                    "upper": float(upper_bound),
                    "Q1": float(Q1),
                    "Q3": float(Q3),
                    "IQR": float(IQR)
                },
                "below_lower": int(sum(numeric_data[col] < lower_bound)),
                "above_upper": int(sum(numeric_data[col] > upper_bound))
            }

        return {
            "outliers_by_column": outliers_by_column,
            "total_columns": len(outliers_by_column),
            "iqr_multiplier": self.iqr_multiplier
        }

    def _detect_zscore_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers using Z-score method.

        Outliers are values with |z-score| > threshold (default 3)

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing outlier information per column
        """
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            return {
                "outliers_by_column": {},
                "message": "No numeric columns found for outlier detection"
            }

        outliers_by_column = {}

        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()

            if len(col_data) == 0 or col_data.std() == 0:
                continue

            # Calculate z-scores
            z_scores = np.abs(stats.zscore(col_data))

            # Find outliers
            outlier_mask = z_scores > self.zscore_threshold
            outlier_indices = col_data[outlier_mask].index.tolist()
            outlier_values = col_data[outlier_mask].tolist()
            outlier_zscores = z_scores[outlier_mask].tolist()

            outliers_by_column[col] = {
                "count": len(outlier_indices),
                "percentage": (len(outlier_indices) / len(col_data)) * 100,
                "indices": outlier_indices,
                "values": outlier_values,
                "z_scores": outlier_zscores,
                "statistics": {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "threshold": self.zscore_threshold
                }
            }

        return {
            "outliers_by_column": outliers_by_column,
            "total_columns": len(outliers_by_column),
            "zscore_threshold": self.zscore_threshold
        }

    def _detect_modified_zscore_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers using Modified Z-score method (MAD-based).

        More robust to outliers than standard Z-score.
        Uses median absolute deviation instead of standard deviation.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing outlier information per column
        """
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            return {
                "outliers_by_column": {},
                "message": "No numeric columns found for outlier detection"
            }

        outliers_by_column = {}
        threshold = 3.5  # Common threshold for modified z-score

        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()

            if len(col_data) == 0:
                continue

            median = col_data.median()
            mad = np.median(np.abs(col_data - median))

            if mad == 0:
                continue

            # Modified z-score = 0.6745 * (x - median) / MAD
            modified_z_scores = np.abs(0.6745 * (col_data - median) / mad)

            # Find outliers
            outlier_mask = modified_z_scores > threshold
            outlier_indices = col_data[outlier_mask].index.tolist()
            outlier_values = col_data[outlier_mask].tolist()
            outlier_zscores = modified_z_scores[outlier_mask].tolist()

            outliers_by_column[col] = {
                "count": len(outlier_indices),
                "percentage": (len(outlier_indices) / len(col_data)) * 100,
                "indices": outlier_indices,
                "values": outlier_values,
                "modified_z_scores": outlier_zscores,
                "statistics": {
                    "median": float(median),
                    "mad": float(mad),
                    "threshold": threshold
                }
            }

        return {
            "outliers_by_column": outliers_by_column,
            "total_columns": len(outliers_by_column),
            "threshold": threshold
        }

    def _detect_isolation_forest_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers using Isolation Forest algorithm.

        Machine learning-based anomaly detection that works well for
        multivariate outlier detection.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing outlier information
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            return {
                "error": "scikit-learn is required for Isolation Forest method",
                "message": "Install with: pip install scikit-learn"
            }

        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            return {
                "outliers": [],
                "message": "No numeric columns found for outlier detection"
            }

        # Remove rows with any NaN values for Isolation Forest
        clean_data = numeric_data.dropna()

        if clean_data.empty or len(clean_data) < 2:
            return {
                "outliers": [],
                "message": "Not enough data for Isolation Forest after removing missing values"
            }

        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )

        predictions = iso_forest.fit_predict(clean_data)
        anomaly_scores = iso_forest.score_samples(clean_data)

        # -1 indicates outlier, 1 indicates inlier
        outlier_mask = predictions == -1
        outlier_indices = clean_data[outlier_mask].index.tolist()

        outlier_records = []
        for idx in outlier_indices:
            record = {
                "index": int(idx),
                "anomaly_score": float(anomaly_scores[clean_data.index == idx][0]),
                "values": clean_data.loc[idx].to_dict()
            }
            outlier_records.append(record)

        # Sort by anomaly score (most anomalous first)
        outlier_records.sort(key=lambda x: x["anomaly_score"])

        return {
            "outliers": outlier_records,
            "count": len(outlier_indices),
            "percentage": (len(outlier_indices) / len(clean_data)) * 100,
            "contamination": self.contamination,
            "total_samples": len(clean_data),
            "features_used": list(clean_data.columns)
        }

    def _summarize_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Summarize outlier statistics for a single method."""
        outliers = self._detect_outliers(data)

        if "error" in outliers or "message" in outliers:
            return outliers

        if self.method == "isolation_forest":
            return {
                "total_outliers": outliers.get("count", 0),
                "percentage": outliers.get("percentage", 0),
                "method": self.method
            }
        else:
            outliers_by_col = outliers.get("outliers_by_column", {})

            total_outliers = sum(col_info["count"] for col_info in outliers_by_col.values())
            columns_with_outliers = [col for col, info in outliers_by_col.items() if info["count"] > 0]

            return {
                "total_outliers": total_outliers,
                "columns_with_outliers": len(columns_with_outliers),
                "columns_list": columns_with_outliers,
                "outlier_counts_by_column": {
                    col: info["count"] for col, info in outliers_by_col.items()
                },
                "method": self.method
            }

    def _summarize_all_methods(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Summarize outliers detected by all methods."""
        summary = {
            "methods": {}
        }

        methods = ["iqr", "zscore", "modified_zscore", "isolation_forest"]

        for method in methods:
            temp_analyzer = OutlierAnalyzer(method=method)
            method_outliers = temp_analyzer._detect_outliers(data)

            if method == "isolation_forest":
                count = method_outliers.get("count", 0)
            else:
                outliers_by_col = method_outliers.get("outliers_by_column", {})
                count = sum(col_info["count"] for col_info in outliers_by_col.values())

            summary["methods"][method] = {
                "total_outliers": count
            }

        return summary
