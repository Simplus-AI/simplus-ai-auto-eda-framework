"""
Data quality assessment module.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from collections import Counter


class DataQualityAnalyzer:
    """
    Assess data quality metrics including missing values, duplicates, and consistency.

    Provides comprehensive data quality analysis including:
    - Missing value patterns and statistics
    - Duplicate detection and analysis
    - Data type analysis and recommendations
    - Data consistency checks
    - Cardinality analysis
    - Value distribution checks
    """

    def __init__(self,
                 missing_threshold: float = 0.5,
                 duplicate_subset: Optional[List[str]] = None,
                 high_cardinality_threshold: int = 50):
        """
        Initialize DataQualityAnalyzer.

        Args:
            missing_threshold: Threshold for flagging high missing rate (default 0.5)
            duplicate_subset: Columns to check for duplicates (None means all columns)
            high_cardinality_threshold: Threshold for high cardinality warning
        """
        self.missing_threshold = missing_threshold
        self.duplicate_subset = duplicate_subset
        self.high_cardinality_threshold = high_cardinality_threshold

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing quality assessment results
        """
        results = {
            "missing_values": self._analyze_missing(data),
            "duplicates": self._analyze_duplicates(data),
            "data_types": self._analyze_types(data),
            "consistency": self._check_consistency(data),
            "cardinality": self._analyze_cardinality(data),
            "quality_score": self._calculate_quality_score(data),
        }
        return results

    def _analyze_missing(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missing values patterns.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing missing value analysis
        """
        if data.empty:
            return {
                "total_missing": 0,
                "total_cells": 0,
                "overall_missing_rate": 0.0,
                "columns": {},
                "message": "Empty DataFrame"
            }

        total_cells = data.shape[0] * data.shape[1]
        total_missing = data.isna().sum().sum()
        overall_missing_rate = (total_missing / total_cells * 100) if total_cells > 0 else 0.0

        columns_info = {}
        high_missing_columns = []

        for col in data.columns:
            missing_count = data[col].isna().sum()
            missing_rate = (missing_count / len(data) * 100) if len(data) > 0 else 0.0

            # Find missing value patterns
            missing_indices = data[data[col].isna()].index.tolist()

            columns_info[col] = {
                "missing_count": int(missing_count),
                "missing_rate": float(missing_rate),
                "missing_indices": missing_indices[:100],  # Limit to first 100
                "has_missing": missing_count > 0
            }

            if missing_rate > (self.missing_threshold * 100):
                high_missing_columns.append({
                    "column": col,
                    "missing_rate": float(missing_rate),
                    "missing_count": int(missing_count)
                })

        # Sort high missing columns by rate
        high_missing_columns.sort(key=lambda x: x["missing_rate"], reverse=True)

        # Analyze missing value patterns
        missing_patterns = self._analyze_missing_patterns(data)

        return {
            "total_missing": int(total_missing),
            "total_cells": int(total_cells),
            "overall_missing_rate": float(overall_missing_rate),
            "columns": columns_info,
            "high_missing_columns": high_missing_columns,
            "high_missing_count": len(high_missing_columns),
            "missing_threshold": self.missing_threshold,
            "patterns": missing_patterns,
            "recommendation": self._get_missing_recommendation(overall_missing_rate, len(high_missing_columns))
        }

    def _analyze_missing_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze patterns in missing values.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary with missing value pattern information
        """
        # Find rows with all values missing
        all_missing_rows = data[data.isna().all(axis=1)].index.tolist()

        # Find rows with any values missing
        any_missing_rows = data[data.isna().any(axis=1)].index.tolist()

        # Find columns with all values missing
        all_missing_cols = [col for col in data.columns if data[col].isna().all()]

        # Calculate completeness by row
        row_completeness = ((~data.isna()).sum(axis=1) / len(data.columns) * 100).describe()

        return {
            "rows_all_missing": len(all_missing_rows),
            "rows_any_missing": len(any_missing_rows),
            "columns_all_missing": all_missing_cols,
            "row_completeness_stats": {
                "mean": float(row_completeness['mean']),
                "min": float(row_completeness['min']),
                "max": float(row_completeness['max']),
                "std": float(row_completeness['std'])
            }
        }

    def _analyze_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect and analyze duplicate rows.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing duplicate analysis
        """
        if data.empty:
            return {
                "duplicate_rows": 0,
                "duplicate_rate": 0.0,
                "message": "Empty DataFrame"
            }

        # Check for exact duplicates
        subset_cols = self.duplicate_subset if self.duplicate_subset else None
        duplicate_mask = data.duplicated(subset=subset_cols, keep=False)
        duplicate_count = duplicate_mask.sum()
        duplicate_rate = (duplicate_count / len(data) * 100) if len(data) > 0 else 0.0

        # Get duplicate indices
        duplicate_indices = data[duplicate_mask].index.tolist()

        # Find unique duplicate groups
        if duplicate_count > 0:
            duplicate_groups = []
            seen_indices = set()

            for idx in duplicate_indices:
                if idx in seen_indices:
                    continue

                # Find all rows that match this row
                if subset_cols:
                    matching_mask = (data[subset_cols] == data.loc[idx, subset_cols]).all(axis=1)
                else:
                    matching_mask = (data == data.loc[idx]).all(axis=1)

                matching_indices = data[matching_mask].index.tolist()

                if len(matching_indices) > 1:
                    duplicate_groups.append({
                        "indices": matching_indices,
                        "count": len(matching_indices)
                    })
                    seen_indices.update(matching_indices)

            # Limit to first 10 groups for performance
            duplicate_groups = duplicate_groups[:10]
        else:
            duplicate_groups = []

        # Analyze duplicates by column combinations
        column_duplicates = {}
        for col in data.columns:
            col_dup_count = data[col].duplicated().sum()
            col_unique_count = data[col].nunique()
            column_duplicates[col] = {
                "duplicate_count": int(col_dup_count),
                "unique_count": int(col_unique_count),
                "total_count": len(data)
            }

        return {
            "duplicate_rows": int(duplicate_count),
            "duplicate_rate": float(duplicate_rate),
            "unique_rows": int(len(data) - duplicate_count + len(duplicate_groups)),
            "duplicate_indices": duplicate_indices[:100],  # Limit to first 100
            "duplicate_groups": duplicate_groups,
            "duplicate_groups_count": len(duplicate_groups),
            "column_duplicates": column_duplicates,
            "subset_columns": subset_cols,
            "recommendation": self._get_duplicate_recommendation(duplicate_rate)
        }

    def _analyze_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data types and suggest improvements.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing data type analysis
        """
        if data.empty:
            return {
                "columns": {},
                "message": "Empty DataFrame"
            }

        type_info = {}
        type_counts = Counter()
        suggestions = []

        for col in data.columns:
            dtype = str(data[col].dtype)
            type_counts[dtype] += 1

            # Analyze column characteristics
            col_info = {
                "dtype": dtype,
                "null_count": int(data[col].isna().sum()),
                "unique_count": int(data[col].nunique()),
                "memory_usage": int(data[col].memory_usage(deep=True))
            }

            # Type-specific analysis
            if pd.api.types.is_numeric_dtype(data[col]):
                col_info["is_numeric"] = True
                col_info["stats"] = {
                    "min": float(data[col].min()) if not data[col].isna().all() else None,
                    "max": float(data[col].max()) if not data[col].isna().all() else None,
                    "mean": float(data[col].mean()) if not data[col].isna().all() else None
                }

                # Check if integer column could be categorical
                if data[col].nunique() < 10 and len(data) > 20:
                    suggestions.append({
                        "column": col,
                        "current_type": dtype,
                        "suggested_type": "category",
                        "reason": f"Only {data[col].nunique()} unique values, consider categorical"
                    })

            elif pd.api.types.is_object_dtype(data[col]):
                col_info["is_object"] = True

                # Try to detect if it could be numeric
                non_null = data[col].dropna()
                if len(non_null) > 0:
                    try:
                        pd.to_numeric(non_null, errors='raise')
                        suggestions.append({
                            "column": col,
                            "current_type": dtype,
                            "suggested_type": "numeric",
                            "reason": "All non-null values can be converted to numeric"
                        })
                    except (ValueError, TypeError):
                        pass

                    # Try to detect datetime
                    try:
                        pd.to_datetime(non_null, errors='raise')
                        suggestions.append({
                            "column": col,
                            "current_type": dtype,
                            "suggested_type": "datetime",
                            "reason": "Values appear to be datetime strings"
                        })
                    except (ValueError, TypeError):
                        pass

                # Check if should be categorical
                if data[col].nunique() < len(data) * 0.5 and data[col].nunique() < 100:
                    suggestions.append({
                        "column": col,
                        "current_type": dtype,
                        "suggested_type": "category",
                        "reason": f"Only {data[col].nunique()} unique values out of {len(data)}"
                    })

            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                col_info["is_datetime"] = True
                col_info["date_range"] = {
                    "min": str(data[col].min()),
                    "max": str(data[col].max())
                }

            elif pd.api.types.is_categorical_dtype(data[col]):
                col_info["is_categorical"] = True
                col_info["categories"] = list(data[col].cat.categories)[:20]  # Limit to 20

            type_info[col] = col_info

        # Calculate memory usage
        total_memory = data.memory_usage(deep=True).sum()

        return {
            "columns": type_info,
            "type_counts": dict(type_counts),
            "total_memory_bytes": int(total_memory),
            "total_memory_mb": float(total_memory / 1024 / 1024),
            "suggestions": suggestions,
            "suggestion_count": len(suggestions)
        }

    def _check_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data consistency and validity.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing consistency checks
        """
        if data.empty:
            return {
                "issues": [],
                "message": "Empty DataFrame"
            }

        issues = []
        numeric_checks = {}
        text_checks = {}

        for col in data.columns:
            # Numeric consistency checks
            if pd.api.types.is_numeric_dtype(data[col]):
                non_null = data[col].dropna()

                if len(non_null) > 0:
                    # Check for infinite values
                    inf_count = np.isinf(non_null).sum()
                    if inf_count > 0:
                        issues.append({
                            "column": col,
                            "issue_type": "infinite_values",
                            "count": int(inf_count),
                            "severity": "high"
                        })

                    # Check for negative values in potentially positive-only columns
                    if any(keyword in col.lower() for keyword in ['age', 'count', 'price', 'amount', 'quantity']):
                        negative_count = (non_null < 0).sum()
                        if negative_count > 0:
                            issues.append({
                                "column": col,
                                "issue_type": "unexpected_negative",
                                "count": int(negative_count),
                                "severity": "medium"
                            })

                    numeric_checks[col] = {
                        "has_infinity": int(inf_count) > 0,
                        "infinity_count": int(inf_count),
                        "has_negative": (non_null < 0).any(),
                        "negative_count": int((non_null < 0).sum()),
                        "has_zero": (non_null == 0).any(),
                        "zero_count": int((non_null == 0).sum())
                    }

            # Text consistency checks
            elif pd.api.types.is_object_dtype(data[col]):
                non_null = data[col].dropna()

                if len(non_null) > 0:
                    # Check for whitespace issues
                    if non_null.astype(str).str.strip().ne(non_null.astype(str)).any():
                        whitespace_count = non_null.astype(str).str.strip().ne(non_null.astype(str)).sum()
                        issues.append({
                            "column": col,
                            "issue_type": "leading_trailing_whitespace",
                            "count": int(whitespace_count),
                            "severity": "low"
                        })

                    # Check for empty strings
                    empty_string_count = (non_null.astype(str) == '').sum()
                    if empty_string_count > 0:
                        issues.append({
                            "column": col,
                            "issue_type": "empty_strings",
                            "count": int(empty_string_count),
                            "severity": "medium"
                        })

                    # Check for case inconsistency
                    unique_values = non_null.unique()
                    lower_values = pd.Series(unique_values).str.lower()
                    if len(unique_values) != len(lower_values.unique()):
                        issues.append({
                            "column": col,
                            "issue_type": "case_inconsistency",
                            "severity": "low"
                        })

                    text_checks[col] = {
                        "has_whitespace_issues": int(non_null.astype(str).str.strip().ne(non_null.astype(str)).sum()) > 0,
                        "has_empty_strings": int(empty_string_count) > 0,
                        "empty_string_count": int(empty_string_count),
                        "max_length": int(non_null.astype(str).str.len().max()),
                        "min_length": int(non_null.astype(str).str.len().min()),
                        "avg_length": float(non_null.astype(str).str.len().mean())
                    }

        # Sort issues by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        issues.sort(key=lambda x: severity_order.get(x["severity"], 3))

        return {
            "issues": issues,
            "issue_count": len(issues),
            "numeric_checks": numeric_checks,
            "text_checks": text_checks,
            "has_critical_issues": any(issue["severity"] == "high" for issue in issues)
        }

    def _analyze_cardinality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze cardinality (unique values) for each column.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing cardinality analysis
        """
        if data.empty:
            return {
                "columns": {},
                "message": "Empty DataFrame"
            }

        cardinality_info = {}
        high_cardinality_cols = []
        low_cardinality_cols = []

        for col in data.columns:
            unique_count = data[col].nunique()
            total_count = len(data)
            cardinality_rate = (unique_count / total_count * 100) if total_count > 0 else 0.0

            col_info = {
                "unique_count": int(unique_count),
                "total_count": int(total_count),
                "cardinality_rate": float(cardinality_rate),
                "is_unique": unique_count == total_count and total_count > 0
            }

            # Classify cardinality
            if unique_count == 1:
                col_info["cardinality_level"] = "constant"
            elif unique_count == total_count:
                col_info["cardinality_level"] = "unique"
            elif unique_count <= 10:
                col_info["cardinality_level"] = "low"
                low_cardinality_cols.append(col)
            elif unique_count > self.high_cardinality_threshold:
                col_info["cardinality_level"] = "high"
                high_cardinality_cols.append(col)
            else:
                col_info["cardinality_level"] = "medium"

            # Get top values for low/medium cardinality
            if unique_count <= 20:
                value_counts = data[col].value_counts().head(10)
                col_info["top_values"] = {
                    str(k): int(v) for k, v in value_counts.items()
                }

            cardinality_info[col] = col_info

        return {
            "columns": cardinality_info,
            "high_cardinality_columns": high_cardinality_cols,
            "low_cardinality_columns": low_cardinality_cols,
            "high_cardinality_threshold": self.high_cardinality_threshold
        }

    def _calculate_quality_score(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate overall data quality score.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary with quality score and components
        """
        if data.empty:
            return {
                "overall_score": 0.0,
                "message": "Empty DataFrame"
            }

        # Component scores (0-100)
        scores = {}

        # 1. Completeness score (missing values)
        total_cells = data.shape[0] * data.shape[1]
        total_missing = data.isna().sum().sum()
        completeness = ((total_cells - total_missing) / total_cells * 100) if total_cells > 0 else 0.0
        scores["completeness"] = float(completeness)

        # 2. Uniqueness score (duplicates)
        duplicate_count = data.duplicated().sum()
        uniqueness = ((len(data) - duplicate_count) / len(data) * 100) if len(data) > 0 else 0.0
        scores["uniqueness"] = float(uniqueness)

        # 3. Consistency score (issues)
        consistency_result = self._check_consistency(data)
        issue_count = consistency_result.get("issue_count", 0)
        max_issues = len(data.columns) * 3  # Rough estimate
        consistency = max(0, (1 - issue_count / max_issues) * 100) if max_issues > 0 else 100.0
        scores["consistency"] = float(consistency)

        # 4. Validity score (data types)
        type_result = self._analyze_types(data)
        suggestion_count = type_result.get("suggestion_count", 0)
        validity = max(0, (1 - suggestion_count / len(data.columns)) * 100) if len(data.columns) > 0 else 100.0
        scores["validity"] = float(validity)

        # Calculate weighted overall score
        weights = {
            "completeness": 0.35,
            "uniqueness": 0.25,
            "consistency": 0.20,
            "validity": 0.20
        }

        overall_score = sum(scores[key] * weights[key] for key in weights.keys())

        # Determine quality level
        if overall_score >= 90:
            quality_level = "excellent"
        elif overall_score >= 75:
            quality_level = "good"
        elif overall_score >= 60:
            quality_level = "fair"
        elif overall_score >= 40:
            quality_level = "poor"
        else:
            quality_level = "critical"

        return {
            "overall_score": float(overall_score),
            "quality_level": quality_level,
            "component_scores": scores,
            "weights": weights,
            "interpretation": self._interpret_quality_score(overall_score)
        }

    def _get_missing_recommendation(self, missing_rate: float, high_missing_count: int) -> str:
        """Generate recommendation based on missing value analysis."""
        if missing_rate == 0:
            return "Excellent! No missing values detected."
        elif missing_rate < 5:
            return f"Low missing rate ({missing_rate:.1f}%). Data quality is good."
        elif missing_rate < 20:
            return f"Moderate missing rate ({missing_rate:.1f}%). Consider imputation strategies."
        else:
            return f"High missing rate ({missing_rate:.1f}%) with {high_missing_count} problematic columns. Investigate data collection process."

    def _get_duplicate_recommendation(self, duplicate_rate: float) -> str:
        """Generate recommendation based on duplicate analysis."""
        if duplicate_rate == 0:
            return "No duplicate rows detected. Data uniqueness is excellent."
        elif duplicate_rate < 5:
            return f"Low duplicate rate ({duplicate_rate:.1f}%). Consider removing duplicates."
        elif duplicate_rate < 20:
            return f"Moderate duplicate rate ({duplicate_rate:.1f}%). Review and remove duplicates."
        else:
            return f"High duplicate rate ({duplicate_rate:.1f}%). Significant duplicates detected - investigate data source."

    def _interpret_quality_score(self, score: float) -> str:
        """Interpret the overall quality score."""
        if score >= 90:
            return "Data quality is excellent. Minimal issues detected."
        elif score >= 75:
            return "Data quality is good. Minor improvements possible."
        elif score >= 60:
            return "Data quality is fair. Several issues should be addressed."
        elif score >= 40:
            return "Data quality is poor. Significant improvements needed."
        else:
            return "Data quality is critical. Major issues require immediate attention."
