"""
Statistical analysis module for descriptive and inferential statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats


class StatisticalAnalyzer:
    """
    Performs statistical analysis on datasets.

    Provides comprehensive statistical analysis including:
    - Descriptive statistics (central tendency, dispersion, shape)
    - Distribution analysis (skewness, kurtosis, range)
    - Normality testing (Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov)
    - Percentile analysis
    - Categorical statistics
    """

    def __init__(self,
                 confidence_level: float = 0.95,
                 normality_alpha: float = 0.05):
        """
        Initialize StatisticalAnalyzer.

        Args:
            confidence_level: Confidence level for intervals (default 0.95)
            normality_alpha: Significance level for normality tests (default 0.05)
        """
        self.confidence_level = confidence_level
        self.normality_alpha = normality_alpha

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis on the dataset.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing statistical analysis results
        """
        results = {
            "descriptive": self._descriptive_stats(data),
            "distributions": self._analyze_distributions(data),
            "normality_tests": self._test_normality(data),
            "categorical": self._analyze_categorical(data),
            "summary": self._generate_summary(data),
        }
        return results

    def _descriptive_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive descriptive statistics.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing descriptive statistics for each column
        """
        if data.empty:
            return {
                "numeric_columns": {},
                "message": "Empty DataFrame"
            }

        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            return {
                "numeric_columns": {},
                "message": "No numeric columns found"
            }

        stats_by_column = {}

        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()

            if len(col_data) == 0:
                stats_by_column[col] = {
                    "message": "All values are missing"
                }
                continue

            # Central tendency
            mean_val = float(col_data.mean())
            median_val = float(col_data.median())
            mode_result = col_data.mode()
            mode_val = float(mode_result.iloc[0]) if len(mode_result) > 0 else None

            # Dispersion
            std_val = float(col_data.std())
            var_val = float(col_data.var())
            range_val = float(col_data.max() - col_data.min())
            iqr_val = float(col_data.quantile(0.75) - col_data.quantile(0.25))
            mad_val = float(np.median(np.abs(col_data - col_data.median())))

            # Percentiles
            percentiles = {
                "min": float(col_data.min()),
                "p5": float(col_data.quantile(0.05)),
                "p10": float(col_data.quantile(0.10)),
                "Q1": float(col_data.quantile(0.25)),
                "median": median_val,
                "Q3": float(col_data.quantile(0.75)),
                "p90": float(col_data.quantile(0.90)),
                "p95": float(col_data.quantile(0.95)),
                "max": float(col_data.max())
            }

            # Shape
            skewness = float(col_data.skew())
            kurtosis = float(col_data.kurtosis())

            # Coefficient of variation (CV)
            cv = (std_val / mean_val * 100) if mean_val != 0 else np.inf

            # Confidence interval for mean
            if len(col_data) > 1:
                ci = stats.t.interval(
                    self.confidence_level,
                    len(col_data) - 1,
                    loc=mean_val,
                    scale=stats.sem(col_data)
                )
                confidence_interval = {
                    "lower": float(ci[0]),
                    "upper": float(ci[1]),
                    "level": self.confidence_level
                }
            else:
                confidence_interval = None

            stats_by_column[col] = {
                "count": len(col_data),
                "missing": int(numeric_data[col].isna().sum()),
                "central_tendency": {
                    "mean": mean_val,
                    "median": median_val,
                    "mode": mode_val
                },
                "dispersion": {
                    "std": std_val,
                    "variance": var_val,
                    "range": range_val,
                    "iqr": iqr_val,
                    "mad": mad_val,
                    "cv": float(cv) if not np.isinf(cv) else None
                },
                "percentiles": percentiles,
                "shape": {
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "skewness_interpretation": self._interpret_skewness(skewness),
                    "kurtosis_interpretation": self._interpret_kurtosis(kurtosis)
                },
                "confidence_interval_mean": confidence_interval
            }

        return {
            "numeric_columns": stats_by_column,
            "total_numeric_columns": len(stats_by_column)
        }

    def _analyze_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze distribution characteristics of numeric columns.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing distribution analysis for each column
        """
        if data.empty:
            return {
                "columns": {},
                "message": "Empty DataFrame"
            }

        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            return {
                "columns": {},
                "message": "No numeric columns found"
            }

        distributions = {}

        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()

            if len(col_data) == 0:
                continue

            # Basic distribution metrics
            skewness = col_data.skew()
            kurtosis = col_data.kurtosis()

            # Value counts for discrete-like data
            unique_count = col_data.nunique()
            is_discrete = unique_count < 20

            distribution_info = {
                "unique_values": int(unique_count),
                "is_discrete": is_discrete,
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "distribution_type": self._classify_distribution(skewness, kurtosis),
            }

            # For discrete-like distributions, add value counts
            if is_discrete:
                value_counts = col_data.value_counts().head(10)
                distribution_info["value_counts"] = {
                    str(k): int(v) for k, v in value_counts.items()
                }
                distribution_info["mode_frequency"] = int(value_counts.iloc[0])
                distribution_info["mode_percentage"] = float(value_counts.iloc[0] / len(col_data) * 100)

            # Check for specific distribution patterns
            distribution_info["patterns"] = self._detect_patterns(col_data)

            distributions[col] = distribution_info

        return {
            "columns": distributions,
            "total_columns": len(distributions)
        }

    def _test_normality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test columns for normality using multiple statistical tests.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing normality test results for each column
        """
        if data.empty:
            return {
                "columns": {},
                "message": "Empty DataFrame"
            }

        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            return {
                "columns": {},
                "message": "No numeric columns found"
            }

        normality_results = {}

        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()

            if len(col_data) < 3:
                normality_results[col] = {
                    "message": "Insufficient data for normality testing (need at least 3 values)"
                }
                continue

            tests = {}

            # Shapiro-Wilk test (best for n < 5000)
            if len(col_data) <= 5000:
                try:
                    stat, p_value = stats.shapiro(col_data)
                    tests["shapiro_wilk"] = {
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "is_normal": p_value > self.normality_alpha,
                        "alpha": self.normality_alpha
                    }
                except Exception as e:
                    tests["shapiro_wilk"] = {"error": str(e)}

            # Kolmogorov-Smirnov test
            try:
                stat, p_value = stats.kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                tests["kolmogorov_smirnov"] = {
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "is_normal": p_value > self.normality_alpha,
                    "alpha": self.normality_alpha
                }
            except Exception as e:
                tests["kolmogorov_smirnov"] = {"error": str(e)}

            # Anderson-Darling test
            try:
                result = stats.anderson(col_data, dist='norm')
                # For 5% significance level (index 2)
                critical_value_5pct = result.critical_values[2]
                tests["anderson_darling"] = {
                    "statistic": float(result.statistic),
                    "critical_values": result.critical_values.tolist(),
                    "significance_levels": result.significance_level.tolist(),
                    "is_normal": result.statistic < critical_value_5pct,
                    "critical_value_5pct": float(critical_value_5pct)
                }
            except Exception as e:
                tests["anderson_darling"] = {"error": str(e)}

            # D'Agostino's K-squared test
            if len(col_data) >= 8:
                try:
                    stat, p_value = stats.normaltest(col_data)
                    tests["dagostino"] = {
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "is_normal": p_value > self.normality_alpha,
                        "alpha": self.normality_alpha
                    }
                except Exception as e:
                    tests["dagostino"] = {"error": str(e)}

            # Overall assessment
            normal_count = sum(1 for test in tests.values()
                             if isinstance(test, dict) and test.get("is_normal", False))
            total_tests = sum(1 for test in tests.values()
                            if isinstance(test, dict) and "is_normal" in test)

            normality_results[col] = {
                "tests": tests,
                "consensus": {
                    "tests_passed": normal_count,
                    "total_tests": total_tests,
                    "percentage_normal": (normal_count / total_tests * 100) if total_tests > 0 else 0,
                    "likely_normal": normal_count >= total_tests / 2 if total_tests > 0 else False
                }
            }

        return {
            "columns": normality_results,
            "total_columns": len(normality_results),
            "alpha": self.normality_alpha
        }

    def _analyze_categorical(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze categorical columns.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing categorical analysis
        """
        if data.empty:
            return {
                "columns": {},
                "message": "Empty DataFrame"
            }

        categorical_data = data.select_dtypes(include=['object', 'category'])

        if categorical_data.empty:
            return {
                "columns": {},
                "message": "No categorical columns found"
            }

        categorical_stats = {}

        for col in categorical_data.columns:
            col_data = data[col].dropna()

            if len(col_data) == 0:
                continue

            # Value counts
            value_counts = col_data.value_counts()

            # Calculate entropy
            probabilities = value_counts / len(col_data)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

            # Mode analysis
            mode_val = value_counts.index[0] if len(value_counts) > 0 else None
            mode_count = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            mode_percentage = float(mode_count / len(col_data) * 100)

            categorical_stats[col] = {
                "count": len(col_data),
                "missing": int(data[col].isna().sum()),
                "unique_values": int(col_data.nunique()),
                "mode": str(mode_val),
                "mode_count": mode_count,
                "mode_percentage": mode_percentage,
                "entropy": float(entropy),
                "max_entropy": float(np.log2(col_data.nunique())),
                "diversity_index": float(entropy / np.log2(col_data.nunique())) if col_data.nunique() > 1 else 0,
                "top_10_values": {
                    str(k): int(v) for k, v in value_counts.head(10).items()
                }
            }

        return {
            "columns": categorical_stats,
            "total_categorical_columns": len(categorical_stats)
        }

    def _generate_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate overall statistical summary.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary containing overall summary statistics
        """
        if data.empty:
            return {"message": "Empty DataFrame"}

        numeric_data = data.select_dtypes(include=[np.number])
        categorical_data = data.select_dtypes(include=['object', 'category'])

        summary = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "numeric_columns": len(numeric_data.columns),
            "categorical_columns": len(categorical_data.columns),
            "other_columns": len(data.columns) - len(numeric_data.columns) - len(categorical_data.columns),
            "total_missing": int(data.isna().sum().sum()),
            "memory_usage_mb": float(data.memory_usage(deep=True).sum() / 1024 / 1024)
        }

        # Numeric summary
        if not numeric_data.empty:
            summary["numeric_summary"] = {
                "mean_of_means": float(numeric_data.mean().mean()),
                "mean_of_stds": float(numeric_data.std().mean()),
                "highly_variable_columns": [
                    col for col in numeric_data.columns
                    if numeric_data[col].std() / (numeric_data[col].mean() + 1e-10) > 1
                ]
            }

        # Categorical summary
        if not categorical_data.empty:
            summary["categorical_summary"] = {
                "avg_unique_values": float(categorical_data.nunique().mean()),
                "high_cardinality_columns": [
                    col for col in categorical_data.columns
                    if data[col].nunique() > 50
                ]
            }

        return summary

    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness value."""
        abs_skew = abs(skewness)
        if abs_skew < 0.5:
            return "approximately symmetric"
        elif abs_skew < 1:
            return "moderately skewed"
        else:
            direction = "right (positive)" if skewness > 0 else "left (negative)"
            return f"highly skewed {direction}"

    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpret kurtosis value."""
        if kurtosis < -1:
            return "platykurtic (flat, thin tails)"
        elif kurtosis > 1:
            return "leptokurtic (peaked, fat tails)"
        else:
            return "mesokurtic (normal-like tails)"

    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """Classify distribution type based on skewness and kurtosis."""
        abs_skew = abs(skewness)
        abs_kurt = abs(kurtosis)

        if abs_skew < 0.5 and abs_kurt < 0.5:
            return "approximately normal"
        elif abs_skew < 0.5:
            return "symmetric non-normal"
        elif skewness > 0:
            return "right-skewed"
        else:
            return "left-skewed"

    def _detect_patterns(self, data: pd.Series) -> Dict[str, Any]:
        """Detect common patterns in the data."""
        patterns = {}

        # Check for uniform distribution
        if len(data) > 10:
            _, p_value = stats.kstest(data, 'uniform',
                                     args=(data.min(), data.max() - data.min()))
            patterns["uniform_like"] = p_value > 0.05

        # Check for exponential pattern
        if (data > 0).all() and len(data) > 10:
            try:
                _, p_value = stats.kstest(data, 'expon',
                                         args=(data.min(), data.mean() - data.min()))
                patterns["exponential_like"] = p_value > 0.05
            except:
                patterns["exponential_like"] = False

        # Check for multimodal
        if len(data) > 20:
            hist, _ = np.histogram(data, bins='auto')
            peaks = np.where(np.diff(np.sign(np.diff(hist))) < 0)[0]
            patterns["potential_modes"] = int(len(peaks) + 1)
            patterns["multimodal"] = len(peaks) > 1

        return patterns
