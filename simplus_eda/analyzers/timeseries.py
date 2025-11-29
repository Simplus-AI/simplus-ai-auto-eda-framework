"""
Time series analysis module for the EDA framework.

This module provides comprehensive time series analysis including:
- ARIMA/SARIMA forecasting
- Trend detection and decomposition
- Seasonality strength metrics
- ACF/PACF analysis
- Multiple time series comparison
- Stationarity testing

Features:
- Automatic model selection
- Seasonal decomposition (additive/multiplicative)
- Trend strength and seasonality strength metrics
- Autocorrelation analysis
- Cross-correlation for comparison
- Anomaly detection in time series
"""

import warnings
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy import stats, signal
from scipy.stats import kurtosis, skew

from simplus_eda.logging_config import get_logger
from simplus_eda.exceptions import AnalysisError
from simplus_eda.progress import ProgressTracker

logger = get_logger(__name__)

# Try to import optional dependencies
try:
    from statsmodels.tsa.stattools import (
        adfuller,
        kpss,
        acf,
        pacf,
        ccf
    )
    from statsmodels.tsa.seasonal import seasonal_decompose, STL
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available. Install with: pip install statsmodels")


# ============================================================================
# Time Series Validator
# ============================================================================

class TimeSeriesValidator:
    """Validate and prepare time series data."""

    @staticmethod
    def validate_series(
        data: Union[pd.Series, pd.DataFrame],
        datetime_col: Optional[str] = None,
        value_col: Optional[str] = None
    ) -> pd.Series:
        """
        Validate and extract time series.

        Args:
            data: Time series data
            datetime_col: DateTime column name (for DataFrame)
            value_col: Value column name (for DataFrame)

        Returns:
            Validated Series with DatetimeIndex

        Raises:
            AnalysisError: If data is invalid
        """
        if isinstance(data, pd.Series):
            series = data.copy()
        elif isinstance(data, pd.DataFrame):
            if value_col is None:
                raise AnalysisError(
                    "value_col required for DataFrame input",
                    details={'columns': list(data.columns)}
                )
            series = data[value_col].copy()

            if datetime_col and datetime_col in data.columns:
                series.index = pd.to_datetime(data[datetime_col])
        else:
            raise AnalysisError(
                "Data must be Series or DataFrame",
                details={'type': type(data).__name__}
            )

        # Ensure numeric
        if not pd.api.types.is_numeric_dtype(series):
            raise AnalysisError(
                "Time series must be numeric",
                details={'dtype': str(series.dtype)}
            )

        # Convert index to datetime if not already
        if not isinstance(series.index, pd.DatetimeIndex):
            try:
                series.index = pd.to_datetime(series.index)
            except Exception as e:
                logger.warning(f"Could not convert index to datetime: {e}")

        # Sort by index
        series = series.sort_index()

        # Check for sufficient data
        if len(series) < 3:
            raise AnalysisError(
                "Time series too short (minimum 3 points)",
                details={'length': len(series)}
            )

        return series

    @staticmethod
    def check_frequency(series: pd.Series) -> Optional[str]:
        """
        Infer time series frequency.

        Args:
            series: Time series with DatetimeIndex

        Returns:
            Frequency string (e.g., 'D', 'M', 'H') or None
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            return None

        try:
            freq = pd.infer_freq(series.index)
            return freq
        except Exception:
            return None

    @staticmethod
    def detect_missing_timestamps(series: pd.Series) -> Dict[str, Any]:
        """
        Detect missing timestamps in regular time series.

        Args:
            series: Time series

        Returns:
            Dictionary with missing timestamp information
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            return {'has_missing': False, 'reason': 'Not a DatetimeIndex'}

        freq = TimeSeriesValidator.check_frequency(series)
        if freq is None:
            return {'has_missing': False, 'reason': 'Irregular frequency'}

        # Create complete date range
        full_range = pd.date_range(
            start=series.index.min(),
            end=series.index.max(),
            freq=freq
        )

        missing = full_range.difference(series.index)

        return {
            'has_missing': len(missing) > 0,
            'missing_count': len(missing),
            'missing_timestamps': missing.tolist() if len(missing) < 10 else missing[:10].tolist(),
            'frequency': freq,
            'total_expected': len(full_range),
            'total_present': len(series)
        }


# ============================================================================
# Stationarity Testing
# ============================================================================

class StationarityTests:
    """Statistical tests for time series stationarity."""

    @staticmethod
    def adf_test(series: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Augmented Dickey-Fuller test for stationarity.

        Args:
            series: Time series
            **kwargs: Additional arguments for adfuller

        Returns:
            Dictionary with test results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for ADF test")

        # Remove NaN values
        series_clean = series.dropna()

        result = adfuller(series_clean, **kwargs)

        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'n_lags': result[2],
            'n_obs': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05,
            'conclusion': 'Stationary' if result[1] < 0.05 else 'Non-stationary'
        }

    @staticmethod
    def kpss_test(series: pd.Series, regression: str = 'c', **kwargs) -> Dict[str, Any]:
        """
        KPSS test for stationarity.

        Args:
            series: Time series
            regression: Type of regression ('c' or 'ct')
            **kwargs: Additional arguments for kpss

        Returns:
            Dictionary with test results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for KPSS test")

        series_clean = series.dropna()

        result = kpss(series_clean, regression=regression, **kwargs)

        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'n_lags': result[2],
            'critical_values': result[3],
            'is_stationary': result[1] > 0.05,
            'conclusion': 'Stationary' if result[1] > 0.05 else 'Non-stationary'
        }

    @staticmethod
    def combined_stationarity_test(series: pd.Series) -> Dict[str, Any]:
        """
        Run both ADF and KPSS tests for robust stationarity check.

        Args:
            series: Time series

        Returns:
            Combined test results
        """
        adf_result = StationarityTests.adf_test(series)
        kpss_result = StationarityTests.kpss_test(series)

        # Determine overall conclusion
        if adf_result['is_stationary'] and kpss_result['is_stationary']:
            conclusion = 'Stationary'
        elif not adf_result['is_stationary'] and not kpss_result['is_stationary']:
            conclusion = 'Non-stationary'
        else:
            conclusion = 'Difference stationary (trend stationary)'

        return {
            'adf': adf_result,
            'kpss': kpss_result,
            'conclusion': conclusion,
            'recommendations': StationarityTests._get_recommendations(conclusion)
        }

    @staticmethod
    def _get_recommendations(conclusion: str) -> List[str]:
        """Get recommendations based on stationarity test results."""
        if conclusion == 'Stationary':
            return [
                "Series is stationary - suitable for ARMA modeling",
                "No differencing needed"
            ]
        elif conclusion == 'Non-stationary':
            return [
                "Series is non-stationary",
                "Apply differencing (d=1 or d=2)",
                "Consider seasonal differencing if seasonal pattern exists",
                "Use I(d) component in ARIMA"
            ]
        else:
            return [
                "Series may have deterministic trend",
                "Consider detrending before modeling",
                "Or include trend in the model"
            ]


# ============================================================================
# Trend Detection and Decomposition
# ============================================================================

class TrendAnalyzer:
    """Detect and analyze trends in time series."""

    @staticmethod
    def detect_trend_mannkendall(series: pd.Series) -> Dict[str, Any]:
        """
        Mann-Kendall trend test (non-parametric).

        Args:
            series: Time series

        Returns:
            Trend test results
        """
        series_clean = series.dropna()
        n = len(series_clean)

        if n < 3:
            raise AnalysisError("Need at least 3 points for trend test")

        # Calculate S statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(series_clean.iloc[j] - series_clean.iloc[i])

        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18

        # Calculate Z statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0

        # Calculate p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Determine trend
        if p_value < 0.05:
            if s > 0:
                trend = 'Increasing'
            else:
                trend = 'Decreasing'
        else:
            trend = 'No significant trend'

        return {
            's_statistic': s,
            'z_statistic': z,
            'p_value': p_value,
            'trend': trend,
            'significant': p_value < 0.05
        }

    @staticmethod
    def detect_trend_linear(series: pd.Series) -> Dict[str, Any]:
        """
        Linear trend detection using least squares.

        Args:
            series: Time series

        Returns:
            Linear trend results
        """
        series_clean = series.dropna()
        x = np.arange(len(series_clean))
        y = series_clean.values

        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Calculate trend strength
        trend_strength = abs(r_value)

        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err,
            'trend_strength': trend_strength,
            'trend_direction': 'Increasing' if slope > 0 else 'Decreasing',
            'significant': p_value < 0.05
        }

    @staticmethod
    def decompose_series(
        series: pd.Series,
        model: str = 'additive',
        period: Optional[int] = None,
        method: str = 'statsmodels'
    ) -> Dict[str, Any]:
        """
        Decompose time series into trend, seasonal, and residual components.

        Args:
            series: Time series
            model: 'additive' or 'multiplicative'
            period: Seasonal period (auto-detected if None)
            method: 'statsmodels' or 'stl'

        Returns:
            Decomposition results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for decomposition")

        series_clean = series.dropna()

        # Auto-detect period if not provided
        if period is None:
            period = TrendAnalyzer._detect_period(series_clean)

        if period is None or period < 2:
            logger.warning("Could not detect period, using default=12")
            period = min(12, len(series_clean) // 2)

        try:
            if method == 'stl':
                # STL decomposition (robust)
                decomposition = STL(series_clean, period=period).fit()
            else:
                # Classical decomposition
                decomposition = seasonal_decompose(
                    series_clean,
                    model=model,
                    period=period,
                    extrapolate_trend='freq'
                )

            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'period': period,
                'model': model,
                'trend_strength': TrendAnalyzer._calculate_component_strength(
                    decomposition.trend.dropna(),
                    series_clean
                ),
                'seasonal_strength': TrendAnalyzer._calculate_component_strength(
                    decomposition.seasonal.dropna(),
                    series_clean
                )
            }
        except Exception as e:
            raise AnalysisError(
                f"Decomposition failed: {e}",
                details={'period': period, 'model': model}
            )

    @staticmethod
    def _detect_period(series: pd.Series) -> Optional[int]:
        """Auto-detect seasonal period using FFT."""
        try:
            # Remove trend
            detrended = signal.detrend(series.dropna().values)

            # Apply FFT
            fft = np.fft.fft(detrended)
            power = np.abs(fft) ** 2
            freq = np.fft.fftfreq(len(detrended))

            # Find dominant frequency (excluding DC component)
            pos_freq_idx = freq > 0
            if not pos_freq_idx.any():
                return None

            dominant_idx = np.argmax(power[pos_freq_idx])
            dominant_freq = freq[pos_freq_idx][dominant_idx]

            if dominant_freq > 0:
                period = int(1 / dominant_freq)
                if 2 <= period <= len(series) // 2:
                    return period

        except Exception as e:
            logger.debug(f"Period detection failed: {e}")

        return None

    @staticmethod
    def _calculate_component_strength(component: pd.Series, original: pd.Series) -> float:
        """
        Calculate strength of a decomposition component.

        Returns value between 0 and 1, where 1 is strongest.
        """
        component_aligned = component.reindex(original.index)
        residual = original - component_aligned

        var_component = component_aligned.var()
        var_residual = residual.var()

        if var_component + var_residual == 0:
            return 0.0

        strength = var_component / (var_component + var_residual)
        return max(0.0, min(1.0, strength))


# ============================================================================
# Seasonality Analysis
# ============================================================================

class SeasonalityAnalyzer:
    """Analyze seasonality in time series."""

    @staticmethod
    def calculate_seasonality_strength(
        series: pd.Series,
        period: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate seasonality strength metrics.

        Args:
            series: Time series
            period: Seasonal period

        Returns:
            Seasonality strength metrics
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for seasonality analysis")

        # Decompose
        decomp_result = TrendAnalyzer.decompose_series(series, period=period)

        seasonal = decomp_result['seasonal']
        residual = decomp_result['residual']

        # Calculate metrics
        var_seasonal = seasonal.var()
        var_residual = residual.var()

        strength = var_seasonal / (var_seasonal + var_residual) if (var_seasonal + var_residual) > 0 else 0

        return {
            'seasonal_strength': strength,
            'period': decomp_result['period'],
            'interpretation': SeasonalityAnalyzer._interpret_strength(strength),
            'seasonal_amplitude': seasonal.std(),
            'seasonal_range': seasonal.max() - seasonal.min()
        }

    @staticmethod
    def test_seasonality(series: pd.Series, period: int) -> Dict[str, Any]:
        """
        Test for presence of seasonality.

        Args:
            series: Time series
            period: Period to test

        Returns:
            Test results
        """
        series_clean = series.dropna()
        n = len(series_clean)

        if n < 2 * period:
            raise AnalysisError(
                f"Need at least {2 * period} points for period={period}",
                details={'length': n}
            )

        # Calculate seasonal means
        seasonal_means = []
        for i in range(period):
            season_values = series_clean.iloc[i::period]
            seasonal_means.append(season_values.mean())

        # ANOVA test
        groups = [series_clean.iloc[i::period].values for i in range(period)]
        f_stat, p_value = stats.f_oneway(*groups)

        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'has_seasonality': p_value < 0.05,
            'seasonal_means': seasonal_means,
            'period': period
        }

    @staticmethod
    def _interpret_strength(strength: float) -> str:
        """Interpret seasonality strength value."""
        if strength < 0.2:
            return 'Very weak'
        elif strength < 0.4:
            return 'Weak'
        elif strength < 0.6:
            return 'Moderate'
        elif strength < 0.8:
            return 'Strong'
        else:
            return 'Very strong'


# ============================================================================
# Autocorrelation Analysis
# ============================================================================

class AutocorrelationAnalyzer:
    """Analyze autocorrelation in time series."""

    @staticmethod
    def compute_acf_pacf(
        series: pd.Series,
        nlags: Optional[int] = None,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compute ACF and PACF.

        Args:
            series: Time series
            nlags: Number of lags (default: min(10*log10(n), n-1))
            alpha: Significance level for confidence intervals

        Returns:
            ACF and PACF results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for ACF/PACF")

        series_clean = series.dropna()

        if nlags is None:
            nlags = min(int(10 * np.log10(len(series_clean))), len(series_clean) - 1)

        # Compute ACF
        acf_values, acf_confint = acf(
            series_clean,
            nlags=nlags,
            alpha=alpha,
            fft=True
        )

        # Compute PACF
        pacf_values, pacf_confint = pacf(
            series_clean,
            nlags=nlags,
            alpha=alpha
        )

        # Find significant lags
        acf_significant = AutocorrelationAnalyzer._find_significant_lags(
            acf_values,
            acf_confint
        )
        pacf_significant = AutocorrelationAnalyzer._find_significant_lags(
            pacf_values,
            pacf_confint
        )

        return {
            'acf': {
                'values': acf_values,
                'confint': acf_confint,
                'significant_lags': acf_significant
            },
            'pacf': {
                'values': pacf_values,
                'confint': pacf_confint,
                'significant_lags': pacf_significant
            },
            'nlags': nlags,
            'alpha': alpha,
            'suggested_ar_order': len(pacf_significant),
            'suggested_ma_order': len(acf_significant)
        }

    @staticmethod
    def _find_significant_lags(values: np.ndarray, confint: np.ndarray) -> List[int]:
        """Find lags where correlation is significant."""
        significant = []
        for i in range(1, len(values)):  # Skip lag 0
            if values[i] < confint[i, 0] or values[i] > confint[i, 1]:
                significant.append(i)
        return significant

    @staticmethod
    def ljung_box_test(series: pd.Series, lags: Optional[int] = None) -> Dict[str, Any]:
        """
        Ljung-Box test for autocorrelation.

        Args:
            series: Time series
            lags: Number of lags to test

        Returns:
            Test results
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox

        series_clean = series.dropna()

        if lags is None:
            lags = min(10, len(series_clean) // 5)

        result = acorr_ljungbox(series_clean, lags=lags, return_df=True)

        return {
            'test_statistics': result['lb_stat'].values,
            'p_values': result['lb_pvalue'].values,
            'lags': list(range(1, lags + 1)),
            'has_autocorrelation': (result['lb_pvalue'] < 0.05).any(),
            'first_significant_lag': result[result['lb_pvalue'] < 0.05].index[0] if (result['lb_pvalue'] < 0.05).any() else None
        }


# ============================================================================
# ARIMA/SARIMA Forecasting
# ============================================================================

class TimeSeriesForecaster:
    """ARIMA and SARIMA forecasting."""

    def __init__(self, verbose: bool = False):
        """
        Initialize forecaster.

        Args:
            verbose: Whether to print progress
        """
        self.verbose = verbose
        self.model = None
        self.model_fit = None

    def auto_arima(
        self,
        series: pd.Series,
        seasonal: bool = False,
        m: Optional[int] = None,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        max_P: int = 2,
        max_D: int = 1,
        max_Q: int = 2,
        information_criterion: str = 'aic'
    ) -> Dict[str, Any]:
        """
        Auto-select best ARIMA/SARIMA model.

        Args:
            series: Time series
            seasonal: Whether to fit SARIMA
            m: Seasonal period
            max_p: Max AR order
            max_d: Max differencing order
            max_q: Max MA order
            max_P: Max seasonal AR order
            max_D: Max seasonal differencing
            max_Q: Max seasonal MA order
            information_criterion: 'aic' or 'bic'

        Returns:
            Best model results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for ARIMA")

        series_clean = series.dropna()

        best_score = np.inf
        best_params = None
        best_model = None

        # Grid search
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    if seasonal and m:
                        for P in range(max_P + 1):
                            for D in range(max_D + 1):
                                for Q in range(max_Q + 1):
                                    try:
                                        model = SARIMAX(
                                            series_clean,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, m)
                                        )
                                        fit = model.fit(disp=False)
                                        score = getattr(fit, information_criterion)

                                        if score < best_score:
                                            best_score = score
                                            best_params = {
                                                'order': (p, d, q),
                                                'seasonal_order': (P, D, Q, m)
                                            }
                                            best_model = fit

                                    except Exception:
                                        continue
                    else:
                        try:
                            model = ARIMA(series_clean, order=(p, d, q))
                            fit = model.fit()
                            score = getattr(fit, information_criterion)

                            if score < best_score:
                                best_score = score
                                best_params = {'order': (p, d, q)}
                                best_model = fit

                        except Exception:
                            continue

        if best_model is None:
            raise AnalysisError("Could not fit any ARIMA model")

        self.model_fit = best_model

        return {
            'params': best_params,
            'aic': best_model.aic,
            'bic': best_model.bic,
            'model': best_model,
            'summary': str(best_model.summary())
        }

    def forecast(
        self,
        steps: int,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Generate forecast.

        Args:
            steps: Number of steps to forecast
            alpha: Significance level for confidence intervals

        Returns:
            Forecast results
        """
        if self.model_fit is None:
            raise AnalysisError("No model fitted. Call auto_arima first.")

        forecast = self.model_fit.forecast(steps=steps)
        forecast_ci = self.model_fit.get_forecast(steps=steps).conf_int(alpha=alpha)

        return {
            'forecast': forecast,
            'lower_bound': forecast_ci.iloc[:, 0],
            'upper_bound': forecast_ci.iloc[:, 1],
            'steps': steps,
            'alpha': alpha
        }


# ============================================================================
# Time Series Comparison
# ============================================================================

class TimeSeriesComparator:
    """Compare multiple time series."""

    @staticmethod
    def compute_cross_correlation(
        series1: pd.Series,
        series2: pd.Series,
        max_lag: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute cross-correlation between two series.

        Args:
            series1: First time series
            series2: Second time series
            max_lag: Maximum lag to compute

        Returns:
            Cross-correlation results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for cross-correlation")

        # Align series
        aligned = pd.DataFrame({'s1': series1, 's2': series2}).dropna()

        if len(aligned) < 3:
            raise AnalysisError("Not enough overlapping data points")

        if max_lag is None:
            max_lag = min(len(aligned) // 4, 40)

        ccf_values = ccf(aligned['s1'], aligned['s2'], adjusted=False)[:max_lag + 1]

        # Find lag with maximum correlation
        max_corr_lag = int(np.argmax(np.abs(ccf_values)))
        max_corr = ccf_values[max_corr_lag]

        return {
            'ccf_values': ccf_values,
            'lags': list(range(max_lag + 1)),
            'max_correlation': max_corr,
            'lag_at_max': max_corr_lag,
            'interpretation': TimeSeriesComparator._interpret_correlation(max_corr, max_corr_lag)
        }

    @staticmethod
    def _interpret_correlation(corr: float, lag: int) -> str:
        """Interpret cross-correlation result."""
        if abs(corr) < 0.3:
            strength = "weak"
        elif abs(corr) < 0.7:
            strength = "moderate"
        else:
            strength = "strong"

        if corr > 0:
            direction = "positive"
        else:
            direction = "negative"

        if lag == 0:
            timing = "simultaneous"
        elif lag > 0:
            timing = f"{lag} lag"
        else:
            timing = f"{abs(lag)} lead"

        return f"{strength.capitalize()} {direction} correlation ({timing})"

    @staticmethod
    def compare_distributions(
        series_dict: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        Compare distributions of multiple time series.

        Args:
            series_dict: Dictionary of series name -> series

        Returns:
            Comparison results
        """
        results = {}

        for name, series in series_dict.items():
            series_clean = series.dropna()

            results[name] = {
                'mean': series_clean.mean(),
                'std': series_clean.std(),
                'min': series_clean.min(),
                'max': series_clean.max(),
                'median': series_clean.median(),
                'skewness': skew(series_clean),
                'kurtosis': kurtosis(series_clean),
                'cv': series_clean.std() / series_clean.mean() if series_clean.mean() != 0 else np.inf
            }

        return results


# ============================================================================
# Main Time Series Analyzer
# ============================================================================

class TimeSeriesAnalyzer:
    """
    Main time series analysis class.

    Example:
        >>> analyzer = TimeSeriesAnalyzer()
        >>> results = analyzer.analyze(series)
    """

    def __init__(
        self,
        enable_forecasting: bool = True,
        enable_decomposition: bool = True,
        enable_acf_pacf: bool = True,
        forecast_steps: int = 10,
        verbose: bool = False
    ):
        """
        Initialize analyzer.

        Args:
            enable_forecasting: Enable ARIMA forecasting
            enable_decomposition: Enable trend decomposition
            enable_acf_pacf: Enable ACF/PACF analysis
            forecast_steps: Number of forecast steps
            verbose: Verbose output
        """
        self.enable_forecasting = enable_forecasting
        self.enable_decomposition = enable_decomposition
        self.enable_acf_pacf = enable_acf_pacf
        self.forecast_steps = forecast_steps
        self.verbose = verbose

    def analyze(
        self,
        series: Union[pd.Series, pd.DataFrame],
        datetime_col: Optional[str] = None,
        value_col: Optional[str] = None,
        seasonal_period: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive time series analysis.

        Args:
            series: Time series data
            datetime_col: DateTime column (for DataFrame)
            value_col: Value column (for DataFrame)
            seasonal_period: Seasonal period

        Returns:
            Analysis results
        """
        # Validate
        series_clean = TimeSeriesValidator.validate_series(
            series,
            datetime_col=datetime_col,
            value_col=value_col
        )

        results = {
            'basic_info': self._analyze_basic_info(series_clean),
            'stationarity': StationarityTests.combined_stationarity_test(series_clean),
        }

        # Trend analysis
        results['trend'] = {
            'mann_kendall': TrendAnalyzer.detect_trend_mannkendall(series_clean),
            'linear': TrendAnalyzer.detect_trend_linear(series_clean)
        }

        # Decomposition
        if self.enable_decomposition:
            try:
                results['decomposition'] = TrendAnalyzer.decompose_series(
                    series_clean,
                    period=seasonal_period
                )
                results['seasonality'] = SeasonalityAnalyzer.calculate_seasonality_strength(
                    series_clean,
                    period=seasonal_period
                )
            except Exception as e:
                logger.warning(f"Decomposition failed: {e}")
                results['decomposition'] = None
                results['seasonality'] = None

        # ACF/PACF
        if self.enable_acf_pacf and STATSMODELS_AVAILABLE:
            try:
                results['autocorrelation'] = AutocorrelationAnalyzer.compute_acf_pacf(series_clean)
            except Exception as e:
                logger.warning(f"ACF/PACF failed: {e}")
                results['autocorrelation'] = None

        # Forecasting
        if self.enable_forecasting and STATSMODELS_AVAILABLE:
            try:
                forecaster = TimeSeriesForecaster(verbose=self.verbose)
                model_result = forecaster.auto_arima(series_clean, seasonal=seasonal_period is not None, m=seasonal_period)
                forecast_result = forecaster.forecast(steps=self.forecast_steps)

                results['forecasting'] = {
                    'model': model_result,
                    'forecast': forecast_result
                }
            except Exception as e:
                logger.warning(f"Forecasting failed: {e}")
                results['forecasting'] = None

        return results

    def _analyze_basic_info(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze basic time series information."""
        return {
            'length': len(series),
            'start': series.index.min() if isinstance(series.index, pd.DatetimeIndex) else None,
            'end': series.index.max() if isinstance(series.index, pd.DatetimeIndex) else None,
            'frequency': TimeSeriesValidator.check_frequency(series),
            'missing_values': series.isna().sum(),
            'missing_percentage': series.isna().mean() * 100,
            'missing_timestamps': TimeSeriesValidator.detect_missing_timestamps(series)
        }


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'TimeSeriesAnalyzer',
    'TimeSeriesValidator',
    'StationarityTests',
    'TrendAnalyzer',
    'SeasonalityAnalyzer',
    'AutocorrelationAnalyzer',
    'TimeSeriesForecaster',
    'TimeSeriesComparator',
    'STATSMODELS_AVAILABLE',
]
