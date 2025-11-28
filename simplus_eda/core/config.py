"""
Configuration management for the EDA framework.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json
from pathlib import Path
from simplus_eda.exceptions import ConfigurationError, InvalidConfigurationError


@dataclass
class EDAConfig:
    """
    Configuration class for EDA analysis with validation.

    This class manages all configuration options for the EDA framework,
    including analysis settings, thresholds, and output preferences.

    Attributes:
        enable_statistical_tests: Whether to run statistical hypothesis tests (default: True)
        enable_visualizations: Whether to generate visualizations (default: True)
        correlation_threshold: Threshold for strong correlation detection (0-1, default: 0.7)
        missing_threshold: Threshold for missing data warnings (0-1, default: 0.1)
        outlier_method: Method for outlier detection (default: 'iqr')
            Options: 'iqr', 'zscore', 'isolation_forest', 'lof'
        outlier_contamination: Expected outlier proportion for ML methods (0-0.5, default: 0.1)
        significance_level: P-value threshold for statistical tests (default: 0.05)
        n_samples_viz: Maximum samples for visualizations (default: 10000)
        random_state: Random seed for reproducibility (default: 42)
        verbose: Enable verbose output (default: False)
        n_jobs: Number of parallel jobs (-1 for all cores, default: 1)
        max_categories: Maximum unique values to treat as categorical (default: 50)
        min_cardinality: Minimum unique values for high cardinality warning (default: 100)
        distribution_test_method: Method for distribution testing (default: 'shapiro')
            Options: 'shapiro', 'ks', 'anderson'
        enable_auto_sampling: Whether to automatically sample large datasets (default: True)
        auto_sample_threshold: Row count threshold for auto-sampling (default: 100000)
        sampling_method: Method for data sampling (default: 'adaptive')
            Options: 'random', 'stratified', 'reservoir', 'systematic', 'adaptive'
        enable_dask: Whether to use Dask for out-of-core computation (default: False)
        dask_threshold: Row count threshold for switching to Dask (default: 1000000)
        dask_npartitions: Number of Dask partitions (None = auto, default: None)
        chunk_size: Chunk size for streaming processing (None = auto, default: None)
        optimize_memory: Whether to optimize DataFrame memory usage (default: True)
        memory_check_enabled: Whether to check available memory before operations (default: True)
        save_results: Whether to save analysis results to file (default: False)
        output_dir: Directory for saving results (default: './eda_results')
        output_format: Format for saving results (default: 'json')
            Options: 'json', 'pickle', 'html'
        custom_params: Additional custom parameters (default: {})

    Example:
        >>> config = EDAConfig(
        ...     correlation_threshold=0.8,
        ...     outlier_method='zscore',
        ...     verbose=True
        ... )
        >>> config.validate()
        >>> analyzer = EDAAnalyzer(config=config)
    """

    # Analysis control
    enable_statistical_tests: bool = True
    enable_visualizations: bool = True
    verbose: bool = False
    n_jobs: int = 1
    random_state: int = 42

    # Thresholds
    correlation_threshold: float = 0.7
    missing_threshold: float = 0.1
    significance_level: float = 0.05
    outlier_contamination: float = 0.1

    # Methods
    outlier_method: str = "iqr"
    distribution_test_method: str = "shapiro"

    # Data handling
    n_samples_viz: int = 10000
    max_categories: int = 50
    min_cardinality: int = 100

    # Large dataset settings
    enable_auto_sampling: bool = True
    auto_sample_threshold: int = 100000  # Auto-sample if rows > threshold
    sampling_method: str = "adaptive"
    enable_dask: bool = False
    dask_threshold: int = 1000000  # Use Dask if rows > threshold
    dask_npartitions: Optional[int] = None  # Auto-determine if None
    chunk_size: Optional[int] = None  # Auto-determine if None (for chunked processing)
    optimize_memory: bool = True
    memory_check_enabled: bool = True

    # Caching settings
    enable_cache: bool = True
    cache_backend: str = "memory"  # 'memory' or 'disk'
    cache_dir: str = "./.simplus_cache"
    cache_max_size: int = 100  # Max entries for memory cache
    cache_max_size_mb: Optional[float] = 500.0  # Max size for disk cache (MB)
    cache_ttl: Optional[float] = None  # Time-to-live in seconds (None = no expiration)

    # Progress tracking settings
    enable_progress: bool = True
    use_tqdm: bool = True  # Use tqdm progress bars
    progress_callbacks: List[Any] = field(default_factory=list)  # Custom progress callbacks

    # Output settings
    save_results: bool = False
    output_dir: str = "./eda_results"
    output_format: str = "json"

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    # Constants for validation
    VALID_OUTLIER_METHODS: List[str] = field(
        default_factory=lambda: ["iqr", "zscore", "isolation_forest", "lof"],
        init=False,
        repr=False
    )
    VALID_DISTRIBUTION_TESTS: List[str] = field(
        default_factory=lambda: ["shapiro", "ks", "anderson"],
        init=False,
        repr=False
    )
    VALID_OUTPUT_FORMATS: List[str] = field(
        default_factory=lambda: ["json", "pickle", "html"],
        init=False,
        repr=False
    )
    VALID_SAMPLING_METHODS: List[str] = field(
        default_factory=lambda: ["random", "stratified", "reservoir", "systematic", "adaptive"],
        init=False,
        repr=False
    )
    VALID_CACHE_BACKENDS: List[str] = field(
        default_factory=lambda: ["memory", "disk"],
        init=False,
        repr=False
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate all configuration parameters.

        Raises:
            ConfigurationError: If any parameter is invalid
        """
        # Validate thresholds (0-1)
        if not 0 <= self.correlation_threshold <= 1:
            raise InvalidConfigurationError(
                "correlation_threshold must be between 0 and 1",
                parameter="correlation_threshold",
                value=self.correlation_threshold,
                valid_values=["0.0 to 1.0"]
            )

        if not 0 <= self.missing_threshold <= 1:
            raise InvalidConfigurationError(
                "missing_threshold must be between 0 and 1",
                parameter="missing_threshold",
                value=self.missing_threshold,
                valid_values=["0.0 to 1.0"]
            )

        if not 0 < self.significance_level < 1:
            raise InvalidConfigurationError(
                "significance_level must be between 0 and 1",
                parameter="significance_level",
                value=self.significance_level,
                valid_values=["0.0 to 1.0 (exclusive)"]
            )

        if not 0 < self.outlier_contamination <= 0.5:
            raise InvalidConfigurationError(
                "outlier_contamination must be between 0 and 0.5",
                parameter="outlier_contamination",
                value=self.outlier_contamination,
                valid_values=["0.0 to 0.5"]
            )

        # Validate methods
        if self.outlier_method not in self.VALID_OUTLIER_METHODS:
            raise InvalidConfigurationError(
                "Invalid outlier detection method",
                parameter="outlier_method",
                value=self.outlier_method,
                valid_values=self.VALID_OUTLIER_METHODS
            )

        if self.distribution_test_method not in self.VALID_DISTRIBUTION_TESTS:
            raise InvalidConfigurationError(
                "Invalid distribution test method",
                parameter="distribution_test_method",
                value=self.distribution_test_method,
                valid_values=self.VALID_DISTRIBUTION_TESTS
            )

        if self.output_format not in self.VALID_OUTPUT_FORMATS:
            raise InvalidConfigurationError(
                "Invalid output format",
                parameter="output_format",
                value=self.output_format,
                valid_values=self.VALID_OUTPUT_FORMATS
            )

        # Validate positive integers
        if self.n_samples_viz <= 0:
            raise InvalidConfigurationError(
                "n_samples_viz must be positive",
                parameter="n_samples_viz",
                value=self.n_samples_viz
            )

        if self.max_categories <= 0:
            raise InvalidConfigurationError(
                "max_categories must be positive",
                parameter="max_categories",
                value=self.max_categories
            )

        if self.min_cardinality <= 0:
            raise InvalidConfigurationError(
                "min_cardinality must be positive",
                parameter="min_cardinality",
                value=self.min_cardinality
            )

        # Validate n_jobs
        if self.n_jobs < -1 or self.n_jobs == 0:
            raise InvalidConfigurationError(
                "n_jobs must be -1 or a positive integer",
                parameter="n_jobs",
                value=self.n_jobs,
                valid_values=["-1 (all CPUs)", "1, 2, 3, ... (specific number)"]
            )

        # Validate sampling method
        if self.sampling_method not in self.VALID_SAMPLING_METHODS:
            raise InvalidConfigurationError(
                "Invalid sampling method",
                parameter="sampling_method",
                value=self.sampling_method,
                valid_values=self.VALID_SAMPLING_METHODS
            )

        # Validate large dataset thresholds
        if self.auto_sample_threshold <= 0:
            raise InvalidConfigurationError(
                "auto_sample_threshold must be positive",
                parameter="auto_sample_threshold",
                value=self.auto_sample_threshold
            )

        if self.dask_threshold <= 0:
            raise InvalidConfigurationError(
                "dask_threshold must be positive",
                parameter="dask_threshold",
                value=self.dask_threshold
            )

        if self.dask_npartitions is not None and self.dask_npartitions <= 0:
            raise InvalidConfigurationError(
                "dask_npartitions must be positive or None",
                parameter="dask_npartitions",
                value=self.dask_npartitions
            )

        if self.chunk_size is not None and self.chunk_size <= 0:
            raise InvalidConfigurationError(
                "chunk_size must be positive or None",
                parameter="chunk_size",
                value=self.chunk_size
            )

        # Validate cache backend
        if self.cache_backend not in self.VALID_CACHE_BACKENDS:
            raise InvalidConfigurationError(
                "Invalid cache backend",
                parameter="cache_backend",
                value=self.cache_backend,
                valid_values=self.VALID_CACHE_BACKENDS
            )

        # Validate cache settings
        if self.cache_max_size <= 0:
            raise InvalidConfigurationError(
                "cache_max_size must be positive",
                parameter="cache_max_size",
                value=self.cache_max_size
            )

        if self.cache_max_size_mb is not None and self.cache_max_size_mb <= 0:
            raise InvalidConfigurationError(
                "cache_max_size_mb must be positive or None",
                parameter="cache_max_size_mb",
                value=self.cache_max_size_mb
            )

        if self.cache_ttl is not None and self.cache_ttl <= 0:
            raise InvalidConfigurationError(
                "cache_ttl must be positive or None",
                parameter="cache_ttl",
                value=self.cache_ttl
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary with all configuration parameters

        Example:
            >>> config = EDAConfig(correlation_threshold=0.8)
            >>> config_dict = config.to_dict()
            >>> assert config_dict['correlation_threshold'] == 0.8
        """
        return {
            "enable_statistical_tests": self.enable_statistical_tests,
            "enable_visualizations": self.enable_visualizations,
            "verbose": self.verbose,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "correlation_threshold": self.correlation_threshold,
            "missing_threshold": self.missing_threshold,
            "significance_level": self.significance_level,
            "outlier_contamination": self.outlier_contamination,
            "outlier_method": self.outlier_method,
            "distribution_test_method": self.distribution_test_method,
            "n_samples_viz": self.n_samples_viz,
            "max_categories": self.max_categories,
            "min_cardinality": self.min_cardinality,
            "enable_auto_sampling": self.enable_auto_sampling,
            "auto_sample_threshold": self.auto_sample_threshold,
            "sampling_method": self.sampling_method,
            "enable_dask": self.enable_dask,
            "dask_threshold": self.dask_threshold,
            "dask_npartitions": self.dask_npartitions,
            "chunk_size": self.chunk_size,
            "optimize_memory": self.optimize_memory,
            "memory_check_enabled": self.memory_check_enabled,
            "enable_cache": self.enable_cache,
            "cache_backend": self.cache_backend,
            "cache_dir": self.cache_dir,
            "cache_max_size": self.cache_max_size,
            "cache_max_size_mb": self.cache_max_size_mb,
            "cache_ttl": self.cache_ttl,
            "enable_progress": self.enable_progress,
            "use_tqdm": self.use_tqdm,
            "save_results": self.save_results,
            "output_dir": self.output_dir,
            "output_format": self.output_format,
            "custom_params": self.custom_params,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EDAConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            EDAConfig instance

        Raises:
            ConfigurationError: If parameters are invalid

        Example:
            >>> config_dict = {'correlation_threshold': 0.8, 'verbose': True}
            >>> config = EDAConfig.from_dict(config_dict)
            >>> assert config.correlation_threshold == 0.8
        """
        # Filter out unknown keys
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    @classmethod
    def from_json(cls, json_path: str) -> "EDAConfig":
        """
        Load configuration from JSON file.

        Args:
            json_path: Path to JSON configuration file

        Returns:
            EDAConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ConfigurationError: If JSON is invalid

        Example:
            >>> config = EDAConfig.from_json('config.json')
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")

        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")

    def to_json(self, json_path: str, indent: int = 2) -> None:
        """
        Save configuration to JSON file.

        Args:
            json_path: Path to save JSON file
            indent: Indentation level for pretty printing (default: 2)

        Example:
            >>> config = EDAConfig(correlation_threshold=0.8)
            >>> config.to_json('my_config.json')
        """
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    def update(self, **kwargs) -> "EDAConfig":
        """
        Create a new config with updated parameters.

        Args:
            **kwargs: Parameters to update

        Returns:
            New EDAConfig instance with updated parameters

        Raises:
            ConfigurationError: If updated parameters are invalid

        Example:
            >>> config = EDAConfig()
            >>> new_config = config.update(correlation_threshold=0.9, verbose=True)
            >>> assert new_config.correlation_threshold == 0.9
        """
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)

    def copy(self) -> "EDAConfig":
        """
        Create a deep copy of the configuration.

        Returns:
            New EDAConfig instance with same parameters

        Example:
            >>> config = EDAConfig(correlation_threshold=0.8)
            >>> config_copy = config.copy()
            >>> assert config_copy.correlation_threshold == 0.8
        """
        return self.from_dict(self.to_dict())

    def get_summary(self) -> str:
        """
        Get human-readable summary of configuration.

        Returns:
            Formatted string with key configuration parameters

        Example:
            >>> config = EDAConfig()
            >>> print(config.get_summary())
        """
        lines = [
            "=== EDA Configuration ===",
            "",
            "Analysis Settings:",
            f"  Statistical Tests: {'Enabled' if self.enable_statistical_tests else 'Disabled'}",
            f"  Visualizations: {'Enabled' if self.enable_visualizations else 'Disabled'}",
            f"  Verbose Output: {'Enabled' if self.verbose else 'Disabled'}",
            f"  Parallel Jobs: {self.n_jobs if self.n_jobs > 0 else 'All CPUs'}",
            "",
            "Thresholds:",
            f"  Correlation Threshold: {self.correlation_threshold}",
            f"  Missing Data Threshold: {self.missing_threshold}",
            f"  Significance Level: {self.significance_level}",
            "",
            "Methods:",
            f"  Outlier Detection: {self.outlier_method}",
            f"  Distribution Test: {self.distribution_test_method}",
            "",
            "Data Handling:",
            f"  Max Visualization Samples: {self.n_samples_viz:,}",
            f"  Max Categorical Values: {self.max_categories}",
            f"  Random State: {self.random_state}",
            "",
            "Large Dataset Settings:",
            f"  Auto Sampling: {'Enabled' if self.enable_auto_sampling else 'Disabled'}",
            f"  Auto Sample Threshold: {self.auto_sample_threshold:,} rows",
            f"  Sampling Method: {self.sampling_method}",
            f"  Dask Backend: {'Enabled' if self.enable_dask else 'Disabled'}",
            f"  Dask Threshold: {self.dask_threshold:,} rows",
            f"  Memory Optimization: {'Enabled' if self.optimize_memory else 'Disabled'}",
            f"  Memory Check: {'Enabled' if self.memory_check_enabled else 'Disabled'}",
            "",
            "Caching Settings:",
            f"  Caching: {'Enabled' if self.enable_cache else 'Disabled'}",
            f"  Cache Backend: {self.cache_backend}",
            f"  Cache Max Size: {self.cache_max_size} entries" if self.cache_backend == 'memory' else f"  Cache Max Size: {self.cache_max_size_mb} MB",
            f"  Cache TTL: {self.cache_ttl}s" if self.cache_ttl else "  Cache TTL: Never expires",
            "",
            "Progress Tracking:",
            f"  Progress Indicators: {'Enabled' if self.enable_progress else 'Disabled'}",
            f"  Progress Bars (tqdm): {'Enabled' if self.use_tqdm else 'Disabled'}",
        ]

        if self.save_results:
            lines.extend([
                "",
                "Output Settings:",
                f"  Save Results: Enabled",
                f"  Output Directory: {self.output_dir}",
                f"  Output Format: {self.output_format}",
            ])

        return "\n".join(lines)

    @staticmethod
    def get_default() -> "EDAConfig":
        """
        Get default configuration.

        Returns:
            EDAConfig with default parameters

        Example:
            >>> config = EDAConfig.get_default()
            >>> assert config.correlation_threshold == 0.7
        """
        return EDAConfig()

    @staticmethod
    def get_quick() -> "EDAConfig":
        """
        Get configuration optimized for quick analysis.

        Returns:
            EDAConfig with settings for faster analysis

        Example:
            >>> config = EDAConfig.get_quick()
            >>> analyzer = EDAAnalyzer(config=config)
        """
        return EDAConfig(
            enable_statistical_tests=False,
            enable_visualizations=False,
            n_samples_viz=1000,
        )

    @staticmethod
    def get_thorough() -> "EDAConfig":
        """
        Get configuration for thorough analysis.

        Returns:
            EDAConfig with settings for comprehensive analysis

        Example:
            >>> config = EDAConfig.get_thorough()
            >>> analyzer = EDAAnalyzer(config=config)
        """
        return EDAConfig(
            enable_statistical_tests=True,
            enable_visualizations=True,
            correlation_threshold=0.5,
            missing_threshold=0.05,
            outlier_method="isolation_forest",
            n_samples_viz=50000,
            verbose=True,
        )

    @staticmethod
    def get_large_dataset() -> "EDAConfig":
        """
        Get configuration optimized for large datasets.

        Returns:
            EDAConfig with settings for memory-efficient large dataset analysis

        Example:
            >>> config = EDAConfig.get_large_dataset()
            >>> analyzer = EDAAnalyzer(config=config)
        """
        return EDAConfig(
            enable_auto_sampling=True,
            auto_sample_threshold=50000,
            sampling_method="adaptive",
            enable_dask=False,  # User can enable if needed
            optimize_memory=True,
            memory_check_enabled=True,
            n_samples_viz=5000,
            n_jobs=-1,  # Use all CPUs
            verbose=True,
        )
