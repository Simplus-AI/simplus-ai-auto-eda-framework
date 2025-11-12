"""
Tests for EDA configuration management.
"""

import pytest
import json
import tempfile
from pathlib import Path

from simplus_eda.core.config import EDAConfig, ConfigurationError


class TestEDAConfigInitialization:
    """Test configuration initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        config = EDAConfig()

        assert config.enable_statistical_tests is True
        assert config.enable_visualizations is True
        assert config.correlation_threshold == 0.7
        assert config.missing_threshold == 0.1
        assert config.outlier_method == "iqr"
        assert config.random_state == 42

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        config = EDAConfig(
            correlation_threshold=0.8,
            missing_threshold=0.2,
            verbose=True,
            outlier_method="zscore"
        )

        assert config.correlation_threshold == 0.8
        assert config.missing_threshold == 0.2
        assert config.verbose is True
        assert config.outlier_method == "zscore"

    def test_all_parameters(self):
        """Test initialization with all parameters."""
        config = EDAConfig(
            enable_statistical_tests=False,
            enable_visualizations=False,
            verbose=True,
            n_jobs=4,
            random_state=123,
            correlation_threshold=0.9,
            missing_threshold=0.05,
            significance_level=0.01,
            outlier_contamination=0.15,
            outlier_method="isolation_forest",
            distribution_test_method="ks",
            n_samples_viz=5000,
            max_categories=100,
            min_cardinality=50,
            save_results=True,
            output_dir="./test_output",
            output_format="json"
        )

        assert config.n_jobs == 4
        assert config.random_state == 123
        assert config.significance_level == 0.01
        assert config.outlier_contamination == 0.15
        assert config.distribution_test_method == "ks"


class TestEDAConfigValidation:
    """Test configuration validation."""

    def test_valid_configuration(self):
        """Test that valid configuration passes validation."""
        config = EDAConfig(
            correlation_threshold=0.8,
            missing_threshold=0.1,
            outlier_method="zscore"
        )
        # Should not raise any exception
        config.validate()

    def test_invalid_correlation_threshold_high(self):
        """Test that correlation threshold > 1 raises error."""
        with pytest.raises(ConfigurationError, match="correlation_threshold must be between 0 and 1"):
            EDAConfig(correlation_threshold=1.5)

    def test_invalid_correlation_threshold_low(self):
        """Test that correlation threshold < 0 raises error."""
        with pytest.raises(ConfigurationError, match="correlation_threshold must be between 0 and 1"):
            EDAConfig(correlation_threshold=-0.1)

    def test_invalid_missing_threshold(self):
        """Test that invalid missing threshold raises error."""
        with pytest.raises(ConfigurationError, match="missing_threshold must be between 0 and 1"):
            EDAConfig(missing_threshold=1.5)

    def test_invalid_significance_level_zero(self):
        """Test that significance level of 0 raises error."""
        with pytest.raises(ConfigurationError, match="significance_level must be between 0 and 1"):
            EDAConfig(significance_level=0)

    def test_invalid_significance_level_one(self):
        """Test that significance level of 1 raises error."""
        with pytest.raises(ConfigurationError, match="significance_level must be between 0 and 1"):
            EDAConfig(significance_level=1.0)

    def test_invalid_outlier_contamination_high(self):
        """Test that outlier contamination > 0.5 raises error."""
        with pytest.raises(ConfigurationError, match="outlier_contamination must be between 0 and 0.5"):
            EDAConfig(outlier_contamination=0.6)

    def test_invalid_outlier_contamination_zero(self):
        """Test that outlier contamination = 0 raises error."""
        with pytest.raises(ConfigurationError, match="outlier_contamination must be between 0 and 0.5"):
            EDAConfig(outlier_contamination=0)

    def test_invalid_outlier_method(self):
        """Test that invalid outlier method raises error."""
        with pytest.raises(ConfigurationError, match="outlier_method must be one of"):
            EDAConfig(outlier_method="invalid_method")

    def test_valid_outlier_methods(self):
        """Test all valid outlier methods."""
        for method in ["iqr", "zscore", "isolation_forest", "lof"]:
            config = EDAConfig(outlier_method=method)
            assert config.outlier_method == method

    def test_invalid_distribution_test_method(self):
        """Test that invalid distribution test method raises error."""
        with pytest.raises(ConfigurationError, match="distribution_test_method must be one of"):
            EDAConfig(distribution_test_method="invalid_test")

    def test_valid_distribution_test_methods(self):
        """Test all valid distribution test methods."""
        for method in ["shapiro", "ks", "anderson"]:
            config = EDAConfig(distribution_test_method=method)
            assert config.distribution_test_method == method

    def test_invalid_output_format(self):
        """Test that invalid output format raises error."""
        with pytest.raises(ConfigurationError, match="output_format must be one of"):
            EDAConfig(output_format="xml")

    def test_valid_output_formats(self):
        """Test all valid output formats."""
        for fmt in ["json", "pickle", "html"]:
            config = EDAConfig(output_format=fmt)
            assert config.output_format == fmt

    def test_invalid_n_samples_viz_zero(self):
        """Test that n_samples_viz = 0 raises error."""
        with pytest.raises(ConfigurationError, match="n_samples_viz must be positive"):
            EDAConfig(n_samples_viz=0)

    def test_invalid_n_samples_viz_negative(self):
        """Test that negative n_samples_viz raises error."""
        with pytest.raises(ConfigurationError, match="n_samples_viz must be positive"):
            EDAConfig(n_samples_viz=-100)

    def test_invalid_max_categories(self):
        """Test that invalid max_categories raises error."""
        with pytest.raises(ConfigurationError, match="max_categories must be positive"):
            EDAConfig(max_categories=0)

    def test_invalid_min_cardinality(self):
        """Test that invalid min_cardinality raises error."""
        with pytest.raises(ConfigurationError, match="min_cardinality must be positive"):
            EDAConfig(min_cardinality=-10)

    def test_invalid_n_jobs_zero(self):
        """Test that n_jobs = 0 raises error."""
        with pytest.raises(ConfigurationError, match="n_jobs must be -1 or positive"):
            EDAConfig(n_jobs=0)

    def test_invalid_n_jobs_negative(self):
        """Test that invalid negative n_jobs raises error."""
        with pytest.raises(ConfigurationError, match="n_jobs must be -1 or positive"):
            EDAConfig(n_jobs=-2)

    def test_valid_n_jobs_minus_one(self):
        """Test that n_jobs = -1 is valid."""
        config = EDAConfig(n_jobs=-1)
        assert config.n_jobs == -1


class TestEDAConfigConversion:
    """Test configuration conversion methods."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = EDAConfig(
            correlation_threshold=0.8,
            verbose=True,
            outlier_method="zscore"
        )
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['correlation_threshold'] == 0.8
        assert config_dict['verbose'] is True
        assert config_dict['outlier_method'] == "zscore"
        assert 'enable_statistical_tests' in config_dict
        assert 'custom_params' in config_dict

    def test_to_dict_all_parameters(self):
        """Test that all parameters are in dictionary."""
        config = EDAConfig()
        config_dict = config.to_dict()

        expected_keys = [
            'enable_statistical_tests', 'enable_visualizations', 'verbose',
            'n_jobs', 'random_state', 'correlation_threshold', 'missing_threshold',
            'significance_level', 'outlier_contamination', 'outlier_method',
            'distribution_test_method', 'n_samples_viz', 'max_categories',
            'min_cardinality', 'save_results', 'output_dir', 'output_format',
            'custom_params'
        ]

        for key in expected_keys:
            assert key in config_dict

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'correlation_threshold': 0.85,
            'verbose': True,
            'outlier_method': 'isolation_forest',
            'n_jobs': 4
        }
        config = EDAConfig.from_dict(config_dict)

        assert config.correlation_threshold == 0.85
        assert config.verbose is True
        assert config.outlier_method == 'isolation_forest'
        assert config.n_jobs == 4

    def test_from_dict_ignores_unknown_keys(self):
        """Test that unknown keys are ignored."""
        config_dict = {
            'correlation_threshold': 0.85,
            'unknown_key': 'unknown_value',
            'another_unknown': 123
        }
        config = EDAConfig.from_dict(config_dict)

        assert config.correlation_threshold == 0.85
        # Should not have unknown keys
        assert not hasattr(config, 'unknown_key')

    def test_round_trip_to_dict_from_dict(self):
        """Test round-trip conversion to/from dict."""
        original = EDAConfig(
            correlation_threshold=0.88,
            verbose=True,
            n_jobs=2
        )
        config_dict = original.to_dict()
        restored = EDAConfig.from_dict(config_dict)

        assert restored.correlation_threshold == original.correlation_threshold
        assert restored.verbose == original.verbose
        assert restored.n_jobs == original.n_jobs


class TestEDAConfigJSON:
    """Test JSON serialization and deserialization."""

    def test_to_json(self, tmp_path):
        """Test saving configuration to JSON file."""
        config = EDAConfig(
            correlation_threshold=0.85,
            verbose=True,
            outlier_method="zscore"
        )

        json_path = tmp_path / "config.json"
        config.to_json(str(json_path))

        assert json_path.exists()

        # Verify content
        with open(json_path, 'r') as f:
            data = json.load(f)

        assert data['correlation_threshold'] == 0.85
        assert data['verbose'] is True
        assert data['outlier_method'] == "zscore"

    def test_to_json_creates_directory(self, tmp_path):
        """Test that to_json creates parent directories."""
        config = EDAConfig()

        json_path = tmp_path / "subdir" / "nested" / "config.json"
        config.to_json(str(json_path))

        assert json_path.exists()
        assert json_path.parent.exists()

    def test_from_json(self, tmp_path):
        """Test loading configuration from JSON file."""
        config_dict = {
            'correlation_threshold': 0.9,
            'verbose': True,
            'n_jobs': 4,
            'outlier_method': 'isolation_forest'
        }

        json_path = tmp_path / "config.json"
        with open(json_path, 'w') as f:
            json.dump(config_dict, f)

        config = EDAConfig.from_json(str(json_path))

        assert config.correlation_threshold == 0.9
        assert config.verbose is True
        assert config.n_jobs == 4
        assert config.outlier_method == 'isolation_forest'

    def test_from_json_file_not_found(self):
        """Test that from_json raises error for missing file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            EDAConfig.from_json("nonexistent_file.json")

    def test_from_json_invalid_json(self, tmp_path):
        """Test that from_json raises error for invalid JSON."""
        json_path = tmp_path / "invalid.json"
        with open(json_path, 'w') as f:
            f.write("{ invalid json content }")

        with pytest.raises(ConfigurationError, match="Invalid JSON"):
            EDAConfig.from_json(str(json_path))

    def test_round_trip_json(self, tmp_path):
        """Test round-trip save/load with JSON."""
        original = EDAConfig(
            correlation_threshold=0.87,
            verbose=True,
            n_samples_viz=5000,
            outlier_method="lof"
        )

        json_path = tmp_path / "config.json"
        original.to_json(str(json_path))
        restored = EDAConfig.from_json(str(json_path))

        assert restored.correlation_threshold == original.correlation_threshold
        assert restored.verbose == original.verbose
        assert restored.n_samples_viz == original.n_samples_viz
        assert restored.outlier_method == original.outlier_method


class TestEDAConfigMethods:
    """Test configuration utility methods."""

    def test_update(self):
        """Test updating configuration parameters."""
        config = EDAConfig(correlation_threshold=0.7)
        updated = config.update(correlation_threshold=0.9, verbose=True)

        # Original should be unchanged
        assert config.correlation_threshold == 0.7
        assert config.verbose is False

        # Updated should have new values
        assert updated.correlation_threshold == 0.9
        assert updated.verbose is True

    def test_update_invalid_value(self):
        """Test that update with invalid value raises error."""
        config = EDAConfig()

        with pytest.raises(ConfigurationError):
            config.update(correlation_threshold=1.5)

    def test_copy(self):
        """Test copying configuration."""
        original = EDAConfig(
            correlation_threshold=0.8,
            verbose=True,
            outlier_method="zscore"
        )
        copy = original.copy()

        # Values should be the same
        assert copy.correlation_threshold == original.correlation_threshold
        assert copy.verbose == original.verbose
        assert copy.outlier_method == original.outlier_method

        # Should be different objects
        assert copy is not original

    def test_get_summary(self):
        """Test getting configuration summary."""
        config = EDAConfig(
            correlation_threshold=0.85,
            verbose=True,
            outlier_method="isolation_forest"
        )
        summary = config.get_summary()

        assert isinstance(summary, str)
        assert "EDA Configuration" in summary
        assert "0.85" in summary
        assert "isolation_forest" in summary
        assert "Enabled" in summary  # For verbose

    def test_get_summary_with_save_results(self):
        """Test summary includes output settings when save_results is True."""
        config = EDAConfig(
            save_results=True,
            output_dir="./my_results",
            output_format="json"
        )
        summary = config.get_summary()

        assert "Output Settings" in summary
        assert "my_results" in summary
        assert "json" in summary


class TestEDAConfigFactoryMethods:
    """Test configuration factory methods."""

    def test_get_default(self):
        """Test default configuration factory."""
        config = EDAConfig.get_default()

        assert config.correlation_threshold == 0.7
        assert config.enable_statistical_tests is True
        assert config.outlier_method == "iqr"

    def test_get_quick(self):
        """Test quick analysis configuration factory."""
        config = EDAConfig.get_quick()

        assert config.enable_statistical_tests is False
        assert config.enable_visualizations is False
        assert config.n_samples_viz == 1000

    def test_get_thorough(self):
        """Test thorough analysis configuration factory."""
        config = EDAConfig.get_thorough()

        assert config.enable_statistical_tests is True
        assert config.enable_visualizations is True
        assert config.correlation_threshold == 0.5
        assert config.missing_threshold == 0.05
        assert config.outlier_method == "isolation_forest"
        assert config.n_samples_viz == 50000
        assert config.verbose is True

    def test_factory_methods_return_valid_configs(self):
        """Test that all factory methods return valid configurations."""
        configs = [
            EDAConfig.get_default(),
            EDAConfig.get_quick(),
            EDAConfig.get_thorough()
        ]

        for config in configs:
            # Should not raise any validation errors
            config.validate()


class TestEDAConfigEdgeCases:
    """Test edge cases and special scenarios."""

    def test_boundary_values(self):
        """Test boundary values for thresholds."""
        # Test minimum valid values
        config1 = EDAConfig(
            correlation_threshold=0.0,
            missing_threshold=0.0,
            significance_level=0.001,
            outlier_contamination=0.001
        )
        assert config1.correlation_threshold == 0.0

        # Test maximum valid values
        config2 = EDAConfig(
            correlation_threshold=1.0,
            missing_threshold=1.0,
            significance_level=0.999,
            outlier_contamination=0.5
        )
        assert config2.correlation_threshold == 1.0

    def test_custom_params(self):
        """Test custom parameters field."""
        custom = {
            'my_param': 'my_value',
            'another_param': 123
        }
        config = EDAConfig(custom_params=custom)

        assert config.custom_params == custom
        assert config.custom_params['my_param'] == 'my_value'

    def test_empty_custom_params(self):
        """Test empty custom parameters."""
        config = EDAConfig()

        assert config.custom_params == {}

    def test_large_n_samples_viz(self):
        """Test very large n_samples_viz value."""
        config = EDAConfig(n_samples_viz=1000000)

        assert config.n_samples_viz == 1000000

    def test_multiple_updates(self):
        """Test chaining multiple updates."""
        config = EDAConfig()
        config = config.update(correlation_threshold=0.8)
        config = config.update(verbose=True)
        config = config.update(n_jobs=4)

        assert config.correlation_threshold == 0.8
        assert config.verbose is True
        assert config.n_jobs == 4


class TestEDAConfigIntegration:
    """Integration tests for configuration."""

    def test_config_with_analyzer(self):
        """Test that config works with EDAAnalyzer."""
        from simplus_eda.core.analyzer import EDAAnalyzer
        import pandas as pd
        import numpy as np

        config = EDAConfig(
            correlation_threshold=0.8,
            missing_threshold=0.15,
            outlier_method="zscore"
        )

        analyzer = EDAAnalyzer(config=config)

        assert analyzer.config.correlation_threshold == 0.8
        assert analyzer.config.missing_threshold == 0.15
        assert analyzer.config.outlier_method == "zscore"

    def test_config_persistence(self, tmp_path):
        """Test saving and loading config for reuse."""
        # Create and save config
        config1 = EDAConfig(
            correlation_threshold=0.88,
            verbose=True,
            outlier_method="isolation_forest",
            n_jobs=4
        )
        config_path = tmp_path / "my_config.json"
        config1.to_json(str(config_path))

        # Load and use config
        config2 = EDAConfig.from_json(str(config_path))

        # Should have same values
        assert config2.correlation_threshold == config1.correlation_threshold
        assert config2.verbose == config1.verbose
        assert config2.outlier_method == config1.outlier_method
        assert config2.n_jobs == config1.n_jobs
