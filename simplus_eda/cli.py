"""
Command-line interface for Simplus EDA Framework.

This module provides a CLI for automated exploratory data analysis using Click.

Examples:
    # Basic analysis
    $ simplus-eda analyze data.csv

    # Generate HTML report
    $ simplus-eda analyze data.csv -o report.html

    # With custom configuration
    $ simplus-eda analyze data.csv -o report.html --correlation-threshold 0.8 --verbose

    # Generate JSON output
    $ simplus-eda analyze data.csv -o results.json -f json

    # Quick analysis (skip statistical tests)
    $ simplus-eda analyze data.csv --quick

    # Multiple formats
    $ simplus-eda analyze data.csv -o report --formats html json pdf

    # Show version
    $ simplus-eda --version

    # Show configuration info
    $ simplus-eda info
"""

import sys
from pathlib import Path
from typing import Optional, List
import json

import click
import pandas as pd

from simplus_eda import SimplusEDA, EDAConfig, __version__
from simplus_eda.utils.data_loader import DataLoader


# Click context settings for better help formatting
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__, prog_name="simplus-eda")
def cli():
    """
    Simplus EDA - Automated Exploratory Data Analysis Framework

    A comprehensive toolkit for automated data analysis, visualization,
    and report generation.

    For more information, visit: https://github.com/simplus-ai/simplus-eda-framework
    """
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file path (default: auto-generated based on input filename)",
)
@click.option(
    "-f",
    "--format",
    type=click.Choice(["html", "json", "pdf"], case_sensitive=False),
    default="html",
    help="Output format (default: html)",
)
@click.option(
    "--formats",
    multiple=True,
    type=click.Choice(["html", "json", "pdf"], case_sensitive=False),
    help="Generate multiple output formats (e.g., --formats html --formats json)",
)
@click.option(
    "-q",
    "--quick",
    is_flag=True,
    help="Quick analysis mode (skip statistical tests for faster results)",
)
@click.option(
    "--correlation-threshold",
    type=float,
    default=0.7,
    help="Correlation detection threshold (default: 0.7)",
)
@click.option(
    "--missing-threshold",
    type=float,
    default=0.5,
    help="Missing value warning threshold (default: 0.5)",
)
@click.option(
    "--outlier-method",
    type=click.Choice(["iqr", "zscore", "modified_zscore", "isolation_forest"],
                      case_sensitive=False),
    default="iqr",
    help="Outlier detection method (default: iqr)",
)
@click.option(
    "--significance-level",
    type=float,
    default=0.05,
    help="Statistical significance level (default: 0.05)",
)
@click.option(
    "--n-samples-viz",
    type=int,
    default=10000,
    help="Maximum samples for visualization (default: 10000)",
)
@click.option(
    "--n-jobs",
    type=int,
    default=1,
    help="Number of parallel jobs, -1 for all CPUs (default: 1)",
)
@click.option(
    "--random-state",
    type=int,
    default=42,
    help="Random seed for reproducibility (default: 42)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to JSON configuration file",
)
@click.option(
    "--title",
    type=str,
    help="Report title (default: auto-generated)",
)
@click.option(
    "--author",
    type=str,
    help="Report author name",
)
@click.option(
    "--company",
    type=str,
    help="Company name for report",
)
@click.option(
    "--no-viz",
    is_flag=True,
    help="Disable visualizations (faster, smaller reports)",
)
def analyze(
    input_file: str,
    output: Optional[str],
    format: str,
    formats: tuple,
    quick: bool,
    correlation_threshold: float,
    missing_threshold: float,
    outlier_method: str,
    significance_level: float,
    n_samples_viz: int,
    n_jobs: int,
    random_state: int,
    verbose: bool,
    config: Optional[str],
    title: Optional[str],
    author: Optional[str],
    company: Optional[str],
    no_viz: bool,
):
    """
    Analyze a dataset and generate an EDA report.

    INPUT_FILE: Path to the data file (CSV, Excel, Parquet, HDF5, or JSON)

    Examples:

        \b
        # Basic HTML report
        $ simplus-eda analyze data.csv

        \b
        # Custom output path
        $ simplus-eda analyze data.csv -o my_report.html

        \b
        # JSON output with verbose logging
        $ simplus-eda analyze data.csv -o results.json -f json --verbose

        \b
        # Multiple formats
        $ simplus-eda analyze data.csv -o report --formats html --formats json

        \b
        # Quick analysis with custom thresholds
        $ simplus-eda analyze data.csv --quick --correlation-threshold 0.8
    """
    try:
        input_path = Path(input_file)

        # Display header
        click.echo(click.style("\n╔════════════════════════════════════════════╗", fg="cyan", bold=True))
        click.echo(click.style("║   Simplus EDA - Automated Analysis        ║", fg="cyan", bold=True))
        click.echo(click.style("╚════════════════════════════════════════════╝\n", fg="cyan", bold=True))

        # Load configuration
        if config:
            if verbose:
                click.echo(f"📋 Loading configuration from: {config}")
            with open(config, "r") as f:
                config_dict = json.load(f)
            eda_config = EDAConfig(**config_dict)
        else:
            # Build configuration from CLI options
            eda_config = EDAConfig(
                enable_statistical_tests=not quick,
                enable_visualizations=not no_viz,
                correlation_threshold=correlation_threshold,
                missing_threshold=missing_threshold,
                outlier_method=outlier_method,
                significance_level=significance_level,
                n_samples_viz=n_samples_viz,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
            )

        # Load data
        if verbose:
            click.echo(f"📂 Loading data from: {input_path}")

        loader = DataLoader()
        data = loader.load_data(str(input_path))

        click.echo(f"✓ Loaded dataset: {data.shape[0]:,} rows × {data.shape[1]} columns")

        # Create EDA instance
        if verbose:
            click.echo("\n🔍 Initializing EDA analyzer...")

        eda = SimplusEDA(config=eda_config)

        # Perform analysis
        click.echo("\n⚙️  Analyzing data...")
        with click.progressbar(
            length=100,
            label="Progress",
            show_eta=True,
            show_percent=True,
        ) as bar:
            eda.analyze(data, quick=quick)
            bar.update(100)

        # Show quality score
        quality_score = eda.get_quality_score()
        quality_color = "green" if quality_score >= 80 else "yellow" if quality_score >= 60 else "red"
        click.echo(f"\n📊 Data Quality Score: {click.style(f'{quality_score:.1f}%', fg=quality_color, bold=True)}")

        # Show summary if verbose
        if verbose:
            click.echo("\n" + "="*60)
            click.echo(eda.summary())
            click.echo("="*60)

        # Determine output formats
        output_formats = list(formats) if formats else [format]

        # Generate default output path if not provided
        if not output:
            output_base = input_path.stem
            output = f"{output_base}_report"
        else:
            output_base = Path(output).stem

        # Generate reports
        click.echo(f"\n📝 Generating report(s)...")

        generated_files = []
        for fmt in output_formats:
            if fmt == "html":
                output_path = f"{output_base}.html"
            elif fmt == "json":
                output_path = f"{output_base}.json"
            elif fmt == "pdf":
                output_path = f"{output_base}.pdf"

            try:
                eda.generate_report(
                    output_path,
                    format=fmt,
                    title=title or f"EDA Report: {input_path.name}",
                    author=author,
                    company=company,
                )
                generated_files.append(output_path)
                click.echo(f"  ✓ {click.style(fmt.upper(), fg='green')}: {output_path}")
            except Exception as e:
                click.echo(f"  ✗ {click.style(fmt.upper(), fg='red')}: Failed - {str(e)}", err=True)

        # Success message
        click.echo(click.style("\n✨ Analysis complete!", fg="green", bold=True))

        # Show insights if verbose
        if verbose:
            insights = eda.get_insights()
            if insights:
                click.echo("\n💡 Key Insights:")
                for category, items in insights.items():
                    if items:
                        click.echo(f"\n  {category.replace('_', ' ').title()}:")
                        for item in items[:3]:  # Show top 3 per category
                            click.echo(f"    • {item}")

        return 0

    except FileNotFoundError as e:
        click.echo(click.style(f"✗ Error: File not found - {e}", fg="red", bold=True), err=True)
        return 1
    except ValueError as e:
        click.echo(click.style(f"✗ Error: Invalid value - {e}", fg="red", bold=True), err=True)
        return 1
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg="red", bold=True), err=True)
        if verbose:
            import traceback
            click.echo("\nTraceback:", err=True)
            click.echo(traceback.format_exc(), err=True)
        return 1


@cli.command()
def info():
    """
    Display framework information and system configuration.
    """
    import platform
    import matplotlib
    import seaborn as sns
    import plotly
    import scipy
    import sklearn

    click.echo(click.style("\n╔════════════════════════════════════════════╗", fg="cyan", bold=True))
    click.echo(click.style("║   Simplus EDA Framework Information       ║", fg="cyan", bold=True))
    click.echo(click.style("╚════════════════════════════════════════════╝\n", fg="cyan", bold=True))

    click.echo(f"Version:           {click.style(__version__, fg='green')}")
    click.echo(f"Python:            {platform.python_version()}")
    click.echo(f"Platform:          {platform.system()} {platform.release()}")

    click.echo(f"\n📦 Dependencies:")
    click.echo(f"  pandas:          {pd.__version__}")
    click.echo(f"  numpy:           {pd.np.__version__}")
    click.echo(f"  matplotlib:      {matplotlib.__version__}")
    click.echo(f"  seaborn:         {sns.__version__}")
    click.echo(f"  plotly:          {plotly.__version__}")
    click.echo(f"  scipy:           {scipy.__version__}")
    click.echo(f"  scikit-learn:    {sklearn.__version__}")
    click.echo(f"  click:           {click.__version__}")

    # Check optional dependencies
    click.echo(f"\n📦 Optional Dependencies:")
    try:
        import weasyprint
        click.echo(f"  weasyprint:      {click.style(weasyprint.__version__ + ' (PDF support enabled)', fg='green')}")
    except ImportError:
        click.echo(f"  weasyprint:      {click.style('Not installed (PDF support disabled)', fg='yellow')}")

    click.echo(f"\n📖 Documentation: https://github.com/simplus-ai/simplus-eda-framework")
    click.echo(f"🐛 Bug Reports:   https://github.com/simplus-ai/simplus-eda-framework/issues\n")


@cli.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="eda_config.json",
    help="Output configuration file path (default: eda_config.json)",
)
@click.option(
    "--profile",
    type=click.Choice(["default", "quick", "thorough"], case_sensitive=False),
    default="default",
    help="Configuration profile preset (default: default)",
)
def init_config(output: str, profile: str):
    """
    Generate a configuration file template.

    This creates a JSON configuration file that can be customized and used
    with the --config option in the analyze command.

    Examples:

        \b
        # Generate default config
        $ simplus-eda init-config

        \b
        # Generate with quick profile
        $ simplus-eda init-config --profile quick -o my_config.json
    """
    try:
        # Create config based on profile
        if profile == "quick":
            config = EDAConfig(
                enable_statistical_tests=False,
                enable_visualizations=True,
                n_samples_viz=5000,
                verbose=False,
            )
        elif profile == "thorough":
            config = EDAConfig(
                enable_statistical_tests=True,
                enable_visualizations=True,
                correlation_threshold=0.5,
                significance_level=0.01,
                n_jobs=-1,
                verbose=True,
            )
        else:  # default
            config = EDAConfig()

        # Save configuration
        config.to_json(output)

        click.echo(click.style(f"\n✓ Configuration file created: {output}", fg="green", bold=True))
        click.echo(f"\nProfile: {click.style(profile, fg='cyan')}")
        click.echo("\nYou can now use this config file with:")
        click.echo(f"  $ simplus-eda analyze data.csv --config {output}")
        click.echo("\nEdit the JSON file to customize settings.\n")

    except Exception as e:
        click.echo(click.style(f"✗ Error creating config: {e}", fg="red", bold=True), err=True)
        return 1


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def validate_config(config_file: str):
    """
    Validate a configuration file.

    CONFIG_FILE: Path to the JSON configuration file to validate

    Example:
        $ simplus-eda validate-config my_config.json
    """
    try:
        with open(config_file, "r") as f:
            config_dict = json.load(f)

        # Try to create EDAConfig (will raise if invalid)
        config = EDAConfig(**config_dict)

        click.echo(click.style(f"\n✓ Configuration is valid!", fg="green", bold=True))
        click.echo(f"\nConfiguration summary:")
        click.echo(f"  Statistical tests: {click.style(str(config.enable_statistical_tests), fg='cyan')}")
        click.echo(f"  Visualizations:    {click.style(str(config.enable_visualizations), fg='cyan')}")
        click.echo(f"  Correlation:       {click.style(str(config.correlation_threshold), fg='cyan')}")
        click.echo(f"  Outlier method:    {click.style(config.outlier_method, fg='cyan')}")
        click.echo(f"  Parallel jobs:     {click.style(str(config.n_jobs), fg='cyan')}")
        click.echo()

    except json.JSONDecodeError as e:
        click.echo(click.style(f"✗ Invalid JSON: {e}", fg="red", bold=True), err=True)
        return 1
    except Exception as e:
        click.echo(click.style(f"✗ Invalid configuration: {e}", fg="red", bold=True), err=True)
        return 1


def main():
    """Entry point for the CLI."""
    try:
        sys.exit(cli())
    except KeyboardInterrupt:
        click.echo(click.style("\n\n✗ Interrupted by user", fg="yellow", bold=True), err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(click.style(f"\n✗ Unexpected error: {e}", fg="red", bold=True), err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
