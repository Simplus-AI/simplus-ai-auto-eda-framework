"""
Pipeline integration example for Simplus EDA Framework.

This example demonstrates how to integrate the EDA framework
into a data processing pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from simplus_eda import EDAAnalyzer, ReportGenerator
from simplus_eda.utils import DataLoader, DataValidator


class DataPipeline:
    """
    Example data pipeline that integrates EDA analysis.
    """

    def __init__(self, output_dir: str = "output"):
        """
        Initialize the pipeline.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eda_analyzer = EDAAnalyzer()
        self.report_generator = ReportGenerator()

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from file.

        Args:
            file_path: Path to data file

        Returns:
            Loaded DataFrame
        """
        print(f"Loading data from {file_path}...")
        loader = DataLoader()
        data = loader.load(file_path)
        print(f"✓ Loaded {data.shape[0]} rows and {data.shape[1]} columns")
        return data

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality.

        Args:
            data: DataFrame to validate

        Returns:
            True if validation passes
        """
        print("\nValidating data...")
        validator = DataValidator()

        # Basic validation
        is_valid, errors = validator.validate_dataframe(data)
        if not is_valid:
            print("✗ Validation errors found:")
            for error in errors:
                print(f"  - {error}")
            return False

        print("✓ Data validation passed")
        return True

    def perform_eda(self, data: pd.DataFrame) -> dict:
        """
        Perform exploratory data analysis.

        Args:
            data: DataFrame to analyze

        Returns:
            Analysis results dictionary
        """
        print("\nPerforming EDA...")
        results = self.eda_analyzer.analyze(data)
        print("✓ EDA analysis completed")
        return results

    def generate_reports(self, results: dict, prefix: str = "pipeline"):
        """
        Generate analysis reports.

        Args:
            results: Analysis results
            prefix: Filename prefix for reports
        """
        print("\nGenerating reports...")

        # JSON report
        json_path = self.output_dir / f"{prefix}_report.json"
        self.report_generator.generate_json(results, str(json_path))
        print(f"✓ JSON report: {json_path}")

        # HTML report
        html_path = self.output_dir / f"{prefix}_report.html"
        self.report_generator.generate_html(results, str(html_path))
        print(f"✓ HTML report: {html_path}")

    def run(self, file_path: str) -> dict:
        """
        Run the complete pipeline.

        Args:
            file_path: Path to input data file

        Returns:
            Analysis results
        """
        print("=" * 60)
        print("Data Pipeline with EDA Integration")
        print("=" * 60)

        # Step 1: Load data
        data = self.load_data(file_path)

        # Step 2: Validate data
        if not self.validate_data(data):
            raise ValueError("Data validation failed")

        # Step 3: Perform EDA
        results = self.perform_eda(data)

        # Step 4: Generate reports
        self.generate_reports(results, prefix="pipeline")

        print("\n" + "=" * 60)
        print("Pipeline execution completed successfully!")
        print("=" * 60)

        return results


def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    data = pd.DataFrame({
        'transaction_id': range(1, 501),
        'amount': np.random.exponential(100, 500),
        'customer_age': np.random.randint(18, 70, 500),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 500),
        'satisfaction_score': np.random.randint(1, 11, 500),
        'is_member': np.random.choice([True, False], 500)
    })

    # Save to CSV
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    data.to_csv("output/sample_data.csv", index=False)
    return "output/sample_data.csv"


def main():
    """Run the pipeline example."""
    # Create sample data
    file_path = create_sample_data()
    print(f"Sample data created: {file_path}\n")

    # Initialize and run pipeline
    pipeline = DataPipeline(output_dir="output")
    results = pipeline.run(file_path)

    # Display summary
    print("\nPipeline Results Summary:")
    print(f"  Analysis completed: ✓")
    print(f"  Reports generated: ✓")
    print(f"  Output directory: output/")


if __name__ == "__main__":
    main()
