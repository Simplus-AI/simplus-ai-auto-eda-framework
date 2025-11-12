"""
Setup configuration for Simplus AI Auto EDA Framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="simplus-eda",
    version="0.1.0",
    author="Simplus AI Team",
    author_email="contact@simplusai.com",
    description="A comprehensive automated exploratory data analysis framework for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/simplus-ai-auto-eda-framework",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/simplus-ai-auto-eda-framework/issues",
        "Documentation": "https://simplus-eda.readthedocs.io",
        "Source Code": "https://github.com/yourusername/simplus-ai-auto-eda-framework",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "pdf": [
            "weasyprint>=54.0",  # For PDF report generation
        ],
    },
    entry_points={
        "console_scripts": [
            "simplus-eda=simplus_eda.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "eda",
        "exploratory-data-analysis",
        "data-analysis",
        "data-science",
        "statistics",
        "visualization",
        "automated-analysis",
    ],
)
