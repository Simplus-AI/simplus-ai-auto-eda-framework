"""
Utility functions and helpers for the EDA framework.
"""

from simplus_eda.utils.data_loader import DataLoader
from simplus_eda.utils.validators import DataValidator
from simplus_eda.utils.formatters import OutputFormatter
from simplus_eda.utils.parallel import ParallelProcessor
from simplus_eda.utils.sampling import (
    DataSampler,
    SamplingMethod,
    smart_sample,
)
from simplus_eda.utils.memory import (
    MemoryProfiler,
    get_system_memory_info,
    optimize_dataframe_memory,
    check_memory_available,
    suggest_optimizations,
    PSUTIL_AVAILABLE,
)
from simplus_eda.utils.chunked import (
    ChunkedDataReader,
    StreamingAggregator,
    process_in_chunks,
    streaming_statistics,
    sample_large_file,
    chunked_value_counts,
)

__all__ = [
    # Core utilities
    "DataLoader",
    "DataValidator",
    "OutputFormatter",
    "ParallelProcessor",
    # Sampling
    "DataSampler",
    "SamplingMethod",
    "smart_sample",
    # Memory
    "MemoryProfiler",
    "get_system_memory_info",
    "optimize_dataframe_memory",
    "check_memory_available",
    "suggest_optimizations",
    "PSUTIL_AVAILABLE",
    # Chunked processing
    "ChunkedDataReader",
    "StreamingAggregator",
    "process_in_chunks",
    "streaming_statistics",
    "sample_large_file",
    "chunked_value_counts",
]
