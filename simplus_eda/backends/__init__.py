"""
Backend implementations for large-scale data processing.

Provides different backends for handling datasets of various sizes:
- Pandas backend (default, in-memory)
- Dask backend (out-of-core, distributed)
"""

from simplus_eda.backends.dask_backend import (
    DaskBackend,
    DASK_AVAILABLE,
    use_dask_if_large,
    compute_if_dask
)

__all__ = [
    'DaskBackend',
    'DASK_AVAILABLE',
    'use_dask_if_large',
    'compute_if_dask',
]
