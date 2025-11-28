"""
Result caching system for the EDA framework.

This module provides caching functionality to avoid re-computation of analysis
results, with support for hash-based dataset fingerprinting, incremental updates,
and cache invalidation strategies.

Features:
- Hash-based dataset fingerprinting
- Configurable cache backends (memory, disk)
- Cache invalidation strategies (TTL, LRU)
- Incremental analysis updates
- Thread-safe operations
"""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import OrderedDict
import threading

import pandas as pd
import numpy as np

from simplus_eda.logging_config import get_logger
from simplus_eda.exceptions import SimplusEDAError

logger = get_logger(__name__)


# ============================================================================
# Dataset Fingerprinting
# ============================================================================

def compute_dataframe_hash(
    df: pd.DataFrame,
    method: str = 'fast',
    include_values: bool = False
) -> str:
    """
    Compute a hash fingerprint of a DataFrame.

    Args:
        df: DataFrame to hash
        method: Hashing method ('fast', 'full', 'structural')
            - 'fast': Hash shape, columns, dtypes, and sample
            - 'full': Hash all values (slow for large datasets)
            - 'structural': Hash only shape and column info
        include_values: Whether to include data values in hash

    Returns:
        Hash string (hexadecimal)

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> hash1 = compute_dataframe_hash(df)
        >>> hash2 = compute_dataframe_hash(df)
        >>> assert hash1 == hash2  # Same data = same hash
    """
    hasher = hashlib.sha256()

    # Always include structural information
    hasher.update(str(df.shape).encode())
    hasher.update('|'.join(df.columns).encode())
    hasher.update('|'.join(str(dt) for dt in df.dtypes).encode())

    if method == 'structural':
        # Only structural info
        pass

    elif method == 'fast' or (method == 'full' and len(df) > 10000):
        # Sample-based hashing for speed
        sample_size = min(1000, len(df))
        sample_indices = np.linspace(0, len(df) - 1, sample_size, dtype=int)
        sample = df.iloc[sample_indices]

        # Hash sample values
        for col in df.columns:
            try:
                col_bytes = sample[col].to_numpy().tobytes()
                hasher.update(col_bytes)
            except (TypeError, ValueError):
                # Handle non-numeric or object types
                col_str = '|'.join(str(v) for v in sample[col].values)
                hasher.update(col_str.encode())

    elif method == 'full' or include_values:
        # Full hash of all values (slow but accurate)
        for col in df.columns:
            try:
                col_bytes = df[col].to_numpy().tobytes()
                hasher.update(col_bytes)
            except (TypeError, ValueError):
                col_str = '|'.join(str(v) for v in df[col].values)
                hasher.update(col_str.encode())

    return hasher.hexdigest()


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute hash of configuration parameters.

    Args:
        config: Configuration dictionary

    Returns:
        Hash string

    Example:
        >>> config = {'threshold': 0.5, 'method': 'test'}
        >>> hash1 = compute_config_hash(config)
        >>> hash2 = compute_config_hash(config)
        >>> assert hash1 == hash2
    """
    # Sort keys for consistent hashing
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()


def compute_cache_key(
    df: pd.DataFrame,
    operation: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    Compute a unique cache key for an analysis operation.

    Args:
        df: Input DataFrame
        operation: Operation name (e.g., 'correlation', 'statistics')
        config: Configuration parameters
        **kwargs: Additional parameters to include in key

    Returns:
        Cache key string

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3]})
        >>> key = compute_cache_key(df, 'stats', config={'method': 'mean'})
    """
    components = [
        operation,
        compute_dataframe_hash(df, method='fast'),
    ]

    if config:
        components.append(compute_config_hash(config))

    if kwargs:
        kwargs_str = json.dumps(kwargs, sort_keys=True)
        components.append(hashlib.sha256(kwargs_str.encode()).hexdigest())

    return '|'.join(components)


# ============================================================================
# Cache Entry
# ============================================================================

@dataclass
class CacheEntry:
    """
    Cache entry with metadata.

    Attributes:
        key: Cache key
        value: Cached value
        created_at: Creation timestamp
        accessed_at: Last access timestamp
        access_count: Number of times accessed
        metadata: Additional metadata
    """
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1

    def age(self) -> float:
        """Get age in seconds."""
        return time.time() - self.created_at

    def is_expired(self, ttl: float) -> bool:
        """Check if entry has expired."""
        return self.age() > ttl

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'key': self.key,
            'created_at': self.created_at,
            'accessed_at': self.accessed_at,
            'access_count': self.access_count,
            'metadata': self.metadata,
        }


# ============================================================================
# Cache Backend Interface
# ============================================================================

class CacheBackend:
    """Base class for cache backends."""

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError

    def set(self, key: str, value: Any, metadata: Dict[str, Any] = None):
        """Set value in cache."""
        raise NotImplementedError

    def delete(self, key: str):
        """Delete value from cache."""
        raise NotImplementedError

    def clear(self):
        """Clear all cache entries."""
        raise NotImplementedError

    def keys(self) -> List[str]:
        """Get all cache keys."""
        raise NotImplementedError

    def size(self) -> int:
        """Get number of cache entries."""
        raise NotImplementedError


# ============================================================================
# Memory Cache Backend
# ============================================================================

class MemoryCacheBackend(CacheBackend):
    """
    In-memory cache backend with LRU eviction.

    Example:
        >>> cache = MemoryCacheBackend(max_size=100, ttl=3600)
        >>> cache.set('key', {'data': [1, 2, 3]})
        >>> value = cache.get('key')
    """

    def __init__(self, max_size: int = 100, ttl: Optional[float] = None):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of entries
            ttl: Time-to-live in seconds (None = no expiration)
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                return None

            # Check expiration
            if self.ttl and entry.is_expired(self.ttl):
                logger.debug(f"Cache entry expired: {key}")
                del self._cache[key]
                return None

            # Update access info and move to end (LRU)
            entry.touch()
            self._cache.move_to_end(key)

            return entry.value

    def set(self, key: str, value: Any, metadata: Dict[str, Any] = None):
        """Set value in cache."""
        with self._lock:
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                metadata=metadata or {}
            )

            # Remove if exists (to update position)
            if key in self._cache:
                del self._cache[key]

            # Add to cache
            self._cache[key] = entry

            # Evict oldest if necessary (LRU)
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                logger.debug(f"Evicting cache entry (LRU): {oldest_key}")
                del self._cache[oldest_key]

    def delete(self, key: str):
        """Delete value from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")

    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())

    def size(self) -> int:
        """Get number of cache entries."""
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = sum(e.access_count for e in self._cache.values())
            avg_age = sum(e.age() for e in self._cache.values()) / len(self._cache) if self._cache else 0

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'total_accesses': total_accesses,
                'average_age_seconds': avg_age,
                'ttl_seconds': self.ttl,
            }


# ============================================================================
# Disk Cache Backend
# ============================================================================

class DiskCacheBackend(CacheBackend):
    """
    Disk-based cache backend.

    Example:
        >>> cache = DiskCacheBackend(cache_dir='./cache', ttl=86400)
        >>> cache.set('key', {'data': [1, 2, 3]})
        >>> value = cache.get('key')
    """

    def __init__(
        self,
        cache_dir: str = './.simplus_cache',
        ttl: Optional[float] = None,
        max_size_mb: Optional[float] = None
    ):
        """
        Initialize disk cache.

        Args:
            cache_dir: Directory for cache files
            ttl: Time-to-live in seconds (None = no expiration)
            max_size_mb: Maximum cache size in MB (None = unlimited)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self.max_size_mb = max_size_mb
        self._lock = threading.RLock()

        logger.debug(f"Disk cache initialized at {self.cache_dir}")

    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use first 2 chars for subdirectory to avoid too many files in one dir
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        subdir = self.cache_dir / safe_key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{safe_key}.pkl"

    def _get_metadata_path(self, key: str) -> Path:
        """Get metadata file path."""
        file_path = self._get_file_path(key)
        return file_path.with_suffix('.meta.json')

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            file_path = self._get_file_path(key)
            meta_path = self._get_metadata_path(key)

            if not file_path.exists():
                return None

            # Check expiration
            if self.ttl:
                age = time.time() - file_path.stat().st_mtime
                if age > self.ttl:
                    logger.debug(f"Cache file expired: {key}")
                    file_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    return None

            # Load value
            try:
                with open(file_path, 'rb') as f:
                    value = pickle.load(f)

                # Update metadata
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    metadata['accessed_at'] = time.time()
                    metadata['access_count'] = metadata.get('access_count', 0) + 1
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f)

                return value

            except Exception as e:
                logger.error(f"Error loading cache file {file_path}: {e}")
                return None

    def set(self, key: str, value: Any, metadata: Dict[str, Any] = None):
        """Set value in cache."""
        with self._lock:
            file_path = self._get_file_path(key)
            meta_path = self._get_metadata_path(key)

            try:
                # Save value
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)

                # Save metadata
                meta = {
                    'key': key,
                    'created_at': time.time(),
                    'accessed_at': time.time(),
                    'access_count': 0,
                    'metadata': metadata or {}
                }
                with open(meta_path, 'w') as f:
                    json.dump(meta, f)

                # Check size limit
                if self.max_size_mb:
                    self._enforce_size_limit()

            except Exception as e:
                logger.error(f"Error saving cache file {file_path}: {e}")

    def delete(self, key: str):
        """Delete value from cache."""
        with self._lock:
            file_path = self._get_file_path(key)
            meta_path = self._get_metadata_path(key)

            file_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Disk cache cleared")

    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            keys = []
            for meta_file in self.cache_dir.rglob('*.meta.json'):
                try:
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)
                        keys.append(meta['key'])
                except Exception:
                    pass
            return keys

    def size(self) -> int:
        """Get number of cache entries."""
        return len(self.keys())

    def get_size_mb(self) -> float:
        """Get total cache size in MB."""
        total_size = 0
        for file_path in self.cache_dir.rglob('*.pkl'):
            total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)

    def _enforce_size_limit(self):
        """Enforce cache size limit by removing oldest entries."""
        if not self.max_size_mb:
            return

        current_size = self.get_size_mb()
        if current_size <= self.max_size_mb:
            return

        logger.info(f"Cache size {current_size:.1f} MB exceeds limit {self.max_size_mb} MB, evicting...")

        # Get all cache files with their ages
        files = []
        for file_path in self.cache_dir.rglob('*.pkl'):
            files.append((file_path, file_path.stat().st_mtime))

        # Sort by age (oldest first)
        files.sort(key=lambda x: x[1])

        # Remove oldest files until under limit
        for file_path, _ in files:
            if self.get_size_mb() <= self.max_size_mb:
                break

            meta_path = file_path.with_suffix('.meta.json')
            file_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            logger.debug(f"Evicted cache file: {file_path.name}")


# ============================================================================
# Result Cache Manager
# ============================================================================

class ResultCache:
    """
    Main cache manager for analysis results.

    Example:
        >>> cache = ResultCache(backend='memory', max_size=100)
        >>> @cache.cached('statistics')
        ... def compute_stats(df):
        ...     return df.describe()
    """

    def __init__(
        self,
        backend: str = 'memory',
        enabled: bool = True,
        **backend_kwargs
    ):
        """
        Initialize result cache.

        Args:
            backend: Cache backend ('memory' or 'disk')
            enabled: Whether caching is enabled
            **backend_kwargs: Backend-specific parameters
        """
        self.enabled = enabled

        if backend == 'memory':
            self.backend = MemoryCacheBackend(**backend_kwargs)
        elif backend == 'disk':
            self.backend = DiskCacheBackend(**backend_kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
        }

        logger.info(f"Result cache initialized (backend={backend}, enabled={enabled})")

    def get(
        self,
        df: pd.DataFrame,
        operation: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Get cached result.

        Args:
            df: Input DataFrame
            operation: Operation name
            config: Configuration parameters
            **kwargs: Additional parameters

        Returns:
            Cached result or None if not found
        """
        if not self.enabled:
            return None

        key = compute_cache_key(df, operation, config, **kwargs)
        value = self.backend.get(key)

        if value is not None:
            self._stats['hits'] += 1
            logger.debug(f"Cache hit: {operation}")
        else:
            self._stats['misses'] += 1
            logger.debug(f"Cache miss: {operation}")

        return value

    def set(
        self,
        df: pd.DataFrame,
        operation: str,
        value: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Cache a result.

        Args:
            df: Input DataFrame
            operation: Operation name
            value: Result to cache
            config: Configuration parameters
            **kwargs: Additional parameters
        """
        if not self.enabled:
            return

        key = compute_cache_key(df, operation, config, **kwargs)
        metadata = {
            'operation': operation,
            'df_shape': df.shape,
            'df_hash': compute_dataframe_hash(df, method='fast'),
        }

        self.backend.set(key, value, metadata)
        self._stats['sets'] += 1
        logger.debug(f"Cached result: {operation}")

    def cached(self, operation: str, config: Optional[Dict[str, Any]] = None):
        """
        Decorator for caching function results.

        Args:
            operation: Operation name
            config: Configuration parameters

        Example:
            >>> cache = ResultCache()
            >>> @cache.cached('statistics')
            ... def compute_stats(df):
            ...     return df.describe()
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(df: pd.DataFrame, *args, **kwargs):
                # Try to get from cache
                cached_result = self.get(df, operation, config, **kwargs)
                if cached_result is not None:
                    return cached_result

                # Compute result
                result = func(df, *args, **kwargs)

                # Cache result
                self.set(df, operation, result, config, **kwargs)

                return result

            return wrapper
        return decorator

    def invalidate(
        self,
        df: Optional[pd.DataFrame] = None,
        operation: Optional[str] = None
    ):
        """
        Invalidate cache entries.

        Args:
            df: DataFrame to invalidate (None = all)
            operation: Operation to invalidate (None = all)
        """
        if df is None and operation is None:
            # Clear all
            self.backend.clear()
            logger.info("Cache invalidated (all)")
        else:
            # TODO: Implement selective invalidation
            logger.warning("Selective cache invalidation not yet implemented, clearing all")
            self.backend.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0

        stats = {
            'enabled': self.enabled,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'sets': self._stats['sets'],
            'hit_rate': hit_rate,
            'size': self.backend.size(),
        }

        # Add backend-specific stats
        if hasattr(self.backend, 'get_stats'):
            stats.update(self.backend.get_stats())
        elif hasattr(self.backend, 'get_size_mb'):
            stats['size_mb'] = self.backend.get_size_mb()

        return stats

    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()

        print("\n=== Cache Statistics ===")
        print(f"Enabled: {stats['enabled']}")
        print(f"Hits: {stats['hits']}")
        print(f"Misses: {stats['misses']}")
        print(f"Hit Rate: {stats['hit_rate']:.1%}")
        print(f"Cached Entries: {stats['size']}")

        if 'size_mb' in stats:
            print(f"Disk Size: {stats['size_mb']:.2f} MB")

        print("=" * 24)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'ResultCache',
    'MemoryCacheBackend',
    'DiskCacheBackend',
    'CacheEntry',
    'compute_dataframe_hash',
    'compute_config_hash',
    'compute_cache_key',
]
