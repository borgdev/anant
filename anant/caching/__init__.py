"""
Advanced Caching System for Anant Knowledge Graph Library

This module provides a comprehensive multi-level caching system with:
- Memory caching with LRU eviction
- Redis distributed caching with failover
- Disk-based persistent caching
- Query-specific caching with intelligent invalidation
- Decorators for easy function memoization
- Automatic cache warming and management

Architecture Overview:
- CacheManager: Central coordination of all cache levels
- MemoryCache: Fast in-memory caching with TTL support  
- RedisCache: Distributed caching with connection pooling
- DiskCache: Persistent caching with compression and encryption
- QueryCache: Specialized caching for database and API queries
- Invalidation: Intelligent cache invalidation strategies
- Decorators: Easy-to-use caching decorators
- Utils: Serialization, key generation, and utilities

The system provides graceful fallbacks when dependencies are unavailable,
ensuring compatibility across different deployment environments.
"""

# Core cache management
from .cache_manager import CacheManager, get_cache_manager, reset_cache_manager

# Individual cache implementations
from .memory_cache import MemoryCache
from .redis_cache import RedisCache, RedisConfig
from .disk_cache import DiskCache
from .query_cache import QueryCache, QueryType, QueryMetadata
from .invalidation import CacheInvalidator, InvalidationStrategy, InvalidationRule
from .decorators import (
    memoize, cache_result, cache_with_lock, cache_async_result, 
    cached_property, CachedProperty, CacheStats, get_function_stats, 
    get_all_stats, clear_all_stats
)
from .utils import (
    KeyGenerator, Serializer, CacheKeyBuilder, safe_serialize, 
    safe_deserialize, compress_data, decompress_data, 
    encode_cache_value, decode_cache_value, get_cache_key_info,
    default_key_generator, default_serializer, fast_serializer, compact_serializer
)

# Global cache manager instance
_cache_manager = None

def get_cache_context():
    """Get the global cache context manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = get_cache_manager()
    return _cache_manager

def is_redis_available():
    """Check if Redis caching is available."""
    return get_cache_context().is_redis_available()

def clear_all_caches():
    """Clear all caches (memory, disk, Redis)."""
    return get_cache_context().clear_all()

def get_cache_stats():
    """Get comprehensive cache statistics."""
    return get_cache_context().get_stats()

def configure_cache(redis_url=None, max_memory_size=None, disk_cache_dir=None):
    """Configure cache settings."""
    return get_cache_context().configure(
        redis_url=redis_url,
        max_memory_size=max_memory_size, 
        disk_cache_dir=disk_cache_dir
    )

__version__ = "1.0.0"

__all__ = [
    # Core management
    "CacheManager",
    "get_cache_manager", 
    "reset_cache_manager",
    
    # Cache implementations
    "MemoryCache",
    "RedisCache", 
    "RedisConfig",
    "DiskCache",
    "QueryCache",
    "QueryType",
    "QueryMetadata",
    
    # Invalidation
    "CacheInvalidator",
    "InvalidationStrategy", 
    "InvalidationRule",
    
    # Decorators
    "memoize",
    "cache_result",
    "cache_with_lock",
    "cache_async_result",
    "cached_property",
    "CachedProperty",
    "CacheStats",
    "get_function_stats",
    "get_all_stats", 
    "clear_all_stats",
    
    # Utilities
    "KeyGenerator",
    "Serializer",
    "CacheKeyBuilder",
    "safe_serialize",
    "safe_deserialize", 
    "compress_data",
    "decompress_data",
    "encode_cache_value",
    "decode_cache_value",
    "get_cache_key_info",
    "default_key_generator",
    "default_serializer",
    "fast_serializer", 
    "compact_serializer",
    
    # Legacy/convenience functions
    "get_cache_context",
    "is_redis_available",
    "clear_all_caches",
    "get_cache_stats",
    "configure_cache",
]