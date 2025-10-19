"""
Cache Manager - Central Cache Coordination

Coordinates multi-level caching across memory, disk, and Redis backends
with intelligent fallback strategies and unified API.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

# Optional imports with graceful fallbacks
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    import diskcache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False
    diskcache = None

try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    xxhash = None

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    msgpack = None

import pickle
import json
import os
import weakref

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels in order of speed."""
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    memory_usage: int = 0
    disk_usage: int = 0
    redis_usage: int = 0
    hit_rate: float = 0.0


class CacheManager:
    """
    Central cache manager with multi-level caching strategy.
    
    Provides a unified interface for caching across memory, disk, and Redis
    with intelligent fallback and cache level optimization.
    """
    
    def __init__(self, 
                 redis_url: Optional[str] = None,
                 max_memory_size: int = 128 * 1024 * 1024,  # 128MB
                 disk_cache_dir: Optional[str] = None,
                 default_ttl: int = 3600):  # 1 hour
        self.max_memory_size = max_memory_size
        self.disk_cache_dir = disk_cache_dir or os.path.expanduser("~/.anant_cache")
        self.default_ttl = default_ttl
        self._stats = CacheStats()
        self._lock = threading.RLock()
        
        # Initialize cache backends
        self._memory_cache = {}
        self._memory_timestamps = {}
        self._memory_ttls = {}
        self._memory_size = 0
        
        # Initialize Redis cache
        self._redis_client = None
        if redis_url and REDIS_AVAILABLE:
            try:
                self._redis_client = redis.from_url(redis_url)
                self._redis_client.ping()  # Test connection
                logger.info(f"Redis cache connected: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using local caches only")
                self._redis_client = None
        elif REDIS_AVAILABLE:
            # Try default Redis connection
            try:
                self._redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self._redis_client.ping()
                logger.info("Redis cache connected: localhost:6379")
            except Exception:
                logger.info("Redis not available, using local caches only")
                self._redis_client = None
        
        # Initialize disk cache
        self._disk_cache = None
        if DISKCACHE_AVAILABLE:
            try:
                os.makedirs(self.disk_cache_dir, exist_ok=True)
                self._disk_cache = diskcache.Cache(self.disk_cache_dir)
                logger.info(f"Disk cache initialized: {self.disk_cache_dir}")
            except Exception as e:
                logger.warning(f"Disk cache initialization failed: {e}")
                self._disk_cache = None
        
        # Cache level preferences (fastest first)
        self._cache_levels = []
        if self._memory_cache is not None:
            self._cache_levels.append(CacheLevel.MEMORY)
        if self._disk_cache is not None:
            self._cache_levels.append(CacheLevel.DISK)
        if self._redis_client is not None:
            self._cache_levels.append(CacheLevel.REDIS)
            
        logger.info(f"Cache manager initialized with levels: {[level.value for level in self._cache_levels]}")
        
    def _generate_key(self, key: str, namespace: str = "") -> str:
        """Generate a cache key with optional namespace."""
        if namespace:
            full_key = f"{namespace}:{key}"
        else:
            full_key = key
            
        # Use fast hashing if available
        if XXHASH_AVAILABLE:
            return xxhash.xxh64(full_key.encode()).hexdigest()
        else:
            return hashlib.sha256(full_key.encode()).hexdigest()[:16]
            
    def _serialize(self, data: Any) -> bytes:
        """Serialize data for storage."""
        if MSGPACK_AVAILABLE:
            try:
                return msgpack.packb(data, use_bin_type=True)
            except Exception:
                pass
        
        # Fallback to pickle
        return pickle.dumps(data)
        
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from storage."""
        if MSGPACK_AVAILABLE:
            try:
                return msgpack.unpackb(data, raw=False)
            except Exception:
                pass
        
        # Fallback to pickle
        return pickle.loads(data)
        
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data."""
        if isinstance(data, (str, bytes)):
            return len(data)
        elif isinstance(data, (int, float)):
            return 8
        elif isinstance(data, (list, tuple)):
            return sum(self._estimate_size(item) for item in data)
        elif isinstance(data, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in data.items())
        else:
            # Fallback: serialize and measure
            try:
                return len(self._serialize(data))
            except Exception:
                return 1024  # Default estimate
                
    def _cleanup_memory_cache(self):
        """Clean up expired entries and enforce size limits."""
        current_time = time.time()
        
        # Remove expired entries
        expired_keys = []
        for key, timestamp in self._memory_timestamps.items():
            ttl = self._memory_ttls.get(key, self.default_ttl)
            if current_time - timestamp > ttl:
                expired_keys.append(key)
                
        for key in expired_keys:
            self._remove_from_memory(key)
            self._stats.evictions += 1
            
        # Enforce size limits (LRU eviction)
        while self._memory_size > self.max_memory_size and self._memory_cache:
            # Find oldest entry
            oldest_key = min(self._memory_timestamps.keys(), 
                           key=lambda k: self._memory_timestamps[k])
            self._remove_from_memory(oldest_key)
            self._stats.evictions += 1
            
    def _remove_from_memory(self, key: str):
        """Remove entry from memory cache."""
        if key in self._memory_cache:
            data_size = self._estimate_size(self._memory_cache[key])
            del self._memory_cache[key]
            del self._memory_timestamps[key]
            if key in self._memory_ttls:
                del self._memory_ttls[key]
            self._memory_size -= data_size
            
    def get(self, key: str, namespace: str = "", default: Any = None) -> Any:
        """
        Get value from cache, checking all levels.
        
        Args:
            key: Cache key
            namespace: Optional namespace for key organization
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        cache_key = self._generate_key(key, namespace)
        
        with self._lock:
            # Check memory cache first
            if cache_key in self._memory_cache:
                # Check if expired
                current_time = time.time()
                timestamp = self._memory_timestamps[cache_key]
                ttl = self._memory_ttls.get(cache_key, self.default_ttl)
                
                if current_time - timestamp <= ttl:
                    self._stats.hits += 1
                    # Update access time
                    self._memory_timestamps[cache_key] = current_time
                    return self._memory_cache[cache_key]
                else:
                    # Expired, remove
                    self._remove_from_memory(cache_key)
                    
            # Check disk cache
            if self._disk_cache is not None:
                try:
                    value = self._disk_cache.get(cache_key)
                    if value is not None:
                        self._stats.hits += 1
                        # Promote to memory cache
                        self._set_memory(cache_key, value, self.default_ttl)
                        return value
                except Exception as e:
                    logger.warning(f"Disk cache get error: {e}")
                    
            # Check Redis cache
            if self._redis_client is not None:
                try:
                    value = self._redis_client.get(cache_key)
                    if value is not None:
                        self._stats.hits += 1
                        # Deserialize and promote to memory
                        deserialized = self._deserialize(value)
                        self._set_memory(cache_key, deserialized, self.default_ttl)
                        return deserialized
                except Exception as e:
                    logger.warning(f"Redis cache get error: {e}")
                    
            self._stats.misses += 1
            return default
            
    def _set_memory(self, cache_key: str, value: Any, ttl: int):
        """Set value in memory cache."""
        data_size = self._estimate_size(value)
        
        # Remove existing entry if present
        if cache_key in self._memory_cache:
            self._remove_from_memory(cache_key)
            
        # Add new entry
        self._memory_cache[cache_key] = value
        self._memory_timestamps[cache_key] = time.time()
        self._memory_ttls[cache_key] = ttl
        self._memory_size += data_size
        
        # Cleanup if needed
        self._cleanup_memory_cache()
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
           namespace: str = "", levels: Optional[List[CacheLevel]] = None) -> bool:
        """
        Set value in cache at specified levels.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            namespace: Optional namespace
            levels: Cache levels to use (all by default)
            
        Returns:
            True if successfully cached at least one level
        """
        cache_key = self._generate_key(key, namespace)
        ttl = ttl or self.default_ttl
        levels = levels or self._cache_levels
        
        success = False
        
        with self._lock:
            # Set in memory cache
            if CacheLevel.MEMORY in levels and CacheLevel.MEMORY in self._cache_levels:
                try:
                    self._set_memory(cache_key, value, ttl)
                    success = True
                except Exception as e:
                    logger.warning(f"Memory cache set error: {e}")
                    
            # Set in disk cache
            if CacheLevel.DISK in levels and self._disk_cache is not None:
                try:
                    self._disk_cache.set(cache_key, value, expire=ttl)
                    success = True
                except Exception as e:
                    logger.warning(f"Disk cache set error: {e}")
                    
            # Set in Redis cache
            if CacheLevel.REDIS in levels and self._redis_client is not None:
                try:
                    serialized = self._serialize(value)
                    self._redis_client.setex(cache_key, ttl, serialized)
                    success = True
                except Exception as e:
                    logger.warning(f"Redis cache set error: {e}")
                    
            if success:
                self._stats.sets += 1
                
        return success
        
    def delete(self, key: str, namespace: str = "") -> bool:
        """
        Delete value from all cache levels.
        
        Args:
            key: Cache key to delete
            namespace: Optional namespace
            
        Returns:
            True if deleted from at least one level
        """
        cache_key = self._generate_key(key, namespace)
        success = False
        
        with self._lock:
            # Delete from memory
            if cache_key in self._memory_cache:
                self._remove_from_memory(cache_key)
                success = True
                
            # Delete from disk
            if self._disk_cache is not None:
                try:
                    if self._disk_cache.delete(cache_key):
                        success = True
                except Exception as e:
                    logger.warning(f"Disk cache delete error: {e}")
                    
            # Delete from Redis
            if self._redis_client is not None:
                try:
                    if self._redis_client.delete(cache_key):
                        success = True
                except Exception as e:
                    logger.warning(f"Redis cache delete error: {e}")
                    
            if success:
                self._stats.deletes += 1
                
        return success
        
    def clear_all(self) -> bool:
        """Clear all caches."""
        success = False
        
        with self._lock:
            # Clear memory
            self._memory_cache.clear()
            self._memory_timestamps.clear()
            self._memory_ttls.clear()
            self._memory_size = 0
            success = True
            
            # Clear disk
            if self._disk_cache is not None:
                try:
                    self._disk_cache.clear()
                    success = True
                except Exception as e:
                    logger.warning(f"Disk cache clear error: {e}")
                    
            # Clear Redis
            if self._redis_client is not None:
                try:
                    self._redis_client.flushdb()
                    success = True
                except Exception as e:
                    logger.warning(f"Redis cache clear error: {e}")
                    
        return success
        
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            # Update hit rate
            total_requests = self._stats.hits + self._stats.misses
            if total_requests > 0:
                self._stats.hit_rate = self._stats.hits / total_requests
                
            # Update memory usage
            self._stats.memory_usage = self._memory_size
            
            # Update disk usage
            if self._disk_cache is not None:
                try:
                    self._stats.disk_usage = self._disk_cache.volume()
                except Exception:
                    pass
                    
            # Update Redis usage
            if self._redis_client is not None:
                try:
                    info = self._redis_client.info('memory')
                    self._stats.redis_usage = info.get('used_memory', 0)
                except Exception:
                    pass
                    
            return self._stats
            
    def is_redis_available(self) -> bool:
        """Check if Redis is available."""
        return self._redis_client is not None
        
    def configure(self, redis_url: Optional[str] = None,
                 max_memory_size: Optional[int] = None,
                 disk_cache_dir: Optional[str] = None) -> bool:
        """
        Reconfigure cache settings.
        
        Args:
            redis_url: Redis connection URL
            max_memory_size: Maximum memory cache size
            disk_cache_dir: Disk cache directory
            
        Returns:
            True if reconfiguration successful
        """
        success = True
        
        with self._lock:
            # Update memory limit
            if max_memory_size is not None:
                self.max_memory_size = max_memory_size
                self._cleanup_memory_cache()
                
            # Update Redis connection
            if redis_url is not None and REDIS_AVAILABLE:
                try:
                    if self._redis_client:
                        self._redis_client.close()
                    self._redis_client = redis.from_url(redis_url)
                    self._redis_client.ping()
                    logger.info(f"Redis reconfigured: {redis_url}")
                except Exception as e:
                    logger.warning(f"Redis reconfiguration failed: {e}")
                    self._redis_client = None
                    success = False
                    
            # Update disk cache
            if disk_cache_dir is not None and DISKCACHE_AVAILABLE:
                try:
                    if self._disk_cache:
                        self._disk_cache.close()
                    os.makedirs(disk_cache_dir, exist_ok=True)
                    self._disk_cache = diskcache.Cache(disk_cache_dir)
                    self.disk_cache_dir = disk_cache_dir
                    logger.info(f"Disk cache reconfigured: {disk_cache_dir}")
                except Exception as e:
                    logger.warning(f"Disk cache reconfiguration failed: {e}")
                    self._disk_cache = None
                    success = False
                    
        return success


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager(**kwargs) -> CacheManager:
    """Get or create the global cache manager instance."""
    global _global_cache_manager
    
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(**kwargs)
        
    return _global_cache_manager


def reset_cache_manager():
    """Reset the global cache manager (useful for testing)."""
    global _global_cache_manager
    if _global_cache_manager is not None:
        _global_cache_manager.clear_all()
    _global_cache_manager = None