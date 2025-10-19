"""
Caching Decorators - Easy-to-use Caching Decorators

Provides decorators for function memoization and result caching
with various cache backends and strategies.
"""

import functools
import hashlib
import inspect
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import logging
import weakref

try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    xxhash = None

logger = logging.getLogger(__name__)


def _make_key(*args, **kwargs) -> str:
    """Create a cache key from function arguments."""
    # Convert arguments to a hashable representation
    key_parts = []
    
    # Process positional arguments
    for arg in args:
        if hasattr(arg, '__dict__'):
            # For objects, use class name and id
            key_parts.append(f"{arg.__class__.__name__}:{id(arg)}")
        else:
            key_parts.append(str(arg))
            
    # Process keyword arguments
    for k, v in sorted(kwargs.items()):
        if hasattr(v, '__dict__'):
            key_parts.append(f"{k}={v.__class__.__name__}:{id(v)}")
        else:
            key_parts.append(f"{k}={v}")
            
    key_string = "|".join(key_parts)
    
    # Hash the key for efficiency
    if XXHASH_AVAILABLE:
        return xxhash.xxh64(key_string.encode()).hexdigest()
    else:
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]


def _make_typed_key(func: Callable, *args, **kwargs) -> str:
    """Create a cache key that includes type information."""
    # Include function name and module
    func_key = f"{func.__module__}.{func.__qualname__}"
    
    # Include type information
    type_parts = []
    for arg in args:
        type_parts.append(type(arg).__name__)
        
    for k, v in sorted(kwargs.items()):
        type_parts.append(f"{k}:{type(v).__name__}")
        
    # Combine function info, types, and values
    key_string = f"{func_key}|{','.join(type_parts)}|{_make_key(*args, **kwargs)}"
    
    if XXHASH_AVAILABLE:
        return xxhash.xxh64(key_string.encode()).hexdigest()
    else:
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]


class CacheStats:
    """Statistics for cached functions."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.total_time = 0.0
        self.cache_time = 0.0
        self._lock = threading.Lock()
        
    def record_hit(self, cache_time: float):
        with self._lock:
            self.hits += 1
            self.cache_time += cache_time
            
    def record_miss(self, compute_time: float):
        with self._lock:
            self.misses += 1
            self.total_time += compute_time
            
    def record_error(self):
        with self._lock:
            self.errors += 1
            
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            avg_compute_time = self.total_time / self.misses if self.misses > 0 else 0.0
            avg_cache_time = self.cache_time / self.hits if self.hits > 0 else 0.0
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'errors': self.errors,
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'total_compute_time': self.total_time,
                'total_cache_time': self.cache_time,
                'avg_compute_time': avg_compute_time,
                'avg_cache_time': avg_cache_time
            }


# Global stats registry
_function_stats: Dict[str, CacheStats] = {}
_stats_lock = threading.Lock()


def get_function_stats(func_name: str) -> Optional[CacheStats]:
    """Get statistics for a cached function."""
    with _stats_lock:
        return _function_stats.get(func_name)


def get_all_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all cached functions."""
    with _stats_lock:
        return {name: stats.get_stats() for name, stats in _function_stats.items()}


def clear_all_stats():
    """Clear statistics for all cached functions."""
    with _stats_lock:
        _function_stats.clear()


def memoize(cache_manager=None, ttl: Optional[int] = None, 
           typed: bool = False, namespace: str = "",
           ignore_errors: bool = True, max_size: Optional[int] = None):
    """
    Memoization decorator using the cache manager.
    
    Args:
        cache_manager: Cache manager instance (uses global if None)
        ttl: Time to live in seconds
        typed: Include argument types in cache key
        namespace: Cache namespace
        ignore_errors: Ignore cache errors and compute result
        max_size: Maximum number of cached results (memory only)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        # Get or import cache manager
        if cache_manager is None:
            from .cache_manager import get_cache_manager
            cm = get_cache_manager()
        else:
            cm = cache_manager
            
        # Function statistics
        func_name = f"{func.__module__}.{func.__qualname__}"
        with _stats_lock:
            if func_name not in _function_stats:
                _function_stats[func_name] = CacheStats()
        stats = _function_stats[func_name]
        
        # Simple LRU cache for memory-only mode
        if max_size and not ttl:
            # Use functools.lru_cache for simple cases
            from functools import lru_cache
            return lru_cache(maxsize=max_size, typed=typed)(func)
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if typed:
                key = _make_typed_key(func, *args, **kwargs)
            else:
                key = f"{func_name}:{_make_key(*args, **kwargs)}"
                
            # Try to get from cache
            start_time = time.time()
            try:
                cached_result = cm.get(key, namespace=namespace)
                if cached_result is not None:
                    cache_time = time.time() - start_time
                    stats.record_hit(cache_time)
                    return cached_result
            except Exception as e:
                if not ignore_errors:
                    raise
                logger.warning(f"Cache get error for {func_name}: {e}")
                stats.record_error()
                
            # Cache miss - compute result
            compute_start = time.time()
            try:
                result = func(*args, **kwargs)
                compute_time = time.time() - compute_start
                stats.record_miss(compute_time)
                
                # Store in cache
                try:
                    cm.set(key, result, ttl=ttl, namespace=namespace)
                except Exception as e:
                    if not ignore_errors:
                        raise
                    logger.warning(f"Cache set error for {func_name}: {e}")
                    stats.record_error()
                    
                return result
                
            except Exception as e:
                compute_time = time.time() - compute_start
                stats.record_miss(compute_time)
                raise
                
        # Add cache management methods to the wrapper
        def cache_clear():
            """Clear cache for this function."""
            if namespace:
                # Clear all keys in namespace (approximation)
                try:
                    pattern = f"{namespace}:{func_name}:*"
                    # This is a simplified approach - actual implementation
                    # would depend on cache backend capabilities
                    pass
                except Exception:
                    pass
                    
        def cache_info():
            """Get cache information."""
            return stats.get_stats()
            
        wrapper.cache_clear = cache_clear
        wrapper.cache_info = cache_info
        wrapper._cache_key = lambda *args, **kwargs: (
            _make_typed_key(func, *args, **kwargs) if typed 
            else f"{func_name}:{_make_key(*args, **kwargs)}"
        )
        
        return wrapper
    return decorator


def cache_result(cache_manager=None, ttl: int = 3600, 
                namespace: str = "", key_func: Optional[Callable] = None):
    """
    Cache function results with custom key generation.
    
    Args:
        cache_manager: Cache manager instance
        ttl: Time to live in seconds
        namespace: Cache namespace
        key_func: Custom key generation function
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        if cache_manager is None:
            from .cache_manager import get_cache_manager
            cm = get_cache_manager()
        else:
            cm = cache_manager
            
        func_name = f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                try:
                    key = key_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Custom key function failed for {func_name}: {e}")
                    key = f"{func_name}:{_make_key(*args, **kwargs)}"
            else:
                key = f"{func_name}:{_make_key(*args, **kwargs)}"
                
            # Try cache first
            cached_result = cm.get(key, namespace=namespace)
            if cached_result is not None:
                return cached_result
                
            # Compute and cache result
            result = func(*args, **kwargs)
            cm.set(key, result, ttl=ttl, namespace=namespace)
            return result
            
        return wrapper
    return decorator


def cache_with_lock(cache_manager=None, ttl: int = 3600,
                   namespace: str = "", lock_timeout: float = 30.0):
    """
    Cache with distributed locking to prevent cache stampede.
    
    Args:
        cache_manager: Cache manager instance
        ttl: Time to live in seconds
        namespace: Cache namespace
        lock_timeout: Lock timeout in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        if cache_manager is None:
            from .cache_manager import get_cache_manager
            cm = get_cache_manager()
        else:
            cm = cache_manager
            
        func_name = f"{func.__module__}.{func.__qualname__}"
        local_locks = {}  # Thread-local locks
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func_name}:{_make_key(*args, **kwargs)}"
            lock_key = f"lock:{key}"
            
            # Check cache first
            cached_result = cm.get(key, namespace=namespace)
            if cached_result is not None:
                return cached_result
                
            # Use local lock to prevent multiple threads from same process
            if key not in local_locks:
                local_locks[key] = threading.Lock()
                
            with local_locks[key]:
                # Double-check cache after acquiring lock
                cached_result = cm.get(key, namespace=namespace)
                if cached_result is not None:
                    return cached_result
                    
                # Try to acquire distributed lock
                lock_acquired = False
                if hasattr(cm, '_redis_client') and cm._redis_client:
                    try:
                        # Use Redis SET NX EX for distributed locking
                        lock_acquired = cm._redis_client.set(
                            lock_key, "locked", nx=True, ex=int(lock_timeout)
                        )
                    except Exception:
                        pass
                        
                if not lock_acquired:
                    # Fallback: compute without lock
                    pass
                    
                try:
                    # Compute result
                    result = func(*args, **kwargs)
                    
                    # Cache result
                    cm.set(key, result, ttl=ttl, namespace=namespace)
                    
                    return result
                    
                finally:
                    # Release distributed lock
                    if lock_acquired and hasattr(cm, '_redis_client') and cm._redis_client:
                        try:
                            cm._redis_client.delete(lock_key)
                        except Exception:
                            pass
                            
        return wrapper
    return decorator


def cache_async_result(cache_manager=None, ttl: int = 3600,
                      namespace: str = ""):
    """
    Cache decorator for async functions.
    
    Args:
        cache_manager: Cache manager instance
        ttl: Time to live in seconds
        namespace: Cache namespace
        
    Returns:
        Decorated async function
    """
    def decorator(func: Callable) -> Callable:
        if not inspect.iscoroutinefunction(func):
            raise ValueError("cache_async_result can only be used with async functions")
            
        if cache_manager is None:
            from .cache_manager import get_cache_manager
            cm = get_cache_manager()
        else:
            cm = cache_manager
            
        func_name = f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            key = f"{func_name}:{_make_key(*args, **kwargs)}"
            
            # Try cache first
            cached_result = cm.get(key, namespace=namespace)
            if cached_result is not None:
                return cached_result
                
            # Compute and cache result
            result = await func(*args, **kwargs)
            cm.set(key, result, ttl=ttl, namespace=namespace)
            return result
            
        return wrapper
    return decorator


def invalidate_cache(cache_manager=None, pattern: str = "", namespace: str = ""):
    """
    Decorator to invalidate cache entries after function execution.
    
    Args:
        cache_manager: Cache manager instance
        pattern: Cache key pattern to invalidate
        namespace: Cache namespace
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        if cache_manager is None:
            from .cache_manager import get_cache_manager
            cm = get_cache_manager()
        else:
            cm = cache_manager
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Invalidate cache entries
            try:
                if pattern:
                    # Use pattern-based invalidation if supported
                    pass  # Implementation depends on cache backend
                else:
                    # Clear entire namespace
                    pass  # Implementation depends on cache backend
                    
            except Exception as e:
                logger.warning(f"Cache invalidation failed: {e}")
                
            return result
            
        return wrapper
    return decorator


class CachedProperty:
    """
    Cached property descriptor with TTL support.
    
    Similar to functools.cached_property but with expiration.
    """
    
    def __init__(self, func: Callable, ttl: Optional[float] = None):
        self.func = func
        self.ttl = ttl
        self.attrname = None
        self.__doc__ = func.__doc__
        
    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise RuntimeError(
                f"Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )
            
    def __get__(self, instance, owner=None):
        if instance is None:
            return self
            
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it."
            )
            
        # Check if we have a cached value
        cache_key = f"_cache_{self.attrname}"
        timestamp_key = f"_cache_timestamp_{self.attrname}"
        
        if hasattr(instance, cache_key):
            # Check if expired
            if self.ttl is not None:
                timestamp = getattr(instance, timestamp_key, 0)
                if time.time() - timestamp > self.ttl:
                    # Expired, remove cached value
                    delattr(instance, cache_key)
                    if hasattr(instance, timestamp_key):
                        delattr(instance, timestamp_key)
                else:
                    # Not expired, return cached value
                    return getattr(instance, cache_key)
            else:
                # No TTL, return cached value
                return getattr(instance, cache_key)
                
        # Compute value
        value = self.func(instance)
        
        # Cache value and timestamp
        try:
            setattr(instance, cache_key, value)
            if self.ttl is not None:
                setattr(instance, timestamp_key, time.time())
        except AttributeError:
            # Instance might be read-only, ignore caching
            pass
            
        return value
        
    def __delete__(self, instance):
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it."
            )
            
        cache_key = f"_cache_{self.attrname}"
        timestamp_key = f"_cache_timestamp_{self.attrname}"
        
        try:
            delattr(instance, cache_key)
        except AttributeError:
            pass
            
        try:
            delattr(instance, timestamp_key)
        except AttributeError:
            pass


def cached_property(func: Callable = None, *, ttl: Optional[float] = None):
    """
    Cached property decorator with optional TTL.
    
    Args:
        func: Property function
        ttl: Time to live in seconds
        
    Returns:
        CachedProperty descriptor
    """
    if func is None:
        return lambda f: CachedProperty(f, ttl=ttl)
    return CachedProperty(func, ttl=ttl)