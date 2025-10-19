"""
Memory Cache - High-Speed In-Memory Caching

Provides fast in-memory caching with LRU eviction, TTL support,
and memory usage monitoring.
"""

import threading
import time
import weakref
from typing import Any, Dict, Optional, Union, Callable, Set
from collections import OrderedDict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryCacheEntry:
    """Entry in the memory cache."""
    value: Any
    timestamp: float
    ttl: Optional[float]
    access_count: int = 0
    size: int = 0


class MemoryCache:
    """
    High-performance in-memory cache with LRU eviction and TTL support.
    
    Features:
    - LRU (Least Recently Used) eviction policy
    - TTL (Time To Live) support per entry
    - Memory usage tracking and limits
    - Thread-safe operations
    - Access frequency tracking
    - Weak reference support for automatic cleanup
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory: int = 128 * 1024 * 1024,  # 128MB
                 default_ttl: Optional[float] = None,
                 cleanup_interval: float = 60.0):  # 1 minute
        self.max_size = max_size
        self.max_memory = max_memory
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        # Thread-safe ordered dict for LRU behavior
        self._cache: OrderedDict[str, MemoryCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._memory_usage = 0
        self._last_cleanup = time.time()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired = 0
        
        # Weak references for automatic cleanup
        self._weak_refs: Set[weakref.ref] = set()
        
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value."""
        if isinstance(value, str):
            return len(value.encode('utf-8'))
        elif isinstance(value, bytes):
            return len(value)
        elif isinstance(value, (int, float)):
            return 8
        elif isinstance(value, (list, tuple)):
            return sum(self._estimate_size(item) for item in value) + 64
        elif isinstance(value, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) 
                      for k, v in value.items()) + 64
        elif isinstance(value, set):
            return sum(self._estimate_size(item) for item in value) + 64
        else:
            # For complex objects, use a heuristic
            try:
                import sys
                return sys.getsizeof(value)
            except Exception:
                return 1024  # Default estimate
                
    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.ttl is not None:
                if current_time - entry.timestamp > entry.ttl:
                    expired_keys.append(key)
                    
        for key in expired_keys:
            self._remove_entry(key)
            self._expired += 1
            
        self._last_cleanup = current_time
        
    def _remove_entry(self, key: str) -> bool:
        """Remove an entry and update memory usage."""
        if key in self._cache:
            entry = self._cache[key]
            self._memory_usage -= entry.size
            del self._cache[key]
            return True
        return False
        
    def _evict_lru(self):
        """Evict least recently used entries to make space."""
        while (len(self._cache) >= self.max_size or 
               self._memory_usage > self.max_memory) and self._cache:
            # Remove the first (oldest) entry
            key = next(iter(self._cache))
            self._remove_entry(key)
            self._evictions += 1
            
    def _maybe_cleanup(self):
        """Perform cleanup if needed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self.cleanup_interval:
            self._cleanup_expired()
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        with self._lock:
            self._maybe_cleanup()
            
            if key not in self._cache:
                self._misses += 1
                return default
                
            entry = self._cache[key]
            current_time = time.time()
            
            # Check if expired
            if entry.ttl is not None:
                if current_time - entry.timestamp > entry.ttl:
                    self._remove_entry(key)
                    self._misses += 1
                    self._expired += 1
                    return default
                    
            # Update access statistics
            entry.access_count += 1
            self._hits += 1
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            return entry.value
            
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            self._maybe_cleanup()
            
            # Calculate size
            size = self._estimate_size(value)
            current_time = time.time()
            
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl
                
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
                
            # Check if single item is too large
            if size > self.max_memory:
                logger.warning(f"Item too large for cache: {size} bytes > {self.max_memory}")
                return False
                
            # Make space if needed
            self._evict_lru()
            
            # Create new entry
            entry = MemoryCacheEntry(
                value=value,
                timestamp=current_time,
                ttl=ttl,
                access_count=1,
                size=size
            )
            
            # Add to cache
            self._cache[key] = entry
            self._memory_usage += size
            
            return True
            
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was present and deleted
        """
        with self._lock:
            return self._remove_entry(key)
            
    def clear(self):
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._memory_usage = 0
            self._weak_refs.clear()
            
    def has_key(self, key: str) -> bool:
        """Check if key exists in cache (without affecting LRU order)."""
        with self._lock:
            if key not in self._cache:
                return False
                
            entry = self._cache[key]
            if entry.ttl is not None:
                current_time = time.time()
                if current_time - entry.timestamp > entry.ttl:
                    self._remove_entry(key)
                    self._expired += 1
                    return False
                    
            return True
            
    def keys(self) -> list:
        """Get list of all keys in cache."""
        with self._lock:
            self._maybe_cleanup()
            return list(self._cache.keys())
            
    def values(self) -> list:
        """Get list of all values in cache."""
        with self._lock:
            self._maybe_cleanup()
            return [entry.value for entry in self._cache.values()]
            
    def items(self):
        """Get iterator over key-value pairs."""
        with self._lock:
            self._maybe_cleanup()
            for key, entry in self._cache.items():
                yield key, entry.value
                
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage': self._memory_usage,
                'max_memory': self.max_memory,
                'memory_utilization': self._memory_usage / self.max_memory if self.max_memory > 0 else 0.0,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'expired': self._expired,
                'total_requests': total_requests
            }
            
    def resize(self, max_size: Optional[int] = None, 
              max_memory: Optional[int] = None):
        """
        Resize cache limits.
        
        Args:
            max_size: New maximum number of entries
            max_memory: New maximum memory usage in bytes
        """
        with self._lock:
            if max_size is not None:
                self.max_size = max_size
                
            if max_memory is not None:
                self.max_memory = max_memory
                
            # Evict entries if we're over the new limits
            self._evict_lru()
            
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self._memory_usage
        
    def get_size(self) -> int:
        """Get current number of entries."""
        return len(self._cache)
        
    def is_full(self) -> bool:
        """Check if cache is at capacity."""
        return (len(self._cache) >= self.max_size or 
                self._memory_usage >= self.max_memory)
                
    def get_oldest_key(self) -> Optional[str]:
        """Get the oldest (least recently used) key."""
        with self._lock:
            if self._cache:
                return next(iter(self._cache))
            return None
            
    def get_newest_key(self) -> Optional[str]:
        """Get the newest (most recently used) key."""
        with self._lock:
            if self._cache:
                return next(reversed(self._cache))
            return None
            
    def touch(self, key: str) -> bool:
        """
        Touch a key to mark it as recently used without retrieving the value.
        
        Args:
            key: Key to touch
            
        Returns:
            True if key exists and was touched
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry = self._cache[key]
                entry.access_count += 1
                return True
            return False
            
    def expire_key(self, key: str, ttl: float) -> bool:
        """
        Set or update TTL for an existing key.
        
        Args:
            key: Key to expire
            ttl: Time to live in seconds
            
        Returns:
            True if key exists and TTL was set
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.ttl = ttl
                entry.timestamp = time.time()  # Reset timestamp
                return True
            return False
            
    def set_weak_ref(self, obj: Any, cleanup_func: Callable[[], None]):
        """
        Set up weak reference for automatic cleanup.
        
        Args:
            obj: Object to create weak reference for
            cleanup_func: Function to call when object is garbage collected
        """
        def cleanup(ref):
            cleanup_func()
            self._weak_refs.discard(ref)
            
        weak_ref = weakref.ref(obj, cleanup)
        self._weak_refs.add(weak_ref)
        
    def __len__(self) -> int:
        """Return number of entries in cache."""
        return len(self._cache)
        
    def __contains__(self, key: str) -> bool:
        """Check if key is in cache."""
        return self.has_key(key)
        
    def __getitem__(self, key: str) -> Any:
        """Get item using dict-like syntax."""
        value = self.get(key)
        if value is None and key not in self._cache:
            raise KeyError(key)
        return value
        
    def __setitem__(self, key: str, value: Any):
        """Set item using dict-like syntax."""
        self.set(key, value)
        
    def __delitem__(self, key: str):
        """Delete item using dict-like syntax."""
        if not self.delete(key):
            raise KeyError(key)