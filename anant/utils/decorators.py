"""
Utility decorators for the Anant library

Provides performance monitoring, caching, and other utility decorators
to enhance functionality and debugging capabilities.
"""

import functools
import time
from typing import Any, Callable, Dict, Optional
import threading


# Simple cache for results
_cache = {}
_cache_lock = threading.Lock()


def performance_monitor(func: Callable) -> Callable:
    """
    Decorator to monitor function performance
    
    Tracks execution time and call frequency for debugging
    and optimization purposes.
    """
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # Store performance info (simplified implementation)
            func_name = f"{func.__module__}.{func.__name__}"
            if not hasattr(wrapper, '_perf_stats'):
                wrapper._perf_stats = {'calls': 0, 'total_time': 0.0}
            
            wrapper._perf_stats['calls'] += 1
            wrapper._perf_stats['total_time'] += duration
            
    return wrapper


def cache_result(func: Callable) -> Callable:
    """
    Simple result caching decorator for properties
    """
    
    cache_key = f"{func.__module__}.{func.__name__}"
    
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Create cache key from self and args/kwargs
        key = (cache_key, id(self), str(args), str(sorted(kwargs.items())))
        
        with _cache_lock:
            if key in _cache:
                return _cache[key]
            
            # Execute function
            result = func(self, *args, **kwargs)
            
            # Store in cache (with simple size management)
            if len(_cache) >= 128:
                # Remove oldest entry (simplified LRU)
                oldest_key = next(iter(_cache))
                del _cache[oldest_key]
            
            _cache[key] = result
            return result
    
    return wrapper