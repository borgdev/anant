"""
Performance Optimization Engine for Anant

This module provides comprehensive performance optimization capabilities including:
- Lazy evaluation frameworks for deferred computation
- Streaming data processing for large datasets
- Advanced caching mechanisms with intelligent cache policies
- Memory optimization strategies and monitoring
- Parallel processing enhancements
- Query optimization and execution planning
"""

from typing import Dict, List, Optional, Union, Any, Callable, Iterator, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import polars as pl
import threading
import time
import hashlib
import gc
import psutil
import weakref
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod

from ..classes import Hypergraph


class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    TTL = "ttl"  # Time To Live
    SIZE_BASED = "size_based"  # Based on memory size


class OptimizationLevel(Enum):
    """Optimization levels for performance tuning"""
    NONE = 0
    BASIC = 1
    AGGRESSIVE = 2
    MAXIMUM = 3


class StreamingMode(Enum):
    """Streaming processing modes"""
    CHUNK_BASED = "chunk"
    ROW_BASED = "row"
    LAZY_EVALUATION = "lazy"
    PIPELINE = "pipeline"


@dataclass
class CacheConfig:
    """Configuration for caching system"""
    policy: CachePolicy = CachePolicy.LRU
    max_size_mb: int = 1024  # 1GB default
    max_entries: int = 10000
    ttl_seconds: int = 3600  # 1 hour
    enable_compression: bool = True
    enable_persistence: bool = False
    persist_path: Optional[Path] = None


@dataclass
class StreamingConfig:
    """Configuration for streaming operations"""
    mode: StreamingMode = StreamingMode.CHUNK_BASED
    chunk_size: int = 10000
    max_memory_mb: int = 512
    enable_parallel: bool = True
    max_workers: int = 4
    buffer_size: int = 1000
    enable_lazy_loading: bool = True


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization"""
    level: OptimizationLevel = OptimizationLevel.BASIC
    enable_caching: bool = True
    enable_streaming: bool = True
    enable_parallel: bool = True
    enable_memory_optimization: bool = True
    enable_query_optimization: bool = True
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    streaming_config: StreamingConfig = field(default_factory=StreamingConfig)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    operation: str
    start_time: float
    end_time: float
    memory_before_mb: float
    memory_after_mb: float
    cache_hits: int = 0
    cache_misses: int = 0
    processed_rows: int = 0
    parallel_workers: int = 1
    
    @property
    def execution_time(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def memory_delta_mb(self) -> float:
        return self.memory_after_mb - self.memory_before_mb
    
    @property
    def cache_hit_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class CacheEntry:
    """Individual cache entry with metadata"""
    
    def __init__(self, key: str, value: Any, size_bytes: int = 0):
        self.key = key
        self.value = value
        self.size_bytes = size_bytes
        self.created_at = time.time()
        self.accessed_at = time.time()
        self.access_count = 1
        self.hit_count = 0
    
    def touch(self):
        """Update access metadata"""
        self.accessed_at = time.time()
        self.access_count += 1
        self.hit_count += 1
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at
    
    @property
    def idle_seconds(self) -> float:
        return time.time() - self.accessed_at


class SmartCache:
    """
    Intelligent caching system with multiple eviction policies and optimization
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: OrderedDict[str, float] = OrderedDict()
        self._frequency_map: Dict[str, int] = defaultdict(int)
        self._total_size_bytes = 0
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_evictions": 0,
            "ttl_evictions": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check TTL if applicable
                if (self.config.policy == CachePolicy.TTL and 
                    entry.age_seconds > self.config.ttl_seconds):
                    self._evict_key(key)
                    self._stats["ttl_evictions"] += 1
                    return None
                
                # Update access patterns
                entry.touch()
                self._access_order.move_to_end(key)
                self._frequency_map[key] += 1
                self._stats["hits"] += 1
                
                return entry.value
            
            self._stats["misses"] += 1
            return None
    
    def put(self, key: str, value: Any, size_bytes: int = 0) -> bool:
        """Put value in cache"""
        with self._lock:
            # Estimate size if not provided
            if size_bytes == 0:
                size_bytes = self._estimate_size(value)
            
            # Check if we need to evict
            while self._should_evict(size_bytes):
                if not self._evict_one():
                    return False  # Could not evict
            
            # Remove existing entry if present
            if key in self._cache:
                self._evict_key(key)
            
            # Add new entry
            entry = CacheEntry(key, value, size_bytes)
            self._cache[key] = entry
            self._access_order[key] = time.time()
            self._frequency_map[key] = 1
            self._total_size_bytes += size_bytes
            
            return True
    
    def _should_evict(self, incoming_size: int) -> bool:
        """Check if eviction is needed"""
        # Size-based eviction
        max_size_bytes = self.config.max_size_mb * 1024 * 1024
        if self._total_size_bytes + incoming_size > max_size_bytes:
            return True
        
        # Entry count-based eviction
        if len(self._cache) >= self.config.max_entries:
            return True
        
        return False
    
    def _evict_one(self) -> bool:
        """Evict one entry based on policy"""
        if not self._cache:
            return False
        
        if self.config.policy == CachePolicy.LRU:
            key_to_evict = next(iter(self._access_order))
        elif self.config.policy == CachePolicy.LFU:
            key_to_evict = min(self._frequency_map.keys(), 
                             key=lambda k: self._frequency_map[k])
        elif self.config.policy == CachePolicy.FIFO:
            key_to_evict = next(iter(self._cache))
        elif self.config.policy == CachePolicy.TTL:
            # Evict expired entries first
            now = time.time()
            expired_keys = [k for k, entry in self._cache.items() 
                          if entry.age_seconds > self.config.ttl_seconds]
            if expired_keys:
                key_to_evict = expired_keys[0]
            else:
                key_to_evict = next(iter(self._access_order))
        elif self.config.policy == CachePolicy.SIZE_BASED:
            # Evict largest entry
            key_to_evict = max(self._cache.keys(), 
                             key=lambda k: self._cache[k].size_bytes)
        else:
            key_to_evict = next(iter(self._cache))
        
        self._evict_key(key_to_evict)
        self._stats["evictions"] += 1
        return True
    
    def _evict_key(self, key: str):
        """Remove specific key from cache"""
        if key in self._cache:
            entry = self._cache[key]
            self._total_size_bytes -= entry.size_bytes
            del self._cache[key]
            self._access_order.pop(key, None)
            self._frequency_map.pop(key, None)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        if isinstance(obj, pl.DataFrame):
            return int(obj.estimated_size("bytes"))
        elif isinstance(obj, (str, bytes)):
            return len(obj)
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_size(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) 
                      for k, v in obj.items())
        else:
            # Rough estimate
            return 1024  # 1KB default
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._frequency_map.clear()
            self._total_size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_ratio = self._stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                **self._stats,
                "total_requests": total_requests,
                "hit_ratio": hit_ratio,
                "cache_size": len(self._cache),
                "total_size_mb": self._total_size_bytes / (1024 * 1024),
                "avg_entry_size_kb": (self._total_size_bytes / len(self._cache) / 1024) 
                                   if self._cache else 0
            }


class LazyDataFrame:
    """
    Simplified lazy evaluation wrapper for DataFrame operations
    """
    
    def __init__(self, data_source: Union[pl.DataFrame, pl.LazyFrame, Callable]):
        if isinstance(data_source, pl.DataFrame):
            self._lazy_frame = data_source.lazy()
        elif isinstance(data_source, pl.LazyFrame):
            self._lazy_frame = data_source
        elif callable(data_source):
            self._data_fn = data_source
            self._lazy_frame = None
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")
        
        self._operations: List[Callable] = []
        self._materialized = False
        self._result: Optional[pl.DataFrame] = None
    
    def filter(self, condition) -> 'LazyDataFrame':
        """Add filter operation"""
        if self._lazy_frame is not None:
            try:
                self._lazy_frame = self._lazy_frame.filter(condition)
            except:
                # Fall back to operation queue
                self._operations.append(lambda df: df.filter(condition))
        else:
            self._operations.append(lambda df: df.filter(condition))
        return self
    
    def select(self, columns) -> 'LazyDataFrame':
        """Add select operation"""
        if self._lazy_frame is not None:
            try:
                self._lazy_frame = self._lazy_frame.select(columns)
            except:
                # Fall back to operation queue
                self._operations.append(lambda df: df.select(columns))
        else:
            self._operations.append(lambda df: df.select(columns))
        return self
    
    def with_columns(self, exprs) -> 'LazyDataFrame':
        """Add column transformation"""
        if self._lazy_frame is not None:
            try:
                self._lazy_frame = self._lazy_frame.with_columns(exprs)
            except:
                # Fall back to operation queue
                self._operations.append(lambda df: df.with_columns(exprs))
        else:
            self._operations.append(lambda df: df.with_columns(exprs))
        return self
    
    def collect(self) -> pl.DataFrame:
        """Materialize the lazy operations"""
        if self._materialized and self._result is not None:
            return self._result
        
        if self._lazy_frame is not None:
            try:
                result = self._lazy_frame.collect()
            except:
                # Fall back to operations if collect fails
                if hasattr(self, '_data_fn'):
                    result = self._data_fn()
                else:
                    result = pl.DataFrame()
                for op in self._operations:
                    result = op(result)
        else:
            # Execute function and apply operations
            result = self._data_fn()
            for op in self._operations:
                result = op(result)
        
        self._result = result
        self._materialized = True
        return result
    
    def __len__(self) -> int:
        return len(self.collect())
    
    def __getitem__(self, key):
        return self.collect()[key]


class StreamProcessor:
    """
    Streaming data processor for large datasets
    """
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self._memory_monitor = MemoryMonitor()
    
    def process_chunks(
        self, 
        data_source: Union[Path, pl.DataFrame, Iterator],
        processor: Callable[[pl.DataFrame], pl.DataFrame],
        output_path: Optional[Path] = None
    ) -> Iterator[pl.DataFrame]:
        """Process data in chunks"""
        
        if isinstance(data_source, Path):
            # Stream from file
            yield from self._process_file_chunks(data_source, processor, output_path)
        elif isinstance(data_source, pl.DataFrame):
            # Stream from DataFrame
            yield from self._process_dataframe_chunks(data_source, processor, output_path)
        else:
            # Stream from iterator
            yield from self._process_iterator_chunks(data_source, processor, output_path)
    
    def _process_file_chunks(
        self, 
        file_path: Path, 
        processor: Callable, 
        output_path: Optional[Path]
    ) -> Iterator[pl.DataFrame]:
        """Process file in chunks"""
        
        if file_path.suffix == '.parquet':
            # Use Polars scan_parquet for lazy loading
            lazy_df = pl.scan_parquet(file_path)
            
            # Process in chunks
            total_rows = lazy_df.collect().height
            for i in range(0, total_rows, self.config.chunk_size):
                chunk = lazy_df.slice(i, self.config.chunk_size).collect()
                processed_chunk = processor(chunk)
                
                if output_path:
                    self._write_chunk(processed_chunk, output_path, i // self.config.chunk_size)
                
                yield processed_chunk
                
                # Memory management
                if self._memory_monitor.get_usage_mb() > self.config.max_memory_mb:
                    gc.collect()
        
        elif file_path.suffix == '.csv':
            # Process CSV in chunks using scan_csv
            lazy_df = pl.scan_csv(file_path)
            
            # Estimate total rows for chunking
            total_rows = lazy_df.collect().height
            for i in range(0, total_rows, self.config.chunk_size):
                chunk = lazy_df.slice(i, self.config.chunk_size).collect()
                processed_chunk = processor(chunk)
                
                if output_path:
                    self._write_chunk(processed_chunk, output_path, i // self.config.chunk_size)
                
                yield processed_chunk
                
                # Memory management
                if self._memory_monitor.get_usage_mb() > self.config.max_memory_mb:
                    gc.collect()
    
    def _process_dataframe_chunks(
        self, 
        df: pl.DataFrame, 
        processor: Callable,
        output_path: Optional[Path]
    ) -> Iterator[pl.DataFrame]:
        """Process DataFrame in chunks"""
        
        total_rows = len(df)
        for i in range(0, total_rows, self.config.chunk_size):
            chunk = df.slice(i, self.config.chunk_size)
            processed_chunk = processor(chunk)
            
            if output_path:
                self._write_chunk(processed_chunk, output_path, i // self.config.chunk_size)
            
            yield processed_chunk
            
            # Memory management
            if self._memory_monitor.get_usage_mb() > self.config.max_memory_mb:
                gc.collect()
    
    def _process_iterator_chunks(
        self, 
        iterator: Iterator, 
        processor: Callable,
        output_path: Optional[Path]
    ) -> Iterator[pl.DataFrame]:
        """Process iterator in chunks"""
        
        chunk_buffer = []
        chunk_id = 0
        
        for item in iterator:
            chunk_buffer.append(item)
            
            if len(chunk_buffer) >= self.config.chunk_size:
                # Convert buffer to DataFrame and process
                chunk_df = pl.DataFrame(chunk_buffer)
                processed_chunk = processor(chunk_df)
                
                if output_path:
                    self._write_chunk(processed_chunk, output_path, chunk_id)
                
                yield processed_chunk
                
                # Reset buffer
                chunk_buffer = []
                chunk_id += 1
                
                # Memory management
                if self._memory_monitor.get_usage_mb() > self.config.max_memory_mb:
                    gc.collect()
        
        # Process remaining items
        if chunk_buffer:
            chunk_df = pl.DataFrame(chunk_buffer)
            processed_chunk = processor(chunk_df)
            
            if output_path:
                self._write_chunk(processed_chunk, output_path, chunk_id)
            
            yield processed_chunk
    
    def _write_chunk(self, chunk: pl.DataFrame, output_path: Path, chunk_id: int):
        """Write processed chunk to output"""
        chunk_path = output_path / f"chunk_{chunk_id:06d}.parquet"
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        chunk.write_parquet(chunk_path)


class MemoryMonitor:
    """
    System memory monitoring and optimization
    """
    
    def __init__(self):
        self.process = psutil.Process()
        self._baseline_memory = self.get_usage_mb()
    
    def get_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def get_system_usage_percent(self) -> float:
        """Get system memory usage percentage"""
        return psutil.virtual_memory().percent
    
    def get_available_mb(self) -> float:
        """Get available system memory in MB"""
        return psutil.virtual_memory().available / (1024 * 1024)
    
    def get_delta_mb(self) -> float:
        """Get memory change since baseline"""
        return self.get_usage_mb() - self._baseline_memory
    
    def optimize_memory(self, aggressive: bool = False):
        """Trigger memory optimization"""
        if aggressive:
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
        else:
            gc.collect()
    
    def set_baseline(self):
        """Set new memory baseline"""
        self._baseline_memory = self.get_usage_mb()


class PerformanceOptimizer:
    """
    Main performance optimization engine that coordinates all optimization strategies
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._cache = SmartCache(config.cache_config) if config.enable_caching else None
        self._stream_processor = StreamProcessor(config.streaming_config) if config.enable_streaming else None
        self._memory_monitor = MemoryMonitor()
        self._metrics: List[PerformanceMetrics] = []
        self._lock = threading.Lock()
    
    def optimize_hypergraph_operation(
        self,
        operation_name: str,
        hypergraph: Hypergraph,
        operation: Callable,
        *args,
        use_cache: bool = True,
        **kwargs
    ) -> Any:
        """
        Optimize a hypergraph operation with caching, memory monitoring, and performance tracking
        """
        start_time = time.time()
        memory_before = self._memory_monitor.get_usage_mb()
        cache_key = self._generate_cache_key(operation_name, hypergraph, args, kwargs)
        
        # Try cache first
        cache_hits = 0
        cache_misses = 0
        
        if use_cache and self._cache:
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                cache_hits = 1
                
                # Record metrics
                end_time = time.time()
                memory_after = self._memory_monitor.get_usage_mb()
                
                metrics = PerformanceMetrics(
                    operation=operation_name,
                    start_time=start_time,
                    end_time=end_time,
                    memory_before_mb=memory_before,
                    memory_after_mb=memory_after,
                    cache_hits=cache_hits,
                    cache_misses=0
                )
                
                self._record_metrics(metrics)
                return cached_result
        
        cache_misses = 1
        
        # Execute operation
        if self.config.enable_parallel and hasattr(operation, '_supports_parallel'):
            result = self._execute_parallel(operation, hypergraph, *args, **kwargs)
        else:
            result = operation(hypergraph, *args, **kwargs)
        
        # Cache result if enabled
        if use_cache and self._cache and result is not None:
            self._cache.put(cache_key, result)
        
        # Memory optimization
        if self.config.enable_memory_optimization:
            if self._memory_monitor.get_delta_mb() > 100:  # 100MB threshold
                self._memory_monitor.optimize_memory()
        
        # Record metrics
        end_time = time.time()
        memory_after = self._memory_monitor.get_usage_mb()
        
        metrics = PerformanceMetrics(
            operation=operation_name,
            start_time=start_time,
            end_time=end_time,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            processed_rows=len(hypergraph.nodes) + len(hypergraph.edges)
        )
        
        self._record_metrics(metrics)
        return result
    
    def create_lazy_dataframe(self, data_source) -> LazyDataFrame:
        """Create lazy DataFrame for deferred computation"""
        return LazyDataFrame(data_source)
    
    def stream_process(
        self,
        data_source,
        processor: Callable,
        output_path: Optional[Path] = None
    ) -> Iterator[pl.DataFrame]:
        """Stream process large datasets"""
        if not self._stream_processor:
            raise RuntimeError("Streaming not enabled in configuration")
        
        return self._stream_processor.process_chunks(data_source, processor, output_path)
    
    def _generate_cache_key(
        self, 
        operation_name: str, 
        hypergraph: Hypergraph, 
        args: tuple, 
        kwargs: dict
    ) -> str:
        """Generate cache key for operation"""
        # Create a unique key based on operation and hypergraph state
        key_parts = [
            operation_name,
            str(hypergraph.num_nodes),
            str(hypergraph.num_edges),
            str(hypergraph.num_incidences)
        ]
        
        # Add args and kwargs
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        
        # Hash for consistent key
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _execute_parallel(self, operation: Callable, hypergraph: Hypergraph, *args, **kwargs):
        """Execute operation in parallel if supported"""
        # This would need operation-specific parallel implementations
        # For now, fall back to sequential
        return operation(hypergraph, *args, **kwargs)
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        with self._lock:
            self._metrics.append(metrics)
            
            # Keep only recent metrics (last 1000)
            if len(self._metrics) > 1000:
                self._metrics = self._metrics[-1000:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        with self._lock:
            if not self._metrics:
                return {"error": "No metrics available"}
            
            # Aggregate metrics by operation
            by_operation = defaultdict(list)
            for metric in self._metrics:
                by_operation[metric.operation].append(metric)
            
            report = {
                "summary": {
                    "total_operations": len(self._metrics),
                    "total_execution_time": sum(m.execution_time for m in self._metrics),
                    "avg_execution_time": sum(m.execution_time for m in self._metrics) / len(self._metrics),
                    "total_cache_hits": sum(m.cache_hits for m in self._metrics),
                    "total_cache_misses": sum(m.cache_misses for m in self._metrics),
                    "overall_cache_hit_ratio": sum(m.cache_hits for m in self._metrics) / 
                                             (sum(m.cache_hits for m in self._metrics) + sum(m.cache_misses for m in self._metrics))
                                             if any(m.cache_hits + m.cache_misses > 0 for m in self._metrics) else 0,
                    "total_memory_usage_mb": sum(m.memory_delta_mb for m in self._metrics),
                    "current_memory_mb": self._memory_monitor.get_usage_mb(),
                    "system_memory_percent": self._memory_monitor.get_system_usage_percent()
                },
                "by_operation": {}
            }
            
            for operation, metrics in by_operation.items():
                op_report = {
                    "count": len(metrics),
                    "total_time": sum(m.execution_time for m in metrics),
                    "avg_time": sum(m.execution_time for m in metrics) / len(metrics),
                    "min_time": min(m.execution_time for m in metrics),
                    "max_time": max(m.execution_time for m in metrics),
                    "cache_hits": sum(m.cache_hits for m in metrics),
                    "cache_misses": sum(m.cache_misses for m in metrics),
                    "cache_hit_ratio": sum(m.cache_hits for m in metrics) / 
                                     (sum(m.cache_hits for m in metrics) + sum(m.cache_misses for m in metrics))
                                     if any(m.cache_hits + m.cache_misses > 0 for m in metrics) else 0,
                    "total_memory_delta_mb": sum(m.memory_delta_mb for m in metrics),
                    "avg_memory_delta_mb": sum(m.memory_delta_mb for m in metrics) / len(metrics)
                }
                report["by_operation"][operation] = op_report
            
            # Add cache statistics if available
            if self._cache:
                report["cache_stats"] = self._cache.get_stats()
            
            return report
    
    def clear_cache(self):
        """Clear all cached data"""
        if self._cache:
            self._cache.clear()
    
    def optimize_memory(self, aggressive: bool = False):
        """Trigger memory optimization"""
        self._memory_monitor.optimize_memory(aggressive)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        return {
            "current_usage_mb": self._memory_monitor.get_usage_mb(),
            "delta_from_baseline_mb": self._memory_monitor.get_delta_mb(),
            "system_usage_percent": self._memory_monitor.get_system_usage_percent(),
            "available_mb": self._memory_monitor.get_available_mb()
        }


# Decorator for easy performance optimization
_global_optimizers = {}

def optimize_performance(
    operation_name: str,
    use_cache: bool = True,
    config: Optional[OptimizationConfig] = None
):
    """Decorator to automatically optimize function performance"""
    
    def decorator(func: Callable):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create optimizer
            optimizer_key = id(func)
            if optimizer_key not in _global_optimizers:
                _global_optimizers[optimizer_key] = PerformanceOptimizer(config or OptimizationConfig())
            
            optimizer = _global_optimizers[optimizer_key]
            
            # First arg should be hypergraph
            if args and isinstance(args[0], Hypergraph):
                hypergraph = args[0]
                return optimizer.optimize_hypergraph_operation(
                    operation_name, hypergraph, func, *args[1:], use_cache=use_cache, **kwargs
                )
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global optimizer instance for convenience
default_optimizer = PerformanceOptimizer(OptimizationConfig())