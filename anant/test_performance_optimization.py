#!/usr/bin/env python3
"""
Test script for Performance Optimization Engine

This script tests the new Performance Optimization capabilities including:
- Smart caching with multiple eviction policies
- Lazy evaluation frameworks  
- Streaming data processing
- Memory optimization
- Performance monitoring and metrics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anant'))

from anant.optimization import (
    PerformanceOptimizer, OptimizationConfig, CacheConfig, StreamingConfig,
    CachePolicy, OptimizationLevel, StreamingMode, SmartCache, LazyDataFrame,
    MemoryMonitor, optimize_performance, default_optimizer
)
from anant.classes import Hypergraph
from anant.factory import SetSystemFactory
import polars as pl
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, cast


def create_test_hypergraph(size: str = "small"):
    """Create test hypergraphs of different sizes"""
    if size == "small":
        edge_data = {
            "edge1": ["node1", "node2", "node3"],
            "edge2": ["node2", "node4"],
            "edge3": ["node1", "node4", "node5"],
            "edge4": ["node3", "node5", "node6"]
        }
    elif size == "medium":
        edge_data = {}
        for i in range(50):
            nodes = [f"node_{j}" for j in range(i, i+5)]
            edge_data[f"edge_{i}"] = nodes
    elif size == "large":
        edge_data = {}
        for i in range(200):
            nodes = [f"node_{j}" for j in range(i, i+8)]
            edge_data[f"edge_{i}"] = nodes
    else:
        raise ValueError(f"Unknown size: {size}")
    
    # Create hypergraph using SetSystemFactory
    incidence_df = SetSystemFactory.from_dict_of_iterables(cast(Dict[str, Iterable], edge_data))
    hg = Hypergraph(incidence_df)
    
    # Add some properties for realism
    for i, node in enumerate(hg.nodes):
        hg._node_properties.set_property(node, "weight", float(i % 10))
        hg._node_properties.set_property(node, "type", f"type_{i % 3}")
    
    for i, edge in enumerate(hg.edges):
        hg._edge_properties.set_property(edge, "strength", float(i % 5))
        hg._edge_properties.set_property(edge, "category", f"cat_{i % 4}")
    
    return hg


def test_smart_cache():
    """Test smart caching with different policies"""
    print("\n=== Testing Smart Cache ===")
    
    policies = [CachePolicy.LRU, CachePolicy.LFU, CachePolicy.FIFO, CachePolicy.SIZE_BASED]
    
    for policy in policies:
        print(f"\nTesting {policy.value} cache policy...")
        
        config = CacheConfig(
            policy=policy,
            max_size_mb=10,  # Small cache for testing
            max_entries=5,
            ttl_seconds=60
        )
        
        cache = SmartCache(config)
        
        # Test basic operations
        test_data = {
            "key1": "value1",
            "key2": [1, 2, 3, 4, 5],
            "key3": {"nested": "data"},
            "key4": pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            "key5": "large_value" * 1000
        }
        
        # Put data
        for key, value in test_data.items():
            result = cache.put(key, value)
            print(f"  Put {key}: {'success' if result else 'failed'}")
        
        # Get data and test hits
        for key in test_data.keys():
            value = cache.get(key)
            status = "hit" if value is not None else "miss"
            print(f"  Get {key}: {status}")
        
        # Add more data to trigger eviction
        for i in range(10):
            cache.put(f"extra_key_{i}", f"extra_value_{i}")
        
        # Check final stats
        stats = cache.get_stats()
        print(f"  Final stats: {stats['cache_size']} entries, "
              f"{stats['hit_ratio']:.2f} hit ratio, "
              f"{stats['evictions']} evictions")


def test_lazy_evaluation():
    """Test lazy evaluation framework"""
    print("\n=== Testing Lazy Evaluation ===")
    
    # Create test DataFrame
    df = pl.DataFrame({
        "id": range(1000),
        "value": [i * 2 for i in range(1000)],
        "category": [f"cat_{i % 5}" for i in range(1000)]
    })
    
    print(f"Original DataFrame: {df.shape}")
    
    # Create lazy wrapper
    lazy_df = LazyDataFrame(df)
    
    # Chain operations without execution
    print("Building lazy operation chain...")
    result_lazy = (lazy_df
                   .filter(pl.col("value") > 100)
                   .select(["id", "value", "category"])
                   .with_columns(pl.col("value").alias("doubled_value")))
    
    print("Operations queued, not yet executed")
    
    # Execute all operations at once
    start_time = time.time()
    result = result_lazy.collect()
    execution_time = time.time() - start_time
    
    print(f"Lazy evaluation completed in {execution_time:.4f}s")
    print(f"Result shape: {result.shape}")
    print(f"First 5 rows:\n{result.head()}")


def test_memory_monitoring():
    """Test memory monitoring capabilities"""
    print("\n=== Testing Memory Monitoring ===")
    
    monitor = MemoryMonitor()
    monitor.set_baseline()
    
    print(f"Baseline memory: {monitor.get_usage_mb():.2f} MB")
    print(f"System memory usage: {monitor.get_system_usage_percent():.1f}%")
    print(f"Available memory: {monitor.get_available_mb():.2f} MB")
    
    # Allocate some memory
    print("\nAllocating memory...")
    big_data = []
    for i in range(100):
        big_data.append(pl.DataFrame({
            "data": range(1000),
            "values": [j * i for j in range(1000)]
        }))
    
    print(f"Memory after allocation: {monitor.get_usage_mb():.2f} MB")
    print(f"Delta from baseline: {monitor.get_delta_mb():.2f} MB")
    
    # Optimize memory
    print("\nOptimizing memory...")
    monitor.optimize_memory(aggressive=True)
    
    print(f"Memory after optimization: {monitor.get_usage_mb():.2f} MB")
    print(f"Delta from baseline: {monitor.get_delta_mb():.2f} MB")


def test_performance_optimizer():
    """Test the main performance optimizer"""
    print("\n=== Testing Performance Optimizer ===")
    
    # Create configuration
    config = OptimizationConfig(
        level=OptimizationLevel.AGGRESSIVE,
        enable_caching=True,
        enable_memory_optimization=True,
        cache_config=CacheConfig(
            policy=CachePolicy.LRU,
            max_size_mb=50,
            max_entries=100
        )
    )
    
    optimizer = PerformanceOptimizer(config)
    hg = create_test_hypergraph("medium")
    
    print(f"Test hypergraph: {hg}")
    
    # Define a test operation
    def analyze_nodes(hypergraph):
        """Test operation that analyzes nodes"""
        nodes_df = hypergraph.to_dataframe("nodes")
        return {
            "node_count": len(nodes_df),
            "avg_degree": nodes_df.get_column("degree").mean() if "degree" in nodes_df.columns else 0,
            "total_weight": nodes_df.get_column("weight").sum() if "weight" in nodes_df.columns else 0
        }
    
    # Run operation multiple times to test caching
    print("\nRunning optimized operations...")
    
    for i in range(5):
        result = optimizer.optimize_hypergraph_operation(
            "analyze_nodes", hg, analyze_nodes
        )
        print(f"  Run {i+1}: {result}")
    
    # Get performance report
    report = optimizer.get_performance_report()
    print(f"\nPerformance Report:")
    print(f"  Total operations: {report['summary']['total_operations']}")
    print(f"  Total execution time: {report['summary']['total_execution_time']:.4f}s")
    print(f"  Average execution time: {report['summary']['avg_execution_time']:.4f}s")
    print(f"  Cache hit ratio: {report['summary']['overall_cache_hit_ratio']:.2f}")
    print(f"  Current memory: {report['summary']['current_memory_mb']:.2f} MB")
    
    if 'cache_stats' in report:
        cache_stats = report['cache_stats']
        print(f"  Cache size: {cache_stats['cache_size']} entries")
        print(f"  Cache hit ratio: {cache_stats['hit_ratio']:.2f}")


@optimize_performance("test_decorated_operation", use_cache=True)
def decorated_test_operation(hypergraph: Hypergraph) -> dict:
    """Test operation with performance optimization decorator"""
    edges_df = hypergraph.to_dataframe("edges")
    return {
        "edge_count": len(edges_df),
        "total_size": edges_df.get_column("size").sum() if "size" in edges_df.columns else 0
    }


def test_optimization_decorator():
    """Test the optimization decorator"""
    print("\n=== Testing Optimization Decorator ===")
    
    hg = create_test_hypergraph("small")
    print(f"Test hypergraph: {hg}")
    
    # Run decorated operation multiple times
    print("\nRunning decorated operations...")
    
    for i in range(3):
        start_time = time.time()
        result = decorated_test_operation(hg)
        end_time = time.time()
        print(f"  Run {i+1}: {result} (took {end_time - start_time:.4f}s)")


def test_streaming_simulation():
    """Test streaming capabilities with simulated data"""
    print("\n=== Testing Streaming Simulation ===")
    
    # Create streaming config
    config = StreamingConfig(
        mode=StreamingMode.CHUNK_BASED,
        chunk_size=100,
        max_memory_mb=50,
        enable_parallel=False  # Simplified for testing
    )
    
    # Create large test DataFrame
    print("Creating large test dataset...")
    large_df = pl.DataFrame({
        "id": range(1000),
        "value": [i * 3 for i in range(1000)],
        "category": [f"cat_{i % 10}" for i in range(1000)],
        "weight": [float(i % 7) for i in range(1000)]
    })
    
    print(f"Dataset size: {large_df.shape}")
    
    # Create stream processor
    from anant.optimization import StreamProcessor
    processor = StreamProcessor(config)
    
    # Define processing function
    def process_chunk(chunk_df):
        """Process each chunk - filter and aggregate"""
        return chunk_df.filter(pl.col("value") > 150).select(["id", "value", "weight"])
    
    # Process in chunks
    print("\nProcessing in chunks...")
    total_processed = 0
    chunk_count = 0
    
    for chunk in processor.process_chunks(large_df, process_chunk):
        chunk_count += 1
        total_processed += len(chunk)
        print(f"  Processed chunk {chunk_count}: {len(chunk)} rows")
        
        if chunk_count >= 5:  # Limit for demo
            break
    
    print(f"\nTotal processed: {total_processed} rows in {chunk_count} chunks")


def main():
    """Run all performance optimization tests"""
    print("Testing Performance Optimization Engine for Anant")
    print("=" * 60)
    
    try:
        test_smart_cache()
        test_lazy_evaluation()
        test_memory_monitoring()
        test_performance_optimizer()
        test_optimization_decorator()
        test_streaming_simulation()
        
        print("\n" + "=" * 60)
        print("All Performance Optimization tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())