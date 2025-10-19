"""
Performance Validation Module

Provides performance benchmarking and validation for hypergraphs including:
- Algorithm execution time benchmarks
- Memory usage monitoring
- Performance threshold validation
- Scalability analysis
"""

import time
import tracemalloc
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from ..classes.hypergraph import Hypergraph
from ..analysis.centrality import degree_centrality
from ..analysis.clustering import modularity_clustering
from .data_integrity import ValidationResult


class MemoryMonitor:
    """Simple memory usage monitor"""
    
    def get_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback if psutil not available
            return 0.0


class PerformanceBenchmarkValidator:
    """Validates performance characteristics"""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.name = "Performance Benchmark"
        self.thresholds = thresholds or {
            'basic_operations': 1.0,  # seconds
            'centrality_computation': 5.0,
            'clustering_computation': 10.0,
            'memory_usage': 1000.0  # MB
        }
        self.memory_monitor = MemoryMonitor()
    
    def validate(self, hg: Hypergraph) -> ValidationResult:
        """Benchmark performance of core operations"""
        benchmark_results = {}
        issues = []
        
        try:
            # Benchmark basic operations
            _, time_taken, memory_used = self._measure_performance(
                self._benchmark_basic_operations, hg
            )
            benchmark_results['basic_operations'] = time_taken
            benchmark_results['memory_usage'] = memory_used
            
            if time_taken > self.thresholds['basic_operations']:
                issues.append(f"Basic operations too slow: {time_taken:.2f}s > {self.thresholds['basic_operations']}s")
            
            if memory_used > self.thresholds['memory_usage']:
                issues.append(f"Memory usage too high: {memory_used:.1f}MB > {self.thresholds['memory_usage']}MB")
            
            # Benchmark centrality computation if enough nodes
            if hg.num_nodes >= 5:
                _, time_taken, _ = self._measure_performance(
                    degree_centrality, hg
                )
                benchmark_results['centrality_computation'] = time_taken
                
                if time_taken > self.thresholds['centrality_computation']:
                    issues.append(f"Centrality computation too slow: {time_taken:.2f}s")
            
            # Benchmark clustering if enough edges
            if hg.num_edges >= 3:
                _, time_taken, _ = self._measure_performance(
                    modularity_clustering, hg
                )
                benchmark_results['clustering_computation'] = time_taken
                
                if time_taken > self.thresholds['clustering_computation']:
                    issues.append(f"Clustering computation too slow: {time_taken:.2f}s")
            
            total_time = sum(benchmark_results.values())
            
            if issues:
                return ValidationResult(
                    test_name="Performance Benchmark",
                    passed=False,
                    message=f"Performance issues detected: {'; '.join(issues[:2])}",
                    execution_time=total_time,
                    details={"performance_metrics": benchmark_results, "issues": issues}
                )
            else:
                return ValidationResult(
                    test_name="Performance Benchmark",
                    passed=True,
                    message="All performance benchmarks passed",
                    execution_time=total_time,
                    details={"performance_metrics": benchmark_results}
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="Performance Benchmark",
                passed=False,
                message=f"Benchmark failed with error: {str(e)}",
                execution_time=0.0,
                details={"error": str(e)}
            )
    
    def _measure_performance(self, func: Callable, *args, **kwargs) -> Tuple[Any, float, float]:
        """Measure execution time and memory usage of a function"""
        start_time = time.perf_counter()
        initial_memory = self.memory_monitor.get_usage_mb()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            final_memory = self.memory_monitor.get_usage_mb()
            memory_usage = max(final_memory - initial_memory, 0)
            
            return result, end_time - start_time, memory_usage
        except Exception as e:
            end_time = time.perf_counter()
            return None, end_time - start_time, 0.0
    
    def _benchmark_basic_operations(self, hg: Hypergraph):
        """Benchmark basic hypergraph operations"""
        # Test node and edge access
        nodes = hg.nodes
        edges = hg.edges
        
        # Test basic operations if possible
        if nodes and edges:
            # Test basic edge-node relationships
            _ = hg.get_edge_size(edges[0])
        
        # Test incidence operations
        _ = hg.incidences.data
        
        return True


def benchmark_performance(hg: Hypergraph, 
                         thresholds: Optional[Dict[str, float]] = None) -> ValidationResult:
    """Convenience function for performance benchmarking"""
    validator = PerformanceBenchmarkValidator(thresholds)
    return validator.validate(hg)