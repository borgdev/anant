#!/usr/bin/env python3
"""
ANANT Performance Benchmark Suite
===============================

Comprehensive benchmarks to measure and track performance improvements.
"""

import time
import psutil
import gc
from typing import Dict, List, Callable, Tuple, Any
from dataclasses import dataclass
import sys


@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    name: str
    duration_ms: float
    memory_delta_mb: float
    success: bool
    error: str = ""
    

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
    def run_benchmark(self, name: str, func: Callable, *args, **kwargs) -> BenchmarkResult:
        """Run a single benchmark and record results"""
        gc.collect()  # Clean slate
        
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = ""
        except Exception as e:
            result = None
            success = False
            error = str(e)[:200]
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        duration_ms = (end_time - start_time) * 1000
        memory_delta = end_memory - start_memory
        
        benchmark_result = BenchmarkResult(
            name=name,
            duration_ms=duration_ms,
            memory_delta_mb=memory_delta,
            success=success,
            error=error
        )
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def print_results(self):
        """Print formatted benchmark results"""
        print("\n" + "="*80)
        print("🚀 ANANT PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        
        categories = {
            'Import Performance': [],
            'Core Operations': [], 
            'Algorithm Performance': [],
            'Memory Usage': [],
            'Polars Operations': []
        }
        
        # Categorize results
        for result in self.results:
            if 'Import' in result.name:
                categories['Import Performance'].append(result)
            elif 'Algorithm' in result.name or 'PageRank' in result.name or 'Community' in result.name:
                categories['Algorithm Performance'].append(result)
            elif 'Polars' in result.name or 'Fast' in result.name:
                categories['Polars Operations'].append(result)
            elif 'Memory' in result.name:
                categories['Memory Usage'].append(result)
            else:
                categories['Core Operations'].append(result)
        
        # Print results by category
        for category, results in categories.items():
            if not results:
                continue
                
            print(f"\n📊 {category}")
            print("-" * (len(category) + 4))
            
            for result in results:
                status = "✅" if result.success else "❌"
                
                # Performance indicators
                perf_indicator = ""
                if result.success:
                    if 'Import' in result.name:
                        if result.duration_ms < 50:
                            perf_indicator = " 🎉 EXCELLENT"
                        elif result.duration_ms < 200:
                            perf_indicator = " ⚡ GOOD" 
                        elif result.duration_ms < 1000:
                            perf_indicator = " 🟡 OK"
                        else:
                            perf_indicator = " 🔴 SLOW"
                    else:
                        if result.duration_ms < 10:
                            perf_indicator = " 🎉 EXCELLENT"
                        elif result.duration_ms < 50:
                            perf_indicator = " ⚡ GOOD"
                        elif result.duration_ms < 200:
                            perf_indicator = " 🟡 OK"
                        else:
                            perf_indicator = " 🔴 SLOW"
                
                print(f"{status} {result.name:30}: {result.duration_ms:8.2f}ms "
                      f"({result.memory_delta_mb:+6.1f}MB){perf_indicator}")
                
                if not result.success:
                    print(f"     ❌ Error: {result.error}")
        
        # Summary statistics
        successful_results = [r for r in self.results if r.success]
        if successful_results:
            total_time = sum(r.duration_ms for r in successful_results)
            total_memory = sum(r.memory_delta_mb for r in successful_results)
            
            print(f"\n📈 SUMMARY STATISTICS")
            print("-" * 25)
            print(f"Total benchmarks: {len(self.results)}")
            print(f"Successful: {len(successful_results)}")
            print(f"Failed: {len(self.results) - len(successful_results)}")
            print(f"Total execution time: {total_time:.1f}ms")
            print(f"Total memory delta: {total_memory:+.1f}MB")
            print(f"Average time per operation: {total_time/len(successful_results):.1f}ms")


def run_comprehensive_benchmarks():
    """Run comprehensive performance benchmarks"""
    
    benchmark = PerformanceBenchmark()
    
    print("🔬 STARTING COMPREHENSIVE ANANT BENCHMARKS")
    print("=" * 55)
    
    # ========================================
    # 1. IMPORT BENCHMARKS
    # ========================================
    print("\n📦 Testing Import Performance...")
    
    # Core imports
    benchmark.run_benchmark("Lazy Utils Import", lambda: __import__('anant.utils.lazy_imports'))
    benchmark.run_benchmark("Polars Utils Import", lambda: __import__('anant.utils.polars_performance'))
    benchmark.run_benchmark("Hypergraph Import", lambda: __import__('anant.classes.hypergraph'))
    benchmark.run_benchmark("KG Core Import", lambda: __import__('anant.kg.core'))
    benchmark.run_benchmark("Hierarchical Import", lambda: __import__('anant.kg.hierarchical'))
    
    # ========================================
    # 2. CORE OPERATION BENCHMARKS  
    # ========================================
    print("\n🔧 Testing Core Operations...")
    
    def create_hypergraph():
        from anant.classes.hypergraph import Hypergraph
        hg = Hypergraph()
        return hg
    
    def create_large_hypergraph():
        from anant.classes.hypergraph import Hypergraph
        hg = Hypergraph()
        for i in range(1000):
            hg.add_edge(f'E{i}', [f'N{j}' for j in range(i, i+4)])
        return hg
    
    def test_kg_operations():
        from anant.kg.core import KnowledgeGraph
        kg = KnowledgeGraph()
        for i in range(100):
            kg.add_node(f'node_{i}')
            if i > 0:
                kg.add_edge([f'node_{i-1}', f'node_{i}'])
        return kg
    
    benchmark.run_benchmark("Hypergraph Creation", create_hypergraph)
    benchmark.run_benchmark("Large Hypergraph (1000 edges)", create_large_hypergraph)
    benchmark.run_benchmark("KG Operations (100 nodes)", test_kg_operations)
    
    # Test nodes/edges access
    hg = create_hypergraph()
    if hg:
        for i in range(50):
            hg.add_edge(f'E{i}', [f'N{j}' for j in range(i, i+3)])
    
    benchmark.run_benchmark("Nodes Access", lambda: len(hg.nodes) if hg else 0)
    benchmark.run_benchmark("Edges Access", lambda: len(hg.edges) if hg and hasattr(hg.edges, '__len__') else 0)
    
    # ========================================
    # 3. POLARS PERFORMANCE BENCHMARKS
    # ========================================
    print("\n⚡ Testing Polars Performance...")
    
    def test_polars_node_degrees():
        from anant.utils.polars_performance import PolarsPerfOps
        from anant.classes.hypergraph import Hypergraph
        
        hg = Hypergraph()
        for i in range(500):
            hg.add_edge(f'E{i}', [f'N{j}' for j in range(i, i+5)])
        
        perf_ops = PolarsPerfOps(hg)
        degrees = perf_ops.fast_node_degrees()
        return len(degrees)
    
    def test_polars_statistics():
        from anant.utils.polars_performance import PolarsPerfOps  
        from anant.classes.hypergraph import Hypergraph
        
        hg = Hypergraph()
        for i in range(300):
            hg.add_edge(f'E{i}', [f'N{j}' for j in range(i, i+4)])
        
        perf_ops = PolarsPerfOps(hg)
        stats = perf_ops.fast_hypergraph_statistics()
        return len(stats)
    
    benchmark.run_benchmark("Fast Node Degrees (500 edges)", test_polars_node_degrees)
    benchmark.run_benchmark("Fast Statistics (300 edges)", test_polars_statistics)
    
    # ========================================
    # 4. ALGORITHM BENCHMARKS
    # ========================================
    print("\n🧮 Testing Algorithm Performance...")
    
    def test_dual_construction():
        from anant.classes.hypergraph import Hypergraph
        hg = Hypergraph()
        for i in range(100):
            hg.add_edge(f'E{i}', [f'N{j}' for j in range(i, i+3)])
        
        dual = hg.dual()
        return dual is not None
    
    benchmark.run_benchmark("Dual Construction (100 edges)", test_dual_construction)
    
    # ========================================
    # 5. MEMORY USAGE BENCHMARKS
    # ========================================  
    print("\n💾 Testing Memory Usage...")
    
    def memory_stress_test():
        from anant.classes.hypergraph import Hypergraph
        graphs = []
        for i in range(10):
            hg = Hypergraph()
            for j in range(100):
                hg.add_edge(f'E{j}', [f'N{k}' for k in range(j, j+3)])
            graphs.append(hg)
        return len(graphs)
    
    benchmark.run_benchmark("Memory Stress Test (10 graphs)", memory_stress_test)
    
    # Print all results
    benchmark.print_results()
    
    # ========================================
    # 6. PERFORMANCE TARGETS CHECK
    # ========================================
    print(f"\n🎯 PERFORMANCE TARGETS CHECK")
    print("-" * 35)
    
    targets = {
        "Import times < 100ms": [r for r in benchmark.results if 'Import' in r.name and r.success and r.duration_ms < 100],
        "Core operations < 50ms": [r for r in benchmark.results if 'Creation' in r.name or 'Access' in r.name and r.success and r.duration_ms < 50],
        "Polars operations < 100ms": [r for r in benchmark.results if 'Fast' in r.name and r.success and r.duration_ms < 100],
        "Memory usage < 50MB per op": [r for r in benchmark.results if r.success and r.memory_delta_mb < 50]
    }
    
    for target_name, matching_results in targets.items():
        total_ops = len([r for r in benchmark.results if any(keyword in r.name for keyword in target_name.split())])
        if total_ops > 0:
            success_rate = (len(matching_results) / total_ops) * 100
            status = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 60 else "❌"
            print(f"{status} {target_name}: {len(matching_results)}/{total_ops} ({success_rate:.1f}%)")
    
    return benchmark


if __name__ == "__main__":
    benchmark_results = run_comprehensive_benchmarks()
    
    print(f"\n🏁 BENCHMARK COMPLETE")
    print(f"Check results above for detailed performance analysis.")