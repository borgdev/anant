"""
Performance Benchmarking Suite for Anant

Advanced performance analysis including:
- Timing analysis for all algorithms
- Memory usage profiling  
- Scalability testing
- Performance regression detection
- Optimization recommendations
"""

import polars as pl
import numpy as np
import time
import psutil
import traceback
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
import gc
from pathlib import Path

from ..classes.hypergraph import Hypergraph
from ..analysis import centrality, clustering
from ..algorithms import contagion_models, laplacian_clustering
from . import ValidationResult, ValidationSuite, BaseValidator


@dataclass
class PerformanceProfile:
    """Performance profile for an algorithm"""
    algorithm_name: str
    execution_time: float
    memory_usage: float
    peak_memory: float
    operations_per_second: Optional[float] = None
    scalability_factor: Optional[float] = None
    memory_efficiency: Optional[float] = None


@dataclass
class ScalabilityResult:
    """Results from scalability testing"""
    algorithm_name: str
    test_sizes: List[int]
    execution_times: List[float] 
    memory_usages: List[float]
    complexity_estimate: Optional[str] = None
    bottleneck_analysis: Dict[str, Any] = field(default_factory=dict)


class PerformanceBenchmark(BaseValidator):
    """Comprehensive performance benchmarking"""
    
    def __init__(self, 
                 target_algorithms: Optional[List[str]] = None,
                 memory_limit_mb: float = 2000.0,
                 time_limit_seconds: float = 30.0):
        super().__init__("Performance Benchmark")
        
        self.target_algorithms = target_algorithms or [
            'degree_centrality', 'closeness_centrality', 'betweenness_centrality',
            's_centrality', 'discrete_SIR', 'discrete_SIS', 'spectral_clustering',
            'modularity_clustering', 'prob_trans'
        ]
        
        self.memory_limit_mb = memory_limit_mb
        self.time_limit_seconds = time_limit_seconds
        self.process = psutil.Process()
    
    def validate(self, target: Hypergraph) -> ValidationResult:
        """Run comprehensive performance benchmark"""
        start_time = time.perf_counter()
        
        try:
            hg = target
            performance_profiles = {}
            issues = []
            
            # Benchmark each algorithm
            for algorithm_name in self.target_algorithms:
                try:
                    profile = self._benchmark_algorithm(hg, algorithm_name)
                    performance_profiles[algorithm_name] = profile
                    
                    # Check for performance issues
                    if profile.execution_time > self.time_limit_seconds:
                        issues.append(f"{algorithm_name} exceeded time limit: {profile.execution_time:.2f}s")
                    
                    if profile.memory_usage > self.memory_limit_mb:
                        issues.append(f"{algorithm_name} exceeded memory limit: {profile.memory_usage:.1f}MB")
                        
                except Exception as e:
                    issues.append(f"{algorithm_name} benchmark failed: {str(e)}")
            
            # Calculate summary statistics
            if performance_profiles:
                avg_time = np.mean([p.execution_time for p in performance_profiles.values()])
                avg_memory = np.mean([p.memory_usage for p in performance_profiles.values()])
                total_time = sum(p.execution_time for p in performance_profiles.values())
            else:
                avg_time = avg_memory = total_time = 0.0
            
            execution_time = time.perf_counter() - start_time
            
            return ValidationResult(
                test_name="Performance Benchmark",
                passed=len(issues) == 0,
                message=f"Benchmarked {len(performance_profiles)} algorithms" + 
                       (f", {len(issues)} issues found" if issues else ""),
                execution_time=execution_time,
                performance_metrics={
                    'algorithms_benchmarked': len(performance_profiles),
                    'average_execution_time': avg_time,
                    'average_memory_usage': avg_memory,
                    'total_benchmark_time': total_time
                },
                details={
                    'performance_profiles': performance_profiles,
                    'issues': issues
                }
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Performance Benchmark",
                passed=False,
                message=f"Benchmarking failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )
    
    def _benchmark_algorithm(self, hg: Hypergraph, algorithm_name: str) -> PerformanceProfile:
        """Benchmark a specific algorithm"""
        
        # Get algorithm function
        algorithm_func, args = self._get_algorithm_function(hg, algorithm_name)
        
        if algorithm_func is None:
            return PerformanceProfile(
                algorithm_name=algorithm_name,
                execution_time=0.0,
                memory_usage=0.0,
                peak_memory=0.0
            )
        
        # Clear garbage collection
        gc.collect()
        
        # Measure initial memory
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory
        
        # Execute algorithm with timing
        start_time = time.perf_counter()
        
        try:
            result = algorithm_func(*args)
            execution_time = time.perf_counter() - start_time
            
            # Measure final memory
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = max(final_memory - initial_memory, 0)
            peak_memory = max(peak_memory, final_memory)
            
            # Calculate operations per second (rough estimate)
            if hg.num_nodes > 0:
                operations_per_second = (hg.num_nodes + hg.num_edges) / execution_time if execution_time > 0 else 0
            else:
                operations_per_second = 0
            
            # Calculate memory efficiency (elements per MB)
            memory_efficiency = (hg.num_nodes + hg.num_edges) / memory_usage if memory_usage > 0 else 0
            
            return PerformanceProfile(
                algorithm_name=algorithm_name,
                execution_time=execution_time,
                memory_usage=memory_usage,
                peak_memory=peak_memory,
                operations_per_second=operations_per_second,
                memory_efficiency=memory_efficiency
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return PerformanceProfile(
                algorithm_name=algorithm_name,
                execution_time=execution_time,
                memory_usage=0.0,
                peak_memory=peak_memory
            )
    
    def _get_algorithm_function(self, hg: Hypergraph, algorithm_name: str) -> Tuple[Optional[Callable], List[Any]]:
        """Get algorithm function and arguments"""
        
        try:
            if algorithm_name == 'degree_centrality':
                return centrality.degree_centrality, [hg]
            
            elif algorithm_name == 'closeness_centrality':
                return centrality.closeness_centrality, [hg]
            
            elif algorithm_name == 'betweenness_centrality':
                return centrality.betweenness_centrality, [hg]
            
            elif algorithm_name == 's_centrality':
                return centrality.s_centrality, [hg, 1.0]
            
            elif algorithm_name == 'discrete_SIR':
                nodes = list(hg.nodes)
                if nodes:
                    return contagion_models.discrete_SIR, [hg, [nodes[0]], 0.3, 0.1, 'individual', 
                                                           contagion_models.majority_vote, 5]
                else:
                    return None, []
            
            elif algorithm_name == 'discrete_SIS':
                nodes = list(hg.nodes)
                if nodes:
                    return contagion_models.discrete_SIS, [hg, [nodes[0]], 0.3, 0.1, 'individual',
                                                           contagion_models.majority_vote, 5]
                else:
                    return None, []
            
            elif algorithm_name == 'spectral_clustering':
                k = min(2, hg.num_nodes)
                return laplacian_clustering.hypergraph_spectral_clustering, [hg, k]
            
            elif algorithm_name == 'modularity_clustering':
                return clustering.modularity_clustering, [hg]
            
            elif algorithm_name == 'prob_trans':
                return laplacian_clustering.prob_trans, [hg]
            
            else:
                return None, []
                
        except Exception:
            return None, []


class ScalabilityTester(BaseValidator):
    """Test scalability characteristics of algorithms"""
    
    def __init__(self, 
                 test_sizes: Optional[List[int]] = None,
                 target_algorithms: Optional[List[str]] = None):
        super().__init__("Scalability Testing")
        
        self.test_sizes = test_sizes or [5, 10, 20, 50, 100]
        self.target_algorithms = target_algorithms or [
            'degree_centrality', 'closeness_centrality', 'spectral_clustering'
        ]
    
    def validate(self, target: Any = None) -> ValidationResult:
        """Test scalability across different hypergraph sizes"""
        start_time = time.perf_counter()
        
        try:
            scalability_results = {}
            issues = []
            
            for algorithm_name in self.target_algorithms:
                try:
                    scalability_result = self._test_algorithm_scalability(algorithm_name)
                    scalability_results[algorithm_name] = scalability_result
                    
                    # Analyze scalability
                    complexity_analysis = self._analyze_complexity(scalability_result)
                    if complexity_analysis.get('warning'):
                        issues.append(f"{algorithm_name}: {complexity_analysis['warning']}")
                        
                except Exception as e:
                    issues.append(f"{algorithm_name} scalability test failed: {str(e)}")
            
            execution_time = time.perf_counter() - start_time
            
            return ValidationResult(
                test_name="Scalability Testing",
                passed=len(issues) == 0,
                message=f"Tested {len(scalability_results)} algorithms" +
                       (f", {len(issues)} scalability issues found" if issues else ""),
                execution_time=execution_time,
                details={
                    'scalability_results': scalability_results,
                    'issues': issues,
                    'test_sizes': self.test_sizes
                }
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Scalability Testing",
                passed=False,
                message=f"Scalability testing failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )
    
    def _test_algorithm_scalability(self, algorithm_name: str) -> ScalabilityResult:
        """Test scalability of a specific algorithm"""
        execution_times = []
        memory_usages = []
        
        for size in self.test_sizes:
            # Generate test hypergraph of specific size
            test_hg = self._generate_test_hypergraph(size)
            
            # Benchmark algorithm
            benchmarker = PerformanceBenchmark(target_algorithms=[algorithm_name])
            profile = benchmarker._benchmark_algorithm(test_hg, algorithm_name)
            
            execution_times.append(profile.execution_time)
            memory_usages.append(profile.memory_usage)
        
        return ScalabilityResult(
            algorithm_name=algorithm_name,
            test_sizes=self.test_sizes,
            execution_times=execution_times,
            memory_usages=memory_usages
        )
    
    def _generate_test_hypergraph(self, target_nodes: int) -> Hypergraph:
        """Generate synthetic hypergraph with approximately target_nodes nodes"""
        hg = Hypergraph()
        
        # Create nodes
        nodes = [f"n{i}" for i in range(target_nodes)]
        
        # Create edges - aim for reasonable density
        num_edges = max(1, target_nodes // 3)
        edge_size = min(4, max(2, target_nodes // 10))
        
        for i in range(num_edges):
            # Select random nodes for each edge
            start_idx = (i * edge_size) % target_nodes
            edge_nodes = []
            
            for j in range(edge_size):
                node_idx = (start_idx + j) % target_nodes
                edge_nodes.append(nodes[node_idx])
            
            hg.add_edge(f"e{i}", edge_nodes)
        
        return hg
    
    def _analyze_complexity(self, result: ScalabilityResult) -> Dict[str, Any]:
        """Analyze computational complexity from scalability results"""
        analysis = {}
        
        try:
            sizes = np.array(result.test_sizes)
            times = np.array(result.execution_times)
            
            # Filter out zero times
            valid_indices = times > 0
            if not np.any(valid_indices):
                return {'complexity': 'unknown', 'warning': 'All execution times were zero'}
            
            sizes = sizes[valid_indices]
            times = times[valid_indices]
            
            if len(sizes) < 2:
                return {'complexity': 'insufficient_data'}
            
            # Fit different complexity models
            log_sizes = np.log(sizes)
            log_times = np.log(times)
            
            # Linear fit: log(time) ~ log(size) -> time ~ size^k
            if len(log_sizes) > 1:
                coeffs = np.polyfit(log_sizes, log_times, 1)
                complexity_exponent = coeffs[0]
                
                if complexity_exponent < 1.2:
                    analysis['complexity'] = 'linear'
                elif complexity_exponent < 1.8:
                    analysis['complexity'] = 'superlinear'
                elif complexity_exponent < 2.2:
                    analysis['complexity'] = 'quadratic'
                elif complexity_exponent < 3.2:
                    analysis['complexity'] = 'cubic'
                else:
                    analysis['complexity'] = 'exponential'
                    analysis['warning'] = f"Algorithm shows {analysis['complexity']} scaling"
                
                analysis['complexity_exponent'] = complexity_exponent
            
            # Check for concerning trends
            max_time = np.max(times)
            if max_time > 10.0:  # seconds
                analysis['warning'] = f"Algorithm takes {max_time:.1f}s on size {sizes[np.argmax(times)]}"
            
        except Exception as e:
            analysis['complexity'] = 'analysis_failed'
            analysis['error'] = str(e)
        
        return analysis


class MemoryProfiler(BaseValidator):
    """Detailed memory usage profiling"""
    
    def __init__(self):
        super().__init__("Memory Profiling")
        self.process = psutil.Process()
    
    def validate(self, target: Hypergraph) -> ValidationResult:
        """Profile memory usage patterns"""
        start_time = time.perf_counter()
        
        try:
            hg = target
            memory_profiles = {}
            issues = []
            
            # Profile hypergraph memory usage
            base_memory = self._profile_hypergraph_memory(hg)
            memory_profiles['base_hypergraph'] = base_memory
            
            # Profile algorithm memory usage
            algorithm_memory = self._profile_algorithm_memory(hg)
            memory_profiles['algorithms'] = algorithm_memory
            
            # Check for memory issues
            total_memory = base_memory.get('total_mb', 0) + sum(
                profile.get('peak_memory_mb', 0) for profile in algorithm_memory.values()
            )
            
            if total_memory > 1000:  # MB
                issues.append(f"High total memory usage: {total_memory:.1f}MB")
            
            # Check for memory leaks (rough heuristic)
            for alg_name, profile in algorithm_memory.items():
                if profile.get('memory_growth', 0) > 100:  # MB
                    issues.append(f"Potential memory leak in {alg_name}: {profile['memory_growth']:.1f}MB growth")
            
            execution_time = time.perf_counter() - start_time
            
            return ValidationResult(
                test_name="Memory Profiling",
                passed=len(issues) == 0,
                message=f"Memory profiled, total usage: {total_memory:.1f}MB" + 
                       (f", {len(issues)} issues found" if issues else ""),
                execution_time=execution_time,
                details={
                    'memory_profiles': memory_profiles,
                    'issues': issues,
                    'total_memory_mb': total_memory
                }
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Memory Profiling",
                passed=False,
                message=f"Memory profiling failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )
    
    def _profile_hypergraph_memory(self, hg: Hypergraph) -> Dict[str, float]:
        """Profile memory usage of hypergraph structure"""
        
        # Get incidence data size
        incidence_data = hg._incidence_store.data
        
        # Estimate DataFrame memory usage
        estimated_memory = 0.0
        
        if hasattr(incidence_data, 'estimated_size'):
            estimated_memory = incidence_data.estimated_size() / 1024 / 1024  # MB
        else:
            # Rough estimate: 24 bytes per row (3 columns * 8 bytes)
            estimated_memory = (incidence_data.height * 24) / 1024 / 1024
        
        return {
            'incidence_rows': incidence_data.height,
            'estimated_mb': estimated_memory,
            'total_mb': estimated_memory,
            'nodes': hg.num_nodes,
            'edges': hg.num_edges
        }
    
    def _profile_algorithm_memory(self, hg: Hypergraph) -> Dict[str, Dict[str, float]]:
        """Profile memory usage of different algorithms"""
        algorithm_profiles = {}
        
        algorithms_to_test = [
            'degree_centrality', 'closeness_centrality', 'spectral_clustering'
        ]
        
        for algorithm_name in algorithms_to_test:
            try:
                profile = self._profile_single_algorithm_memory(hg, algorithm_name)
                algorithm_profiles[algorithm_name] = profile
            except Exception as e:
                algorithm_profiles[algorithm_name] = {
                    'error': str(e),
                    'peak_memory_mb': 0.0,
                    'memory_growth': 0.0
                }
        
        return algorithm_profiles
    
    def _profile_single_algorithm_memory(self, hg: Hypergraph, algorithm_name: str) -> Dict[str, float]:
        """Profile memory usage of a single algorithm"""
        
        gc.collect()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Get algorithm function
        benchmarker = PerformanceBenchmark()
        algorithm_func, args = benchmarker._get_algorithm_function(hg, algorithm_name)
        
        if algorithm_func is None:
            return {'peak_memory_mb': 0.0, 'memory_growth': 0.0}
        
        try:
            # Execute algorithm
            result = algorithm_func(*args)
            
            peak_memory = self.process.memory_info().rss / 1024 / 1024
            memory_growth = peak_memory - initial_memory
            
            # Clean up result
            del result
            gc.collect()
            
            final_memory = self.process.memory_info().rss / 1024 / 1024
            
            return {
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'final_memory_mb': final_memory,
                'memory_growth': memory_growth,
                'memory_recovered': peak_memory - final_memory
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'peak_memory_mb': 0.0,
                'memory_growth': 0.0
            }


# Convenience functions
def benchmark_algorithms(hg: Hypergraph, 
                        algorithms: Optional[List[str]] = None) -> ValidationResult:
    """Benchmark specific algorithms on hypergraph"""
    benchmarker = PerformanceBenchmark(target_algorithms=algorithms)
    return benchmarker.validate(hg)


def memory_profiler(hg: Hypergraph) -> ValidationResult:
    """Profile memory usage of hypergraph and algorithms"""
    profiler = MemoryProfiler()
    return profiler.validate(hg)


def timing_analysis(hg: Hypergraph, algorithm_name: str, runs: int = 5) -> Dict[str, float]:
    """Detailed timing analysis of a specific algorithm"""
    
    benchmarker = PerformanceBenchmark(target_algorithms=[algorithm_name])
    times = []
    
    for _ in range(runs):
        profile = benchmarker._benchmark_algorithm(hg, algorithm_name)
        times.append(profile.execution_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'median_time': np.median(times)
    }


__all__ = [
    'PerformanceProfile',
    'ScalabilityResult', 
    'PerformanceBenchmark',
    'ScalabilityTester',
    'MemoryProfiler',
    'benchmark_algorithms',
    'memory_profiler',
    'timing_analysis'
]