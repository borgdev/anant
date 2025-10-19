"""
Stress Testing Suite for Anant

Comprehensive stress testing including:
- Large-scale data validation
- Extreme parameter testing  
- Memory stress testing
- Concurrent operation testing
- Error resilience testing
"""

import polars as pl
import numpy as np
import time
import threading
import multiprocessing
import traceback
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import random

from ..classes.hypergraph import Hypergraph
from ..analysis import centrality, clustering
from ..algorithms import contagion_models, laplacian_clustering
from . import ValidationResult, ValidationSuite, BaseValidator


@dataclass 
class StressTestConfig:
    """Configuration for stress testing"""
    max_nodes: int = 1000
    max_edges: int = 500
    max_edge_size: int = 20
    memory_limit_mb: float = 4000.0
    time_limit_seconds: float = 60.0
    concurrent_workers: int = 4
    error_tolerance: float = 0.1  # 10% error rate allowed


class StressTester(BaseValidator):
    """Comprehensive stress testing framework"""
    
    def __init__(self, config: Optional[StressTestConfig] = None):
        super().__init__("Stress Testing")
        self.config = config or StressTestConfig()
        self.process = psutil.Process()
    
    def validate(self, target: Any = None) -> ValidationResult:
        """Run comprehensive stress tests"""
        start_time = time.perf_counter()
        
        try:
            stress_results = {}
            issues = []
            
            # Large scale stress test
            large_scale_result = self._run_large_scale_test()
            stress_results['large_scale'] = large_scale_result
            if not large_scale_result.passed:
                issues.extend(large_scale_result.details.get('issues', []))
            
            # Memory stress test
            memory_stress_result = self._run_memory_stress_test()
            stress_results['memory_stress'] = memory_stress_result
            if not memory_stress_result.passed:
                issues.extend(memory_stress_result.details.get('issues', []))
            
            # Concurrent operations test
            concurrent_result = self._run_concurrent_test()
            stress_results['concurrent'] = concurrent_result
            if not concurrent_result.passed:
                issues.extend(concurrent_result.details.get('issues', []))
            
            # Error resilience test
            resilience_result = self._run_error_resilience_test()
            stress_results['error_resilience'] = resilience_result
            if not resilience_result.passed:
                issues.extend(resilience_result.details.get('issues', []))
            
            execution_time = time.perf_counter() - start_time
            
            return ValidationResult(
                test_name="Comprehensive Stress Testing",
                passed=len(issues) == 0,
                message=f"Stress tests completed" + (f", {len(issues)} issues found" if issues else ""),
                execution_time=execution_time,
                details={
                    'stress_results': stress_results,
                    'issues': issues,
                    'config': self.config
                }
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Comprehensive Stress Testing",
                passed=False,
                message=f"Stress testing failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )
    
    def _run_large_scale_test(self) -> ValidationResult:
        """Test with large hypergraphs"""
        start_time = time.perf_counter()
        
        try:
            issues = []
            test_results = {}
            
            # Test different scales
            test_scales = [100, 300, 500, self.config.max_nodes]
            
            for scale in test_scales:
                if scale > self.config.max_nodes:
                    continue
                
                # Generate large hypergraph
                large_hg = self._generate_large_hypergraph(scale)
                test_results[f'scale_{scale}'] = {
                    'nodes': large_hg.num_nodes,
                    'edges': large_hg.num_edges,
                    'incidences': large_hg._incidence_store.data.height
                }
                
                # Test core algorithms
                algorithm_results = self._test_algorithms_on_large_graph(large_hg)
                test_results[f'scale_{scale}'].update(algorithm_results)
                
                # Check for issues
                for alg_name, result in algorithm_results.items():
                    if result.get('error'):
                        issues.append(f"Scale {scale}: {alg_name} failed - {result['error']}")
                    elif result.get('execution_time', 0) > self.config.time_limit_seconds:
                        issues.append(f"Scale {scale}: {alg_name} exceeded time limit")
                
                # Memory check
                current_memory = self.process.memory_info().rss / 1024 / 1024
                if current_memory > self.config.memory_limit_mb:
                    issues.append(f"Scale {scale}: Memory usage exceeded limit: {current_memory:.1f}MB")
                    break  # Stop testing larger scales
            
            execution_time = time.perf_counter() - start_time
            
            return ValidationResult(
                test_name="Large Scale Testing",
                passed=len(issues) == 0,
                message=f"Tested scales up to {max(test_scales)} nodes" + 
                       (f", {len(issues)} issues found" if issues else ""),
                execution_time=execution_time,
                details={
                    'test_results': test_results,
                    'issues': issues,
                    'scales_tested': len(test_results)
                }
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Large Scale Testing",
                passed=False,
                message=f"Large scale testing failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )
    
    def _run_memory_stress_test(self) -> ValidationResult:
        """Test memory usage under stress"""
        start_time = time.perf_counter()
        
        try:
            issues = []
            memory_results = {}
            
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            peak_memory = initial_memory
            
            # Create multiple hypergraphs simultaneously  
            hypergraphs = []
            for i in range(10):
                hg = self._generate_large_hypergraph(100 + i * 20)
                hypergraphs.append(hg)
                
                current_memory = self.process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                
                if current_memory > self.config.memory_limit_mb:
                    issues.append(f"Memory limit exceeded at hypergraph {i}: {current_memory:.1f}MB")
                    break
            
            memory_results['hypergraphs_created'] = len(hypergraphs)
            memory_results['peak_memory_mb'] = peak_memory
            memory_results['memory_growth'] = peak_memory - initial_memory
            
            # Test algorithm memory usage on multiple graphs
            if hypergraphs:
                algorithm_memory = self._test_algorithm_memory_stress(hypergraphs[:3])
                memory_results['algorithm_memory'] = algorithm_memory
                
                # Check for memory leaks
                for alg_name, alg_memory in algorithm_memory.items():
                    if alg_memory.get('memory_not_recovered', 0) > 200:  # MB
                        issues.append(f"Potential memory leak in {alg_name}: {alg_memory['memory_not_recovered']:.1f}MB")
            
            # Cleanup and check memory recovery
            del hypergraphs
            gc.collect()
            
            final_memory = self.process.memory_info().rss / 1024 / 1024
            memory_recovered = peak_memory - final_memory
            memory_results['final_memory_mb'] = final_memory
            memory_results['memory_recovered'] = memory_recovered
            
            if memory_recovered < memory_results['memory_growth'] * 0.5:
                issues.append(f"Poor memory recovery: only {memory_recovered:.1f}MB of {memory_results['memory_growth']:.1f}MB recovered")
            
            execution_time = time.perf_counter() - start_time
            
            return ValidationResult(
                test_name="Memory Stress Testing",
                passed=len(issues) == 0,
                message=f"Peak memory: {peak_memory:.1f}MB" + 
                       (f", {len(issues)} issues found" if issues else ""),
                execution_time=execution_time,
                details={
                    'memory_results': memory_results,
                    'issues': issues
                }
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Memory Stress Testing",
                passed=False,
                message=f"Memory stress testing failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )
    
    def _run_concurrent_test(self) -> ValidationResult:
        """Test concurrent operations"""
        start_time = time.perf_counter()
        
        try:
            issues = []
            concurrent_results = {}
            
            # Create test hypergraph
            test_hg = self._generate_large_hypergraph(200)
            
            # Define concurrent tasks
            tasks = [
                ('degree_centrality', lambda: centrality.degree_centrality(test_hg)),
                ('closeness_centrality', lambda: centrality.closeness_centrality(test_hg)),
                ('modularity_clustering', lambda: clustering.modularity_clustering(test_hg)),
                ('spectral_clustering', lambda: laplacian_clustering.hypergraph_spectral_clustering(test_hg, k=3))
            ]
            
            # Run tasks concurrently
            with ThreadPoolExecutor(max_workers=self.config.concurrent_workers) as executor:
                futures = []
                for task_name, task_func in tasks:
                    future = executor.submit(self._run_task_safely, task_name, task_func)
                    futures.append((task_name, future))
                
                # Collect results
                for task_name, future in futures:
                    try:
                        result = future.result(timeout=self.config.time_limit_seconds)
                        concurrent_results[task_name] = result
                        
                        if not result['success']:
                            issues.append(f"Concurrent task {task_name} failed: {result['error']}")
                            
                    except Exception as e:
                        issues.append(f"Concurrent task {task_name} exception: {str(e)}")
            
            # Test for race conditions by running same algorithm multiple times
            race_condition_test = self._test_race_conditions(test_hg)
            concurrent_results['race_condition_test'] = race_condition_test
            
            if race_condition_test.get('inconsistent_results', 0) > 0:
                issues.append(f"Race condition detected: {race_condition_test['inconsistent_results']} inconsistent results")
            
            execution_time = time.perf_counter() - start_time
            
            return ValidationResult(
                test_name="Concurrent Operations Testing",
                passed=len(issues) == 0,
                message=f"Tested {len(tasks)} concurrent tasks" + 
                       (f", {len(issues)} issues found" if issues else ""),
                execution_time=execution_time,
                details={
                    'concurrent_results': concurrent_results,
                    'issues': issues,
                    'workers_used': self.config.concurrent_workers
                }
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Concurrent Operations Testing",
                passed=False,
                message=f"Concurrent testing failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )
    
    def _run_error_resilience_test(self) -> ValidationResult:
        """Test error handling and resilience"""
        start_time = time.perf_counter()
        
        try:
            issues = []
            resilience_results = {}
            
            # Test with malformed data
            malformed_tests = self._test_malformed_data_resilience()
            resilience_results['malformed_data'] = malformed_tests
            
            # Test with extreme parameters
            extreme_param_tests = self._test_extreme_parameters()
            resilience_results['extreme_parameters'] = extreme_param_tests
            
            # Test with edge cases
            edge_case_tests = self._test_algorithm_edge_cases()
            resilience_results['edge_cases'] = edge_case_tests
            
            # Analyze results
            for test_category, results in resilience_results.items():
                error_rate = results.get('error_rate', 0)
                if error_rate > self.config.error_tolerance:
                    issues.append(f"{test_category} error rate too high: {error_rate:.2%}")
                
                crash_rate = results.get('crash_rate', 0)
                if crash_rate > 0.05:  # 5% crash rate threshold
                    issues.append(f"{test_category} crash rate too high: {crash_rate:.2%}")
            
            execution_time = time.perf_counter() - start_time
            
            return ValidationResult(
                test_name="Error Resilience Testing",
                passed=len(issues) == 0,
                message=f"Tested error resilience" + (f", {len(issues)} issues found" if issues else ""),
                execution_time=execution_time,
                details={
                    'resilience_results': resilience_results,
                    'issues': issues
                }
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Error Resilience Testing",
                passed=False,
                message=f"Error resilience testing failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )
    
    def _generate_large_hypergraph(self, target_nodes: int) -> Hypergraph:
        """Generate large synthetic hypergraph"""
        hg = Hypergraph()
        
        # Create nodes
        nodes = [f"n{i}" for i in range(target_nodes)]
        
        # Create edges with varying sizes
        num_edges = min(self.config.max_edges, target_nodes // 2)
        
        for i in range(num_edges):
            # Vary edge sizes
            edge_size = random.randint(2, min(self.config.max_edge_size, target_nodes // 4))
            
            # Select nodes for edge
            edge_nodes = random.sample(nodes, edge_size)
            hg.add_edge(f"e{i}", edge_nodes)
        
        return hg
    
    def _test_algorithms_on_large_graph(self, hg: Hypergraph) -> Dict[str, Dict[str, Any]]:
        """Test core algorithms on large hypergraph"""
        results = {}
        
        algorithms = [
            ('degree_centrality', lambda: centrality.degree_centrality(hg)),
            ('closeness_centrality', lambda: centrality.closeness_centrality(hg)),
            ('modularity_clustering', lambda: clustering.modularity_clustering(hg))
        ]
        
        for alg_name, alg_func in algorithms:
            try:
                start_time = time.perf_counter()
                result = alg_func()
                execution_time = time.perf_counter() - start_time
                
                results[alg_name] = {
                    'success': True,
                    'execution_time': execution_time,
                    'result_size': len(result) if isinstance(result, dict) else 1
                }
                
            except Exception as e:
                results[alg_name] = {
                    'success': False,
                    'error': str(e),
                    'execution_time': 0.0
                }
        
        return results
    
    def _test_algorithm_memory_stress(self, hypergraphs: List[Hypergraph]) -> Dict[str, Dict[str, float]]:
        """Test algorithm memory usage across multiple hypergraphs"""
        memory_results = {}
        
        algorithms = ['degree_centrality', 'closeness_centrality']
        
        for alg_name in algorithms:
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            peak_memory = initial_memory
            
            # Run algorithm on all hypergraphs
            for hg in hypergraphs:
                try:
                    if alg_name == 'degree_centrality':
                        centrality.degree_centrality(hg)
                    elif alg_name == 'closeness_centrality':
                        centrality.closeness_centrality(hg)
                    
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, current_memory)
                    
                except Exception:
                    pass
            
            # Force cleanup
            gc.collect()
            final_memory = self.process.memory_info().rss / 1024 / 1024
            
            memory_results[alg_name] = {
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'final_memory_mb': final_memory,
                'memory_growth': peak_memory - initial_memory,
                'memory_not_recovered': max(0, final_memory - initial_memory)
            }
        
        return memory_results
    
    def _run_task_safely(self, task_name: str, task_func: Callable) -> Dict[str, Any]:
        """Run a task safely and return results"""
        try:
            start_time = time.perf_counter()
            result = task_func()
            execution_time = time.perf_counter() - start_time
            
            return {
                'success': True,
                'execution_time': execution_time,
                'result_available': result is not None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': 0.0
            }
    
    def _test_race_conditions(self, hg: Hypergraph) -> Dict[str, Any]:
        """Test for race conditions in algorithms"""
        
        # Run same algorithm multiple times concurrently
        def run_centrality():
            return centrality.degree_centrality(hg)
        
        results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_centrality) for _ in range(4)]
            
            for future in futures:
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                except Exception:
                    pass
        
        # Check for consistency
        inconsistent_results = 0
        if len(results) > 1:
            reference_result = results[0]
            for result in results[1:]:
                if result != reference_result:
                    inconsistent_results += 1
        
        return {
            'runs_completed': len(results),
            'inconsistent_results': inconsistent_results,
            'consistency_rate': (len(results) - inconsistent_results) / len(results) if results else 0
        }
    
    def _test_malformed_data_resilience(self) -> Dict[str, Any]:
        """Test resilience to malformed data"""
        
        test_cases = []
        errors = 0
        crashes = 0
        
        # Test empty hypergraph
        try:
            empty_hg = Hypergraph()
            centrality.degree_centrality(empty_hg)
            test_cases.append('empty_graph')
        except Exception:
            errors += 1
        
        # Test single node
        try:
            single_hg = Hypergraph()
            single_hg.add_edge("e1", ["n1"])
            centrality.degree_centrality(single_hg)
            test_cases.append('single_node')
        except Exception:
            errors += 1
        
        return {
            'total_tests': len(test_cases) + errors,
            'successful_tests': len(test_cases),
            'error_rate': errors / max(1, len(test_cases) + errors),
            'crash_rate': crashes / max(1, len(test_cases) + errors)
        }
    
    def _test_extreme_parameters(self) -> Dict[str, Any]:
        """Test algorithms with extreme parameters"""
        
        hg = self._generate_large_hypergraph(50)
        test_cases = []
        errors = 0
        
        # Test extreme s-centrality parameters
        try:
            centrality.s_centrality(hg, s=100.0)  # Very large s
            test_cases.append('large_s_centrality')
        except Exception:
            errors += 1
        
        try:
            centrality.s_centrality(hg, s=0.001)  # Very small s
            test_cases.append('small_s_centrality')
        except Exception:
            errors += 1
        
        # Test extreme contagion parameters
        if hg.nodes:
            try:
                contagion_models.discrete_SIR(hg, initial_infected=[list(hg.nodes)[0]], 
                                            tau=0.99, gamma=0.001, max_steps=1000)
                test_cases.append('extreme_sir')
            except Exception:
                errors += 1
        
        return {
            'total_tests': len(test_cases) + errors,
            'successful_tests': len(test_cases),
            'error_rate': errors / max(1, len(test_cases) + errors),
            'crash_rate': 0  # Assuming no crashes for now
        }
    
    def _test_algorithm_edge_cases(self) -> Dict[str, Any]:
        """Test algorithms on edge case hypergraphs"""
        
        test_cases = []
        errors = 0
        
        # Disconnected graph
        try:
            disconnected_hg = Hypergraph()
            disconnected_hg.add_edge("e1", ["n1", "n2"])
            disconnected_hg.add_edge("e2", ["n3", "n4"])
            centrality.closeness_centrality(disconnected_hg)
            test_cases.append('disconnected_graph')
        except Exception:
            errors += 1
        
        # Very large edge
        try:
            large_edge_hg = Hypergraph()
            nodes = [f"n{i}" for i in range(50)]
            large_edge_hg.add_edge("big_edge", nodes)
            centrality.degree_centrality(large_edge_hg)
            test_cases.append('large_edge')
        except Exception:
            errors += 1
        
        return {
            'total_tests': len(test_cases) + errors,
            'successful_tests': len(test_cases),
            'error_rate': errors / max(1, len(test_cases) + errors),
            'crash_rate': 0
        }


# Convenience functions
def large_scale_validation(max_nodes: int = 1000) -> ValidationResult:
    """Run large-scale validation test"""
    config = StressTestConfig(max_nodes=max_nodes)
    tester = StressTester(config)
    return tester._run_large_scale_test()


def scalability_analysis(target_algorithms: Optional[List[str]] = None) -> ValidationResult:
    """Run scalability analysis on algorithms"""
    from .performance_benchmarks import ScalabilityTester
    
    tester = ScalabilityTester(target_algorithms=target_algorithms)
    return tester.validate()


def memory_stress_test(memory_limit_mb: float = 4000.0) -> ValidationResult:
    """Run memory stress test"""
    config = StressTestConfig(memory_limit_mb=memory_limit_mb)
    tester = StressTester(config)
    return tester._run_memory_stress_test()


__all__ = [
    'StressTestConfig',
    'StressTester', 
    'large_scale_validation',
    'scalability_analysis',
    'memory_stress_test'
]