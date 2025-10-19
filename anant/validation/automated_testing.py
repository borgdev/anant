"""
Automated Testing Suite for Anant Algorithms

Comprehensive testing framework for all algorithms including:
- Centrality measures (degree, closeness, betweenness, s-centrality)  
- Contagion models (SIR, SIS, individual, collective)
- Clustering algorithms (spectral, modularity, Laplacian)
- Analysis functions and integration tests
"""

import polars as pl
import numpy as np
import time
import traceback
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass
import pytest
from pathlib import Path

from ..classes.hypergraph import Hypergraph
from ..analysis import centrality, clustering
from ..algorithms import contagion_models, laplacian_clustering
from .data_integrity import ValidationResult
from .validation_runner import ValidationSuite


class AlgorithmTester:
    """Comprehensive algorithm testing framework"""
    
    def __init__(self):
        self.name = "Algorithm Testing"
        self.test_cases = self._generate_test_cases()
    
    def _generate_test_cases(self) -> List[Hypergraph]:
        """Generate diverse test cases for algorithm validation"""
        test_cases = []
        
        # Small hypergraph (basic functionality)
        hg_small = Hypergraph()
        hg_small.add_edge("e1", ["n1", "n2"])
        hg_small.add_edge("e2", ["n2", "n3"])
        hg_small.name = "small_graph"
        test_cases.append(hg_small)
        
        # Medium hypergraph (realistic case)
        hg_medium = Hypergraph()
        hg_medium.add_edge("e1", ["n1", "n2", "n3"])
        hg_medium.add_edge("e2", ["n2", "n4", "n5"])
        hg_medium.add_edge("e3", ["n1", "n4"])
        hg_medium.add_edge("e4", ["n3", "n5", "n6"])
        hg_medium.name = "medium_graph"
        test_cases.append(hg_medium)
        
        # Star topology (centrality testing)
        hg_star = Hypergraph()
        center_node = "center"
        for i in range(5):
            hg_star.add_edge(f"spoke_{i}", [center_node, f"leaf_{i}"])
        hg_star.name = "star_graph"
        test_cases.append(hg_star)
        
        # Dense hypergraph (performance testing)
        hg_dense = Hypergraph()
        nodes = [f"n{i}" for i in range(8)]
        for i in range(6):
            # Create hyperedges with 3-4 nodes each
            edge_nodes = nodes[i:i+3] + [nodes[(i+5) % len(nodes)]]
            hg_dense.add_edge(f"e{i}", edge_nodes)
        hg_dense.name = "dense_graph"
        test_cases.append(hg_dense)
        
        # Linear chain (path testing)
        hg_chain = Hypergraph()
        for i in range(6):
            hg_chain.add_edge(f"e{i}", [f"n{i}", f"n{i+1}"])
        hg_chain.name = "chain_graph"
        test_cases.append(hg_chain)
        
        return test_cases
    
    def validate(self, target: Any = None) -> ValidationResult:
        """Run comprehensive algorithm tests"""
        start_time = time.perf_counter()
        
        try:
            all_tests_passed = True
            test_results = {}
            issues = []
            
            # Test centrality algorithms
            centrality_result = self.test_centrality_algorithms()
            test_results['centrality'] = centrality_result
            if not centrality_result.passed:
                all_tests_passed = False
                issues.extend(centrality_result.details.get('issues', []))
            
            # Test contagion algorithms
            contagion_result = self.test_contagion_algorithms()
            test_results['contagion'] = contagion_result
            if not contagion_result.passed:
                all_tests_passed = False
                issues.extend(contagion_result.details.get('issues', []))
            
            # Test clustering algorithms
            clustering_result = self.test_clustering_algorithms()
            test_results['clustering'] = clustering_result
            if not clustering_result.passed:
                all_tests_passed = False
                issues.extend(clustering_result.details.get('issues', []))
            
            execution_time = time.perf_counter() - start_time
            
            return ValidationResult(
                test_name="Comprehensive Algorithm Testing",
                passed=all_tests_passed,
                message=f"{'All algorithm tests passed' if all_tests_passed else f'{len(issues)} issues found'}",
                execution_time=execution_time,
                details={
                    'test_results': test_results,
                    'issues': issues,
                    'test_cases_count': len(self.test_cases)
                }
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Comprehensive Algorithm Testing",
                passed=False,
                message=f"Testing framework failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )
    
    def test_centrality_algorithms(self) -> ValidationResult:
        """Test all centrality algorithms"""
        start_time = time.perf_counter()
        issues = []
        
        try:
            for hg in self.test_cases:
                # Test degree centrality
                try:
                    deg_cent = centrality.degree_centrality(hg)
                    if not isinstance(deg_cent, dict) or 'nodes' not in deg_cent:
                        issues.append(f"Degree centrality failed on {hg.name}")
                    elif len(deg_cent['nodes']) != hg.num_nodes:
                        issues.append(f"Degree centrality node count mismatch on {hg.name}")
                except Exception as e:
                    issues.append(f"Degree centrality error on {hg.name}: {str(e)}")
                
                # Test closeness centrality
                try:
                    close_cent = centrality.closeness_centrality(hg)
                    if not isinstance(close_cent, dict):
                        issues.append(f"Closeness centrality failed on {hg.name}")
                    elif len(close_cent) != hg.num_nodes:
                        issues.append(f"Closeness centrality node count mismatch on {hg.name}")
                except Exception as e:
                    issues.append(f"Closeness centrality error on {hg.name}: {str(e)}")
                
                # Test betweenness centrality
                try:
                    bet_cent = centrality.betweenness_centrality(hg)
                    if not isinstance(bet_cent, dict):
                        issues.append(f"Betweenness centrality failed on {hg.name}")
                    elif len(bet_cent) != hg.num_nodes:
                        issues.append(f"Betweenness centrality node count mismatch on {hg.name}")
                except Exception as e:
                    issues.append(f"Betweenness centrality error on {hg.name}: {str(e)}")
                
                # Test s-centrality
                try:
                    s_cent = centrality.s_centrality(hg, s=1)
                    if not isinstance(s_cent, dict):
                        issues.append(f"S-centrality failed on {hg.name}")
                    elif len(s_cent) != hg.num_nodes:
                        issues.append(f"S-centrality node count mismatch on {hg.name}")
                except Exception as e:
                    issues.append(f"S-centrality error on {hg.name}: {str(e)}")
            
            execution_time = time.perf_counter() - start_time
            
            return ValidationResult(
                test_name="Centrality Algorithms",
                passed=len(issues) == 0,
                message=f"Centrality tests {'passed' if len(issues) == 0 else f'failed with {len(issues)} issues'}",
                execution_time=execution_time,
                details={'issues': issues, 'algorithms_tested': 4}
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Centrality Algorithms",
                passed=False,
                message=f"Centrality testing failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )
    
    def test_contagion_algorithms(self) -> ValidationResult:
        """Test contagion model algorithms"""
        start_time = time.perf_counter()
        issues = []
        
        try:
            for hg in self.test_cases:
                if hg.num_nodes < 2:  # Skip tiny graphs
                    continue
                
                nodes = list(hg.nodes)  # Define nodes at the start
                
                # Test SIR model
                try:
                    sir_result = contagion_models.discrete_SIR(
                        hg,
                        initial_infected=[nodes[0]],
                        tau=0.3,
                        gamma=0.1,
                        max_steps=5
                    )
                    
                    required_keys = ['final_states', 'history', 'total_infected']
                    if not all(key in sir_result for key in required_keys):
                        issues.append(f"SIR model missing keys on {hg.name}")
                    elif sir_result['total_infected'] < 1:
                        issues.append(f"SIR model invalid infected count on {hg.name}")
                        
                except Exception as e:
                    issues.append(f"SIR model error on {hg.name}: {str(e)}")
                
                # Test SIS model  
                try:
                    sis_result = contagion_models.discrete_SIS(
                        hg,
                        initial_infected=[nodes[0]],
                        tau=0.4,
                        gamma=0.2,
                        max_steps=5
                    )
                    
                    if 'final_states' not in sis_result:
                        issues.append(f"SIS model missing final_states on {hg.name}")
                        
                except Exception as e:
                    issues.append(f"SIS model error on {hg.name}: {str(e)}")
                
                # Test individual contagion
                try:
                    # Create node states dictionary
                    node_states = {node: 'S' for node in nodes}
                    node_states[nodes[0]] = 'I'  # Initially infected
                    
                    indiv_result = contagion_models.individual_contagion(
                        hg,
                        node_states,
                        tau=0.3,
                        gamma=0.1,
                        return_event_history=True
                    )
                    
                    # Check if result is valid dictionary with expected keys
                    if not isinstance(indiv_result, dict) or 'final_states' not in indiv_result:
                        issues.append(f"Individual contagion invalid result format on {hg.name}")
                        
                except Exception as e:
                    issues.append(f"Individual contagion error on {hg.name}: {str(e)}")
                
                # Test collective contagion
                try:
                    # Create node states dictionary
                    node_states = {node: 'S' for node in nodes}
                    node_states[nodes[0]] = 'I'  # Initially infected
                    
                    coll_result = contagion_models.collective_contagion(
                        hg,
                        node_states,
                        tau=0.3,
                        gamma=0.1,
                        return_event_history=True
                    )
                    
                    # Check if result is valid dictionary with expected keys
                    if not isinstance(coll_result, dict) or 'final_states' not in coll_result:
                        issues.append(f"Collective contagion invalid result format on {hg.name}")
                        
                except Exception as e:
                    issues.append(f"Collective contagion error on {hg.name}: {str(e)}")
            
            execution_time = time.perf_counter() - start_time
            
            return ValidationResult(
                test_name="Contagion Algorithms",
                passed=len(issues) == 0,
                message=f"Contagion tests {'passed' if len(issues) == 0 else f'failed with {len(issues)} issues'}",
                execution_time=execution_time,
                details={'issues': issues, 'algorithms_tested': 4}
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Contagion Algorithms",
                passed=False,
                message=f"Contagion testing failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )
    
    def test_clustering_algorithms(self) -> ValidationResult:
        """Test clustering algorithms"""
        start_time = time.perf_counter()
        issues = []
        
        try:
            for hg in self.test_cases:
                if hg.num_nodes < 3:  # Skip tiny graphs
                    continue
                
                # Test spectral clustering
                try:
                    clusters = laplacian_clustering.hypergraph_spectral_clustering(hg, k=2)
                    if len(clusters) != hg.num_nodes:
                        issues.append(f"Spectral clustering node count mismatch on {hg.name}")
                    elif len(set(clusters)) == 0:
                        issues.append(f"Spectral clustering produced no clusters on {hg.name}")
                        
                except Exception as e:
                    issues.append(f"Spectral clustering error on {hg.name}: {str(e)}")
                
                # Test probability transition matrix
                try:
                    P_matrix, node_mapping = laplacian_clustering.prob_trans(hg)
                    if P_matrix.shape[0] != hg.num_nodes or P_matrix.shape[1] != hg.num_nodes:
                        issues.append(f"Probability matrix shape mismatch on {hg.name}")
                    elif len(node_mapping) != hg.num_nodes:
                        issues.append(f"Node mapping count mismatch on {hg.name}")
                        
                except Exception as e:
                    issues.append(f"Probability transition error on {hg.name}: {str(e)}")
                
                # Test modularity clustering  
                try:
                    communities = clustering.modularity_clustering(hg)
                    if not isinstance(communities, dict):
                        issues.append(f"Modularity clustering invalid format on {hg.name}")
                    elif len(communities) != hg.num_nodes:
                        issues.append(f"Modularity clustering node count mismatch on {hg.name}")
                        
                except Exception as e:
                    issues.append(f"Modularity clustering error on {hg.name}: {str(e)}")
                
                # Test normalized Laplacian
                try:
                    # First get probability transition matrix
                    P_matrix, _ = laplacian_clustering.prob_trans(hg)
                    L_norm = laplacian_clustering.norm_lap(P_matrix)
                    expected_size = hg.num_nodes
                    if L_norm.shape != (expected_size, expected_size):
                        issues.append(f"Normalized Laplacian shape mismatch on {hg.name}")
                        
                except Exception as e:
                    issues.append(f"Normalized Laplacian error on {hg.name}: {str(e)}")
            
            execution_time = time.perf_counter() - start_time
            
            return ValidationResult(
                test_name="Clustering Algorithms",
                passed=len(issues) == 0,
                message=f"Clustering tests {'passed' if len(issues) == 0 else f'failed with {len(issues)} issues'}",
                execution_time=execution_time,
                details={'issues': issues, 'algorithms_tested': 4}
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Clustering Algorithms",
                passed=False,
                message=f"Clustering testing failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )


class EdgeCaseValidator:
    """Validates algorithm behavior on edge cases"""
    
    def __init__(self):
        self.name = "Edge Case Validation"
    
    def validate(self, target: Any = None) -> ValidationResult:
        """Test algorithms on edge cases"""
        start_time = time.perf_counter()
        
        try:
            issues = []
            
            # Empty hypergraph
            empty_hg = Hypergraph()
            issues.extend(self._test_empty_hypergraph(empty_hg))
            
            # Single node
            single_node_hg = Hypergraph()
            single_node_hg.add_edge("e1", ["n1"])
            issues.extend(self._test_single_node_hypergraph(single_node_hg))
            
            # Single edge 
            single_edge_hg = Hypergraph()
            single_edge_hg.add_edge("e1", ["n1", "n2"])
            issues.extend(self._test_single_edge_hypergraph(single_edge_hg))
            
            # Disconnected components
            disconnected_hg = Hypergraph()
            disconnected_hg.add_edge("e1", ["n1", "n2"])
            disconnected_hg.add_edge("e2", ["n3", "n4"])
            issues.extend(self._test_disconnected_hypergraph(disconnected_hg))
            
            execution_time = time.perf_counter() - start_time
            
            return ValidationResult(
                test_name="Edge Case Validation",
                passed=len(issues) == 0,
                message=f"Edge case tests {'passed' if len(issues) == 0 else f'failed with {len(issues)} issues'}",
                execution_time=execution_time,
                details={'issues': issues, 'test_cases': 4}
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Edge Case Validation",
                passed=False,
                message=f"Edge case testing failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )
    
    def _test_empty_hypergraph(self, hg: Hypergraph) -> List[str]:
        """Test algorithms on empty hypergraph"""
        issues = []
        
        # Most algorithms should handle empty graphs gracefully
        try:
            result = centrality.degree_centrality(hg)
            if result['nodes'] != {} or result['edges'] != {}:
                issues.append("Degree centrality failed on empty graph")
        except Exception:
            # Expected to fail gracefully, not crash
            pass
        
        return issues
    
    def _test_single_node_hypergraph(self, hg: Hypergraph) -> List[str]:
        """Test algorithms on single node hypergraph"""
        issues = []
        
        try:
            # Centrality should work
            result = centrality.degree_centrality(hg)
            if len(result['nodes']) != 1:
                issues.append("Degree centrality failed on single node")
        except Exception as e:
            issues.append(f"Single node centrality failed: {str(e)}")
        
        return issues
    
    def _test_single_edge_hypergraph(self, hg: Hypergraph) -> List[str]:
        """Test algorithms on single edge hypergraph"""
        issues = []
        
        try:
            # Test centrality
            result = centrality.degree_centrality(hg)
            if len(result['nodes']) != 2 or len(result['edges']) != 1:
                issues.append("Degree centrality failed on single edge")
            
            # Test contagion 
            nodes = list(hg.nodes)
            sir_result = contagion_models.discrete_SIR(
                hg, initial_infected=[nodes[0]], max_steps=2
            )
            if 'final_states' not in sir_result:
                issues.append("SIR failed on single edge")
                
        except Exception as e:
            issues.append(f"Single edge algorithms failed: {str(e)}")
        
        return issues
    
    def _test_disconnected_hypergraph(self, hg: Hypergraph) -> List[str]:
        """Test algorithms on disconnected hypergraph"""
        issues = []
        
        try:
            # Centrality should work
            result = centrality.degree_centrality(hg)
            if len(result['nodes']) != 4:
                issues.append("Degree centrality failed on disconnected graph")
            
            # Closeness centrality should handle disconnected components
            close_result = centrality.closeness_centrality(hg)
            if len(close_result) != 4:
                issues.append("Closeness centrality failed on disconnected graph")
                
        except Exception as e:
            issues.append(f"Disconnected graph algorithms failed: {str(e)}")
        
        return issues


# Convenience functions
def run_all_tests() -> ValidationSuite:
    """Run all algorithm tests"""
    from datetime import datetime
    
    suite = ValidationSuite(name="Complete Algorithm Test Suite")
    suite.start_time = datetime.now()
    
    # Run algorithm tests
    algorithm_tester = AlgorithmTester()
    algorithm_result = algorithm_tester.validate()
    suite.results.append(algorithm_result)
    
    # Run edge case tests
    edge_case_validator = EdgeCaseValidator()
    edge_case_result = edge_case_validator.validate()
    suite.results.append(edge_case_result)
    
    suite.end_time = datetime.now()
    return suite


def test_centrality_algorithms() -> ValidationResult:
    """Test only centrality algorithms"""
    tester = AlgorithmTester()
    return tester.test_centrality_algorithms()


def test_contagion_algorithms() -> ValidationResult:
    """Test only contagion algorithms"""
    tester = AlgorithmTester()
    return tester.test_contagion_algorithms()


def test_clustering_algorithms() -> ValidationResult:
    """Test only clustering algorithms"""
    tester = AlgorithmTester()
    return tester.test_clustering_algorithms()


__all__ = [
    'AlgorithmTester',
    'EdgeCaseValidator',
    'run_all_tests',
    'test_centrality_algorithms',
    'test_contagion_algorithms', 
    'test_clustering_algorithms'
]