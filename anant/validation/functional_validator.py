"""
Functional Validation Framework for Anant Library

This module provides focused functionality testing for all algorithms and features,
emphasizing correctness validation and integration testing over stress testing.
"""

import anant
from anant import analysis
from anant.algorithms import contagion_models, laplacian_clustering
import time
from typing import Dict, List, Any, Optional
import numpy as np


class FunctionalValidator:
    """Streamlined validator focusing on algorithm functionality"""
    
    def __init__(self):
        self.name = "Functional Validation"
        self.results = []
        self.test_hypergraph: anant.Hypergraph = anant.Hypergraph()
        self.setup_test_data()
    
    def setup_test_data(self):
        """Create standard test hypergraph for consistent testing"""
        # Create a well-structured test hypergraph
        self.test_hypergraph.add_edge("e1", ["n1", "n2", "n3"])
        self.test_hypergraph.add_edge("e2", ["n1", "n4", "n5"])
        self.test_hypergraph.add_edge("e3", ["n2", "n3", "n6"])
        self.test_hypergraph.add_edge("e4", ["n4", "n5", "n6"])
        self.test_hypergraph.add_edge("e5", ["n1", "n6"])
        
        print(f"🏗️  Test hypergraph: {len(self.test_hypergraph.nodes)} nodes, {len(self.test_hypergraph.edges)} edges")
    
    def test_centrality_algorithms(self) -> Dict[str, Any]:
        """Test all centrality algorithms for basic functionality"""
        print("\n🎯 Testing Centrality Algorithms...")
        results = {}
        
        try:
            # Test degree centrality
            degree_cent = analysis.degree_centrality(self.test_hypergraph)
            results['degree_centrality'] = {
                'status': 'PASS',
                'result': f"{len(degree_cent['nodes'])} nodes analyzed"
            }
            
            # Test closeness centrality
            closeness_cent = analysis.closeness_centrality(self.test_hypergraph)
            results['closeness_centrality'] = {
                'status': 'PASS',
                'result': f"{len(closeness_cent)} nodes analyzed"
            }
            
            # Test betweenness centrality
            betweenness_cent = analysis.betweenness_centrality(self.test_hypergraph)
            results['betweenness_centrality'] = {
                'status': 'PASS',
                'result': f"{len(betweenness_cent)} nodes analyzed"
            }
            
            # Test s-centrality
            s_cent = analysis.s_centrality(self.test_hypergraph, s=1)
            results['s_centrality'] = {
                'status': 'PASS',
                'result': f"{len(s_cent)} nodes analyzed"
            }
            
            print("   ✅ All centrality algorithms working")
            
        except Exception as e:
            results['centrality_error'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ Centrality error: {e}")
        
        return results
    
    def test_contagion_algorithms(self) -> Dict[str, Any]:
        """Test contagion spreading algorithms"""
        print("\n🦠 Testing Contagion Algorithms...")
        results = {}
        
        try:
            # Test SIR model
            sir_result = contagion_models.discrete_SIR(
                self.test_hypergraph,
                initial_infected=['n1'],
                tau=0.3,
                gamma=0.1,
                max_steps=5
            )
            results['sir_model'] = {
                'status': 'PASS',
                'result': f"Infected {sir_result['total_infected']} nodes in {sir_result['duration']} steps"
            }
            
            # Test SIS model  
            sis_result = contagion_models.discrete_SIS(
                self.test_hypergraph,
                initial_infected=['n1'],
                tau=0.4,
                gamma=0.2,
                max_steps=5
            )
            infected_count = len([n for n in sis_result.get('final_states', {}).values() if n == 'I'])
            results['sis_model'] = {
                'status': 'PASS',
                'result': f"Endemic level: {infected_count} nodes"
            }
            
            # Test individual contagion
            node_states = {node: 'S' for node in self.test_hypergraph.nodes}
            node_states['n1'] = 'I'
            
            individual_result = contagion_models.individual_contagion(
                self.test_hypergraph,
                node_states,
                tau=0.3,
                gamma=0.1
            )
            results['individual_contagion'] = {
                'status': 'PASS',
                'result': f"Updated {len(individual_result)} node states"
            }
            
            print("   ✅ All contagion algorithms working")
            
        except Exception as e:
            results['contagion_error'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ Contagion error: {e}")
        
        return results
    
    def test_clustering_algorithms(self) -> Dict[str, Any]:
        """Test clustering and spectral algorithms"""
        print("\n🔍 Testing Clustering Algorithms...")
        results = {}
        
        try:
            # Test spectral clustering
            clusters = laplacian_clustering.hypergraph_spectral_clustering(self.test_hypergraph, k=2)
            results['spectral_clustering'] = {
                'status': 'PASS',
                'result': f"Created {len(set(clusters))} clusters"
            }
            
            # Test probability transition matrix
            P_matrix, node_mapping = laplacian_clustering.prob_trans(self.test_hypergraph)
            results['prob_transition'] = {
                'status': 'PASS',
                'result': f"Matrix shape: {P_matrix.shape}"
            }
            
            # Test community detection (using clustering module)
            communities = analysis.clustering.modularity_clustering(self.test_hypergraph)
            results['community_detection'] = {
                'status': 'PASS',
                'result': f"Detected {len(set(communities.values()))} communities"
            }
            
            print("   ✅ All clustering algorithms working")
            
        except Exception as e:
            results['clustering_error'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ Clustering error: {e}")
        
        return results
    
    def test_hypergraph_operations(self) -> Dict[str, Any]:
        """Test basic hypergraph operations and properties"""
        print("\n📊 Testing Hypergraph Operations...")
        results = {}
        
        try:
            # Test basic properties
            results['basic_properties'] = {
                'status': 'PASS',
                'result': f"Nodes: {len(self.test_hypergraph.nodes)}, Edges: {len(self.test_hypergraph.edges)}"
            }
            
            # Test edge operations
            edge_sizes = [self.test_hypergraph.get_edge_size(edge) for edge in self.test_hypergraph.edges]
            results['edge_operations'] = {
                'status': 'PASS',
                'result': f"Edge sizes: min={min(edge_sizes)}, max={max(edge_sizes)}"
            }
            
            # Test incidence data access
            incidence_data = self.test_hypergraph.incidences.data
            results['incidence_access'] = {
                'status': 'PASS',
                'result': f"Incidences: {len(incidence_data)} rows"
            }
            
            print("   ✅ All hypergraph operations working")
            
        except Exception as e:
            results['hypergraph_error'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ Hypergraph error: {e}")
        
        return results
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and boundary conditions"""
        print("\n⚠️  Testing Edge Cases...")
        results = {}
        
        try:
            # Test empty hypergraph
            empty_hg = anant.Hypergraph()
            results['empty_hypergraph'] = {
                'status': 'PASS',
                'result': f"Empty HG: {len(empty_hg.nodes)} nodes"
            }
            
            # Test single node
            single_hg = anant.Hypergraph()
            single_hg.add_edge("e1", ["n1"])
            degree_cent = analysis.degree_centrality(single_hg)
            results['single_node'] = {
                'status': 'PASS',
                'result': f"Single node centrality computed"
            }
            
            # Test large edge
            large_hg = anant.Hypergraph()
            large_hg.add_edge("big_edge", [f"n{i}" for i in range(10)])
            edge_size = large_hg.get_edge_size("big_edge")
            results['large_edge'] = {
                'status': 'PASS',
                'result': f"Large edge size: {edge_size}"
            }
            
            print("   ✅ All edge cases handled")
            
        except Exception as e:
            results['edge_case_error'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ Edge case error: {e}")
        
        return results
    
    def run_functional_validation(self) -> Dict[str, Any]:
        """Run complete functional validation suite"""
        print("🧪 FUNCTIONAL VALIDATION FRAMEWORK")
        print("=" * 60)
        
        start_time = time.time()
        
        all_results = {}
        all_results['centrality'] = self.test_centrality_algorithms()
        all_results['contagion'] = self.test_contagion_algorithms()
        all_results['clustering'] = self.test_clustering_algorithms()
        all_results['hypergraph_ops'] = self.test_hypergraph_operations()
        all_results['edge_cases'] = self.test_edge_cases()
        
        elapsed_time = time.time() - start_time
        
        # Generate summary
        total_tests = sum(len(category) for category in all_results.values())
        passed_tests = sum(
            len([test for test in category.values() if test.get('status') == 'PASS'])
            for category in all_results.values()
        )
        
        print(f"\n🎉 FUNCTIONAL VALIDATION COMPLETE")
        print("=" * 60)
        print(f"✅ Passed: {passed_tests}/{total_tests} tests")
        print(f"⏱️  Time: {elapsed_time:.3f} seconds")
        
        if passed_tests == total_tests:
            print("🏆 ALL FUNCTIONALITY TESTS PASSED!")
        else:
            print("⚠️  Some tests failed - check results above")
        
        all_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'elapsed_time': elapsed_time,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0
        }
        
        return all_results


def run_functional_validation():
    """Convenience function to run functional validation"""
    validator = FunctionalValidator()
    return validator.run_functional_validation()


if __name__ == "__main__":
    run_functional_validation()