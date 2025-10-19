"""
Test Critical Fixes for ANANT Library
=====================================

Comprehensive test suite for all the critical fixes implemented
based on the FIBO analysis experience.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import polars as pl
import numpy as np
from typing import Dict, List, Any

from anant.classes.hypergraph import Hypergraph
from anant.classes.incidence_store import IncidenceStore
from anant.algorithms.clustering import hypergraph_clustering, community_detection, spectral_clustering
from anant.algorithms.sampling import SmartSampler, auto_scale_algorithm, get_sampling_recommendations
from anant.algorithms.centrality_enhanced import (
    enhanced_centrality_analysis, betweenness_centrality, closeness_centrality,
    eigenvector_centrality, harmonic_centrality
)


class TestIncidenceStoreFixes:
    """Test fixes for IncidenceStore interface"""
    
    def test_edge_column_property(self):
        """Test that edge_column property is available"""
        store = IncidenceStore()
        assert hasattr(store, 'edge_column')
        assert store.edge_column == 'edge_id'
    
    def test_node_column_property(self):
        """Test that node_column property is available"""
        store = IncidenceStore()
        assert hasattr(store, 'node_column')
        assert store.node_column == 'node_id'
    
    def test_weight_column_property(self):
        """Test that weight_column property is available"""
        store = IncidenceStore()
        assert hasattr(store, 'weight_column')
        assert store.weight_column == 'weight'
    
    def test_property_access_with_data(self):
        """Test property access with actual data"""
        data = pl.DataFrame({
            'edge_id': ['e1', 'e1', 'e2'],
            'node_id': ['n1', 'n2', 'n1'],
            'weight': [1.0, 0.8, 1.0]
        })
        store = IncidenceStore(data)
        
        # Should be able to access column names
        assert store.edge_column in store.data.columns
        assert store.node_column in store.data.columns
        assert store.weight_column in store.data.columns


class TestClusteringFixes:
    """Test fixes for clustering algorithms"""
    
    def setup_method(self):
        """Create test hypergraph"""
        edge_dict = {
            'e1': ['n1', 'n2', 'n3'],
            'e2': ['n2', 'n3', 'n4'],
            'e3': ['n1', 'n4'],
            'e4': ['n5', 'n6'],
            'e5': ['n6', 'n7']
        }
        self.hg = Hypergraph.from_dict(edge_dict)
    
    def test_clustering_no_crash(self):
        """Test that clustering algorithms don't crash with fixed interface"""
        try:
            # These should not crash anymore
            result = hypergraph_clustering(self.hg, algorithm='modularity')
            assert isinstance(result, dict)
            assert len(result) == len(self.hg.nodes)
            
            result = hypergraph_clustering(self.hg, algorithm='spectral', n_clusters=2)
            assert isinstance(result, dict)
            assert len(result) == len(self.hg.nodes)
            
            result = hypergraph_clustering(self.hg, algorithm='hierarchical', n_clusters=2)
            assert isinstance(result, dict)
            assert len(result) == len(self.hg.nodes)
            
        except AttributeError as e:
            pytest.fail(f"Clustering algorithm crashed with AttributeError: {e}")
    
    def test_community_detection(self):
        """Test community detection works"""
        result = community_detection(self.hg)
        assert isinstance(result, dict)
        assert all(node in result for node in self.hg.nodes)
        assert all(isinstance(cluster_id, int) for cluster_id in result.values())
    
    def test_clustering_with_weights(self):
        """Test clustering with weighted edges"""
        # Add some weights
        for i, edge in enumerate(self.hg.edges):
            for node in self.hg.incidences.get_edge_nodes(edge):
                self.hg.incidences.set_weight(edge, node, 1.0 + i * 0.1)
        
        result = hypergraph_clustering(self.hg, algorithm='spectral', 
                                     weight_column='weight', n_clusters=2)
        assert isinstance(result, dict)
        assert len(result) == len(self.hg.nodes)


class TestSmartSampling:
    """Test intelligent sampling capabilities"""
    
    def setup_method(self):
        """Create larger test hypergraph for sampling"""
        # Create a larger graph for meaningful sampling tests
        edge_dict = {}
        for i in range(100):  # 100 edges
            edge_dict[f'e{i}'] = [f'n{j}' for j in range(i, min(i+5, 200))]  # Up to 200 nodes
        
        self.large_hg = Hypergraph.from_dict(edge_dict)
        self.sampler = SmartSampler(self.large_hg)
    
    def test_sampler_initialization(self):
        """Test sampler initializes correctly"""
        assert self.sampler.hypergraph == self.large_hg
        assert self.sampler.strategy == 'adaptive'
        assert hasattr(self.sampler, '_graph_stats')
        assert 'num_nodes' in self.sampler._graph_stats
    
    def test_optimal_sample_size_determination(self):
        """Test optimal sample size calculation"""
        size_clustering = self.sampler.determine_optimal_sample_size('clustering')
        size_centrality = self.sampler.determine_optimal_sample_size('centrality')
        size_general = self.sampler.determine_optimal_sample_size('general')
        
        # Clustering should suggest smaller samples (more expensive)
        assert size_clustering <= size_centrality
        assert all(isinstance(size, int) for size in [size_clustering, size_centrality, size_general])
        assert all(size > 0 for size in [size_clustering, size_centrality, size_general])
    
    def test_adaptive_sampling(self):
        """Test adaptive sampling creates valid subgraphs"""
        sample_size = 50
        subgraph = self.sampler.adaptive_sample(sample_size)
        
        assert isinstance(subgraph, Hypergraph)
        assert len(subgraph.nodes) <= sample_size
        assert len(subgraph.nodes) > 0
        assert len(subgraph.edges) > 0
    
    def test_different_sampling_strategies(self):
        """Test different sampling strategies"""
        strategies = ['degree_based', 'stratified', 'random', 'adaptive']
        
        for strategy in strategies:
            sampler = SmartSampler(self.large_hg, strategy=strategy)
            subgraph = sampler.adaptive_sample(30)
            
            assert isinstance(subgraph, Hypergraph)
            assert len(subgraph.nodes) <= 30
            assert len(subgraph.nodes) > 0
    
    def test_auto_scale_algorithm(self):
        """Test auto-scaling algorithm wrapper"""
        def dummy_algorithm(hg, **kwargs):
            return {node: 1.0 for node in hg.nodes}
        
        result = auto_scale_algorithm(
            self.large_hg, dummy_algorithm, 'test_algorithm', max_nodes=50
        )
        
        assert isinstance(result, dict)
        # Should return results for all nodes (either sampled or extended)
        assert len(result) >= 50  # May extend results to full graph
    
    def test_sampling_recommendations(self):
        """Test sampling recommendations"""
        recommendations = get_sampling_recommendations(self.large_hg)
        
        assert isinstance(recommendations, dict)
        assert 'total_nodes' in recommendations
        assert 'recommended_sampling' in recommendations
        assert 'optimal_sample_sizes' in recommendations
        assert 'recommended_strategy' in recommendations


class TestEnhancedCentrality:
    """Test enhanced centrality measures"""
    
    def setup_method(self):
        """Create test hypergraph for centrality"""
        edge_dict = {
            'e1': ['n1', 'n2', 'n3'],
            'e2': ['n2', 'n3', 'n4'],
            'e3': ['n3', 'n4', 'n5'],
            'e4': ['n1', 'n5'],
            'e5': ['n6', 'n7']  # Separate component
        }
        self.hg = Hypergraph.from_dict(edge_dict)
    
    def test_enhanced_centrality_analysis(self):
        """Test comprehensive centrality analysis"""
        result = enhanced_centrality_analysis(self.hg, sample_large_graphs=False)
        
        assert isinstance(result, pl.DataFrame)
        assert 'node_id' in result.columns
        assert len(result) == len(self.hg.nodes)
        
        # Check for centrality columns
        expected_measures = ['degree', 'betweenness', 'closeness', 'eigenvector', 'harmonic']
        for measure in expected_measures:
            col_name = f'{measure}_centrality'
            if col_name in result.columns:
                # All values should be numeric
                assert result[col_name].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
    
    def test_individual_centrality_measures(self):
        """Test individual centrality measures"""
        measures = {
            'betweenness': betweenness_centrality,
            'closeness': closeness_centrality,
            'eigenvector': eigenvector_centrality,
            'harmonic': harmonic_centrality
        }
        
        for measure_name, measure_func in measures.items():
            try:
                result = measure_func(self.hg)
                assert isinstance(result, dict)
                assert len(result) == len(self.hg.nodes)
                assert all(isinstance(score, (int, float)) for score in result.values())
                assert all(node in result for node in self.hg.nodes)
                
            except Exception as e:
                pytest.fail(f"{measure_name} centrality failed: {e}")
    
    def test_centrality_with_weights(self):
        """Test centrality measures with weighted edges"""
        # Add weights
        for i, edge in enumerate(self.hg.edges):
            for node in self.hg.incidences.get_edge_nodes(edge):
                self.hg.incidences.set_weight(edge, node, 1.0 + i * 0.2)
        
        result = enhanced_centrality_analysis(
            self.hg, 
            measures=['degree', 'betweenness'], 
            weight_column='weight',
            sample_large_graphs=False
        )
        
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(self.hg.nodes)
    
    def test_centrality_edge_cases(self):
        """Test centrality with edge cases"""
        # Single node
        single_node_hg = Hypergraph.from_dict({'e1': ['n1']})
        result = enhanced_centrality_analysis(single_node_hg, sample_large_graphs=False)
        assert len(result) == 1
        
        # Disconnected graph
        disconnected_hg = Hypergraph.from_dict({
            'e1': ['n1', 'n2'],
            'e2': ['n3', 'n4']
        })
        result = enhanced_centrality_analysis(disconnected_hg, sample_large_graphs=False)
        assert len(result) == 4


class TestPerformanceImprovements:
    """Test performance monitoring and improvements"""
    
    def test_clustering_performance_on_medium_graph(self):
        """Test clustering performance on medium-sized graph"""
        # Create medium graph (should be fast)
        edge_dict = {f'e{i}': [f'n{j}' for j in range(i, i+3)] 
                    for i in range(50)}  # 50 edges, ~150 nodes
        
        medium_hg = Hypergraph.from_dict(edge_dict)
        
        # These should complete reasonably fast
        import time
        
        start_time = time.time()
        result = hypergraph_clustering(medium_hg, algorithm='modularity')
        clustering_time = time.time() - start_time
        
        assert isinstance(result, dict)
        assert clustering_time < 10.0  # Should complete in under 10 seconds
    
    def test_centrality_performance_with_sampling(self):
        """Test centrality performance with auto-sampling"""
        # Create larger graph
        edge_dict = {f'e{i}': [f'n{j}' for j in range(i, i+4)] 
                    for i in range(200)}  # Larger graph
        
        large_hg = Hypergraph.from_dict(edge_dict)
        
        import time
        start_time = time.time()
        
        # Should use sampling automatically for large graphs
        result = enhanced_centrality_analysis(
            large_hg, 
            measures=['degree', 'betweenness'],
            sample_large_graphs=True,
            max_nodes=100
        )
        
        analysis_time = time.time() - start_time
        
        assert isinstance(result, pl.DataFrame)
        assert analysis_time < 30.0  # Should complete reasonably fast with sampling


class TestIntegrationFixes:
    """Integration tests for all fixes working together"""
    
    def test_full_pipeline_small_graph(self):
        """Test complete analysis pipeline on small graph"""
        edge_dict = {
            'e1': ['n1', 'n2', 'n3'],
            'e2': ['n2', 'n3', 'n4'],
            'e3': ['n1', 'n4', 'n5']
        }
        hg = Hypergraph.from_dict(edge_dict)
        
        # Should be able to run full analysis without crashes
        try:
            # Clustering
            clusters = hypergraph_clustering(hg, algorithm='spectral', n_clusters=2)
            assert isinstance(clusters, dict)
            
            # Centrality
            centrality = enhanced_centrality_analysis(hg, sample_large_graphs=False)
            assert isinstance(centrality, pl.DataFrame)
            
            # Sampling recommendations
            recommendations = get_sampling_recommendations(hg)
            assert isinstance(recommendations, dict)
            
        except Exception as e:
            pytest.fail(f"Full pipeline failed: {e}")
    
    def test_full_pipeline_large_graph(self):
        """Test complete analysis pipeline on large graph with sampling"""
        # Create larger graph
        edge_dict = {f'e{i}': [f'n{j}' for j in range(i*3, i*3+5)] 
                    for i in range(100)}  # ~500 nodes
        
        large_hg = Hypergraph.from_dict(edge_dict)
        
        try:
            # Should use sampling automatically
            centrality = enhanced_centrality_analysis(
                large_hg, 
                measures=['degree', 'betweenness'],
                sample_large_graphs=True,
                max_nodes=200
            )
            assert isinstance(centrality, pl.DataFrame)
            
            # Clustering with sampling
            def clustering_func(hg, **kwargs):
                return hypergraph_clustering(hg, algorithm='modularity', **kwargs)
            
            clusters = auto_scale_algorithm(
                large_hg, clustering_func, 'clustering', max_nodes=200
            )
            assert isinstance(clusters, dict)
            
        except Exception as e:
            pytest.fail(f"Large graph pipeline failed: {e}")


def run_all_tests():
    """Run all tests and report results"""
    test_classes = [
        TestIncidenceStoreFixes,
        TestClusteringFixes,
        TestSmartSampling,
        TestEnhancedCentrality,
        TestPerformanceImprovements,
        TestIntegrationFixes
    ]
    
    results = {}
    
    for test_class in test_classes:
        class_name = test_class.__name__
        print(f"\n=== Running {class_name} ===")
        
        test_instance = test_class()
        methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        passed = 0
        failed = 0
        
        for method_name in methods:
            try:
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                method = getattr(test_instance, method_name)
                method()
                
                print(f"  ✓ {method_name}")
                passed += 1
                
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                failed += 1
        
        results[class_name] = {'passed': passed, 'failed': failed}
        print(f"  Results: {passed} passed, {failed} failed")
    
    # Summary
    print(f"\n=== Test Summary ===")
    total_passed = sum(r['passed'] for r in results.values())
    total_failed = sum(r['failed'] for r in results.values())
    
    for class_name, result in results.items():
        status = "✓" if result['failed'] == 0 else "✗"
        print(f"{status} {class_name}: {result['passed']}/{result['passed'] + result['failed']}")
    
    print(f"\nOverall: {total_passed} passed, {total_failed} failed")
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)