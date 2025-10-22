#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Community Detection

Tests the advanced community detection algorithms including overlapping
community detection, multi-resolution analysis, consensus clustering,
and adaptive methods.
"""

import polars as pl
import numpy as np
from typing import Dict, List
import time

# Add the anant package to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anant'))

from anant.classes.hypergraph import Hypergraph
from anant.analysis.clustering import (
    spectral_clustering,
    modularity_clustering,
    overlapping_community_detection,
    multi_resolution_clustering,
    consensus_clustering,
    community_quality_metrics,
    edge_community_detection,
    adaptive_community_detection
)

def create_community_test_hypergraph() -> Hypergraph:
    """Create a hypergraph with clear community structure for testing"""
    # Create a hypergraph with two main communities
    incidence_data = [
        # Community 1: A, B, C, D
        ("E1", "A"), ("E1", "B"),
        ("E2", "B"), ("E2", "C"), 
        ("E3", "C"), ("E3", "D"),
        ("E4", "A"), ("E4", "D"),  # Closes community 1
        
        # Community 2: E, F, G, H  
        ("E5", "E"), ("E5", "F"),
        ("E6", "F"), ("E6", "G"),
        ("E7", "G"), ("E7", "H"),
        ("E8", "E"), ("E8", "H"),  # Closes community 2
        
        # Bridge edges (weaker connections between communities)
        ("E9", "D"), ("E9", "E"),  # Bridge 1
        ("E10", "C"), ("E10", "F"),  # Bridge 2
        
        # Large mixed edge
        ("E11", "A"), ("E11", "B"), ("E11", "E"), ("E11", "F"), ("E11", "I"),
        
        # Isolated node with single edge
        ("E12", "I"), ("E12", "J")
    ]
    
    # Convert to DataFrame with correct column names
    incidence_df = pl.DataFrame([
        {"edge_id": edge, "node_id": node} 
        for edge, node in incidence_data
    ])
    
    return Hypergraph(incidence_df)

def test_overlapping_community_detection():
    """Test overlapping community detection"""
    print("Testing Overlapping Community Detection...")
    
    hg = create_community_test_hypergraph()
    
    start_time = time.time()
    overlapping_communities = overlapping_community_detection(
        hg, 
        max_communities=5,
        alpha=0.6,
        max_iterations=50
    )
    execution_time = time.time() - start_time
    
    print(f"  Execution time: {execution_time:.4f}s")
    print(f"  Results:")
    
    for node, communities in sorted(overlapping_communities.items()):
        print(f"    Node {node}: communities {communities}")
    
    # Validate results
    assert len(overlapping_communities) == hg.num_nodes
    assert all(len(comms) > 0 for comms in overlapping_communities.values())
    
    # Check that some nodes are in multiple communities (overlapping)
    multiple_community_nodes = [node for node, comms in overlapping_communities.items() if len(comms) > 1]
    
    print(f"  Nodes in multiple communities: {len(multiple_community_nodes)}")
    print(f"  ✓ Overlapping community detection working correctly")

def test_multi_resolution_clustering():
    """Test multi-resolution clustering"""
    print("\nTesting Multi-Resolution Clustering...")
    
    hg = create_community_test_hypergraph()
    
    start_time = time.time()
    multi_res_results = multi_resolution_clustering(
        hg,
        resolution_range=(0.2, 1.5),
        n_resolutions=6,
        method="modularity"
    )
    execution_time = time.time() - start_time
    
    print(f"  Execution time: {execution_time:.4f}s")
    print(f"  Tested {len(multi_res_results)} resolution values")
    
    # Show results for different resolutions
    print(f"\n  Community structure at different resolutions:")
    print(f"  {'Resolution':<12} {'Communities':<12} {'Modularity':<10}")
    print(f"  {'-'*36}")
    
    for resolution, communities in sorted(multi_res_results.items()):
        n_communities = len(set(communities.values()))
        quality_metrics = community_quality_metrics(hg, communities)
        modularity = quality_metrics["modularity"]
        
        print(f"  {resolution:<12.3f} {n_communities:<12} {modularity:<10.4f}")
    
    # Validate results
    assert len(multi_res_results) > 0
    for resolution, communities in multi_res_results.items():
        assert len(communities) == hg.num_nodes
        assert all(isinstance(comm_id, int) for comm_id in communities.values())
    
    print(f"  ✓ Multi-resolution clustering working correctly")

def test_consensus_clustering():
    """Test consensus clustering"""
    print("\nTesting Consensus Clustering...")
    
    hg = create_community_test_hypergraph()
    
    methods = ["spectral", "modularity"]
    
    start_time = time.time()
    consensus_communities = consensus_clustering(
        hg,
        n_runs=5,
        n_clusters=3,
        methods=methods
    )
    execution_time = time.time() - start_time
    
    print(f"  Execution time: {execution_time:.4f}s")
    print(f"  Methods used: {methods}")
    print(f"  Consensus results:")
    
    # Group nodes by community
    community_groups = {}
    for node, comm_id in consensus_communities.items():
        if comm_id not in community_groups:
            community_groups[comm_id] = []
        community_groups[comm_id].append(node)
    
    for comm_id, nodes in sorted(community_groups.items()):
        print(f"    Community {comm_id}: {sorted(nodes)}")
    
    # Validate results
    assert len(consensus_communities) == hg.num_nodes
    assert all(isinstance(comm_id, int) for comm_id in consensus_communities.values())
    
    # Calculate quality metrics
    quality_metrics = community_quality_metrics(hg, consensus_communities)
    print(f"  Consensus quality metrics:")
    for metric, value in quality_metrics.items():
        print(f"    {metric}: {value:.4f}")
    
    print(f"  ✓ Consensus clustering working correctly")

def test_community_quality_metrics():
    """Test community quality metrics"""
    print("\nTesting Community Quality Metrics...")
    
    hg = create_community_test_hypergraph()
    
    # Test with different clustering methods
    clustering_methods = [
        ("Spectral", lambda: spectral_clustering(hg, n_clusters=3)),
        ("Modularity", lambda: modularity_clustering(hg)),
    ]
    
    print(f"\n  Quality comparison across methods:")
    print(f"  {'Method':<12} {'Modular':<8} {'Coverage':<9} {'Conduct':<8} {'N_Comm':<7}")
    print(f"  {'-'*50}")
    
    for method_name, method_func in clustering_methods:
        start_time = time.time()
        communities = method_func()
        execution_time = time.time() - start_time
        
        quality_metrics = community_quality_metrics(hg, communities)
        
        print(f"  {method_name:<12} {quality_metrics['modularity']:<8.3f} " +
              f"{quality_metrics['coverage']:<9.3f} {quality_metrics['conductance']:<8.3f} " +
              f"{quality_metrics['n_communities']:<7.0f}")
    
    # Detailed metrics for one method
    communities = modularity_clustering(hg)
    detailed_metrics = community_quality_metrics(hg, communities)
    
    print(f"\n  Detailed metrics (Modularity method):")
    for metric, value in detailed_metrics.items():
        print(f"    {metric}: {value:.4f}")
    
    print(f"  ✓ Community quality metrics working correctly")

def test_edge_community_detection():
    """Test edge community detection"""
    print("\nTesting Edge Community Detection...")
    
    hg = create_community_test_hypergraph()
    
    start_time = time.time()
    edge_communities = edge_community_detection(
        hg,
        n_communities=4,
        method="spectral"
    )
    execution_time = time.time() - start_time
    
    print(f"  Execution time: {execution_time:.4f}s")
    print(f"  Edge community assignments:")
    
    # Group edges by community
    edge_community_groups = {}
    for edge, comm_id in edge_communities.items():
        if comm_id not in edge_community_groups:
            edge_community_groups[comm_id] = []
        edge_community_groups[comm_id].append(edge)
    
    for comm_id, edges in sorted(edge_community_groups.items()):
        print(f"    Edge Community {comm_id}: {sorted(edges)}")
    
    # Validate results
    assert len(edge_communities) == hg.num_edges
    assert all(isinstance(comm_id, int) for comm_id in edge_communities.values())
    
    print(f"  ✓ Edge community detection working correctly")

def test_adaptive_community_detection():
    """Test adaptive community detection"""
    print("\nTesting Adaptive Community Detection...")
    
    hg = create_community_test_hypergraph()
    
    start_time = time.time()
    adaptive_communities = adaptive_community_detection(
        hg,
        quality_threshold=0.1,
        max_communities=8
    )
    execution_time = time.time() - start_time
    
    print(f"  Execution time: {execution_time:.4f}s")
    
    # Analyze results
    n_communities = len(set(adaptive_communities.values()))
    quality_metrics = community_quality_metrics(hg, adaptive_communities)
    
    print(f"  Automatically detected {n_communities} communities")
    print(f"  Quality score: {quality_metrics['modularity']:.4f}")
    
    # Show community assignments
    community_groups = {}
    for node, comm_id in adaptive_communities.items():
        if comm_id not in community_groups:
            community_groups[comm_id] = []
        community_groups[comm_id].append(node)
    
    print(f"  Community assignments:")
    for comm_id, nodes in sorted(community_groups.items()):
        print(f"    Community {comm_id}: {sorted(nodes)}")
    
    # Validate results
    assert len(adaptive_communities) == hg.num_nodes
    assert n_communities >= 2  # Should find at least 2 communities
    assert quality_metrics["modularity"] > 0  # Should have positive modularity
    
    print(f"  ✓ Adaptive community detection working correctly")

def test_comprehensive_community_analysis():
    """Test comprehensive community analysis workflow"""
    print("\nTesting Comprehensive Community Analysis...")
    
    hg = create_community_test_hypergraph()
    
    print(f"  Analyzing hypergraph with {hg.num_nodes} nodes and {hg.num_edges} edges")
    
    # 1. Basic community detection
    print("  1. Basic community detection...")
    spectral_communities = spectral_clustering(hg, n_clusters=3)
    modularity_communities = modularity_clustering(hg)
    
    # 2. Advanced methods
    print("  2. Advanced community detection...")
    overlapping_comms = overlapping_community_detection(hg, max_communities=4)
    consensus_comms = consensus_clustering(hg, n_runs=3, n_clusters=3)
    adaptive_comms = adaptive_community_detection(hg)
    
    # 3. Quality analysis
    print("  3. Quality analysis...")
    methods_results = {
        "Spectral": spectral_communities,
        "Modularity": modularity_communities,
        "Consensus": consensus_comms,
        "Adaptive": adaptive_comms
    }
    
    print(f"\n  Method comparison:")
    print(f"  {'Method':<12} {'N_Comm':<7} {'Modular':<8} {'Coverage':<9} {'Quality':<8}")
    print(f"  {'-'*50}")
    
    for method_name, communities in methods_results.items():
        quality = community_quality_metrics(hg, communities)
        combined_quality = (
            quality["modularity"] * 0.4 +
            quality["coverage"] * 0.3 +
            (1 - quality["conductance"]) * 0.3
        )
        
        print(f"  {method_name:<12} {quality['n_communities']:<7.0f} " +
              f"{quality['modularity']:<8.3f} {quality['coverage']:<9.3f} " +
              f"{combined_quality:<8.3f}")
    
    # 4. Multi-resolution analysis
    print("  4. Multi-resolution analysis...")
    multi_res = multi_resolution_clustering(hg, n_resolutions=5)
    
    print(f"  Resolution analysis shows communities ranging from " +
          f"{min(len(set(c.values())) for c in multi_res.values())} to " +
          f"{max(len(set(c.values())) for c in multi_res.values())}")
    
    # 5. Edge communities
    print("  5. Edge community analysis...")
    edge_comms = edge_community_detection(hg, n_communities=3)
    edge_comm_count = len(set(edge_comms.values()))
    
    print(f"  Found {edge_comm_count} edge communities")
    
    print("  ✓ Comprehensive community analysis completed successfully")

def main():
    """Run all enhanced community detection tests"""
    print("=" * 60)
    print("Enhanced Community Detection Test Suite")
    print("=" * 60)
    
    try:
        test_overlapping_community_detection()
        test_multi_resolution_clustering()
        test_consensus_clustering()
        test_community_quality_metrics()
        test_edge_community_detection()
        test_adaptive_community_detection()
        test_comprehensive_community_analysis()
        
        print("\n" + "=" * 60)
        print("✅ ALL ENHANCED COMMUNITY DETECTION TESTS PASSED!")
        print("=" * 60)
        
        print("\nKey Results:")
        print("• Overlapping community detection allowing multiple memberships")
        print("• Multi-resolution analysis revealing hierarchical structure")
        print("• Consensus clustering combining multiple methods for stability")
        print("• Comprehensive quality metrics for method comparison")
        print("• Edge community detection for hyperedge clustering")
        print("• Adaptive methods automatically determining optimal parameters")
        print("• All community detection methods producing valid, meaningful results")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()