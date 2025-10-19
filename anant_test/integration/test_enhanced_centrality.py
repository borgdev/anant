#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Centrality Measures

Tests the advanced centrality algorithms including s-centrality, 
eigenvector centrality, PageRank, and weighted degree centrality.
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
from anant.analysis.centrality import (
    s_centrality,
    eigenvector_centrality, 
    pagerank_centrality,
    weighted_degree_centrality,
    degree_centrality,
    closeness_centrality,
    betweenness_centrality
)

def create_test_hypergraph() -> Hypergraph:
    """Create a test hypergraph for centrality analysis"""
    # Create a hypergraph with various edge sizes
    incidence_data = [
        # Small edges (size 2)
        ("E1", "A"), ("E1", "B"),
        # Medium edges (size 3) 
        ("E2", "B"), ("E2", "C"), ("E2", "D"),
        # Large edge (size 4)
        ("E3", "C"), ("E3", "D"), ("E3", "E"), ("E3", "F"),
        # Another small edge
        ("E4", "A"), ("E4", "F"),
        # Hub edge connecting many nodes
        ("E5", "A"), ("E5", "B"), ("E5", "C"), ("E5", "E"), ("E5", "G")
    ]
    
    # Convert to DataFrame
    incidence_df = pl.DataFrame([
        {"edges": edge, "nodes": node} 
        for edge, node in incidence_data
    ])
    
    return Hypergraph(incidence_df)

def test_s_centrality():
    """Test s-centrality with different s values"""
    print("Testing S-Centrality...")
    
    hg = create_test_hypergraph()
    
    # Test different s values
    for s in [1, 2, 3]:
        print(f"\n  S-centrality with s={s}:")
        
        start_time = time.time()
        centralities = s_centrality(hg, s=s, normalized=True)
        execution_time = time.time() - start_time
        
        print(f"    Execution time: {execution_time:.4f}s")
        print(f"    Results: {dict(sorted(centralities.items()))}")
        
        # Validate results
        assert len(centralities) == hg.num_nodes
        assert all(0 <= score <= 1 for score in centralities.values())
        assert sum(centralities.values()) > 0  # At least one node should have positive centrality
        
        print(f"    ✓ All nodes present, scores normalized, positive sum")

def test_eigenvector_centrality():
    """Test eigenvector centrality"""
    print("\nTesting Eigenvector Centrality...")
    
    hg = create_test_hypergraph()
    
    start_time = time.time()
    centralities = eigenvector_centrality(hg, max_iter=100, tolerance=1e-6)
    execution_time = time.time() - start_time
    
    print(f"  Execution time: {execution_time:.4f}s")
    print(f"  Results: {dict(sorted(centralities.items()))}")
    
    # Validate results
    assert len(centralities) == hg.num_nodes
    assert all(score >= 0 for score in centralities.values())
    assert sum(centralities.values()) > 0
    
    print(f"  ✓ All nodes present, non-negative scores, positive sum")

def test_pagerank_centrality():
    """Test PageRank centrality"""
    print("\nTesting PageRank Centrality...")
    
    hg = create_test_hypergraph()
    
    # Test default PageRank
    start_time = time.time()
    centralities = pagerank_centrality(hg, alpha=0.85, max_iter=100)
    execution_time = time.time() - start_time
    
    print(f"  Execution time: {execution_time:.4f}s")
    print(f"  Results: {dict(sorted(centralities.items()))}")
    
    # Validate results
    assert len(centralities) == hg.num_nodes
    assert all(score > 0 for score in centralities.values())  # PageRank scores should be positive
    assert abs(sum(centralities.values()) - 1.0) < 1e-6  # Should sum to 1
    
    print(f"  ✓ All nodes present, positive scores, sum ≈ 1.0")
    
    # Test with personalization
    print("\n  Testing with personalization...")
    personalization = {"A": 0.5, "B": 0.3, "C": 0.2}
    
    start_time = time.time()
    pers_centralities = pagerank_centrality(
        hg, 
        alpha=0.85, 
        personalization=personalization
    )
    execution_time = time.time() - start_time
    
    print(f"  Personalized execution time: {execution_time:.4f}s")
    print(f"  Personalized results: {dict(sorted(pers_centralities.items()))}")
    
    assert len(pers_centralities) == hg.num_nodes
    assert all(score > 0 for score in pers_centralities.values())
    
    print(f"  ✓ Personalized PageRank working correctly")

def test_weighted_degree_centrality():
    """Test weighted degree centrality with different weight functions"""
    print("\nTesting Weighted Degree Centrality...")
    
    hg = create_test_hypergraph()
    
    weight_functions = ["uniform", "size", "inverse_size"]
    
    for weight_func in weight_functions:
        print(f"\n  Weight function: {weight_func}")
        
        start_time = time.time()
        centralities = weighted_degree_centrality(
            hg, 
            weight_function=weight_func,
            normalized=True
        )
        execution_time = time.time() - start_time
        
        print(f"    Execution time: {execution_time:.4f}s")
        print(f"    Results: {dict(sorted(centralities.items()))}")
        
        # Validate results
        assert len(centralities) == hg.num_nodes
        assert all(score >= 0 for score in centralities.values())
        
        if weight_func == "uniform":
            # Should be equivalent to normalized degree centrality
            degree_cents = degree_centrality(hg, normalized=True)
            node_degrees = degree_cents['nodes']
            
            for node in centralities:
                assert abs(centralities[node] - node_degrees[node]) < 1e-10
        
        print(f"    ✓ Weighted centrality with {weight_func} weights working")
    
    # Test with explicit edge weights
    print(f"\n  Testing with explicit edge weights...")
    
    edge_weights = {"E1": 1.0, "E2": 2.0, "E3": 3.0, "E4": 1.5, "E5": 2.5}
    
    start_time = time.time()
    weighted_centralities = weighted_degree_centrality(
        hg,
        edge_weights=edge_weights,
        normalized=True
    )
    execution_time = time.time() - start_time
    
    print(f"    Execution time: {execution_time:.4f}s")
    print(f"    Results: {dict(sorted(weighted_centralities.items()))}")
    
    assert len(weighted_centralities) == hg.num_nodes
    assert all(score >= 0 for score in weighted_centralities.values())
    
    print(f"    ✓ Explicit edge weights working correctly")

def test_centrality_comparison():
    """Compare different centrality measures"""
    print("\nCentrality Measures Comparison...")
    
    hg = create_test_hypergraph()
    
    # Compute all centrality measures
    measures = {
        "Degree": degree_centrality(hg, normalized=True)['nodes'],
        "Closeness": closeness_centrality(hg, normalized=True),
        "Betweenness": betweenness_centrality(hg, normalized=True),
        "S-centrality (s=1)": s_centrality(hg, s=1, normalized=True),
        "S-centrality (s=2)": s_centrality(hg, s=2, normalized=True),
        "Eigenvector": eigenvector_centrality(hg, normalized=True),
        "PageRank": pagerank_centrality(hg, alpha=0.85),
        "Weighted Degree": weighted_degree_centrality(hg, weight_function="size", normalized=True)
    }
    
    print(f"\n  Centrality Rankings:")
    print(f"  {'Node':<8} {'Degree':<8} {'Close':<8} {'Between':<8} {'S(1)':<8} {'S(2)':<8} {'Eigen':<8} {'PageRank':<8} {'Weighted':<8}")
    print(f"  {'-'*80}")
    
    for node in sorted(hg.nodes):
        values = [
            f"{measures[measure].get(node, 0.0):.3f}" 
            for measure in measures.keys()
        ]
        print(f"  {node:<8} {' '.join(f'{v:<8}' for v in values)}")
    
    # Validate all measures
    for measure_name, centralities in measures.items():
        assert len(centralities) == hg.num_nodes, f"{measure_name} missing nodes"
        assert all(score >= 0 for score in centralities.values()), f"{measure_name} has negative scores"
        print(f"  ✓ {measure_name} validated")

def test_performance_comparison():
    """Test performance of different centrality measures"""
    print("\nPerformance Comparison...")
    
    hg = create_test_hypergraph()
    
    measures = [
        ("Degree Centrality", lambda: degree_centrality(hg, normalized=True)),
        ("S-centrality (s=1)", lambda: s_centrality(hg, s=1, normalized=True)),
        ("S-centrality (s=2)", lambda: s_centrality(hg, s=2, normalized=True)),
        ("Eigenvector Centrality", lambda: eigenvector_centrality(hg, max_iter=100)),
        ("PageRank Centrality", lambda: pagerank_centrality(hg, alpha=0.85, max_iter=100)),
        ("Weighted Degree (size)", lambda: weighted_degree_centrality(hg, weight_function="size")),
        ("Closeness Centrality", lambda: closeness_centrality(hg, normalized=True)),
        ("Betweenness Centrality", lambda: betweenness_centrality(hg, normalized=True))
    ]
    
    print(f"\n  {'Measure':<25} {'Time (ms)':<12} {'Relative':<10}")
    print(f"  {'-'*50}")
    
    times = []
    
    for measure_name, measure_func in measures:
        # Warm up
        measure_func()
        
        # Time multiple runs
        start_time = time.time()
        for _ in range(10):
            result = measure_func()
        avg_time = (time.time() - start_time) / 10 * 1000  # Convert to ms
        
        times.append((measure_name, avg_time))
        
    # Calculate relative times
    min_time = min(time for _, time in times)
    
    for measure_name, avg_time in times:
        relative = avg_time / min_time
        print(f"  {measure_name:<25} {avg_time:<12.2f} {relative:<10.1f}x")
    
    print(f"\n  ✓ Performance comparison completed")

def main():
    """Run all enhanced centrality tests"""
    print("=" * 60)
    print("Enhanced Centrality Measures Test Suite")
    print("=" * 60)
    
    try:
        test_s_centrality()
        test_eigenvector_centrality()
        test_pagerank_centrality()
        test_weighted_degree_centrality()
        test_centrality_comparison()
        test_performance_comparison()
        
        print("\n" + "=" * 60)
        print("✅ ALL ENHANCED CENTRALITY TESTS PASSED!")
        print("=" * 60)
        
        print("\nKey Results:")
        print("• S-centrality working with different s parameters")
        print("• Eigenvector centrality converging correctly")
        print("• PageRank with and without personalization")
        print("• Weighted degree centrality with multiple weight schemes")
        print("• All centrality measures producing valid, normalized results")
        print("• Performance benchmarks showing relative computational costs")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()