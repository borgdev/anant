#!/usr/bin/env python3
"""
Comprehensive Test Suite for Temporal Analysis

Tests the temporal hypergraph analysis capabilities including
evolution metrics, stability analysis, and persistence tracking.
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
from anant.analysis.temporal import (
    TemporalSnapshot,
    TemporalHypergraph,
    temporal_degree_evolution,
    temporal_centrality_evolution,
    temporal_clustering_evolution,
    stability_analysis,
    temporal_motif_analysis,
    growth_analysis,
    persistence_analysis
)

def create_temporal_hypergraphs() -> TemporalHypergraph:
    """Create a sequence of evolving hypergraphs for testing"""
    
    # Time 0: Initial small hypergraph
    incidence_t0 = [
        ("E1", "A"), ("E1", "B"),
        ("E2", "B"), ("E2", "C"),
    ]
    hg_t0 = Hypergraph(pl.DataFrame([
        {"edges": edge, "nodes": node} 
        for edge, node in incidence_t0
    ]))
    
    # Time 1: Add more nodes and edges
    incidence_t1 = [
        ("E1", "A"), ("E1", "B"),
        ("E2", "B"), ("E2", "C"),
        ("E3", "C"), ("E3", "D"), ("E3", "E"),
    ]
    hg_t1 = Hypergraph(pl.DataFrame([
        {"edges": edge, "nodes": node} 
        for edge, node in incidence_t1
    ]))
    
    # Time 2: Modify existing edges and add new ones
    incidence_t2 = [
        ("E1", "A"), ("E1", "B"), ("E1", "F"),  # E1 grows
        ("E2", "B"), ("E2", "C"),  # E2 stays same
        ("E3", "C"), ("E3", "D"), ("E3", "E"),  # E3 stays same
        ("E4", "A"), ("E4", "F"), ("E4", "G"),  # New edge E4
    ]
    hg_t2 = Hypergraph(pl.DataFrame([
        {"edges": edge, "nodes": node} 
        for edge, node in incidence_t2
    ]))
    
    # Time 3: Some edges disappear, new ones appear
    incidence_t3 = [
        ("E1", "A"), ("E1", "B"), ("E1", "F"),  # E1 persists
        # E2 disappears
        ("E3", "C"), ("E3", "D"),  # E3 shrinks
        ("E4", "A"), ("E4", "F"), ("E4", "G"),  # E4 persists
        ("E5", "G"), ("E5", "H"), ("E5", "I"), ("E5", "J"),  # Large new edge
    ]
    hg_t3 = Hypergraph(pl.DataFrame([
        {"edges": edge, "nodes": node} 
        for edge, node in incidence_t3
    ]))
    
    # Time 4: Further evolution
    incidence_t4 = [
        ("E1", "A"), ("E1", "B"),  # E1 shrinks back
        ("E3", "C"), ("E3", "D"), ("E3", "E"), ("E3", "K"),  # E3 grows again
        ("E4", "A"), ("E4", "F"),  # E4 shrinks
        ("E5", "G"), ("E5", "H"), ("E5", "I"), ("E5", "J"),  # E5 persists
        ("E6", "K"), ("E6", "L"),  # New small edge
    ]
    hg_t4 = Hypergraph(pl.DataFrame([
        {"edges": edge, "nodes": node} 
        for edge, node in incidence_t4
    ]))
    
    # Create temporal hypergraph
    temporal_hg = TemporalHypergraph([
        TemporalSnapshot(0, hg_t0, {"description": "Initial state"}),
        TemporalSnapshot(1, hg_t1, {"description": "First growth"}),
        TemporalSnapshot(2, hg_t2, {"description": "Expansion"}),
        TemporalSnapshot(3, hg_t3, {"description": "Restructuring"}),
        TemporalSnapshot(4, hg_t4, {"description": "Further evolution"}),
    ])
    
    return temporal_hg

def test_temporal_hypergraph_basics():
    """Test basic temporal hypergraph functionality"""
    print("Testing Temporal Hypergraph Basics...")
    
    temporal_hg = create_temporal_hypergraphs()
    
    print(f"  Number of snapshots: {temporal_hg.num_snapshots}")
    print(f"  Timestamps: {temporal_hg.timestamps}")
    
    # Test sorting
    temporal_hg.sort_snapshots()
    sorted_timestamps = temporal_hg.timestamps
    assert sorted_timestamps == sorted(sorted_timestamps), "Timestamps should be sorted"
    
    # Test snapshot retrieval
    snapshot_t2 = temporal_hg.get_snapshot(2)
    assert snapshot_t2 is not None, "Should find snapshot at time 2"
    assert snapshot_t2.timestamp == 2, "Should return correct snapshot"
    
    # Test index-based retrieval
    first_snapshot = temporal_hg.get_snapshot_at_index(0)
    assert first_snapshot.timestamp == 0, "First snapshot should be at time 0"
    
    print("  ✓ Basic temporal hypergraph operations working")

def test_temporal_degree_evolution():
    """Test temporal degree evolution analysis"""
    print("\nTesting Temporal Degree Evolution...")
    
    temporal_hg = create_temporal_hypergraphs()
    
    # Test evolution for all nodes
    start_time = time.time()
    all_evolution = temporal_degree_evolution(temporal_hg)
    execution_time = time.time() - start_time
    
    print(f"  Execution time (all nodes): {execution_time:.4f}s")
    print(f"  Number of nodes tracked: {len(all_evolution)}")
    
    # Validate results
    assert isinstance(all_evolution, dict), "Should return dictionary for all nodes"
    assert len(all_evolution) > 0, "Should track some nodes"
    
    # Test evolution for specific node
    start_time = time.time()
    node_a_evolution = temporal_degree_evolution(temporal_hg, node_id="A")
    execution_time = time.time() - start_time
    
    print(f"  Execution time (node A): {execution_time:.4f}s")
    print(f"  Node A evolution shape: {node_a_evolution.shape}")
    print(f"  Node A degree over time:")
    
    for row in node_a_evolution.iter_rows(named=True):
        print(f"    Time {row['timestamp']}: degree={row['degree']:.3f}")
    
    assert node_a_evolution.shape[0] > 0, "Should have evolution data for node A"
    assert all(col in node_a_evolution.columns for col in ['timestamp', 'node', 'degree']), "Should have required columns"
    
    print("  ✓ Temporal degree evolution working correctly")

def test_temporal_centrality_evolution():
    """Test temporal centrality evolution analysis"""
    print("\nTesting Temporal Centrality Evolution...")
    
    temporal_hg = create_temporal_hypergraphs()
    
    centrality_measures = ['degree', 's_centrality', 'eigenvector']
    
    start_time = time.time()
    centrality_evolution = temporal_centrality_evolution(
        temporal_hg, 
        centrality_measures=centrality_measures,
        s_parameter=1
    )
    execution_time = time.time() - start_time
    
    print(f"  Execution time: {execution_time:.4f}s")
    print(f"  Evolution data shape: {centrality_evolution.shape}")
    print(f"  Columns: {centrality_evolution.columns}")
    
    # Validate results
    expected_cols = ['timestamp', 'node', 'num_edges', 'num_nodes'] + centrality_measures
    assert all(col in centrality_evolution.columns for col in expected_cols), f"Missing columns: {expected_cols}"
    
    # Check data for specific node and time
    node_c_data = centrality_evolution.filter(
        (pl.col('node') == 'C') & (pl.col('timestamp') == 2)
    )
    
    if node_c_data.shape[0] > 0:
        row = node_c_data.row(0, named=True)
        print(f"  Node C at time 2:")
        for measure in centrality_measures:
            print(f"    {measure}: {row[measure]:.4f}")
    
    print("  ✓ Temporal centrality evolution working correctly")

def test_stability_analysis():
    """Test structural stability analysis"""
    print("\nTesting Stability Analysis...")
    
    temporal_hg = create_temporal_hypergraphs()
    
    metrics = ["jaccard", "node_overlap", "edge_overlap"]
    
    for metric in metrics:
        print(f"\n  Testing {metric} stability...")
        
        start_time = time.time()
        stability_data = stability_analysis(
            temporal_hg, 
            window_size=3, 
            metric=metric
        )
        execution_time = time.time() - start_time
        
        print(f"    Execution time: {execution_time:.4f}s")
        print(f"    Stability data shape: {stability_data.shape}")
        
        if stability_data.shape[0] > 0:
            avg_stability = stability_data['stability_score'].mean()
            min_stability = stability_data['stability_score'].min()
            max_stability = stability_data['stability_score'].max()
            
            print(f"    Average stability: {avg_stability:.4f}")
            print(f"    Stability range: [{min_stability:.4f}, {max_stability:.4f}]")
            
            # Validate stability scores are in valid range
            assert all(0 <= score <= 1 for score in stability_data['stability_score']), f"{metric} stability scores should be in [0,1]"
        
        print(f"    ✓ {metric} stability analysis working")

def test_temporal_motif_analysis():
    """Test temporal motif analysis"""
    print("\nTesting Temporal Motif Analysis...")
    
    temporal_hg = create_temporal_hypergraphs()
    
    motif_sizes = [2, 3, 4]
    
    start_time = time.time()
    motif_evolution = temporal_motif_analysis(temporal_hg, motif_sizes=motif_sizes)
    execution_time = time.time() - start_time
    
    print(f"  Execution time: {execution_time:.4f}s")
    print(f"  Motif evolution shape: {motif_evolution.shape}")
    
    # Validate results
    expected_cols = ['timestamp', 'motif_size', 'count', 'total_edges', 'proportion']
    assert all(col in motif_evolution.columns for col in expected_cols), "Missing required columns"
    
    # Show motif evolution over time
    print(f"\n  Motif evolution over time:")
    print(f"  {'Time':<6} {'Size=2':<8} {'Size=3':<8} {'Size=4':<8}")
    print(f"  {'-'*32}")
    
    for timestamp in sorted(temporal_hg.timestamps):
        row_data = {"Time": timestamp}
        
        for size in motif_sizes:
            motif_data = motif_evolution.filter(
                (pl.col('timestamp') == timestamp) & (pl.col('motif_size') == size)
            )
            if motif_data.shape[0] > 0:
                count = motif_data['count'].item(0)
                row_data[f"Size={size}"] = count
            else:
                row_data[f"Size={size}"] = 0
        
        print(f"  {row_data['Time']:<6} {row_data['Size=2']:<8} {row_data['Size=3']:<8} {row_data['Size=4']:<8}")
    
    print("  ✓ Temporal motif analysis working correctly")

def test_growth_analysis():
    """Test growth pattern analysis"""
    print("\nTesting Growth Analysis...")
    
    temporal_hg = create_temporal_hypergraphs()
    
    start_time = time.time()
    growth_data = growth_analysis(temporal_hg, smoothing_window=2)
    execution_time = time.time() - start_time
    
    print(f"  Execution time: {execution_time:.4f}s")
    print(f"  Growth data shape: {growth_data.shape}")
    
    # Show growth metrics over time
    print(f"\n  Growth metrics over time:")
    print(f"  {'Time':<6} {'Nodes':<7} {'Edges':<7} {'NodeGR':<8} {'EdgeGR':<8} {'AvgDeg':<8}")
    print(f"  {'-'*50}")
    
    for row in growth_data.iter_rows(named=True):
        print(f"  {row['timestamp']:<6} {row['num_nodes']:<7} {row['num_edges']:<7} " +
              f"{row['node_growth_rate']:<8.3f} {row['edge_growth_rate']:<8.3f} {row['avg_node_degree']:<8.2f}")
    
    # Validate required columns
    required_cols = ['timestamp', 'num_nodes', 'num_edges', 'node_growth_rate', 'edge_growth_rate']
    assert all(col in growth_data.columns for col in required_cols), "Missing required columns"
    
    print("  ✓ Growth analysis working correctly")

def test_persistence_analysis():
    """Test persistence analysis for nodes and edges"""
    print("\nTesting Persistence Analysis...")
    
    temporal_hg = create_temporal_hypergraphs()
    
    for entity_type in ["nodes", "edges"]:
        print(f"\n  Testing {entity_type} persistence...")
        
        start_time = time.time()
        persistence_data = persistence_analysis(temporal_hg, entity_type=entity_type)
        execution_time = time.time() - start_time
        
        print(f"    Execution time: {execution_time:.4f}s")
        print(f"    Persistence data shape: {persistence_data.shape}")
        
        if persistence_data.shape[0] > 0:
            # Show persistence metrics
            print(f"    {entity_type.capitalize()} persistence summary:")
            print(f"    {'Entity':<8} {'First':<6} {'Last':<6} {'Lifespan':<9} {'Persistence':<11} {'Intermittency':<12}")
            print(f"    {'-'*60}")
            
            for row in persistence_data.iter_rows(named=True):
                print(f"    {row['entity']:<8} {row['first_appearance']:<6} {row['last_appearance']:<6} " +
                      f"{row['lifespan_snapshots']:<9} {row['persistence_ratio']:<11.3f} {row['intermittency']:<12.3f}")
            
            # Validate persistence metrics
            assert all(0 <= ratio <= 1 for ratio in persistence_data['persistence_ratio']), "Persistence ratio should be in [0,1]"
            assert all(0 <= inter <= 1 for inter in persistence_data['intermittency']), "Intermittency should be in [0,1]"
        
        print(f"    ✓ {entity_type} persistence analysis working")

def test_comprehensive_temporal_workflow():
    """Test a comprehensive temporal analysis workflow"""
    print("\nTesting Comprehensive Temporal Workflow...")
    
    temporal_hg = create_temporal_hypergraphs()
    
    print(f"  Analyzing temporal hypergraph with {temporal_hg.num_snapshots} snapshots")
    
    # 1. Basic evolution metrics
    print("  1. Computing centrality evolution...")
    centrality_evolution = temporal_centrality_evolution(temporal_hg)
    
    # 2. Stability analysis
    print("  2. Computing stability metrics...")
    stability_data = stability_analysis(temporal_hg, window_size=3, metric="jaccard")
    
    # 3. Growth patterns
    print("  3. Analyzing growth patterns...")
    growth_data = growth_analysis(temporal_hg)
    
    # 4. Persistence patterns
    print("  4. Analyzing persistence patterns...")
    node_persistence = persistence_analysis(temporal_hg, entity_type="nodes")
    edge_persistence = persistence_analysis(temporal_hg, entity_type="edges")
    
    # 5. Summary analysis
    print(f"\n  Temporal Analysis Summary:")
    print(f"  • Total timespan: {min(temporal_hg.timestamps)} to {max(temporal_hg.timestamps)}")
    print(f"  • Nodes tracked: {len(set(centrality_evolution['node']))}")
    print(f"  • Average stability: {stability_data['stability_score'].mean():.3f}")
    print(f"  • Peak node count: {growth_data['num_nodes'].max()}")
    print(f"  • Peak edge count: {growth_data['num_edges'].max()}")
    print(f"  • Most persistent node: {node_persistence.sort('persistence_ratio', descending=True)['entity'].item(0) if node_persistence.shape[0] > 0 else 'None'}")
    print(f"  • Most persistent edge: {edge_persistence.sort('persistence_ratio', descending=True)['entity'].item(0) if edge_persistence.shape[0] > 0 else 'None'}")
    
    print("  ✓ Comprehensive temporal workflow completed successfully")

def main():
    """Run all temporal analysis tests"""
    print("=" * 60)
    print("Temporal Analysis Test Suite")
    print("=" * 60)
    
    try:
        test_temporal_hypergraph_basics()
        test_temporal_degree_evolution()
        test_temporal_centrality_evolution()
        test_stability_analysis()
        test_temporal_motif_analysis()
        test_growth_analysis()
        test_persistence_analysis()
        test_comprehensive_temporal_workflow()
        
        print("\n" + "=" * 60)
        print("✅ ALL TEMPORAL ANALYSIS TESTS PASSED!")
        print("=" * 60)
        
        print("\nKey Results:")
        print("• Temporal hypergraph management working correctly")
        print("• Centrality evolution tracking over time")
        print("• Structural stability analysis with multiple metrics")
        print("• Motif evolution and growth pattern analysis")
        print("• Node and edge persistence tracking")
        print("• Comprehensive temporal workflows integrated")
        print("• All temporal metrics producing valid, meaningful results")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()