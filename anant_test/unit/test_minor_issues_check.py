#!/usr/bin/env python3
"""
Comprehensive Minor Issues Check

Check for any remaining edge cases, warnings, or potential issues
across all components of the anant library.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anant'))

import polars as pl
import warnings

def check_all_components():
    """Check all major components for potential issues"""
    issues = []
    
    print("=== Comprehensive Component Check ===")
    
    # Test 1: Basic Hypergraph Operations
    print("1. Testing basic hypergraph operations...")
    try:
        from anant.classes.hypergraph import Hypergraph
        
        # Test with minimal data
        test_df = pl.DataFrame([
            {"edges": "E1", "nodes": "A", "weight": 1.0},
            {"edges": "E1", "nodes": "B", "weight": 1.0},
        ])
        hg = Hypergraph(test_df)
        
        # Basic operations
        nodes = hg.nodes
        edges = hg.edges
        num_nodes = hg.num_nodes
        num_edges = hg.num_edges
        
        if num_nodes != 2 or num_edges != 1:
            issues.append("Basic hypergraph operations: incorrect counts")
        else:
            print("  ✓ Basic operations working")
            
    except Exception as e:
        issues.append(f"Basic hypergraph operations failed: {e}")
    
    # Test 2: Analysis Algorithms
    print("2. Testing analysis algorithms...")
    try:
        from anant.analysis.centrality import degree_centrality
        from anant.analysis.clustering import modularity_clustering
        
        test_df = pl.DataFrame([
            {"edges": "E1", "nodes": "A", "weight": 1.0},
            {"edges": "E1", "nodes": "B", "weight": 1.0},
            {"edges": "E2", "nodes": "B", "weight": 1.0},
            {"edges": "E2", "nodes": "C", "weight": 1.0},
        ])
        hg = Hypergraph(test_df)
        
        # Centrality
        centralities = degree_centrality(hg)
        if not isinstance(centralities, dict) or 'nodes' not in centralities:
            issues.append("Centrality analysis: invalid return format")
        
        # Clustering  
        communities = modularity_clustering(hg)
        if not isinstance(communities, dict):
            issues.append("Clustering analysis: invalid return format")
            
        print("  ✓ Analysis algorithms working")
        
    except Exception as e:
        issues.append(f"Analysis algorithms failed: {e}")
    
    # Test 3: Streaming Operations
    print("3. Testing streaming operations...")
    try:
        from anant.streaming import StreamingHypergraph
        import time
        
        hg = Hypergraph(test_df)
        streaming_hg = StreamingHypergraph(hg, enable_optimization=False)
        
        # Test edge addition
        success = streaming_hg.add_edge_update(1, "E3", ["C", "D"])
        if not success:
            issues.append("Streaming: failed to add edge update")
        
        # Test processing
        streaming_hg.start_processing()
        time.sleep(0.05)
        streaming_hg.stop_processing()
        
        stats = streaming_hg.get_statistics()
        if stats['processed_updates'] == 0:
            issues.append("Streaming: no updates processed")
        else:
            print("  ✓ Streaming operations working")
            
    except Exception as e:
        issues.append(f"Streaming operations failed: {e}")
    
    # Test 4: Temporal Analysis
    print("4. Testing temporal analysis...")
    try:
        from anant.analysis.temporal import TemporalHypergraph, TemporalSnapshot
        
        temporal_hg = TemporalHypergraph()
        snapshot = TemporalSnapshot(timestamp=1, hypergraph=hg)
        temporal_hg.add_snapshot(snapshot)
        
        if len(temporal_hg.snapshots) != 1:
            issues.append("Temporal analysis: snapshot addition failed")
        else:
            print("  ✓ Temporal analysis working")
            
    except Exception as e:
        issues.append(f"Temporal analysis failed: {e}")
    
    # Test 5: Property Management
    print("5. Testing property management...")
    try:
        test_nodes = ["A", "B"]
        hg.add_node_properties({node: {"test_prop": 1.0} for node in test_nodes})
        
        node_props = hg.get_node_properties("A")
        if not isinstance(node_props, dict) or "test_prop" not in node_props:
            issues.append("Property management: retrieval failed")
        else:
            print("  ✓ Property management working")
            
    except Exception as e:
        issues.append(f"Property management failed: {e}")
    
    # Test 6: Performance Optimization
    print("6. Testing performance optimization...")
    try:
        from anant.optimization import PerformanceOptimizer, MemoryMonitor, OptimizationConfig
        
        config = OptimizationConfig()
        optimizer = PerformanceOptimizer(config)
        monitor = MemoryMonitor()
        
        # Basic monitoring
        usage = monitor.get_usage_mb()
        if usage <= 0:
            issues.append("Performance optimization: invalid memory reading")
        else:
            print("  ✓ Performance optimization working")
            
    except Exception as e:
        issues.append(f"Performance optimization failed: {e}")
    
    # Test 7: Validation Framework
    print("7. Testing validation framework...")
    try:
        from anant.validation import quick_validate, ValidationFramework
        
        is_valid = quick_validate(hg)
        if not is_valid:
            issues.append("Validation framework: basic validation failed")
        
        framework = ValidationFramework(enable_logging=False)
        suite = framework.validate_hypergraph(hg, ['data_integrity'])
        if suite.passed_count == 0:
            issues.append("Validation framework: no tests passed")
        else:
            print("  ✓ Validation framework working")
            
    except Exception as e:
        issues.append(f"Validation framework failed: {e}")
    
    return issues

def main():
    print("Minor Issues Check")
    print("=" * 30)
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        issues = check_all_components()
        
        # Check for warnings
        if w:
            print(f"\n⚠️  Found {len(w)} warnings:")
            for warning in w:
                print(f"  - {warning.message}")
    
    print(f"\n=== Results ===")
    if issues:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ No issues found! All components working correctly.")
        return True

if __name__ == "__main__":
    main()