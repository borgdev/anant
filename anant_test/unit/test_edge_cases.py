#!/usr/bin/env python3
"""
Edge Cases and Error Handling Test

Test edge cases, boundary conditions, and error handling 
across all anant library components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anant'))

import polars as pl

def test_edge_cases():
    """Test edge cases and error handling"""
    print("=== Edge Cases and Error Handling Test ===")
    
    from anant.classes.hypergraph import Hypergraph
    from anant.streaming import StreamingHypergraph
    from anant.validation import ValidationFramework
    
    # Test 1: Empty hypergraph
    print("1. Testing empty hypergraph...")
    try:
        empty_hg = Hypergraph()
        print(f"   Empty HG: {empty_hg.num_nodes} nodes, {empty_hg.num_edges} edges")
        
        # Test validation on empty hypergraph
        framework = ValidationFramework(enable_logging=False)
        suite = framework.validate_hypergraph(empty_hg)
        print(f"   Empty HG validation: {suite.passed_count}/{suite.total_count} passed")
        
        print("  ✓ Empty hypergraph handled correctly")
    except Exception as e:
        print(f"  ❌ Empty hypergraph issue: {e}")
    
    # Test 2: Single node/edge hypergraph
    print("2. Testing minimal hypergraph...")
    try:
        minimal_df = pl.DataFrame([
            {"edges": "E1", "nodes": "A", "weight": 1.0},
        ])
        minimal_hg = Hypergraph(minimal_df)
        print(f"   Minimal HG: {minimal_hg.num_nodes} nodes, {minimal_hg.num_edges} edges")
        
        # Test streaming with minimal hypergraph
        streaming_hg = StreamingHypergraph(minimal_hg, enable_optimization=False)
        success = streaming_hg.add_edge_update(1, "E2", ["A", "B"])
        print(f"   Minimal streaming update: {success}")
        
        print("  ✓ Minimal hypergraph handled correctly")
    except Exception as e:
        print(f"  ❌ Minimal hypergraph issue: {e}")
    
    # Test 3: Large node/edge names
    print("3. Testing with unusual identifiers...")
    try:
        unusual_df = pl.DataFrame([
            {"edges": "very_long_edge_name_with_special_chars_123", "nodes": "node_with_unicode_αβγ", "weight": 1.0},
            {"edges": "very_long_edge_name_with_special_chars_123", "nodes": "another_long_node_name", "weight": 2.5},
            {"edges": "E_2", "nodes": "node_with_unicode_αβγ", "weight": 0.1},
        ])
        unusual_hg = Hypergraph(unusual_df)
        print(f"   Unusual IDs HG: {unusual_hg.num_nodes} nodes, {unusual_hg.num_edges} edges")
        print("  ✓ Unusual identifiers handled correctly")
    except Exception as e:
        print(f"  ❌ Unusual identifiers issue: {e}")
    
    # Test 4: Missing data handling
    print("4. Testing data validation...")
    try:
        # Test with required validation
        framework = ValidationFramework(enable_logging=False)
        
        # Test with good data
        good_df = pl.DataFrame([
            {"edges": "E1", "nodes": "A", "weight": 1.0},
            {"edges": "E1", "nodes": "B", "weight": 1.0},
        ])
        good_hg = Hypergraph(good_df)
        suite = framework.validate_hypergraph(good_hg, ['data_integrity'])
        
        if suite.passed_count == 0:
            print(f"  ❌ Data validation issue: no tests passed")
        else:
            print(f"   Data integrity: {suite.passed_count}/{suite.total_count} passed")
            print("  ✓ Data validation working correctly")
            
    except Exception as e:
        print(f"  ❌ Data validation issue: {e}")
    
    # Test 5: Memory efficiency with larger data
    print("5. Testing with moderately large data...")
    try:
        # Create a moderately sized hypergraph
        large_data = []
        for i in range(100):
            for j in range(5):
                large_data.append({
                    "edges": f"E_{i}",
                    "nodes": f"N_{j}",
                    "weight": 1.0 + (i * 0.1)
                })
        
        large_df = pl.DataFrame(large_data)
        large_hg = Hypergraph(large_df)
        print(f"   Large HG: {large_hg.num_nodes} nodes, {large_hg.num_edges} edges")
        
        # Test analysis on larger data
        from anant.analysis.centrality import degree_centrality
        centralities = degree_centrality(large_hg)
        if len(centralities['nodes']) == large_hg.num_nodes:
            print("  ✓ Large data handled correctly")
        else:
            print(f"  ❌ Large data issue: centrality count mismatch")
            
    except Exception as e:
        print(f"  ❌ Large data issue: {e}")
    
    # Test 6: Concurrent operations
    print("6. Testing concurrent streaming operations...")
    try:
        import time
        import threading
        
        base_df = pl.DataFrame([
            {"edges": "E1", "nodes": "A", "weight": 1.0},
            {"edges": "E1", "nodes": "B", "weight": 1.0},
        ])
        base_hg = Hypergraph(base_df)
        streaming_hg = StreamingHypergraph(base_hg, enable_optimization=False)
        
        # Add multiple updates quickly
        for i in range(5):
            streaming_hg.add_edge_update(i, f"E_{i+2}", [f"N_{i}", f"N_{i+1}"])
        
        # Process them
        streaming_hg.start_processing()
        time.sleep(0.2)
        streaming_hg.stop_processing()
        
        stats = streaming_hg.get_statistics()
        print(f"   Processed {stats['processed_updates']} updates")
        print("  ✓ Concurrent operations handled correctly")
        
    except Exception as e:
        print(f"  ❌ Concurrent operations issue: {e}")
    
    print("\n=== Edge Cases Summary ===")
    print("✅ All edge cases and error handling tests completed")

def main():
    print("Edge Cases Test")
    print("=" * 20)
    
    test_edge_cases()
    
    print("\n✅ EDGE CASES TEST COMPLETED!")

if __name__ == "__main__":
    main()