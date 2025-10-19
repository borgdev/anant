#!/usr/bin/env python3
"""
Simple Test for Streaming Capabilities

A minimal test to verify streaming functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anant'))

import polars as pl
import time

def test_basic_streaming():
    """Test basic streaming functionality"""
    print("Testing Basic Streaming...")
    
    # Import streaming components
    from anant.streaming import StreamingUpdate, StreamingHypergraph
    from anant.classes.hypergraph import Hypergraph
    
    # Create initial hypergraph
    initial_df = pl.DataFrame([
        {"edges": "E1", "nodes": "A", "weight": 1.0},
        {"edges": "E1", "nodes": "B", "weight": 1.0},
        {"edges": "E2", "nodes": "B", "weight": 1.0},
        {"edges": "E2", "nodes": "C", "weight": 1.0},
    ])
    
    initial_hg = Hypergraph(initial_df)
    print(f"  Initial: {initial_hg.num_nodes} nodes, {initial_hg.num_edges} edges")
    
    # Create streaming hypergraph
    streaming_hg = StreamingHypergraph(initial_hg, enable_optimization=False)
    
    # Add an update
    success = streaming_hg.add_edge_update(
        timestamp=1,
        edge_id="E3",
        nodes=["C", "D"]
    )
    print(f"  Added update: {success}")
    
    # Start and stop processing quickly
    streaming_hg.start_processing()
    time.sleep(0.1)
    streaming_hg.stop_processing()
    
    # Check results
    stats = streaming_hg.get_statistics()
    final_hg = streaming_hg.current_hypergraph
    
    print(f"  Final: {final_hg.num_nodes} nodes, {final_hg.num_edges} edges")
    print(f"  Processed: {stats['processed_updates']} updates")
    
    assert stats['processed_updates'] > 0
    print("  ✓ Basic streaming test passed")

def main():
    print("Simple Streaming Test")
    print("=" * 30)
    
    try:
        test_basic_streaming()
        print("\n✅ SIMPLE STREAMING TEST PASSED!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()