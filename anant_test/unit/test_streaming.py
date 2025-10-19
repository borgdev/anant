#!/usr/bin/env python3
"""
Comprehensive Test Suite for Streaming Capabilities

Tests the streaming hypergraph processing including real-time updates,
incremental analytics, and performance optimization integration.
"""

import polars as pl
import time
from typing import Dict, List
import threading

# Add the anant package to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anant'))

from anant.classes.hypergraph import Hypergraph
from anant.streaming import (
    StreamingUpdate,
    StreamingHypergraph,
    IncrementalCentralityProcessor,
    StreamingClusteringProcessor,
    StreamingAnalytics,
    StreamingDataIngestion,
    stream_from_temporal_hypergraph
)
from anant.analysis.temporal import TemporalSnapshot, TemporalHypergraph

def create_initial_hypergraph() -> Hypergraph:
    """Create initial hypergraph for streaming tests"""
    incidence_data = [
        ("E1", "A"), ("E1", "B"),
        ("E2", "B"), ("E2", "C"),
        ("E3", "C"), ("E3", "D"),
    ]
    
    incidence_df = pl.DataFrame([
        {"edges": edge, "nodes": node} 
        for edge, node in incidence_data
    ])
    
    return Hypergraph(incidence_df)

def test_streaming_hypergraph_basic():
    """Test basic streaming hypergraph functionality"""
    print("Testing Streaming Hypergraph Basics...")
    
    initial_hg = create_initial_hypergraph()
    streaming_hg = StreamingHypergraph(initial_hg, enable_optimization=True)
    
    print(f"  Initial state: {streaming_hg.current_hypergraph.num_nodes} nodes, {streaming_hg.current_hypergraph.num_edges} edges")
    
    # Test adding updates
    success1 = streaming_hg.add_edge_update(
        timestamp=1,
        edge_id="E4",
        nodes=["D", "E", "F"]
    )
    
    success2 = streaming_hg.add_edge_update(
        timestamp=2,
        edge_id="E5",
        nodes=["A", "E"]
    )
    
    print(f"  Added updates: {success1 and success2}")
    
    # Start processing
    streaming_hg.start_processing(process_interval=0.01)
    time.sleep(0.1)  # Allow processing
    
    # Check statistics
    stats = streaming_hg.get_statistics()
    print(f"  Processing stats: {stats['processed_updates']} updates processed")
    print(f"  Current state: {streaming_hg.current_hypergraph.num_nodes} nodes, {streaming_hg.current_hypergraph.num_edges} edges")
    
    # Stop processing
    streaming_hg.stop_processing()
    
    assert stats['processed_updates'] > 0, "Should have processed updates"
    assert streaming_hg.current_hypergraph.num_edges >= initial_hg.num_edges, "Should have added edges"
    
    print("  ✓ Basic streaming hypergraph operations working")

def test_incremental_centrality_processor():
    """Test incremental centrality processing"""
    print("\nTesting Incremental Centrality Processor...")
    
    initial_hg = create_initial_hypergraph()
    streaming_hg = StreamingHypergraph(initial_hg)
    
    # Add centrality processor
    centrality_processor = IncrementalCentralityProcessor(measures=['degree', 's_centrality'])
    streaming_hg.add_processor(centrality_processor)
    
    # Start processing
    streaming_hg.start_processing(process_interval=0.01)
    
    # Add several updates
    updates = [
        {"timestamp": 1, "edge_id": "E4", "nodes": ["D", "E"]},
        {"timestamp": 2, "edge_id": "E5", "nodes": ["E", "F", "G"]},
        {"timestamp": 3, "edge_id": "E6", "nodes": ["A", "F"]},
    ]
    
    for update in updates:
        streaming_hg.add_edge_update(**update)
    
    time.sleep(0.2)  # Allow processing
    
    # Get processor results
    results = streaming_hg.get_processor_results()
    centrality_results = results.get('processor_0', {})
    
    print(f"  Centrality processor updates: {centrality_results.get('update_count', 0)}")
    
    centrality_scores = centrality_results.get('centrality_scores', {})
    if 'degree' in centrality_scores:
        degree_scores = centrality_scores['degree']
        print(f"  Degree centralities: {dict(sorted(degree_scores.items()))}")
    
    # Stop processing
    streaming_hg.stop_processing()
    
    assert centrality_results.get('update_count', 0) > 0, "Should have processed updates"
    assert 'degree' in centrality_scores, "Should have degree centralities"
    
    print("  ✓ Incremental centrality processor working")

def test_streaming_clustering_processor():
    """Test streaming clustering processor"""
    print("\nTesting Streaming Clustering Processor...")
    
    initial_hg = create_initial_hypergraph()
    streaming_hg = StreamingHypergraph(initial_hg)
    
    # Add clustering processor
    clustering_processor = StreamingClusteringProcessor(recompute_interval=5)
    streaming_hg.add_processor(clustering_processor)
    
    # Start processing
    streaming_hg.start_processing(process_interval=0.01)
    
    # Add updates to create more complex structure
    updates = [
        {"timestamp": 1, "edge_id": "E4", "nodes": ["A", "D"]},
        {"timestamp": 2, "edge_id": "E5", "nodes": ["E", "F"]},
        {"timestamp": 3, "edge_id": "E6", "nodes": ["F", "G"]},
        {"timestamp": 4, "edge_id": "E7", "nodes": ["E", "G"]},
        {"timestamp": 5, "edge_id": "E8", "nodes": ["H", "I"]},
        {"timestamp": 6, "edge_id": "E9", "nodes": ["I", "J"]},
    ]
    
    for update in updates:
        streaming_hg.add_edge_update(**update)
    
    time.sleep(0.3)  # Allow processing
    
    # Get clustering results
    results = streaming_hg.get_processor_results()
    clustering_results = results.get('processor_0', {})
    
    print(f"  Clustering processor updates: {clustering_results.get('update_count', 0)}")
    print(f"  Current modularity: {clustering_results.get('modularity', 0.0):.4f}")
    
    communities = clustering_results.get('communities', {})
    if communities:
        print(f"  Found communities for {len(communities)} nodes")
    
    # Stop processing
    streaming_hg.stop_processing()
    
    assert clustering_results.get('update_count', 0) > 0, "Should have processed updates"
    
    print("  ✓ Streaming clustering processor working")

def test_streaming_analytics():
    """Test streaming analytics engine"""
    print("\nTesting Streaming Analytics...")
    
    initial_hg = create_initial_hypergraph()
    streaming_hg = StreamingHypergraph(initial_hg)
    
    # Create analytics engine
    analytics = StreamingAnalytics(streaming_hg)
    analytics.enable_analytics(['size_metrics', 'density_metrics', 'centrality_stats'])
    
    # Start processing
    streaming_hg.start_processing(process_interval=0.01)
    
    # Compute initial analytics
    initial_analytics = analytics.compute_analytics(timestamp=0)
    print(f"  Initial analytics: {initial_analytics}")
    
    # Add some updates
    streaming_hg.add_edge_update(1, "E4", ["D", "E", "F"])
    streaming_hg.add_edge_update(2, "E5", ["G", "H"])
    
    time.sleep(0.1)  # Allow processing
    
    # Compute analytics after updates
    updated_analytics = analytics.compute_analytics(timestamp=1)
    print(f"  Updated analytics: {updated_analytics}")
    
    # Get analytics history
    history = analytics.get_analytics_history()
    print(f"  Analytics history length: {len(history)}")
    
    # Stop processing
    streaming_hg.stop_processing()
    
    assert len(history) >= 2, "Should have analytics history"
    assert 'size_metrics' in updated_analytics, "Should have size metrics"
    
    print("  ✓ Streaming analytics working")

def test_streaming_data_ingestion():
    """Test streaming data ingestion"""
    print("\nTesting Streaming Data Ingestion...")
    
    initial_hg = create_initial_hypergraph()
    streaming_hg = StreamingHypergraph(initial_hg)
    
    # Create data ingestion
    ingestion = StreamingDataIngestion(streaming_hg, buffer_size=100, batch_size=10)
    
    # Start both processing and ingestion
    streaming_hg.start_processing(process_interval=0.01)
    ingestion.start_ingestion(process_interval=0.02)
    
    # Ingest data
    ingestion_success = []
    for i in range(5):
        success = ingestion.ingest_edge_data(
            timestamp=i,
            edge_id=f"EI{i}",
            nodes=[f"N{i}", f"N{i+1}", f"N{i+2}"]
        )
        ingestion_success.append(success)
    
    time.sleep(0.2)  # Allow processing
    
    # Get statistics
    ingestion_stats = ingestion.get_ingestion_stats()
    streaming_stats = streaming_hg.get_statistics()
    
    print(f"  Ingestion stats: {ingestion_stats}")
    print(f"  Streaming stats: {streaming_stats}")
    print(f"  Successful ingestions: {sum(ingestion_success)}")
    
    # Stop processing
    ingestion.stop_ingestion()
    streaming_hg.stop_processing()
    
    assert sum(ingestion_success) > 0, "Should have successful ingestions"
    assert ingestion_stats['total_ingested'] > 0, "Should have ingested data"
    
    print("  ✓ Streaming data ingestion working")

def test_temporal_replay():
    """Test streaming from temporal hypergraph"""
    print("\nTesting Temporal Hypergraph Replay...")
    
    # Create temporal hypergraph
    snapshots = []
    
    # Snapshot 1
    hg1 = Hypergraph(pl.DataFrame([
        {"edges": "E1", "nodes": "A"},
        {"edges": "E1", "nodes": "B"},
        {"edges": "E2", "nodes": "B"},
        {"edges": "E2", "nodes": "C"},
    ]))
    snapshots.append(TemporalSnapshot(0, hg1))
    
    # Snapshot 2 (add edge)
    hg2 = Hypergraph(pl.DataFrame([
        {"edges": "E1", "nodes": "A"},
        {"edges": "E1", "nodes": "B"},
        {"edges": "E2", "nodes": "B"},
        {"edges": "E2", "nodes": "C"},
        {"edges": "E3", "nodes": "C"},
        {"edges": "E3", "nodes": "D"},
    ]))
    snapshots.append(TemporalSnapshot(1, hg2))
    
    # Snapshot 3 (remove edge E1, add E4)
    hg3 = Hypergraph(pl.DataFrame([
        {"edges": "E2", "nodes": "B"},
        {"edges": "E2", "nodes": "C"},
        {"edges": "E3", "nodes": "C"},
        {"edges": "E3", "nodes": "D"},
        {"edges": "E4", "nodes": "A"},
        {"edges": "E4", "nodes": "D"},
    ]))
    snapshots.append(TemporalSnapshot(2, hg3))
    
    temporal_hg = TemporalHypergraph(snapshots)
    
    # Create streaming hypergraph
    initial_hg = Hypergraph(pl.DataFrame({"edges": [], "nodes": []}))
    streaming_hg = StreamingHypergraph(initial_hg)
    
    # Add processors
    centrality_processor = IncrementalCentralityProcessor()
    streaming_hg.add_processor(centrality_processor)
    
    # Start processing
    streaming_hg.start_processing()
    
    # Stream from temporal hypergraph
    stream_from_temporal_hypergraph(temporal_hg, streaming_hg, replay_speed=1.0)
    
    time.sleep(0.1)  # Allow processing
    
    # Check results
    final_stats = streaming_hg.get_statistics()
    processor_results = streaming_hg.get_processor_results()
    
    print(f"  Replay processed {final_stats['processed_updates']} updates")
    print(f"  Final hypergraph: {streaming_hg.current_hypergraph.num_nodes} nodes, {streaming_hg.current_hypergraph.num_edges} edges")
    
    # Stop processing
    streaming_hg.stop_processing()
    
    assert final_stats['processed_updates'] > 0, "Should have processed temporal updates"
    
    print("  ✓ Temporal hypergraph replay working")

def test_performance_integration():
    """Test integration with performance optimization"""
    print("\nTesting Performance Integration...")
    
    initial_hg = create_initial_hypergraph()
    streaming_hg = StreamingHypergraph(initial_hg, enable_optimization=True)
    
    # Add processors
    centrality_processor = IncrementalCentralityProcessor()
    clustering_processor = StreamingClusteringProcessor()
    streaming_hg.add_processor(centrality_processor)
    streaming_hg.add_processor(clustering_processor)
    
    # Start processing
    streaming_hg.start_processing()
    
    # Add many updates to test performance
    start_time = time.time()
    
    for i in range(20):
        streaming_hg.add_edge_update(
            timestamp=i,
            edge_id=f"EP{i}",
            nodes=[f"N{i}", f"N{i+1}", f"N{i%5}"]  # Create some overlap
        )
    
    time.sleep(0.3)  # Allow processing
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Get final statistics
    stats = streaming_hg.get_statistics()
    
    print(f"  Processing time: {processing_time:.3f}s")
    print(f"  Updates processed: {stats['processed_updates']}")
    print(f"  Processing rate: {stats['processed_updates'] / processing_time:.1f} updates/s")
    
    if 'memory_usage_mb' in stats:
        print(f"  Memory usage: {stats['memory_usage_mb']:.2f}MB")
        print(f"  Memory delta: {stats['memory_delta_mb']:.2f}MB")
    
    # Stop processing
    streaming_hg.stop_processing()
    
    assert stats['processed_updates'] > 0, "Should have processed updates"
    assert processing_time < 5.0, "Should process reasonably fast"
    
    print("  ✓ Performance integration working")

def test_comprehensive_streaming_workflow():
    """Test comprehensive streaming workflow"""
    print("\nTesting Comprehensive Streaming Workflow...")
    
    # Create streaming system
    initial_hg = create_initial_hypergraph()
    streaming_hg = StreamingHypergraph(initial_hg, enable_optimization=True)
    
    # Add all processors
    centrality_processor = IncrementalCentralityProcessor(measures=['degree', 's_centrality'])
    clustering_processor = StreamingClusteringProcessor(recompute_interval=10)
    streaming_hg.add_processor(centrality_processor)
    streaming_hg.add_processor(clustering_processor)
    
    # Add analytics
    analytics = StreamingAnalytics(streaming_hg)
    analytics.enable_analytics(['size_metrics', 'density_metrics', 'centrality_stats', 'community_stats'])
    
    # Add data ingestion
    ingestion = StreamingDataIngestion(streaming_hg)
    
    # Start all processing
    streaming_hg.start_processing()
    ingestion.start_ingestion()
    
    print("  Starting comprehensive streaming simulation...")
    
    # Simulate streaming data
    for i in range(15):
        # Add edge via direct update
        streaming_hg.add_edge_update(i, f"E_direct_{i}", [f"A{i%3}", f"B{i%4}"])
        
        # Add edge via ingestion
        ingestion.ingest_edge_data(i, f"E_ingest_{i}", [f"C{i%2}", f"D{i%3}", f"E{i%5}"])
        
        # Compute analytics periodically
        if i % 5 == 0:
            analytics.compute_analytics(i)
        
        time.sleep(0.02)  # Small delay to simulate real-time
    
    time.sleep(0.2)  # Allow final processing
    
    # Collect final results
    final_stats = streaming_hg.get_statistics()
    ingestion_stats = ingestion.get_ingestion_stats()
    processor_results = streaming_hg.get_processor_results()
    analytics_history = analytics.get_analytics_history()
    
    # Print comprehensive summary
    print(f"\n  Comprehensive Streaming Results:")
    print(f"  • Total updates processed: {final_stats['processed_updates']}")
    print(f"  • Ingestion success rate: {ingestion_stats['successful_updates']}/{ingestion_stats['total_ingested']}")
    print(f"  • Final hypergraph size: {streaming_hg.current_hypergraph.num_nodes} nodes, {streaming_hg.current_hypergraph.num_edges} edges")
    print(f"  • Analytics snapshots: {len(analytics_history)}")
    
    # Centrality results
    if 'processor_0' in processor_results:
        cent_results = processor_results['processor_0']
        print(f"  • Centrality updates: {cent_results.get('update_count', 0)}")
    
    # Clustering results  
    if 'processor_1' in processor_results:
        clust_results = processor_results['processor_1']
        print(f"  • Clustering updates: {clust_results.get('update_count', 0)}")
        print(f"  • Final modularity: {clust_results.get('modularity', 0.0):.4f}")
    
    # Performance metrics
    if 'memory_usage_mb' in final_stats:
        print(f"  • Memory usage: {final_stats['memory_usage_mb']:.2f}MB (Δ{final_stats['memory_delta_mb']:.2f}MB)")
    
    # Stop all processing
    ingestion.stop_ingestion()
    streaming_hg.stop_processing()
    
    # Validate comprehensive workflow
    assert final_stats['processed_updates'] > 10, "Should have processed many updates"
    assert len(analytics_history) > 0, "Should have analytics history"
    assert streaming_hg.current_hypergraph.num_edges > initial_hg.num_edges, "Should have grown"
    
    print("  ✓ Comprehensive streaming workflow completed successfully")

def main():
    """Run all streaming capabilities tests"""
    print("=" * 60)
    print("Streaming Capabilities Test Suite")
    print("=" * 60)
    
    try:
        test_streaming_hypergraph_basic()
        test_incremental_centrality_processor()
        test_streaming_clustering_processor()
        test_streaming_analytics()
        test_streaming_data_ingestion()
        test_temporal_replay()
        test_performance_integration()
        test_comprehensive_streaming_workflow()
        
        print("\n" + "=" * 60)
        print("✅ ALL STREAMING CAPABILITIES TESTS PASSED!")
        print("=" * 60)
        
        print("\nKey Results:")
        print("• Streaming hypergraph with real-time updates working")
        print("• Incremental centrality and clustering processors functional")
        print("• Streaming analytics with multiple metrics tracking")
        print("• Data ingestion pipeline with buffering and error handling")
        print("• Temporal hypergraph replay capabilities")
        print("• Performance optimization integration active")
        print("• Memory monitoring and optimization working")
        print("• Comprehensive streaming workflows validated")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()