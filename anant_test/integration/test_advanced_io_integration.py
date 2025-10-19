#!/usr/bin/env python3
"""
Comprehensive Test Suite for Advanced I/O and Integration Features

Tests all components of the Advanced I/O system:
- Enhanced file format support
- Database connectivity
- Streaming data processing  
- Data transformation utilities
"""

import asyncio
import tempfile
import json
import csv
from pathlib import Path
import polars as pl
import sys
import os

# Add anant to path
sys.path.insert(0, os.path.dirname(__file__))

# Import our Advanced I/O modules
from anant.io import (
    # Enhanced formats
    quick_import, quick_export, EnhancedFileFormats, ImportExportConfig,
    
    # Database connectivity
    create_sqlite_manager, DatabaseConfig,
    
    # Streaming processing
    create_streaming_hypergraph, StreamEvent, StreamEventType,
    
    # Data transformation
    quick_clean, quick_quality_check, quick_convert,
    assess_data_quality, DataCleaner, HypergraphConverter
)

from anant.classes.hypergraph import Hypergraph


async def test_enhanced_file_formats():
    """Test enhanced file format support"""
    print("🧪 Testing Enhanced File Formats...")
    
    # Create test hypergraph
    hg = Hypergraph()
    hg.add_edge("e1", ["n1", "n2", "n3"], weight=1.5)
    hg.add_edge("e2", ["n2", "n3", "n4"], weight=2.0)
    hg.add_edge("e3", ["n1", "n4", "n5"], weight=1.0)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test JSON format
        json_file = temp_path / "test.json"
        quick_export(hg, json_file)
        loaded_hg = quick_import(json_file)
        assert loaded_hg.num_nodes == hg.num_nodes
        assert loaded_hg.num_edges == hg.num_edges
        print("✅ JSON format: Export/Import successful")
        
        # Test CSV format
        csv_file = temp_path / "test.csv"
        quick_export(hg, csv_file, csv_format="incidences")
        loaded_hg = quick_import(csv_file, csv_format="incidences")
        assert loaded_hg.num_nodes == hg.num_nodes
        assert loaded_hg.num_edges == hg.num_edges
        print("✅ CSV format: Export/Import successful")
        
        # Test GraphML format
        graphml_file = temp_path / "test.graphml"
        quick_export(hg, graphml_file)
        print("✅ GraphML format: Export successful")
        
        # Test configuration
        config = ImportExportConfig(include_weights=True, validate_schema=True)
        formats = EnhancedFileFormats(config)
        enhanced_json = temp_path / "enhanced.json"
        formats.export(hg, enhanced_json)
        loaded_enhanced = formats.import_file(enhanced_json)
        assert loaded_enhanced.num_nodes == hg.num_nodes
        print("✅ Enhanced formats with configuration: Successful")


async def test_database_connectivity():
    """Test database connectivity"""
    print("\n🗄️ Testing Database Connectivity...")
    
    # Clear any global state before starting
    from anant.classes.hypergraph import Hypergraph as FreshHypergraph
    FreshHypergraph.clear_global_cache()
    
    # Create test hypergraph - use a completely fresh instance
    hg = FreshHypergraph()
    print(f"Initial fresh hypergraph nodes: {list(hg.nodes)}")
    print(f"Initial fresh hypergraph edges: {list(hg.edges)}")
    
    # Ensure the hypergraph is truly empty
    assert len(hg.nodes) == 0, f"Fresh hypergraph should have 0 nodes, got {len(hg.nodes)}"
    assert len(list(hg.edges)) == 0, f"Fresh hypergraph should have 0 edges, got {len(list(hg.edges))}"
    
    hg.add_edge("db_e1", ["db_n1", "db_n2", "db_n3"])
    print(f"After adding db_e1: {list(hg.nodes)}")
    hg.add_edge("db_e2", ["db_n2", "db_n3", "db_n4"])
    
    print(f"Original hypergraph nodes: {list(hg.nodes)}")
    print(f"Original hypergraph edges: {list(hg.edges)}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_hypergraphs.db"
        
        # Test SQLite connectivity
        db_manager = create_sqlite_manager(db_path)
        
        async with db_manager:
            # Save hypergraph
            saved_name = await db_manager.save(hg, "test_hypergraph", {"version": 1})
            assert saved_name == "test_hypergraph"
            print("✅ SQLite: Save successful")
            
            # Load hypergraph
            loaded_hg = await db_manager.load("test_hypergraph")
            print(f"Original: {hg.num_nodes} nodes, {hg.num_edges} edges")
            print(f"Loaded: {loaded_hg.num_nodes} nodes, {loaded_hg.num_edges} edges")
            print(f"Loaded hypergraph nodes: {list(loaded_hg.nodes)}")
            print(f"Loaded hypergraph edges: {list(loaded_hg.edges)}")
            
            # For now, just check that edges are correct and loaded nodes are the expected ones
            assert loaded_hg.num_edges == hg.num_edges, "Edge count should match"
            expected_nodes = {'db_n1', 'db_n2', 'db_n3', 'db_n4'}
            loaded_nodes = set(loaded_hg.nodes)
            assert loaded_nodes == expected_nodes, f"Expected nodes {expected_nodes}, got {loaded_nodes}"
            print("✅ SQLite: Load successful")
            
            # List hypergraphs
            hg_list = await db_manager.list()
            assert len(hg_list) == 1
            assert hg_list[0]["name"] == "test_hypergraph"
            print("✅ SQLite: List successful")
            
            # Delete hypergraph
            deleted = await db_manager.delete("test_hypergraph")
            assert deleted == True
            print("✅ SQLite: Delete successful")


async def test_streaming_processing():
    """Test streaming data processing"""
    print("\n🌊 Testing Streaming Processing...")
    
    # Create streaming hypergraph
    processor = await create_streaming_hypergraph()
    
    # Track updates
    updates_received = []
    
    def update_callback(event, hg):
        updates_received.append({
            "event_type": event.event_type.value,
            "nodes": hg.num_nodes,
            "edges": hg.num_edges
        })
    
    processor.subscribe(update_callback)
    
    # Test adding edges
    add_event = StreamEvent(
        event_type=StreamEventType.ADD_EDGE,
        data={
            "edge_id": "stream_e1",
            "nodes": ["stream_n1", "stream_n2", "stream_n3"],
            "weight": 1.5
        }
    )
    await processor.add_event(add_event)
    
    # Brief wait for processing
    await asyncio.sleep(0.1)
    
    assert processor.hypergraph.num_edges == 1
    assert processor.hypergraph.num_nodes == 3
    print("✅ Streaming: Edge addition successful")
    
    # Test removing edges
    remove_event = StreamEvent(
        event_type=StreamEventType.REMOVE_EDGE,
        data={"edge_id": "stream_e1"}
    )
    await processor.add_event(remove_event)
    
    await asyncio.sleep(0.1)
    
    assert processor.hypergraph.num_edges == 0
    print("✅ Streaming: Edge removal successful")
    
    # Test metrics
    metrics = processor.get_metrics()
    assert metrics["events_processed"] >= 2
    print("✅ Streaming: Metrics tracking successful")
    
    await processor.stop()


def test_data_transformation():
    """Test data transformation utilities"""
    print("\n🔄 Testing Data Transformation...")
    
    # Create test dataset with quality issues
    test_data = pl.DataFrame({
        "transaction_id": ["t1", "t2", "t3", "t4", "t1"],  # Duplicate
        "item_id": ["item1", "item2", None, "item3", "item1"],  # Missing value
        "quantity": ["1", "2", "3", "invalid", "1"],  # Type inconsistency (all strings)
        "price": [10.5, 20.0, 15.5, 25.0, 10.5]
    })
    
    # Test data quality assessment
    quality_report = quick_quality_check(test_data)
    assert quality_report.total_records == 5
    assert quality_report.overall_score < 100  # Should detect issues
    print(f"✅ Data Quality: Assessment successful (Score: {quality_report.overall_score:.1f})")
    
    # Test data cleaning
    cleaned_data = quick_clean(test_data)
    assert len(cleaned_data) <= len(test_data)  # Should remove/fix issues
    print("✅ Data Cleaning: Successful")
    
    # Create valid transaction data for conversion
    transaction_data = pl.DataFrame({
        "transaction_id": ["t1", "t1", "t1", "t2", "t2", "t3"],
        "item_id": ["item1", "item2", "item3", "item1", "item4", "item2"]
    })
    
    # Test hypergraph conversion
    hg = quick_convert(
        transaction_data, 
        strategy="transaction",
        transaction_col="transaction_id",
        item_col="item_id"
    )
    
    assert hg.num_edges == 3  # Three transactions
    assert hg.num_nodes >= 4  # At least 4 unique items
    print("✅ Hypergraph Conversion: Successful")
    
    # Test feature engineering
    from anant.io.data_transformation import FeatureEngineer
    engineer = FeatureEngineer()
    
    node_features = engineer.create_node_features(hg)
    edge_features = engineer.create_edge_features(hg)
    global_features = engineer.create_global_features(hg)
    
    assert len(node_features) == hg.num_nodes
    assert len(edge_features) == hg.num_edges
    assert isinstance(global_features, dict)
    print("✅ Feature Engineering: Successful")


def test_integration_workflow():
    """Test complete integration workflow"""
    print("\n🔗 Testing Integration Workflow...")
    
    # Create sample data file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample CSV data
        csv_file = temp_path / "sample_data.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "item_id", "rating", "timestamp"])
            writer.writerow(["u1", "i1", "5", "2023-01-01"])
            writer.writerow(["u1", "i2", "4", "2023-01-02"])
            writer.writerow(["u2", "i1", "3", "2023-01-03"])
            writer.writerow(["u2", "i3", "5", "2023-01-04"])
            writer.writerow(["u3", "i2", "4", "2023-01-05"])
        
        # Complete workflow: Load -> Clean -> Convert -> Export
        
        # 1. Load and assess quality
        raw_data = pl.read_csv(csv_file)
        quality_report = quick_quality_check(raw_data)
        print(f"✅ Workflow Step 1: Data loaded and assessed (Score: {quality_report.overall_score:.1f})")
        
        # 2. Clean data
        clean_data = quick_clean(raw_data)
        print("✅ Workflow Step 2: Data cleaned")
        
        # 3. Convert to hypergraph (users as nodes, items as hyperedges)
        hg = quick_convert(
            clean_data,
            strategy="bipartite",
            source_col="user_id",
            target_col="item_id"
        )
        print(f"✅ Workflow Step 3: Hypergraph created ({hg.num_nodes} nodes, {hg.num_edges} edges)")
        
        # 4. Export to multiple formats
        json_output = temp_path / "output.json"
        csv_output = temp_path / "output.csv"
        
        quick_export(hg, json_output)
        quick_export(hg, csv_output, csv_format="incidences")
        
        assert json_output.exists()
        assert csv_output.exists()
        print("✅ Workflow Step 4: Multi-format export successful")
        
        # 5. Verify round-trip consistency
        loaded_hg = quick_import(json_output)
        assert loaded_hg.num_nodes == hg.num_nodes
        assert loaded_hg.num_edges == hg.num_edges
        print("✅ Workflow Step 5: Round-trip consistency verified")


async def run_all_tests():
    """Run all Advanced I/O tests"""
    print("🚀 Starting Advanced I/O & Integration Test Suite")
    print("=" * 60)
    
    # Clear any global state before starting tests
    from anant.classes.hypergraph import Hypergraph
    Hypergraph.clear_global_cache()
    
    try:
        # Test each component
        await test_enhanced_file_formats()
        await test_database_connectivity()
        await test_streaming_processing()
        test_data_transformation()
        test_integration_workflow()
        
        print("\n" + "=" * 60)
        print("🎉 All Advanced I/O & Integration tests passed!")
        print("✅ Enhanced file format support: Working")
        print("✅ Database connectivity: Working")
        print("✅ Streaming data processing: Working")
        print("✅ Data transformation utilities: Working")
        print("✅ End-to-end integration workflow: Working")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n🎯 Advanced I/O and Integration features are fully functional!")
        print("🚀 Ready for production use with comprehensive I/O capabilities.")
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")