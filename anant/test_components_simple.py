#!/usr/bin/env python3
"""
Simple Component Test for Anant Library

This script tests the basic functionality of implemented components
using their actual API.
"""

import sys
import polars as pl
import numpy as np
from pathlib import Path
import tempfile
import traceback

def test_setsystem_factory():
    """Test SetSystemFactory with actual API"""
    print("🧪 Testing SetSystemFactory...")
    
    try:
        from anant.factory.setsystem_factory import SetSystemFactory
        
        # Test dictionary of iterables (most common use case)
        print("  📋 Testing dictionary of iterables...")
        dict_data = {
            'meeting1': ['Alice', 'Bob', 'Charlie'],
            'meeting2': ['Bob', 'David', 'Eve'], 
            'meeting3': ['Alice', 'Eve', 'Frank']
        }
        df = SetSystemFactory.from_dict_of_iterables(dict_data)
        print(f"    ✅ Created DataFrame with {df.height} incidences")
        print(f"    ✅ Columns: {df.columns}")
        
        # Test DataFrame input
        print("  📋 Testing DataFrame input...")
        input_df = pl.DataFrame({
            'edge_id': ['e1', 'e1', 'e2', 'e2', 'e3'],
            'node_id': ['n1', 'n2', 'n1', 'n3', 'n2']
        })
        df2 = SetSystemFactory.from_dataframe(input_df, edge_col='edge_id', node_col='node_id')
        print(f"    ✅ Created DataFrame with {df2.height} incidences")
        
        return True
        
    except Exception as e:
        print(f"    ❌ SetSystemFactory test failed: {e}")
        traceback.print_exc()
        return False

def test_hypergraph():
    """Test Hypergraph with actual API"""
    print("\n🧪 Testing Hypergraph...")
    
    try:
        from anant.classes.hypergraph import Hypergraph
        from anant.factory.setsystem_factory import SetSystemFactory
        
        # Create test data
        print("  📋 Creating hypergraph...")
        edge_data = {
            'meeting1': ['Alice', 'Bob', 'Charlie'],
            'meeting2': ['Bob', 'David', 'Eve'],
            'meeting3': ['Alice', 'Eve', 'Frank']
        }
        
        # Create setsystem and hypergraph
        setsystem_df = SetSystemFactory.from_dict_of_iterables(edge_data)
        hg = Hypergraph(setsystem=setsystem_df)
        print(f"    ✅ Created hypergraph with {hg.num_edges} edges and {hg.num_nodes} nodes")
        
        # Test basic properties
        print("  📋 Testing basic properties...")
        print(f"    ✅ Number of edges: {hg.num_edges}")
        print(f"    ✅ Number of nodes: {hg.num_nodes}")
        print(f"    ✅ Number of incidences: {hg.num_incidences}")
        
        # Test edge and node lists
        edges = hg.edges
        nodes = hg.nodes
        print(f"    ✅ Edge list: {list(edges)[:3]}...")
        print(f"    ✅ Node list: {list(nodes)[:3]}...")
        
        # Test node properties
        print("  📋 Testing node properties...")
        node_props = pl.DataFrame({
            'uid': list(nodes),
            'department': ['Engineering', 'Sales', 'Engineering', 'Marketing', 'Sales', 'HR'][:len(nodes)],
            'experience': list(range(len(nodes)))
        })
        hg.add_node_properties(node_props)
        print("    ✅ Added node properties successfully")
        
        # Test edge properties  
        print("  📋 Testing edge properties...")
        edge_props = pl.DataFrame({
            'uid': list(edges),
            'room': ['A101', 'B202', 'A101'][:len(edges)],
            'duration': [60, 90, 45][:len(edges)]
        })
        hg.add_edge_properties(edge_props)
        print("    ✅ Added edge properties successfully")
        
        # Test statistics
        print("  📋 Testing statistics...")
        stats = hg.get_statistics()
        print(f"    ✅ Statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Hypergraph test failed: {e}")
        traceback.print_exc()
        return False

def test_parquet_io():
    """Test parquet I/O operations"""
    print("\n🧪 Testing Parquet I/O...")
    
    try:
        from anant.classes.hypergraph import Hypergraph
        from anant.factory.setsystem_factory import SetSystemFactory
        from anant.io.parquet_io import AnantIO
        
        # Create test hypergraph
        print("  📋 Creating test hypergraph...")
        edge_data = {
            'team_alpha': ['Alice', 'Bob', 'Charlie'],
            'team_beta': ['David', 'Eve', 'Frank']
        }
        setsystem_df = SetSystemFactory.from_dict_of_iterables(edge_data)
        hg = Hypergraph(setsystem=setsystem_df)
        
        # Add some properties
        node_props = pl.DataFrame({
            'uid': list(hg.nodes),
            'skill_level': [8, 6, 9, 7, 8, 5][:len(hg.nodes)]
        })
        hg.add_node_properties(node_props)
        
        # Test saving to parquet
        print("  📋 Testing parquet save...")
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_hypergraph"
            
            # Save with snappy compression
            print("    📁 Saving with snappy compression...")
            AnantIO.save_hypergraph_parquet(hg, save_path, compression="snappy")
            
            # Verify files were created
            files = list(save_path.glob("*.parquet"))
            print(f"    ✅ Created {len(files)} parquet files")
            
            # Test loading
            print("    📁 Loading from parquet...")
            loaded_hg = AnantIO.load_hypergraph_parquet(save_path)
            
            # Verify loaded data
            print(f"    ✅ Loaded hypergraph: {loaded_hg.num_edges} edges, {loaded_hg.num_nodes} nodes")
            
            # Basic verification
            assert loaded_hg.num_edges == hg.num_edges
            assert loaded_hg.num_nodes == hg.num_nodes
            print("    ✅ Data integrity verified")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Parquet I/O test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_benchmark():
    """Test performance benchmarking framework"""
    print("\n🧪 Testing Performance Benchmark...")
    
    try:
        from anant.utils.benchmarks import PerformanceBenchmark
        from anant.classes.hypergraph import Hypergraph
        from anant.factory.setsystem_factory import SetSystemFactory
        
        print("  📋 Creating test datasets...")
        
        # Create simple test datasets
        small_data = {
            'e1': ['n1', 'n2', 'n3'],
            'e2': ['n2', 'n3', 'n4'],
            'e3': ['n1', 'n4', 'n5']
        }
        
        medium_data = {}
        for i in range(20):
            nodes = [f"node_{j}" for j in range(i, i+4)]
            medium_data[f"edge_{i}"] = nodes
        
        # Test benchmark initialization
        print("  📋 Initializing benchmark...")
        benchmark = PerformanceBenchmark()
        print("    ✅ Benchmark framework initialized")
        
        # Test simple hypergraph construction timing
        print("  📋 Testing construction timing...")
        import time
        
        start_time = time.time()
        setsystem_df = SetSystemFactory.from_dict_of_iterables(medium_data)
        hg = Hypergraph(setsystem=setsystem_df)
        construction_time = time.time() - start_time
        
        print(f"    ✅ Construction time: {construction_time:.4f}s")
        print(f"    ✅ Hypergraph size: {hg.num_edges} edges, {hg.num_nodes} nodes")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Performance benchmark test failed: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test basic integration between components"""
    print("\n🧪 Testing Basic Integration...")
    
    try:
        from anant.classes.hypergraph import Hypergraph
        from anant.factory.setsystem_factory import SetSystemFactory
        from anant.io.parquet_io import AnantIO
        
        print("  📋 Creating integrated workflow...")
        
        # Create dataset using factory
        friendship_data = {
            'group_1': ['Alice', 'Bob', 'Charlie'],
            'group_2': ['David', 'Eve', 'Frank'],
            'group_3': ['Alice', 'Eve', 'Grace']
        }
        
        # Create hypergraph
        setsystem_df = SetSystemFactory.from_dict_of_iterables(friendship_data)
        hg = Hypergraph(setsystem=setsystem_df)
        print(f"    ✅ Created hypergraph: {hg.num_edges} groups, {hg.num_nodes} people")
        
        # Add properties
        node_props = pl.DataFrame({
            'uid': list(hg.nodes),
            'age': [25, 30, 28, 35, 26, 32, 29][:len(hg.nodes)]
        })
        hg.add_node_properties(node_props)
        
        edge_props = pl.DataFrame({
            'uid': list(hg.edges),
            'formation_date': ['2023-01-15', '2023-03-22', '2023-05-10'][:len(hg.edges)]
        })
        hg.add_edge_properties(edge_props)
        
        # Test save/load cycle
        print("  📋 Testing save/load cycle...")
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "integration_test"
            
            # Save
            AnantIO.save_hypergraph_parquet(hg, save_path, compression='snappy')
            
            # Load
            loaded_hg = AnantIO.load_hypergraph_parquet(save_path)
            
            # Verify
            assert loaded_hg.num_edges == hg.num_edges
            assert loaded_hg.num_nodes == hg.num_nodes
            
            print(f"    ✅ Save/load cycle successful")
            print(f"    ✅ Original: {hg.num_edges} edges, {hg.num_nodes} nodes")
            print(f"    ✅ Loaded: {loaded_hg.num_edges} edges, {loaded_hg.num_nodes} nodes")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all component tests"""
    print("🚀 Anant Library Component Test Suite")
    print("=" * 50)
    
    tests = [
        ("SetSystem Factory", test_setsystem_factory),
        ("Hypergraph Class", test_hypergraph),
        ("Parquet I/O", test_parquet_io),
        ("Performance Benchmark", test_performance_benchmark),
        ("Basic Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {name} - PASSED")
            else:
                print(f"❌ {name} - FAILED")
        except Exception as e:
            print(f"❌ {name} - ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All component tests passed! Anant library is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())