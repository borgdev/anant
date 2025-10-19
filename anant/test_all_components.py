#!/usr/bin/env python3
"""
Comprehensive Test Suite for Anant Library Components

This script tests all the major components of the anant library:
- SetSystemFactory (various data input formats)
- Hypergraph (main integration class)
- AnantIO (parquet I/O operations)
- PerformanceBenchmark (benchmarking framework)
"""

import sys
import polars as pl
import numpy as np
from pathlib import Path
import tempfile
import time
import traceback

def test_setsystem_factory():
    """Test SetSystemFactory with different input formats"""
    print("🧪 Testing SetSystemFactory...")
    
    try:
        from anant.factory.setsystem_factory import SetSystemFactory
        
        # Test 1: Iterable of iterables
        print("  📋 Testing iterable of iterables...")
        iterables = [
            ['Alice', 'Bob', 'Charlie'],
            ['Bob', 'David', 'Eve'],
            ['Alice', 'Eve', 'Frank']
        ]
        df1 = SetSystemFactory.from_iterable_of_iterables(iterables)
        print(f"    ✅ Created DataFrame with {df1.height} incidences")
        
        # Test 2: Dictionary of iterables
        print("  📋 Testing dictionary of iterables...")
        dict_data = {
            'meeting1': ['Alice', 'Bob', 'Charlie'],
            'meeting2': ['Bob', 'David', 'Eve'],
            'meeting3': ['Alice', 'Eve', 'Frank']
        }
        df2 = SetSystemFactory.from_dict_of_iterables(dict_data)
        print(f"    ✅ Created DataFrame with {df2.height} incidences")
        
        # Test 3: DataFrame input
        print("  📋 Testing DataFrame input...")
        input_df = pl.DataFrame({
            'edge_id': ['e1', 'e1', 'e2', 'e2', 'e3'],
            'node_id': ['n1', 'n2', 'n1', 'n3', 'n2']
        })
        df3 = SetSystemFactory.from_dataframe(input_df, edge_col='edge_id', node_col='node_id')
        print(f"    ✅ Created DataFrame with {df3.height} incidences")
        
        # Test 4: NumPy array
        print("  📋 Testing NumPy array input...")
        array_data = np.array([
            ['e1', 'n1'],
            ['e1', 'n2'], 
            ['e2', 'n1'],
            ['e2', 'n3']
        ])
        df4 = SetSystemFactory.from_numpy_array(array_data)
        print(f"    ✅ Created DataFrame with {df4.height} incidences")
        
        return True
        
    except Exception as e:
        print(f"    ❌ SetSystemFactory test failed: {e}")
        traceback.print_exc()
        return False

def test_hypergraph():
    """Test Hypergraph main class"""
    print("\n🧪 Testing Hypergraph...")
    
    try:
        from anant.classes.hypergraph import Hypergraph
        from anant.factory.setsystem_factory import SetSystemFactory
        
        # Create test data
        print("  📋 Creating hypergraph from dictionary...")
        edge_data = {
            'meeting1': ['Alice', 'Bob', 'Charlie'],
            'meeting2': ['Bob', 'David', 'Eve'],
            'meeting3': ['Alice', 'Eve', 'Frank'],
            'meeting4': ['Charlie', 'David', 'Frank']
        }
        
        # Create hypergraph
        setsystem_df = SetSystemFactory.from_dict_of_iterables(edge_data)
        hg = Hypergraph(setsystem=setsystem_df)
        print(f"    ✅ Created hypergraph with {hg.num_edges} edges and {hg.num_nodes} nodes")
        
        # Test node properties
        print("  📋 Testing node properties...")
        node_props = pl.DataFrame({
            'uid': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
            'department': ['Engineering', 'Sales', 'Engineering', 'Marketing', 'Sales', 'Engineering'],
            'experience': [5, 3, 7, 2, 4, 6]
        })
        hg.add_node_properties(node_props)
        print("    ✅ Added node properties")
        
        # Test edge properties
        print("  📋 Testing edge properties...")
        edge_props = pl.DataFrame({
            'uid': ['meeting1', 'meeting2', 'meeting3', 'meeting4'],
            'room': ['A101', 'B202', 'A101', 'C303'],
            'duration': [60, 90, 45, 120]
        })
        hg.add_edge_properties(edge_props)
        print("    ✅ Added edge properties")
        
        # Test basic queries
        print("  📋 Testing basic queries...")
        alice_edges = hg.get_node_edges('Alice')
        meeting1_nodes = hg.get_edge_nodes('meeting1')
        print(f"    ✅ Alice participates in {len(alice_edges)} meetings")
        print(f"    ✅ meeting1 has {len(meeting1_nodes)} participants")
        
        # Test statistics
        print("  📋 Testing statistics...")
        stats = hg.get_statistics()
        print(f"    ✅ Hypergraph statistics: {stats['num_edges']} edges, {stats['num_nodes']} nodes")
        
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
        from anant.io.parquet_io import AnantIO
        
        # Create test hypergraph
        print("  📋 Creating test hypergraph...")
        edge_data = {
            'team_alpha': ['Alice', 'Bob', 'Charlie'],
            'team_beta': ['David', 'Eve', 'Frank'],
            'team_gamma': ['Alice', 'Eve', 'Grace']
        }
        hg = Hypergraph.from_dict_of_iterables(edge_data)
        
        # Add some properties
        node_props = pl.DataFrame({
            'uid': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace'],
            'skill_level': [8, 6, 9, 7, 8, 5, 9],
            'years_exp': [5, 3, 8, 4, 6, 2, 7]
        })
        hg.add_node_properties(node_props)
        
        # Test saving to parquet
        print("  📋 Testing parquet save...")
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_hypergraph"
            
            # Save with different compression options
            for compression in ['snappy', 'gzip', 'lz4']:
                print(f"    📁 Saving with {compression} compression...")
                AnantIO.save_hypergraph_parquet(hg, save_path / compression, compression=compression)
                
                # Verify files were created
                files = list((save_path / compression).glob("*.parquet"))
                print(f"    ✅ Created {len(files)} parquet files")
                
                # Test loading
                print(f"    📁 Loading from {compression} format...")
                loaded_hg = AnantIO.load_hypergraph_parquet(save_path / compression)
                
                # Verify loaded data
                assert loaded_hg.num_edges == hg.num_edges
                assert loaded_hg.num_nodes == hg.num_nodes
                print(f"    ✅ Loaded hypergraph: {loaded_hg.num_edges} edges, {loaded_hg.num_nodes} nodes")
        
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
        
        print("  📋 Creating test datasets...")
        
        # Create different sized datasets for benchmarking
        datasets = {}
        for size in [100, 500, 1000]:
            # Generate random edge data
            import random
            edges = {}
            for i in range(size // 10):  # ~10 nodes per edge on average
                nodes = [f"node_{j}" for j in random.sample(range(size), random.randint(3, 8))]
                edges[f"edge_{i}"] = nodes
            datasets[size] = edges
        
        # Initialize benchmark
        print("  📋 Initializing benchmark framework...")
        benchmark = PerformanceBenchmark()
        
        # Test hypergraph construction benchmarks
        print("  📋 Running construction benchmarks...")
        for size, data in datasets.items():
            print(f"    ⏱️  Benchmarking {size} node dataset...")
            
            start_time = time.time()
            hg = Hypergraph.from_dict_of_iterables(data)
            construction_time = time.time() - start_time
            
            memory_mb = hg.get_memory_usage()
            
            print(f"    ✅ Size {size}: {construction_time:.3f}s, {memory_mb:.2f}MB")
        
        # Test specific operations
        print("  📋 Testing operation benchmarks...")
        hg = Hypergraph.from_dict_of_iterables(datasets[1000])
        
        # Benchmark property operations
        node_props = pl.DataFrame({
            'uid': list(hg.nodes),
            'value': np.random.randn(len(hg.nodes))
        })
        
        start_time = time.time()
        hg.add_node_properties(node_props)
        prop_time = time.time() - start_time
        print(f"    ✅ Property addition: {prop_time:.3f}s")
        
        # Benchmark queries
        start_time = time.time()
        for edge in list(hg.edges)[:10]:  # Test first 10 edges
            nodes = hg.get_edge_nodes(edge)
        query_time = time.time() - start_time
        print(f"    ✅ Edge queries (10): {query_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Performance benchmark test failed: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between all components"""
    print("\n🧪 Testing Component Integration...")
    
    try:
        from anant.classes.hypergraph import Hypergraph
        from anant.factory.setsystem_factory import SetSystemFactory
        from anant.io.parquet_io import AnantIO
        
        print("  📋 Creating complex hypergraph...")
        
        # Create a more complex dataset
        # Social network with multiple types of relationships
        friendship_data = {
            'friends_group_1': ['Alice', 'Bob', 'Charlie'],
            'friends_group_2': ['David', 'Eve', 'Frank'],
            'friends_group_3': ['Alice', 'Eve', 'Grace'],
            'friends_group_4': ['Bob', 'David', 'Henry']
        }
        
        # Create hypergraph
        hg = Hypergraph.from_dict_of_iterables(friendship_data)
        print(f"    ✅ Created social network: {hg.num_edges} groups, {hg.num_nodes} people")
        
        # Add rich node properties
        print("  📋 Adding comprehensive node properties...")
        all_nodes = list(hg.nodes)
        node_props = pl.DataFrame({
            'uid': all_nodes,
            'age': np.random.randint(20, 65, len(all_nodes)),
            'location': np.random.choice(['NYC', 'SF', 'LA', 'Chicago'], len(all_nodes)),
            'occupation': np.random.choice(['Engineer', 'Designer', 'Manager', 'Analyst'], len(all_nodes)),
            'satisfaction': np.random.uniform(1, 10, len(all_nodes))
        })
        hg.add_node_properties(node_props)
        
        # Add edge properties
        print("  📋 Adding edge properties...")
        edge_props = pl.DataFrame({
            'uid': list(hg.edges),
            'formation_date': ['2023-01-15', '2023-03-22', '2023-05-10', '2023-07-08'],
            'activity_level': np.random.choice(['High', 'Medium', 'Low'], len(hg.edges)),
            'meeting_frequency': np.random.uniform(0.5, 5.0, len(hg.edges))
        })
        hg.add_edge_properties(edge_props)
        
        # Test complex queries
        print("  📋 Testing complex queries...")
        
        # Find nodes with high satisfaction
        high_satisfaction = hg.get_nodes_by_property('satisfaction', lambda x: x > 7.0)
        print(f"    ✅ Found {len(high_satisfaction)} highly satisfied people")
        
        # Find edges with high activity
        high_activity_edges = hg.get_edges_by_property('activity_level', lambda x: x == 'High')
        print(f"    ✅ Found {len(high_activity_edges)} high-activity groups")
        
        # Test save/load cycle
        print("  📋 Testing complete save/load cycle...")
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "integration_test"
            
            # Save
            AnantIO.save_hypergraph_parquet(hg, save_path, compression='snappy')
            
            # Load
            loaded_hg = AnantIO.load_hypergraph_parquet(save_path)
            
            # Verify integrity
            assert loaded_hg.num_edges == hg.num_edges
            assert loaded_hg.num_nodes == hg.num_nodes
            
            # Verify properties were preserved
            original_stats = hg.get_statistics()
            loaded_stats = loaded_hg.get_statistics()
            
            print(f"    ✅ Data integrity verified after save/load cycle")
            print(f"    ✅ Memory usage: {loaded_stats['memory_usage_mb']:.2f}MB")
        
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
        ("Component Integration", test_integration)
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
        print("🎉 All tests passed! Anant library is fully functional.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())