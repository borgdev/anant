#!/usr/bin/env python3
"""
Test suite for the refactored Hypergraph implementation
======================================================

This test suite validates that the modular Hypergraph architecture
maintains full functionality while providing better maintainability.
"""

import sys
import traceback
from pathlib import Path

# Add the anant package to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_functionality():
    """Test basic hypergraph operations."""
    print("🧪 Testing basic functionality...")
    
    try:
        from anant.classes.hypergraph_refactored import HypergraphRefactored as Hypergraph
        
        # Test 1: Create empty hypergraph
        hg = Hypergraph()
        assert hg.num_nodes() == 0
        assert hg.num_edges() == 0
        print("✅ Empty hypergraph creation")
        
        # Test 2: Add nodes and edges
        hg.add_edge('e1', ['n1', 'n2', 'n3'])
        hg.add_edge('e2', ['n2', 'n4'])
        
        assert hg.num_nodes() == 4
        assert hg.num_edges() == 2
        assert hg.get_node_degree('n2') == 2  # n2 is in both edges
        assert hg.get_edge_size('e1') == 3
        print("✅ Node and edge operations")
        
        # Test 3: From dict creation
        edge_dict = {
            'project1': ['alice', 'bob', 'charlie'],
            'project2': ['bob', 'diana'],
            'project3': ['charlie', 'diana', 'eve']
        }
        
        hg2 = Hypergraph.from_dict(edge_dict)
        assert hg2.num_nodes() == 5
        assert hg2.num_edges() == 3
        print("✅ Dictionary creation")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_algorithm_operations():
    """Test algorithm operations."""
    print("🧪 Testing algorithm operations...")
    
    try:
        from anant.classes.hypergraph_refactored import HypergraphRefactored as Hypergraph
        
        # Create a connected hypergraph
        edge_dict = {
            'e1': ['a', 'b'],
            'e2': ['b', 'c'],
            'e3': ['c', 'd']
        }
        
        hg = Hypergraph.from_dict(edge_dict)
        
        # Test shortest path
        path = hg.shortest_path('a', 'd')
        assert path is not None
        assert len(path) >= 3  # At least a->b->c->d
        print("✅ Shortest path algorithm")
        
        # Test connectivity
        assert hg.is_connected()
        components = hg.connected_components()
        assert len(components) == 1
        print("✅ Connectivity analysis")
        
        # Test neighbors
        neighbors_b = hg.neighbors('b')
        assert 'a' in neighbors_b
        assert 'c' in neighbors_b
        print("✅ Neighbor queries")
        
        return True
        
    except Exception as e:
        print(f"❌ Algorithm operations test failed: {e}")
        traceback.print_exc()
        return False

def test_centrality_operations():
    """Test centrality operations."""
    print("🧪 Testing centrality operations...")
    
    try:
        from anant.classes.hypergraph_refactored import HypergraphRefactored as Hypergraph
        
        # Create a star-like hypergraph where one node is central
        edge_dict = {
            'e1': ['center', 'a'],
            'e2': ['center', 'b'],
            'e3': ['center', 'c'],
            'e4': ['a', 'b']  # Additional connection
        }
        
        hg = Hypergraph.from_dict(edge_dict)
        
        # Test degree centrality
        degree_centrality = hg.centrality_ops.degree_centrality()
        assert 'center' in degree_centrality
        assert degree_centrality['center'] > degree_centrality['a']
        print("✅ Degree centrality")
        
        # Test betweenness centrality
        betweenness = hg.betweenness_centrality()
        assert 'center' in betweenness
        print("✅ Betweenness centrality")
        
        # Test closeness centrality
        closeness = hg.closeness_centrality()
        assert 'center' in closeness
        print("✅ Closeness centrality")
        
        return True
        
    except Exception as e:
        print(f"❌ Centrality operations test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_operations():
    """Test performance operations."""
    print("🧪 Testing performance operations...")
    
    try:
        from anant.classes.hypergraph_refactored import HypergraphRefactored as Hypergraph
        
        # Create hypergraph with some data
        hg = Hypergraph.from_dict({
            'e1': ['n1', 'n2'],
            'e2': ['n2', 'n3'],
            'e3': ['n3', 'n4']
        })
        
        # Test index building
        hg._build_performance_indexes()
        assert hg._indexes_built
        print("✅ Index building")
        
        # Test performance stats
        stats = hg.get_performance_stats()
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        print("✅ Performance statistics")
        
        # Test batch operations
        degrees = hg.performance_ops.get_multiple_node_degrees(['n1', 'n2', 'n3'])
        assert len(degrees) == 3
        print("✅ Batch operations")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance operations test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Hypergraph Refactoring Tests")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_algorithm_operations,
        test_centrality_operations,
        test_performance_operations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Hypergraph refactoring successful.")
        return 0
    else:
        print("💥 Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)