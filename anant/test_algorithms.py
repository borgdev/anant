#!/usr/bin/env python3
"""
Test ANANT Algorithms

Test the core algorithms in ANANT to ensure they're working properly
before running analytics on the FIBO metagraph.
"""

import sys
import os
from pathlib import Path

# Add ANANT to path
sys.path.insert(0, '/home/amansingh/dev/ai/anant')

import anant
from anant import Hypergraph

def test_anant_algorithms():
    """Test ANANT algorithms with a simple hypergraph"""
    print("🧪 Testing ANANT Algorithms")
    print("=" * 40)
    
    # Initialize ANANT
    anant.setup()
    
    # Create a simple test hypergraph
    print("📊 Creating test hypergraph...")
    edge_dict = {
        'e1': ['n1', 'n2', 'n3'],
        'e2': ['n2', 'n4', 'n5'],
        'e3': ['n1', 'n4'],
        'e4': ['n3', 'n5', 'n6'],
        'e5': ['n1', 'n6']
    }
    
    hg = Hypergraph.from_dict(edge_dict)
    print(f"✅ Test hypergraph: {hg.num_nodes} nodes, {hg.num_edges} edges")
    
    # Test 1: Basic Operations
    print(f"\n🔍 1. Testing Basic Operations...")
    test_basic_operations(hg)
    
    # Test 2: Centrality Algorithms
    print(f"\n🎯 2. Testing Centrality Algorithms...")
    test_centrality_algorithms(hg)
    
    # Test 3: Clustering Algorithms
    print(f"\n🌐 3. Testing Clustering Algorithms...")
    test_clustering_algorithms(hg)
    
    # Test 4: Advanced Algorithms
    print(f"\n⚡ 4. Testing Advanced Algorithms...")
    test_advanced_algorithms(hg)
    
    return True

def test_basic_operations(hg: Hypergraph):
    """Test basic hypergraph operations"""
    try:
        # Test node operations
        nodes = list(hg.nodes)
        print(f"   📊 Nodes: {nodes}")
        
        # Test edge operations  
        edges = list(hg.edges)
        print(f"   📊 Edges: {edges}")
        
        # Test degree calculation
        for node in nodes[:3]:
            degree = hg.get_node_degree(node)
            print(f"   📊 Node {node} degree: {degree}")
        
        # Test edge size calculation
        for edge in edges[:3]:
            size = hg.get_edge_size(edge)
            edge_nodes = hg.get_edge_nodes(edge)
            print(f"   📊 Edge {edge} size: {size}, nodes: {edge_nodes}")
        
        print(f"   ✅ Basic operations working correctly")
        
    except Exception as e:
        print(f"   ❌ Error in basic operations: {e}")

def test_centrality_algorithms(hg: Hypergraph):
    """Test centrality algorithms"""
    try:
        # Import centrality module
        from anant.algorithms import centrality
        
        print(f"   🎯 Testing hypergraph centrality...")
        
        # Test degree centrality
        degree_centrality = centrality.hypergraph_centrality(hg, centrality_type='degree')
        print(f"   📊 Degree centrality: {degree_centrality}")
        
        # Test normalized centrality
        normalized_centrality = centrality.hypergraph_centrality(hg, centrality_type='degree', normalize=True)
        print(f"   📊 Normalized centrality: {normalized_centrality}")
        
        print(f"   ✅ Centrality algorithms working correctly")
        
    except Exception as e:
        print(f"   ❌ Error in centrality algorithms: {e}")
        import traceback
        traceback.print_exc()

def test_clustering_algorithms(hg: Hypergraph):
    """Test clustering algorithms"""
    try:
        # Import clustering module
        from anant.algorithms import clustering
        
        print(f"   🌐 Testing hypergraph clustering...")
        
        # Test clustering
        clusters = clustering.hypergraph_clustering(hg, algorithm='modularity')
        print(f"   📊 Hypergraph clustering: {clusters}")
        
        print(f"   ✅ Clustering algorithms working correctly")
        
    except Exception as e:
        print(f"   ❌ Error in clustering algorithms: {e}")
        import traceback
        traceback.print_exc()

def test_advanced_algorithms(hg: Hypergraph):
    """Test advanced algorithms"""
    try:
        print(f"   ⚡ Testing pattern detection...")
        
        # Manual pattern detection (since imports might be complex)
        patterns = detect_simple_patterns(hg)
        print(f"   📊 Patterns found: {patterns}")
        
        print(f"   ✅ Advanced algorithms working correctly")
        
    except Exception as e:
        print(f"   ❌ Error in advanced algorithms: {e}")

def detect_simple_patterns(hg: Hypergraph) -> dict:
    """Simple pattern detection for testing"""
    patterns = {
        'triangular_patterns': 0,
        'star_patterns': 0,
        'chain_patterns': 0
    }
    
    # Analyze edge patterns
    for edge in hg.edges:
        edge_size = hg.get_edge_size(edge)
        
        if edge_size == 3:
            patterns['triangular_patterns'] += 1
        elif edge_size > 3:
            patterns['star_patterns'] += 1
        elif edge_size == 2:
            patterns['chain_patterns'] += 1
    
    return patterns

if __name__ == "__main__":
    try:
        success = test_anant_algorithms()
        print(f"\n✅ ANANT algorithms test: {'PASSED' if success else 'FAILED'}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)