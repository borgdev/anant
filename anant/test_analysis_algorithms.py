#!/usr/bin/env python3
"""
Anant Analysis Algorithms Test Suite

Test all the advanced analysis algorithms including centrality,
clustering, structural analysis, and spectral methods.
"""

import sys
import numpy as np
import polars as pl
from pathlib import Path
import traceback

def test_centrality_algorithms():
    """Test centrality measures"""
    print("🧪 Testing Centrality Algorithms...")
    
    try:
        from anant import Hypergraph, SetSystemFactory
        from anant.analysis.centrality import (
            degree_centrality, node_degree_centrality, edge_degree_centrality,
            closeness_centrality, betweenness_centrality
        )
        
        # Create test hypergraph
        edge_data = {
            'group1': ['Alice', 'Bob', 'Charlie'],
            'group2': ['Bob', 'David', 'Eve'],
            'group3': ['Alice', 'Eve', 'Frank'],
            'group4': ['Charlie', 'David']
        }
        
        setsystem = SetSystemFactory.from_dict_of_iterables(edge_data)
        hg = Hypergraph(setsystem=setsystem)
        
        print(f"  📊 Test hypergraph: {hg.num_nodes} nodes, {hg.num_edges} edges")
        
        # Test degree centrality
        print("  🔢 Testing degree centrality...")
        centrality = degree_centrality(hg)
        assert 'nodes' in centrality and 'edges' in centrality
        print(f"    ✅ Node centrality (top 3): {dict(list(centrality['nodes'].items())[:3])}")
        print(f"    ✅ Edge centrality (top 3): {dict(list(centrality['edges'].items())[:3])}")
        
        # Test node degree centrality
        print("  🔢 Testing node degree centrality...")
        node_cent = node_degree_centrality(hg, normalized=True)
        assert len(node_cent) == hg.num_nodes
        print(f"    ✅ Normalized node centrality computed for {len(node_cent)} nodes")
        
        # Test edge degree centrality
        print("  🔢 Testing edge degree centrality...")
        edge_cent = edge_degree_centrality(hg, normalized=True)
        assert len(edge_cent) == hg.num_edges
        print(f"    ✅ Normalized edge centrality computed for {len(edge_cent)} edges")
        
        # Test closeness centrality
        print("  📏 Testing closeness centrality...")
        close_cent = closeness_centrality(hg)
        assert len(close_cent) == hg.num_nodes
        print(f"    ✅ Closeness centrality computed for {len(close_cent)} nodes")
        print(f"    ✅ Sample values: {dict(list(close_cent.items())[:3])}")
        
        # Test betweenness centrality (with sampling for speed)
        print("  🌉 Testing betweenness centrality...")
        between_cent = betweenness_centrality(hg, sample_size=3)
        assert len(between_cent) == hg.num_nodes
        print(f"    ✅ Betweenness centrality computed for {len(between_cent)} nodes")
        print(f"    ✅ Sample values: {dict(list(between_cent.items())[:3])}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Centrality test failed: {e}")
        traceback.print_exc()
        return False


def test_clustering_algorithms():
    """Test clustering and community detection"""
    print("\n🧪 Testing Clustering Algorithms...")
    
    try:
        from anant import Hypergraph, SetSystemFactory
        from anant.analysis.clustering import (
            modularity_clustering, community_detection, evaluate_clustering
        )
        
        # Create test hypergraph with clear community structure
        edge_data = {
            # Community 1
            'team_a1': ['Alice', 'Bob', 'Charlie'],
            'team_a2': ['Alice', 'Bob', 'Diana'],
            'team_a3': ['Charlie', 'Diana'],
            
            # Community 2
            'team_b1': ['Eve', 'Frank', 'Grace'],
            'team_b2': ['Eve', 'Henry', 'Grace'],
            'team_b3': ['Frank', 'Henry'],
            
            # Bridge
            'bridge': ['Diana', 'Eve']
        }
        
        setsystem = SetSystemFactory.from_dict_of_iterables(edge_data)
        hg = Hypergraph(setsystem=setsystem)
        
        print(f"  📊 Test hypergraph: {hg.num_nodes} nodes, {hg.num_edges} edges")
        
        # Test modularity clustering
        print("  🏘️ Testing modularity clustering...")
        mod_clusters = modularity_clustering(hg, resolution=1.0)
        assert len(mod_clusters) == hg.num_nodes
        n_communities = len(set(mod_clusters.values()))
        print(f"    ✅ Found {n_communities} communities using modularity")
        print(f"    ✅ Sample assignment: {dict(list(mod_clusters.items())[:4])}")
        
        # Test unified community detection interface
        print("  🔍 Testing community detection interface...")
        communities = community_detection(hg, method="modularity")
        assert len(communities) == hg.num_nodes
        print(f"    ✅ Unified interface returned {len(set(communities.values()))} communities")
        
        # Test clustering evaluation
        print("  📈 Testing clustering evaluation...")
        eval_results = evaluate_clustering(hg, communities)
        assert 'n_clusters' in eval_results
        assert 'modularity_score' in eval_results
        print(f"    ✅ Evaluation metrics: {len(eval_results)} measures computed")
        print(f"    ✅ Modularity score: {eval_results['modularity_score']:.3f}")
        print(f"    ✅ Within-cluster edge ratio: {eval_results['within_cluster_edge_ratio']:.3f}")
        
        # Test with scikit-learn algorithms (if available)
        try:
            print("  🔬 Testing spectral clustering...")
            from anant.analysis.clustering import spectral_clustering
            spec_clusters = spectral_clustering(hg, n_clusters=2)
            print(f"    ✅ Spectral clustering found {len(set(spec_clusters.values()))} clusters")
        except ImportError:
            print("    ⚠️ Scikit-learn not available, skipping spectral clustering")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Clustering test failed: {e}")
        traceback.print_exc()
        return False


def test_structural_analysis():
    """Test structural analysis algorithms"""
    print("\n🧪 Testing Structural Analysis...")
    
    try:
        from anant import Hypergraph, SetSystemFactory
        from anant.analysis.structural import (
            connected_components, hypergraph_diameter, clustering_coefficient,
            global_clustering_coefficient, structural_summary
        )
        
        # Create test hypergraph with known structure
        edge_data = {
            # Connected component 1
            'comp1_e1': ['A', 'B', 'C'],
            'comp1_e2': ['B', 'C', 'D'],
            'comp1_e3': ['A', 'D'],
            
            # Connected component 2 (isolated)
            'comp2_e1': ['X', 'Y', 'Z'],
            'comp2_e2': ['Y', 'Z']
        }
        
        setsystem = SetSystemFactory.from_dict_of_iterables(edge_data)
        hg = Hypergraph(setsystem=setsystem)
        
        print(f"  📊 Test hypergraph: {hg.num_nodes} nodes, {hg.num_edges} edges")
        
        # Test connected components
        print("  🔗 Testing connected components...")
        components = connected_components(hg)
        print(f"    ✅ Found {len(components)} connected components")
        print(f"    ✅ Component sizes: {[len(comp) for comp in components]}")
        
        # Test diameter (should be None for disconnected graph)
        print("  📏 Testing hypergraph diameter...")
        diameter = hypergraph_diameter(hg)
        print(f"    ✅ Diameter: {diameter} (None for disconnected graph)")
        
        # Test with connected subgraph
        print("  📐 Testing on connected component...")
        connected_edges = {'e1': ['A', 'B', 'C'], 'e2': ['B', 'C', 'D'], 'e3': ['A', 'D']}
        connected_setsystem = SetSystemFactory.from_dict_of_iterables(connected_edges)
        connected_hg = Hypergraph(setsystem=connected_setsystem)
        
        connected_diameter = hypergraph_diameter(connected_hg)
        print(f"    ✅ Connected diameter: {connected_diameter}")
        
        # Test clustering coefficient
        print("  🕸️ Testing clustering coefficient...")
        clustering_coeffs = clustering_coefficient(connected_hg)
        if isinstance(clustering_coeffs, dict):
            assert len(clustering_coeffs) == connected_hg.num_nodes
            print(f"    ✅ Local clustering coefficients computed for {len(clustering_coeffs)} nodes")
        else:
            print(f"    ✅ Global clustering coefficient: {clustering_coeffs:.3f}")
        
        global_clustering = global_clustering_coefficient(connected_hg)
        print(f"    ✅ Global clustering coefficient: {global_clustering:.3f}")
        
        # Test structural summary
        print("  📋 Testing structural summary...")
        summary = structural_summary(hg)
        assert 'num_nodes' in summary
        assert 'num_edges' in summary
        assert 'density' in summary
        print(f"    ✅ Structural summary contains {len(summary)} metrics")
        print(f"    ✅ Is connected: {summary['is_connected']}")
        print(f"    ✅ Density: {summary['density']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Structural analysis test failed: {e}")
        traceback.print_exc()
        return False


def test_spectral_analysis():
    """Test spectral analysis algorithms"""
    print("\n🧪 Testing Spectral Analysis...")
    
    try:
        from anant import Hypergraph, SetSystemFactory
        from anant.analysis.spectral import (
            node_laplacian, edge_laplacian, laplacian_spectrum,
            algebraic_connectivity, spectral_embedding
        )
        
        # Create test hypergraph
        edge_data = {
            'edge1': ['A', 'B', 'C'],
            'edge2': ['B', 'C', 'D'],
            'edge3': ['C', 'D', 'E'],
            'edge4': ['A', 'E']
        }
        
        setsystem = SetSystemFactory.from_dict_of_iterables(edge_data)
        hg = Hypergraph(setsystem=setsystem)
        
        print(f"  📊 Test hypergraph: {hg.num_nodes} nodes, {hg.num_edges} edges")
        
        # Test node Laplacian
        print("  🔢 Testing node Laplacian...")
        node_L = node_laplacian(hg, normalized=True)
        assert node_L.shape == (hg.num_nodes, hg.num_nodes)
        print(f"    ✅ Node Laplacian shape: {node_L.shape}")
        print(f"    ✅ Laplacian is symmetric: {np.allclose(node_L, node_L.T)}")
        
        # Test edge Laplacian
        print("  🔢 Testing edge Laplacian...")
        edge_L = edge_laplacian(hg, normalized=True)
        assert edge_L.shape == (hg.num_edges, hg.num_edges)
        print(f"    ✅ Edge Laplacian shape: {edge_L.shape}")
        print(f"    ✅ Laplacian is symmetric: {np.allclose(edge_L, edge_L.T)}")
        
        # Test Laplacian spectrum
        print("  🌈 Testing Laplacian spectrum...")
        eigenvalues, eigenvectors = laplacian_spectrum(node_L, k=3)
        assert len(eigenvalues) <= 3
        print(f"    ✅ Computed {len(eigenvalues)} eigenvalues")
        print(f"    ✅ Smallest eigenvalues: {eigenvalues[:3]}")
        
        # Test algebraic connectivity
        print("  🔗 Testing algebraic connectivity...")
        alg_conn = algebraic_connectivity(hg)
        print(f"    ✅ Algebraic connectivity: {alg_conn:.4f}")
        
        # Test spectral embedding
        print("  🗺️ Testing spectral embedding...")
        node_embedding = spectral_embedding(hg, n_components=2, method="node")
        assert node_embedding.shape == (hg.num_nodes, 2)
        print(f"    ✅ Node embedding shape: {node_embedding.shape}")
        
        edge_embedding = spectral_embedding(hg, n_components=2, method="edge")
        assert edge_embedding.shape == (hg.num_edges, 2)
        print(f"    ✅ Edge embedding shape: {edge_embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Spectral analysis test failed: {e}")
        traceback.print_exc()
        return False


def test_analysis_integration():
    """Test integration between different analysis modules"""
    print("\n🧪 Testing Analysis Integration...")
    
    try:
        from anant import Hypergraph, SetSystemFactory
        from anant.analysis import degree_centrality
        from anant.analysis.clustering import community_detection
        from anant.analysis.structural import structural_summary
        from anant.analysis.spectral import algebraic_connectivity
        
        # Create comprehensive test hypergraph
        edge_data = {
            'research_team': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'dev_team': ['Bob', 'Eve', 'Frank'],
            'design_team': ['Charlie', 'Grace', 'Henry'],
            'leadership': ['Alice', 'Eve', 'Grace'],
            'project_alpha': ['Alice', 'Bob', 'Frank'],
            'project_beta': ['Diana', 'Grace', 'Henry']
        }
        
        setsystem = SetSystemFactory.from_dict_of_iterables(edge_data)
        hg = Hypergraph(setsystem=setsystem)
        
        print(f"  📊 Integration test hypergraph: {hg.num_nodes} nodes, {hg.num_edges} edges")
        
        # Run multiple analyses
        print("  🔄 Running integrated analysis workflow...")
        
        # Centrality analysis
        centrality = degree_centrality(hg)
        most_central_node = max(centrality['nodes'].items(), key=lambda x: x[1])
        print(f"    ✅ Most central node: {most_central_node[0]} (centrality: {most_central_node[1]:.3f})")
        
        # Community detection
        communities = community_detection(hg, method="modularity")
        n_communities = len(set(communities.values()))
        print(f"    ✅ Detected {n_communities} communities")
        
        # Structural analysis
        structure = structural_summary(hg)
        print(f"    ✅ Structural analysis: density={structure['density']:.3f}, connected={structure['is_connected']}")
        
        # Spectral analysis
        connectivity = algebraic_connectivity(hg)
        print(f"    ✅ Algebraic connectivity: {connectivity:.4f}")
        
        # Cross-validate results
        print("  🔍 Cross-validating analysis results...")
        
        # Check that most central node is in largest community
        largest_community = max(set(communities.values()), key=lambda x: list(communities.values()).count(x))
        central_node_community = communities[most_central_node[0]]
        print(f"    ✅ Central node community check: {central_node_community == largest_community}")
        
        # Check connectivity consistency
        is_connected = structure['is_connected']
        has_positive_connectivity = connectivity > 0
        print(f"    ✅ Connectivity consistency: {is_connected == has_positive_connectivity}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all analysis algorithm tests"""
    print("🚀 Anant Analysis Algorithms Test Suite")
    print("=" * 60)
    
    tests = [
        ("Centrality Algorithms", test_centrality_algorithms),
        ("Clustering Algorithms", test_clustering_algorithms),
        ("Structural Analysis", test_structural_analysis),
        ("Spectral Analysis", test_spectral_analysis),
        ("Analysis Integration", test_analysis_integration)
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
    
    print("\n" + "=" * 60)
    print(f"📊 Analysis Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All analysis algorithm tests passed! Advanced features working correctly.")
        return 0
    else:
        print("⚠️  Some analysis tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())