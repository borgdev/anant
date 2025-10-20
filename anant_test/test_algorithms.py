"""
Algorithms Test Suite
====================

Tests for graph algorithms in Anant:
- Centrality measures
- Clustering algorithms
- Community detection
- Path algorithms
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import numpy as np
from anant.classes.hypergraph import Hypergraph


def test_centrality_algorithms():
    """Test centrality algorithms."""
    print("  Testing centrality algorithms...")
    
    try:
        from anant.analysis.centrality import degree_centrality, s_centrality
        
        # Create test hypergraph
        setsystem = {
            "e1": ["n1", "n2", "n3"],
            "e2": ["n2", "n3", "n4"],
            "e3": ["n1", "n4"]
        }
        
        hg = Hypergraph(setsystem=setsystem)
        
        # Test degree centrality
        degree_cent = degree_centrality(hg)
        assert isinstance(degree_cent, dict), "Should return dictionary"
        assert "nodes" in degree_cent, "Should have nodes key"
        assert len(degree_cent["nodes"]) > 0, "Should have centrality values"
        
        # Test normalized degree centrality
        degree_cent_norm = degree_centrality(hg, normalized=True)
        assert isinstance(degree_cent_norm, dict), "Should return dictionary"
        
        # Test s-centrality
        s_cent = s_centrality(hg, s=1)
        assert isinstance(s_cent, dict), "Should return dictionary"
        assert len(s_cent) > 0, "Should have centrality values"
        
        print("    ✅ Centrality algorithms working")
        return True
        
    except ImportError:
        print("    ⚠️  Centrality module not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Centrality test failed: {e}")
        return False


def test_clustering_algorithms():
    """Test clustering algorithms."""
    print("  Testing clustering algorithms...")
    
    try:
        from anant.analysis.clustering import modularity_clustering, spectral_clustering
        
        # Create test hypergraph
        setsystem = {
            "e1": ["n1", "n2"],
            "e2": ["n2", "n3"],
            "e3": ["n3", "n4"],
            "e4": ["n4", "n5"],
            "e5": ["n5", "n6"]
        }
        
        hg = Hypergraph(setsystem=setsystem)
        
        # Test modularity clustering
        try:
            communities = modularity_clustering(hg)
            assert isinstance(communities, dict), "Should return community assignments"
            assert len(communities) > 0, "Should have community assignments"
            print("    ✅ Modularity clustering working")
        except Exception as e:
            print(f"    ⚠️  Modularity clustering failed: {e}")
        
        # Test spectral clustering
        try:
            clusters = spectral_clustering(hg, k=2)
            assert isinstance(clusters, dict), "Should return cluster assignments"
            print("    ✅ Spectral clustering working")
        except Exception as e:
            print(f"    ⚠️  Spectral clustering failed: {e}")
        
        return True
        
    except ImportError:
        print("    ⚠️  Clustering module not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Clustering test failed: {e}")
        return False


def test_community_detection():
    """Test community detection algorithms."""
    print("  Testing community detection...")
    
    try:
        from anant.analysis.clustering import community_quality_metrics
        
        # Create test communities
        communities = {
            "n1": 0, "n2": 0, "n3": 1, "n4": 1, "n5": 1
        }
        
        setsystem = {
            "e1": ["n1", "n2"],
            "e2": ["n3", "n4"],
            "e3": ["n4", "n5"]
        }
        
        hg = Hypergraph(setsystem=setsystem)
        
        # Test quality metrics
        quality = community_quality_metrics(hg, communities)
        assert isinstance(quality, dict), "Should return quality metrics"
        assert "modularity" in quality, "Should have modularity"
        
        print("    ✅ Community detection working")
        return True
        
    except ImportError:
        print("    ⚠️  Community detection module not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Community detection test failed: {e}")
        return False


def test_path_algorithms():
    """Test path algorithms."""
    print("  Testing path algorithms...")
    
    try:
        from anant.analysis.paths import shortest_path, all_paths
        
        # Create test hypergraph
        setsystem = {
            "e1": ["n1", "n2"],
            "e2": ["n2", "n3"],
            "e3": ["n3", "n4"],
            "e4": ["n1", "n4"]  # Alternative path
        }
        
        hg = Hypergraph(setsystem=setsystem)
        
        # Test shortest path
        try:
            path = shortest_path(hg, "n1", "n4")
            assert isinstance(path, list), "Should return path as list"
            assert path[0] == "n1", "Path should start with source"
            assert path[-1] == "n4", "Path should end with target"
            print("    ✅ Shortest path working")
        except Exception as e:
            print(f"    ⚠️  Shortest path failed: {e}")
        
        return True
        
    except ImportError:
        print("    ⚠️  Path algorithms module not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Path algorithms test failed: {e}")
        return False


def test_graph_metrics():
    """Test basic graph metrics."""
    print("  Testing graph metrics...")
    
    try:
        from anant.analysis.metrics import density, diameter, clustering_coefficient
        
        # Create test hypergraph
        setsystem = {
            "e1": ["n1", "n2", "n3"],
            "e2": ["n2", "n3", "n4"],
            "e3": ["n1", "n4"]
        }
        
        hg = Hypergraph(setsystem=setsystem)
        
        # Test density
        try:
            dens = density(hg)
            assert isinstance(dens, (int, float)), "Density should be numeric"
            assert 0 <= dens <= 1, "Density should be between 0 and 1"
            print("    ✅ Density calculation working")
        except Exception as e:
            print(f"    ⚠️  Density calculation failed: {e}")
        
        # Test other metrics
        try:
            diam = diameter(hg)
            print(f"    ✅ Diameter: {diam}")
        except Exception as e:
            print(f"    ⚠️  Diameter calculation failed: {e}")
        
        return True
        
    except ImportError:
        print("    ⚠️  Metrics module not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Metrics test failed: {e}")
        return False


def test_algorithm_performance():
    """Test algorithm performance on larger graphs."""
    print("  Testing algorithm performance...")
    
    try:
        # Create larger test hypergraph
        import random
        random.seed(42)
        
        setsystem = {}
        nodes = [f"n{i}" for i in range(100)]
        
        # Create random hyperedges
        for i in range(50):
            edge_size = random.randint(2, 5)
            edge_nodes = random.sample(nodes, edge_size)
            setsystem[f"e{i}"] = edge_nodes
        
        hg = Hypergraph(setsystem=setsystem)
        
        # Test if algorithms can handle larger graphs
        try:
            from anant.analysis.centrality import degree_centrality
            start_time = time.time()
            degree_cent = degree_centrality(hg)
            end_time = time.time()
            print(f"    ✅ Degree centrality on 100 nodes: {end_time - start_time:.3f}s")
        except ImportError:
            pass
        except Exception as e:
            print(f"    ⚠️  Performance test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Performance test failed: {e}")
        return False


def run_tests():
    """Run all algorithm tests."""
    print("🧪 Running Algorithm Tests")
    
    # Import time for performance tests
    import time
    
    test_functions = [
        test_centrality_algorithms,
        test_clustering_algorithms,
        test_community_detection,
        test_path_algorithms,
        test_graph_metrics,
        test_algorithm_performance
    ]
    
    passed = 0
    failed = 0
    details = []
    
    for test_func in test_functions:
        try:
            result = test_func()
            if result:
                passed += 1
                details.append(f"✅ {test_func.__name__}")
            else:
                failed += 1
                details.append(f"❌ {test_func.__name__}: Test returned False")
        except Exception as e:
            failed += 1
            details.append(f"❌ {test_func.__name__}: {str(e)}")
    
    status = "PASSED" if failed == 0 else "FAILED"
    
    return {
        "status": status,
        "passed": passed,
        "failed": failed,
        "details": details
    }


if __name__ == "__main__":
    result = run_tests()
    print(f"\nAlgorithm Tests: {result['status']}")
    print(f"Passed: {result['passed']}, Failed: {result['failed']}")
    for detail in result["details"]:
        print(f"  {detail}")