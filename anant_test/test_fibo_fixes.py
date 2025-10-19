"""
Test Critical Fixes with FIBO Dataset
====================================

Test the critical fixes using the actual FIBO dataset to ensure
they resolve the issues encountered during the original analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
from anant.classes.hypergraph import Hypergraph
from anant.io.parquet_io import AnantIO
from anant.algorithms.clustering import hypergraph_clustering
from anant.algorithms.centrality_enhanced import enhanced_centrality_analysis
from anant.algorithms.sampling import get_sampling_recommendations, SmartSampler
from anant.utils.performance import PerformanceProfiler


def test_fibo_fixes():
    """Test critical fixes with actual FIBO dataset"""
    
    print("=== Testing Critical Fixes with FIBO Dataset ===")
    
    fibo_path = "/home/amansingh/dev/ai/anant/prod.jsonld/fibo_metagraphs/fibo_unified_metagraph"
    
    if not os.path.exists(fibo_path):
        print("❌ FIBO dataset not found, skipping test")
        return False
    
    # Load FIBO metagraph
    print("📂 Loading FIBO unified metagraph...")
    start_time = time.time()
    
    try:
        io_handler = AnantIO()
        fibo_hg = io_handler.load_hypergraph_parquet(fibo_path)
        load_time = time.time() - start_time
        
        print(f"✅ Loaded FIBO hypergraph in {load_time:.2f}s")
        print(f"   📊 Nodes: {len(fibo_hg.nodes):,}")
        print(f"   🔗 Edges: {len(fibo_hg.edges):,}")
        print(f"   📈 Incidences: {fibo_hg.incidences.num_incidences():,}")
        
    except Exception as e:
        print(f"❌ Failed to load FIBO dataset: {e}")
        return False
    
    # Test 1: IncidenceStore fixes
    print("\n🔧 Test 1: IncidenceStore Interface Fixes")
    try:
        # These should now work without AttributeError
        edge_col = fibo_hg.incidences.edge_column
        node_col = fibo_hg.incidences.node_column
        weight_col = fibo_hg.incidences.weight_column
        
        print(f"✅ IncidenceStore properties accessible:")
        print(f"   - edge_column: {edge_col}")
        print(f"   - node_column: {node_col}")
        print(f"   - weight_column: {weight_col}")
        
    except AttributeError as e:
        print(f"❌ IncidenceStore interface still broken: {e}")
        return False
    
    # Test 2: Smart Sampling
    print("\n🎯 Test 2: Intelligent Sampling")
    try:
        recommendations = get_sampling_recommendations(fibo_hg)
        print(f"✅ Sampling recommendations generated:")
        print(f"   - Total nodes: {recommendations['total_nodes']:,}")
        print(f"   - Recommended sampling: {recommendations['recommended_sampling']}")
        print(f"   - Strategy: {recommendations['recommended_strategy']}")
        print(f"   - Optimal sizes: {recommendations['optimal_sample_sizes']}")
        
        # Test sampling
        sampler = SmartSampler(fibo_hg, strategy='adaptive')
        sample_hg = sampler.adaptive_sample(sample_size=1000, algorithm='centrality')
        
        print(f"✅ Adaptive sampling successful:")
        print(f"   - Sampled nodes: {len(sample_hg.nodes):,}")
        print(f"   - Sampled edges: {len(sample_hg.edges):,}")
        
    except Exception as e:
        print(f"❌ Sampling failed: {e}")
        return False
    
    # Test 3: Clustering fixes
    print("\n🔗 Test 3: Clustering Algorithm Fixes")
    
    with PerformanceProfiler("clustering_test") as profiler:
        try:
            # This should no longer crash with AttributeError
            profiler.checkpoint("start_clustering")
            
            # Test on sampled graph for performance
            clusters = hypergraph_clustering(
                sample_hg, 
                algorithm='modularity'
            )
            profiler.checkpoint("modularity_complete")
            
            print(f"✅ Modularity clustering successful:")
            print(f"   - Clusters: {len(set(clusters.values()))}")
            print(f"   - Nodes clustered: {len(clusters)}")
            
            # Test spectral clustering
            spectral_clusters = hypergraph_clustering(
                sample_hg,
                algorithm='spectral',
                n_clusters=5
            )
            profiler.checkpoint("spectral_complete")
            
            print(f"✅ Spectral clustering successful:")
            print(f"   - Clusters: {len(set(spectral_clusters.values()))}")
            print(f"   - Nodes clustered: {len(spectral_clusters)}")
            
        except Exception as e:
            print(f"❌ Clustering failed: {e}")
            return False
    
    clustering_report = profiler.get_report()
    print(f"   ⏱️  Total clustering time: {clustering_report['total_execution_time']:.2f}s")
    
    # Test 4: Enhanced Centrality
    print("\n⭐ Test 4: Enhanced Centrality Measures")
    
    with PerformanceProfiler("centrality_test") as profiler:
        try:
            profiler.checkpoint("start_centrality")
            
            # Test enhanced centrality on sampled graph
            centrality_results = enhanced_centrality_analysis(
                sample_hg,
                measures=['degree', 'betweenness', 'closeness'],
                sample_large_graphs=True,
                max_nodes=1000
            )
            profiler.checkpoint("centrality_complete")
            
            print(f"✅ Enhanced centrality analysis successful:")
            print(f"   - Measures computed: {len([col for col in centrality_results.columns if col.endswith('_centrality')])}")
            print(f"   - Nodes analyzed: {len(centrality_results)}")
            
            # Show top central nodes
            if 'degree_centrality' in centrality_results.columns:
                top_nodes = centrality_results.sort('degree_centrality', descending=True).head(3)
                print(f"   - Top 3 central nodes by degree:")
                for row in top_nodes.iter_rows(named=True):
                    print(f"     • {row['node_id']}: {row['degree_centrality']:.3f}")
            
        except Exception as e:
            print(f"❌ Enhanced centrality failed: {e}")
            return False
    
    centrality_report = profiler.get_report()
    print(f"   ⏱️  Total centrality time: {centrality_report['total_execution_time']:.2f}s")
    
    # Test 5: Integration Test
    print("\n🚀 Test 5: Full Integration Test")
    
    try:
        # Run full analysis pipeline
        with PerformanceProfiler("full_pipeline") as profiler:
            
            profiler.checkpoint("start_full_analysis")
            
            # Create medium sample for full analysis
            medium_sample = sampler.adaptive_sample(sample_size=500, algorithm='general')
            profiler.checkpoint("sampling_complete")
            
            # Run clustering
            final_clusters = hypergraph_clustering(medium_sample, algorithm='modularity')
            profiler.checkpoint("clustering_complete")
            
            # Run centrality
            final_centrality = enhanced_centrality_analysis(
                medium_sample,
                measures=['degree', 'betweenness'],
                sample_large_graphs=False
            )
            profiler.checkpoint("centrality_complete")
            
            print(f"✅ Full integration test successful:")
            print(f"   - Sample size: {len(medium_sample.nodes)} nodes")
            print(f"   - Clusters identified: {len(set(final_clusters.values()))}")
            print(f"   - Centrality measures: {len([col for col in final_centrality.columns if col.endswith('_centrality')])}")
        
        integration_report = profiler.get_report()
        print(f"   ⏱️  Full pipeline time: {integration_report['total_execution_time']:.2f}s")
        
        # Show checkpoint breakdown
        print(f"   📊 Performance breakdown:")
        for checkpoint in integration_report['checkpoints']:
            print(f"     • {checkpoint['name']}: {checkpoint['elapsed_time']:.2f}s")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    print("\n🎉 ALL CRITICAL FIXES VALIDATED SUCCESSFULLY!")
    print("\n📋 Summary of Improvements:")
    print("   ✅ IncidenceStore interface fixed (edge_column/node_column properties)")
    print("   ✅ Clustering algorithms no longer crash")
    print("   ✅ Intelligent sampling implemented and working")
    print("   ✅ Enhanced centrality measures available")
    print("   ✅ Performance monitoring active")
    print("   ✅ Full integration pipeline working")
    
    return True


if __name__ == "__main__":
    success = test_fibo_fixes()
    
    if success:
        print("\n🌟 FIBO testing completed successfully!")
        print("   The critical fixes resolve all issues encountered in the original analysis.")
        exit(0)
    else:
        print("\n💥 FIBO testing failed!")
        print("   Some critical fixes may need additional work.")
        exit(1)