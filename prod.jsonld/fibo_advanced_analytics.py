#!/usr/bin/env python3
"""
FIBO Advanced Analytics

Advanced algorithmic analysis of the FIBO metagraph using ANANT's
sophisticated algorithms including centrality, clustering, and pattern detection.
"""

import sys
import os
from pathlib import Path
import time

# Add ANANT to path
sys.path.insert(0, '/home/amansingh/dev/ai/anant')

import anant
from anant import Hypergraph
from anant.io import AnantIO
from anant.algorithms import centrality, clustering, pattern_detection

def run_advanced_fibo_analytics():
    """Run advanced algorithmic analytics on FIBO metagraph"""
    print("🏦 FIBO Advanced Algorithmic Analytics")
    print("=" * 50)
    
    # Initialize ANANT
    anant.setup()
    
    # Load the FIBO metagraph
    metagraph_path = "/home/amansingh/dev/ai/anant/prod.jsonld/fibo_metagraphs/fibo_unified_metagraph"
    
    try:
        print("📂 Loading FIBO metagraph...")
        hg = AnantIO.load_hypergraph_parquet(metagraph_path)
        print(f"✅ Loaded: {hg.num_nodes:,} nodes, {hg.num_edges:,} edges")
        
    except Exception as e:
        print(f"❌ Error loading metagraph: {e}")
        return False
    
    # 1. Centrality Analysis
    print(f"\n🎯 1. CENTRALITY ANALYSIS")
    print("-" * 30)
    run_centrality_analysis(hg)
    
    # 2. Clustering Analysis
    print(f"\n🌐 2. CLUSTERING ANALYSIS")
    print("-" * 30)
    run_clustering_analysis(hg)
    
    # 3. Pattern Detection
    print(f"\n🔍 3. PATTERN DETECTION")
    print("-" * 30)
    run_pattern_detection(hg)
    
    # 4. Financial Hub Analysis
    print(f"\n💰 4. FINANCIAL HUB ANALYSIS")
    print("-" * 30)
    run_financial_hub_analysis(hg)
    
    return True

def run_centrality_analysis(hg: Hypergraph):
    """Analyze centrality in the FIBO metagraph"""
    print("   🎯 Computing centrality measures...")
    
    try:
        # Sample nodes for centrality analysis (subset for performance)
        node_sample = list(hg.nodes)[:1000]
        print(f"   📊 Analyzing {len(node_sample)} nodes...")
        
        # Simple degree centrality
        centrality_scores = {}
        for node in node_sample:
            degree = hg.get_node_degree(node)
            centrality_scores[node] = degree
        
        # Find top central nodes
        top_central = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"   🔝 Top 10 Central Nodes (by degree):")
        for i, (node, score) in enumerate(top_central, 1):
            node_name = str(node).split('/')[-1] if '/' in str(node) else str(node)
            print(f"     {i:2d}. {node_name[:50]:<50} (degree: {score})")
        
        # Analyze centrality distribution
        scores = list(centrality_scores.values())
        avg_centrality = sum(scores) / len(scores)
        max_centrality = max(scores)
        
        print(f"   📈 Centrality Statistics:")
        print(f"     Average centrality: {avg_centrality:.2f}")
        print(f"     Maximum centrality: {max_centrality}")
        print(f"     Centrality ratio: {max_centrality/avg_centrality:.2f}")
        
    except Exception as e:
        print(f"   ❌ Error in centrality analysis: {e}")

def run_clustering_analysis(hg: Hypergraph):
    """Analyze clustering in the FIBO metagraph"""
    print("   🌐 Computing clustering patterns...")
    
    try:
        # Sample for clustering analysis
        node_sample = list(hg.nodes)[:500]
        edge_sample = list(hg.edges)[:500]
        
        print(f"   📊 Analyzing {len(node_sample)} nodes, {len(edge_sample)} edges...")
        
        # Simple clustering analysis - find dense subgraphs
        dense_regions = find_dense_regions(hg, node_sample)
        
        print(f"   🔍 Dense regions found: {len(dense_regions)}")
        
        # Analyze cluster sizes
        if dense_regions:
            cluster_sizes = [len(cluster) for cluster in dense_regions]
            avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes)
            max_cluster_size = max(cluster_sizes)
            
            print(f"   📈 Clustering Statistics:")
            print(f"     Number of clusters: {len(dense_regions)}")
            print(f"     Average cluster size: {avg_cluster_size:.2f}")
            print(f"     Largest cluster size: {max_cluster_size}")
            
            # Show sample of largest cluster
            largest_cluster = max(dense_regions, key=len)
            print(f"   🏆 Largest cluster sample:")
            for i, node in enumerate(list(largest_cluster)[:5]):
                node_name = str(node).split('/')[-1] if '/' in str(node) else str(node)
                print(f"     • {node_name[:40]}")
        
    except Exception as e:
        print(f"   ❌ Error in clustering analysis: {e}")

def find_dense_regions(hg: Hypergraph, nodes: list) -> list:
    """Find densely connected regions"""
    clusters = []
    visited = set()
    
    for node in nodes[:100]:  # Limit for performance
        if node in visited:
            continue
            
        # Find neighbors
        neighbors = set()
        node_edges = hg.get_node_edges(node)
        
        for edge in node_edges[:10]:  # Limit edges per node
            edge_nodes = hg.get_edge_nodes(edge)
            neighbors.update(edge_nodes)
        
        # If node has enough neighbors, consider it a cluster seed
        if len(neighbors) > 5:
            cluster = {node} | neighbors
            clusters.append(cluster)
            visited.update(cluster)
    
    return clusters

def run_pattern_detection(hg: Hypergraph):
    """Detect patterns in the FIBO metagraph"""
    print("   🔍 Detecting structural patterns...")
    
    try:
        # Financial pattern detection
        patterns = detect_financial_patterns(hg)
        
        print(f"   📊 Pattern Detection Results:")
        for pattern_type, count in patterns.items():
            print(f"     {pattern_type}: {count} instances")
        
        # Detect common substructures
        substructures = detect_common_substructures(hg)
        print(f"   🧬 Common substructures: {len(substructures)}")
        
    except Exception as e:
        print(f"   ❌ Error in pattern detection: {e}")

def detect_financial_patterns(hg: Hypergraph) -> dict:
    """Detect financial domain patterns"""
    patterns = {
        'regulatory_patterns': 0,
        'market_patterns': 0,
        'instrument_patterns': 0,
        'entity_patterns': 0
    }
    
    # Sample nodes for pattern detection
    for node in list(hg.nodes)[:1000]:
        node_str = str(node).lower()
        
        if any(term in node_str for term in ['regulation', 'compliance', 'law']):
            patterns['regulatory_patterns'] += 1
        elif any(term in node_str for term in ['market', 'exchange', 'trading']):
            patterns['market_patterns'] += 1
        elif any(term in node_str for term in ['bond', 'equity', 'derivative', 'security']):
            patterns['instrument_patterns'] += 1
        elif any(term in node_str for term in ['bank', 'corporation', 'entity', 'organization']):
            patterns['entity_patterns'] += 1
    
    return patterns

def detect_common_substructures(hg: Hypergraph) -> list:
    """Detect common substructures"""
    substructures = []
    
    # Simple substructure detection - look for repeated edge patterns
    edge_patterns = {}
    
    for edge in list(hg.edges)[:200]:  # Sample edges
        edge_nodes = hg.get_edge_nodes(edge)
        if len(edge_nodes) == 3:  # RDF triple pattern
            # Create pattern signature
            pattern = tuple(sorted([str(n).split('/')[-1] for n in edge_nodes]))
            edge_patterns[pattern] = edge_patterns.get(pattern, 0) + 1
    
    # Find common patterns (appearing multiple times)
    common_patterns = {k: v for k, v in edge_patterns.items() if v > 1}
    
    return list(common_patterns.keys())

def run_financial_hub_analysis(hg: Hypergraph):
    """Analyze financial hubs and their importance"""
    print("   💰 Analyzing financial hubs...")
    
    try:
        # Find financial domain hubs
        financial_hubs = find_financial_hubs(hg)
        
        print(f"   🏦 Financial hubs identified: {len(financial_hubs)}")
        
        # Analyze hub characteristics
        for i, (hub, metrics) in enumerate(financial_hubs[:5], 1):
            hub_name = str(hub).split('/')[-1] if '/' in str(hub) else str(hub)
            print(f"   {i}. {hub_name[:40]:<40} (connections: {metrics['degree']}, type: {metrics['type']})")
        
        # Hub connectivity analysis
        if financial_hubs:
            degrees = [h[1]['degree'] for h in financial_hubs]
            avg_hub_degree = sum(degrees) / len(degrees)
            print(f"   📊 Average hub connectivity: {avg_hub_degree:.2f}")
        
    except Exception as e:
        print(f"   ❌ Error in financial hub analysis: {e}")

def find_financial_hubs(hg: Hypergraph) -> list:
    """Identify key financial hubs"""
    hubs = []
    
    # Financial keywords for hub identification
    financial_keywords = [
        'bank', 'market', 'security', 'bond', 'equity', 'fund', 
        'financial', 'monetary', 'currency', 'derivative'
    ]
    
    # Analyze high-degree nodes in financial contexts
    for node in list(hg.nodes)[:2000]:
        node_str = str(node).lower()
        degree = hg.get_node_degree(node)
        
        # Check if node is financial and has high connectivity
        if degree > 10:  # High connectivity threshold
            hub_type = 'general'
            for keyword in financial_keywords:
                if keyword in node_str:
                    hub_type = keyword
                    break
            
            hubs.append((node, {
                'degree': degree,
                'type': hub_type
            }))
    
    # Sort by degree
    hubs.sort(key=lambda x: x[1]['degree'], reverse=True)
    
    return hubs

if __name__ == "__main__":
    try:
        success = run_advanced_fibo_analytics()
        print(f"\n🎉 Advanced FIBO analytics: {'COMPLETED' if success else 'FAILED'}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)