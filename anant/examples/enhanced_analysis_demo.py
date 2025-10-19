"""
Enhanced Analysis Features Demonstration
========================================

Comprehensive demonstration of Anant's advanced analysis capabilities.
"""

import sys
sys.path.append('/home/amansingh/dev/ai/anant')

import anant
import anant.algorithms as algorithms
import polars as pl
import numpy as np


def create_sample_hypergraph():
    """Create a sample hypergraph for analysis."""
    # Create a more complex hypergraph with multiple edge types and weights
    data = pl.DataFrame({
        'edge_id': [
            'research_team', 'research_team', 'research_team', 'research_team',
            'paper_authors', 'paper_authors', 'paper_authors',
            'conference_committee', 'conference_committee', 'conference_committee', 'conference_committee',
            'collaboration', 'collaboration',
            'department', 'department', 'department', 'department', 'department',
            'project_alpha', 'project_alpha', 'project_alpha',
            'project_beta', 'project_beta'
        ],
        'node_id': [
            'alice', 'bob', 'charlie', 'diana',  # research team
            'alice', 'eve', 'frank',  # paper authors
            'bob', 'charlie', 'grace', 'henry',  # conference committee
            'diana', 'eve',  # collaboration
            'alice', 'bob', 'charlie', 'grace', 'henry',  # department
            'alice', 'frank', 'grace',  # project alpha
            'bob', 'diana'  # project beta
        ],
        'weight': [
            1.0, 0.8, 0.9, 1.1,  # research team weights
            1.2, 0.7, 0.9,  # paper author weights
            0.6, 0.8, 1.0, 0.9,  # committee weights
            1.3, 1.1,  # collaboration weights
            0.5, 0.6, 0.7, 0.8, 0.9,  # department weights
            1.4, 1.0, 0.8,  # project alpha weights
            0.9, 1.2  # project beta weights
        ],
        'importance': [
            'high', 'medium', 'high', 'high',
            'very_high', 'medium', 'high',
            'medium', 'high', 'high', 'medium',
            'very_high', 'very_high',
            'low', 'low', 'low', 'medium', 'medium',
            'high', 'medium', 'medium',
            'high', 'very_high'
        ]
    })
    
    return anant.Hypergraph(data=data)


def demonstrate_centrality_analysis(hg):
    """Demonstrate centrality analysis capabilities."""
    print("🎯 CENTRALITY ANALYSIS")
    print("=" * 50)
    
    # Weighted node centrality
    centrality = algorithms.weighted_node_centrality(hg, 'weight', normalize=True)
    print(f"📊 Node Centrality (weighted by 'weight' column):")
    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    for node, score in sorted_centrality:
        print(f"   {node}: {score:.3f}")
    
    # Edge centrality
    edge_centrality = algorithms.edge_centrality(hg, 'size', normalize=True)
    print(f"\n📊 Edge Centrality (by size):")
    sorted_edge_centrality = sorted(edge_centrality.items(), key=lambda x: x[1], reverse=True)
    for edge, score in sorted_edge_centrality:
        print(f"   {edge}: {score:.3f}")
    
    return centrality


def demonstrate_community_detection(hg):
    """Demonstrate community detection capabilities."""
    print("\n🏘️  COMMUNITY DETECTION")
    print("=" * 50)
    
    # Community detection
    communities = algorithms.community_detection(hg, 'weight')
    
    # Group nodes by community
    community_groups = {}
    for node, community_id in communities.items():
        if community_id not in community_groups:
            community_groups[community_id] = []
        community_groups[community_id].append(node)
    
    print(f"🎭 Detected {len(community_groups)} communities:")
    for community_id, nodes in community_groups.items():
        print(f"   Community {community_id}: {nodes}")
    
    return communities


def demonstrate_comprehensive_analysis(hg):
    """Demonstrate comprehensive analysis workflow."""
    print("\n🔬 COMPREHENSIVE ANALYSIS")
    print("=" * 50)
    
    # Analyze hypergraph structure
    print(f"📈 Hypergraph Structure:")
    print(f"   Nodes: {hg.num_nodes}")
    print(f"   Edges: {hg.num_edges}")
    print(f"   Total incidences: {len(hg.incidences.data)}")
    
    # Node degree analysis
    data = hg.incidences.data
    node_degrees = (
        data
        .group_by('node_id')
        .agg([pl.count().alias('degree')])
        .sort('degree', descending=True)
    )
    
    print(f"\n📊 Node Degree Distribution:")
    for row in node_degrees.iter_rows(named=True):
        print(f"   {row['node_id']}: {row['degree']} edges")
    
    # Edge size analysis
    edge_sizes = (
        data
        .group_by('edge_id')
        .agg([pl.count().alias('size')])
        .sort('size', descending=True)
    )
    
    print(f"\n📊 Edge Size Distribution:")
    for row in edge_sizes.iter_rows(named=True):
        print(f"   {row['edge_id']}: {row['size']} nodes")
    
    # Weight analysis
    weight_stats = data['weight'].describe()
    print(f"\n📊 Weight Statistics:")
    print(f"   Mean: {data['weight'].mean():.3f}")
    print(f"   Std: {data['weight'].std():.3f}")
    print(f"   Min: {data['weight'].min():.3f}")
    print(f"   Max: {data['weight'].max():.3f}")


def demonstrate_advanced_queries(hg):
    """Demonstrate advanced querying capabilities."""
    print("\n🔍 ADVANCED QUERIES")
    print("=" * 50)
    
    data = hg.incidences.data
    
    # Find nodes with high weights
    high_weight_nodes = (
        data
        .filter(pl.col('weight') > 1.0)
        .group_by('node_id')
        .agg([
            pl.col('weight').mean().alias('avg_weight'),
            pl.count().alias('high_weight_edges')
        ])
        .sort('avg_weight', descending=True)
    )
    
    print("🎯 Nodes with high-weight connections:")
    for row in high_weight_nodes.iter_rows(named=True):
        print(f"   {row['node_id']}: avg weight {row['avg_weight']:.3f} across {row['high_weight_edges']} edges")
    
    # Find large edges
    large_edges = (
        data
        .group_by('edge_id')
        .agg([pl.count().alias('size')])
        .filter(pl.col('size') >= 4)
        .sort('size', descending=True)
    )
    
    print(f"\n🎯 Large edges (4+ nodes):")
    for row in large_edges.iter_rows(named=True):
        edge_nodes = data.filter(pl.col('edge_id') == row['edge_id'])['node_id'].to_list()
        print(f"   {row['edge_id']}: {edge_nodes}")


def main():
    """Main demonstration function."""
    print("🚀 ANANT ENHANCED ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Create sample hypergraph
    print("📊 Creating sample academic collaboration hypergraph...")
    hg = create_sample_hypergraph()
    print(f"✅ Created hypergraph with {hg.num_nodes} nodes and {hg.num_edges} edges")
    
    # Demonstrate analysis capabilities
    centrality = demonstrate_centrality_analysis(hg)
    communities = demonstrate_community_detection(hg)
    demonstrate_comprehensive_analysis(hg)
    demonstrate_advanced_queries(hg)
    
    # Summary insights
    print("\n💡 ANALYSIS INSIGHTS")
    print("=" * 50)
    
    # Find most central person
    top_person = max(centrality.items(), key=lambda x: x[1])
    print(f"🌟 Most central person: {top_person[0]} (centrality: {top_person[1]:.3f})")
    
    # Find largest community
    community_sizes = {}
    for person, community_id in communities.items():
        community_sizes[community_id] = community_sizes.get(community_id, 0) + 1
    
    largest_community_id = max(community_sizes.items(), key=lambda x: x[1])[0]
    largest_community_members = [p for p, c in communities.items() if c == largest_community_id]
    print(f"👥 Largest community: Community {largest_community_id} with {len(largest_community_members)} members: {largest_community_members}")
    
    print(f"\n🎉 Analysis complete! Anant's enhanced features successfully analyzed:")
    print(f"   ✅ Centrality measures computed")
    print(f"   ✅ Community structure detected") 
    print(f"   ✅ Advanced queries executed")
    print(f"   ✅ Statistical insights generated")


if __name__ == "__main__":
    main()