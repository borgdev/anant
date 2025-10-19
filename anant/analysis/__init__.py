"""
Anant Analysis Module

This module provides advanced algorithms for hypergraph analysis including:
- Centrality measures (degree, closeness, betweenness, s-centrality, eigenvector, PageRank)
- Community detection and clustering
- Structural analysis and metrics
- Spectral analysis using Laplacians
- Temporal analysis for dynamic hypergraphs

All algorithms are optimized for Polars DataFrames and designed for
high-performance analysis of large hypergraphs.
"""

from .centrality import (
    degree_centrality,
    node_degree_centrality,
    edge_degree_centrality,
    closeness_centrality,
    betweenness_centrality,
    s_centrality,
    eigenvector_centrality,
    pagerank_centrality,
    weighted_degree_centrality
)

from .clustering import (
    spectral_clustering,
    modularity_clustering,
    hierarchical_clustering,
    community_detection,
    overlapping_community_detection,
    multi_resolution_clustering,
    consensus_clustering,
    community_quality_metrics,
    edge_community_detection,
    adaptive_community_detection
)

from .structural import (
    connected_components,
    hypergraph_diameter,
    clustering_coefficient,
    global_clustering_coefficient,
    local_clustering_coefficient
)

from .spectral import (
    node_laplacian,
    edge_laplacian,
    hypergraph_laplacian,
    laplacian_spectrum,
    algebraic_connectivity
)

from .temporal import (
    TemporalSnapshot,
    TemporalHypergraph,
    temporal_degree_evolution,
    temporal_centrality_evolution,
    temporal_clustering_evolution,
    stability_analysis,
    temporal_motif_analysis,
    growth_analysis,
    persistence_analysis
)

__all__ = [
    # Centrality measures
    'degree_centrality',
    'node_degree_centrality', 
    'edge_degree_centrality',
    'closeness_centrality',
    'betweenness_centrality',
    's_centrality',
    'eigenvector_centrality',
    'pagerank_centrality',
    'weighted_degree_centrality',
    
    # Clustering and community detection
    'spectral_clustering',
    'modularity_clustering', 
    'hierarchical_clustering',
    'community_detection',
    'overlapping_community_detection',
    'multi_resolution_clustering',
    'consensus_clustering',
    'community_quality_metrics',
    'edge_community_detection',
    'adaptive_community_detection',
    
    # Structural analysis
    'connected_components',
    'hypergraph_diameter',
    'clustering_coefficient',
    'global_clustering_coefficient',
    'local_clustering_coefficient',
    
    # Spectral analysis
    'node_laplacian',
    'edge_laplacian', 
    'hypergraph_laplacian',
    'laplacian_spectrum',
    'algebraic_connectivity',
    
    # Temporal analysis
    'TemporalSnapshot',
    'TemporalHypergraph',
    'temporal_degree_evolution',
    'temporal_centrality_evolution',
    'temporal_clustering_evolution',
    'stability_analysis',
    'temporal_motif_analysis',
    'growth_analysis',
    'persistence_analysis'
]