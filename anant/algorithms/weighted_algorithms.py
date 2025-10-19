"""
Weighted Hypergraph Algorithms
==============================

Advanced algorithms specifically designed for weighted hypergraphs,
including weight-based analysis and optimization.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from collections import defaultdict
from ..utils.decorators import performance_monitor
from ..utils.extras import safe_import

logger = logging.getLogger(__name__)

# Optional dependencies
scipy = safe_import('scipy')
networkx = safe_import('networkx')


@performance_monitor
def weighted_degree_distribution(hypergraph,
                                weight_column: str,
                                normalize: bool = True) -> Dict[str, Any]:
    """
    Analyze weighted degree distribution of hypergraph nodes.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        normalize: Whether to normalize the distribution
        
    Returns:
        Dictionary with degree distribution analysis
    """
    try:
        data = hypergraph.incidences.data
        
        if weight_column not in data.columns:
            logger.warning(f"Weight column '{weight_column}' not found")
            return {}
        
        # Calculate weighted degrees
        weighted_degrees = (
            data
            .group_by(hypergraph.incidences.node_column)
            .agg([
                pl.col(weight_column).sum().alias('weighted_degree'),
                pl.count().alias('edge_count')
            ])
        )
        
        degrees = weighted_degrees['weighted_degree'].to_numpy()
        edge_counts = weighted_degrees['edge_count'].to_numpy()
        
        # Calculate distribution statistics
        distribution = {
            'degrees': degrees.tolist(),
            'edge_counts': edge_counts.tolist(),
            'statistics': {
                'mean_weighted_degree': float(np.mean(degrees)),
                'std_weighted_degree': float(np.std(degrees)),
                'min_weighted_degree': float(np.min(degrees)),
                'max_weighted_degree': float(np.max(degrees)),
                'median_weighted_degree': float(np.median(degrees)),
                'total_weight': float(np.sum(degrees))
            },
            'distribution_bins': _create_degree_bins(degrees, normalize)
        }
        
        logger.info(f"Analyzed weighted degree distribution for {len(degrees)} nodes")
        return distribution
        
    except Exception as e:
        logger.error(f"Error in weighted degree distribution analysis: {e}")
        return {}


@performance_monitor
def weighted_shortest_paths(hypergraph,
                           weight_column: str,
                           source_node: Optional[str] = None,
                           target_node: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate weighted shortest paths in hypergraph.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights (used as inverse distances)
        source_node: Source node (if None, calculates from all nodes)
        target_node: Target node (if None, calculates to all nodes)
        
    Returns:
        Dictionary with shortest path results
    """
    try:
        if not networkx:
            logger.warning("NetworkX required for shortest path calculations")
            return {}
        
        # Convert hypergraph to weighted graph
        G = _hypergraph_to_weighted_graph(hypergraph, weight_column)
        
        if source_node and target_node:
            # Single source-target path
            try:
                path = networkx.shortest_path(G, source_node, target_node, weight='distance')
                length = networkx.shortest_path_length(G, source_node, target_node, weight='distance')
                
                return {
                    'path': path,
                    'length': length,
                    'source': source_node,
                    'target': target_node
                }
            except networkx.NetworkXNoPath:
                return {
                    'path': None,
                    'length': float('inf'),
                    'source': source_node,
                    'target': target_node
                }
        
        elif source_node:
            # Single source, all targets
            lengths = networkx.single_source_shortest_path_length(G, source_node, weight='distance')
            paths = networkx.single_source_shortest_path(G, source_node)
            
            return {
                'source': source_node,
                'lengths': dict(lengths),
                'paths': dict(paths)
            }
        
        else:
            # All pairs shortest paths (expensive for large graphs)
            if len(G.nodes()) > 1000:
                logger.warning("Large graph detected, skipping all-pairs shortest paths")
                return {}
            
            lengths = dict(networkx.all_pairs_shortest_path_length(G, weight='distance'))
            
            return {
                'all_pairs_lengths': lengths,
                'diameter': _calculate_graph_diameter(lengths),
                'average_path_length': _calculate_average_path_length(lengths)
            }
        
    except Exception as e:
        logger.error(f"Error in weighted shortest paths calculation: {e}")
        return {}


@performance_monitor
def weighted_connectivity_analysis(hypergraph,
                                  weight_column: str,
                                  threshold: Optional[float] = None) -> Dict[str, Any]:
    """
    Analyze connectivity patterns in weighted hypergraph.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        threshold: Weight threshold for connectivity analysis
        
    Returns:
        Dictionary with connectivity analysis results
    """
    try:
        data = hypergraph.incidences.data
        
        if weight_column not in data.columns:
            logger.warning(f"Weight column '{weight_column}' not found")
            return {}
        
        # Filter by threshold if provided
        if threshold is not None:
            filtered_data = data.filter(pl.col(weight_column) >= threshold)
        else:
            filtered_data = data
        
        # Analyze edge weight distribution
        weights = filtered_data[weight_column].to_numpy()
        
        # Calculate connectivity metrics
        edge_size_weights = (
            filtered_data
            .group_by(hypergraph.incidences.edge_column)
            .agg([
                pl.count().alias('edge_size'),
                pl.col(weight_column).mean().alias('avg_weight'),
                pl.col(weight_column).sum().alias('total_weight')
            ])
        )
        
        # Node connectivity analysis
        node_connectivity = (
            filtered_data
            .group_by(hypergraph.incidences.node_column)
            .agg([
                pl.col(hypergraph.incidences.edge_column).n_unique().alias('connected_edges'),
                pl.col(weight_column).sum().alias('total_weight'),
                pl.col(weight_column).mean().alias('avg_weight')
            ])
        )
        
        # Calculate connectivity strength distribution
        connectivity_strength = _calculate_connectivity_strength(
            hypergraph, filtered_data, weight_column
        )
        
        analysis = {
            'weight_statistics': {
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'min': float(np.min(weights)),
                'max': float(np.max(weights)),
                'threshold_used': threshold
            },
            'edge_analysis': {
                'total_edges': len(edge_size_weights),
                'avg_edge_size': float(edge_size_weights['edge_size'].mean()),
                'avg_edge_weight': float(edge_size_weights['avg_weight'].mean()),
                'total_graph_weight': float(edge_size_weights['total_weight'].sum())
            },
            'node_analysis': {
                'total_nodes': len(node_connectivity),
                'avg_node_degree': float(node_connectivity['connected_edges'].mean()),
                'avg_node_weight': float(node_connectivity['avg_weight'].mean()),
                'connectivity_distribution': connectivity_strength
            }
        }
        
        logger.info(f"Connectivity analysis completed with threshold {threshold}")
        return analysis
        
    except Exception as e:
        logger.error(f"Error in weighted connectivity analysis: {e}")
        return {}


@performance_monitor
def weight_based_importance(hypergraph,
                           weight_column: str,
                           importance_method: str = 'weight_centrality',
                           normalize: bool = True) -> Dict[str, float]:
    """
    Calculate node importance based on weights.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        importance_method: Method for calculating importance
        normalize: Whether to normalize importance scores
        
    Returns:
        Dictionary mapping node IDs to importance scores
    """
    try:
        data = hypergraph.incidences.data
        
        if weight_column not in data.columns:
            logger.warning(f"Weight column '{weight_column}' not found")
            return {}
        
        if importance_method == 'weight_centrality':
            # Sum of all edge weights for each node
            importance_data = (
                data
                .group_by(hypergraph.incidences.node_column)
                .agg([
                    pl.col(weight_column).sum().alias('importance')
                ])
            )
        
        elif importance_method == 'weighted_pagerank':
            return _weighted_pagerank_importance(hypergraph, weight_column, normalize)
        
        elif importance_method == 'weight_diversity':
            # Based on diversity of edge weights
            importance_data = (
                data
                .group_by(hypergraph.incidences.node_column)
                .agg([
                    pl.col(weight_column).std().alias('weight_std'),
                    pl.col(weight_column).mean().alias('weight_mean'),
                    pl.count().alias('edge_count')
                ])
                .with_columns([
                    (pl.col('weight_std') * pl.col('edge_count')).alias('importance')
                ])
            )
        
        elif importance_method == 'harmonic_centrality':
            # Harmonic mean of edge weights
            importance_data = (
                data
                .group_by(hypergraph.incidences.node_column)
                .agg([
                    (pl.count() / pl.col(weight_column).map_elements(lambda x: 1/x if x > 0 else 0, return_dtype=pl.Float64).sum()).alias('importance')
                ])
            )
        
        else:
            # Default to weight centrality
            return weight_based_importance(hypergraph, weight_column, 'weight_centrality', normalize)
        
        # Convert to dictionary
        importance_dict = dict(zip(
            importance_data[hypergraph.incidences.node_column].to_list(),
            importance_data['importance'].to_list()
        ))
        
        # Ensure all nodes are included
        for node in hypergraph.nodes:
            if node not in importance_dict:
                importance_dict[node] = 0.0
        
        # Normalize if requested
        if normalize and importance_dict:
            max_importance = max(importance_dict.values())
            if max_importance > 0:
                importance_dict = {
                    node: importance / max_importance 
                    for node, importance in importance_dict.items()
                }
        
        logger.info(f"Calculated {importance_method} importance for {len(importance_dict)} nodes")
        return importance_dict
        
    except Exception as e:
        logger.error(f"Error calculating weight-based importance: {e}")
        return {}


@performance_monitor
def weighted_clustering_coefficient(hypergraph,
                                   weight_column: str,
                                   node: Optional[str] = None) -> Union[float, Dict[str, float]]:
    """
    Calculate weighted clustering coefficient for hypergraph nodes.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        node: Specific node to analyze (if None, analyzes all nodes)
        
    Returns:
        Clustering coefficient(s)
    """
    try:
        if not networkx:
            logger.warning("NetworkX required for clustering coefficient calculation")
            return 0.0 if node else {}
        
        # Convert to weighted graph
        G = _hypergraph_to_weighted_graph(hypergraph, weight_column)
        
        if node:
            # Single node clustering coefficient
            if node in G:
                return networkx.clustering(G, node, weight='weight')
            else:
                return 0.0
        else:
            # All nodes clustering coefficients
            clustering = networkx.clustering(G, weight='weight')
            
            # Ensure all hypergraph nodes are included
            for hg_node in hypergraph.nodes:
                if hg_node not in clustering:
                    clustering[hg_node] = 0.0
            
            return clustering
        
    except Exception as e:
        logger.error(f"Error calculating weighted clustering coefficient: {e}")
        return 0.0 if node else {}


def _hypergraph_to_weighted_graph(hypergraph, weight_column: str):
    """Convert hypergraph to NetworkX weighted graph."""
    if not networkx:
        raise ImportError("NetworkX is required for this operation")
    
    G = networkx.Graph()
    G.add_nodes_from(hypergraph.nodes)
    
    data = hypergraph.incidences.data
    edge_nodes = defaultdict(list)
    
    for row in data.iter_rows(named=True):
        edge_id = row[hypergraph.incidences.edge_column]
        node_id = row[hypergraph.incidences.node_column]
        weight = row.get(weight_column, 1.0) if weight_column in row else 1.0
        edge_nodes[edge_id].append((node_id, weight))
    
    # Create weighted edges
    for edge_id, node_weight_pairs in edge_nodes.items():
        for i in range(len(node_weight_pairs)):
            for j in range(i + 1, len(node_weight_pairs)):
                node1, w1 = node_weight_pairs[i]
                node2, w2 = node_weight_pairs[j]
                
                # Weight is geometric mean, distance is inverse
                edge_weight = np.sqrt(w1 * w2)
                distance = 1.0 / max(edge_weight, 1e-10)
                
                if G.has_edge(node1, node2):
                    # Use minimum distance (maximum weight)
                    current_distance = G[node1][node2]['distance']
                    G[node1][node2]['distance'] = min(current_distance, distance)
                    G[node1][node2]['weight'] = max(G[node1][node2]['weight'], edge_weight)
                else:
                    G.add_edge(node1, node2, weight=edge_weight, distance=distance)
    
    return G


def _create_degree_bins(degrees: np.ndarray, normalize: bool = True) -> Dict[str, Any]:
    """Create histogram bins for degree distribution."""
    try:
        if len(degrees) == 0:
            return {'bin_edges': [], 'bin_counts': []}
        
        # Determine optimal number of bins
        num_bins = min(50, max(10, int(np.sqrt(len(degrees)))))
        
        counts, bin_edges = np.histogram(degrees, bins=num_bins)
        
        if normalize:
            counts = counts / np.sum(counts)
        
        return {
            'bin_edges': bin_edges.tolist(),
            'bin_counts': counts.tolist(),
            'num_bins': num_bins
        }
        
    except Exception as e:
        logger.error(f"Error creating degree bins: {e}")
        return {'bin_edges': [], 'bin_counts': []}


def _calculate_connectivity_strength(hypergraph, data: pl.DataFrame, weight_column: str) -> Dict[str, Any]:
    """Calculate connectivity strength distribution."""
    try:
        # Calculate pairwise connectivity strengths
        edge_connections = defaultdict(float)
        
        # Group by edge to find node pairs
        edge_groups = data.group_by(hypergraph.incidences.edge_column)
        
        for edge_id, edge_data in edge_groups:
            nodes_weights = list(zip(
                edge_data[hypergraph.incidences.node_column].to_list(),
                edge_data[weight_column].to_list()
            ))
            
            # Calculate pairwise connectivity strength
            for i, (node1, w1) in enumerate(nodes_weights):
                for j, (node2, w2) in enumerate(nodes_weights[i+1:], i+1):
                    pair_key = tuple(sorted([node1, node2]))
                    edge_connections[pair_key] += np.sqrt(w1 * w2)
        
        if not edge_connections:
            return {'strengths': [], 'statistics': {}}
        
        strengths = list(edge_connections.values())
        
        return {
            'strengths': strengths,
            'statistics': {
                'mean': float(np.mean(strengths)),
                'std': float(np.std(strengths)),
                'min': float(np.min(strengths)),
                'max': float(np.max(strengths)),
                'total_connections': len(strengths)
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating connectivity strength: {e}")
        return {'strengths': [], 'statistics': {}}


def _weighted_pagerank_importance(hypergraph, weight_column: str, normalize: bool = True) -> Dict[str, float]:
    """Calculate PageRank-based importance using edge weights."""
    try:
        if not networkx:
            logger.warning("NetworkX required for PageRank calculation")
            return {}
        
        G = _hypergraph_to_weighted_graph(hypergraph, weight_column)
        
        if len(G.edges()) == 0:
            # No edges, equal importance
            equal_importance = 1.0 / len(hypergraph.nodes) if normalize else 1.0
            return {node: equal_importance for node in hypergraph.nodes}
        
        pagerank = networkx.pagerank(G, weight='weight')
        
        # Ensure all hypergraph nodes are included
        for node in hypergraph.nodes:
            if node not in pagerank:
                pagerank[node] = 0.0
        
        return pagerank
        
    except Exception as e:
        logger.error(f"Error calculating weighted PageRank: {e}")
        return {}


def _calculate_graph_diameter(lengths: Dict[str, Dict[str, float]]) -> float:
    """Calculate graph diameter from all-pairs shortest paths."""
    try:
        max_length = 0.0
        for source in lengths:
            for target, length in lengths[source].items():
                if length != float('inf'):
                    max_length = max(max_length, length)
        
        return max_length
        
    except Exception as e:
        logger.error(f"Error calculating graph diameter: {e}")
        return float('inf')


def _calculate_average_path_length(lengths: Dict[str, Dict[str, float]]) -> float:
    """Calculate average path length from all-pairs shortest paths."""
    try:
        total_length = 0.0
        count = 0
        
        for source in lengths:
            for target, length in lengths[source].items():
                if source != target and length != float('inf'):
                    total_length += length
                    count += 1
        
        return total_length / count if count > 0 else float('inf')
        
    except Exception as e:
        logger.error(f"Error calculating average path length: {e}")
        return float('inf')


def weighted_analysis_summary(hypergraph,
                             weight_column: str,
                             include_paths: bool = False) -> Dict[str, Any]:
    """
    Comprehensive weighted analysis summary.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        include_paths: Whether to include shortest path analysis (expensive)
        
    Returns:
        Dictionary with comprehensive weighted analysis
    """
    try:
        summary = {
            'degree_distribution': weighted_degree_distribution(hypergraph, weight_column),
            'connectivity': weighted_connectivity_analysis(hypergraph, weight_column),
            'importance': {
                'weight_centrality': weight_based_importance(hypergraph, weight_column, 'weight_centrality'),
                'weight_diversity': weight_based_importance(hypergraph, weight_column, 'weight_diversity')
            },
            'clustering': weighted_clustering_coefficient(hypergraph, weight_column)
        }
        
        if include_paths and len(hypergraph.nodes) <= 100:  # Limit for performance
            summary['shortest_paths'] = weighted_shortest_paths(hypergraph, weight_column)
        
        logger.info("Completed comprehensive weighted analysis")
        return summary
        
    except Exception as e:
        logger.error(f"Error in weighted analysis summary: {e}")
        return {}