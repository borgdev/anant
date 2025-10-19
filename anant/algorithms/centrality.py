"""
Hypergraph Centrality Measures
==============================

Advanced centrality algorithms for hypergraph analysis, supporting
weighted and unweighted hypergraphs with efficient implementations.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from collections import defaultdict
from ..utils.decorators import performance_monitor
from ..utils.extras import safe_import

logger = logging.getLogger(__name__)

# Optional dependencies for advanced algorithms
scipy = safe_import('scipy')
networkx = safe_import('networkx')


@performance_monitor
def hypergraph_centrality(hypergraph, 
                         centrality_type: str = 'degree',
                         weight_column: Optional[str] = None,
                         normalize: bool = True) -> Dict[str, float]:
    """
    Compute centrality measures for hypergraph nodes.
    
    Args:
        hypergraph: Anant Hypergraph instance
        centrality_type: Type of centrality ('degree', 'eigenvector', 'betweenness', 'closeness')
        weight_column: Column name for edge weights (if any)
        normalize: Whether to normalize centrality values
        
    Returns:
        Dictionary mapping node IDs to centrality values
    """
    if centrality_type == 'degree':
        return weighted_node_centrality(hypergraph, weight_column, normalize)
    elif centrality_type == 'eigenvector':
        return eigenvector_centrality(hypergraph, weight_column, normalize)
    elif centrality_type == 'betweenness':
        return betweenness_centrality(hypergraph, weight_column, normalize)
    elif centrality_type == 'closeness':
        return closeness_centrality(hypergraph, weight_column, normalize)
    else:
        raise ValueError(f"Unknown centrality type: {centrality_type}")


@performance_monitor
def weighted_node_centrality(hypergraph, 
                           weight_column: Optional[str] = None,
                           normalize: bool = True) -> Dict[str, float]:
    """
    Compute weighted degree centrality for hypergraph nodes.
    
    For hypergraphs, degree centrality considers both the number of edges
    a node participates in and optionally the weights of those edges.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        normalize: Whether to normalize by maximum possible degree
        
    Returns:
        Dictionary mapping node IDs to centrality values
    """
    try:
        data = hypergraph.incidences.data
        
        if weight_column and weight_column in data.columns:
            # Weighted degree centrality
            centrality_data = (
                data
                .group_by('node_id')
                .agg([
                    pl.col(weight_column).sum().alias('weighted_degree'),
                    pl.count().alias('edge_count')
                ])
            )
        else:
            # Standard degree centrality (count of edges)
            centrality_data = (
                data
                .group_by('node_id')
                .agg([
                    pl.count().alias('degree')
                ])
            )
        
        # Convert to dictionary
        if weight_column and weight_column in data.columns:
            centrality_dict = dict(zip(
                centrality_data['node_id'].to_list(),
                centrality_data['weighted_degree'].to_list()
            ))
        else:
            centrality_dict = dict(zip(
                centrality_data['node_id'].to_list(),
                centrality_data['degree'].to_list()
            ))
        
        # Ensure all nodes are included (some might have degree 0)
        for node in hypergraph.nodes:
            if node not in centrality_dict:
                centrality_dict[node] = 0.0
        
        # Normalize if requested
        if normalize and centrality_dict:
            max_centrality = max(centrality_dict.values())
            if max_centrality > 0:
                centrality_dict = {
                    node: centrality / max_centrality 
                    for node, centrality in centrality_dict.items()
                }
        
        logger.info(f"Computed weighted node centrality for {len(centrality_dict)} nodes")
        return centrality_dict
        
    except Exception as e:
        logger.error(f"Error computing weighted node centrality: {e}")
        return {}


@performance_monitor
def edge_centrality(hypergraph,
                   centrality_type: str = 'size',
                   weight_column: Optional[str] = None,
                   normalize: bool = True) -> Dict[str, float]:
    """
    Compute centrality measures for hypergraph edges.
    
    Args:
        hypergraph: Anant Hypergraph instance
        centrality_type: Type of edge centrality ('size', 'weight', 'connectivity')
        weight_column: Column name for edge weights
        normalize: Whether to normalize centrality values
        
    Returns:
        Dictionary mapping edge IDs to centrality values
    """
    try:
        data = hypergraph.incidences.data
        
        if centrality_type == 'size':
            # Edge centrality based on edge size (number of nodes)
            edge_centrality_data = (
                data
                .group_by(hypergraph.incidences.edge_column)
                .agg([
                    pl.count().alias('edge_size')
                ])
            )
            centrality_dict = dict(zip(
                edge_centrality_data[hypergraph.incidences.edge_column].to_list(),
                edge_centrality_data['edge_size'].to_list()
            ))
            
        elif centrality_type == 'weight' and weight_column and weight_column in data.columns:
            # Edge centrality based on weights
            edge_centrality_data = (
                data
                .group_by(hypergraph.incidences.edge_column)
                .agg([
                    pl.col(weight_column).mean().alias('avg_weight')
                ])
            )
            centrality_dict = dict(zip(
                edge_centrality_data[hypergraph.incidences.edge_column].to_list(),
                edge_centrality_data['avg_weight'].to_list()
            ))
            
        elif centrality_type == 'connectivity':
            # Edge centrality based on connectivity (nodes' degree sum)
            node_degrees = weighted_node_centrality(hypergraph, weight_column, False)
            
            edge_connectivity = defaultdict(float)
            for row in data.iter_rows(named=True):
                edge_id = row[hypergraph.incidences.edge_column]
                node_id = row[hypergraph.incidences.node_column]
                edge_connectivity[edge_id] += node_degrees.get(node_id, 0)
            
            centrality_dict = dict(edge_connectivity)
            
        else:
            # Default to size-based centrality
            return edge_centrality(hypergraph, 'size', weight_column, normalize)
        
        # Normalize if requested
        if normalize and centrality_dict:
            max_centrality = max(centrality_dict.values())
            if max_centrality > 0:
                centrality_dict = {
                    edge: centrality / max_centrality 
                    for edge, centrality in centrality_dict.items()
                }
        
        logger.info(f"Computed edge centrality for {len(centrality_dict)} edges")
        return centrality_dict
        
    except Exception as e:
        logger.error(f"Error computing edge centrality: {e}")
        return {}


@performance_monitor
def eigenvector_centrality(hypergraph,
                          weight_column: Optional[str] = None,
                          normalize: bool = True,
                          max_iterations: int = 100,
                          tolerance: float = 1e-6) -> Dict[str, float]:
    """
    Compute eigenvector centrality for hypergraph nodes.
    
    This implementation uses the hypergraph adjacency matrix approach,
    where nodes are connected if they share hyperedges.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        normalize: Whether to normalize centrality values
        max_iterations: Maximum iterations for power method
        tolerance: Convergence tolerance
        
    Returns:
        Dictionary mapping node IDs to centrality values
    """
    try:
        if not scipy:
            logger.warning("SciPy not available, falling back to degree centrality")
            return weighted_node_centrality(hypergraph, weight_column, normalize)
        
        # Build node adjacency matrix
        nodes = list(hypergraph.nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n_nodes = len(nodes)
        
        # Initialize adjacency matrix
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        # Build adjacency based on shared hyperedges
        data = hypergraph.incidences.data
        edge_nodes = defaultdict(list)
        
        for row in data.iter_rows(named=True):
            edge_id = row[hypergraph.incidences.edge_column]
            node_id = row[hypergraph.incidences.node_column]
            weight = row.get(weight_column, 1.0) if weight_column else 1.0
            edge_nodes[edge_id].append((node_id, weight))
        
        # Create adjacency matrix
        for edge_id, node_weight_pairs in edge_nodes.items():
            nodes_in_edge = [pair[0] for pair in node_weight_pairs]
            weights = [pair[1] for pair in node_weight_pairs]
            
            # Connect all pairs of nodes in the same hyperedge
            for i, (node1, w1) in enumerate(node_weight_pairs):
                for j, (node2, w2) in enumerate(node_weight_pairs):
                    if i != j:
                        idx1, idx2 = node_to_idx[node1], node_to_idx[node2]
                        # Weight by geometric mean of node weights in the edge
                        edge_weight = np.sqrt(w1 * w2) if weight_column else 1.0
                        adj_matrix[idx1, idx2] += edge_weight
        
        # Compute eigenvector centrality using power method
        if n_nodes == 0:
            return {}
        
        # Handle disconnected graph
        if np.allclose(adj_matrix, 0):
            centrality_dict = {node: 1.0/n_nodes for node in nodes}
        else:
            # Use scipy's eigenvalue computation
            eigenvalues, eigenvectors = scipy.linalg.eig(adj_matrix)
            
            # Find largest eigenvalue
            max_eigenvalue_idx = np.argmax(np.real(eigenvalues))
            principal_eigenvector = np.real(eigenvectors[:, max_eigenvalue_idx])
            
            # Ensure positive values
            if np.sum(principal_eigenvector) < 0:
                principal_eigenvector = -principal_eigenvector
            
            centrality_dict = dict(zip(nodes, principal_eigenvector))
        
        # Normalize if requested
        if normalize and centrality_dict:
            max_centrality = max(centrality_dict.values())
            if max_centrality > 0:
                centrality_dict = {
                    node: centrality / max_centrality 
                    for node, centrality in centrality_dict.items()
                }
        
        logger.info(f"Computed eigenvector centrality for {len(centrality_dict)} nodes")
        return centrality_dict
        
    except Exception as e:
        logger.error(f"Error computing eigenvector centrality: {e}")
        # Fallback to degree centrality
        return weighted_node_centrality(hypergraph, weight_column, normalize)


@performance_monitor
def betweenness_centrality(hypergraph,
                          weight_column: Optional[str] = None,
                          normalize: bool = True) -> Dict[str, float]:
    """
    Compute betweenness centrality for hypergraph nodes.
    
    This is an approximation that converts the hypergraph to a graph
    representation and computes traditional betweenness centrality.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        normalize: Whether to normalize centrality values
        
    Returns:
        Dictionary mapping node IDs to centrality values
    """
    try:
        if not networkx:
            logger.warning("NetworkX not available, falling back to degree centrality")
            return weighted_node_centrality(hypergraph, weight_column, normalize)
        
        # Convert hypergraph to NetworkX graph
        G = networkx.Graph()
        
        # Add all nodes
        G.add_nodes_from(hypergraph.nodes)
        
        # Add edges based on hyperedge co-membership
        data = hypergraph.incidences.data
        edge_nodes = defaultdict(list)
        
        for row in data.iter_rows(named=True):
            edge_id = row[hypergraph.incidences.edge_column]
            node_id = row[hypergraph.incidences.node_column]
            weight = row.get(weight_column, 1.0) if weight_column else 1.0
            edge_nodes[edge_id].append((node_id, weight))
        
        # Create graph edges
        for edge_id, node_weight_pairs in edge_nodes.items():
            nodes_in_edge = [pair[0] for pair in node_weight_pairs]
            
            # Connect all pairs in clique-like fashion
            for i in range(len(nodes_in_edge)):
                for j in range(i + 1, len(nodes_in_edge)):
                    node1, node2 = nodes_in_edge[i], nodes_in_edge[j]
                    
                    if weight_column:
                        # Weight by minimum weight in the hyperedge
                        w1, w2 = node_weight_pairs[i][1], node_weight_pairs[j][1]
                        edge_weight = min(w1, w2)
                    else:
                        edge_weight = 1.0
                    
                    if G.has_edge(node1, node2):
                        G[node1][node2]['weight'] += edge_weight
                    else:
                        G.add_edge(node1, node2, weight=edge_weight)
        
        # Compute betweenness centrality
        if weight_column:
            centrality_dict = networkx.betweenness_centrality(
                G, weight='weight', normalized=normalize
            )
        else:
            centrality_dict = networkx.betweenness_centrality(
                G, normalized=normalize
            )
        
        # Ensure all nodes are included
        for node in hypergraph.nodes:
            if node not in centrality_dict:
                centrality_dict[node] = 0.0
        
        logger.info(f"Computed betweenness centrality for {len(centrality_dict)} nodes")
        return centrality_dict
        
    except Exception as e:
        logger.error(f"Error computing betweenness centrality: {e}")
        # Fallback to degree centrality
        return weighted_node_centrality(hypergraph, weight_column, normalize)


@performance_monitor
def closeness_centrality(hypergraph,
                        weight_column: Optional[str] = None,
                        normalize: bool = True) -> Dict[str, float]:
    """
    Compute closeness centrality for hypergraph nodes.
    
    Uses shortest path distances in the graph representation of the hypergraph.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights (used as inverse distances)
        normalize: Whether to normalize centrality values
        
    Returns:
        Dictionary mapping node IDs to centrality values
    """
    try:
        if not networkx:
            logger.warning("NetworkX not available, falling back to degree centrality")
            return weighted_node_centrality(hypergraph, weight_column, normalize)
        
        # Convert hypergraph to NetworkX graph (similar to betweenness)
        G = networkx.Graph()
        G.add_nodes_from(hypergraph.nodes)
        
        data = hypergraph.incidences.data
        edge_nodes = defaultdict(list)
        
        for row in data.iter_rows(named=True):
            edge_id = row[hypergraph.incidences.edge_column]
            node_id = row[hypergraph.incidences.node_column]
            weight = row.get(weight_column, 1.0) if weight_column else 1.0
            edge_nodes[edge_id].append((node_id, weight))
        
        # Create graph edges with distances (inverse of weights)
        for edge_id, node_weight_pairs in edge_nodes.items():
            nodes_in_edge = [pair[0] for pair in node_weight_pairs]
            
            for i in range(len(nodes_in_edge)):
                for j in range(i + 1, len(nodes_in_edge)):
                    node1, node2 = nodes_in_edge[i], nodes_in_edge[j]
                    
                    if weight_column:
                        w1, w2 = node_weight_pairs[i][1], node_weight_pairs[j][1]
                        # Use harmonic mean of weights, then invert for distance
                        weight_avg = 2 * w1 * w2 / (w1 + w2) if (w1 + w2) > 0 else 1.0
                        distance = 1.0 / max(weight_avg, 1e-10)
                    else:
                        distance = 1.0
                    
                    if G.has_edge(node1, node2):
                        # Use minimum distance (maximum weight)
                        current_distance = G[node1][node2]['weight']
                        G[node1][node2]['weight'] = min(current_distance, distance)
                    else:
                        G.add_edge(node1, node2, weight=distance)
        
        # Compute closeness centrality
        if len(G.edges()) > 0:
            centrality_dict = networkx.closeness_centrality(
                G, distance='weight' if weight_column else None
            )
        else:
            # No edges, all nodes have equal centrality
            centrality_dict = {node: 1.0/len(hypergraph.nodes) for node in hypergraph.nodes}
        
        # Ensure all nodes are included
        for node in hypergraph.nodes:
            if node not in centrality_dict:
                centrality_dict[node] = 0.0
        
        # Additional normalization if requested
        if normalize and centrality_dict:
            max_centrality = max(centrality_dict.values())
            if max_centrality > 0:
                centrality_dict = {
                    node: centrality / max_centrality 
                    for node, centrality in centrality_dict.items()
                }
        
        logger.info(f"Computed closeness centrality for {len(centrality_dict)} nodes")
        return centrality_dict
        
    except Exception as e:
        logger.error(f"Error computing closeness centrality: {e}")
        # Fallback to degree centrality
        return weighted_node_centrality(hypergraph, weight_column, normalize)


def centrality_comparison(hypergraph,
                         weight_column: Optional[str] = None,
                         centrality_types: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Compare multiple centrality measures for a hypergraph.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        centrality_types: List of centrality types to compute
        
    Returns:
        DataFrame with nodes and their centrality values
    """
    if centrality_types is None:
        centrality_types = ['degree', 'eigenvector', 'betweenness', 'closeness']
    
    results = {'node_id': list(hypergraph.nodes)}
    
    for centrality_type in centrality_types:
        try:
            centrality = hypergraph_centrality(
                hypergraph, centrality_type, weight_column, normalize=True
            )
            results[f'{centrality_type}_centrality'] = [
                centrality.get(node, 0.0) for node in results['node_id']
            ]
        except Exception as e:
            logger.warning(f"Failed to compute {centrality_type} centrality: {e}")
            results[f'{centrality_type}_centrality'] = [0.0] * len(results['node_id'])
    
    return pl.DataFrame(results)