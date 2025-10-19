"""
Enhanced Centrality Measures for Hypergraphs
===========================================

Extended centrality measures beyond basic degree centrality,
including betweenness, closeness, and eigenvector centrality
adapted for hypergraph structures.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from collections import defaultdict, deque
from ..utils.decorators import performance_monitor
from ..utils.extras import safe_import
from .sampling import auto_scale_algorithm

logger = logging.getLogger(__name__)

# Optional dependencies
scipy = safe_import('scipy')
networkx = safe_import('networkx')


@performance_monitor
def enhanced_centrality_analysis(hypergraph,
                               measures: Optional[List[str]] = None,
                               weight_column: Optional[str] = None,
                               sample_large_graphs: bool = True,
                               max_nodes: int = 2000) -> pl.DataFrame:
    """
    Compute multiple centrality measures for hypergraph nodes
    
    Args:
        hypergraph: Anant Hypergraph instance
        measures: List of centrality measures to compute
        weight_column: Column name for edge weights
        sample_large_graphs: Whether to sample large graphs for performance
        max_nodes: Maximum nodes before sampling
        
    Returns:
        DataFrame with centrality scores for all nodes
    """
    
    if measures is None:
        measures = ['degree', 'betweenness', 'closeness', 'eigenvector', 'harmonic']
    
    # Initialize results with node IDs
    results = {'node_id': list(hypergraph.nodes)}
    
    # Auto-scale for large graphs
    if sample_large_graphs and len(hypergraph.nodes) > max_nodes:
        logger.info(f"Large graph detected ({len(hypergraph.nodes)} nodes), using sampling for centrality analysis")
    
    # Compute each centrality measure
    for measure in measures:
        try:
            if measure == 'degree':
                scores = degree_centrality(hypergraph, weight_column)
            elif measure == 'betweenness':
                if sample_large_graphs:
                    scores = auto_scale_algorithm(
                        hypergraph, betweenness_centrality, 'centrality',
                        max_nodes=max_nodes, weight_column=weight_column
                    )
                else:
                    scores = betweenness_centrality(hypergraph, weight_column)
            elif measure == 'closeness':
                if sample_large_graphs:
                    scores = auto_scale_algorithm(
                        hypergraph, closeness_centrality, 'centrality',
                        max_nodes=max_nodes, weight_column=weight_column
                    )
                else:
                    scores = closeness_centrality(hypergraph, weight_column)
            elif measure == 'eigenvector':
                if sample_large_graphs:
                    scores = auto_scale_algorithm(
                        hypergraph, eigenvector_centrality, 'centrality',
                        max_nodes=max_nodes, weight_column=weight_column
                    )
                else:
                    scores = eigenvector_centrality(hypergraph, weight_column)
            elif measure == 'harmonic':
                if sample_large_graphs:
                    scores = auto_scale_algorithm(
                        hypergraph, harmonic_centrality, 'centrality',
                        max_nodes=max_nodes, weight_column=weight_column
                    )
                else:
                    scores = harmonic_centrality(hypergraph, weight_column)
            else:
                logger.warning(f"Unknown centrality measure: {measure}")
                continue
            
            # Add scores to results
            results[f'{measure}_centrality'] = [
                scores.get(node, 0.0) for node in results['node_id']
            ]
            
            logger.info(f"Computed {measure} centrality for {len(scores)} nodes")
            
        except Exception as e:
            logger.error(f"Error computing {measure} centrality: {e}")
            # Add zeros for failed computation
            results[f'{measure}_centrality'] = [0.0] * len(results['node_id'])
    
    return pl.DataFrame(results)


@performance_monitor
def degree_centrality(hypergraph,
                     weight_column: Optional[str] = None,
                     normalized: bool = True) -> Dict[str, float]:
    """
    Compute degree centrality for hypergraph nodes
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        normalized: Whether to normalize by maximum possible degree
        
    Returns:
        Dictionary mapping node IDs to centrality scores
    """
    
    centrality = {}
    
    for node in hypergraph.nodes:
        if weight_column:
            # Weighted degree centrality
            incident_edges = hypergraph.incidences.get_node_edges(node)
            weighted_degree = 0.0
            
            for edge in incident_edges:
                weight = hypergraph.incidences.get_weight(edge, node)
                if weight is not None:
                    weighted_degree += weight
            
            centrality[node] = weighted_degree
        else:
            # Simple degree centrality
            centrality[node] = float(hypergraph.incidences.get_node_degree(node))
    
    # Normalize if requested
    if normalized and centrality:
        max_degree = max(centrality.values())
        if max_degree > 0:
            centrality = {node: score / max_degree for node, score in centrality.items()}
    
    return centrality


@performance_monitor
def betweenness_centrality(hypergraph,
                          weight_column: Optional[str] = None,
                          normalized: bool = True,
                          approximate: bool = False,
                          k_samples: int = 100) -> Dict[str, float]:
    """
    Compute betweenness centrality for hypergraph nodes
    
    Uses hypergraph shortest paths to measure how often a node lies
    on shortest paths between other nodes.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        normalized: Whether to normalize scores
        approximate: Use sampling for large graphs
        k_samples: Number of samples for approximation
        
    Returns:
        Dictionary mapping node IDs to centrality scores
    """
    
    nodes = list(hypergraph.nodes)
    n_nodes = len(nodes)
    
    if n_nodes <= 1:
        return {node: 0.0 for node in nodes}
    
    # Initialize centrality scores
    centrality = {node: 0.0 for node in nodes}
    
    # For large graphs, use sampling
    if approximate and n_nodes > k_samples:
        sample_nodes = np.random.choice(nodes, size=k_samples, replace=False)
        scaling_factor = n_nodes / k_samples
    else:
        sample_nodes = nodes
        scaling_factor = 1.0
    
    # Compute shortest paths and accumulate betweenness
    for source in sample_nodes:
        # Single-source shortest paths
        distances, predecessors = _single_source_shortest_paths(
            hypergraph, source, weight_column
        )
        
        # Accumulate betweenness scores
        for target in nodes:
            if target != source and target in distances:
                paths = _get_all_shortest_paths(predecessors, source, target)
                
                for path in paths:
                    # Count intermediate nodes
                    for i in range(1, len(path) - 1):
                        intermediate = path[i]
                        centrality[intermediate] += scaling_factor / len(paths)
    
    # Normalize if requested
    if normalized and n_nodes > 2:
        normalization = (n_nodes - 1) * (n_nodes - 2)
        centrality = {node: score / normalization for node, score in centrality.items()}
    
    return centrality


@performance_monitor
def closeness_centrality(hypergraph,
                        weight_column: Optional[str] = None,
                        normalized: bool = True) -> Dict[str, float]:
    """
    Compute closeness centrality for hypergraph nodes
    
    Measures how close a node is to all other nodes in the hypergraph.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        normalized: Whether to normalize scores
        
    Returns:
        Dictionary mapping node IDs to centrality scores
    """
    
    nodes = list(hypergraph.nodes)
    n_nodes = len(nodes)
    centrality = {}
    
    for node in nodes:
        # Compute shortest paths from this node to all others
        distances, _ = _single_source_shortest_paths(hypergraph, node, weight_column)
        
        # Sum of distances to reachable nodes
        reachable_distances = [d for d in distances.values() if d < float('inf')]
        
        if len(reachable_distances) > 1:  # Exclude self
            total_distance = sum(reachable_distances) - distances.get(node, 0)
            n_reachable = len(reachable_distances) - 1
            
            if total_distance > 0:
                closeness = n_reachable / total_distance
                
                # Normalize by fraction of nodes reachable
                if normalized:
                    closeness *= (n_reachable / (n_nodes - 1))
                
                centrality[node] = closeness
            else:
                centrality[node] = 0.0
        else:
            centrality[node] = 0.0
    
    return centrality


@performance_monitor
def eigenvector_centrality(hypergraph,
                          weight_column: Optional[str] = None,
                          max_iterations: int = 100,
                          tolerance: float = 1e-6) -> Dict[str, float]:
    """
    Compute eigenvector centrality for hypergraph nodes
    
    A node's centrality is proportional to the sum of centralities
    of nodes connected to it through hyperedges.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        max_iterations: Maximum iterations for power method
        tolerance: Convergence tolerance
        
    Returns:
        Dictionary mapping node IDs to centrality scores
    """
    
    nodes = list(hypergraph.nodes)
    n_nodes = len(nodes)
    
    if n_nodes == 0:
        return {}
    
    if n_nodes == 1:
        return {nodes[0]: 1.0}
    
    # Build adjacency matrix
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    adjacency = np.zeros((n_nodes, n_nodes))
    
    # Create adjacency matrix from hyperedges
    for edge in hypergraph.edges:
        edge_nodes = hypergraph.incidences.get_edge_nodes(edge)
        
        # For each pair of nodes in the hyperedge
        for i, node1 in enumerate(edge_nodes):
            for j, node2 in enumerate(edge_nodes):
                if i != j:
                    idx1, idx2 = node_to_idx[node1], node_to_idx[node2]
                    
                    if weight_column:
                        w1 = hypergraph.incidences.get_weight(edge, node1) or 1.0
                        w2 = hypergraph.incidences.get_weight(edge, node2) or 1.0
                        weight = np.sqrt(w1 * w2)
                    else:
                        weight = 1.0
                    
                    adjacency[idx1, idx2] += weight
    
    # Power iteration method
    x = np.ones(n_nodes) / n_nodes  # Initial vector
    
    for iteration in range(max_iterations):
        x_new = adjacency.dot(x)
        
        # Normalize
        norm = np.linalg.norm(x_new)
        if norm > 0:
            x_new = x_new / norm
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tolerance:
            logger.debug(f"Eigenvector centrality converged after {iteration + 1} iterations")
            break
        
        x = x_new
    
    # Create result dictionary
    centrality = {node: float(x[node_to_idx[node]]) for node in nodes}
    
    return centrality


@performance_monitor
def harmonic_centrality(hypergraph,
                       weight_column: Optional[str] = None,
                       normalized: bool = True) -> Dict[str, float]:
    """
    Compute harmonic centrality for hypergraph nodes
    
    Harmonic centrality uses the harmonic mean of distances,
    which handles disconnected graphs better than closeness centrality.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        normalized: Whether to normalize scores
        
    Returns:
        Dictionary mapping node IDs to centrality scores
    """
    
    nodes = list(hypergraph.nodes)
    n_nodes = len(nodes)
    centrality = {}
    
    for node in nodes:
        # Compute shortest paths from this node to all others
        distances, _ = _single_source_shortest_paths(hypergraph, node, weight_column)
        
        # Sum of reciprocals of distances (harmonic mean)
        harmonic_sum = 0.0
        for target, distance in distances.items():
            if target != node and distance > 0 and distance < float('inf'):
                harmonic_sum += 1.0 / distance
        
        if normalized and n_nodes > 1:
            harmonic_sum /= (n_nodes - 1)
        
        centrality[node] = harmonic_sum
    
    return centrality


def _single_source_shortest_paths(hypergraph,
                                 source: str,
                                 weight_column: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    Compute shortest paths from a source node using modified Dijkstra's algorithm
    
    Returns:
        Tuple of (distances dict, predecessors dict)
    """
    
    nodes = set(hypergraph.nodes)
    distances = {node: float('inf') for node in nodes}
    predecessors = {node: [] for node in nodes}
    distances[source] = 0.0
    
    # Priority queue: (distance, node)
    queue = [(0.0, source)]
    visited = set()
    
    while queue:
        current_dist, current_node = min(queue)
        queue.remove((current_dist, current_node))
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        # Get neighbors through hyperedges
        neighbors = _get_hypergraph_neighbors(hypergraph, current_node, weight_column)
        
        for neighbor, edge_weight in neighbors.items():
            if neighbor not in visited:
                new_distance = current_dist + edge_weight
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = [current_node]
                    queue.append((new_distance, neighbor))
                elif new_distance == distances[neighbor] and current_node not in predecessors[neighbor]:
                    predecessors[neighbor].append(current_node)
    
    return distances, predecessors


def _get_hypergraph_neighbors(hypergraph,
                             node: str,
                             weight_column: Optional[str] = None) -> Dict[str, float]:
    """Get neighbors of a node with edge weights"""
    
    neighbors = {}
    incident_edges = hypergraph.incidences.get_node_edges(node)
    
    for edge in incident_edges:
        edge_nodes = hypergraph.incidences.get_edge_nodes(edge)
        
        for neighbor in edge_nodes:
            if neighbor != node:
                if weight_column:
                    # Use minimum weight among node-edge connections
                    node_weight = hypergraph.incidences.get_weight(edge, node) or 1.0
                    neighbor_weight = hypergraph.incidences.get_weight(edge, neighbor) or 1.0
                    edge_weight = min(node_weight, neighbor_weight)
                else:
                    edge_weight = 1.0
                
                # Take minimum weight if multiple edges connect the same nodes
                if neighbor not in neighbors or edge_weight < neighbors[neighbor]:
                    neighbors[neighbor] = edge_weight
    
    return neighbors


def _get_all_shortest_paths(predecessors: Dict[str, List[str]],
                           source: str,
                           target: str) -> List[List[str]]:
    """Get all shortest paths from source to target"""
    
    if target == source:
        return [[source]]
    
    if not predecessors[target]:
        return []
    
    paths = []
    
    def _build_paths(current_target, current_path):
        if current_target == source:
            paths.append([source] + current_path)
            return
        
        for pred in predecessors[current_target]:
            _build_paths(pred, [current_target] + current_path)
    
    _build_paths(target, [])
    
    return paths


@performance_monitor
def centrality_ranking(centrality_scores: Dict[str, float],
                      top_k: Optional[int] = None) -> List[Tuple[str, float]]:
    """
    Rank nodes by centrality scores
    
    Args:
        centrality_scores: Dictionary of node -> centrality score
        top_k: Number of top nodes to return (all if None)
        
    Returns:
        List of (node_id, score) tuples sorted by score (descending)
    """
    
    ranked = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
    
    if top_k is not None:
        ranked = ranked[:top_k]
    
    return ranked


@performance_monitor
def centrality_comparison(hypergraph,
                         measures: Optional[List[str]] = None,
                         weight_column: Optional[str] = None) -> pl.DataFrame:
    """
    Compare multiple centrality measures and identify consensus rankings
    
    Args:
        hypergraph: Anant Hypergraph instance
        measures: List of centrality measures to compare
        weight_column: Column name for edge weights
        
    Returns:
        DataFrame with comparative centrality analysis
    """
    
    # Get centrality scores
    centrality_df = enhanced_centrality_analysis(hypergraph, measures, weight_column)
    
    # Add ranking columns
    for measure in measures or ['degree', 'betweenness', 'closeness', 'eigenvector']:
        col_name = f'{measure}_centrality'
        if col_name in centrality_df.columns:
            # Add ranking column
            rank_col = f'{measure}_rank'
            centrality_df = centrality_df.with_columns(
                pl.col(col_name).rank(method='ordinal', descending=True).alias(rank_col)
            )
    
    # Calculate average rank for consensus
    rank_columns = [col for col in centrality_df.columns if col.endswith('_rank')]
    if rank_columns:
        centrality_df = centrality_df.with_columns(
            pl.mean_horizontal(rank_columns).alias('avg_rank')
        )
        
        # Sort by average rank
        centrality_df = centrality_df.sort('avg_rank')
    
    return centrality_df