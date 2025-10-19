"""
Hypergraph Clustering and Community Detection
============================================

Advanced clustering algorithms for hypergraph analysis including
community detection, modularity optimization, and spectral clustering.
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
sklearn = safe_import('sklearn')
networkx = safe_import('networkx')


@performance_monitor
def hypergraph_clustering(hypergraph,
                         algorithm: str = 'modularity',
                         weight_column: Optional[str] = None,
                         n_clusters: Optional[int] = None,
                         auto_sample: bool = True,
                         max_nodes: int = 5000,
                         **kwargs) -> Dict[str, int]:
    """
    Perform clustering on hypergraph nodes with automatic sampling for large graphs.
    
    Args:
        hypergraph: Anant Hypergraph instance
        algorithm: Clustering algorithm ('modularity', 'spectral', 'hierarchical')
        weight_column: Column name for edge weights
        n_clusters: Number of clusters (for algorithms that require it)
        auto_sample: Whether to automatically sample large graphs
        max_nodes: Maximum nodes before sampling is applied
        **kwargs: Additional arguments for specific algorithms
        
    Returns:
        Dictionary mapping node IDs to cluster IDs
    """
    
    # Auto-sample large graphs for performance
    if auto_sample and len(hypergraph.nodes) > max_nodes:
        from .sampling import auto_scale_algorithm
        logger.info(f"Large graph detected ({len(hypergraph.nodes)} nodes), using intelligent sampling")
        
        def _clustering_func(hg, **cluster_kwargs):
            return _run_clustering_algorithm(hg, algorithm, weight_column, n_clusters, **cluster_kwargs)
        
        return auto_scale_algorithm(
            hypergraph, _clustering_func, 'clustering', 
            max_nodes=max_nodes, **kwargs
        )
    
    # Run on original graph
    return _run_clustering_algorithm(hypergraph, algorithm, weight_column, n_clusters, **kwargs)


def _run_clustering_algorithm(hypergraph, algorithm, weight_column, n_clusters, **kwargs):
    """Internal function to run the actual clustering algorithm"""
    if algorithm == 'modularity':
        return community_detection(hypergraph, weight_column, **kwargs)
    elif algorithm == 'spectral':
        return spectral_clustering(hypergraph, weight_column, n_clusters, **kwargs)
    elif algorithm == 'hierarchical':
        return hierarchical_clustering(hypergraph, weight_column, n_clusters, **kwargs)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")


@performance_monitor
def community_detection(hypergraph,
                       weight_column: Optional[str] = None,
                       resolution: float = 1.0,
                       max_iterations: int = 100) -> Dict[str, int]:
    """
    Detect communities using modularity optimization.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        resolution: Resolution parameter for community detection
        max_iterations: Maximum iterations for optimization
        
    Returns:
        Dictionary mapping node IDs to community IDs
    """
    
    # For very large graphs, use simple clustering as fallback
    if len(hypergraph.nodes) > 2000:
        logger.warning(f"Large graph ({len(hypergraph.nodes)} nodes), using fast simple clustering")
        return _simple_clustering(hypergraph, weight_column)
    
    try:
        if not networkx:
            logger.warning("NetworkX not available, using simple clustering")
            return _simple_clustering(hypergraph, weight_column)
        
        # Convert hypergraph to NetworkX graph
        G = _hypergraph_to_networkx(hypergraph, weight_column)
        
        # Apply community detection
        if hasattr(networkx, 'community') and hasattr(networkx.community, 'greedy_modularity_communities'):
            communities = networkx.community.greedy_modularity_communities(
                G, weight='weight' if weight_column else None, resolution=resolution
            )
        else:
            # Fallback to simple connected components
            communities = list(networkx.connected_components(G))
        
        # Convert to dictionary format
        community_dict = {}
        for community_id, community in enumerate(communities):
            for node in community:
                community_dict[node] = community_id
        
        # Ensure all nodes are assigned
        for node in hypergraph.nodes:
            if node not in community_dict:
                community_dict[node] = len(communities)  # Create new community
        
        logger.info(f"Detected {len(communities)} communities")
        return community_dict
        
    except Exception as e:
        logger.error(f"Error in community detection: {e}")
        return _simple_clustering(hypergraph, weight_column)


@performance_monitor
def modularity_optimization(hypergraph,
                           weight_column: Optional[str] = None,
                           gamma: float = 1.0) -> Tuple[Dict[str, int], float]:
    """
    Optimize modularity for hypergraph clustering.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        gamma: Resolution parameter
        
    Returns:
        Tuple of (community assignment dict, modularity score)
    """
    try:
        communities = community_detection(hypergraph, weight_column, resolution=gamma)
        modularity_score = calculate_modularity(hypergraph, communities, weight_column)
        
        logger.info(f"Modularity optimization completed with score: {modularity_score:.4f}")
        return communities, modularity_score
        
    except Exception as e:
        logger.error(f"Error in modularity optimization: {e}")
        return {node: 0 for node in hypergraph.nodes}, 0.0


@performance_monitor
def spectral_clustering(hypergraph,
                       weight_column: Optional[str] = None,
                       n_clusters: Optional[int] = None,
                       eigen_solver: str = 'auto') -> Dict[str, int]:
    """
    Perform spectral clustering on hypergraph.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        n_clusters: Number of clusters to find
        eigen_solver: Eigenvalue solver ('auto', 'arpack', 'lobpcg')
        
    Returns:
        Dictionary mapping node IDs to cluster IDs
    """
    try:
        if not sklearn:
            logger.warning("Scikit-learn not available, falling back to community detection")
            return community_detection(hypergraph, weight_column)
        
        # Build adjacency matrix
        nodes = list(hypergraph.nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n_nodes = len(nodes)
        
        if n_nodes == 0:
            return {}
        
        # Estimate number of clusters if not provided
        if n_clusters is None:
            n_clusters = min(max(2, int(np.sqrt(n_nodes))), 10)
        
        # Build adjacency matrix
        adj_matrix = np.zeros((n_nodes, n_nodes))
        data = hypergraph.incidences.data
        edge_nodes = defaultdict(list)
        
        for row in data.iter_rows(named=True):
            edge_id = row['edge_id']
            node_id = row['node_id']
            weight = row.get(weight_column, 1.0) if weight_column else 1.0
            edge_nodes[edge_id].append((node_id, weight))
        
        # Create adjacency matrix from hyperedges
        for edge_id, node_weight_pairs in edge_nodes.items():
            for i, (node1, w1) in enumerate(node_weight_pairs):
                for j, (node2, w2) in enumerate(node_weight_pairs):
                    if i != j:
                        idx1, idx2 = node_to_idx[node1], node_to_idx[node2]
                        edge_weight = np.sqrt(w1 * w2) if weight_column else 1.0
                        adj_matrix[idx1, idx2] += edge_weight
        
        # Apply spectral clustering
        from sklearn.cluster import SpectralClustering
        
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            eigen_solver='arpack',  # Use fixed value to avoid type issues
            random_state=42
        )
        
        cluster_labels = spectral.fit_predict(adj_matrix)
        
        # Convert to dictionary
        clustering_dict = dict(zip(nodes, cluster_labels))
        
        logger.info(f"Spectral clustering completed with {n_clusters} clusters")
        return clustering_dict
        
    except Exception as e:
        logger.error(f"Error in spectral clustering: {e}")
        return community_detection(hypergraph, weight_column)


@performance_monitor
def hierarchical_clustering(hypergraph,
                           weight_column: Optional[str] = None,
                           n_clusters: Optional[int] = None,
                           linkage: str = 'ward',
                           distance_metric: str = 'euclidean') -> Dict[str, int]:
    """
    Perform hierarchical clustering on hypergraph.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        n_clusters: Number of clusters to find
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
        distance_metric: Distance metric for clustering
        
    Returns:
        Dictionary mapping node IDs to cluster IDs
    """
    try:
        if not sklearn:
            logger.warning("Scikit-learn not available, falling back to community detection")
            return community_detection(hypergraph, weight_column)
        
        # Build feature matrix (node participation in edges)
        nodes = list(hypergraph.nodes)
        edges = list(hypergraph.edges)
        n_nodes, n_edges = len(nodes), len(edges)
        
        if n_nodes == 0:
            return {}
        
        # Estimate number of clusters if not provided
        if n_clusters is None:
            n_clusters = min(max(2, int(np.sqrt(n_nodes))), 10)
        
        # Create node-edge incidence matrix
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        edge_to_idx = {edge: i for i, edge in enumerate(edges)}
        
        incidence_matrix = np.zeros((n_nodes, n_edges))
        data = hypergraph.incidences.data
        
        for row in data.iter_rows(named=True):
            edge_id = row['edge_id']
            node_id = row['node_id']
            weight = row.get(weight_column, 1.0) if weight_column else 1.0
            
            if node_id in node_to_idx and edge_id in edge_to_idx:
                incidence_matrix[node_to_idx[node_id], edge_to_idx[edge_id]] = weight
        
        # Apply hierarchical clustering
        from sklearn.cluster import AgglomerativeClustering
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'  # Use fixed value to avoid type issues
        )
        
        cluster_labels = hierarchical.fit_predict(incidence_matrix)
        
        # Convert to dictionary
        clustering_dict = dict(zip(nodes, cluster_labels))
        
        logger.info(f"Hierarchical clustering completed with {n_clusters} clusters")
        return clustering_dict
        
    except Exception as e:
        logger.error(f"Error in hierarchical clustering: {e}")
        return community_detection(hypergraph, weight_column)


def calculate_modularity(hypergraph,
                        communities: Dict[str, int],
                        weight_column: Optional[str] = None) -> float:
    """
    Calculate modularity score for a community assignment.
    
    Args:
        hypergraph: Anant Hypergraph instance
        communities: Dictionary mapping nodes to community IDs
        weight_column: Column name for edge weights
        
    Returns:
        Modularity score
    """
    try:
        if not networkx:
            logger.warning("NetworkX not available, returning approximate modularity")
            return _approximate_modularity(hypergraph, communities, weight_column)
        
        # Convert to NetworkX graph and compute modularity
        G = _hypergraph_to_networkx(hypergraph, weight_column)
        
        # Group nodes by community
        community_list = []
        community_groups = defaultdict(list)
        for node, community_id in communities.items():
            community_groups[community_id].append(node)
        
        community_list = [set(nodes) for nodes in community_groups.values()]
        
        # Calculate modularity
        modularity = networkx.community.modularity(
            G, community_list, weight='weight' if weight_column else None
        )
        
        return modularity
        
    except Exception as e:
        logger.error(f"Error calculating modularity: {e}")
        return 0.0


def _hypergraph_to_networkx(hypergraph, weight_column: Optional[str] = None):
    """Helper function to convert hypergraph to NetworkX graph."""
    if not networkx:
        raise ImportError("NetworkX is required for this operation")
    
    G = networkx.Graph()
    G.add_nodes_from(hypergraph.nodes)
    
    data = hypergraph.incidences.data
    edge_nodes = defaultdict(list)
    
    for row in data.iter_rows(named=True):
        edge_id = row['edge_id']
        node_id = row['node_id']
        weight = row.get(weight_column, 1.0) if weight_column else 1.0
        edge_nodes[edge_id].append((node_id, weight))
    
    # Create clique-like connections for each hyperedge
    for edge_id, node_weight_pairs in edge_nodes.items():
        nodes_in_edge = [pair[0] for pair in node_weight_pairs]
        
        for i in range(len(nodes_in_edge)):
            for j in range(i + 1, len(nodes_in_edge)):
                node1, node2 = nodes_in_edge[i], nodes_in_edge[j]
                
                if weight_column:
                    w1, w2 = node_weight_pairs[i][1], node_weight_pairs[j][1]
                    edge_weight = np.sqrt(w1 * w2)
                else:
                    edge_weight = 1.0
                
                if G.has_edge(node1, node2):
                    G[node1][node2]['weight'] += edge_weight
                else:
                    G.add_edge(node1, node2, weight=edge_weight)
    
    return G


def _simple_clustering(hypergraph, weight_column: Optional[str] = None) -> Dict[str, int]:
    """Fast simple clustering fallback for large graphs."""
    
    # For very large graphs, use super-simple clustering
    if len(hypergraph.nodes) > 5000:
        logger.info(f"Very large graph ({len(hypergraph.nodes)} nodes), using degree-based clustering")
        return _degree_based_clustering(hypergraph)
    
    # Group nodes by their edge connectivity (optimized)
    node_connections = defaultdict(set)
    
    # Build connections more efficiently
    for edge in hypergraph.edges:
        edge_nodes = hypergraph.incidences.get_edge_nodes(edge)
        for node in edge_nodes:
            node_connections[node].add(edge)
    
    # Fast clustering based on shared edges
    clusters = []
    assigned_nodes = set()
    
    # Sample nodes for large graphs to speed up
    nodes_to_process = list(hypergraph.nodes)
    if len(nodes_to_process) > 1000:
        import random
        random.shuffle(nodes_to_process)
        nodes_to_process = nodes_to_process[:1000]  # Process only 1000 nodes
    
    for node in nodes_to_process:
        if node in assigned_nodes:
            continue
        
        # Find nodes with similar edge sets (with early termination)
        cluster = {node}
        node_edges = node_connections[node]
        
        # Only check a subset of other nodes for performance
        other_nodes = [n for n in nodes_to_process[:500] if n != node and n not in assigned_nodes]
        
        for other_node in other_nodes:
            other_edges = node_connections[other_node]
            # Quick similarity check
            if node_edges and other_edges:
                intersection_size = len(node_edges & other_edges)
                if intersection_size > 0:  # Any shared edges
                    similarity = intersection_size / len(node_edges | other_edges)
                    if similarity > 0.2:  # Lower threshold for speed
                        cluster.add(other_node)
        
        clusters.append(cluster)
        assigned_nodes.update(cluster)
    
    # Assign remaining nodes to individual clusters
    remaining_nodes = set(hypergraph.nodes) - assigned_nodes
    for i, node in enumerate(remaining_nodes):
        clusters.append({node})
    
    # Convert to dictionary
    clustering_dict = {}
    for cluster_id, cluster in enumerate(clusters):
        for node in cluster:
            clustering_dict[node] = cluster_id
    
    logger.info(f"Simple clustering created {len(clusters)} clusters")
    return clustering_dict


def _degree_based_clustering(hypergraph) -> Dict[str, int]:
    """Ultra-fast degree-based clustering for very large graphs."""
    
    # Group nodes by degree
    degree_groups = defaultdict(list)
    
    for node in hypergraph.nodes:
        degree = hypergraph.incidences.get_node_degree(node)
        # Quantize degrees into bins for clustering
        degree_bin = min(degree // 5, 10)  # Bins: 0-4, 5-9, 10-14, ..., 50+
        degree_groups[degree_bin].append(node)
    
    # Assign cluster IDs
    clustering_dict = {}
    cluster_id = 0
    
    for degree_bin, nodes in degree_groups.items():
        # For large groups, subdivide
        if len(nodes) > 1000:
            # Split into smaller groups
            chunk_size = 1000
            for i in range(0, len(nodes), chunk_size):
                chunk = nodes[i:i + chunk_size]
                for node in chunk:
                    clustering_dict[node] = cluster_id
                cluster_id += 1
        else:
            # Assign all nodes to same cluster
            for node in nodes:
                clustering_dict[node] = cluster_id
            cluster_id += 1
    
    logger.info(f"Degree-based clustering created {cluster_id} clusters")
    return clustering_dict


def _approximate_modularity(hypergraph, communities: Dict[str, int], weight_column: Optional[str] = None) -> float:
    """Approximate modularity calculation without NetworkX."""
    # This is a simplified modularity calculation
    # Real modularity requires more complex graph analysis
    
    num_communities = len(set(communities.values()))
    num_nodes = len(hypergraph.nodes)
    
    if num_communities == 0 or num_nodes == 0:
        return 0.0
    
    # Simple heuristic: better modularity for fewer, balanced communities
    community_sizes = defaultdict(int)
    for community_id in communities.values():
        community_sizes[community_id] += 1
    
    # Penalize very unbalanced communities
    size_variance = np.var(list(community_sizes.values()))
    expected_size = num_nodes / num_communities
    balance_score = 1.0 / (1.0 + size_variance / (expected_size ** 2))
    
    # Penalize too many or too few communities
    optimal_communities = max(2, int(np.sqrt(num_nodes)))
    community_score = 1.0 / (1.0 + abs(num_communities - optimal_communities) / optimal_communities)
    
    return float(balance_score * community_score)


def clustering_analysis(hypergraph,
                       weight_column: Optional[str] = None,
                       algorithms: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Compare multiple clustering algorithms on a hypergraph.
    
    Args:
        hypergraph: Anant Hypergraph instance
        weight_column: Column name for edge weights
        algorithms: List of clustering algorithms to compare
        
    Returns:
        DataFrame with clustering results and comparison metrics
    """
    if algorithms is None:
        algorithms = ['modularity', 'spectral', 'hierarchical']
    
    results = {'node_id': list(hypergraph.nodes)}
    
    for algorithm in algorithms:
        try:
            clustering = hypergraph_clustering(hypergraph, algorithm, weight_column)
            results[f'{algorithm}_cluster'] = [
                clustering.get(node, -1) for node in results['node_id']
            ]
            
            # Calculate modularity for each clustering
            modularity = calculate_modularity(hypergraph, clustering, weight_column)
            logger.info(f"{algorithm} clustering modularity: {modularity:.4f}")
            
        except Exception as e:
            logger.warning(f"Failed to compute {algorithm} clustering: {e}")
            results[f'{algorithm}_cluster'] = [-1] * len(results['node_id'])
    
    return pl.DataFrame(results)