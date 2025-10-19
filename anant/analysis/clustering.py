"""
Clustering and Community Detection for Hypergraphs

This module implements advanced clustering algorithms adapted for hypergraphs,
including overlapping community detection, multi-resolution analysis, and
enhanced modularity optimization utilizing Polars for high-performance computation.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from ..classes.hypergraph import Hypergraph
from .spectral import node_laplacian, hypergraph_laplacian


def spectral_clustering(
    hg: Hypergraph,
    n_clusters: int,
    method: str = "node",
    random_state: Optional[int] = None
) -> Dict[str, int]:
    """
    Perform spectral clustering on hypergraph.
    
    Args:
        hg: Hypergraph instance
        n_clusters: Number of clusters to find
        method: "node" for node clustering, "edge" for edge clustering
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary mapping node/edge IDs to cluster labels
    """
    try:
        from sklearn.cluster import SpectralClustering
    except ImportError:
        raise ImportError("scikit-learn is required for spectral clustering")
    
    if method == "node":
        # Get node Laplacian
        laplacian = node_laplacian(hg)
        entities = list(hg.nodes)
    elif method == "edge":
        # Use hypergraph Laplacian for edge clustering
        laplacian = hypergraph_laplacian(hg)
        entities = list(hg.edges)
    else:
        raise ValueError("method must be 'node' or 'edge'")
    
    if len(entities) < n_clusters:
        # Not enough entities to cluster
        return {entity: 0 for entity in entities}
    
    # Perform spectral clustering
    clusterer = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=random_state
    )
    
    # Convert Laplacian to affinity matrix (similarity)
    # Use exponential transformation: exp(-beta * distance)
    beta = 1.0
    affinity = np.exp(-beta * laplacian)
    
    cluster_labels = clusterer.fit_predict(affinity)
    
    return dict(zip(entities, cluster_labels))


def modularity_clustering(
    hg: Hypergraph,
    resolution: float = 1.0,
    max_iterations: int = 100
) -> Dict[str, int]:
    """
    Community detection using modularity optimization.
    
    Implements a hypergraph extension of modularity-based clustering.
    
    Args:
        hg: Hypergraph instance
        resolution: Resolution parameter for modularity
        max_iterations: Maximum iterations for optimization
        
    Returns:
        Dictionary mapping node IDs to cluster labels
    """
    nodes = list(hg.nodes)
    n = len(nodes)
    
    if n <= 1:
        return {node: 0 for node in nodes}
    
    # Initialize each node to its own community
    communities = {node: i for i, node in enumerate(nodes)}
    
    # Get incidence data
    incidence_df = hg.incidences.data
    
    # Calculate node degrees
    node_degrees = (
        incidence_df
        .group_by("node_id")
        .agg(pl.count("edge_id").alias("degree"))
    )
    degrees = dict(zip(
        node_degrees["node_id"].to_list(),
        node_degrees["degree"].to_list()
    ))
    
    # Calculate edge sizes
    edge_sizes = (
        incidence_df
        .group_by("edge_id")
        .agg(pl.count("node_id").alias("size"))
    )
    edge_size_dict = dict(zip(
        edge_sizes["edge_id"].to_list(),
        edge_sizes["size"].to_list()
    ))
    
    # Total number of incidences
    total_incidences = hg.num_edges
    
    def compute_modularity():
        """Compute current modularity."""
        modularity = 0.0
        
        # Group nodes by community
        community_nodes = {}
        for node, comm in communities.items():
            if comm not in community_nodes:
                community_nodes[comm] = []
            community_nodes[comm].append(node)
        
        for comm_nodes in community_nodes.values():
            if len(comm_nodes) <= 1:
                continue
                
            # Calculate within-community connections
            within_edges = 0
            expected_edges = 0
            
            for edge_id in hg.edges:
                edge_nodes = (
                    incidence_df
                    .filter(pl.col("edge_id") == edge_id)
                    .select("node_id")
                    .to_series()
                    .to_list()
                )
                
                # Count nodes in this community that are in this edge
                comm_nodes_in_edge = len([n for n in edge_nodes if n in comm_nodes])
                
                if comm_nodes_in_edge >= 2:
                    # Multiple community nodes in this edge
                    edge_size = edge_size_dict[edge_id]
                    contribution = comm_nodes_in_edge * (comm_nodes_in_edge - 1) / (edge_size * (edge_size - 1))
                    within_edges += contribution
            
            # Calculate expected edges based on degree sequence
            comm_degree_sum = sum(degrees.get(node, 0) for node in comm_nodes)
            expected_edges += (comm_degree_sum * comm_degree_sum) / (2 * total_incidences)
            
            modularity += within_edges - resolution * expected_edges
        
        return modularity / total_incidences
    
    # Optimization loop
    best_modularity = compute_modularity()
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for node in nodes:
            current_comm = communities[node]
            best_comm = current_comm
            best_delta = 0.0
            
            # Try moving node to neighboring communities
            neighbors = set()
            node_edges = (
                incidence_df
                .filter(pl.col("node_id") == node)
                .select("edge_id")
                .to_series()
                .to_list()
            )
            
            for edge in node_edges:
                edge_nodes = (
                    incidence_df
                    .filter(pl.col("edge_id") == edge)
                    .select("node_id")
                    .to_series()
                    .to_list()
                )
                neighbors.update(edge_nodes)
            
            neighbor_comms = {communities[neighbor] for neighbor in neighbors if neighbor != node}
            
            for target_comm in neighbor_comms:
                if target_comm == current_comm:
                    continue
                
                # Try moving node to target community
                communities[node] = target_comm
                new_modularity = compute_modularity()
                delta = new_modularity - best_modularity
                
                if delta > best_delta:
                    best_delta = delta
                    best_comm = target_comm
                
                # Restore original community
                communities[node] = current_comm
            
            # Apply best move if it improves modularity
            if best_delta > 0:
                communities[node] = best_comm
                best_modularity += best_delta
                improved = True
    
    # Relabel communities to be consecutive integers
    unique_comms = list(set(communities.values()))
    comm_relabel = {old: new for new, old in enumerate(unique_comms)}
    
    return {node: comm_relabel[comm] for node, comm in communities.items()}


def hierarchical_clustering(
    hg: Hypergraph,
    n_clusters: int,
    linkage: str = "ward",
    metric: str = "euclidean"
) -> Dict[str, int]:
    """
    Perform hierarchical clustering on hypergraph nodes.
    
    Uses embedding based on shared edge relationships.
    
    Args:
        hg: Hypergraph instance
        n_clusters: Number of clusters to find
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
        metric: Distance metric for clustering
        
    Returns:
        Dictionary mapping node IDs to cluster labels
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        raise ImportError("scikit-learn is required for hierarchical clustering")
    
    nodes = list(hg.nodes)
    edges = list(hg.edges)
    
    if len(nodes) < n_clusters:
        return {node: 0 for node in nodes}
    
    # Create node-edge incidence matrix
    incidence_matrix = np.zeros((len(nodes), len(edges)))
    
    incidence_df = hg.incidences.data
    for i, node in enumerate(nodes):
        node_edges = (
            incidence_df
            .filter(pl.col("node_id") == node)
            .select("edge_id")
            .to_series()
            .to_list()
        )
        for edge in node_edges:
            j = edges.index(edge)
            incidence_matrix[i, j] = 1
    
    # Perform hierarchical clustering
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,  # type: ignore
        metric=metric
    )
    
    cluster_labels = clusterer.fit_predict(incidence_matrix)
    
    return dict(zip(nodes, cluster_labels))


def community_detection(
    hg: Hypergraph,
    method: str = "modularity",
    **kwargs
) -> Dict[str, int]:
    """
    Unified interface for community detection.
    
    Args:
        hg: Hypergraph instance
        method: Detection method ('modularity', 'spectral', 'hierarchical')
        **kwargs: Additional arguments for specific methods
        
    Returns:
        Dictionary mapping node IDs to community labels
    """
    if method == "modularity":
        return modularity_clustering(hg, **kwargs)
    elif method == "spectral":
        # Default to reasonable number of clusters if not specified
        if "n_clusters" not in kwargs:
            kwargs["n_clusters"] = min(10, max(2, len(hg.nodes) // 10))
        return spectral_clustering(hg, **kwargs)
    elif method == "hierarchical":
        # Default to reasonable number of clusters if not specified
        if "n_clusters" not in kwargs:
            kwargs["n_clusters"] = min(10, max(2, len(hg.nodes) // 10))
        return hierarchical_clustering(hg, **kwargs)
    else:
        raise ValueError(f"Unknown community detection method: {method}")


def evaluate_clustering(
    hg: Hypergraph,
    clusters: Dict[str, int]
) -> Dict[str, float]:
    """
    Evaluate clustering quality using various metrics.
    
    Args:
        hg: Hypergraph instance
        clusters: Dictionary mapping node IDs to cluster labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    nodes = list(clusters.keys())
    n_clusters = len(set(clusters.values()))
    
    # Calculate within-cluster vs between-cluster edge density
    incidence_df = hg.incidences.data
    
    within_cluster_edges = 0
    between_cluster_edges = 0
    total_edges = 0
    
    for edge_id in hg.edges:
        edge_nodes = (
            incidence_df
            .filter(pl.col("edge_id") == edge_id)
            .select("node_id")
            .to_series()
            .to_list()
        )
        
        if len(edge_nodes) >= 2:
            total_edges += 1
            edge_clusters = [clusters[node] for node in edge_nodes if node in clusters]
            
            if len(set(edge_clusters)) == 1:
                # All nodes in same cluster
                within_cluster_edges += 1
            else:
                # Nodes span multiple clusters
                between_cluster_edges += 1
    
    # Calculate metrics
    if total_edges > 0:
        within_ratio = within_cluster_edges / total_edges
        between_ratio = between_cluster_edges / total_edges
    else:
        within_ratio = 0.0
        between_ratio = 0.0
    
    # Calculate cluster size statistics
    cluster_sizes = {}
    for cluster_id in clusters.values():
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
    
    size_values = list(cluster_sizes.values())
    avg_cluster_size = np.mean(size_values)
    cluster_size_std = np.std(size_values)
    
    return {
        "n_clusters": n_clusters,
        "within_cluster_edge_ratio": float(within_ratio),
        "between_cluster_edge_ratio": float(between_ratio),
        "average_cluster_size": float(avg_cluster_size),
        "cluster_size_std": float(cluster_size_std),
        "modularity_score": _calculate_modularity(hg, clusters)
    }


def _calculate_modularity(hg: Hypergraph, clusters: Dict[str, int]) -> float:
    """Calculate modularity score for given clustering."""
    # Simplified modularity calculation
    incidence_df = hg.incidences.data
    total_incidences = hg.num_edges
    
    if total_incidences == 0:
        return 0.0
    
    # Group nodes by cluster
    cluster_nodes = {}
    for node, cluster_id in clusters.items():
        if cluster_id not in cluster_nodes:
            cluster_nodes[cluster_id] = []
        cluster_nodes[cluster_id].append(node)
    
    modularity = 0.0
    
    for cluster_id, nodes_in_cluster in cluster_nodes.items():
        if len(nodes_in_cluster) <= 1:
            continue
        
        # Count edges within this cluster
        within_edges = 0
        for edge_id in hg.edges:
            edge_nodes = (
                incidence_df
                .filter(pl.col("edge_id") == edge_id)
                .select("node_id")
                .to_series()
                .to_list()
            )
            
            nodes_in_edge_and_cluster = [n for n in edge_nodes if n in nodes_in_cluster]
            if len(nodes_in_edge_and_cluster) >= 2:
                within_edges += 1
        
        # Calculate expected edges (simplified)
        cluster_degree_sum = 0
        for node in nodes_in_cluster:
            node_degree = (
                incidence_df
                .filter(pl.col("node_id") == node)
                .height
            )
            cluster_degree_sum += node_degree
        
        expected_edges = (cluster_degree_sum * cluster_degree_sum) / (2 * total_incidences)
        modularity += within_edges - expected_edges
    
    return modularity / total_incidences


def overlapping_community_detection(
    hg: Hypergraph,
    max_communities: int = 10,
    alpha: float = 0.5,
    max_iterations: int = 100
) -> Dict[str, List[int]]:
    """
    Detect overlapping communities in hypergraph.
    
    Nodes can belong to multiple communities. Uses a probabilistic
    approach based on edge membership patterns.
    
    Args:
        hg: Hypergraph instance
        max_communities: Maximum number of communities to detect
        alpha: Parameter controlling community overlap (0-1)
        max_iterations: Maximum optimization iterations
        
    Returns:
        Dictionary mapping node IDs to list of community IDs
    """
    nodes = list(hg.nodes)
    edges = list(hg.edges)
    n_nodes = len(nodes)
    n_edges = len(edges)
    
    if n_nodes <= 1:
        return {node: [0] for node in nodes}
    
    # Initialize node-community membership probabilities
    np.random.seed(42)  # For reproducibility
    node_comm_prob = np.random.rand(n_nodes, max_communities)
    node_comm_prob = node_comm_prob / node_comm_prob.sum(axis=1, keepdims=True)
    
    # Get incidence matrix representation
    incidence_df = hg.incidences.data
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edge_to_idx = {edge: i for i, edge in enumerate(edges)}
    
    # Build edge-node mapping
    edge_nodes = {}
    for edge in edges:
        edge_nodes[edge] = (
            incidence_df
            .filter(pl.col("edge_id") == edge)
            .select("node_id")
            .to_series()
            .to_list()
        )
    
    # Optimization loop
    for iteration in range(max_iterations):
        old_prob = node_comm_prob.copy()
        
        # Update community probabilities for each node
        for i, node in enumerate(nodes):
            # Get edges containing this node
            node_edges = [edge for edge in edges if node in edge_nodes[edge]]
            
            if not node_edges:
                continue
            
            # Calculate new probabilities based on edge communities
            new_probs = np.zeros(max_communities)
            
            for edge in node_edges:
                edge_node_indices = [node_to_idx[n] for n in edge_nodes[edge]]
                
                # Calculate edge-community affinity
                for c in range(max_communities):
                    edge_comm_strength = np.mean([node_comm_prob[idx, c] for idx in edge_node_indices])
                    new_probs[c] += edge_comm_strength
            
            # Normalize and apply smoothing
            if new_probs.sum() > 0:
                new_probs = new_probs / new_probs.sum()
                node_comm_prob[i] = alpha * new_probs + (1 - alpha) * node_comm_prob[i]
        
        # Check convergence
        if np.allclose(old_prob, node_comm_prob, atol=1e-6):
            break
    
    # Extract community assignments (threshold-based)
    threshold = 1.0 / max_communities  # Minimum probability to be in community
    overlapping_communities = {}
    
    for i, node in enumerate(nodes):
        communities = [c for c in range(max_communities) if node_comm_prob[i, c] > threshold]
        if not communities:  # Assign to highest probability community
            communities = [np.argmax(node_comm_prob[i])]
        overlapping_communities[node] = communities
    
    return overlapping_communities


def multi_resolution_clustering(
    hg: Hypergraph,
    resolution_range: Tuple[float, float] = (0.1, 2.0),
    n_resolutions: int = 10,
    method: str = "modularity"
) -> Dict[float, Dict[str, int]]:
    """
    Perform multi-resolution community detection.
    
    Explores community structure at different resolution scales
    to understand hierarchical organization.
    
    Args:
        hg: Hypergraph instance
        resolution_range: Range of resolution parameters to explore
        n_resolutions: Number of resolution values to test
        method: Clustering method ("modularity" or "spectral")
        
    Returns:
        Dictionary mapping resolution values to community assignments
    """
    resolutions = np.linspace(resolution_range[0], resolution_range[1], n_resolutions)
    multi_res_results = {}
    
    for resolution in resolutions:
        if method == "modularity":
            communities = modularity_clustering(hg, resolution=resolution)
        elif method == "spectral":
            # Estimate number of clusters based on resolution
            n_clusters = max(2, min(len(hg.nodes) // 2, int(10 / resolution)))
            communities = spectral_clustering(hg, n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        multi_res_results[resolution] = communities
    
    return multi_res_results


def consensus_clustering(
    hg: Hypergraph,
    n_runs: int = 10,
    n_clusters: int = 5,
    methods: List[str] = ["spectral", "modularity"]
) -> Dict[str, int]:
    """
    Perform consensus clustering across multiple methods and runs.
    
    Combines results from multiple clustering runs to find
    stable community structure.
    
    Args:
        hg: Hypergraph instance
        n_runs: Number of clustering runs per method
        n_clusters: Number of clusters for methods that require it
        methods: List of clustering methods to use
        
    Returns:
        Dictionary mapping node IDs to consensus cluster labels
    """
    nodes = list(hg.nodes)
    n_nodes = len(nodes)
    
    if n_nodes <= 1:
        return {node: 0 for node in nodes}
    
    # Collect all clustering results
    all_clusterings = []
    
    for method in methods:
        for run in range(n_runs):
            try:
                if method == "spectral":
                    clustering = spectral_clustering(hg, n_clusters=n_clusters, random_state=run)
                elif method == "modularity":
                    clustering = modularity_clustering(hg, max_iterations=50)
                elif method == "hierarchical":
                    clustering = hierarchical_clustering(hg, n_clusters=n_clusters)
                else:
                    continue
                
                all_clusterings.append(clustering)
            except Exception:
                # Skip failed clustering attempts
                continue
    
    if not all_clusterings:
        # Fallback to simple clustering
        return {node: 0 for node in nodes}
    
    # Build consensus matrix
    consensus_matrix = np.zeros((n_nodes, n_nodes))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    for clustering in all_clusterings:
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i <= j and node1 in clustering and node2 in clustering:
                    if clustering[node1] == clustering[node2]:
                        consensus_matrix[i, j] += 1
                        consensus_matrix[j, i] += 1
    
    # Normalize consensus matrix
    consensus_matrix = consensus_matrix / len(all_clusterings)
    
    # Perform clustering on consensus matrix
    try:
        from sklearn.cluster import SpectralClustering
        clusterer = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        consensus_labels = clusterer.fit_predict(consensus_matrix)
        
        return {node: int(label) for node, label in zip(nodes, consensus_labels)}
    except ImportError:
        # Fallback: use simple threshold-based clustering
        return threshold_clustering_from_consensus(consensus_matrix, nodes, threshold=0.5)


def threshold_clustering_from_consensus(
    consensus_matrix: np.ndarray,
    nodes: List[str],
    threshold: float = 0.5
) -> Dict[str, int]:
    """Helper function for consensus clustering without sklearn"""
    n_nodes = len(nodes)
    visited = [False] * n_nodes
    clusters = {}
    cluster_id = 0
    
    for i in range(n_nodes):
        if visited[i]:
            continue
        
        # Start new cluster
        cluster_nodes = [i]
        visited[i] = True
        
        # Add connected nodes above threshold
        for j in range(i + 1, n_nodes):
            if not visited[j] and consensus_matrix[i, j] > threshold:
                cluster_nodes.append(j)
                visited[j] = True
        
        # Assign cluster labels
        for node_idx in cluster_nodes:
            clusters[nodes[node_idx]] = cluster_id
        
        cluster_id += 1
    
    return clusters


def community_quality_metrics(
    hg: Hypergraph,
    communities: Dict[str, int]
) -> Dict[str, float]:
    """
    Compute quality metrics for community detection results.
    
    Args:
        hg: Hypergraph instance
        communities: Community assignments
        
    Returns:
        Dictionary of quality metrics
    """
    # Basic metrics
    n_communities = len(set(communities.values()))
    modularity = _calculate_modularity(hg, communities)
    
    # Coverage: fraction of edges with endpoints in same community
    incidence_df = hg.incidences.data
    total_edges = hg.num_edges
    covered_edges = 0
    
    for edge in hg.edges:
        edge_nodes = (
            incidence_df
            .filter(pl.col("edge_id") == edge)
            .select("node_id")
            .to_series()
            .to_list()
        )
        
        edge_communities = {communities.get(node, -1) for node in edge_nodes}
        if len(edge_communities) == 1 and -1 not in edge_communities:
            covered_edges += 1
    
    coverage = covered_edges / total_edges if total_edges > 0 else 0.0
    
    # Conductance: ratio of cut edges to total edges
    cut_edges = total_edges - covered_edges
    conductance = cut_edges / total_edges if total_edges > 0 else 0.0
    
    # Community size statistics
    community_sizes = {}
    for community_id in communities.values():
        community_sizes[community_id] = community_sizes.get(community_id, 0) + 1
    
    avg_community_size = float(np.mean(list(community_sizes.values())))
    std_community_size = float(np.std(list(community_sizes.values())))
    
    return {
        "n_communities": float(n_communities),
        "modularity": float(modularity),
        "coverage": float(coverage),
        "conductance": float(conductance),
        "avg_community_size": avg_community_size,
        "std_community_size": std_community_size,
        "largest_community": float(max(community_sizes.values()) if community_sizes else 0),
        "smallest_community": float(min(community_sizes.values()) if community_sizes else 0)
    }


def edge_community_detection(
    hg: Hypergraph,
    n_communities: int = 5,
    method: str = "spectral"
) -> Dict[str, int]:
    """
    Detect communities among edges rather than nodes.
    
    Edges that share many nodes are clustered together.
    
    Args:
        hg: Hypergraph instance
        n_communities: Number of edge communities to find
        method: Clustering method ("spectral" or "similarity")
        
    Returns:
        Dictionary mapping edge IDs to community labels
    """
    edges = list(hg.edges)
    n_edges = len(edges)
    
    if n_edges <= 1:
        return {edge: 0 for edge in edges}
    
    if n_edges < n_communities:
        return {edge: i for i, edge in enumerate(edges)}
    
    # Build edge-edge similarity matrix
    incidence_df = hg.incidences.data
    similarity_matrix = np.zeros((n_edges, n_edges))
    
    edge_nodes_dict = {}
    for edge in edges:
        edge_nodes_dict[edge] = set(
            incidence_df
            .filter(pl.col("edge_id") == edge)
            .select("node_id")
            .to_series()
            .to_list()
        )
    
    # Calculate Jaccard similarity between edges
    for i, edge1 in enumerate(edges):
        for j, edge2 in enumerate(edges):
            if i <= j:
                nodes1 = edge_nodes_dict[edge1]
                nodes2 = edge_nodes_dict[edge2]
                
                intersection = len(nodes1.intersection(nodes2))
                union = len(nodes1.union(nodes2))
                
                jaccard_sim = intersection / union if union > 0 else 0.0
                similarity_matrix[i, j] = jaccard_sim
                similarity_matrix[j, i] = jaccard_sim
    
    # Perform clustering
    if method == "spectral":
        try:
            from sklearn.cluster import SpectralClustering
            clusterer = SpectralClustering(
                n_clusters=n_communities,
                affinity='precomputed',
                random_state=42
            )
            edge_labels = clusterer.fit_predict(similarity_matrix)
            return {edge: int(label) for edge, label in zip(edges, edge_labels)}
        except ImportError:
            method = "similarity"  # Fallback
    
    if method == "similarity":
        # Simple threshold-based clustering
        return threshold_clustering_from_consensus(similarity_matrix, edges, threshold=0.3)
    
    raise ValueError(f"Unknown method: {method}")


def adaptive_community_detection(
    hg: Hypergraph,
    quality_threshold: float = 0.3,
    max_communities: int = 20
) -> Dict[str, int]:
    """
    Automatically determine optimal number of communities.
    
    Tries different numbers of communities and selects based
    on quality metrics.
    
    Args:
        hg: Hypergraph instance
        quality_threshold: Minimum quality threshold
        max_communities: Maximum number of communities to try
        
    Returns:
        Dictionary mapping node IDs to community labels
    """
    nodes = list(hg.nodes)
    n_nodes = len(nodes)
    
    if n_nodes <= 1:
        return {node: 0 for node in nodes}
    
    best_clustering = None
    best_quality = -float('inf')
    
    # Try different numbers of communities
    min_communities = 2
    max_test_communities = min(max_communities, n_nodes // 2)
    
    for n_comms in range(min_communities, max_test_communities + 1):
        try:
            # Try multiple methods
            methods_to_try = [
                ("spectral", lambda: spectral_clustering(hg, n_clusters=n_comms)),
                ("modularity", lambda: modularity_clustering(hg))
            ]
            
            for method_name, method_func in methods_to_try:
                clustering = method_func()
                quality_metrics = community_quality_metrics(hg, clustering)
                
                # Combined quality score
                quality_score = (
                    quality_metrics["modularity"] * 0.4 +
                    quality_metrics["coverage"] * 0.3 +
                    (1 - quality_metrics["conductance"]) * 0.3
                )
                
                if quality_score > best_quality and quality_score > quality_threshold:
                    best_quality = quality_score
                    best_clustering = clustering
                    
        except Exception:
            continue
    
    # Fallback to modularity if no good clustering found
    if best_clustering is None:
        best_clustering = modularity_clustering(hg)
    
    return best_clustering