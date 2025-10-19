"""
Centrality Measures for Hypergraphs

This module implements various centrality measures adapted for hypergraphs,
utilizing Polars for high-performance computation.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from ..classes.hypergraph import Hypergraph


def degree_centrality(
    hg: Hypergraph, 
    normalized: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compute degree centrality for both nodes and edges.
    
    For nodes: number of edges the node participates in
    For edges: number of nodes in the edge (edge size)
    
    Args:
        hg: Hypergraph instance
        normalized: Whether to normalize by maximum possible degree
        
    Returns:
        Dictionary with 'nodes' and 'edges' centrality scores
    """
    node_centrality = node_degree_centrality(hg, normalized)
    edge_centrality = edge_degree_centrality(hg, normalized)
    
    return {
        'nodes': node_centrality,
        'edges': edge_centrality
    }


def node_degree_centrality(
    hg: Hypergraph, 
    normalized: bool = True
) -> Dict[str, float]:
    """
    Compute node degree centrality (number of edges each node participates in).
    
    Args:
        hg: Hypergraph instance
        normalized: Whether to normalize by maximum possible degree
        
    Returns:
        Dictionary mapping node IDs to centrality scores
    """
    # Get node degrees from incidence store
    incidence_df = hg._incidence_store.data
    
    # Count edges per node
    node_degrees = (
        incidence_df
        .group_by("nodes")
        .agg(pl.count("edges").alias("degree"))
        .sort("nodes")
    )
    
    # Convert to dictionary
    degrees_dict = dict(zip(
        node_degrees["nodes"].to_list(),
        node_degrees["degree"].to_list()
    ))
    
    if normalized and hg.num_edges > 1:
        # Normalize by maximum possible degree (total number of edges)
        max_degree = hg.num_edges
        degrees_dict = {node: degree / max_degree 
                       for node, degree in degrees_dict.items()}
    
    return degrees_dict


def edge_degree_centrality(
    hg: Hypergraph, 
    normalized: bool = True
) -> Dict[str, float]:
    """
    Compute edge degree centrality (size of each edge).
    
    Args:
        hg: Hypergraph instance
        normalized: Whether to normalize by maximum possible size
        
    Returns:
        Dictionary mapping edge IDs to centrality scores
    """
    # Get edge sizes from incidence store
    incidence_df = hg._incidence_store.data
    
    # Count nodes per edge
    edge_sizes = (
        incidence_df
        .group_by("edges")
        .agg(pl.count("nodes").alias("size"))
        .sort("edges")
    )
    
    # Convert to dictionary
    sizes_dict = dict(zip(
        edge_sizes["edges"].to_list(),
        edge_sizes["size"].to_list()
    ))
    
    if normalized and hg.num_nodes > 1:
        # Normalize by maximum possible size (total number of nodes)
        max_size = hg.num_nodes
        sizes_dict = {edge: size / max_size 
                     for edge, size in sizes_dict.items()}
    
    return sizes_dict


def closeness_centrality(
    hg: Hypergraph,
    normalized: bool = True
) -> Dict[str, float]:
    """
    Compute closeness centrality for nodes in hypergraph.
    
    Uses distance through shared edges. Two nodes are distance 1
    if they share an edge, distance 2 if they share a neighbor, etc.
    
    Args:
        hg: Hypergraph instance
        normalized: Whether to normalize by graph size
        
    Returns:
        Dictionary mapping node IDs to closeness centrality scores
    """
    nodes = list(hg.nodes)
    n = len(nodes)
    
    if n <= 1:
        return {node: 0.0 for node in nodes}
    
    # Build adjacency information
    incidence_df = hg._incidence_store.data
    
    # Create node-to-edges mapping
    node_edges = (
        incidence_df
        .group_by("nodes")
        .agg(pl.col("edges").alias("edge_list"))
    )
    
    node_to_edges = dict(zip(
        node_edges["nodes"].to_list(),
        node_edges["edge_list"].to_list()
    ))
    
    centralities = {}
    
    for source_node in nodes:
        # BFS to find shortest paths
        distances = {source_node: 0}
        queue = [source_node]
        visited = {source_node}
        
        while queue:
            current = queue.pop(0)
            current_dist = distances[current]
            
            # Get all edges containing current node
            if current in node_to_edges:
                current_edges = set(node_to_edges[current])
                
                # Find all nodes sharing these edges
                neighbors = set()
                for edge in current_edges:
                    edge_nodes = (
                        incidence_df
                        .filter(pl.col("edges") == edge)
                        .select("nodes")
                        .to_series()
                        .to_list()
                    )
                    neighbors.update(edge_nodes)
                
                # Update distances for unvisited neighbors
                for neighbor in neighbors:
                    if neighbor not in visited:
                        distances[neighbor] = current_dist + 1
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        # Calculate closeness centrality
        if len(distances) > 1:
            total_distance = sum(distances.values())
            if total_distance > 0:
                closeness = (len(distances) - 1) / total_distance
                if normalized:
                    closeness *= (len(distances) - 1) / (n - 1)
            else:
                closeness = 0.0
        else:
            closeness = 0.0
            
        centralities[source_node] = closeness
    
    return centralities


def betweenness_centrality(
    hg: Hypergraph,
    normalized: bool = True,
    sample_size: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute betweenness centrality for nodes in hypergraph.
    
    Measures how often a node lies on shortest paths between other nodes.
    For large graphs, can use sampling for approximation.
    
    Args:
        hg: Hypergraph instance
        normalized: Whether to normalize by total possible paths
        sample_size: If provided, sample this many source nodes for approximation
        
    Returns:
        Dictionary mapping node IDs to betweenness centrality scores
    """
    nodes = list(hg.nodes)
    n = len(nodes)
    
    if n <= 2:
        return {node: 0.0 for node in nodes}
    
    # Initialize betweenness scores
    betweenness = {node: 0.0 for node in nodes}
    
    # Build adjacency information
    incidence_df = hg._incidence_store.data
    node_edges = (
        incidence_df
        .group_by("nodes")
        .agg(pl.col("edges").alias("edge_list"))
    )
    
    node_to_edges = dict(zip(
        node_edges["nodes"].to_list(),
        node_edges["edge_list"].to_list()
    ))
    
    # Sample nodes if requested
    if sample_size and sample_size < n:
        import random
        source_nodes = random.sample(nodes, sample_size)
    else:
        source_nodes = nodes
    
    for source in source_nodes:
        # BFS to find all shortest paths from source
        predecessors = {node: [] for node in nodes}
        distances = {source: 0}
        sigma = {node: 0 for node in nodes}
        sigma[source] = 1
        
        queue = [source]
        stack = []
        
        while queue:
            current = queue.pop(0)
            stack.append(current)
            current_dist = distances[current]
            
            # Get neighbors through shared edges
            if current in node_to_edges:
                current_edges = set(node_to_edges[current])
                neighbors = set()
                
                for edge in current_edges:
                    edge_nodes = (
                        incidence_df
                        .filter(pl.col("edges") == edge)
                        .select("nodes")
                        .to_series()
                        .to_list()
                    )
                    neighbors.update(edge_nodes)
                
                for neighbor in neighbors:
                    if neighbor != current:
                        if neighbor not in distances:
                            # First time visiting neighbor
                            distances[neighbor] = current_dist + 1
                            queue.append(neighbor)
                        
                        if distances[neighbor] == current_dist + 1:
                            # Neighbor is on shortest path through current
                            sigma[neighbor] += sigma[current]
                            predecessors[neighbor].append(current)
        
        # Accumulate betweenness scores
        delta = {node: 0.0 for node in nodes}
        
        while stack:
            node = stack.pop()
            for pred in predecessors[node]:
                delta[pred] += (sigma[pred] / sigma[node]) * (1 + delta[node])
            
            if node != source:
                betweenness[node] += delta[node]
    
    # Normalize if requested
    if normalized:
        # Normalization factor for undirected graphs
        norm_factor = 2.0 / ((n - 1) * (n - 2))
        if sample_size and sample_size < n:
            # Adjust for sampling
            norm_factor *= n / sample_size
        
        betweenness = {node: score * norm_factor 
                      for node, score in betweenness.items()}
    
    return betweenness


def s_centrality(
    hg: Hypergraph,
    s: int = 1,
    normalized: bool = True
) -> Dict[str, float]:
    """
    Compute s-centrality for nodes in hypergraph.
    
    S-centrality measures the importance of a node based on the sizes
    of the edges it participates in, with parameter s controlling
    the emphasis on larger edges.
    
    Args:
        hg: Hypergraph instance
        s: Parameter controlling emphasis on edge size (default=1)
        normalized: Whether to normalize by maximum possible score
        
    Returns:
        Dictionary mapping node IDs to s-centrality scores
    """
    incidence_df = hg._incidence_store.data
    
    # Calculate edge sizes
    edge_sizes = (
        incidence_df
        .group_by("edges")
        .agg(pl.count("nodes").alias("size"))
    )
    
    # Join with incidence to get edge sizes for each node-edge pair
    node_edge_sizes = (
        incidence_df
        .join(edge_sizes, on="edges")
        .with_columns([
            # Compute s-centrality contribution: (size - 1)^s
            (pl.col("size") - 1).pow(s).alias("contribution")
        ])
    )
    
    # Sum contributions per node
    s_centralities = (
        node_edge_sizes
        .group_by("nodes")
        .agg(pl.sum("contribution").alias("s_centrality"))
        .sort("nodes")
    )
    
    # Convert to dictionary
    centralities_dict = dict(zip(
        s_centralities["nodes"].to_list(),
        s_centralities["s_centrality"].to_list()
    ))
    
    # Fill in nodes that might not appear (isolated nodes)
    all_nodes = set(hg.nodes)
    for node in all_nodes:
        if node not in centralities_dict:
            centralities_dict[node] = 0.0
    
    if normalized and hg.num_edges > 0:
        # Normalize by maximum possible score
        max_edge_size = max(hg._incidence_store.get_edge_size(edge) for edge in hg.edges)
        max_possible = hg.num_edges * ((max_edge_size - 1) ** s)
        
        if max_possible > 0:
            centralities_dict = {
                node: score / max_possible 
                for node, score in centralities_dict.items()
            }
    
    return centralities_dict


def eigenvector_centrality(
    hg: Hypergraph,
    max_iter: int = 100,
    tolerance: float = 1e-6,
    normalized: bool = True
) -> Dict[str, float]:
    """
    Compute eigenvector centrality for nodes in hypergraph.
    
    Uses the node projection of the hypergraph to compute eigenvector
    centrality via power iteration method.
    
    Args:
        hg: Hypergraph instance
        max_iter: Maximum number of iterations for power method
        tolerance: Convergence tolerance for power method
        normalized: Whether to normalize the final scores
        
    Returns:
        Dictionary mapping node IDs to eigenvector centrality scores
    """
    nodes = list(hg.nodes)
    n = len(nodes)
    
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: 1.0}
    
    # Build node-to-node adjacency matrix through edge projections
    incidence_df = hg._incidence_store.data
    
    # For each edge, create pairwise connections between all nodes in the edge
    node_connections = []
    
    for edge in hg.edges:
        edge_nodes = (
            incidence_df
            .filter(pl.col("edges") == edge)
            .select("nodes")
            .to_series()
            .to_list()
        )
        
        # Add connections between all pairs of nodes in this edge
        for i, node1 in enumerate(edge_nodes):
            for j, node2 in enumerate(edge_nodes):
                if i != j:  # No self-loops
                    node_connections.append({"from_node": node1, "to_node": node2})
    
    if not node_connections:
        # No connections, return uniform distribution
        return {node: 1.0 / n for node in nodes}
    
    # Create adjacency DataFrame
    connections_df = pl.DataFrame(node_connections)
    
    # Count connections and create adjacency matrix representation
    adjacency = (
        connections_df
        .group_by(["from_node", "to_node"])
        .agg(pl.count().alias("weight"))
    )
    
    # Initialize eigenvector with uniform values
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    x = np.ones(n) / n
    
    # Power iteration
    for iteration in range(max_iter):
        x_new = np.zeros(n)
        
        # Matrix-vector multiplication using Polars operations
        for row in adjacency.iter_rows(named=True):
            from_idx = node_to_idx[row["from_node"]]
            to_idx = node_to_idx[row["to_node"]]
            weight = row["weight"]
            x_new[to_idx] += weight * x[from_idx]
        
        # Normalize
        norm = np.linalg.norm(x_new)
        if norm == 0:
            break
        x_new = x_new / norm
        
        # Check convergence
        if np.allclose(x, x_new, atol=tolerance):
            break
        
        x = x_new
    
    # Create result dictionary
    centralities = dict(zip(nodes, x.tolist()))
    
    if normalized:
        # Normalize to [0, 1] range
        max_score = max(centralities.values()) if centralities.values() else 1.0
        if max_score > 0:
            centralities = {node: score / max_score 
                          for node, score in centralities.items()}
    
    return centralities


def pagerank_centrality(
    hg: Hypergraph,
    alpha: float = 0.85,
    max_iter: int = 100,
    tolerance: float = 1e-6,
    personalization: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compute PageRank centrality for nodes in hypergraph.
    
    Adapts PageRank algorithm to hypergraphs by treating the node
    projection as the underlying graph structure.
    
    Args:
        hg: Hypergraph instance
        alpha: Damping parameter (probability of following links)
        max_iter: Maximum number of iterations
        tolerance: Convergence tolerance
        personalization: Personalization vector (if None, uniform distribution)
        
    Returns:
        Dictionary mapping node IDs to PageRank scores
    """
    nodes = list(hg.nodes)
    n = len(nodes)
    
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: 1.0}
    
    # Build transition matrix from hypergraph structure
    incidence_df = hg._incidence_store.data
    
    # Calculate node degrees (out-degrees for PageRank)
    node_degrees = (
        incidence_df
        .group_by("nodes")
        .agg(pl.count("edges").alias("degree"))
    )
    
    node_degree_dict = dict(zip(
        node_degrees["nodes"].to_list(),
        node_degrees["degree"].to_list()
    ))
    
    # Build node-to-node connections through shared edges
    node_connections = []
    
    for edge in hg.edges:
        edge_nodes = (
            incidence_df
            .filter(pl.col("edges") == edge)
            .select("nodes")
            .to_series()
            .to_list()
        )
        
        # Each node in edge connects to all other nodes in edge
        for node1 in edge_nodes:
            for node2 in edge_nodes:
                if node1 != node2:
                    node_connections.append({"from_node": node1, "to_node": node2})
    
    if not node_connections:
        # No connections, return uniform distribution
        return {node: 1.0 / n for node in nodes}
    
    # Create adjacency DataFrame and calculate transition probabilities
    connections_df = pl.DataFrame(node_connections)
    
    # Count outgoing connections per node
    out_connections = (
        connections_df
        .group_by("from_node")
        .agg(pl.count().alias("out_degree"))
    )
    
    out_degree_dict = dict(zip(
        out_connections["from_node"].to_list(),
        out_connections["out_degree"].to_list()
    ))
    
    # Initialize PageRank values
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    if personalization is None:
        personalization = {node: 1.0 / n for node in nodes}
    
    # Normalize personalization vector
    pers_sum = sum(personalization.values())
    if pers_sum > 0:
        personalization = {node: val / pers_sum 
                         for node, val in personalization.items()}
    
    x = np.array([personalization.get(node, 1.0 / n) for node in nodes])
    
    # Power iteration for PageRank
    for iteration in range(max_iter):
        x_new = np.zeros(n)
        
        # Add personalization term
        for i, node in enumerate(nodes):
            x_new[i] += (1 - alpha) * personalization.get(node, 1.0 / n)
        
        # Add link-based contributions
        for row in connections_df.iter_rows(named=True):
            from_node = row["from_node"]
            to_node = row["to_node"]
            
            from_idx = node_to_idx[from_node]
            to_idx = node_to_idx[to_node]
            
            # Transition probability
            out_degree = out_degree_dict.get(from_node, 1)
            transition_prob = 1.0 / out_degree
            
            x_new[to_idx] += alpha * transition_prob * x[from_idx]
        
        # Check convergence
        if np.allclose(x, x_new, atol=tolerance):
            break
        
        x = x_new
    
    # Create result dictionary
    return dict(zip(nodes, x.tolist()))


def weighted_degree_centrality(
    hg: Hypergraph,
    edge_weights: Optional[Dict[str, float]] = None,
    weight_function: str = "uniform",
    normalized: bool = True
) -> Dict[str, float]:
    """
    Compute weighted degree centrality for nodes.
    
    Args:
        hg: Hypergraph instance
        edge_weights: Explicit edge weights (if None, computed from weight_function)
        weight_function: How to compute edge weights ('uniform', 'size', 'inverse_size')
        normalized: Whether to normalize by maximum possible weighted degree
        
    Returns:
        Dictionary mapping node IDs to weighted degree centrality scores
    """
    incidence_df = hg._incidence_store.data
    
    # Compute edge weights if not provided
    if edge_weights is None:
        if weight_function == "uniform":
            edge_weights = {edge: 1.0 for edge in hg.edges}
        elif weight_function == "size":
            # Weight by edge size
            edge_sizes = (
                incidence_df
                .group_by("edges")
                .agg(pl.count("nodes").alias("size"))
            )
            edge_weights = dict(zip(
                edge_sizes["edges"].to_list(),
                edge_sizes["size"].to_list()
            ))
        elif weight_function == "inverse_size":
            # Weight by inverse edge size (smaller edges get higher weight)
            edge_sizes = (
                incidence_df
                .group_by("edges")
                .agg(pl.count("nodes").alias("size"))
            )
            edge_weights = {
                edge: 1.0 / size 
                for edge, size in zip(
                    edge_sizes["edges"].to_list(),
                    edge_sizes["size"].to_list()
                )
            }
        else:
            raise ValueError(f"Unknown weight_function: {weight_function}")
    
    # Add edge weights to incidence data
    edge_weight_list = [
        {"edges": edge, "weight": edge_weights.get(edge, 1.0)}
        for edge in hg.edges
    ]
    
    weights_df = pl.DataFrame(edge_weight_list)
    
    # Ensure data types match for join
    weights_df = weights_df.with_columns([
        pl.col("edges").cast(incidence_df["edges"].dtype)
    ])
    
    # Join with incidence and sum weights per node
    weighted_degrees = (
        incidence_df
        .join(weights_df, on="edges")
        .group_by("nodes")
        .agg(pl.sum("weight").alias("weighted_degree"))
        .sort("nodes")
    )
    
    # Convert to dictionary
    centralities_dict = dict(zip(
        weighted_degrees["nodes"].to_list(),
        weighted_degrees["weighted_degree"].to_list()
    ))
    
    # Fill in nodes that might not appear (isolated nodes)
    all_nodes = set(hg.nodes)
    for node in all_nodes:
        if node not in centralities_dict:
            centralities_dict[node] = 0.0
    
    if normalized:
        # Normalize by maximum possible weighted degree
        total_weight = sum(edge_weights.values())
        if total_weight > 0:
            centralities_dict = {
                node: score / total_weight 
                for node, score in centralities_dict.items()
            }
    
    return centralities_dict