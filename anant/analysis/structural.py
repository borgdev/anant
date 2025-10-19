"""
Structural Analysis for Hypergraphs

This module implements structural analysis algorithms for hypergraphs,
including connectivity, diameter, and clustering coefficients.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Set
from collections import deque
from ..classes.hypergraph import Hypergraph


def connected_components(hg: Hypergraph) -> List[Set[str]]:
    """
    Find connected components in the hypergraph.
    
    Two nodes are connected if there is a path between them through edges.
    
    Args:
        hg: Hypergraph instance
        
    Returns:
        List of sets, each containing nodes in a connected component
    """
    nodes = set(hg.nodes)
    visited = set()
    components = []
    
    incidence_df = hg._incidence_store.data
    
    # Build adjacency information
    node_neighbors = {}
    for node in nodes:
        node_edges = (
            incidence_df
            .filter(pl.col("nodes") == node)
            .select("edges")
            .to_series()
            .to_list()
        )
        
        neighbors = set()
        for edge in node_edges:
            edge_nodes = (
                incidence_df
                .filter(pl.col("edges") == edge)
                .select("nodes")
                .to_series()
                .to_list()
            )
            neighbors.update(edge_nodes)
        
        neighbors.discard(node)  # Remove self
        node_neighbors[node] = neighbors
    
    # DFS to find components
    for start_node in nodes:
        if start_node not in visited:
            component = set()
            stack = [start_node]
            
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    component.add(node)
                    
                    # Add unvisited neighbors to stack
                    for neighbor in node_neighbors.get(node, []):
                        if neighbor not in visited:
                            stack.append(neighbor)
            
            if component:
                components.append(component)
    
    return components


def hypergraph_diameter(hg: Hypergraph) -> Optional[int]:
    """
    Compute the diameter of the hypergraph.
    
    The diameter is the longest shortest path between any two nodes.
    Returns None if the graph is disconnected.
    
    Args:
        hg: Hypergraph instance
        
    Returns:
        Diameter of the hypergraph or None if disconnected
    """
    nodes = list(hg.nodes)
    n = len(nodes)
    
    if n <= 1:
        return 0
    
    # Check if graph is connected
    components = connected_components(hg)
    if len(components) > 1:
        return None  # Disconnected graph
    
    incidence_df = hg._incidence_store.data
    
    # Build adjacency information
    node_neighbors = {}
    for node in nodes:
        node_edges = (
            incidence_df
            .filter(pl.col("nodes") == node)
            .select("edges")
            .to_series()
            .to_list()
        )
        
        neighbors = set()
        for edge in node_edges:
            edge_nodes = (
                incidence_df
                .filter(pl.col("edges") == edge)
                .select("nodes")
                .to_series()
                .to_list()
            )
            neighbors.update(edge_nodes)
        
        neighbors.discard(node)
        node_neighbors[node] = neighbors
    
    max_distance = 0
    
    # BFS from each node to find shortest paths
    for start_node in nodes:
        distances = {start_node: 0}
        queue = deque([start_node])
        
        while queue:
            current = queue.popleft()
            current_dist = distances[current]
            
            for neighbor in node_neighbors.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        
        # Update maximum distance
        if len(distances) == n:  # All nodes reachable
            max_dist_from_start = max(distances.values())
            max_distance = max(max_distance, max_dist_from_start)
        else:
            return None  # Graph is disconnected
    
    return max_distance


def clustering_coefficient(
    hg: Hypergraph,
    node: Optional[str] = None
) -> Union[float, Dict[str, float]]:
    """
    Compute clustering coefficient(s) for hypergraph.
    
    The clustering coefficient measures how close a node's neighbors
    are to being a clique.
    
    Args:
        hg: Hypergraph instance
        node: Specific node to compute for (None for all nodes)
        
    Returns:
        Clustering coefficient for specific node or dict for all nodes
    """
    if node is not None:
        return local_clustering_coefficient(hg, node)
    else:
        return {n: local_clustering_coefficient(hg, n) for n in hg.nodes}


def local_clustering_coefficient(hg: Hypergraph, node: str) -> float:
    """
    Compute local clustering coefficient for a specific node.
    
    Args:
        hg: Hypergraph instance
        node: Node ID
        
    Returns:
        Local clustering coefficient
    """
    incidence_df = hg._incidence_store.data
    
    # Get edges containing the node
    node_edges = (
        incidence_df
        .filter(pl.col("nodes") == node)
        .select("edges")
        .to_series()
        .to_list()
    )
    
    if len(node_edges) < 2:
        return 0.0
    
    # Get all neighbors of the node
    neighbors = set()
    for edge in node_edges:
        edge_nodes = (
            incidence_df
            .filter(pl.col("edges") == edge)
            .select("nodes")
            .to_series()
            .to_list()
        )
        neighbors.update(edge_nodes)
    
    neighbors.discard(node)
    neighbors = list(neighbors)
    k = len(neighbors)
    
    if k < 2:
        return 0.0
    
    # Count edges between neighbors
    edges_between_neighbors = 0
    
    for edge_id in hg.edges:
        if edge_id not in node_edges:  # Don't count edges containing the node itself
            edge_nodes = (
                incidence_df
                .filter(pl.col("edges") == edge_id)
                .select("nodes")
                .to_series()
                .to_list()
            )
            
            # Count how many neighbors are in this edge
            neighbors_in_edge = len([n for n in edge_nodes if n in neighbors])
            
            if neighbors_in_edge >= 2:
                # This edge connects neighbors
                edges_between_neighbors += 1
    
    # Maximum possible edges between k neighbors
    max_possible_edges = k * (k - 1) // 2
    
    if max_possible_edges == 0:
        return 0.0
    
    return edges_between_neighbors / max_possible_edges


def global_clustering_coefficient(hg: Hypergraph) -> float:
    """
    Compute global clustering coefficient for the hypergraph.
    
    This is the average of all local clustering coefficients.
    
    Args:
        hg: Hypergraph instance
        
    Returns:
        Global clustering coefficient
    """
    local_coefficients = []
    
    for node in hg.nodes:
        coeff = local_clustering_coefficient(hg, node)
        local_coefficients.append(coeff)
    
    if not local_coefficients:
        return 0.0
    
    return sum(local_coefficients) / len(local_coefficients)


def hypergraph_density(hg: Hypergraph) -> float:
    """
    Compute the density of the hypergraph.
    
    Density is the ratio of actual edges to maximum possible edges.
    
    Args:
        hg: Hypergraph instance
        
    Returns:
        Hypergraph density
    """
    n_nodes = hg.num_nodes
    n_edges = hg.num_edges
    
    if n_nodes <= 1:
        return 0.0
    
    # Maximum possible edges (power set minus empty set and singletons)
    max_possible_edges = 2**n_nodes - n_nodes - 1
    
    if max_possible_edges == 0:
        return 0.0
    
    return n_edges / max_possible_edges


def edge_overlap(hg: Hypergraph, edge1: str, edge2: str) -> float:
    """
    Compute the overlap between two edges.
    
    Overlap is measured as Jaccard similarity: |A ∩ B| / |A ∪ B|
    
    Args:
        hg: Hypergraph instance
        edge1: First edge ID
        edge2: Second edge ID
        
    Returns:
        Overlap coefficient between 0 and 1
    """
    incidence_df = hg._incidence_store.data
    
    # Get nodes in each edge
    nodes1 = set(
        incidence_df
        .filter(pl.col("edges") == edge1)
        .select("nodes")
        .to_series()
        .to_list()
    )
    
    nodes2 = set(
        incidence_df
        .filter(pl.col("edges") == edge2)
        .select("nodes")
        .to_series()
        .to_list()
    )
    
    intersection = len(nodes1.intersection(nodes2))
    union = len(nodes1.union(nodes2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def average_edge_overlap(hg: Hypergraph) -> float:
    """
    Compute the average overlap between all pairs of edges.
    
    Args:
        hg: Hypergraph instance
        
    Returns:
        Average edge overlap
    """
    edges = list(hg.edges)
    n_edges = len(edges)
    
    if n_edges < 2:
        return 0.0
    
    total_overlap = 0.0
    n_pairs = 0
    
    for i in range(n_edges):
        for j in range(i + 1, n_edges):
            overlap = edge_overlap(hg, edges[i], edges[j])
            total_overlap += overlap
            n_pairs += 1
    
    if n_pairs == 0:
        return 0.0
    
    return total_overlap / n_pairs


def node_similarity(hg: Hypergraph, node1: str, node2: str) -> float:
    """
    Compute similarity between two nodes based on shared edges.
    
    Uses Jaccard similarity of edge sets.
    
    Args:
        hg: Hypergraph instance
        node1: First node ID
        node2: Second node ID
        
    Returns:
        Similarity coefficient between 0 and 1
    """
    incidence_df = hg._incidence_store.data
    
    # Get edges for each node
    edges1 = set(
        incidence_df
        .filter(pl.col("nodes") == node1)
        .select("edges")
        .to_series()
        .to_list()
    )
    
    edges2 = set(
        incidence_df
        .filter(pl.col("nodes") == node2)
        .select("edges")
        .to_series()
        .to_list()
    )
    
    intersection = len(edges1.intersection(edges2))
    union = len(edges1.union(edges2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def structural_summary(hg: Hypergraph) -> Dict[str, Union[int, float, None]]:
    """
    Compute a comprehensive structural summary of the hypergraph.
    
    Args:
        hg: Hypergraph instance
        
    Returns:
        Dictionary of structural metrics
    """
    components = connected_components(hg)
    
    return {
        "num_nodes": hg.num_nodes,
        "num_edges": hg.num_edges,
        "num_incidences": hg.num_incidences,
        "density": hypergraph_density(hg),
        "num_connected_components": len(components),
        "largest_component_size": max(len(comp) for comp in components) if components else 0,
        "diameter": hypergraph_diameter(hg),
        "global_clustering_coefficient": global_clustering_coefficient(hg),
        "average_edge_overlap": average_edge_overlap(hg),
        "is_connected": len(components) <= 1
    }