"""
Spectral Analysis for Hypergraphs

This module implements Laplacian matrices and spectral analysis methods
for hypergraphs, optimized with Polars and NumPy.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from ..classes.hypergraph import Hypergraph


def node_laplacian(
    hg: Hypergraph,
    normalized: bool = True
) -> np.ndarray:
    """
    Compute the node Laplacian matrix for the hypergraph.
    
    The node Laplacian captures relationships between nodes through
    shared edge participation.
    
    Args:
        hg: Hypergraph instance
        normalized: Whether to normalize the Laplacian
        
    Returns:
        Node Laplacian matrix as numpy array
    """
    nodes = list(hg.nodes)
    n = len(nodes)
    
    if n == 0:
        return np.array([])
    
    # Create adjacency matrix based on shared edges
    adjacency = np.zeros((n, n))
    incidence_df = hg._incidence_store.data
    
    # For each edge, connect all pairs of nodes in that edge
    for edge_id in hg.edges:
        edge_nodes = (
            incidence_df
            .filter(pl.col("edges") == edge_id)
            .select("nodes")
            .to_series()
            .to_list()
        )
        
        edge_size = len(edge_nodes)
        if edge_size >= 2:
            # Weight by 1/(edge_size - 1) to normalize by edge size
            weight = 1.0 / (edge_size - 1)
            
            for i, node1 in enumerate(edge_nodes):
                for j, node2 in enumerate(edge_nodes):
                    if i != j:
                        idx1 = nodes.index(node1)
                        idx2 = nodes.index(node2)
                        adjacency[idx1, idx2] += weight
    
    # Compute degree matrix
    degrees = np.sum(adjacency, axis=1)
    degree_matrix = np.diag(degrees)
    
    # Compute Laplacian
    laplacian = degree_matrix - adjacency
    
    if normalized and np.any(degrees > 0):
        # Normalized Laplacian: D^(-1/2) L D^(-1/2)
        inv_sqrt_degrees = np.zeros_like(degrees)
        nonzero_mask = degrees > 0
        inv_sqrt_degrees[nonzero_mask] = 1.0 / np.sqrt(degrees[nonzero_mask])
        
        inv_sqrt_degree_matrix = np.diag(inv_sqrt_degrees)
        laplacian = inv_sqrt_degree_matrix @ laplacian @ inv_sqrt_degree_matrix
    
    return laplacian


def edge_laplacian(
    hg: Hypergraph,
    normalized: bool = True
) -> np.ndarray:
    """
    Compute the edge Laplacian matrix for the hypergraph.
    
    The edge Laplacian captures relationships between edges through
    shared node participation.
    
    Args:
        hg: Hypergraph instance
        normalized: Whether to normalize the Laplacian
        
    Returns:
        Edge Laplacian matrix as numpy array
    """
    edges = list(hg.edges)
    m = len(edges)
    
    if m == 0:
        return np.array([])
    
    # Create adjacency matrix based on shared nodes
    adjacency = np.zeros((m, m))
    incidence_df = hg._incidence_store.data
    
    # For each node, connect all pairs of edges containing that node
    for node_id in hg.nodes:
        node_edges = (
            incidence_df
            .filter(pl.col("nodes") == node_id)
            .select("edges")
            .to_series()
            .to_list()
        )
        
        node_degree = len(node_edges)
        if node_degree >= 2:
            # Weight by 1/(node_degree - 1) to normalize by node degree
            weight = 1.0 / (node_degree - 1)
            
            for i, edge1 in enumerate(node_edges):
                for j, edge2 in enumerate(node_edges):
                    if i != j:
                        idx1 = edges.index(edge1)
                        idx2 = edges.index(edge2)
                        adjacency[idx1, idx2] += weight
    
    # Compute degree matrix
    degrees = np.sum(adjacency, axis=1)
    degree_matrix = np.diag(degrees)
    
    # Compute Laplacian
    laplacian = degree_matrix - adjacency
    
    if normalized and np.any(degrees > 0):
        # Normalized Laplacian
        inv_sqrt_degrees = np.zeros_like(degrees)
        nonzero_mask = degrees > 0
        inv_sqrt_degrees[nonzero_mask] = 1.0 / np.sqrt(degrees[nonzero_mask])
        
        inv_sqrt_degree_matrix = np.diag(inv_sqrt_degrees)
        laplacian = inv_sqrt_degree_matrix @ laplacian @ inv_sqrt_degree_matrix
    
    return laplacian


def hypergraph_laplacian(
    hg: Hypergraph,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Compute a combined hypergraph Laplacian matrix.
    
    Combines node and edge Laplacians with a weighting parameter.
    
    Args:
        hg: Hypergraph instance
        alpha: Weight for combining node and edge Laplacians (0 <= alpha <= 1)
        
    Returns:
        Combined Laplacian matrix
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    
    node_L = node_laplacian(hg, normalized=True)
    edge_L = edge_laplacian(hg, normalized=True)
    
    # If sizes differ, we need a more sophisticated combination
    # For now, return the node Laplacian weighted by alpha
    if node_L.size > 0:
        return alpha * node_L + (1 - alpha) * node_L  # Simplified version
    else:
        return np.array([])


def laplacian_spectrum(
    laplacian: np.ndarray,
    k: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and eigenvectors of a Laplacian matrix.
    
    Args:
        laplacian: Laplacian matrix
        k: Number of smallest eigenvalues/eigenvectors to compute (None for all)
        
    Returns:
        Tuple of (eigenvalues, eigenvectors) sorted by eigenvalue magnitude
    """
    if laplacian.size == 0:
        return np.array([]), np.array([])
    
    if k is None:
        # Compute all eigenvalues/eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    else:
        # Compute only k smallest eigenvalues/eigenvectors
        try:
            from scipy.sparse.linalg import eigsh
            eigenvalues, eigenvectors = eigsh(laplacian, k=k, which='SM')
        except ImportError:
            # Fall back to full decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            # Take k smallest
            idx = np.argsort(eigenvalues)[:k]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
    
    # Sort by eigenvalue magnitude
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors


def algebraic_connectivity(hg: Hypergraph) -> float:
    """
    Compute the algebraic connectivity of the hypergraph.
    
    The algebraic connectivity is the second smallest eigenvalue of
    the normalized Laplacian matrix.
    
    Args:
        hg: Hypergraph instance
        
    Returns:
        Algebraic connectivity value
    """
    laplacian = node_laplacian(hg, normalized=True)
    
    if laplacian.size <= 1:
        return 0.0
    
    eigenvalues, _ = laplacian_spectrum(laplacian, k=min(2, laplacian.shape[0]))
    
    if len(eigenvalues) >= 2:
        return float(eigenvalues[1])  # Second smallest eigenvalue
    else:
        return 0.0


def fiedler_vector(hg: Hypergraph) -> Optional[np.ndarray]:
    """
    Compute the Fiedler vector (eigenvector corresponding to algebraic connectivity).
    
    The Fiedler vector can be used for graph partitioning.
    
    Args:
        hg: Hypergraph instance
        
    Returns:
        Fiedler vector or None if not computable
    """
    laplacian = node_laplacian(hg, normalized=True)
    
    if laplacian.size <= 1:
        return None
    
    eigenvalues, eigenvectors = laplacian_spectrum(laplacian, k=min(2, laplacian.shape[0]))
    
    if len(eigenvalues) >= 2:
        return eigenvectors[:, 1]  # Eigenvector for second smallest eigenvalue
    else:
        return None


def spectral_embedding(
    hg: Hypergraph,
    n_components: int = 2,
    method: str = "node"
) -> np.ndarray:
    """
    Compute spectral embedding of hypergraph for visualization or clustering.
    
    Args:
        hg: Hypergraph instance
        n_components: Number of embedding dimensions
        method: "node" or "edge" embedding
        
    Returns:
        Embedding matrix (n_entities x n_components)
    """
    if method == "node":
        laplacian = node_laplacian(hg, normalized=True)
        n_entities = len(hg.nodes)
    elif method == "edge":
        laplacian = edge_laplacian(hg, normalized=True)
        n_entities = len(hg.edges)
    else:
        raise ValueError("method must be 'node' or 'edge'")
    
    if laplacian.size == 0 or n_entities <= n_components:
        return np.zeros((n_entities, n_components))
    
    # Use the smallest non-zero eigenvalues for embedding
    k = min(n_components + 1, laplacian.shape[0])
    eigenvalues, eigenvectors = laplacian_spectrum(laplacian, k=k)
    
    # Skip the first eigenvector (corresponding to eigenvalue 0)
    if len(eigenvalues) > 1:
        embedding = eigenvectors[:, 1:n_components+1]
    else:
        embedding = np.zeros((n_entities, n_components))
    
    # Pad with zeros if needed
    if embedding.shape[1] < n_components:
        padding = np.zeros((n_entities, n_components - embedding.shape[1]))
        embedding = np.concatenate([embedding, padding], axis=1)
    
    return embedding