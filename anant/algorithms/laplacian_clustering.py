"""
Hypergraph Laplacian and Spectral Clustering
============================================

Implementation of hypergraph random walks, Laplacian matrices, and spectral clustering
algorithms. Based on the methodology described in:

K. Hayashi, S. Aksoy, C. Park, H. Park, "Hypergraph random walks, Laplacians, and clustering",
Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2020.
https://doi.org/10.1145/3340531.3412034

This module provides hypergraph-specific generalizations of graph Laplacians and 
spectral clustering methods.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings

# Optional dependencies with graceful fallback
try:
    from scipy.sparse import csr_matrix, diags, identity
    from scipy.sparse.linalg import eigs
    from sklearn.cluster import KMeans
    from sklearn import preprocessing
    HAS_SCIPY_SKLEARN = True
except ImportError:
    HAS_SCIPY_SKLEARN = False
    warnings.warn("scipy and/or sklearn not available. Some laplacian clustering features disabled.")


def prob_trans(hg, weights=False, index=True, check_connected=True):
    """
    Compute the probability transition matrix of a random walk on hypergraph vertices.
    
    At each step in the walk, the next vertex is chosen by:
    1. Selecting a hyperedge e containing the vertex with probability proportional to w(e)
    2. Selecting a vertex v within e with probability proportional to γ(v,e)
    
    Parameters
    ----------
    hg : Hypergraph
        The hypergraph to analyze
    weights : bool, default False
        Whether to use edge and vertex weights
    index : bool, default True
        Whether to return node indices mapping
    check_connected : bool, default True
        Whether to check if hypergraph is connected
        
    Returns
    -------
    P : numpy.ndarray or scipy.sparse matrix
        Probability transition matrix
    node_index : dict, optional
        Mapping from nodes to matrix indices (if index=True)
    """
    if not HAS_SCIPY_SKLEARN:
        raise ImportError("scipy is required for probability transition matrix computation")
    
    nodes = list(hg.nodes)
    num_nodes = len(nodes)
    
    if num_nodes == 0:
        return np.array([[]])
    
    # Create node index mapping
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Build incidence matrix
    edges = list(hg.edges)
    num_edges = len(edges)
    
    if num_edges == 0:
        # Return identity matrix for disconnected nodes
        P = np.eye(num_nodes)
        return (P, node_to_idx) if index else P
    
    # Incidence matrix: nodes x edges
    B = np.zeros((num_nodes, num_edges))
    edge_weights = np.ones(num_edges)
    
    for j, edge in enumerate(edges):
        edge_nodes = list(hg.get_edge_nodes(edge))
        
        if weights:
            # Use edge weights if available
            try:
                edge_weight = hg.incidences.data.filter(
                    hg.incidences.data['edge_id'] == edge
                ).select('weight').to_numpy().mean()
                edge_weights[j] = edge_weight if not np.isnan(edge_weight) else 1.0
            except:
                edge_weights[j] = 1.0
        
        for node in edge_nodes:
            if node in node_to_idx:
                i = node_to_idx[node]
                vertex_weight = 1.0
                
                if weights:
                    # Use vertex weights if available in incidence data
                    try:
                        vertex_weight = hg.incidences.data.filter(
                            (hg.incidences.data['edge_id'] == edge) & 
                            (hg.incidences.data['node_id'] == node)
                        ).select('weight').to_numpy()[0, 0]
                        vertex_weight = vertex_weight if not np.isnan(vertex_weight) else 1.0
                    except:
                        vertex_weight = 1.0
                
                B[i, j] = vertex_weight
    
    # Compute degree matrices
    D_v = np.diag(np.sum(B * edge_weights, axis=1))  # Node degrees
    D_e = np.diag(edge_weights)  # Edge weights
    D_e_inv = np.diag(1.0 / (edge_weights + 1e-12))  # Avoid division by zero
    
    # Compute edge vertex degrees
    Omega = np.diag(np.sum(B, axis=0))  # Edge sizes
    Omega_inv = np.diag(1.0 / (np.sum(B, axis=0) + 1e-12))
    
    # Probability transition matrix
    # P = D_v^{-1} * B * D_e * Omega^{-1} * B^T
    try:
        D_v_inv = np.linalg.pinv(D_v)  # Use pseudo-inverse for numerical stability
        P = D_v_inv @ B @ D_e @ Omega_inv @ B.T
    except np.linalg.LinAlgError:
        # Fallback to simple random walk
        warnings.warn("Numerical issues computing transition matrix, using simple random walk")
        degrees = np.sum(B, axis=1)
        degrees[degrees == 0] = 1  # Avoid division by zero
        P = np.diag(1.0 / degrees) @ B @ B.T
    
    # Check for connectivity if requested
    if check_connected:
        # Simple connectivity check: see if all nodes are reachable
        connected_components = _find_connected_components(P)
        if len(connected_components) > 1:
            warnings.warn(f"Hypergraph has {len(connected_components)} connected components")
    
    if index:
        return P, node_to_idx
    else:
        return P


def get_pi(P):
    """
    Compute the stationary distribution of a probability transition matrix.
    
    Parameters
    ----------
    P : numpy.ndarray
        Probability transition matrix
        
    Returns
    -------
    pi : numpy.ndarray
        Stationary distribution
    """
    if not HAS_SCIPY_SKLEARN:
        raise ImportError("scipy is required for stationary distribution computation")
    
    try:
        # Find leading eigenvector
        eigenvals, eigenvecs = eigs(P.T, k=1, which='LR')
        pi = np.real(eigenvecs[:, 0])
        
        # Normalize to be a probability distribution
        pi = np.abs(pi)  # Ensure non-negative
        pi = pi / np.sum(pi)
        
        return pi
    
    except:
        # Fallback: uniform distribution
        warnings.warn("Could not compute stationary distribution, using uniform")
        return np.ones(P.shape[0]) / P.shape[0]


def norm_lap(P, pi=None):
    """
    Compute the normalized Laplacian matrix from a probability transition matrix.
    
    Parameters
    ----------
    P : numpy.ndarray
        Probability transition matrix
    pi : numpy.ndarray, optional
        Stationary distribution. If None, computed automatically
        
    Returns
    -------
    L : numpy.ndarray
        Normalized Laplacian matrix
    """
    if pi is None:
        pi = get_pi(P)
    
    # Normalized Laplacian: L = I - (1/2)(Π^{1/2} P Π^{-1/2} + Π^{-1/2} P^T Π^{1/2})
    sqrt_pi = np.sqrt(pi + 1e-12)
    inv_sqrt_pi = 1.0 / sqrt_pi
    
    Pi_sqrt = np.diag(sqrt_pi)
    Pi_inv_sqrt = np.diag(inv_sqrt_pi)
    
    # Symmetrized transition matrix
    P_sym = Pi_sqrt @ P @ Pi_inv_sqrt
    P_sym_T = Pi_inv_sqrt @ P.T @ Pi_sqrt
    
    I = np.eye(P.shape[0])
    L = I - 0.5 * (P_sym + P_sym_T)
    
    return L


def spec_clus(L, k, method='kmeans'):
    """
    Perform spectral clustering using the normalized Laplacian.
    
    Parameters
    ----------
    L : numpy.ndarray
        Normalized Laplacian matrix
    k : int
        Number of clusters
    method : str, default 'kmeans'
        Clustering method ('kmeans')
        
    Returns
    -------
    clusters : numpy.ndarray
        Cluster assignments for each node
    eigenvals : numpy.ndarray
        Eigenvalues used for clustering
    eigenvecs : numpy.ndarray
        Eigenvectors used for clustering
    """
    if not HAS_SCIPY_SKLEARN:
        raise ImportError("scipy and sklearn are required for spectral clustering")
    
    try:
        # Compute k smallest eigenvalues and eigenvectors
        eigenvals, eigenvecs = eigs(L, k=k, which='SM')
        
        # Sort by eigenvalue
        idx = np.argsort(np.real(eigenvals))
        eigenvals = np.real(eigenvals[idx])
        eigenvecs = np.real(eigenvecs[:, idx])
        
        # Use eigenvectors as features for clustering
        X = eigenvecs
        
        # Normalize rows
        X = preprocessing.normalize(X, norm='l2', axis=1)
        
        # Apply k-means clustering
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return clusters, eigenvals, eigenvecs
    
    except Exception as e:
        warnings.warn(f"Spectral clustering failed: {e}, using random assignment")
        return np.random.randint(0, k, size=L.shape[0]), None, None


def hypergraph_spectral_clustering(hg, k, weights=False, return_details=False):
    """
    Perform spectral clustering on a hypergraph.
    
    This is the main high-level interface for hypergraph spectral clustering.
    
    Parameters
    ----------
    hg : Hypergraph
        The hypergraph to cluster
    k : int
        Number of clusters
    weights : bool, default False
        Whether to use edge and vertex weights
    return_details : bool, default False
        Whether to return detailed computation results
        
    Returns
    -------
    clusters : dict
        Mapping from nodes to cluster assignments
    details : dict, optional
        Detailed computation results (if return_details=True)
    """
    if not HAS_SCIPY_SKLEARN:
        raise ImportError("scipy and sklearn are required for spectral clustering")
    
    # Step 1: Compute probability transition matrix
    P, node_index = prob_trans(hg, weights=weights, index=True)
    
    # Step 2: Compute stationary distribution
    pi = get_pi(P)
    
    # Step 3: Compute normalized Laplacian
    L = norm_lap(P, pi)
    
    # Step 4: Perform spectral clustering
    cluster_assignments, eigenvals, eigenvecs = spec_clus(L, k)
    
    # Convert back to node names
    idx_to_node = {i: node for node, i in node_index.items()}
    clusters = {idx_to_node[i]: cluster_assignments[i] for i in range(len(cluster_assignments))}
    
    if return_details:
        details = {
            'transition_matrix': P,
            'stationary_distribution': pi,
            'laplacian': L,
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs,
            'node_index': node_index
        }
        return clusters, details
    else:
        return clusters


def _find_connected_components(P, threshold=1e-10):
    """
    Find connected components in the transition matrix.
    
    Parameters
    ----------
    P : numpy.ndarray
        Probability transition matrix
    threshold : float
        Threshold for considering edges
        
    Returns
    -------
    components : list
        List of connected components (as lists of node indices)
    """
    n = P.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []
    
    def dfs(node, component):
        visited[node] = True
        component.append(node)
        
        # Check outgoing edges
        for neighbor in range(n):
            if not visited[neighbor] and P[node, neighbor] > threshold:
                dfs(neighbor, component)
        
        # Check incoming edges
        for neighbor in range(n):
            if not visited[neighbor] and P[neighbor, node] > threshold:
                dfs(neighbor, component)
    
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)
    
    return components


def analyze_clustering_quality(hg, clusters, weights=False):
    """
    Analyze the quality of a clustering result.
    
    Parameters
    ----------
    hg : Hypergraph
        The hypergraph that was clustered
    clusters : dict
        Clustering result mapping nodes to cluster IDs
    weights : bool, default False
        Whether to consider weights in analysis
        
    Returns
    -------
    analysis : dict
        Quality metrics for the clustering
    """
    num_clusters = len(set(clusters.values()))
    cluster_sizes = {}
    
    # Count cluster sizes
    for cluster_id in clusters.values():
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
    
    # Analyze edge distribution across clusters
    intra_cluster_edges = 0
    inter_cluster_edges = 0
    total_edges = 0
    
    for edge in hg.edges:
        edge_nodes = list(hg.get_edge_nodes(edge))
        edge_clusters = set(clusters[node] for node in edge_nodes if node in clusters)
        
        total_edges += 1
        if len(edge_clusters) == 1:
            intra_cluster_edges += 1
        else:
            inter_cluster_edges += 1
    
    # Compute modularity-like measure
    if total_edges > 0:
        intra_cluster_ratio = intra_cluster_edges / total_edges
    else:
        intra_cluster_ratio = 0.0
    
    analysis = {
        'num_clusters': num_clusters,
        'cluster_sizes': cluster_sizes,
        'intra_cluster_edges': intra_cluster_edges,
        'inter_cluster_edges': inter_cluster_edges,
        'total_edges': total_edges,
        'intra_cluster_ratio': intra_cluster_ratio,
        'modularity_estimate': intra_cluster_ratio  # Simplified modularity
    }
    
    return analysis


# Export main functions
__all__ = [
    'prob_trans',
    'get_pi', 
    'norm_lap',
    'spec_clus',
    'hypergraph_spectral_clustering',
    'analyze_clustering_quality'
]