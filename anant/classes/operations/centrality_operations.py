"""
Centrality Operations for Hypergraphs
=====================================

Node centrality measure implementations including:
- Betweenness centrality 
- Closeness centrality
- Eigenvector centrality
- PageRank algorithm
- HITS algorithm (Hubs and Authorities)
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class CentralityOperations:
    """
    Handles centrality measure calculations for hypergraphs.
    
    This class provides methods for computing various node centrality
    measures that quantify the importance of nodes in the network.
    """
    
    def __init__(self, hypergraph):
        """
        Initialize centrality operations.
        
        Args:
            hypergraph: Reference to parent Hypergraph instance
        """
        self.hg = hypergraph
        self.logger = logger.getChild(self.__class__.__name__)
    
    def betweenness_centrality(self, normalized: bool = True) -> Dict[Any, float]:
        """
        Calculate betweenness centrality for all nodes.
        
        Betweenness centrality measures how often a node lies on the
        shortest paths between other nodes.
        
        Args:
            normalized: Whether to normalize values
            
        Returns:
            Dictionary mapping nodes to betweenness centrality values
        """
        nodes = list(self.hg.core_ops.nodes())
        n = len(nodes)
        
        if n <= 2:
            return {node: 0.0 for node in nodes}
        
        betweenness = {node: 0.0 for node in nodes}
        
        # Calculate shortest paths between all pairs
        for source in nodes:
            # Get all shortest paths from source
            all_paths = self.hg.algorithm_ops.all_shortest_paths(source)
            
            for target, paths in all_paths.items():
                if source != target and paths:
                    # Count how many shortest paths pass through each node
                    path_count = len(paths)
                    
                    for path in paths:
                        # Skip source and target
                        for intermediate in path[1:-1]:
                            betweenness[intermediate] += 1.0 / path_count
        
        # Normalize if requested
        if normalized and n > 2:
            # Normalization factor for undirected graphs
            norm_factor = 2.0 / ((n - 1) * (n - 2))
            betweenness = {node: value * norm_factor for node, value in betweenness.items()}
        
        return betweenness
    
    def closeness_centrality(self, normalized: bool = True) -> Dict[Any, float]:
        """
        Calculate closeness centrality for all nodes.
        
        Closeness centrality measures how close a node is to all other
        nodes in the graph (inverse of average distance).
        
        Args:
            normalized: Whether to normalize values
            
        Returns:
            Dictionary mapping nodes to closeness centrality values
        """
        nodes = list(self.hg.core_ops.nodes())
        closeness = {}
        
        for node in nodes:
            # Get shortest path lengths to all other nodes
            distances = self.hg.algorithm_ops.breadth_first_search(node)
            
            # Remove self-distance
            distances.pop(node, None)
            
            if not distances:
                closeness[node] = 0.0
                continue
            
            # Calculate average distance
            total_distance = sum(distances.values())
            reachable_nodes = len(distances)
            
            if total_distance == 0:
                closeness[node] = 0.0
            else:
                # Closeness is inverse of average distance
                avg_distance = total_distance / reachable_nodes
                closeness[node] = 1.0 / avg_distance
                
                # Normalize by fraction of reachable nodes
                if normalized:
                    total_nodes = len(nodes) - 1  # Exclude self
                    if total_nodes > 0:
                        closeness[node] *= reachable_nodes / total_nodes
        
        return closeness
    
    def eigenvector_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> Dict[Any, float]:
        """
        Calculate eigenvector centrality using power iteration.
        
        Eigenvector centrality measures the influence of a node based on
        the centrality of its neighbors.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Dictionary mapping nodes to eigenvector centrality values
        """
        nodes = list(self.hg.core_ops.nodes())
        n = len(nodes)
        
        if n == 0:
            return {}
        
        if n == 1:
            return {nodes[0]: 1.0}
        
        # Create adjacency matrix
        adjacency = self.hg.matrix_ops.adjacency_matrix(nodes)
        
        # Initialize centrality vector
        centrality = np.ones(n) / n
        
        # Power iteration
        for iteration in range(max_iter):
            old_centrality = centrality.copy()
            
            # Multiply by adjacency matrix
            centrality = adjacency.dot(centrality)
            
            # Normalize
            norm = np.linalg.norm(centrality)
            if norm > 0:
                centrality = centrality / norm
            
            # Check convergence
            if np.allclose(centrality, old_centrality, atol=tol):
                break
        
        # Convert to dictionary
        return {nodes[i]: float(centrality[i]) for i in range(n)}
    
    def pagerank(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> Dict[Any, float]:
        """
        Calculate PageRank centrality.
        
        PageRank models the behavior of a random walker that follows
        links with probability alpha and jumps randomly otherwise.
        
        Args:
            alpha: Damping parameter (0.0 to 1.0)
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Dictionary mapping nodes to PageRank values
        """
        nodes = list(self.hg.core_ops.nodes())
        n = len(nodes)
        
        if n == 0:
            return {}
        
        if n == 1:
            return {nodes[0]: 1.0}
        
        # Initialize PageRank values
        pagerank = np.ones(n) / n
        
        # Create transition matrix
        adjacency = self.hg.matrix_ops.adjacency_matrix(nodes)
        
        # Convert to row-stochastic matrix
        transition = adjacency.astype(float)
        for i in range(n):
            row_sum = transition[i].sum()
            if row_sum > 0:
                transition[i] = transition[i] / row_sum
            else:
                # Handle dangling nodes (no outlinks)
                transition[i] = 1.0 / n
        
        # Power iteration
        for iteration in range(max_iter):
            old_pagerank = pagerank.copy()
            
            # PageRank update formula
            pagerank = (1 - alpha) / n + alpha * transition.T.dot(pagerank)
            
            # Check convergence
            if np.allclose(pagerank, old_pagerank, atol=tol):
                break
        
        # Convert to dictionary
        return {nodes[i]: float(pagerank[i]) for i in range(n)}
    
    def hits(self, max_iter: int = 100, tol: float = 1e-6) -> Tuple[Dict[Any, float], Dict[Any, float]]:
        """
        Calculate HITS (Hyperlink-Induced Topic Search) centrality.
        
        HITS computes two scores for each node:
        - Authority score: based on incoming links from good hubs
        - Hub score: based on outgoing links to good authorities
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Tuple of (hub_scores, authority_scores) dictionaries
        """
        nodes = list(self.hg.core_ops.nodes())
        n = len(nodes)
        
        if n == 0:
            return {}, {}
        
        if n == 1:
            return {nodes[0]: 1.0}, {nodes[0]: 1.0}
        
        # Initialize scores
        hub_scores = np.ones(n)
        auth_scores = np.ones(n)
        
        # Create adjacency matrix
        adjacency = self.hg.matrix_ops.adjacency_matrix(nodes)
        
        # Power iteration
        for iteration in range(max_iter):
            old_hub = hub_scores.copy()
            old_auth = auth_scores.copy()
            
            # Update authority scores (sum of hub scores of nodes linking to this node)
            auth_scores = adjacency.T.dot(hub_scores)
            
            # Update hub scores (sum of authority scores of nodes this node links to)
            hub_scores = adjacency.dot(auth_scores)
            
            # Normalize scores
            hub_norm = np.linalg.norm(hub_scores)
            auth_norm = np.linalg.norm(auth_scores)
            
            if hub_norm > 0:
                hub_scores = hub_scores / hub_norm
            if auth_norm > 0:
                auth_scores = auth_scores / auth_norm
            
            # Check convergence
            if (np.allclose(hub_scores, old_hub, atol=tol) and 
                np.allclose(auth_scores, old_auth, atol=tol)):
                break
        
        # Convert to dictionaries
        hub_dict = {nodes[i]: float(hub_scores[i]) for i in range(n)}
        auth_dict = {nodes[i]: float(auth_scores[i]) for i in range(n)}
        
        return hub_dict, auth_dict
    
    def degree_centrality(self, normalized: bool = True) -> Dict[Any, float]:
        """
        Calculate degree centrality for all nodes.
        
        Degree centrality is simply the degree of each node,
        optionally normalized by the maximum possible degree.
        
        Args:
            normalized: Whether to normalize by maximum possible degree
            
        Returns:
            Dictionary mapping nodes to degree centrality values
        """
        nodes = list(self.hg.core_ops.nodes())
        n = len(nodes)
        
        degree_centrality = {}
        
        for node in nodes:
            degree = self.hg.core_ops.get_node_degree(node)
            
            if normalized and n > 1:
                # Normalize by maximum possible degree
                degree_centrality[node] = degree / (n - 1)
            else:
                degree_centrality[node] = float(degree)
        
        return degree_centrality
    
    def load_centrality(self) -> Dict[Any, float]:
        """
        Calculate load centrality (edge betweenness centrality).
        
        Load centrality measures how much "load" or "stress" 
        each edge carries in terms of shortest paths.
        
        Returns:
            Dictionary mapping edges to load centrality values
        """
        edges = list(self.hg.core_ops.edges())
        nodes = list(self.hg.core_ops.nodes())
        
        load_centrality = {edge: 0.0 for edge in edges}
        
        # Calculate shortest paths between all node pairs
        for source in nodes:
            for target in nodes:
                if source != target:
                    paths = self.hg.algorithm_ops.all_shortest_paths(source, target)
                    
                    if target in paths and paths[target]:
                        num_paths = len(paths[target])
                        
                        # Count edges used in shortest paths
                        for path in paths[target]:
                            # Convert path to edges
                            for i in range(len(path) - 1):
                                node1, node2 = path[i], path[i + 1]
                                
                                # Find edges connecting these nodes
                                edges1 = set(self.hg.core_ops.get_node_edges(node1))
                                edges2 = set(self.hg.core_ops.get_node_edges(node2))
                                shared_edges = edges1 & edges2
                                
                                # Distribute load among shared edges
                                if shared_edges:
                                    load_per_edge = 1.0 / (num_paths * len(shared_edges))
                                    for edge in shared_edges:
                                        load_centrality[edge] += load_per_edge
        
        return load_centrality
    
    def katz_centrality(self, alpha: float = 0.1, beta: float = 1.0, 
                       max_iter: int = 100, tol: float = 1e-6) -> Dict[Any, float]:
        """
        Calculate Katz centrality.
        
        Katz centrality measures influence based on the number of walks
        of all lengths, with longer walks weighted less.
        
        Args:
            alpha: Attenuation factor (should be < 1/largest_eigenvalue)
            beta: Weight given to immediate neighbors
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Dictionary mapping nodes to Katz centrality values
        """
        nodes = list(self.hg.core_ops.nodes())
        n = len(nodes)
        
        if n == 0:
            return {}
        
        # Initialize centrality values
        centrality = np.ones(n) * beta
        
        # Create adjacency matrix
        adjacency = self.hg.matrix_ops.adjacency_matrix(nodes)
        
        # Power iteration
        for iteration in range(max_iter):
            old_centrality = centrality.copy()
            
            # Katz centrality update
            centrality = beta + alpha * adjacency.T.dot(centrality)
            
            # Check convergence
            if np.allclose(centrality, old_centrality, atol=tol):
                break
        
        # Convert to dictionary
        return {nodes[i]: float(centrality[i]) for i in range(n)}