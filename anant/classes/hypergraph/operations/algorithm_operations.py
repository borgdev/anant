"""
Algorithm operations for Hypergraph

Handles graph algorithms including centrality measures, PageRank, HITS,
shortest paths, connected components, and other graph-theoretic algorithms.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import random
import math
from collections import deque, defaultdict
from ....exceptions import HypergraphError, ValidationError, NodeError


class AlgorithmOperations:
    """
    Graph algorithm operations for hypergraph
    
    Provides implementations of various graph algorithms adapted for hypergraphs,
    including centrality measures, shortest paths, random walks, and more.
    """
    
    def __init__(self, hypergraph):
        """
        Initialize AlgorithmOperations
        
        Parameters
        ----------
        hypergraph : Hypergraph
            Parent hypergraph instance
        """
        if hypergraph is None:
            raise HypergraphError("Hypergraph instance cannot be None")
        self.hypergraph = hypergraph
    
    def random_walk(self, start_node: Any, steps: int = 100, seed: Optional[int] = None) -> List[Any]:
        """
        Perform a random walk on the hypergraph
        
        Parameters
        ----------
        start_node : Any
            Starting node for the walk
        steps : int, default 100
            Number of steps to take
        seed : Optional[int]
            Random seed for reproducibility
            
        Returns
        -------
        List[Any]
            List of nodes visited during the walk
            
        Raises
        ------
        ValidationError
            If start_node is not in the hypergraph
        HypergraphError
            If random walk fails
        """
        if start_node not in self.hypergraph.nodes:
            raise ValidationError(f"Start node {start_node} not in hypergraph")
        
        if steps <= 0:
            raise ValidationError("Steps must be positive")
        
        try:
            if seed is not None:
                random.seed(seed)
            
            walk = [start_node]
            current_node = start_node
            
            for _ in range(steps):
                # Get edges containing current node
                node_edges = self.hypergraph.incidences.get_node_edges(current_node)
                
                if not node_edges:
                    break
                
                # Choose random edge
                random_edge = random.choice(list(node_edges))
                
                # Get nodes in this edge
                edge_nodes = list(self.hypergraph.incidences.get_edge_nodes(random_edge))
                
                # Remove current node from choices
                possible_next = [n for n in edge_nodes if n != current_node]
                
                if not possible_next:
                    break
                
                # Choose random next node
                current_node = random.choice(possible_next)
                walk.append(current_node)
            
            return walk
            
        except Exception as e:
            raise HypergraphError(f"Random walk failed: {e}")
    
    def shortest_path(self, source: Any, target: Any) -> Optional[List[Any]]:
        """
        Find shortest path between two nodes using BFS
        
        Parameters
        ----------
        source : Any
            Source node
        target : Any
            Target node
            
        Returns
        -------
        Optional[List[Any]]
            Shortest path as list of nodes, or None if no path exists
            
        Raises
        ------
        ValidationError
            If source or target nodes don't exist
        HypergraphError
            If shortest path calculation fails
        """
        if source not in self.hypergraph.nodes:
            raise ValidationError(f"Source node {source} not in hypergraph")
        if target not in self.hypergraph.nodes:
            raise ValidationError(f"Target node {target} not in hypergraph")
        
        if source == target:
            return [source]
        
        try:
            # BFS to find shortest path
            queue = deque([(source, [source])])
            visited = {source}
            
            while queue:
                current, path = queue.popleft()
                
                # Get neighbors through shared edges
                neighbors = set()
                for edge in self.hypergraph.incidences.get_node_edges(current):
                    edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                    neighbors.update(n for n in edge_nodes if n != current)
                
                for neighbor in neighbors:
                    if neighbor == target:
                        return path + [neighbor]
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
            
            return None  # No path found
            
        except Exception as e:
            raise HypergraphError(f"Shortest path calculation failed: {e}")
    
    def connected_components(self) -> List[List[Any]]:
        """
        Find all connected components in the hypergraph
        
        Returns
        -------
        List[List[Any]]
            List of connected components, each as a list of nodes
            
        Raises
        ------
        HypergraphError
            If connected components calculation fails
        """
        try:
            if not self.hypergraph.nodes:
                return []
            
            visited = set()
            components = []
            
            def dfs(node, component):
                """DFS to explore connected component"""
                visited.add(node)
                component.append(node)
                
                # Get neighbors through shared edges
                neighbors = set()
                for edge in self.hypergraph.incidences.get_node_edges(node):
                    edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                    neighbors.update(n for n in edge_nodes if n != node)
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        dfs(neighbor, component)
            
            for node in self.hypergraph.nodes:
                if node not in visited:
                    component = []
                    dfs(node, component)
                    components.append(component)
            
            return components
            
        except Exception as e:
            raise HypergraphError(f"Connected components calculation failed: {e}")
    
    def betweenness_centrality(self, normalized: bool = True) -> Dict[Any, float]:
        """
        Calculate betweenness centrality for all nodes
        
        Parameters
        ----------
        normalized : bool, default True
            Whether to normalize the values
            
        Returns
        -------
        Dict[Any, float]
            Dictionary mapping nodes to centrality values
            
        Raises
        ------
        HypergraphError
            If centrality calculation fails
        """
        try:
            nodes = list(self.hypergraph.nodes)
            if not nodes:
                return {}
            
            centrality = {node: 0.0 for node in nodes}
            
            # For each pair of nodes, find shortest paths and count betweenness
            for source in nodes:
                # Single-source shortest paths using BFS
                stack = []
                paths = {node: [] for node in nodes}
                paths[source] = [source]
                
                sigma = {node: 0.0 for node in nodes}
                sigma[source] = 1.0
                
                distances = {source: 0}
                queue = [source]
                
                while queue:
                    current = queue.pop(0)
                    stack.append(current)
                    
                    # Get neighbors through shared edges
                    neighbors = set()
                    for edge in self.hypergraph.incidences.get_node_edges(current):
                        edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                        neighbors.update(n for n in edge_nodes if n != current)
                    
                    for neighbor in neighbors:
                        # First time visiting neighbor
                        if neighbor not in distances:
                            distances[neighbor] = distances[current] + 1
                            queue.append(neighbor)
                        
                        # Shortest path to neighbor via current
                        if distances[neighbor] == distances[current] + 1:
                            sigma[neighbor] += sigma[current]
                            paths[neighbor].append(current)
                
                # Accumulate betweenness centrality
                delta = {node: 0.0 for node in nodes}
                
                # Back-propagate dependencies
                while stack:
                    current = stack.pop()
                    for predecessor in paths[current]:
                        if predecessor != current:
                            delta[predecessor] += (sigma[predecessor] / sigma[current]) * (1 + delta[current])
                    
                    if current != source:
                        centrality[current] += delta[current]
            
            # Normalize if requested
            if normalized and len(nodes) > 2:
                # Normalization factor for betweenness centrality
                norm = 2.0 / ((len(nodes) - 1) * (len(nodes) - 2))
                centrality = {node: value * norm for node, value in centrality.items()}
            
            return centrality
            
        except Exception as e:
            raise HypergraphError(f"Betweenness centrality calculation failed: {e}")
    
    def closeness_centrality(self, normalized: bool = True) -> Dict[Any, float]:
        """
        Calculate closeness centrality for all nodes
        
        Parameters
        ----------
        normalized : bool, default True
            Whether to normalize the values
            
        Returns
        -------
        Dict[Any, float]
            Dictionary mapping nodes to centrality values
            
        Raises
        ------
        HypergraphError
            If centrality calculation fails
        """
        try:
            nodes = list(self.hypergraph.nodes)
            if not nodes:
                return {}
            
            centrality = {}
            
            for node in nodes:
                # Calculate shortest distances to all other nodes using BFS
                distances = {}
                queue = [(node, 0)]
                visited = {node}
                
                while queue:
                    current, dist = queue.pop(0)
                    distances[current] = dist
                    
                    # Get neighbors through shared edges
                    neighbors = set()
                    for edge in self.hypergraph.incidences.get_node_edges(current):
                        edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                        neighbors.update(n for n in edge_nodes if n != current)
                    
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))
                
                if len(distances) <= 1:
                    centrality[node] = 0.0
                    continue
                
                # Sum of distances (excluding distance to self)
                total_distance = sum(d for n, d in distances.items() if d > 0)
                
                if total_distance == 0:
                    centrality[node] = 0.0
                else:
                    # Closeness is inverse of average distance
                    n_reachable = len([d for d in distances.values() if d > 0])
                    closeness = (n_reachable - 1) / total_distance if n_reachable > 1 else 0.0
                    
                    if normalized and len(nodes) > 1:
                        # Normalize by the maximum possible closeness
                        closeness *= (n_reachable - 1) / (len(nodes) - 1)
                    
                    centrality[node] = closeness
            
            return centrality
            
        except Exception as e:
            raise HypergraphError(f"Closeness centrality calculation failed: {e}")
    
    def eigenvector_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> Dict[Any, float]:
        """
        Calculate eigenvector centrality using power iteration
        
        Parameters
        ----------
        max_iter : int, default 100
            Maximum number of iterations
        tol : float, default 1e-6
            Convergence tolerance
            
        Returns
        -------
        Dict[Any, float]
            Dictionary mapping nodes to centrality values
            
        Raises
        ------
        ValidationError
            If parameters are invalid
        HypergraphError
            If centrality calculation fails
        """
        if max_iter <= 0:
            raise ValidationError("max_iter must be positive")
        if tol <= 0:
            raise ValidationError("tol must be positive")
        
        try:
            nodes = list(self.hypergraph.nodes)
            n = len(nodes)
            
            if n == 0:
                return {}
            
            # Initialize centrality values
            centrality = {node: 1.0 for node in nodes}
            
            # Power iteration
            for iteration in range(max_iter):
                old_centrality = centrality.copy()
                
                # Update centrality values
                for node in nodes:
                    new_value = 0.0
                    
                    # Sum centrality of neighbors
                    neighbors = set()
                    for edge in self.hypergraph.incidences.get_node_edges(node):
                        edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                        neighbors.update(n for n in edge_nodes if n != node)
                    
                    for neighbor in neighbors:
                        new_value += old_centrality[neighbor]
                    
                    centrality[node] = new_value
                
                # Normalize
                norm = sum(centrality.values()) or 1.0
                centrality = {node: value / norm for node, value in centrality.items()}
                
                # Check convergence
                max_change = max(abs(centrality[node] - old_centrality[node]) 
                               for node in nodes)
                
                if max_change < tol:
                    break
            
            return centrality
            
        except Exception as e:
            raise HypergraphError(f"Eigenvector centrality calculation failed: {e}")
    
    def hits(self, max_iter: int = 100, tol: float = 1e-6) -> Tuple[Dict[Any, float], Dict[Any, float]]:
        """
        HITS algorithm (Hyperlink-Induced Topic Search) adapted for hypergraphs
        
        Parameters
        ----------
        max_iter : int, default 100
            Maximum iterations
        tol : float, default 1e-6
            Convergence tolerance
            
        Returns
        -------
        Tuple[Dict[Any, float], Dict[Any, float]]
            Tuple of (hub_scores, authority_scores)
            
        Raises
        ------
        ValidationError
            If parameters are invalid
        HypergraphError
            If HITS calculation fails
        """
        if max_iter <= 0:
            raise ValidationError("max_iter must be positive")
        if tol <= 0:
            raise ValidationError("tol must be positive")
        
        try:
            nodes = list(self.hypergraph.nodes)
            n = len(nodes)
            
            if n == 0:
                return {}, {}
            
            # Initialize scores
            hub_scores = {node: 1.0 for node in nodes}
            auth_scores = {node: 1.0 for node in nodes}
            
            # Build adjacency information
            node_to_neighbors = {}
            for node in nodes:
                neighbors = set()
                for edge in self.hypergraph.incidences.get_node_edges(node):
                    edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                    neighbors.update(n for n in edge_nodes if n != node)
                node_to_neighbors[node] = neighbors
            
            for iteration in range(max_iter):
                old_hub_scores = hub_scores.copy()
                old_auth_scores = auth_scores.copy()
                
                # Update authority scores
                for node in nodes:
                    auth_scores[node] = sum(hub_scores[neighbor] 
                                          for neighbor in node_to_neighbors.get(node, []))
                
                # Update hub scores
                for node in nodes:
                    hub_scores[node] = sum(auth_scores[neighbor] 
                                         for neighbor in node_to_neighbors.get(node, []))
                
                # Normalize scores
                hub_norm = sum(score ** 2 for score in hub_scores.values()) ** 0.5
                auth_norm = sum(score ** 2 for score in auth_scores.values()) ** 0.5
                
                if hub_norm > 0:
                    hub_scores = {node: score / hub_norm for node, score in hub_scores.items()}
                if auth_norm > 0:
                    auth_scores = {node: score / auth_norm for node, score in auth_scores.items()}
                
                # Check convergence
                hub_change = max(abs(hub_scores[node] - old_hub_scores[node]) for node in nodes)
                auth_change = max(abs(auth_scores[node] - old_auth_scores[node]) for node in nodes)
                
                if hub_change < tol and auth_change < tol:
                    break
            
            return hub_scores, auth_scores
            
        except Exception as e:
            raise HypergraphError(f"HITS calculation failed: {e}")
    
    def pagerank(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> Dict[Any, float]:
        """
        PageRank algorithm adapted for hypergraphs
        
        Parameters
        ----------
        alpha : float, default 0.85
            Damping parameter (between 0 and 1)
        max_iter : int, default 100
            Maximum iterations
        tol : float, default 1e-6
            Convergence tolerance
            
        Returns
        -------
        Dict[Any, float]
            Dictionary mapping nodes to PageRank values
            
        Raises
        ------
        ValidationError
            If parameters are invalid
        HypergraphError
            If PageRank calculation fails
        """
        if not 0 <= alpha <= 1:
            raise ValidationError("alpha must be between 0 and 1")
        if max_iter <= 0:
            raise ValidationError("max_iter must be positive")
        if tol <= 0:
            raise ValidationError("tol must be positive")
        
        try:
            nodes = list(self.hypergraph.nodes)
            n = len(nodes)
            
            if n == 0:
                return {}
            
            # Initialize PageRank values
            pr_values = {node: 1.0 / n for node in nodes}
            
            # Build transition matrix (simplified)
            node_to_neighbors = {}
            for node in nodes:
                neighbors = set()
                for edge in self.hypergraph.incidences.get_node_edges(node):
                    edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                    neighbors.update(n for n in edge_nodes if n != node)
                node_to_neighbors[node] = neighbors
            
            for iteration in range(max_iter):
                old_pr_values = pr_values.copy()
                
                # Calculate new PageRank values
                for node in nodes:
                    rank_sum = 0.0
                    
                    # Sum contributions from neighbors
                    for other_node in nodes:
                        if node in node_to_neighbors.get(other_node, set()):
                            out_degree = len(node_to_neighbors.get(other_node, set()))
                            if out_degree > 0:
                                rank_sum += old_pr_values[other_node] / out_degree
                    
                    pr_values[node] = (1 - alpha) / n + alpha * rank_sum
                
                # Check convergence
                max_change = max(abs(pr_values[node] - old_pr_values[node]) 
                               for node in nodes)
                
                if max_change < tol:
                    break
            
            return pr_values
            
        except Exception as e:
            raise HypergraphError(f"PageRank calculation failed: {e}")
    
    def minimum_spanning_tree(self):
        """
        Find minimum spanning tree (simplified for hypergraphs)
        
        Note: This converts hyperedges to pairwise edges for MST calculation
        
        Returns
        -------
        Hypergraph
            Minimum spanning tree as a hypergraph
            
        Raises
        ------
        HypergraphError
            If MST calculation fails
        """
        try:
            # Convert to pairwise edges first
            pairwise_edges = {}
            edge_counter = 0
            
            for edge in self.hypergraph.edges:
                edge_nodes = list(self.hypergraph.incidences.get_edge_nodes(edge))
                # Create pairwise connections for MST
                for i, node1 in enumerate(edge_nodes):
                    for node2 in edge_nodes[i+1:]:
                        edge_id = f"mst_edge_{edge_counter}"
                        pairwise_edges[edge_id] = [node1, node2]
                        edge_counter += 1
            
            # For simplicity, return a spanning tree structure
            if not pairwise_edges:
                # Import hypergraph class dynamically to avoid circular imports
                return type(self.hypergraph)({}, name=f"mst_{self.hypergraph.name}")
            
            # Use a simple approach - DFS to ensure connectivity
            visited = set()
            mst_edges = {}
            start_node = next(iter(self.hypergraph.nodes))
            
            def dfs_mst(node):
                """DFS for MST construction"""
                visited.add(node)
                for edge_id, (n1, n2) in pairwise_edges.items():
                    if node == n1 and n2 not in visited:
                        mst_edges[edge_id] = [n1, n2]
                        dfs_mst(n2)
                    elif node == n2 and n1 not in visited:
                        mst_edges[edge_id] = [n1, n2]
                        dfs_mst(n1)
            
            dfs_mst(start_node)
            return type(self.hypergraph)(mst_edges, name=f"mst_{self.hypergraph.name}")
            
        except Exception as e:
            raise HypergraphError(f"MST calculation failed: {e}")
    
    def max_flow(self, source: Any, sink: Any) -> float:
        """
        Maximum flow between source and sink (simplified)
        
        Parameters
        ----------
        source : Any
            Source node
        sink : Any
            Sink node
            
        Returns
        -------
        float
            Maximum flow value
            
        Raises
        ------
        ValidationError
            If source or sink nodes don't exist
        HypergraphError
            If max flow calculation fails
        """
        if source not in self.hypergraph.nodes:
            raise ValidationError(f"Source node {source} not in hypergraph")
        if sink not in self.hypergraph.nodes:
            raise ValidationError(f"Sink node {sink} not in hypergraph")
        
        try:
            if source == sink:
                return 0.0
            
            # Simplified max flow - count edge-disjoint paths
            # This is a placeholder implementation
            path = self.shortest_path(source, sink)
            return 1.0 if path else 0.0
            
        except Exception as e:
            raise HypergraphError(f"Max flow calculation failed: {e}")
    
    def min_cut(self, source: Any, sink: Any) -> float:
        """
        Minimum cut between source and sink
        
        Parameters
        ----------
        source : Any
            Source node
        sink : Any
            Sink node
            
        Returns
        -------
        float
            Minimum cut value
            
        Raises
        ------
        ValidationError
            If source or sink nodes don't exist
        HypergraphError
            If min cut calculation fails
        """
        try:
            # By max-flow min-cut theorem, min cut = max flow
            return self.max_flow(source, sink)
        except Exception as e:
            raise HypergraphError(f"Min cut calculation failed: {e}")
    
    def degree_centrality(self, normalized: bool = True) -> Dict[Any, float]:
        """
        Calculate degree centrality for all nodes
        
        Parameters
        ----------
        normalized : bool, default True
            Whether to normalize by (n-1) where n is number of nodes
            
        Returns
        -------
        Dict[Any, float]
            Dictionary mapping nodes to degree centrality values
            
        Raises
        ------
        HypergraphError
            If degree centrality calculation fails
        """
        try:
            nodes = list(self.hypergraph.nodes)
            if not nodes:
                return {}
            
            centrality = {}
            n = len(nodes)
            
            for node in nodes:
                degree = self.hypergraph.get_node_degree(node)
                
                if normalized and n > 1:
                    centrality[node] = degree / (n - 1)
                else:
                    centrality[node] = float(degree)
            
            return centrality
            
        except Exception as e:
            raise HypergraphError(f"Degree centrality calculation failed: {e}")