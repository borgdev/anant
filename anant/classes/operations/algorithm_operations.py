"""
Algorithm Operations for Hypergraphs
====================================

Graph algorithm implementations including:
- Path finding (shortest path, all shortest paths)
- Connectivity analysis (connected components, diameter)
- Graph transformations (dual graph, line graph)
- Random walks and traversal algorithms
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Set
import logging
from collections import deque, defaultdict
import random

logger = logging.getLogger(__name__)


class AlgorithmOperations:
    """
    Handles graph algorithm operations for hypergraphs.
    
    This class provides methods for path finding, connectivity analysis,
    graph transformations, and traversal algorithms.
    """
    
    def __init__(self, hypergraph):
        """
        Initialize algorithm operations.
        
        Args:
            hypergraph: Reference to parent Hypergraph instance
        """
        self.hg = hypergraph
        self.logger = logger.getChild(self.__class__.__name__)
    
    def shortest_path(self, source: Any, target: Any) -> Optional[List[Any]]:
        """
        Find the shortest path between two nodes using BFS.
        
        Args:
            source: Source node
            target: Target node
            
        Returns:
            List of nodes representing the shortest path, or None if no path exists
        """
        if not self.hg.core_ops.has_node(source) or not self.hg.core_ops.has_node(target):
            return None
        
        if source == target:
            return [source]
        
        # BFS to find shortest path
        queue = deque([(source, [source])])
        visited = {source}
        
        while queue:
            current_node, path = queue.popleft()
            
            # Get neighbors (nodes that share edges with current node)
            neighbors = set()
            for edge_id in self.hg.core_ops.get_node_edges(current_node):
                edge_nodes = self.hg.core_ops.get_edge_nodes(edge_id)
                neighbors.update(node for node in edge_nodes if node != current_node)
            
            for neighbor in neighbors:
                if neighbor == target:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None  # No path found
    
    def connected_components(self) -> List[List[Any]]:
        """
        Find all connected components in the hypergraph.
        
        Returns:
            List of connected components, where each component is a list of nodes
        """
        visited = set()
        components = []
        
        for node in self.hg.core_ops.nodes():
            if node not in visited:
                # BFS to find component
                component = []
                queue = [node]
                visited.add(node)
                
                while queue:
                    current = queue.pop(0)
                    component.append(current)
                    
                    # Get neighbors
                    neighbors = set()
                    for edge_id in self.hg.core_ops.get_node_edges(current):
                        edge_nodes = self.hg.core_ops.get_edge_nodes(edge_id)
                        neighbors.update(node for node in edge_nodes if node != current)
                    
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                components.append(component)
        
        return components
    
    def diameter(self) -> Optional[int]:
        """
        Calculate the diameter of the hypergraph (longest shortest path).
        
        Returns:
            Diameter of the graph, or None if graph is disconnected
        """
        components = self.connected_components()
        
        # If more than one component, diameter is undefined
        if len(components) > 1:
            return None
        
        if len(components) == 0 or len(components[0]) <= 1:
            return 0
        
        nodes = list(self.hg.core_ops.nodes())
        max_distance = 0
        
        # Check all pairs (inefficient but correct for small graphs)
        for i, source in enumerate(nodes):
            for target in nodes[i+1:]:
                path = self.shortest_path(source, target)
                if path is None:
                    return None  # Disconnected
                max_distance = max(max_distance, len(path) - 1)
        
        return max_distance
    
    def all_shortest_paths(self, source: Any, target: Any = None) -> Dict[Any, List[List[Any]]]:
        """
        Find all shortest paths between nodes.
        
        Args:
            source: Source node
            target: Target node (if None, finds paths to all nodes)
            
        Returns:
            Dictionary mapping target nodes to lists of shortest paths
        """
        if source not in self.hg.core_ops.nodes():
            return {}
        
        # Use BFS to find shortest paths
        queue = deque([(source, [source])])
        distances = {source: 0}
        all_paths = defaultdict(list)
        visited_at_distance = defaultdict(set)
        visited_at_distance[0].add(source)
        
        while queue:
            current_node, path = queue.popleft()
            current_distance = len(path) - 1
            
            # Get neighbors
            neighbors = set()
            for edge_id in self.hg.core_ops.get_node_edges(current_node):
                edge_nodes = self.hg.core_ops.get_edge_nodes(edge_id)
                neighbors.update(node for node in edge_nodes if node != current_node)
            
            for neighbor in neighbors:
                new_distance = current_distance + 1
                
                if neighbor not in distances:
                    # First time visiting this node
                    distances[neighbor] = new_distance
                    all_paths[neighbor].append(path + [neighbor])
                    visited_at_distance[new_distance].add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
                elif distances[neighbor] == new_distance:
                    # Found another path of same length
                    all_paths[neighbor].append(path + [neighbor])
        
        # Filter by target if specified
        if target is not None:
            return {target: all_paths[target]} if target in all_paths else {}
        
        return dict(all_paths)
    
    def is_connected(self) -> bool:
        """
        Check if the hypergraph is connected.
        
        Returns:
            True if the hypergraph is connected
        """
        components = self.connected_components()
        return len(components) <= 1
    
    def neighbors(self, node_id: Any) -> Set[Any]:
        """
        Get all neighbors of a node (nodes that share edges).
        
        Args:
            node_id: Node identifier
            
        Returns:
            Set of neighbor node identifiers
        """
        neighbors = set()
        for edge_id in self.hg.core_ops.get_node_edges(node_id):
            edge_nodes = self.hg.core_ops.get_edge_nodes(edge_id)
            neighbors.update(node for node in edge_nodes if node != node_id)
        return neighbors
    
    def dual(self):
        """
        Create the dual hypergraph where nodes and edges are swapped.
        
        Returns:
            Dual hypergraph
        """
        # Import here to avoid circular imports
        from ..hypergraph_refactored import HypergraphRefactored as Hypergraph
        
        if self.hg.incidences.data.is_empty():
            return Hypergraph(name=f"{self.hg.name}_dual")
        
        # Import polars
        import polars as pl
        
        # Swap node_id and edge_id columns
        dual_data = self.hg.incidences.data.select([
            pl.col('node_id').alias('edge_id'),
            pl.col('edge_id').alias('node_id'),
            pl.col('weight')
        ])
        
        from ..incidence_store import IncidenceStore
        dual_store = IncidenceStore(dual_data)
        
        return Hypergraph(
            setsystem=dual_store,
            name=f"{self.hg.name}_dual"
        )
    
    def s_line_graph(self, s: int = 2):
        """
        Create the s-line graph where edges sharing at least s nodes become connected.
        
        Args:
            s: Minimum number of shared nodes for edge connection
            
        Returns:
            S-line graph hypergraph
        """
        # Import here to avoid circular imports
        from ..hypergraph_refactored import HypergraphRefactored as Hypergraph
        
        if s <= 0:
            raise ValueError("s must be positive")
        
        if self.hg.incidences.data.is_empty():
            return Hypergraph(name=f"{self.hg.name}_s{s}_line")
        
        # Get all edges and their nodes
        edge_nodes = {}
        for edge_id in self.hg.core_ops.edges():
            edge_nodes[edge_id] = set(self.hg.core_ops.get_edge_nodes(edge_id))
        
        # Create new edges for edge pairs sharing >= s nodes
        new_edges = {}
        edge_counter = 0
        
        edges = list(edge_nodes.keys())
        for i, edge1 in enumerate(edges):
            for edge2 in edges[i+1:]:
                shared_nodes = edge_nodes[edge1] & edge_nodes[edge2]
                if len(shared_nodes) >= s:
                    new_edge_id = f"line_{edge_counter}"
                    new_edges[new_edge_id] = [edge1, edge2]
                    edge_counter += 1
        
        return Hypergraph.from_dict(new_edges, name=f"{self.hg.name}_s{s}_line")
    
    def random_walk(self, start_node: Any, steps: int = 100) -> List[Any]:
        """
        Perform a random walk starting from a given node.
        
        Args:
            start_node: Starting node for the walk
            steps: Number of steps to take
            
        Returns:
            List of nodes visited during the walk
        """
        if not self.hg.core_ops.has_node(start_node):
            raise ValueError(f"Start node '{start_node}' not found in hypergraph")
        
        walk = [start_node]
        current = start_node
        
        for _ in range(steps):
            neighbors = list(self.neighbors(current))
            if not neighbors:
                break  # No neighbors, walk ends
            
            # Choose random neighbor
            current = random.choice(neighbors)
            walk.append(current)
        
        return walk
    
    def breadth_first_search(self, start_node: Any, max_depth: Optional[int] = None) -> Dict[Any, int]:
        """
        Perform breadth-first search from a starting node.
        
        Args:
            start_node: Starting node
            max_depth: Maximum depth to explore (None = unlimited)
            
        Returns:
            Dictionary mapping nodes to their distance from start
        """
        if not self.hg.core_ops.has_node(start_node):
            raise ValueError(f"Start node '{start_node}' not found in hypergraph")
        
        distances = {start_node: 0}
        queue = deque([start_node])
        
        while queue:
            current = queue.popleft()
            current_distance = distances[current]
            
            if max_depth is not None and current_distance >= max_depth:
                continue
            
            for neighbor in self.neighbors(current):
                if neighbor not in distances:
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)
        
        return distances
    
    def depth_first_search(self, start_node: Any, max_depth: Optional[int] = None) -> List[Any]:
        """
        Perform depth-first search from a starting node.
        
        Args:
            start_node: Starting node
            max_depth: Maximum depth to explore (None = unlimited)
            
        Returns:
            List of nodes in DFS order
        """
        if not self.hg.core_ops.has_node(start_node):
            raise ValueError(f"Start node '{start_node}' not found in hypergraph")
        
        visited = set()
        result = []
        
        def dfs_recursive(node: Any, depth: int):
            if max_depth is not None and depth > max_depth:
                return
            
            if node in visited:
                return
            
            visited.add(node)
            result.append(node)
            
            for neighbor in self.neighbors(node):
                dfs_recursive(neighbor, depth + 1)
        
        dfs_recursive(start_node, 0)
        return result
    
    def find_cycles(self) -> List[List[Any]]:
        """
        Find all simple cycles in the hypergraph.
        
        Returns:
            List of cycles, where each cycle is a list of nodes
        """
        cycles = []
        visited_global = set()
        
        for start_node in self.hg.core_ops.nodes():
            if start_node in visited_global:
                continue
            
            # DFS to find cycles starting from this node
            stack = [(start_node, [start_node], set([start_node]))]
            
            while stack:
                current, path, visited_in_path = stack.pop()
                
                for neighbor in self.neighbors(current):
                    if neighbor == start_node and len(path) > 2:
                        # Found a cycle back to start
                        cycles.append(path + [start_node])
                    elif neighbor not in visited_in_path:
                        new_path = path + [neighbor]
                        new_visited = visited_in_path | {neighbor}
                        stack.append((neighbor, new_path, new_visited))
            
            visited_global.add(start_node)
        
        # Remove duplicate cycles (same cycle in different directions)
        unique_cycles = []
        for cycle in cycles:
            # Normalize cycle to start with smallest element
            min_idx = cycle.index(min(cycle[:-1]))  # Exclude last element (duplicate of first)
            normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
            
            if normalized not in unique_cycles:
                unique_cycles.append(normalized)
        
        return unique_cycles