"""
Core operations for Hypergraph

Handles basic graph structure operations including node/edge addition, removal, and basic queries.
"""

from typing import Any, Dict, List, Optional, Set, Iterable
import polars as pl
from ....exceptions import (
    AnantError, 
    ValidationError, 
    HypergraphError,
    NodeError,
    EdgeError,
    IncidenceError
)


class CoreOperations:
    """
    Core graph structure operations
    
    Handles fundamental hypergraph operations like adding/removing nodes and edges,
    basic queries, and structure validation.
    """
    
    def __init__(self, hypergraph):
        """
        Initialize CoreOperations
        
        Parameters
        ----------
        hypergraph : Hypergraph
            Parent hypergraph instance
        """
        if hypergraph is None:
            raise ValidationError("Hypergraph instance cannot be None")
        self.hypergraph = hypergraph
    
    def nodes(self) -> set:
        """
        Get all nodes in the hypergraph
        
        Returns
        -------
        set
            Set of all node identifiers
        """
        try:
            if not hasattr(self.hypergraph, 'incidences') or self.hypergraph.incidences is None:
                return set()
            
            return self.hypergraph.incidences.get_all_nodes()
        except Exception as e:
            raise HypergraphError(f"Failed to get nodes: {e}")
    
    def edges(self) -> set:
        """
        Get all edges in the hypergraph
        
        Returns
        -------
        set
            Set of all edge identifiers
        """
        try:
            if not hasattr(self.hypergraph, 'incidences') or self.hypergraph.incidences is None:
                return set()
            
            return self.hypergraph.incidences.get_all_edges()
        except Exception as e:
            raise HypergraphError(f"Failed to get edges: {e}")
    
    def has_node(self, node_id: Any) -> bool:
        """
        Check if a node exists in the hypergraph
        
        Parameters
        ----------
        node_id : Any
            The node ID to check for
            
        Returns
        -------
        bool
            True if the node exists, False otherwise
        """
        try:
            self.hypergraph._build_performance_indexes()
            return node_id in (self.hypergraph._nodes_cache or set())
        except Exception as e:
            raise IncidenceError(f"Failed to check node existence: {e}")
    
    def has_edge(self, edge_id: Any) -> bool:
        """
        Check if an edge exists in the hypergraph
        
        Parameters
        ----------
        edge_id : Any
            The edge ID to check for
            
        Returns
        -------
        bool
            True if the edge exists, False otherwise
        """
        try:
            self.hypergraph._build_performance_indexes()
            return edge_id in (self.hypergraph._edges_cache or set())
        except Exception as e:
            raise IncidenceError(f"Failed to check edge existence: {e}")
    
    def add_node(self, node_id: Any, properties: Optional[Dict] = None) -> None:
        """
        Add a node to the hypergraph
        
        Parameters
        ----------
        node_id : Any
            Unique identifier for the node
        properties : Optional[Dict]
            Optional properties for the node
            
        Raises
        ------
        ValidationError
            If node_id is None or invalid
        HypergraphError
            If node addition fails
        """
        if node_id is None:
            raise ValidationError("Node ID cannot be None")
        
        try:
            # For isolated nodes, add to properties if provided
            if properties:
                if not hasattr(self.hypergraph, 'properties') or self.hypergraph.properties is None:
                    raise HypergraphError("Properties store not available")
                self.hypergraph.properties.set_node_properties(node_id, properties)
            
            # Clear cache to ensure consistency
            self.hypergraph._invalidate_cache()
            
        except Exception as e:
            raise HypergraphError(f"Failed to add node {node_id}: {e}")
    
    def add_nodes_from(self, nodes: Iterable[Any]) -> None:
        """
        Add multiple nodes to the hypergraph
        
        Parameters
        ----------
        nodes : Iterable[Any]
            Iterable of node identifiers
            
        Raises
        ------
        ValidationError
            If nodes is not iterable
        HypergraphError
            If node addition fails
        """
        if nodes is None:
            raise ValidationError("Nodes cannot be None")
        
        try:
            nodes_list = list(nodes)
        except TypeError:
            raise ValidationError("Nodes must be iterable")
        
        for node in nodes_list:
            self.add_node(node)
    
    def add_edge(self, edge_id: Any, node_list: List[Any], weight: float = 1.0, 
                 properties: Optional[Dict] = None) -> None:
        """
        Add an edge to the hypergraph
        
        Parameters
        ----------
        edge_id : Any
            Unique identifier for the edge
        node_list : List[Any]
            List of nodes connected by this edge
        weight : float, default 1.0
            Weight of the edge
        properties : Optional[Dict]
            Optional properties for the edge
            
        Raises
        ------
        ValidationError
            If parameters are invalid
        HypergraphError
            If edge addition fails
        """
        if edge_id is None:
            raise ValidationError("Edge ID cannot be None")
        if not isinstance(node_list, list) or len(node_list) == 0:
            raise ValidationError("Node list must be a non-empty list")
        if not isinstance(weight, (int, float)):
            raise ValidationError("Weight must be a number")
        
        try:
            # Create new rows for the edge
            new_rows = []
            for node_id in node_list:
                if node_id is None:
                    raise ValidationError("Node ID in edge cannot be None")
                new_rows.append({
                    'edge_id': edge_id,
                    'node_id': node_id,
                    'weight': weight
                })
            
            if new_rows:
                new_data = pl.DataFrame(new_rows)
                
                if not hasattr(self.hypergraph, 'incidences') or self.hypergraph.incidences is None:
                    raise HypergraphError("Incidence store not available")
                
                if self.hypergraph.incidences.data.is_empty():
                    self.hypergraph.incidences.data = new_data
                else:
                    self.hypergraph.incidences.data = pl.concat([self.hypergraph.incidences.data, new_data])
            
            # Add edge properties if provided
            if properties:
                if not hasattr(self.hypergraph, 'properties') or self.hypergraph.properties is None:
                    raise HypergraphError("Properties store not available")
                self.hypergraph.properties.set_edge_properties(edge_id, properties)
            
            # Update performance indexes incrementally if built
            if hasattr(self.hypergraph, '_indexes_built') and self.hypergraph._indexes_built:
                self._update_indexes_for_new_edge(edge_id, node_list)
            else:
                # Clear cache to force rebuild
                self.hypergraph._invalidate_cache()
                
        except Exception as e:
            raise HypergraphError(f"Failed to add edge {edge_id}: {e}")
    
    def remove_node(self, node_id: Any) -> None:
        """
        Remove a node and all its incident edges
        
        Parameters
        ----------
        node_id : Any
            The node to remove
            
        Raises
        ------
        ValidationError
            If node_id is None
        HypergraphError
            If node removal fails
        """
        if node_id is None:
            raise ValidationError("Node ID cannot be None")
        
        if not self.has_node(node_id):
            return  # Node doesn't exist, nothing to remove
        
        try:
            if not hasattr(self.hypergraph, 'incidences') or self.hypergraph.incidences is None:
                raise HypergraphError("Incidence store not available")
            
            # Remove all edges containing this node
            edges_to_remove = self.hypergraph.incidences.get_node_edges(node_id)
            for edge_id in edges_to_remove:
                self.remove_edge(edge_id)
            
            # Remove node properties if they exist
            if hasattr(self.hypergraph, 'properties') and self.hypergraph.properties is not None:
                # Note: PropertyStore might not have a direct remove method, 
                # this would need to be implemented in PropertyStore
                pass
            
            # Clear cache to ensure consistency
            self.hypergraph._invalidate_cache()
            
        except Exception as e:
            raise HypergraphError(f"Failed to remove node {node_id}: {e}")
    
    def remove_edge(self, edge_id: Any) -> None:
        """
        Remove an edge from the hypergraph
        
        Parameters
        ----------
        edge_id : Any
            The edge to remove
            
        Raises
        ------
        ValidationError
            If edge_id is None
        HypergraphError
            If edge removal fails
        """
        if edge_id is None:
            raise ValidationError("Edge ID cannot be None")
        
        if not self.has_edge(edge_id):
            return  # Edge doesn't exist, nothing to remove
        
        try:
            if not hasattr(self.hypergraph, 'incidences') or self.hypergraph.incidences is None:
                raise HypergraphError("Incidence store not available")
            
            # Remove edge from incidence data
            self.hypergraph.incidences.data = self.hypergraph.incidences.data.filter(
                pl.col('edge_id') != edge_id
            )
            
            # Remove edge properties if they exist
            if hasattr(self.hypergraph, 'properties') and self.hypergraph.properties is not None:
                # Note: PropertyStore might not have a direct remove method,
                # this would need to be implemented in PropertyStore
                pass
            
            # Clear cache to ensure consistency
            self.hypergraph._invalidate_cache()
            
        except Exception as e:
            raise HypergraphError(f"Failed to remove edge {edge_id}: {e}")
    
    def neighbors(self, node_id: Any) -> Set[Any]:
        """
        Get neighbors of a node (nodes connected through any shared edge)
        
        Parameters
        ----------
        node_id : Any
            The node to find neighbors for
            
        Returns
        -------
        Set[Any]
            Set of neighboring nodes
            
        Raises
        ------
        ValidationError
            If node_id is None or node doesn't exist
        IncidenceError
            If neighbor calculation fails
        """
        if node_id is None:
            raise ValidationError("Node ID cannot be None")
        
        if not self.has_node(node_id):
            raise ValidationError(f"Node {node_id} does not exist")
        
        try:
            neighbors = set()
            
            if not hasattr(self.hypergraph, 'incidences') or self.hypergraph.incidences is None:
                return neighbors
            
            # Get all edges containing this node
            node_edges = self.hypergraph.incidences.get_node_edges(node_id)
            
            # For each edge, add all other nodes as neighbors
            for edge_id in node_edges:
                edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge_id)
                neighbors.update(n for n in edge_nodes if n != node_id)
            
            return neighbors
            
        except Exception as e:
            raise IncidenceError(f"Failed to get neighbors for node {node_id}: {e}")
    
    def get_edge_nodes(self, edge_id: Any) -> List[Any]:
        """
        Get nodes connected by an edge
        
        Parameters
        ----------
        edge_id : Any
            The edge identifier
            
        Returns
        -------
        List[Any]
            List of nodes connected by the edge
            
        Raises
        ------
        ValidationError
            If edge_id is None or edge doesn't exist
        IncidenceError
            If edge node retrieval fails
        """
        if edge_id is None:
            raise ValidationError("Edge ID cannot be None")
        
        if not self.has_edge(edge_id):
            raise ValidationError(f"Edge {edge_id} does not exist")
        
        try:
            if not hasattr(self.hypergraph, 'incidences') or self.hypergraph.incidences is None:
                return []
            
            return self.hypergraph.incidences.get_edge_nodes(edge_id)
            
        except Exception as e:
            raise IncidenceError(f"Failed to get nodes for edge {edge_id}: {e}")
    
    def get_node_edges(self, node_id: Any) -> List[Any]:
        """
        Get edges incident to a node
        
        Parameters
        ----------
        node_id : Any
            The node identifier
            
        Returns
        -------
        List[Any]
            List of edges incident to the node
            
        Raises
        ------
        ValidationError
            If node_id is None or node doesn't exist  
        IncidenceError
            If node edge retrieval fails
        """
        if node_id is None:
            raise ValidationError("Node ID cannot be None")
        
        if not self.has_node(node_id):
            raise ValidationError(f"Node {node_id} does not exist")
        
        try:
            if not hasattr(self.hypergraph, 'incidences') or self.hypergraph.incidences is None:
                return []
            
            return self.hypergraph.incidences.get_node_edges(node_id)
            
        except Exception as e:
            raise IncidenceError(f"Failed to get edges for node {node_id}: {e}")
    
    def is_empty(self) -> bool:
        """
        Check if the hypergraph is empty (no edges)
        
        Returns
        -------
        bool
            True if hypergraph has no edges, False otherwise
        """
        try:
            if not hasattr(self.hypergraph, 'incidences') or self.hypergraph.incidences is None:
                return True
            return self.hypergraph.incidences.data.is_empty()
        except Exception:
            return True
    
    def _update_indexes_for_new_edge(self, edge_id: Any, node_list: List[Any]) -> None:
        """
        Update performance indexes when adding a new edge
        
        Parameters
        ----------
        edge_id : Any
            The new edge identifier
        node_list : List[Any]
            List of nodes in the edge
        """
        try:
            # Initialize caches if needed
            if not hasattr(self.hypergraph, '_edges_cache') or self.hypergraph._edges_cache is None:
                self.hypergraph._edges_cache = set()
            if not hasattr(self.hypergraph, '_nodes_cache') or self.hypergraph._nodes_cache is None:
                self.hypergraph._nodes_cache = set()
            if not hasattr(self.hypergraph, '_edge_to_nodes'):
                self.hypergraph._edge_to_nodes = {}
            if not hasattr(self.hypergraph, '_node_to_edges'):
                self.hypergraph._node_to_edges = {}
            if not hasattr(self.hypergraph, '_edge_sizes'):
                self.hypergraph._edge_sizes = {}
            if not hasattr(self.hypergraph, '_node_degrees'):
                self.hypergraph._node_degrees = {}
            
            # Update edge cache
            self.hypergraph._edges_cache.add(edge_id)
            node_set = set(node_list)
            self.hypergraph._edge_to_nodes[edge_id] = node_set
            self.hypergraph._edge_sizes[edge_id] = len(node_set)
            
            # Update node caches
            for node_id in node_list:
                self.hypergraph._nodes_cache.add(node_id)
                
                if node_id not in self.hypergraph._node_to_edges:
                    self.hypergraph._node_to_edges[node_id] = set()
                    
                self.hypergraph._node_to_edges[node_id].add(edge_id)
                self.hypergraph._node_degrees[node_id] = len(self.hypergraph._node_to_edges[node_id])
                
        except Exception as e:
            # If index update fails, invalidate cache to force rebuild
            self.hypergraph._invalidate_cache()
            raise IncidenceError(f"Failed to update performance indexes: {e}")