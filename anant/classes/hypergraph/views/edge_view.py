"""
EdgeView class for Hypergraph operations

Provides view functionality for accessing edge-node relationships with optimized performance.
"""

from typing import Any, List, Iterator, Optional


class EdgeView:
    """
    View class for accessing edge-node relationships
    
    This class provides a dictionary-like interface for accessing edge data
    while maintaining optimization through the parent hypergraph's caching system.
    """
    
    def __init__(self, hypergraph):
        """
        Initialize EdgeView
        
        Parameters
        ----------
        hypergraph : Hypergraph
            The parent hypergraph instance
        """
        if hypergraph is None:
            raise ValueError("hypergraph cannot be None")
        self.hypergraph = hypergraph
    
    def __getitem__(self, edge_id: Any) -> List[Any]:
        """
        Get nodes for an edge
        
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
        KeyError
            If the edge does not exist
        """
        if not self.hypergraph.has_edge(edge_id):
            raise KeyError(f"Edge {edge_id} not found in hypergraph")
        return self.hypergraph.get_edge_nodes(edge_id)
    
    def __iter__(self) -> Iterator[Any]:
        """
        Iterate through edge IDs using optimized cache
        
        Returns
        -------
        Iterator[Any]
            Iterator over edge identifiers
        """
        try:
            self.hypergraph._build_performance_indexes()
            
            if hasattr(self.hypergraph, '_edges_cache') and self.hypergraph._edges_cache:
                return iter(self.hypergraph._edges_cache)
            else:
                return iter([])
        except Exception as e:
            raise RuntimeError(f"Failed to iterate over edges: {e}")
    
    def __len__(self) -> int:
        """
        Number of edges
        
        Returns
        -------
        int
            Total number of edges in the hypergraph
        """
        return self.hypergraph.num_edges
    
    def __contains__(self, edge_id: Any) -> bool:
        """
        Check if edge exists using optimized cache
        
        Parameters
        ----------
        edge_id : Any
            The edge identifier to check
            
        Returns
        -------
        bool
            True if edge exists, False otherwise
        """
        try:
            self.hypergraph._build_performance_indexes()
            if hasattr(self.hypergraph, '_edges_cache'):
                return edge_id in (self.hypergraph._edges_cache or set())
            else:
                return self.hypergraph.has_edge(edge_id)
        except Exception as e:
            raise RuntimeError(f"Failed to check edge existence: {e}")
    
    def get(self, edge_id: Any, default: Optional[List[Any]] = None) -> List[Any]:
        """
        Get nodes for an edge with default fallback
        
        Parameters
        ----------
        edge_id : Any
            The edge identifier
        default : List[Any], optional
            Default value to return if edge doesn't exist
            
        Returns
        -------
        List[Any]
            List of nodes connected by the edge, or default if edge doesn't exist
        """
        if default is None:
            default = []
            
        try:
            if self.hypergraph.has_edge(edge_id):
                return self.hypergraph.get_edge_nodes(edge_id)
            return default
        except Exception:
            return default
    
    def keys(self) -> Iterator[Any]:
        """
        Get iterator over edge identifiers
        
        Returns
        -------
        Iterator[Any]
            Iterator over edge identifiers
        """
        return iter(self)
    
    def values(self) -> Iterator[List[Any]]:
        """
        Get iterator over edge node lists
        
        Returns
        -------
        Iterator[List[Any]]
            Iterator over lists of nodes for each edge
        """
        for edge_id in self:
            yield self.hypergraph.get_edge_nodes(edge_id)
    
    def items(self) -> Iterator[tuple]:
        """
        Get iterator over (edge_id, nodes) pairs
        
        Returns
        -------
        Iterator[tuple]
            Iterator over (edge_id, node_list) tuples
        """
        for edge_id in self:
            yield edge_id, self.hypergraph.get_edge_nodes(edge_id)
    
    def __repr__(self) -> str:
        """String representation of EdgeView"""
        return f"EdgeView(hypergraph='{self.hypergraph.name}', num_edges={len(self)})"
    
    def __str__(self) -> str:
        """String representation of EdgeView"""
        return self.__repr__()