"""
Core Hypergraph Operations
=========================

Basic hypergraph structure operations including:
- Node and edge management (add, remove, query)
- Basic graph properties (degrees, sizes, membership)
- Property management
- Basic graph operations (subgraphs, copying)
"""

from typing import Dict, List, Optional, Any, Union, Iterable, Set
import polars as pl
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CoreOperations:
    """
    Handles core hypergraph structure operations.
    
    This class provides methods for basic node/edge management,
    property handling, and fundamental graph operations.
    """
    
    def __init__(self, hypergraph):
        """
        Initialize core operations.
        
        Args:
            hypergraph: Reference to parent Hypergraph instance
        """
        self.hg = hypergraph
        self.logger = logger.getChild(self.__class__.__name__)
        
    def nodes(self) -> set:
        """Get all nodes in the hypergraph."""
        if self.hg._nodes_cache is None:
            if self.hg.incidences.data.is_empty():
                self.hg._nodes_cache = set()
            else:
                self.hg._nodes_cache = set(self.hg.incidences.data.select('node_id').unique().to_series())
        return self.hg._nodes_cache
    
    def edges(self) -> 'EdgeView':
        """Get edge view for the hypergraph."""
        return EdgeView(self.hg)
    
    def num_nodes(self) -> int:
        """Get number of nodes."""
        return len(self.nodes())
    
    def num_edges(self) -> int:
        """Get number of edges."""
        if self.hg._edges_cache is None:
            if self.hg.incidences.data.is_empty():
                return 0
            else:
                return self.hg.incidences.data.select('edge_id').unique().height
        return len(self.hg._edges_cache)
    
    def num_incidences(self) -> int:
        """Get total number of incidences (node-edge relationships)."""
        return self.hg.incidences.data.height
    
    def get_node_degree(self, node_id: Any) -> int:
        """
        Get degree of a node (number of edges it participates in).
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node degree
        """
        if not self.hg._indexes_built:
            self.hg._build_performance_indexes()
            
        if node_id in self.hg._node_degrees:
            self.hg._performance_stats['cache_hits'] += 1
            return self.hg._node_degrees[node_id]
        
        self.hg._performance_stats['cache_misses'] += 1
        
        if self.hg.incidences.data.is_empty():
            degree = 0
        else:
            degree = self.hg.incidences.data.filter(
                pl.col('node_id') == node_id
            ).select('edge_id').unique().height
        
        self.hg._node_degrees[node_id] = degree
        return degree
    
    def get_edge_size(self, edge_id: Any) -> int:
        """
        Get size of an edge (number of nodes in the edge).
        
        Args:
            edge_id: Edge identifier
            
        Returns:
            Edge size
        """
        if not self.hg._indexes_built:
            self.hg._build_performance_indexes()
            
        if edge_id in self.hg._edge_sizes:
            self.hg._performance_stats['cache_hits'] += 1
            return self.hg._edge_sizes[edge_id]
        
        self.hg._performance_stats['cache_misses'] += 1
        
        if self.hg.incidences.data.is_empty():
            size = 0
        else:
            size = self.hg.incidences.data.filter(
                pl.col('edge_id') == edge_id
            ).select('node_id').unique().height
        
        self.hg._edge_sizes[edge_id] = size
        return size
    
    def get_node_edges(self, node_id: Any) -> List[Any]:
        """
        Get all edges that contain a given node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            List of edge identifiers
        """
        if not self.hg._indexes_built:
            self.hg._build_performance_indexes()
            
        if node_id in self.hg._node_to_edges:
            self.hg._performance_stats['cache_hits'] += 1
            return list(self.hg._node_to_edges[node_id])
        
        self.hg._performance_stats['cache_misses'] += 1
        
        if self.hg.incidences.data.is_empty():
            return []
        
        edges = self.hg.incidences.data.filter(
            pl.col('node_id') == node_id
        ).select('edge_id').unique().to_series().to_list()
        
        self.hg._node_to_edges[node_id] = set(edges)
        return edges
    
    def get_edge_nodes(self, edge_id: Any) -> List[Any]:
        """
        Get all nodes in a given edge.
        
        Args:
            edge_id: Edge identifier
            
        Returns:
            List of node identifiers
        """
        if not self.hg._indexes_built:
            self.hg._build_performance_indexes()
            
        if edge_id in self.hg._edge_to_nodes:
            self.hg._performance_stats['cache_hits'] += 1
            return list(self.hg._edge_to_nodes[edge_id])
        
        self.hg._performance_stats['cache_misses'] += 1
        
        if self.hg.incidences.data.is_empty():
            return []
        
        nodes = self.hg.incidences.data.filter(
            pl.col('edge_id') == edge_id
        ).select('node_id').to_series().to_list()
        
        self.hg._edge_to_nodes[edge_id] = set(nodes)
        return nodes
    
    def add_node(self, node_id: Any, properties: Optional[Dict] = None):
        """
        Add a node to the hypergraph.
        
        Args:
            node_id: Node identifier
            properties: Optional node properties
        """
        if properties:
            self.hg.properties.set_node_properties(node_id, properties)
        
        # Node will be implicitly added when edges are created
        # Mark as dirty for cache invalidation
        self.hg._dirty = True
        self.hg._invalidate_cache()
    
    def add_nodes_from(self, nodes: Iterable[Any]):
        """
        Add multiple nodes to the hypergraph.
        
        Args:
            nodes: Iterable of node identifiers
        """
        for node in nodes:
            self.add_node(node)
    
    def add_edge(self, edge_id: Any, node_list: List[Any], weight: float = 1.0, properties: Optional[Dict] = None):
        """
        Add an edge to the hypergraph.
        
        Args:
            edge_id: Edge identifier  
            node_list: List of nodes in the edge
            weight: Edge weight (default 1.0)
            properties: Optional edge properties
        """
        if not node_list:
            raise ValueError("Edge must contain at least one node")
        
        # Create incidence records
        new_rows = []
        for node_id in node_list:
            new_rows.append({
                'edge_id': edge_id,
                'node_id': node_id,
                'weight': weight
            })
        
        new_data = pl.DataFrame(new_rows)
        
        if self.hg.incidences.data.is_empty():
            self.hg.incidences.data = new_data
        else:
            self.hg.incidences.data = pl.concat([self.hg.incidences.data, new_data])
        
        # Add properties if provided
        if properties:
            self.hg.properties.set_edge_properties(edge_id, properties)
        
        # Invalidate caches
        self.hg._dirty = True
        self.hg._invalidate_cache()
        
        self.logger.debug(f"Added edge '{edge_id}' with {len(node_list)} nodes")
    
    def remove_node(self, node_id: Any):
        """
        Remove a node and all its incident edges.
        
        Args:
            node_id: Node identifier to remove
        """
        if self.hg.incidences.data.is_empty():
            return
        
        # Remove all incidences involving this node
        self.hg.incidences.data = self.hg.incidences.data.filter(
            pl.col('node_id') != node_id
        )
        
        # Remove node properties
        self.hg.properties.remove_node_properties(node_id)
        
        # Invalidate caches
        self.hg._dirty = True
        self.hg._invalidate_cache()
        
        self.logger.debug(f"Removed node '{node_id}'")
    
    def remove_edge(self, edge_id: Any):
        """
        Remove an edge from the hypergraph.
        
        Args:
            edge_id: Edge identifier to remove
        """
        if self.hg.incidences.data.is_empty():
            return
        
        # Remove all incidences for this edge
        self.hg.incidences.data = self.hg.incidences.data.filter(
            pl.col('edge_id') != edge_id
        )
        
        # Remove edge properties
        self.hg.properties.remove_edge_properties(edge_id)
        
        # Invalidate caches
        self.hg._dirty = True
        self.hg._invalidate_cache()
        
        self.logger.debug(f"Removed edge '{edge_id}'")
    
    def has_node(self, node_id: Any) -> bool:
        """
        Check if a node exists in the hypergraph.
        
        Args:
            node_id: Node identifier to check
            
        Returns:
            True if node exists
        """
        if self.hg._nodes_cache is not None:
            return node_id in self.hg._nodes_cache
        
        if self.hg.incidences.data.is_empty():
            return False
        
        return self.hg.incidences.data.filter(
            pl.col('node_id') == node_id
        ).height > 0
    
    def has_edge(self, edge_id: Any) -> bool:
        """
        Check if an edge exists in the hypergraph.
        
        Args:
            edge_id: Edge identifier to check
            
        Returns:
            True if edge exists
        """
        if self.hg._edges_cache is not None:
            return edge_id in self.hg._edges_cache
        
        if self.hg.incidences.data.is_empty():
            return False
        
        return self.hg.incidences.data.filter(
            pl.col('edge_id') == edge_id
        ).height > 0
    
    def clear(self):
        """Clear all nodes and edges from the hypergraph."""
        self.hg.incidences.data = pl.DataFrame({
            'edge_id': pl.Series([], dtype=pl.Utf8),
            'node_id': pl.Series([], dtype=pl.Utf8),
            'weight': pl.Series([], dtype=pl.Float64)
        })
        
        self.hg.properties.clear()
        self.hg._invalidate_cache()
        
        self.logger.info("Cleared hypergraph")
    
    def copy(self):
        """
        Create a deep copy of the hypergraph.
        
        Returns:
            New Hypergraph instance with copied data
        """
        # Import here to avoid circular imports
        from ..hypergraph_refactored import HypergraphRefactored as Hypergraph
        
        new_hg = Hypergraph(
            setsystem=self.hg.incidences.copy(),
            properties=self.hg.properties.copy(),
            name=f"{self.hg.name}_copy"
        )
        
        # Copy metadata
        new_hg.metadata = self.hg.metadata.copy()
        
        return new_hg
    
    def subhypergraph(self, nodes: Optional[List[Any]] = None, edges: Optional[List[Any]] = None):
        """
        Create a subhypergraph with specified nodes and/or edges.
        
        Args:
            nodes: List of nodes to include (None = all nodes)
            edges: List of edges to include (None = all edges)
            
        Returns:
            New Hypergraph instance with subset of data
        """
        # Import here to avoid circular imports
        from ..hypergraph_refactored import HypergraphRefactored as Hypergraph
        
        if self.hg.incidences.data.is_empty():
            return Hypergraph()
        
        # Filter data based on nodes and edges
        filtered_data = self.hg.incidences.data
        
        if nodes is not None:
            filtered_data = filtered_data.filter(
                pl.col('node_id').is_in(nodes)
            )
        
        if edges is not None:
            filtered_data = filtered_data.filter(
                pl.col('edge_id').is_in(edges)
            )
        
        # Create new hypergraph
        from ..incidence_store import IncidenceStore
        new_store = IncidenceStore(filtered_data)
        
        # Filter properties
        filtered_props = {}
        if nodes is not None or edges is not None:
            # This is a simplified property filtering
            # In practice, you'd want more sophisticated filtering
            pass
        
        return Hypergraph(
            setsystem=new_store,
            properties=filtered_props,
            name=f"{self.hg.name}_subgraph"
        )


class EdgeView:
    """
    View class for hypergraph edges.
    Provides iteration and access to edge information.
    """
    
    def __init__(self, hypergraph):
        self.hg = hypergraph
    
    def __iter__(self):
        """Iterate over edge identifiers."""
        if self.hg.incidences.data.is_empty():
            return iter([])
        
        edges = self.hg.incidences.data.select('edge_id').unique().to_series()
        return iter(edges)
    
    def __len__(self):
        """Get number of edges."""
        if self.hg.incidences.data.is_empty():
            return 0
        return self.hg.incidences.data.select('edge_id').unique().height
    
    def __contains__(self, edge_id):
        """Check if edge exists."""
        return self.hg.core_ops.has_edge(edge_id)