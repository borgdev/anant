"""
Set operations for Hypergraph

Handles mathematical set operations like union, intersection, difference,
and subgraph extraction for hypergraph structures.
"""

from typing import Dict, List, Set, Any, Optional, Union, Iterable
import polars as pl
from ....exceptions import HypergraphError, ValidationError


class SetOperations:
    """
    Set operations for hypergraph
    
    Provides mathematical set operations and subgraph extraction
    functionality for hypergraphs including union, intersection,
    difference, and various filtering operations.
    """
    
    def __init__(self, hypergraph):
        """
        Initialize SetOperations
        
        Parameters
        ----------
        hypergraph : Hypergraph
            Parent hypergraph instance
        """
        if hypergraph is None:
            raise HypergraphError("Hypergraph instance cannot be None")
        self.hypergraph = hypergraph
    
    def union(self, other: 'Hypergraph', node_merge_strategy: str = 'union', 
              edge_merge_strategy: str = 'union') -> 'Hypergraph':
        """
        Compute union of two hypergraphs
        
        Parameters
        ----------
        other : Hypergraph
            Other hypergraph to union with
        node_merge_strategy : str, default 'union'
            How to handle node conflicts: 'union', 'left', 'right'
        edge_merge_strategy : str, default 'union' 
            How to handle edge conflicts: 'union', 'left', 'right'
            
        Returns
        -------
        Hypergraph
            New hypergraph containing union of both inputs
            
        Raises
        ------
        ValidationError
            If merge strategies are invalid
        HypergraphError
            If union operation fails
        """
        try:
            from ..core.hypergraph import Hypergraph
            
            if not hasattr(other, 'nodes') or not hasattr(other, 'edges'):
                raise ValidationError("other must be a Hypergraph instance")
            
            valid_strategies = {'union', 'left', 'right'}
            if node_merge_strategy not in valid_strategies:
                raise ValidationError(f"Invalid node_merge_strategy: {node_merge_strategy}")
            if edge_merge_strategy not in valid_strategies:
                raise ValidationError(f"Invalid edge_merge_strategy: {edge_merge_strategy}")
            
            # Collect all nodes and edges
            if node_merge_strategy == 'union':
                all_nodes = set(self.hypergraph.nodes) | set(other.nodes)
            elif node_merge_strategy == 'left':
                all_nodes = set(self.hypergraph.nodes)
            else:  # right
                all_nodes = set(other.nodes)
            
            # Build new edge system
            new_edges = {}
            
            # Add edges from left hypergraph
            if edge_merge_strategy in {'union', 'left'}:
                for edge in self.hypergraph.edges:
                    edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                    # Only include nodes that are in the final node set
                    filtered_nodes = [n for n in edge_nodes if n in all_nodes]
                    if len(filtered_nodes) >= 1:  # Keep edges with at least 1 node
                        new_edges[edge] = filtered_nodes
            
            # Add edges from right hypergraph
            if edge_merge_strategy in {'union', 'right'}:
                for edge in other.edges:
                    edge_nodes = other.incidences.get_edge_nodes(edge)
                    # Only include nodes that are in the final node set
                    filtered_nodes = [n for n in edge_nodes if n in all_nodes]
                    if len(filtered_nodes) >= 1:
                        if edge in new_edges:
                            # Merge edge nodes for duplicate edges
                            new_edges[edge] = list(set(new_edges[edge]) | set(filtered_nodes))
                        else:
                            new_edges[edge] = filtered_nodes
            
            # Create union hypergraph
            result = Hypergraph(setsystem=new_edges, 
                              name=f"{self.hypergraph.name}_union_{other.name}")
            
            # Merge properties if both have property stores
            if hasattr(self.hypergraph, 'properties') and hasattr(other, 'properties'):
                self._merge_properties(result, other, node_merge_strategy, edge_merge_strategy)
            
            return result
            
        except Exception as e:
            raise HypergraphError(f"Union operation failed: {e}")
    
    def intersection(self, other: 'Hypergraph', mode: str = 'edges') -> 'Hypergraph':
        """
        Compute intersection of two hypergraphs
        
        Parameters
        ----------
        other : Hypergraph
            Other hypergraph to intersect with
        mode : str, default 'edges'
            Intersection mode:
            - 'edges': Intersection of edges (nodes that appear in both)
            - 'nodes': Only nodes that exist in both graphs
            - 'structure': Exact structural intersection
            
        Returns
        -------
        Hypergraph
            New hypergraph containing intersection
            
        Raises
        ------
        ValidationError
            If mode is invalid
        HypergraphError
            If intersection operation fails
        """
        try:
            from ..core.hypergraph import Hypergraph
            
            if not hasattr(other, 'nodes') or not hasattr(other, 'edges'):
                raise ValidationError("other must be a Hypergraph instance")
            
            valid_modes = {'edges', 'nodes', 'structure'}
            if mode not in valid_modes:
                raise ValidationError(f"Invalid mode: {mode}. Valid modes: {valid_modes}")
            
            if mode == 'nodes':
                # Intersection of nodes only
                common_nodes = set(self.hypergraph.nodes) & set(other.nodes)
                return self.subgraph_nodes(list(common_nodes))
            
            elif mode == 'structure':
                # Exact structural intersection - edges that exist in both with same nodes
                common_edges = {}
                
                for edge in self.hypergraph.edges:
                    if edge in other.edges:
                        left_nodes = set(self.hypergraph.incidences.get_edge_nodes(edge))
                        right_nodes = set(other.incidences.get_edge_nodes(edge))
                        
                        if left_nodes == right_nodes:
                            common_edges[edge] = list(left_nodes)
                
                return Hypergraph(setsystem=common_edges, 
                                name=f"{self.hypergraph.name}_intersect_{other.name}")
            
            else:  # edges mode
                # Intersection based on common edges (with potentially different nodes)
                common_edge_ids = set(self.hypergraph.edges) & set(other.edges)
                common_edges = {}
                
                for edge in common_edge_ids:
                    left_nodes = set(self.hypergraph.incidences.get_edge_nodes(edge))
                    right_nodes = set(other.incidences.get_edge_nodes(edge))
                    
                    # Use intersection of nodes for each edge
                    intersect_nodes = list(left_nodes & right_nodes)
                    if intersect_nodes:
                        common_edges[edge] = intersect_nodes
                
                return Hypergraph(setsystem=common_edges,
                                name=f"{self.hypergraph.name}_intersect_{other.name}")
            
        except Exception as e:
            raise HypergraphError(f"Intersection operation failed: {e}")
    
    def difference(self, other: 'Hypergraph', mode: str = 'edges') -> 'Hypergraph':
        """
        Compute difference of two hypergraphs (self - other)
        
        Parameters
        ----------
        other : Hypergraph
            Other hypergraph to subtract
        mode : str, default 'edges'
            Difference mode:
            - 'edges': Remove edges that exist in other
            - 'nodes': Remove nodes that exist in other
            - 'structure': Remove exact structural matches
            
        Returns
        -------
        Hypergraph
            New hypergraph containing difference
            
        Raises
        ------
        ValidationError
            If mode is invalid
        HypergraphError
            If difference operation fails
        """
        try:
            from ..core.hypergraph import Hypergraph
            
            if not hasattr(other, 'nodes') or not hasattr(other, 'edges'):
                raise ValidationError("other must be a Hypergraph instance")
            
            valid_modes = {'edges', 'nodes', 'structure'}
            if mode not in valid_modes:
                raise ValidationError(f"Invalid mode: {mode}. Valid modes: {valid_modes}")
            
            if mode == 'nodes':
                # Remove nodes that exist in other
                remaining_nodes = set(self.hypergraph.nodes) - set(other.nodes)
                return self.subgraph_nodes(list(remaining_nodes))
            
            elif mode == 'structure':
                # Remove edges with exact structural matches
                result_edges = {}
                
                for edge in self.hypergraph.edges:
                    if edge in other.edges:
                        left_nodes = set(self.hypergraph.incidences.get_edge_nodes(edge))
                        right_nodes = set(other.incidences.get_edge_nodes(edge))
                        
                        # Keep edge only if structure is different
                        if left_nodes != right_nodes:
                            result_edges[edge] = list(left_nodes)
                    else:
                        # Keep edges not in other
                        edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                        result_edges[edge] = edge_nodes
                
                return Hypergraph(setsystem=result_edges,
                                name=f"{self.hypergraph.name}_minus_{other.name}")
            
            else:  # edges mode
                # Remove edges by ID
                other_edge_ids = set(other.edges)
                result_edges = {}
                
                for edge in self.hypergraph.edges:
                    if edge not in other_edge_ids:
                        edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                        result_edges[edge] = edge_nodes
                
                return Hypergraph(setsystem=result_edges,
                                name=f"{self.hypergraph.name}_minus_{other.name}")
            
        except Exception as e:
            raise HypergraphError(f"Difference operation failed: {e}")
    
    def subgraph_nodes(self, nodes: Iterable[Any]) -> 'Hypergraph':
        """
        Extract subgraph containing only specified nodes
        
        Parameters
        ----------
        nodes : Iterable[Any]
            Nodes to include in subgraph
            
        Returns
        -------
        Hypergraph
            New hypergraph with only specified nodes and their edges
            
        Raises
        ------
        ValidationError
            If nodes parameter is invalid
        HypergraphError
            If subgraph extraction fails
        """
        try:
            from ..core.hypergraph import Hypergraph
            
            if nodes is None:
                raise ValidationError("nodes cannot be None")
            
            node_set = set(nodes)
            if not node_set:
                return Hypergraph(name=f"{self.hypergraph.name}_subgraph_empty")
            
            # Check that all nodes exist
            existing_nodes = set(self.hypergraph.nodes)
            invalid_nodes = node_set - existing_nodes
            if invalid_nodes:
                raise ValidationError(f"Nodes not found in hypergraph: {invalid_nodes}")
            
            # Extract edges that have at least one node in the subset
            subgraph_edges = {}
            for edge in self.hypergraph.edges:
                edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                
                # Include nodes that are in our target set
                filtered_nodes = [n for n in edge_nodes if n in node_set]
                
                # Keep edge if it has at least one node in our set
                if filtered_nodes:
                    subgraph_edges[edge] = filtered_nodes
            
            return Hypergraph(setsystem=subgraph_edges,
                            name=f"{self.hypergraph.name}_subgraph_{len(node_set)}nodes")
            
        except Exception as e:
            raise HypergraphError(f"Node subgraph extraction failed: {e}")
    
    def subgraph_edges(self, edges: Iterable[Any]) -> 'Hypergraph':
        """
        Extract subgraph containing only specified edges
        
        Parameters
        ----------
        edges : Iterable[Any]
            Edges to include in subgraph
            
        Returns
        -------
        Hypergraph
            New hypergraph with only specified edges
            
        Raises
        ------
        ValidationError
            If edges parameter is invalid
        HypergraphError
            If subgraph extraction fails
        """
        try:
            from ..core.hypergraph import Hypergraph
            
            if edges is None:
                raise ValidationError("edges cannot be None")
            
            edge_set = set(edges)
            if not edge_set:
                return Hypergraph(name=f"{self.hypergraph.name}_subgraph_empty")
            
            # Check that all edges exist
            existing_edges = set(self.hypergraph.edges)
            invalid_edges = edge_set - existing_edges
            if invalid_edges:
                raise ValidationError(f"Edges not found in hypergraph: {invalid_edges}")
            
            # Extract specified edges with their nodes
            subgraph_edges = {}
            for edge in edge_set:
                edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                subgraph_edges[edge] = edge_nodes
            
            return Hypergraph(setsystem=subgraph_edges,
                            name=f"{self.hypergraph.name}_subgraph_{len(edge_set)}edges")
            
        except Exception as e:
            raise HypergraphError(f"Edge subgraph extraction failed: {e}")
    
    def filter_nodes(self, predicate) -> 'Hypergraph':
        """
        Filter nodes based on a predicate function
        
        Parameters
        ----------
        predicate : callable
            Function that takes a node and returns True/False
            
        Returns
        -------
        Hypergraph
            New hypergraph with filtered nodes
            
        Raises
        ------
        ValidationError
            If predicate is not callable
        HypergraphError
            If filtering fails
        """
        try:
            if not callable(predicate):
                raise ValidationError("predicate must be callable")
            
            # Apply predicate to filter nodes
            filtered_nodes = [node for node in self.hypergraph.nodes if predicate(node)]
            
            return self.subgraph_nodes(filtered_nodes)
            
        except Exception as e:
            raise HypergraphError(f"Node filtering failed: {e}")
    
    def filter_edges(self, predicate) -> 'Hypergraph':
        """
        Filter edges based on a predicate function
        
        Parameters
        ----------
        predicate : callable
            Function that takes an edge and returns True/False
            
        Returns
        -------
        Hypergraph
            New hypergraph with filtered edges
            
        Raises
        ------
        ValidationError
            If predicate is not callable
        HypergraphError
            If filtering fails
        """
        try:
            if not callable(predicate):
                raise ValidationError("predicate must be callable")
            
            # Apply predicate to filter edges
            filtered_edges = [edge for edge in self.hypergraph.edges if predicate(edge)]
            
            return self.subgraph_edges(filtered_edges)
            
        except Exception as e:
            raise HypergraphError(f"Edge filtering failed: {e}")
    
    def filter_by_degree(self, min_degree: int = 1, max_degree: Optional[int] = None) -> 'Hypergraph':
        """
        Filter nodes by their degree (number of incident edges)
        
        Parameters
        ----------
        min_degree : int, default 1
            Minimum degree (inclusive)
        max_degree : Optional[int]
            Maximum degree (inclusive), None for no upper limit
            
        Returns
        -------
        Hypergraph
            New hypergraph with nodes filtered by degree
            
        Raises
        ------
        ValidationError
            If degree parameters are invalid
        HypergraphError
            If filtering fails
        """
        try:
            if min_degree < 0:
                raise ValidationError("min_degree must be non-negative")
            if max_degree is not None and max_degree < min_degree:
                raise ValidationError("max_degree must be >= min_degree")
            
            def degree_predicate(node):
                degree = len(self.hypergraph.incidences.get_node_edges(node))
                if max_degree is None:
                    return degree >= min_degree
                else:
                    return min_degree <= degree <= max_degree
            
            return self.filter_nodes(degree_predicate)
            
        except Exception as e:
            raise HypergraphError(f"Degree filtering failed: {e}")
    
    def filter_by_edge_size(self, min_size: int = 1, max_size: Optional[int] = None) -> 'Hypergraph':
        """
        Filter edges by their size (number of incident nodes)
        
        Parameters
        ----------
        min_size : int, default 1
            Minimum edge size (inclusive)
        max_size : Optional[int]
            Maximum edge size (inclusive), None for no upper limit
            
        Returns
        -------
        Hypergraph
            New hypergraph with edges filtered by size
            
        Raises
        ------
        ValidationError
            If size parameters are invalid
        HypergraphError
            If filtering fails
        """
        try:
            if min_size < 1:
                raise ValidationError("min_size must be positive")
            if max_size is not None and max_size < min_size:
                raise ValidationError("max_size must be >= min_size")
            
            def size_predicate(edge):
                size = len(self.hypergraph.incidences.get_edge_nodes(edge))
                if max_size is None:
                    return size >= min_size
                else:
                    return min_size <= size <= max_size
            
            return self.filter_edges(size_predicate)
            
        except Exception as e:
            raise HypergraphError(f"Edge size filtering failed: {e}")
    
    def largest_connected_component(self) -> 'Hypergraph':
        """
        Extract the largest connected component
        
        Returns
        -------
        Hypergraph
            New hypergraph containing only the largest connected component
            
        Raises
        ------
        HypergraphError
            If component extraction fails
        """
        try:
            # Find connected components using DFS
            visited_nodes = set()
            components = []
            
            for start_node in self.hypergraph.nodes:
                if start_node not in visited_nodes:
                    # DFS to find component
                    component_nodes = set()
                    stack = [start_node]
                    
                    while stack:
                        node = stack.pop()
                        if node not in visited_nodes:
                            visited_nodes.add(node)
                            component_nodes.add(node)
                            
                            # Add all neighbors through hyperedges
                            for edge in self.hypergraph.incidences.get_node_edges(node):
                                for neighbor in self.hypergraph.incidences.get_edge_nodes(edge):
                                    if neighbor not in visited_nodes:
                                        stack.append(neighbor)
                    
                    components.append(component_nodes)
            
            if not components:
                return Hypergraph(name=f"{self.hypergraph.name}_empty_component")
            
            # Find largest component
            largest_component = max(components, key=len)
            
            return self.subgraph_nodes(list(largest_component))
            
        except Exception as e:
            raise HypergraphError(f"Connected component extraction failed: {e}")
    
    def _merge_properties(self, result_hg, other_hg, node_strategy: str, edge_strategy: str):
        """
        Helper method to merge properties from two hypergraphs
        
        Parameters
        ----------
        result_hg : Hypergraph
            Target hypergraph for merged properties
        other_hg : Hypergraph
            Source hypergraph for properties
        node_strategy : str
            Strategy for merging node properties
        edge_strategy : str
            Strategy for merging edge properties
        """
        try:
            # Merge node properties
            if hasattr(self.hypergraph.properties, 'node_properties'):
                for node in result_hg.nodes:
                    if node in self.hypergraph.nodes:
                        left_props = self.hypergraph.properties.get_node_properties(node)
                        if left_props:
                            result_hg.properties.set_node_properties(node, left_props)
                    
                    if node_strategy == 'union' and node in other_hg.nodes:
                        right_props = other_hg.properties.get_node_properties(node)
                        if right_props:
                            # Merge with existing properties
                            existing = result_hg.properties.get_node_properties(node) or {}
                            existing.update(right_props)
                            result_hg.properties.set_node_properties(node, existing)
            
            # Merge edge properties
            if hasattr(self.hypergraph.properties, 'edge_properties'):
                for edge in result_hg.edges:
                    if edge in self.hypergraph.edges:
                        left_props = self.hypergraph.properties.get_edge_properties(edge)
                        if left_props:
                            result_hg.properties.set_edge_properties(edge, left_props)
                    
                    if edge_strategy == 'union' and edge in other_hg.edges:
                        right_props = other_hg.properties.get_edge_properties(edge)
                        if right_props:
                            # Merge with existing properties
                            existing = result_hg.properties.get_edge_properties(edge) or {}
                            existing.update(right_props)
                            result_hg.properties.set_edge_properties(edge, existing)
                            
        except Exception as e:
            # Property merging is non-critical, log warning but don't fail
            pass