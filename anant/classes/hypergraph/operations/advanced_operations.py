"""
Advanced operations for Hypergraph

Handles complex hypergraph transformations, dual graphs, line graphs,
advanced analysis, and specialized hypergraph theory operations.
"""

from typing import Dict, List, Set, Any, Optional, Union, Iterable, Tuple
import polars as pl
from collections import defaultdict, Counter
import itertools
import math
from ....exceptions import HypergraphError, ValidationError


class AdvancedOperations:
    """
    Advanced operations for hypergraph
    
    Provides complex hypergraph transformations and advanced analysis
    including dual graphs, line graphs, transformations, and specialized
    hypergraph theory operations.
    """
    
    def __init__(self, hypergraph):
        """
        Initialize AdvancedOperations
        
        Parameters
        ----------
        hypergraph : Hypergraph
            Parent hypergraph instance
        """
        if hypergraph is None:
            raise HypergraphError("Hypergraph instance cannot be None")
        self.hypergraph = hypergraph
    
    def dual(self) -> 'Hypergraph':
        """
        Compute dual hypergraph where nodes and edges are swapped
        
        In the dual hypergraph:
        - Original nodes become edges
        - Original edges become nodes
        - Incidence relationships are preserved
        
        Returns
        -------
        Hypergraph
            Dual hypergraph
            
        Raises
        ------
        HypergraphError
            If dual computation fails
        """
        try:
            from ..core.hypergraph import Hypergraph
            
            # Build dual edge system
            dual_edges = {}
            
            # Each original node becomes an edge in the dual
            for node in self.hypergraph.nodes:
                # Get all edges incident to this node
                incident_edges = self.hypergraph.incidences.get_node_edges(node)
                if incident_edges:
                    dual_edges[node] = list(incident_edges)
            
            return Hypergraph(setsystem=dual_edges, 
                            name=f"{self.hypergraph.name}_dual")
            
        except Exception as e:
            raise HypergraphError(f"Dual hypergraph computation failed: {e}")
    
    def line_graph(self) -> 'Hypergraph':
        """
        Compute line graph where edges become nodes
        
        In the line graph:
        - Original edges become nodes
        - Two nodes (original edges) are connected if they share a common node
        - Results in a traditional graph (all edges are binary)
        
        Returns
        -------
        Hypergraph
            Line graph representation
            
        Raises
        ------
        HypergraphError
            If line graph computation fails
        """
        try:
            from ..core.hypergraph import Hypergraph
            
            # Build adjacency between edges
            line_edges = {}
            edges_list = list(self.hypergraph.edges)
            
            # Find pairs of edges that share at least one node
            for i, edge1 in enumerate(edges_list):
                for j, edge2 in enumerate(edges_list[i+1:], i+1):
                    nodes1 = set(self.hypergraph.incidences.get_edge_nodes(edge1))
                    nodes2 = set(self.hypergraph.incidences.get_edge_nodes(edge2))
                    
                    # If edges share nodes, connect them in line graph
                    if nodes1 & nodes2:  # Intersection is non-empty
                        edge_id = f"L_{edge1}_{edge2}"
                        line_edges[edge_id] = [edge1, edge2]
            
            return Hypergraph(setsystem=line_edges,
                            name=f"{self.hypergraph.name}_line")
            
        except Exception as e:
            raise HypergraphError(f"Line graph computation failed: {e}")
    
    def adjacency_graph(self, threshold: int = 1) -> 'Hypergraph':
        """
        Create adjacency graph where nodes are connected if they share edges
        
        Parameters
        ----------
        threshold : int, default 1
            Minimum number of shared edges required for connection
            
        Returns
        -------
        Hypergraph
            Adjacency graph
            
        Raises
        ------
        ValidationError
            If threshold is invalid
        HypergraphError
            If adjacency graph computation fails
        """
        try:
            from ..core.hypergraph import Hypergraph
            
            if threshold < 1:
                raise ValidationError("threshold must be positive")
            
            # Count shared edges between all node pairs
            shared_count = defaultdict(int)
            
            for edge in self.hypergraph.edges:
                edge_nodes = list(self.hypergraph.incidences.get_edge_nodes(edge))
                
                # Count shared edges for all pairs of nodes in this edge
                for i, node1 in enumerate(edge_nodes):
                    for j, node2 in enumerate(edge_nodes[i+1:], i+1):
                        pair = tuple(sorted([node1, node2]))
                        shared_count[pair] += 1
            
            # Create edges for pairs that meet threshold
            adj_edges = {}
            edge_counter = 0
            
            for (node1, node2), count in shared_count.items():
                if count >= threshold:
                    edge_id = f"A_{node1}_{node2}"
                    adj_edges[edge_id] = [node1, node2]
                    edge_counter += 1
            
            return Hypergraph(setsystem=adj_edges,
                            name=f"{self.hypergraph.name}_adjacency_t{threshold}")
            
        except Exception as e:
            raise HypergraphError(f"Adjacency graph computation failed: {e}")
    
    def clique_graph(self, min_size: int = 2) -> 'Hypergraph':
        """
        Create clique graph from maximal cliques in the adjacency structure
        
        Parameters
        ----------
        min_size : int, default 2
            Minimum clique size to include
            
        Returns
        -------
        Hypergraph
            Clique graph where each clique becomes a node
            
        Raises
        ------
        ValidationError
            If min_size is invalid
        HypergraphError
            If clique computation fails
        """
        try:
            from ..core.hypergraph import Hypergraph
            
            if min_size < 2:
                raise ValidationError("min_size must be at least 2")
            
            # First get adjacency relationships
            adjacency = defaultdict(set)
            
            for edge in self.hypergraph.edges:
                edge_nodes = list(self.hypergraph.incidences.get_edge_nodes(edge))
                
                # All pairs in a hyperedge are adjacent
                for i, node1 in enumerate(edge_nodes):
                    for j, node2 in enumerate(edge_nodes[i+1:], i+1):
                        adjacency[node1].add(node2)
                        adjacency[node2].add(node1)
            
            # Find maximal cliques using Bron-Kerbosch algorithm (simplified)
            cliques = self._find_maximal_cliques(adjacency)
            
            # Filter cliques by minimum size
            large_cliques = [clique for clique in cliques if len(clique) >= min_size]
            
            if not large_cliques:
                return Hypergraph(name=f"{self.hypergraph.name}_cliques_empty")
            
            # Create clique graph - each clique becomes a node
            # Cliques are connected if they share nodes
            clique_edges = {}
            
            for i, clique1 in enumerate(large_cliques):
                for j, clique2 in enumerate(large_cliques[i+1:], i+1):
                    # If cliques share nodes, connect them
                    if set(clique1) & set(clique2):
                        edge_id = f"C_{i}_{j}"
                        clique_edges[edge_id] = [f"clique_{i}", f"clique_{j}"]
            
            return Hypergraph(setsystem=clique_edges,
                            name=f"{self.hypergraph.name}_cliques")
            
        except Exception as e:
            raise HypergraphError(f"Clique graph computation failed: {e}")
    
    def k_uniform_projection(self, k: int) -> 'Hypergraph':
        """
        Project to k-uniform hypergraph by filtering/splitting edges
        
        Parameters
        ----------
        k : int
            Target uniformity (all edges will have exactly k nodes)
            
        Returns
        -------
        Hypergraph
            k-uniform hypergraph
            
        Raises
        ------
        ValidationError
            If k is invalid
        HypergraphError
            If projection fails
        """
        try:
            from ..core.hypergraph import Hypergraph
            
            if k < 1:
                raise ValidationError("k must be positive")
            
            uniform_edges = {}
            edge_counter = 0
            
            for edge in self.hypergraph.edges:
                edge_nodes = list(self.hypergraph.incidences.get_edge_nodes(edge))
                
                if len(edge_nodes) == k:
                    # Edge already has correct size
                    uniform_edges[edge] = edge_nodes
                elif len(edge_nodes) > k:
                    # Split large edge into k-sized subsets
                    for i, subset in enumerate(itertools.combinations(edge_nodes, k)):
                        new_edge_id = f"{edge}_sub_{i}"
                        uniform_edges[new_edge_id] = list(subset)
                        edge_counter += 1
                # Ignore edges smaller than k
            
            return Hypergraph(setsystem=uniform_edges,
                            name=f"{self.hypergraph.name}_{k}_uniform")
            
        except Exception as e:
            raise HypergraphError(f"k-uniform projection failed: {e}")
    
    def bipartite_projection(self, node_type_attribute: Optional[str] = None) -> Tuple['Hypergraph', 'Hypergraph']:
        """
        Create bipartite projections onto two node types
        
        Parameters
        ----------
        node_type_attribute : str, optional
            Attribute name to determine node types. If None, use simple heuristic
            
        Returns
        -------
        Tuple[Hypergraph, Hypergraph]
            Two projected graphs for each node type
            
        Raises
        ------
        HypergraphError
            If bipartite projection fails
        """
        try:
            from ..core.hypergraph import Hypergraph
            
            # Partition nodes into two types
            if node_type_attribute and hasattr(self.hypergraph, 'properties'):
                # Use attribute to partition nodes
                type_a_nodes = []
                type_b_nodes = []
                
                for node in self.hypergraph.nodes:
                    node_props = self.hypergraph.properties.get_node_properties(node)
                    if node_props and node_props.get(node_type_attribute) == 'A':
                        type_a_nodes.append(node)
                    else:
                        type_b_nodes.append(node)
            else:
                # Simple heuristic: partition by node degree
                node_degrees = {}
                for node in self.hypergraph.nodes:
                    node_degrees[node] = len(self.hypergraph.incidences.get_node_edges(node))
                
                sorted_nodes = sorted(self.hypergraph.nodes, key=lambda x: node_degrees[x])
                mid = len(sorted_nodes) // 2
                type_a_nodes = sorted_nodes[:mid]
                type_b_nodes = sorted_nodes[mid:]
            
            # Create projections
            proj_a_edges = {}
            proj_b_edges = {}
            
            # Project type A nodes (connected if they share edges)
            a_shared = defaultdict(set)
            b_shared = defaultdict(set)
            
            for edge in self.hypergraph.edges:
                edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                
                # Nodes of type A in this edge
                a_in_edge = [n for n in edge_nodes if n in type_a_nodes]
                b_in_edge = [n for n in edge_nodes if n in type_b_nodes]
                
                # Connect all pairs within type A
                for i, node1 in enumerate(a_in_edge):
                    for j, node2 in enumerate(a_in_edge[i+1:], i+1):
                        pair = tuple(sorted([node1, node2]))
                        a_shared[pair].add(edge)
                
                # Connect all pairs within type B
                for i, node1 in enumerate(b_in_edge):
                    for j, node2 in enumerate(b_in_edge[i+1:], i+1):
                        pair = tuple(sorted([node1, node2]))
                        b_shared[pair].add(edge)
            
            # Create projection edges
            for i, (pair, shared_edges) in enumerate(a_shared.items()):
                proj_a_edges[f"PA_{i}"] = list(pair)
            
            for i, (pair, shared_edges) in enumerate(b_shared.items()):
                proj_b_edges[f"PB_{i}"] = list(pair)
            
            proj_a = Hypergraph(setsystem=proj_a_edges,
                              name=f"{self.hypergraph.name}_proj_A")
            proj_b = Hypergraph(setsystem=proj_b_edges,
                              name=f"{self.hypergraph.name}_proj_B")
            
            return proj_a, proj_b
            
        except Exception as e:
            raise HypergraphError(f"Bipartite projection failed: {e}")
    
    def complement(self) -> 'Hypergraph':
        """
        Compute complement hypergraph
        
        The complement contains all possible hyperedges that are NOT in the original.
        Warning: This can be very large for large node sets.
        
        Returns
        -------
        Hypergraph
            Complement hypergraph
            
        Raises
        ------
        HypergraphError
            If complement computation fails or result would be too large
        """
        try:
            from ..core.hypergraph import Hypergraph
            
            nodes = list(self.hypergraph.nodes)
            n = len(nodes)
            
            if n > 10:
                raise HypergraphError(f"Complement too large for {n} nodes. "
                                    f"Would generate 2^{n} - 1 possible edges.")
            
            # Get existing edges as sets for fast lookup
            existing_edges = set()
            for edge in self.hypergraph.edges:
                edge_nodes = tuple(sorted(self.hypergraph.incidences.get_edge_nodes(edge)))
                existing_edges.add(edge_nodes)
            
            # Generate all possible non-empty subsets
            complement_edges = {}
            edge_counter = 0
            
            for r in range(1, n + 1):  # All subset sizes from 1 to n
                for subset in itertools.combinations(nodes, r):
                    if subset not in existing_edges:
                        edge_id = f"COMP_{edge_counter}"
                        complement_edges[edge_id] = list(subset)
                        edge_counter += 1
            
            return Hypergraph(setsystem=complement_edges,
                            name=f"{self.hypergraph.name}_complement")
            
        except Exception as e:
            raise HypergraphError(f"Complement computation failed: {e}")
    
    def tensor_product(self, other: 'Hypergraph') -> 'Hypergraph':
        """
        Compute tensor product with another hypergraph
        
        Parameters
        ----------
        other : Hypergraph
            Other hypergraph for tensor product
            
        Returns
        -------
        Hypergraph
            Tensor product hypergraph
            
        Raises
        ------
        ValidationError
            If other is not a valid hypergraph
        HypergraphError
            If tensor product computation fails
        """
        try:
            from ..core.hypergraph import Hypergraph
            
            if not hasattr(other, 'nodes') or not hasattr(other, 'edges'):
                raise ValidationError("other must be a Hypergraph instance")
            
            # Create product nodes (Cartesian product of node sets)
            product_edges = {}
            edge_counter = 0
            
            # For each edge pair, create product edges
            for edge1 in self.hypergraph.edges:
                nodes1 = self.hypergraph.incidences.get_edge_nodes(edge1)
                
                for edge2 in other.edges:
                    nodes2 = other.incidences.get_edge_nodes(edge2)
                    
                    # Create Cartesian product of nodes
                    product_nodes = []
                    for n1 in nodes1:
                        for n2 in nodes2:
                            product_nodes.append(f"({n1},{n2})")
                    
                    if product_nodes:
                        edge_id = f"T_{edge1}_{edge2}"
                        product_edges[edge_id] = product_nodes
                        edge_counter += 1
            
            return Hypergraph(setsystem=product_edges,
                            name=f"{self.hypergraph.name}_tensor_{other.name}")
            
        except Exception as e:
            raise HypergraphError(f"Tensor product computation failed: {e}")
    
    def contractible_analysis(self) -> Dict[str, Any]:
        """
        Analyze contractible structures in the hypergraph
        
        Returns
        -------
        Dict[str, Any]
            Analysis results including contractible edges and components
            
        Raises
        ------
        HypergraphError
            If analysis fails
        """
        try:
            analysis = {
                'total_edges': len(self.hypergraph.edges),
                'total_nodes': len(self.hypergraph.nodes),
                'contractible_edges': [],
                'pendant_edges': [],
                'singleton_edges': [],
                'uniform_sizes': Counter(),
                'max_edge_size': 0,
                'min_edge_size': float('inf')
            }
            
            for edge in self.hypergraph.edges:
                edge_nodes = list(self.hypergraph.incidences.get_edge_nodes(edge))
                edge_size = len(edge_nodes)
                
                analysis['uniform_sizes'][edge_size] += 1
                analysis['max_edge_size'] = max(analysis['max_edge_size'], edge_size)
                analysis['min_edge_size'] = min(analysis['min_edge_size'], edge_size)
                
                if edge_size == 1:
                    analysis['singleton_edges'].append(edge)
                elif edge_size == 2:
                    # Check if this is a pendant edge (one node has degree 1)
                    node1, node2 = edge_nodes
                    deg1 = len(self.hypergraph.incidences.get_node_edges(node1))
                    deg2 = len(self.hypergraph.incidences.get_node_edges(node2))
                    
                    if deg1 == 1 or deg2 == 1:
                        analysis['pendant_edges'].append(edge)
                
                # Check if edge is contractible (removing it doesn't disconnect graph)
                # Simplified heuristic: edge is contractible if all its nodes have degree > 1
                all_high_degree = all(
                    len(self.hypergraph.incidences.get_node_edges(node)) > 1 
                    for node in edge_nodes
                )
                if all_high_degree and edge_size >= 2:
                    analysis['contractible_edges'].append(edge)
            
            if analysis['min_edge_size'] == float('inf'):
                analysis['min_edge_size'] = 0
            
            return analysis
            
        except Exception as e:
            raise HypergraphError(f"Contractible analysis failed: {e}")
    
    def _find_maximal_cliques(self, adjacency: Dict[Any, Set[Any]]) -> List[List[Any]]:
        """
        Find maximal cliques using simplified Bron-Kerbosch algorithm
        
        Parameters
        ----------
        adjacency : Dict[Any, Set[Any]]
            Adjacency list representation
            
        Returns
        -------
        List[List[Any]]
            List of maximal cliques
        """
        cliques = []
        
        def bron_kerbosch(r, p, x):
            if not p and not x:
                # Found maximal clique
                cliques.append(list(r))
                return
            
            # Choose pivot with most connections
            pivot = max(p | x, key=lambda v: len(adjacency[v] & p), default=None)
            if pivot is None:
                return
            
            # Iterate over vertices not connected to pivot
            for v in p - adjacency[pivot]:
                bron_kerbosch(
                    r | {v},
                    p & adjacency[v],
                    x & adjacency[v]
                )
                p.remove(v)
                x.add(v)
        
        all_nodes = set(adjacency.keys())
        bron_kerbosch(set(), all_nodes, set())
        
        return cliques
    
    def homomorphism_check(self, target: 'Hypergraph') -> bool:
        """
        Check if there exists a homomorphism to target hypergraph
        
        A hypergraph homomorphism is a function that maps nodes to nodes
        and preserves the hyperedge structure.
        
        Parameters
        ----------
        target : Hypergraph
            Target hypergraph
            
        Returns
        -------
        bool
            True if homomorphism exists
            
        Raises
        ------
        ValidationError
            If target is invalid
        HypergraphError
            If homomorphism check fails
        """
        try:
            if not hasattr(target, 'nodes') or not hasattr(target, 'edges'):
                raise ValidationError("target must be a Hypergraph instance")
            
            # Simple necessary conditions
            if len(self.hypergraph.nodes) > len(target.nodes):
                return False
            
            if len(self.hypergraph.edges) > len(target.edges):
                return False
            
            # Check edge size compatibility
            source_sizes = [len(self.hypergraph.incidences.get_edge_nodes(e)) 
                          for e in self.hypergraph.edges]
            target_sizes = [len(target.incidences.get_edge_nodes(e)) 
                          for e in target.edges]
            
            source_size_count = Counter(source_sizes)
            target_size_count = Counter(target_sizes)
            
            # Target must have at least as many edges of each size
            for size, count in source_size_count.items():
                if target_size_count[size] < count:
                    return False
            
            # For small graphs, try to find actual mapping (brute force)
            if len(self.hypergraph.nodes) <= 6 and len(target.nodes) <= 6:
                return self._try_homomorphism_mapping(target)
            
            # For larger graphs, return True if necessary conditions pass
            return True
            
        except Exception as e:
            raise HypergraphError(f"Homomorphism check failed: {e}")
    
    def _try_homomorphism_mapping(self, target: 'Hypergraph') -> bool:
        """
        Try to find actual homomorphism mapping (for small graphs)
        
        Parameters
        ----------
        target : Hypergraph
            Target hypergraph
            
        Returns
        -------
        bool
            True if mapping found
        """
        try:
            from itertools import permutations
            
            source_nodes = list(self.hypergraph.nodes)
            target_nodes = list(target.nodes)
            
            # Try all possible mappings
            for mapping_tuple in itertools.permutations(target_nodes, len(source_nodes)):
                mapping = dict(zip(source_nodes, mapping_tuple))
                
                # Check if this mapping preserves all hyperedges
                valid = True
                for edge in self.hypergraph.edges:
                    edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
                    mapped_nodes = [mapping[node] for node in edge_nodes]
                    
                    # Check if mapped nodes form an edge in target
                    found_edge = False
                    for target_edge in target.edges:
                        target_edge_nodes = target.incidences.get_edge_nodes(target_edge)
                        if set(mapped_nodes).issubset(set(target_edge_nodes)):
                            found_edge = True
                            break
                    
                    if not found_edge:
                        valid = False
                        break
                
                if valid:
                    return True
            
            return False
            
        except Exception:
            return False  # If mapping fails, return False