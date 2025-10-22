"""
Core Hypergraph implementation for Anant Library

Provides the main Hypergraph class with enhanced functionality through 
modular operations delegation pattern.
"""

from typing import Dict, List, Optional, Any, Union, Set, Iterable, Tuple
import polars as pl
import uuid
from pathlib import Path

# Import supporting classes
from ....utils.decorators import performance_monitor
from ...incidence_store import IncidenceStore
from ...property_store import PropertyStore
from ....exceptions import HypergraphError, ValidationError

# Import operation modules
from ..operations.core_operations import CoreOperations
from ..operations.performance_operations import PerformanceOperations
from ..operations.io_operations import IOOperations
from ..operations.algorithm_operations import AlgorithmOperations
from ..operations.visualization_operations import VisualizationOperations
from ..operations.set_operations import SetOperations
from ..operations.advanced_operations import AdvancedOperations

# Import utility classes
from .property_wrapper import PropertyWrapper
from ..views.edge_view import EdgeView


class Hypergraph:
    """
    Main Hypergraph class for the Anant library
    
    A hypergraph is a generalization of a graph where edges (hyperedges) can connect
    any number of vertices, not just two. This implementation provides:
    
    - High-performance operations using Polars DataFrames  
    - Advanced property management for nodes and edges
    - Flexible data import/export capabilities
    - Comprehensive validation and debugging support
    - Modular architecture with specialized operation modules
    
    Parameters
    ----------
    setsystem : dict, IncidenceStore, or None
        The underlying set system defining node-edge relationships
    data : pl.DataFrame, optional
        Raw incidence data as a Polars DataFrame
    properties : dict, optional
        Initial properties for nodes and edges
    name : str, optional
        Name identifier for the hypergraph
        
    Examples
    --------
    >>> from anant.classes.hypergraph import Hypergraph
    >>> import polars as pl
    
    Create from dict:
    >>> setsystem = {'e1': ['n1', 'n2'], 'e2': ['n1', 'n3']}
    >>> hg = Hypergraph(setsystem)
    
    Create from DataFrame:
    >>> data = pl.DataFrame({
    ...     'edge_id': ['e1', 'e1', 'e2'],
    ...     'node_id': ['n1', 'n2', 'n1'], 
    ...     'weight': [1.0, 0.8, 1.0]
    ... })
    >>> hg = Hypergraph.from_dataframe(data, 'node_id', 'edge_id')
    """
    
    def __init__(
        self, 
        setsystem: Optional[Union[dict, IncidenceStore]] = None,
        data: Optional[pl.DataFrame] = None,
        properties: Optional[dict] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Hypergraph instance
        
        Parameters
        ----------
        setsystem : dict, IncidenceStore, or None
            The underlying set system defining node-edge relationships
        data : pl.DataFrame, optional
            Raw incidence data as a Polars DataFrame
        properties : dict, optional
            Initial properties for nodes and edges
        name : str, optional
            Name identifier for the hypergraph
        **kwargs
            Additional metadata and configuration options
        """
        try:
            # Basic attributes
            self.name = name or "Hypergraph"
            self._instance_id = str(uuid.uuid4())
            
            # Initialize core data structures
            self._initialize_data_structures(setsystem, data, properties)
            
            # Initialize operation modules
            self._initialize_operations()
            
            # Initialize performance optimization structures
            self._initialize_performance_structures()
            
            # Store metadata
            self.metadata = kwargs.get('metadata', {})
            
            # Build initial indexes if data exists
            if not self.incidences.data.is_empty():
                self._build_performance_indexes()
                
        except Exception as e:
            raise HypergraphError(f"Failed to initialize hypergraph: {e}")
    
    def _initialize_data_structures(self, setsystem, data, properties):
        """Initialize core data structures"""
        try:
            # Initialize incidence store
            if isinstance(setsystem, IncidenceStore):
                self.incidences = setsystem
            elif setsystem is not None:
                # Convert dict setsystem to IncidenceStore
                self.incidences = IncidenceStore.from_dict(setsystem)
            elif data is not None:
                # Create from DataFrame
                self.incidences = IncidenceStore(data)
            else:
                # Empty hypergraph
                self.incidences = IncidenceStore()
            
            # Initialize property store
            self.properties = PropertyStore(properties or {})
            
            # Instance-specific cache for computed properties
            self._cache = {}
            
        except Exception as e:
            raise HypergraphError(f"Failed to initialize data structures: {e}")
    
    def _initialize_operations(self):
        """Initialize operation modules"""
        try:
            # Core operations (basic graph structure)
            self._core_ops = CoreOperations(self)
            
            # Performance operations (indexing, caching, batch operations)
            self._performance_ops = PerformanceOperations(self)
            
            # I/O operations (save/load, format conversion)
            self._io_ops = IOOperations(self)
            
            # Algorithm operations (centrality, PageRank, graph algorithms)
            self._algorithm_ops = AlgorithmOperations(self)
            
            # Visualization operations (layout, drawing, coordinates)
            self._visualization_ops = VisualizationOperations(self)
            
            # Set operations (union, intersection, difference, subgraphs)
            self._set_ops = SetOperations(self)
            
            # Advanced operations (dual graphs, transformations, analysis)
            self._advanced_ops = AdvancedOperations(self)
            
        except Exception as e:
            raise HypergraphError(f"Failed to initialize operation modules: {e}")
    
    def _initialize_performance_structures(self):
        """Initialize performance optimization structures"""
        try:
            # Performance tracking
            self._indexes_built = False
            self._dirty = True
            
            # Adjacency lists for O(1) lookups
            from collections import defaultdict
            self._node_to_edges: Dict[Any, set] = defaultdict(set)
            self._edge_to_nodes: Dict[Any, set] = defaultdict(set)
            
            # Degree cache for instant degree queries
            self._node_degrees: Dict[Any, int] = {}
            self._edge_sizes: Dict[Any, int] = {}
            
            # Node and edge sets for fast membership testing
            self._nodes_cache: Optional[set] = None
            self._edges_cache: Optional[set] = None
            
            # Performance counters
            self._performance_stats = {
                'cache_hits': 0,
                'cache_misses': 0,
                'index_builds': 0,
                'batch_operations': 0
            }
            
        except Exception as e:
            raise HypergraphError(f"Failed to initialize performance structures: {e}")
    
    # ================================================================
    # CORE GRAPH OPERATIONS (delegated to CoreOperations)
    # ================================================================
    
    def has_node(self, node_id: Any) -> bool:
        """Check if a node exists in the hypergraph"""
        return self._core_ops.has_node(node_id)
    
    def has_edge(self, edge_id: Any) -> bool:
        """Check if an edge exists in the hypergraph"""
        return self._core_ops.has_edge(edge_id)
    
    def add_node(self, node_id: Any, properties: Optional[Dict] = None) -> None:
        """Add a node to the hypergraph"""
        self._core_ops.add_node(node_id, properties)
    
    def add_entities_from(self, entities: Iterable[Any]) -> None:
        """Add multiple entities to the hypergraph"""
        self._core_ops.add_nodes_from(entities)
    
    # Backward compatibility alias
    def add_nodes_from(self, nodes: Iterable[Any]) -> None:
        """Add multiple nodes to the hypergraph"""
        for node in nodes:
            self.add_node(node)
    
    def add_edge(self, edge_id: Any, node_list: List[Any], weight: float = 1.0, 
                 properties: Optional[Dict] = None) -> None:
        """Add an edge to the hypergraph"""
        self._core_ops.add_edge(edge_id, node_list, weight, properties)
    
    def remove_node(self, node_id: Any) -> None:
        """Remove a node and all its incident edges"""
        self._core_ops.remove_node(node_id)
    
    def remove_edge(self, edge_id: Any) -> None:
        """Remove an edge from the hypergraph"""
        self._core_ops.remove_edge(edge_id)
    
    def neighbors(self, node_id: Any) -> Set[Any]:
        """Get nodes that share edges with the given node"""
        neighbors = set()
        # Get all edges incident to this node
        node_edges = self.incidences.get_node_edges(node_id)
        
        # For each edge, get all nodes and add them as neighbors (excluding the node itself)
        for edge_id in node_edges:
            edge_nodes = self.incidences.get_edge_nodes(edge_id)
            neighbors.update(n for n in edge_nodes if n != str(node_id))
        
        return neighbors
    
    def get_edge_nodes(self, edge_id: Any) -> List[Any]:
        """Get nodes connected by an edge"""
        return self._core_ops.get_edge_nodes(edge_id)
    
    def get_node_edges(self, node_id: Any) -> List[Any]:
        """Get edges incident to a node"""
        return self._core_ops.get_node_edges(node_id)
    
    def is_empty(self) -> bool:
        """Check if the hypergraph is empty"""
        return self._core_ops.is_empty()
    
    # ================================================================
    # PROPERTIES AND BASIC QUERIES
    # ================================================================
    
    @property
    def nodes(self) -> set:
        """Return the set of all nodes in the hypergraph"""
        return set(self.incidences.get_all_nodes())
    
    @property 
    def edges(self) -> EdgeView:
        """Get edge view for accessing edge-node relationships"""
        if 'edges' not in self._cache:
            self._cache['edges'] = EdgeView(self)
        return self._cache['edges']
    
    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the hypergraph"""
        return len(self.nodes)
    
    @property
    def num_edges(self) -> int:
        """Number of edges in O(1) time"""
        self._build_performance_indexes()
        return len(self._edges_cache or set())
    
    @property
    def num_incidences(self) -> int:
        """Number of incidences (total node-edge relationships)"""
        try:
            return len(self.incidences.data) if not self.incidences.data.is_empty() else 0
        except Exception:
            return 0
    
    # ================================================================
    # PERFORMANCE OPTIMIZATION (delegated to PerformanceOperations)
    # ================================================================
    
    def _build_performance_indexes(self):
        """Build performance indexes (internal)"""
        self._performance_ops.build_performance_indexes()
    
    def build_performance_indexes(self):
        """Build performance indexes for O(1) operations"""
        self._build_performance_indexes()
    
    def _invalidate_cache(self):
        """Invalidate all cached data to force rebuild"""
        self._performance_ops.invalidate_cache()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        return self._performance_ops.get_performance_stats()
    
    def print_performance_report(self) -> None:
        """Print detailed performance analysis"""
        self._performance_ops.print_performance_report()
    
    def get_node_degree(self, node_id: Any) -> int:
        """Get node degree in O(1) time using pre-computed cache"""
        return self._performance_ops.get_node_degree(node_id)
    
    def node_degree(self, node_id: Any) -> int:
        """Get the degree of a node (alias for get_node_degree)"""
        return self.get_node_degree(node_id)
    
    def get_edge_size(self, edge_id: Any) -> int:
        """Get edge size in O(1) time using pre-computed cache"""
        return self._performance_ops.get_edge_size(edge_id)
    
    def get_multiple_node_degrees(self, node_ids: List[Any]) -> Dict[Any, int]:
        """Get degrees for multiple nodes in one optimized operation"""
        return self._performance_ops.get_multiple_node_degrees(node_ids)
    
    def get_multiple_edge_nodes(self, edge_ids: List[Any]) -> Dict[Any, List[Any]]:
        """Get nodes for multiple edges in one optimized operation"""
        return self._performance_ops.get_multiple_edge_nodes(edge_ids)
    
    def get_multiple_node_edges(self, node_ids: List[Any]) -> Dict[Any, List[Any]]:
        """Get edges for multiple nodes in one optimized operation"""
        return self._performance_ops.get_multiple_node_edges(node_ids)
    
    @classmethod
    def clear_global_cache(cls):
        """Clear any global caches to ensure fresh state"""
        PerformanceOperations.clear_global_cache()
    
    # ================================================================
    # I/O OPERATIONS (delegated to IOOperations)
    # ================================================================
    
    def to_dict(self) -> Dict[Any, List[Any]]:
        """Convert hypergraph to edge dictionary representation"""
        return self._io_ops.to_dict()
    
    def to_dataframe(self) -> pl.DataFrame:
        """Get the underlying incidence DataFrame"""
        return self._io_ops.to_dataframe()
    
    def save(self, filepath: Union[str, Path], format: Optional[str] = None) -> None:
        """Save hypergraph to file"""
        self._io_ops.save(filepath, format)
    
    @classmethod
    def load(cls, filepath: Union[str, Path], format: Optional[str] = None) -> 'Hypergraph':
        """Load hypergraph from file"""
        return IOOperations.load(cls, filepath, format)
    
    def to_json(self) -> str:
        """Convert hypergraph to JSON representation"""
        return self._io_ops.to_json()
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Hypergraph':
        """Create hypergraph from JSON representation"""
        return IOOperations.from_json(cls, json_str)
    
    def to_gexf(self) -> str:
        """Export hypergraph to GEXF format"""
        return self._io_ops.to_gexf()
    
    @classmethod
    def from_gexf(cls, gexf_str: str, name: str = "from_gexf") -> 'Hypergraph':
        """Create hypergraph from GEXF string"""
        return IOOperations.from_gexf(cls, gexf_str, name)
    
    def to_graphml(self) -> str:
        """Export hypergraph to GraphML format"""
        return self._io_ops.to_graphml()
    
    @classmethod
    def from_graphml(cls, graphml_str: str, name: str = "from_graphml") -> 'Hypergraph':
        """Create hypergraph from GraphML string"""
        return IOOperations.from_graphml(cls, graphml_str, name)
    
    def to_networkx(self):
        """Convert hypergraph to NetworkX graph"""
        return self._io_ops.to_networkx()
    
    @classmethod
    def from_networkx(cls, nx_graph, name: str = "from_networkx") -> 'Hypergraph':
        """Create hypergraph from NetworkX graph"""
        return IOOperations.from_networkx(cls, nx_graph, name)
    
    # ================================================================
    # ALGORITHM OPERATIONS (delegated to AlgorithmOperations)
    # ================================================================
    
    def random_walk(self, start_node: Any, steps: int = 100, seed: Optional[int] = None) -> List[Any]:
        """Perform a random walk on the hypergraph starting from the given node"""
        return self._algorithm_ops.random_walk(start_node, steps, seed)
    
    def shortest_path(self, source: Any, target: Any) -> Optional[List[Any]]:
        """Find shortest path between two nodes"""
        return self._algorithm_ops.shortest_path(source, target)
    
    def connected_components(self) -> List[List[Any]]:
        """Find all connected components in the hypergraph"""
        return self._algorithm_ops.connected_components()
    
    def degree_centrality(self, normalized: bool = True) -> Dict[Any, float]:
        """Calculate degree centrality for all nodes"""
        return self._algorithm_ops.degree_centrality(normalized)
    
    def betweenness_centrality(self, normalized: bool = True) -> Dict[Any, float]:
        """Calculate betweenness centrality for all nodes"""
        return self._algorithm_ops.betweenness_centrality(normalized)
    
    def closeness_centrality(self, normalized: bool = True) -> Dict[Any, float]:
        """Calculate closeness centrality for all nodes"""
        return self._algorithm_ops.closeness_centrality(normalized)
    
    def eigenvector_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> Dict[Any, float]:
        """Calculate eigenvector centrality using power iteration"""
        return self._algorithm_ops.eigenvector_centrality(max_iter, tol)
    
    def hits(self, max_iter: int = 100, tol: float = 1e-6) -> Tuple[Dict[Any, float], Dict[Any, float]]:
        """HITS algorithm adapted for hypergraphs"""
        return self._algorithm_ops.hits(max_iter, tol)
    
    def pagerank(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> Dict[Any, float]:
        """PageRank algorithm adapted for hypergraphs"""
        return self._algorithm_ops.pagerank(alpha, max_iter, tol)
    
    def minimum_spanning_tree(self) -> 'Hypergraph':
        """Find minimum spanning tree"""
        return self._algorithm_ops.minimum_spanning_tree()
    
    def max_flow(self, source: Any, sink: Any) -> float:
        """Maximum flow between source and sink"""
        return self._algorithm_ops.max_flow(source, sink)
    
    def min_cut(self, source: Any, sink: Any) -> float:
        """Minimum cut between source and sink"""
        return self._algorithm_ops.min_cut(source, sink)
    
    # ================================================================
    # UTILITY METHODS
    # ================================================================
    
    def __str__(self) -> str:
        """String representation of hypergraph"""
        return f"Hypergraph(name='{self.name}', nodes={self.num_nodes}, edges={self.num_edges})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()
    
    def __len__(self) -> int:
        """Number of edges in the hypergraph"""
        return self.num_edges
    
    def __bool__(self) -> bool:
        """True if hypergraph is non-empty"""
        return not self.is_empty()
    
    # ================================================================
    # CLASS METHODS AND STATIC METHODS (TODO: move to IOOperations)
    # ================================================================
    
    @classmethod
    def from_dataframe(cls, data: pl.DataFrame, node_col: str = 'node_id', 
                      edge_col: str = 'edge_id', weight_col: Optional[str] = None,
                      name: Optional[str] = None) -> 'Hypergraph':
        """
        Create hypergraph from Polars DataFrame
        
        Args:
            data: Polars DataFrame with node-edge relationships
            node_col: Column name containing node identifiers
            edge_col: Column name containing edge identifiers  
            weight_col: Optional column name containing edge weights
            name: Optional name for the hypergraph
        """
        return cls._io_ops.from_dataframe(data, node_col, edge_col, weight_col, name)    @classmethod  
    def from_dict(cls, edge_dict: Dict[Any, List[Any]], name: Optional[str] = None) -> 'Hypergraph':
        """
        Create hypergraph from dictionary
        
        Parameters
        ----------
        edge_dict : Dict[Any, List[Any]]
            Dictionary mapping edge IDs to lists of node IDs
        name : str, optional
            Name for the hypergraph
            
        Returns
        -------
        Hypergraph
            New hypergraph instance
        """
        try:
            if not edge_dict:
                return cls(name=name)
            
            return cls(setsystem=edge_dict, name=name)
            
        except Exception as e:
            raise HypergraphError(f"Failed to create hypergraph from dict: {e}")
    
    # ===================================================================
    # VISUALIZATION OPERATIONS DELEGATION METHODS  
    # ===================================================================
    
    def get_layout_coordinates(self, layout_type: str = "spring", **kwargs) -> Dict[Any, Tuple[float, float]]:
        """Generate layout coordinates for visualization"""
        return self._visualization_ops.get_layout_coordinates(layout_type, **kwargs)
    
    def layout(self, algorithm: str = 'spring', **kwargs) -> Dict[Any, Tuple[float, float]]:
        """Generate layout coordinates (alias)"""
        return self._visualization_ops.layout(algorithm, **kwargs)
    
    def draw(self, layout: str = "spring", node_size: int = 300, 
             edge_color: str = "gray", node_color: str = "lightblue",
             with_labels: bool = True, figsize: Tuple[int, int] = (10, 8), **kwargs) -> None:
        """Draw the hypergraph using matplotlib
        
        Args:
            layout: Layout algorithm to use
            node_size: Size of node markers
            edge_color: Color for edges
            node_color: Color for nodes
            with_labels: Whether to show node labels
            figsize: Figure size tuple
        """
        return self._visualization_ops.draw(layout, node_size, edge_color, node_color, 
                                          with_labels, figsize, **kwargs)
    
    def get_node_positions_array(self, layout: str = "spring", **kwargs):
        """Get node positions as numpy arrays for external visualization"""
        return self._visualization_ops.get_node_positions_array(layout, **kwargs)
    
    def export_layout(self, layout: str = "spring", filename: Optional[str] = None, **kwargs) -> Dict[Any, Tuple[float, float]]:
        """Export layout coordinates to file or return them"""
        return self._visualization_ops.export_layout(layout, filename, **kwargs)
    
    # ===================================================================
    # SET OPERATIONS DELEGATION METHODS  
    # ===================================================================
    
    def union(self, other: 'Hypergraph', node_merge_strategy: str = 'union', 
              edge_merge_strategy: str = 'union') -> 'Hypergraph':
        """Compute union of two hypergraphs"""
        return self._set_ops.union(other, node_merge_strategy, edge_merge_strategy)
    
    def intersection(self, other: 'Hypergraph', mode: str = 'edges') -> 'Hypergraph':
        """Compute intersection of two hypergraphs"""
        return self._set_ops.intersection(other, mode)
    
    def difference(self, other: 'Hypergraph', mode: str = 'edges') -> 'Hypergraph':
        """Compute difference of two hypergraphs (self - other)"""
        return self._set_ops.difference(other, mode)
    
    def subgraph_nodes(self, nodes: Iterable[Any]) -> 'Hypergraph':
        """Extract subgraph containing only specified nodes"""
        return self._set_ops.subgraph_nodes(nodes)
    
    def subgraph_edges(self, edges: Iterable[Any]) -> 'Hypergraph':
        """Extract subgraph containing only specified edges"""
        return self._set_ops.subgraph_edges(edges)
    
    def filter_nodes(self, predicate) -> 'Hypergraph':
        """Filter nodes based on a predicate function"""
        return self._set_ops.filter_nodes(predicate)
    
    def filter_edges(self, predicate) -> 'Hypergraph':
        """Filter edges based on a predicate function"""
        return self._set_ops.filter_edges(predicate)
    
    def filter_nodes_by_degree(self, min_degree: int = None, max_degree: int = None) -> Set[Any]:
        """Filter nodes by degree constraints"""
        return self._performance_ops.filter_nodes_by_degree(min_degree, max_degree)
    
    def filter_by_edge_size(self, min_size: int = 1, max_size: Optional[int] = None) -> 'Hypergraph':
        """Filter edges by their size (number of incident nodes)"""
        return self._set_ops.filter_by_edge_size(min_size, max_size)
    
    def largest_connected_component(self) -> 'Hypergraph':
        """Extract the largest connected component"""
        return self._set_ops.largest_connected_component()
    
    # Set operation aliases for intuitive usage
    def __or__(self, other: 'Hypergraph') -> 'Hypergraph':
        """Union operator (|)"""
        return self.union(other)
    
    def __and__(self, other: 'Hypergraph') -> 'Hypergraph':
        """Intersection operator (&)"""
        return self.intersection(other)
    
    def __sub__(self, other: 'Hypergraph') -> 'Hypergraph':
        """Difference operator (-)"""
        return self.difference(other)
    
    # ===================================================================
    # ADVANCED OPERATIONS DELEGATION METHODS  
    # ===================================================================
    
    def dual(self) -> 'Hypergraph':
        """Compute dual hypergraph where nodes and edges are swapped"""
        return self._advanced_ops.dual()
    
    def line_graph(self) -> 'Hypergraph':
        """Compute line graph where edges become nodes"""
        return self._advanced_ops.line_graph()
    
    def adjacency_graph(self, threshold: int = 1) -> 'Hypergraph':
        """Create adjacency graph where nodes are connected if they share edges"""
        return self._advanced_ops.adjacency_graph(threshold)
    
    def clique_graph(self, min_size: int = 2) -> 'Hypergraph':
        """Create clique graph from maximal cliques in the adjacency structure"""
        return self._advanced_ops.clique_graph(min_size)
    
    def k_uniform_projection(self, k: int) -> 'Hypergraph':
        """Project to k-uniform hypergraph by filtering/splitting edges"""
        return self._advanced_ops.k_uniform_projection(k)
    
    def bipartite_projection(self, node_type_attribute: Optional[str] = None) -> Tuple['Hypergraph', 'Hypergraph']:
        """Create bipartite projections onto two node types"""
        return self._advanced_ops.bipartite_projection(node_type_attribute)
    
    def complement(self) -> 'Hypergraph':
        """Compute complement hypergraph"""
        return self._advanced_ops.complement()
    
    def tensor_product(self, other: 'Hypergraph') -> 'Hypergraph':
        """Compute tensor product with another hypergraph"""
        return self._advanced_ops.tensor_product(other)
    
    def contractible_analysis(self) -> Dict[str, Any]:
        """Analyze contractible structures in the hypergraph"""
        return self._advanced_ops.contractible_analysis()
    
    def homomorphism_check(self, target: 'Hypergraph') -> bool:
        """Check if there exists a homomorphism to target hypergraph"""
        return self._advanced_ops.homomorphism_check(target)
    
    # ============================================================================
    # Missing Functionality - Added for Comprehensive Analysis
    # ============================================================================
    
    def k_core_decomposition(self) -> Dict[Any, int]:
        """
        K-core decomposition for hypergraph clustering
        
        Returns
        -------
        Dict[Any, int]
            Mapping of nodes to their k-core numbers
        """
        try:
            return self.algorithm.k_core_decomposition()
        except AttributeError:
            # Fallback implementation using Polars
            import polars as pl
            
            # Get degree for each node
            degrees = {}
            for node in self.nodes:
                degrees[node] = len(self.get_node_edges(node))
            
            # Simple k-core algorithm
            k_cores = {}
            nodes_to_process = set(self.nodes)
            
            current_k = 1
            while nodes_to_process:
                # Find nodes with degree < current_k
                to_remove = {node for node in nodes_to_process if degrees[node] < current_k}
                
                if not to_remove:
                    # All remaining nodes have k-core >= current_k
                    for node in nodes_to_process:
                        k_cores[node] = current_k
                    current_k += 1
                    continue
                
                # Remove low-degree nodes and update neighbors
                for node in to_remove:
                    k_cores[node] = current_k - 1
                    for edge_id in self.get_node_edges(node):
                        edge_nodes = self.get_edge_nodes(edge_id)
                        for neighbor in edge_nodes:
                            if neighbor in degrees:
                                degrees[neighbor] -= 1
                
                nodes_to_process -= to_remove
            
            return k_cores
    
    def diameter(self) -> int:
        """
        Compute hypergraph diameter using Polars-optimized BFS
        
        Returns
        -------
        int
            The diameter (maximum shortest path) of the hypergraph
        """
        try:
            return self.algorithm.diameter()
        except AttributeError:
            # Fallback implementation using BFS
            if self.num_nodes == 0:
                return 0
            
            max_distance = 0
            all_nodes = list(self.nodes)
            
            for start_node in all_nodes:
                distances = {start_node: 0}
                queue = deque([start_node])
                
                while queue:
                    current = queue.popleft()
                    current_dist = distances[current]
                    
                    # Get neighbors through hyperedges
                    neighbors = self.neighbors(current)
                    
                    for neighbor in neighbors:
                        if neighbor not in distances:
                            distances[neighbor] = current_dist + 1
                            queue.append(neighbor)
                            max_distance = max(max_distance, current_dist + 1)
            
            return max_distance
    
    def modularity(self, communities: Optional[Dict[Any, int]] = None) -> float:
        """
        Compute hypergraph modularity using Polars for efficient computation
        
        Parameters
        ----------
        communities : Dict[Any, int], optional
            Node to community mapping. If None, detects communities automatically.
            
        Returns
        -------
        float
            Modularity score
        """
        try:
            return self.algorithm.modularity(communities)
        except AttributeError:
            # Fallback implementation
            if communities is None:
                # Simple community detection - each connected component
                communities = {}
                visited = set()
                community_id = 0
                
                for node in self.nodes:
                    if node not in visited:
                        # BFS to find connected component
                        queue = deque([node])
                        visited.add(node)
                        communities[node] = community_id
                        
                        while queue:
                            current = queue.popleft()
                            for neighbor in self.neighbors(current):
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    communities[neighbor] = community_id
                                    queue.append(neighbor)
                        
                        community_id += 1
            
            # Compute modularity score
            total_edges = self.num_edges
            if total_edges == 0:
                return 0.0
            
            # Calculate modularity using hypergraph formula
            modularity_score = 0.0
            
            for edge_id in self.edges:
                edge_nodes = self.get_edge_nodes(edge_id)
                edge_size = len(edge_nodes)
                
                # Count same-community connections in this hyperedge
                community_counts = {}
                for node in edge_nodes:
                    comm = communities.get(node, 0)
                    community_counts[comm] = community_counts.get(comm, 0) + 1
                
                # Add to modularity score
                for count in community_counts.values():
                    if count > 1:
                        modularity_score += count * (count - 1) / (edge_size * (edge_size - 1))
            
            # Normalize
            return modularity_score / total_edges if total_edges > 0 else 0.0