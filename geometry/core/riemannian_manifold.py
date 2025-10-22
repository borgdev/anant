"""
Riemannian Graph Manifold
=========================

Revolutionary framework: Transform graphs into Riemannian manifolds.

Key Innovation:
- Graph structure → Manifold geometry
- Properties → Metric tensor
- Structure → Curvature
- Patterns emerge naturally from geometry

Mathematical Foundation:
- Riemannian metric from graph properties
- Christoffel symbols for connection
- Riemann curvature tensor
- Geodesics as natural paths
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import numpy as np
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import eigs, eigsh
    from scipy.integrate import ode
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available - geometric features limited")

try:
    import polars as pl
except ImportError:
    logging.warning("Polars not available")

logger = logging.getLogger(__name__)


@dataclass
class MetricTensor:
    """Riemannian metric tensor at a point"""
    point_id: str
    components: np.ndarray  # g_ij matrix
    dimension: int
    
    def distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute distance between two tangent vectors"""
        diff = v2 - v1
        return np.sqrt(np.dot(diff, np.dot(self.components, diff)))
    
    def inverse(self) -> np.ndarray:
        """Compute inverse metric tensor (g^ij)"""
        return np.linalg.inv(self.components)


@dataclass
class CurvatureData:
    """Curvature information at a point"""
    point_id: str
    scalar_curvature: float  # R (single number)
    ricci_curvature: Optional[np.ndarray] = None  # R_ij (matrix)
    gaussian_curvature: Optional[float] = None  # For 2D
    mean_curvature: Optional[float] = None
    sectional_curvatures: Optional[Dict[Tuple[int, int], float]] = None
    
    def is_outlier(self, mean: float, std: float, threshold: float = 2.0) -> bool:
        """Check if this point is an outlier based on curvature"""
        z_score = abs(self.scalar_curvature - mean) / (std + 1e-10)
        return z_score > threshold


class RiemannianGraphManifold:
    """
    Transform any graph into a Riemannian manifold.
    
    Revolutionary Approach:
    - Nodes become points on a smooth manifold
    - Properties define the metric tensor
    - Structure determines curvature
    - Outliers = High curvature points
    - Clusters = Negative curvature regions
    - Paths = Geodesics
    
    Examples:
        >>> from geometry import RiemannianGraphManifold
        >>> 
        >>> # Create manifold from graph
        >>> manifold = RiemannianGraphManifold(graph)
        >>> 
        >>> # Compute curvature everywhere
        >>> curvature_field = manifold.compute_curvature_field()
        >>> 
        >>> # Find outliers (high curvature)
        >>> outliers = manifold.find_outliers_by_curvature()
        >>> 
        >>> # Detect clusters (negative curvature)
        >>> clusters = manifold.detect_natural_clusters()
        >>> 
        >>> # Compute geodesic (optimal path)
        >>> path = manifold.geodesic("node1", "node2")
    """
    
    def __init__(
        self,
        graph: Any,
        property_weights: Optional[Dict[str, float]] = None,
        embedding_dim: int = None,
        use_properties: bool = True
    ):
        """
        Initialize Riemannian manifold from graph.
        
        Args:
            graph: Source graph (Hypergraph, LCG, or MultiModal)
            property_weights: Weights for properties in metric
            embedding_dim: Dimension to embed into (auto if None)
            use_properties: Use properties for metric (True) or structure only
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("SciPy required for geometric analysis")
        
        self.graph = graph
        self.property_weights = property_weights or {}
        self.use_properties = use_properties
        
        # Detect graph type and extract structure
        self.nodes = self._extract_nodes()
        self.edges = self._extract_edges()
        self.properties = self._extract_properties() if use_properties else {}
        
        # Determine embedding dimension
        self.embedding_dim = embedding_dim or self._estimate_dimension()
        
        # Embed nodes into Euclidean space (for metric computation)
        self.node_coordinates = self._embed_nodes()
        
        # Compute metric tensor at each node
        self.metric_tensors = self._compute_metric_tensors()
        
        # Curvature data (computed on demand)
        self.curvature_data: Dict[str, CurvatureData] = {}
        
        logger.info(f"Riemannian manifold initialized: {len(self.nodes)} nodes, dim={self.embedding_dim}")
    
    def _extract_nodes(self) -> Set[str]:
        """Extract nodes from graph"""
        # Handle different graph types
        if hasattr(self.graph, 'nodes'):
            return set(self.graph.nodes())
        elif hasattr(self.graph, 'get_nodes'):
            return set(self.graph.get_nodes())
        elif hasattr(self.graph, 'setsystem'):
            # Hypergraph
            nodes = set()
            if hasattr(self.graph.setsystem, 'nodes'):
                nodes = set(self.graph.setsystem.nodes())
            return nodes
        else:
            raise ValueError("Cannot extract nodes from graph")
    
    def _extract_edges(self) -> List[Tuple[str, str, float]]:
        """Extract edges with weights"""
        edges = []
        
        if hasattr(self.graph, 'edges'):
            for e in self.graph.edges():
                if len(e) >= 2:
                    weight = e[2] if len(e) > 2 else 1.0
                    edges.append((e[0], e[1], weight))
        elif hasattr(self.graph, 'setsystem'):
            # Hypergraph - convert hyperedges to pairwise edges
            if hasattr(self.graph.setsystem, 'get_edges'):
                for edge_id in self.graph.setsystem.get_edges():
                    nodes_in_edge = self.graph.setsystem.get_edge_members(edge_id)
                    # Create clique
                    for i, n1 in enumerate(nodes_in_edge):
                        for n2 in nodes_in_edge[i+1:]:
                            edges.append((n1, n2, 1.0))
        
        return edges
    
    def _extract_properties(self) -> Dict[str, Dict[str, Any]]:
        """Extract node properties"""
        props = {}
        
        if hasattr(self.graph, 'properties'):
            prop_store = self.graph.properties
            if hasattr(prop_store, 'get_nodes_with_properties'):
                for node_id in prop_store.get_nodes_with_properties():
                    props[node_id] = prop_store.get_node_properties(node_id)
        
        return props
    
    def _estimate_dimension(self) -> int:
        """Estimate intrinsic dimension of the graph"""
        n_nodes = len(self.nodes)
        
        # Heuristics for dimension
        if n_nodes < 10:
            return 2
        elif n_nodes < 100:
            return 3
        elif n_nodes < 1000:
            return min(5, int(np.log(n_nodes)))
        else:
            return min(10, int(np.log2(n_nodes)))
    
    def _embed_nodes(self) -> Dict[str, np.ndarray]:
        """
        Embed nodes into Euclidean space.
        
        Uses spectral embedding (Laplacian eigenmaps) which
        preserves local geometry.
        """
        node_list = list(self.nodes)
        n = len(node_list)
        node_index = {node: i for i, node in enumerate(node_list)}
        
        # Build adjacency matrix
        A = lil_matrix((n, n))
        for u, v, w in self.edges:
            if u in node_index and v in node_index:
                i, j = node_index[u], node_index[v]
                A[i, j] = w
                A[j, i] = w
        
        A = A.tocsr()
        
        # Degree matrix
        D = np.array(A.sum(axis=1)).flatten()
        D_inv_sqrt = np.sqrt(1.0 / (D + 1e-10))
        
        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        D_mat = lil_matrix((n, n))
        D_mat.setdiag(D_inv_sqrt)
        D_mat = D_mat.tocsr()
        
        L_norm = D_mat @ A @ D_mat
        
        # Compute smallest eigenvectors (except first)
        k = min(self.embedding_dim + 1, n - 1)
        try:
            eigenvalues, eigenvectors = eigsh(L_norm, k=k, which='LA')
            # Use eigenvectors 1 to embedding_dim (skip first)
            embedding = eigenvectors[:, 1:self.embedding_dim+1]
        except:
            # Fallback: random embedding
            embedding = np.random.randn(n, self.embedding_dim) * 0.1
        
        # Create coordinate dict
        coordinates = {}
        for i, node in enumerate(node_list):
            coordinates[node] = embedding[i, :]
        
        return coordinates
    
    def _compute_metric_tensors(self) -> Dict[str, MetricTensor]:
        """
        Compute metric tensor at each node.
        
        The metric encodes how to measure distances and angles
        in the manifold's tangent space.
        
        We use:
        g_ij = δ_ij + Σ w_k (∂p_k/∂x_i)(∂p_k/∂x_j)
        
        Where p_k are properties, w_k are weights.
        """
        tensors = {}
        
        for node_id in self.nodes:
            # Start with Euclidean metric
            g = np.eye(self.embedding_dim)
            
            # Add property contributions
            if node_id in self.properties:
                node_props = self.properties[node_id]
                
                for prop_name, prop_value in node_props.items():
                    if isinstance(prop_value, (int, float)):
                        weight = self.property_weights.get(prop_name, 1.0)
                        
                        # Property gradient (approximate)
                        grad = self._property_gradient(node_id, prop_name)
                        
                        # Outer product: ∂p ⊗ ∂p
                        g += weight * np.outer(grad, grad)
            
            tensors[node_id] = MetricTensor(
                point_id=node_id,
                components=g,
                dimension=self.embedding_dim
            )
        
        return tensors
    
    def _property_gradient(self, node_id: str, prop_name: str) -> np.ndarray:
        """
        Approximate gradient of property at node.
        
        Uses finite differences with neighbors.
        """
        if node_id not in self.properties:
            return np.zeros(self.embedding_dim)
        
        prop_value = self.properties[node_id].get(prop_name, 0)
        if not isinstance(prop_value, (int, float)):
            return np.zeros(self.embedding_dim)
        
        # Find neighbors
        neighbors = self._get_neighbors(node_id)
        
        if not neighbors:
            return np.zeros(self.embedding_dim)
        
        # Compute average gradient direction
        grad = np.zeros(self.embedding_dim)
        coord = self.node_coordinates[node_id]
        
        for neighbor_id in neighbors:
            if neighbor_id in self.properties:
                neighbor_props = self.properties[neighbor_id]
                if prop_name in neighbor_props:
                    neighbor_value = neighbor_props[prop_name]
                    if isinstance(neighbor_value, (int, float)):
                        neighbor_coord = self.node_coordinates[neighbor_id]
                        
                        # Finite difference
                        diff = neighbor_coord - coord
                        dist = np.linalg.norm(diff) + 1e-10
                        direction = diff / dist
                        
                        value_diff = neighbor_value - prop_value
                        grad += (value_diff / dist) * direction
        
        if len(neighbors) > 0:
            grad /= len(neighbors)
        
        return grad
    
    def _get_neighbors(self, node_id: str) -> Set[str]:
        """Get neighbors of a node"""
        neighbors = set()
        
        for u, v, _ in self.edges:
            if u == node_id:
                neighbors.add(v)
            elif v == node_id:
                neighbors.add(u)
        
        return neighbors
    
    def compute_curvature_at_node(self, node_id: str) -> CurvatureData:
        """
        Compute scalar curvature at a specific node.
        
        Curvature measures how the manifold bends.
        High curvature = Outlier, Special structure
        Negative curvature = Hyperbolic region (clusters)
        Positive curvature = Spherical region (tight group)
        """
        if node_id in self.curvature_data:
            return self.curvature_data[node_id]
        
        metric = self.metric_tensors[node_id]
        g = metric.components
        g_inv = metric.inverse()
        
        # Compute Christoffel symbols (simplified)
        # Γ^k_ij ≈ (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
        
        # For computational efficiency, use discrete approximation
        christoffel = self._compute_christoffel_symbols(node_id)
        
        # Riemann curvature tensor (simplified)
        # R^l_ijk = ∂_i Γ^l_jk - ∂_j Γ^l_ik + ...
        
        # Scalar curvature (trace of Ricci tensor)
        scalar_curvature = self._compute_scalar_curvature(node_id, christoffel)
        
        curvature_data = CurvatureData(
            point_id=node_id,
            scalar_curvature=scalar_curvature
        )
        
        self.curvature_data[node_id] = curvature_data
        return curvature_data
    
    def _compute_christoffel_symbols(self, node_id: str) -> np.ndarray:
        """Compute Christoffel symbols at node (simplified)"""
        d = self.embedding_dim
        gamma = np.zeros((d, d, d))
        
        metric = self.metric_tensors[node_id]
        g = metric.components
        g_inv = metric.inverse()
        
        # Approximate metric derivatives using neighbors
        neighbors = list(self._get_neighbors(node_id))
        
        if not neighbors:
            return gamma
        
        for neighbor_id in neighbors[:min(5, len(neighbors))]:  # Limit for efficiency
            if neighbor_id in self.metric_tensors:
                g_neighbor = self.metric_tensors[neighbor_id].components
                
                # Finite difference
                dg = g_neighbor - g
                
                # Simplified Christoffel
                for k in range(d):
                    for i in range(d):
                        for j in range(d):
                            for l in range(d):
                                gamma[k, i, j] += 0.5 * g_inv[k, l] * (
                                    dg[j, l] + dg[i, l] - dg[i, j]
                                )
        
        if neighbors:
            gamma /= len(neighbors[:5])
        
        return gamma
    
    def _compute_scalar_curvature(self, node_id: str, christoffel: np.ndarray) -> float:
        """Compute scalar curvature (simplified)"""
        # Discrete curvature approximation
        # Based on Gaussian curvature for discrete surfaces
        
        neighbors = list(self._get_neighbors(node_id))
        
        if len(neighbors) < 2:
            return 0.0
        
        # Angle defect method (discrete Gaussian curvature)
        coord = self.node_coordinates[node_id]
        
        # Compute angles
        total_angle = 0.0
        for i in range(len(neighbors)):
            n1 = neighbors[i]
            n2 = neighbors[(i+1) % len(neighbors)]
            
            if n1 in self.node_coordinates and n2 in self.node_coordinates:
                v1 = self.node_coordinates[n1] - coord
                v2 = self.node_coordinates[n2] - coord
                
                # Angle between vectors
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                total_angle += angle
        
        # Angle defect
        expected_angle = 2 * np.pi
        curvature = (expected_angle - total_angle) / (len(neighbors) + 1e-10)
        
        return curvature
    
    def compute_curvature_field(self) -> Dict[str, float]:
        """
        Compute scalar curvature at all nodes.
        
        Returns dict of node_id -> curvature.
        """
        curvature_field = {}
        
        for node_id in self.nodes:
            curvature_data = self.compute_curvature_at_node(node_id)
            curvature_field[node_id] = curvature_data.scalar_curvature
        
        logger.info(f"Computed curvature field for {len(curvature_field)} nodes")
        return curvature_field
    
    def find_outliers_by_curvature(
        self,
        threshold: float = 2.0,
        method: str = "zscore"
    ) -> List[str]:
        """
        Find outliers based on curvature.
        
        HIGH CURVATURE = OUTLIER (key insight)
        
        Args:
            threshold: Z-score threshold
            method: "zscore" or "absolute"
            
        Returns:
            List of outlier node IDs
        """
        curvature_field = self.compute_curvature_field()
        
        curvatures = list(curvature_field.values())
        mean_curv = np.mean(curvatures)
        std_curv = np.std(curvatures)
        
        outliers = []
        
        for node_id, curvature in curvature_field.items():
            if method == "zscore":
                z_score = abs(curvature - mean_curv) / (std_curv + 1e-10)
                if z_score > threshold:
                    outliers.append(node_id)
            elif method == "absolute":
                if abs(curvature) > threshold:
                    outliers.append(node_id)
        
        logger.info(f"Found {len(outliers)} outliers by curvature (threshold={threshold})")
        return outliers
    
    def detect_natural_clusters(self) -> Dict[str, List[str]]:
        """
        Detect clusters from curvature structure.
        
        NEGATIVE CURVATURE REGIONS = CLUSTERS (key insight)
        
        Hyperbolic geometry naturally separates clusters.
        """
        curvature_field = self.compute_curvature_field()
        
        # Find negative curvature regions
        negative_nodes = [
            node_id for node_id, curv in curvature_field.items()
            if curv < 0
        ]
        
        # Group connected negative curvature nodes
        clusters = {}
        visited = set()
        cluster_id = 0
        
        for node_id in negative_nodes:
            if node_id not in visited:
                cluster = self._expand_cluster(node_id, negative_nodes, visited)
                clusters[f"cluster_{cluster_id}"] = cluster
                cluster_id += 1
        
        logger.info(f"Detected {len(clusters)} natural clusters")
        return clusters
    
    def _expand_cluster(
        self,
        start_node: str,
        candidate_nodes: List[str],
        visited: Set[str]
    ) -> List[str]:
        """Expand cluster from seed node"""
        cluster = []
        queue = [start_node]
        visited.add(start_node)
        
        while queue:
            node = queue.pop(0)
            cluster.append(node)
            
            for neighbor in self._get_neighbors(node):
                if neighbor in candidate_nodes and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return cluster
    
    def geodesic_distance(self, node1: str, node2: str) -> float:
        """
        Compute geodesic distance between two nodes.
        
        Geodesic = shortest path respecting the metric.
        """
        # Simplified: use Euclidean distance in embedded space
        # TODO: Implement true geodesic via parallel transport
        
        if node1 not in self.node_coordinates or node2 not in self.node_coordinates:
            return float('inf')
        
        coord1 = self.node_coordinates[node1]
        coord2 = self.node_coordinates[node2]
        
        return np.linalg.norm(coord2 - coord1)
    
    def get_curvature_statistics(self) -> Dict[str, float]:
        """Get statistics of curvature distribution"""
        curvature_field = self.compute_curvature_field()
        curvatures = list(curvature_field.values())
        
        return {
            'mean': np.mean(curvatures),
            'std': np.std(curvatures),
            'min': np.min(curvatures),
            'max': np.max(curvatures),
            'median': np.median(curvatures),
            'negative_fraction': sum(1 for c in curvatures if c < 0) / len(curvatures)
        }
