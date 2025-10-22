"""
Modal Metrics and Centrality Measures
=====================================

Advanced metrics and centrality measures for multi-modal hypergraph analysis.
"""

from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

try:
    import polars as pl
    import numpy as np
    from scipy import stats
    from scipy.spatial.distance import cosine as cosine_distance
except ImportError as e:
    logging.warning(f"Optional dependencies not available: {e}")

logger = logging.getLogger(__name__)


@dataclass
class MultiModalCentrality:
    """
    Centrality scores across multiple modalities.
    
    Attributes:
        entity_id: Entity identifier
        metric: Centrality metric used
        per_modality: Scores per modality
        aggregated: Aggregated score
        aggregation_method: Method used for aggregation
        rank: Rank among all entities
        percentile: Percentile score
    """
    entity_id: str
    metric: str
    per_modality: Dict[str, float]
    aggregated: float
    aggregation_method: str = "weighted_average"
    rank: Optional[int] = None
    percentile: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return (f"MultiModalCentrality(entity={self.entity_id}, "
                f"metric={self.metric}, score={self.aggregated:.3f})")


@dataclass
class ModalCorrelation:
    """
    Correlation between two modalities.
    
    Attributes:
        modality_a: First modality
        modality_b: Second modality
        correlation_value: Correlation score [0, 1]
        method: Correlation method used
        shared_entities: Number of shared entities
        total_entities_a: Total entities in modality A
        total_entities_b: Total entities in modality B
    """
    modality_a: str
    modality_b: str
    correlation_value: float
    method: str
    shared_entities: int
    total_entities_a: int
    total_entities_b: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def overlap_ratio(self) -> float:
        """Ratio of shared entities to total unique entities."""
        total_unique = self.total_entities_a + self.total_entities_b - self.shared_entities
        return self.shared_entities / total_unique if total_unique > 0 else 0.0
    
    def __repr__(self):
        return (f"ModalCorrelation({self.modality_a} ↔ {self.modality_b}, "
                f"{self.method}={self.correlation_value:.3f})")


class ModalMetrics:
    """
    Comprehensive metrics calculator for multi-modal hypergraphs.
    
    Provides advanced centrality measures, correlation analysis,
    and statistical metrics across modalities.
    """
    
    def __init__(self, multi_modal_hypergraph):
        """
        Initialize modal metrics calculator.
        
        Args:
            multi_modal_hypergraph: MultiModalHypergraph instance
        """
        self.mmhg = multi_modal_hypergraph
        self.centrality_cache: Dict[Tuple[str, str], float] = {}
        
        logger.info("Initialized ModalMetrics")
    
    def compute_degree_centrality(
        self,
        modality: str,
        entity_id: str,
        normalized: bool = True
    ) -> float:
        """
        Compute degree centrality for an entity in a modality.
        
        Args:
            modality: Modality name
            entity_id: Entity to analyze
            normalized: Whether to normalize by max possible degree
            
        Returns:
            Degree centrality score
        """
        cache_key = (modality, entity_id, "degree")
        if cache_key in self.centrality_cache:
            return self.centrality_cache[cache_key]
        
        hg = self.mmhg.get_modality(modality)
        degree = self.mmhg._get_entity_connections(hg, entity_id)
        
        if normalized:
            nodes = self.mmhg._get_nodes_from_hypergraph(hg)
            max_degree = len(nodes) - 1
            degree = degree / max_degree if max_degree > 0 else 0.0
        
        self.centrality_cache[cache_key] = float(degree)
        return float(degree)
    
    def compute_betweenness_centrality(
        self,
        modality: str,
        entity_id: str,
        normalized: bool = True
    ) -> float:
        """
        Compute betweenness centrality for an entity.
        
        Measures how often an entity lies on shortest paths between
        other entities.
        
        Args:
            modality: Modality name
            entity_id: Entity to analyze
            normalized: Whether to normalize
            
        Returns:
            Betweenness centrality score
        """
        cache_key = (modality, entity_id, "betweenness")
        if cache_key in self.centrality_cache:
            return self.centrality_cache[cache_key]
        
        hg = self.mmhg.get_modality(modality)
        nodes = self.mmhg._get_nodes_from_hypergraph(hg)
        
        if entity_id not in nodes:
            return 0.0
        
        # Build adjacency for shortest path computation
        adjacency = self._build_adjacency(hg)
        
        betweenness = 0.0
        
        # For each pair of other nodes
        for source in nodes:
            if source == entity_id:
                continue
            
            # BFS to find shortest paths from source
            paths = self._find_all_shortest_paths(adjacency, source)
            
            for target in nodes:
                if target == entity_id or target == source:
                    continue
                
                # Check if entity is on shortest path
                if target in paths and entity_id in paths:
                    # Count paths through entity
                    if self._is_on_path(paths, source, entity_id, target):
                        betweenness += 1.0
        
        # Normalize
        if normalized:
            n = len(nodes)
            normalizer = (n - 1) * (n - 2) / 2
            betweenness = betweenness / normalizer if normalizer > 0 else 0.0
        
        self.centrality_cache[cache_key] = betweenness
        return betweenness
    
    def compute_closeness_centrality(
        self,
        modality: str,
        entity_id: str,
        normalized: bool = True
    ) -> float:
        """
        Compute closeness centrality for an entity.
        
        Measures average distance to all other entities.
        
        Args:
            modality: Modality name
            entity_id: Entity to analyze
            normalized: Whether to normalize
            
        Returns:
            Closeness centrality score
        """
        cache_key = (modality, entity_id, "closeness")
        if cache_key in self.centrality_cache:
            return self.centrality_cache[cache_key]
        
        hg = self.mmhg.get_modality(modality)
        nodes = self.mmhg._get_nodes_from_hypergraph(hg)
        
        if entity_id not in nodes or len(nodes) <= 1:
            return 0.0
        
        # Build adjacency
        adjacency = self._build_adjacency(hg)
        
        # BFS to compute distances
        distances = self._bfs_distances(adjacency, entity_id)
        
        # Compute closeness
        reachable_distances = [d for d in distances.values() if d > 0 and d != float('inf')]
        
        if not reachable_distances:
            return 0.0
        
        avg_distance = sum(reachable_distances) / len(reachable_distances)
        closeness = 1.0 / avg_distance if avg_distance > 0 else 0.0
        
        # Normalize by fraction of reachable nodes
        if normalized:
            n = len(nodes)
            closeness = closeness * (len(reachable_distances) / (n - 1))
        
        self.centrality_cache[cache_key] = closeness
        return closeness
    
    def compute_eigenvector_centrality(
        self,
        modality: str,
        entity_id: str,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> float:
        """
        Compute eigenvector centrality using power iteration.
        
        Measures influence based on connections to influential entities.
        
        Args:
            modality: Modality name
            entity_id: Entity to analyze
            max_iterations: Maximum power iterations
            tolerance: Convergence tolerance
            
        Returns:
            Eigenvector centrality score
        """
        cache_key = (modality, entity_id, "eigenvector")
        if cache_key in self.centrality_cache:
            return self.centrality_cache[cache_key]
        
        hg = self.mmhg.get_modality(modality)
        nodes = list(self.mmhg._get_nodes_from_hypergraph(hg))
        
        if entity_id not in nodes or len(nodes) <= 1:
            return 0.0
        
        # Build adjacency matrix
        adjacency = self._build_adjacency(hg)
        n = len(nodes)
        
        # Initialize centrality vector
        centrality = {node: 1.0 / n for node in nodes}
        
        # Power iteration
        for iteration in range(max_iterations):
            old_centrality = centrality.copy()
            
            # Update centrality
            for node in nodes:
                score = 0.0
                neighbors = adjacency.get(node, set())
                for neighbor in neighbors:
                    score += old_centrality.get(neighbor, 0.0)
                centrality[node] = score
            
            # Normalize
            total = sum(centrality.values())
            if total > 0:
                centrality = {k: v / total for k, v in centrality.items()}
            
            # Check convergence
            diff = sum(abs(centrality[node] - old_centrality[node]) for node in nodes)
            if diff < tolerance:
                break
        
        score = centrality.get(entity_id, 0.0)
        self.centrality_cache[cache_key] = score
        return score
    
    def compute_multi_modal_centrality_batch(
        self,
        entity_ids: List[str],
        metric: str = "degree",
        aggregation: str = "weighted_average"
    ) -> List[MultiModalCentrality]:
        """
        Compute centrality for multiple entities efficiently.
        
        Args:
            entity_ids: List of entities to analyze
            metric: Centrality metric
            aggregation: Aggregation method
            
        Returns:
            List of MultiModalCentrality results
        """
        results = []
        
        for entity_id in entity_ids:
            centrality = self.mmhg.compute_cross_modal_centrality(
                entity_id,
                metric=metric,
                aggregation=aggregation
            )
            
            results.append(MultiModalCentrality(
                entity_id=entity_id,
                metric=metric,
                per_modality=centrality['per_modality'],
                aggregated=centrality['aggregated'],
                aggregation_method=aggregation
            ))
        
        # Add rankings
        results.sort(key=lambda x: x.aggregated, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1
            result.percentile = (len(results) - i) / len(results) * 100
        
        return results
    
    def compute_correlation_matrix(
        self,
        method: str = "jaccard"
    ) -> Dict[Tuple[str, str], ModalCorrelation]:
        """
        Compute correlation matrix for all modality pairs.
        
        Args:
            method: Correlation method
            
        Returns:
            Dict mapping modality pairs to correlations
        """
        modalities = self.mmhg.list_modalities()
        correlations = {}
        
        for i, mod_a in enumerate(modalities):
            for mod_b in modalities[i+1:]:
                corr_value = self.mmhg.compute_modal_correlation(
                    mod_a, mod_b, method=method
                )
                
                # Get entity counts
                nodes_a = self.mmhg._get_nodes_from_hypergraph(
                    self.mmhg.get_modality(mod_a)
                )
                nodes_b = self.mmhg._get_nodes_from_hypergraph(
                    self.mmhg.get_modality(mod_b)
                )
                shared = len(nodes_a & nodes_b)
                
                correlation = ModalCorrelation(
                    modality_a=mod_a,
                    modality_b=mod_b,
                    correlation_value=corr_value,
                    method=method,
                    shared_entities=shared,
                    total_entities_a=len(nodes_a),
                    total_entities_b=len(nodes_b)
                )
                
                correlations[(mod_a, mod_b)] = correlation
        
        return correlations
    
    def _build_adjacency(self, hypergraph: Any) -> Dict[str, Set[str]]:
        """Build adjacency list from hypergraph."""
        adjacency = defaultdict(set)
        
        try:
            if hasattr(hypergraph, 'incidences'):
                df = hypergraph.incidences.data
                
                if 'edges' in df.columns and 'nodes' in df.columns:
                    # Group by edge to find connected nodes
                    edge_groups = df.group_by('edges').agg([
                        pl.col('nodes')
                    ])
                    
                    for edge_nodes in edge_groups['nodes']:
                        nodes = list(edge_nodes)
                        # Add edges between all pairs in hyperedge
                        for i, node1 in enumerate(nodes):
                            for node2 in nodes[i+1:]:
                                adjacency[str(node1)].add(str(node2))
                                adjacency[str(node2)].add(str(node1))
        except Exception as e:
            logger.warning(f"Could not build adjacency: {e}")
        
        return dict(adjacency)
    
    def _bfs_distances(
        self,
        adjacency: Dict[str, Set[str]],
        source: str
    ) -> Dict[str, int]:
        """Compute shortest path distances using BFS."""
        distances = {source: 0}
        queue = deque([source])
        
        while queue:
            node = queue.popleft()
            current_dist = distances[node]
            
            for neighbor in adjacency.get(node, set()):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        
        return distances
    
    def _find_all_shortest_paths(
        self,
        adjacency: Dict[str, Set[str]],
        source: str
    ) -> Dict[str, Set[str]]:
        """Find all nodes reachable via shortest paths."""
        distances = self._bfs_distances(adjacency, source)
        
        # Build predecessors for path reconstruction
        predecessors = defaultdict(set)
        for node, dist in distances.items():
            for neighbor in adjacency.get(node, set()):
                if neighbor in distances and distances[neighbor] == dist - 1:
                    predecessors[node].add(neighbor)
        
        return dict(predecessors)
    
    def _is_on_path(
        self,
        predecessors: Dict[str, Set[str]],
        source: str,
        intermediate: str,
        target: str
    ) -> bool:
        """Check if intermediate node is on path from source to target."""
        if intermediate not in predecessors:
            return False
        
        # Trace back from target
        current = target
        visited = set()
        
        while current != source and current not in visited:
            if current == intermediate:
                return True
            visited.add(current)
            
            preds = predecessors.get(current, set())
            if not preds:
                break
            current = next(iter(preds))
        
        return False
    
    def compute_modal_diversity(self, entity_id: str) -> float:
        """
        Compute diversity of entity's modal participation.
        
        Higher score = more evenly distributed across modalities.
        
        Args:
            entity_id: Entity to analyze
            
        Returns:
            Diversity score [0, 1]
        """
        entity_index = self.mmhg._build_entity_index()
        
        if entity_id not in entity_index:
            return 0.0
        
        modalities = entity_index[entity_id]
        total_modalities = len(self.mmhg.list_modalities())
        
        if total_modalities <= 1:
            return 1.0
        
        # Shannon entropy-based diversity
        participation_ratio = len(modalities) / total_modalities
        
        # Get activity distribution across modalities
        activity = []
        for mod in modalities:
            hg = self.mmhg.get_modality(mod)
            activity.append(self.mmhg._get_entity_connections(hg, entity_id))
        
        if not activity or sum(activity) == 0:
            return participation_ratio
        
        # Normalize
        total_activity = sum(activity)
        proportions = [a / total_activity for a in activity]
        
        # Compute Shannon entropy
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in proportions)
        max_entropy = np.log2(len(modalities)) if len(modalities) > 0 else 1
        
        diversity = (entropy / max_entropy) if max_entropy > 0 else 0
        
        return float(diversity * participation_ratio)
    
    def generate_metrics_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report."""
        entity_index = self.mmhg._build_entity_index()
        modalities = self.mmhg.list_modalities()
        
        # Compute correlation matrix
        correlations = self.compute_correlation_matrix()
        
        # Sample centrality for top entities
        top_entities = list(entity_index.keys())[:100]
        centralities = self.compute_multi_modal_centrality_batch(
            top_entities,
            metric="degree"
        )
        
        return {
            'total_entities': len(entity_index),
            'total_modalities': len(modalities),
            'modal_correlations': {
                f"{k[0]}-{k[1]}": v.correlation_value
                for k, v in correlations.items()
            },
            'avg_correlation': np.mean([
                v.correlation_value for v in correlations.values()
            ]) if correlations else 0.0,
            'top_central_entities': [
                {'entity': c.entity_id, 'score': c.aggregated}
                for c in centralities[:10]
            ],
            'centrality_distribution': {
                'mean': np.mean([c.aggregated for c in centralities]),
                'std': np.std([c.aggregated for c in centralities]),
                'min': np.min([c.aggregated for c in centralities]),
                'max': np.max([c.aggregated for c in centralities])
            } if centralities else {}
        }
