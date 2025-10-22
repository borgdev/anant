"""
Multi-Modal Hypergraph Implementation
======================================

Core class for managing and analyzing multiple relationship modalities simultaneously.
Extends Anant's core Hypergraph class with multi-modal capabilities.
"""

from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add Anant to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    import polars as pl
    import numpy as np
    from scipy import stats
    from sklearn.preprocessing import normalize
except ImportError as e:
    logging.warning(f"Optional dependencies not available: {e}")

# Import Anant core Hypergraph
try:
    from anant.classes.hypergraph.core.hypergraph import Hypergraph as AnantHypergraph
    ANANT_AVAILABLE = True
except ImportError:
    ANANT_AVAILABLE = False
    logging.warning("Anant core library not available. Using standalone mode.")
    # Create a base class placeholder
    class AnantHypergraph:
        """Placeholder when Anant is not available"""
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)


@dataclass
class ModalityConfig:
    """Configuration for a single modality"""
    name: str
    weight: float = 1.0
    description: str = ""
    edge_column: str = "edges"
    node_column: str = "nodes"
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


class MultiModalHypergraph(AnantHypergraph):
    """
    Multi-Modal Hypergraph for Cross-Domain Relationship Analysis
    
    Extends Anant's core Hypergraph with multi-modal capabilities.
    Manages multiple hypergraphs representing different relationship types
    (modalities) and provides methods for cross-modal analysis.
    
    This class inherits all functionality from Anant's Hypergraph and adds:
    - Multi-modality management
    - Cross-modal pattern detection
    - Inter-modal relationship discovery
    - Modal correlation analysis
    - Aggregate centrality metrics
    
    Examples:
        >>> mmhg = MultiModalHypergraph()
        >>> mmhg.add_modality("purchases", purchase_hg, weight=2.0)
        >>> mmhg.add_modality("reviews", review_hg, weight=1.0)
        >>> patterns = mmhg.detect_cross_modal_patterns()
    """
    
    def __init__(
        self, 
        name: str = "multi_modal_hypergraph",
        setsystem: Optional[Union[dict, Any]] = None,
        data: Optional[pl.DataFrame] = None,
        properties: Optional[dict] = None,
        **kwargs
    ):
        """
        Initialize multi-modal hypergraph.
        
        Args:
            name: Name identifier for this multi-modal hypergraph
            setsystem: Optional underlying set system (for base Hypergraph)
            data: Optional DataFrame data (for base Hypergraph)
            properties: Optional properties dict (for base Hypergraph)
            **kwargs: Additional arguments passed to base Hypergraph
        """
        # Initialize base Hypergraph class if Anant is available
        if ANANT_AVAILABLE:
            super().__init__(
                setsystem=setsystem,
                data=data,
                properties=properties,
                name=name,
                **kwargs
            )
        else:
            # Standalone mode
            self.name = name
        
        # Multi-modal specific attributes
        self.modalities: Dict[str, Any] = {}  # modality_name -> hypergraph
        self.modality_configs: Dict[str, ModalityConfig] = {}
        self.cross_modal_cache: Dict[str, Any] = {}
        self._entity_index: Optional[Dict[str, Set[str]]] = None
        
        logger.info(f"Initialized MultiModalHypergraph: {name} (Anant: {ANANT_AVAILABLE})")
    
    def add_modality(
        self,
        name: str,
        hypergraph: Any,
        weight: float = 1.0,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a relationship modality to the multi-modal hypergraph.
        
        Args:
            name: Unique name for this modality (e.g., "purchases", "reviews")
            hypergraph: Anant Hypergraph instance for this modality
            weight: Importance weight for this modality (default: 1.0)
            description: Human-readable description
            metadata: Additional metadata about this modality
            
        Examples:
            >>> mmhg.add_modality("collaboration", collab_hg, weight=1.5)
            >>> mmhg.add_modality("citation", citation_hg, weight=1.0)
        """
        if name in self.modalities:
            logger.warning(f"Modality '{name}' already exists. Replacing...")
        
        self.modalities[name] = hypergraph
        self.modality_configs[name] = ModalityConfig(
            name=name,
            weight=weight,
            description=description,
            metadata=metadata or {}
        )
        
        # Invalidate caches
        self._entity_index = None
        self.cross_modal_cache.clear()
        
        logger.info(f"Added modality '{name}' with weight {weight}")
    
    def remove_modality(self, name: str) -> None:
        """Remove a modality from the multi-modal hypergraph."""
        if name not in self.modalities:
            raise ValueError(f"Modality '{name}' not found")
        
        del self.modalities[name]
        del self.modality_configs[name]
        self._entity_index = None
        self.cross_modal_cache.clear()
        
        logger.info(f"Removed modality '{name}'")
    
    def get_modality(self, name: str) -> Any:
        """Get hypergraph for a specific modality."""
        if name not in self.modalities:
            raise ValueError(f"Modality '{name}' not found")
        return self.modalities[name]
    
    def list_modalities(self) -> List[str]:
        """Get list of all modality names."""
        return list(self.modalities.keys())
    
    def get_modality_info(self, name: str) -> ModalityConfig:
        """Get configuration info for a modality."""
        if name not in self.modality_configs:
            raise ValueError(f"Modality '{name}' not found")
        return self.modality_configs[name]
    
    def _build_entity_index(self) -> Dict[str, Set[str]]:
        """
        Build index of which modalities each entity participates in.
        
        Returns:
            Dict mapping entity_id -> set of modality names
        """
        if self._entity_index is not None:
            return self._entity_index
        
        entity_index = defaultdict(set)
        
        for modality_name, hypergraph in self.modalities.items():
            # Extract nodes from hypergraph
            try:
                nodes = self._get_nodes_from_hypergraph(hypergraph)
                for node in nodes:
                    entity_index[str(node)].add(modality_name)
            except Exception as e:
                logger.warning(f"Could not extract nodes from modality '{modality_name}': {e}")
        
        self._entity_index = dict(entity_index)
        return self._entity_index
    
    def _get_nodes_from_hypergraph(self, hypergraph: Any) -> Set[str]:
        """Extract node set from a hypergraph (handles different formats)."""
        # Try different methods to extract nodes
        if hasattr(hypergraph, 'nodes'):
            if callable(hypergraph.nodes):
                return set(str(n) for n in hypergraph.nodes())
            return set(str(n) for n in hypergraph.nodes)
        elif hasattr(hypergraph, 'get_nodes'):
            return set(str(n) for n in hypergraph.get_nodes())
        elif hasattr(hypergraph, 'incidences'):
            # Anant/Polars-based hypergraph
            if hasattr(hypergraph.incidences, 'data'):
                df = hypergraph.incidences.data
                # Try both 'node_id' (Anant) and 'nodes' (custom) column names
                if 'node_id' in df.columns:
                    return set(str(n) for n in df['node_id'].unique())
                elif 'nodes' in df.columns:
                    return set(str(n) for n in df['nodes'].unique())
        
        logger.warning("Could not extract nodes from hypergraph")
        return set()
    
    def find_modal_bridges(
        self,
        min_modalities: int = 2,
        min_connections: int = 1
    ) -> Dict[str, Set[str]]:
        """
        Find entities that bridge multiple modalities.
        
        Args:
            min_modalities: Minimum number of modalities entity must appear in
            min_connections: Minimum connections in each modality
            
        Returns:
            Dict mapping entity_id -> set of modalities it bridges
            
        Examples:
            >>> bridges = mmhg.find_modal_bridges(min_modalities=3)
            >>> print(f"Found {len(bridges)} entities bridging 3+ modalities")
        """
        entity_index = self._build_entity_index()
        
        bridges = {
            entity_id: modalities
            for entity_id, modalities in entity_index.items()
            if len(modalities) >= min_modalities
        }
        
        # Filter by minimum connections if requested
        if min_connections > 1:
            filtered_bridges = {}
            for entity_id, modalities in bridges.items():
                qualified_modalities = set()
                for modality_name in modalities:
                    hg = self.modalities[modality_name]
                    connections = self._get_entity_connections(hg, entity_id)
                    if connections >= min_connections:
                        qualified_modalities.add(modality_name)
                
                if len(qualified_modalities) >= min_modalities:
                    filtered_bridges[entity_id] = qualified_modalities
            
            bridges = filtered_bridges
        
        logger.info(f"Found {len(bridges)} modal bridges (min_modalities={min_modalities})")
        return bridges
    
    def _get_entity_connections(self, hypergraph: Any, entity_id: str) -> int:
        """Get number of connections for an entity in a hypergraph."""
        try:
            if hasattr(hypergraph, 'degree'):
                if callable(hypergraph.degree):
                    return hypergraph.degree(entity_id)
                return hypergraph.degree.get(entity_id, 0)
            elif hasattr(hypergraph, 'incidences'):
                if hasattr(hypergraph.incidences, 'data'):
                    df = hypergraph.incidences.data
                    if 'nodes' in df.columns:
                        return len(df.filter(pl.col('nodes') == entity_id))
        except:
            pass
        return 0
    
    def detect_cross_modal_patterns(
        self,
        min_support: int = 5,
        pattern_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect patterns that exist across multiple modalities.
        
        Args:
            min_support: Minimum number of instances for a pattern
            pattern_types: Types of patterns to detect (default: all)
            
        Returns:
            List of detected cross-modal patterns
            
        Examples:
            >>> patterns = mmhg.detect_cross_modal_patterns(min_support=10)
            >>> for pattern in patterns:
            ...     print(f"{pattern['type']}: {pattern['description']}")
        """
        patterns = []
        
        # Pattern 1: Entities active in multiple modalities
        bridges = self.find_modal_bridges(min_modalities=2)
        if len(bridges) >= min_support:
            patterns.append({
                'type': 'modal_bridge',
                'description': f'{len(bridges)} entities active across multiple modalities',
                'entities': list(bridges.keys())[:100],  # Sample
                'support': len(bridges)
            })
        
        # Pattern 2: Modality co-occurrence patterns
        modality_pairs = self._find_modality_cooccurrence(min_support)
        patterns.extend(modality_pairs)
        
        # Pattern 3: Sequential modal patterns
        sequential_patterns = self._find_sequential_patterns(min_support)
        patterns.extend(sequential_patterns)
        
        logger.info(f"Detected {len(patterns)} cross-modal patterns")
        return patterns
    
    def _find_modality_cooccurrence(self, min_support: int) -> List[Dict[str, Any]]:
        """Find which modalities frequently occur together."""
        entity_index = self._build_entity_index()
        
        # Count modality pairs
        pair_counts = defaultdict(int)
        for entity_id, modalities in entity_index.items():
            if len(modalities) < 2:
                continue
            modality_list = sorted(modalities)
            for i, mod1 in enumerate(modality_list):
                for mod2 in modality_list[i+1:]:
                    pair_counts[(mod1, mod2)] += 1
        
        # Filter by support
        patterns = []
        for (mod1, mod2), count in pair_counts.items():
            if count >= min_support:
                patterns.append({
                    'type': 'modality_cooccurrence',
                    'description': f'Modalities "{mod1}" and "{mod2}" co-occur',
                    'modalities': [mod1, mod2],
                    'support': count
                })
        
        return patterns
    
    def _find_sequential_patterns(self, min_support: int) -> List[Dict[str, Any]]:
        """Find temporal sequential patterns across modalities."""
        patterns = []
        entity_index = self._build_entity_index()
        
        # Analyze modal participation sequences
        # Find common transitions between modalities
        transitions = defaultdict(int)
        
        for entity, modalities in entity_index.items():
            if len(modalities) < 2:
                continue
            
            # Create ordered pairs (simple sequence analysis)
            mod_list = sorted(modalities)  # Note: could use temporal order if available
            for i in range(len(mod_list) - 1):
                transition = (mod_list[i], mod_list[i + 1])
                transitions[transition] += 1
        
        # Find frequent transitions
        for (mod_from, mod_to), count in transitions.items():
            if count >= min_support:
                patterns.append({
                    'type': 'modal_transition',
                    'description': f'Entities transition from "{mod_from}" to "{mod_to}"',
                    'modalities': [mod_from, mod_to],
                    'support': count,
                    'metadata': {
                        'from_modality': mod_from,
                        'to_modality': mod_to,
                        'transition_count': count
                    }
                })
        
        return patterns
    
    def compute_cross_modal_centrality(
        self,
        node_id: str,
        metric: str = "degree",
        aggregation: str = "weighted_average"
    ) -> Dict[str, Any]:
        """
        Compute centrality for a node across all modalities.
        
        Args:
            node_id: Node to analyze
            metric: Centrality metric ("degree", "betweenness", "closeness")
            aggregation: How to aggregate ("weighted_average", "max", "min", "sum")
            
        Returns:
            Dict with centrality scores per modality and aggregated score
            
        Examples:
            >>> centrality = mmhg.compute_cross_modal_centrality(
            ...     "customer_123",
            ...     metric="degree",
            ...     aggregation="weighted_average"
            ... )
            >>> print(f"Overall centrality: {centrality['aggregated']}")
        """
        centrality_scores = {}
        
        for modality_name, hypergraph in self.modalities.items():
            try:
                score = self._compute_centrality(hypergraph, node_id, metric)
                centrality_scores[modality_name] = score
            except Exception as e:
                logger.warning(f"Could not compute centrality for '{node_id}' in '{modality_name}': {e}")
                centrality_scores[modality_name] = 0.0
        
        # Aggregate scores
        aggregated = self._aggregate_scores(centrality_scores, aggregation)
        
        return {
            'node_id': node_id,
            'metric': metric,
            'aggregation': aggregation,
            'per_modality': centrality_scores,
            'aggregated': aggregated
        }
    
    def _compute_centrality(
        self,
        hypergraph: Any,
        node_id: str,
        metric: str
    ) -> float:
        """Compute centrality for a node in a hypergraph."""
        if metric == "degree":
            return float(self._get_entity_connections(hypergraph, node_id))
        elif metric == "betweenness":
            # Import modal metrics for advanced centrality
            try:
                from .modal_metrics import ModalMetrics
                # Get modality name by searching
                for mod_name, hg in self.modalities.items():
                    if hg == hypergraph:
                        metrics = ModalMetrics(self)
                        return metrics.compute_betweenness_centrality(mod_name, node_id)
            except:
                pass
            return 0.0
        elif metric == "closeness":
            # Import modal metrics for advanced centrality
            try:
                from .modal_metrics import ModalMetrics
                # Get modality name by searching
                for mod_name, hg in self.modalities.items():
                    if hg == hypergraph:
                        metrics = ModalMetrics(self)
                        return metrics.compute_closeness_centrality(mod_name, node_id)
            except:
                pass
            return 0.0
        elif metric == "eigenvector":
            # Import modal metrics for advanced centrality
            try:
                from .modal_metrics import ModalMetrics
                # Get modality name by searching
                for mod_name, hg in self.modalities.items():
                    if hg == hypergraph:
                        metrics = ModalMetrics(self)
                        return metrics.compute_eigenvector_centrality(mod_name, node_id)
            except:
                pass
            return 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _aggregate_scores(
        self,
        scores: Dict[str, float],
        method: str
    ) -> float:
        """Aggregate scores across modalities."""
        if not scores:
            return 0.0
        
        values = list(scores.values())
        
        if method == "weighted_average":
            total_weight = sum(
                self.modality_configs[mod].weight
                for mod in scores.keys()
            )
            if total_weight == 0:
                return 0.0
            weighted_sum = sum(
                score * self.modality_configs[mod].weight
                for mod, score in scores.items()
            )
            return weighted_sum / total_weight
        
        elif method == "max":
            return max(values)
        elif method == "min":
            return min(values)
        elif method == "sum":
            return sum(values)
        elif method == "average":
            return sum(values) / len(values)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def discover_inter_modal_relationships(
        self,
        source_modality: str,
        target_modality: str,
        relationship_type: str = "implicit"
    ) -> List[Dict[str, Any]]:
        """
        Discover relationships between entities in different modalities.
        
        Args:
            source_modality: Source modality name
            target_modality: Target modality name
            relationship_type: Type of relationship to find
            
        Returns:
            List of discovered inter-modal relationships
            
        Examples:
            >>> relationships = mmhg.discover_inter_modal_relationships(
            ...     source_modality="purchases",
            ...     target_modality="reviews"
            ... )
            >>> print(f"Found {len(relationships)} cross-modal connections")
        """
        if source_modality not in self.modalities:
            raise ValueError(f"Source modality '{source_modality}' not found")
        if target_modality not in self.modalities:
            raise ValueError(f"Target modality '{target_modality}' not found")
        
        source_hg = self.modalities[source_modality]
        target_hg = self.modalities[target_modality]
        
        # Get nodes from each modality
        source_nodes = self._get_nodes_from_hypergraph(source_hg)
        target_nodes = self._get_nodes_from_hypergraph(target_hg)
        
        # Find nodes that appear in both
        common_nodes = source_nodes & target_nodes
        
        relationships = []
        for node in common_nodes:
            relationships.append({
                'node_id': node,
                'source_modality': source_modality,
                'target_modality': target_modality,
                'relationship_type': relationship_type,
                'source_degree': self._get_entity_connections(source_hg, node),
                'target_degree': self._get_entity_connections(target_hg, node)
            })
        
        logger.info(f"Discovered {len(relationships)} inter-modal relationships "
                   f"between '{source_modality}' and '{target_modality}'")
        
        return relationships
    
    def compute_modal_correlation(
        self,
        modality_a: str,
        modality_b: str,
        method: str = "jaccard"
    ) -> float:
        """
        Compute correlation between two modalities.
        
        Args:
            modality_a: First modality name
            modality_b: Second modality name
            method: Correlation method ("jaccard", "overlap", "cosine")
            
        Returns:
            Correlation score between 0 and 1
            
        Examples:
            >>> corr = mmhg.compute_modal_correlation("purchases", "reviews")
            >>> print(f"Modality correlation: {corr:.3f}")
        """
        if modality_a not in self.modalities or modality_b not in self.modalities:
            raise ValueError("One or both modalities not found")
        
        hg_a = self.modalities[modality_a]
        hg_b = self.modalities[modality_b]
        
        nodes_a = self._get_nodes_from_hypergraph(hg_a)
        nodes_b = self._get_nodes_from_hypergraph(hg_b)
        
        if method == "jaccard":
            intersection = len(nodes_a & nodes_b)
            union = len(nodes_a | nodes_b)
            return intersection / union if union > 0 else 0.0
        
        elif method == "overlap":
            intersection = len(nodes_a & nodes_b)
            min_size = min(len(nodes_a), len(nodes_b))
            return intersection / min_size if min_size > 0 else 0.0
        
        elif method == "cosine":
            intersection = len(nodes_a & nodes_b)
            denominator = (len(nodes_a) * len(nodes_b)) ** 0.5
            return intersection / denominator if denominator > 0 else 0.0
        
        else:
            raise ValueError(f"Unknown correlation method: {method}")
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the multi-modal hypergraph."""
        entity_index = self._build_entity_index()
        
        modality_stats = {}
        for name, hg in self.modalities.items():
            nodes = self._get_nodes_from_hypergraph(hg)
            modality_stats[name] = {
                'num_nodes': len(nodes),
                'weight': self.modality_configs[name].weight,
                'description': self.modality_configs[name].description
            }
        
        # Compute modal participation distribution
        participation_counts = defaultdict(int)
        for modalities in entity_index.values():
            participation_counts[len(modalities)] += 1
        
        return {
            'name': self.name,
            'num_modalities': len(self.modalities),
            'modalities': list(self.modalities.keys()),
            'total_unique_entities': len(entity_index),
            'modality_stats': modality_stats,
            'modal_participation_distribution': dict(participation_counts),
            'avg_modalities_per_entity': sum(
                len(mods) for mods in entity_index.values()
            ) / len(entity_index) if entity_index else 0
        }
    
    def __repr__(self) -> str:
        return (f"MultiModalHypergraph(name='{self.name}', "
                f"modalities={len(self.modalities)}, "
                f"entities={len(self._build_entity_index()) if self.modalities else 0})")
