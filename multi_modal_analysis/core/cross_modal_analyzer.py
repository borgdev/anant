"""
Cross-Modal Pattern Analyzer
=============================

Advanced pattern detection and relationship discovery across multiple modalities.
"""

from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import logging

try:
    import polars as pl
    import numpy as np
    from scipy import stats
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
except ImportError as e:
    logging.warning(f"Optional dependencies not available: {e}")

logger = logging.getLogger(__name__)


@dataclass
class CrossModalPattern:
    """
    Represents a pattern detected across multiple modalities.
    
    Attributes:
        pattern_id: Unique identifier
        pattern_type: Type of pattern (bridge, cooccurrence, sequential, etc.)
        modalities: List of modalities involved
        entities: Entities exhibiting this pattern
        support: Number of instances
        confidence: Pattern confidence score [0, 1]
        metadata: Additional pattern metadata
    """
    pattern_id: str
    pattern_type: str
    modalities: List[str]
    entities: List[str]
    support: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)
    
    def __repr__(self):
        return (f"CrossModalPattern(type={self.pattern_type}, "
                f"modalities={len(self.modalities)}, support={self.support})")


@dataclass
class InterModalRelationship:
    """
    Represents a relationship between entities in different modalities.
    
    Attributes:
        source_entity: Entity in source modality
        target_entity: Entity in target modality
        source_modality: Source modality name
        target_modality: Target modality name
        relationship_type: Type of relationship
        strength: Relationship strength [0, 1]
        metadata: Additional relationship data
    """
    source_entity: str
    target_entity: str
    source_modality: str
    target_modality: str
    relationship_type: str = "implicit"
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.now)
    
    def __repr__(self):
        return (f"InterModalRelationship({self.source_entity} "
                f"[{self.source_modality}] → {self.target_entity} "
                f"[{self.target_modality}], strength={self.strength:.2f})")


class CrossModalAnalyzer:
    """
    Advanced analyzer for cross-modal patterns and relationships.
    
    Provides sophisticated algorithms for:
    - Pattern mining across modalities
    - Anomaly detection
    - Relationship inference
    - Temporal pattern tracking
    """
    
    def __init__(self, multi_modal_hypergraph):
        """
        Initialize cross-modal analyzer.
        
        Args:
            multi_modal_hypergraph: MultiModalHypergraph instance to analyze
        """
        self.mmhg = multi_modal_hypergraph
        self.pattern_cache: Dict[str, List[CrossModalPattern]] = {}
        self.relationship_cache: Dict[Tuple[str, str], List[InterModalRelationship]] = {}
        
        logger.info("Initialized CrossModalAnalyzer")
    
    def mine_frequent_patterns(
        self,
        min_support: int = 5,
        min_modalities: int = 2,
        max_pattern_size: int = 10
    ) -> List[CrossModalPattern]:
        """
        Mine frequent patterns across modalities using itemset mining.
        
        Args:
            min_support: Minimum number of pattern instances
            min_modalities: Minimum modalities involved
            max_pattern_size: Maximum entities per pattern
            
        Returns:
            List of discovered patterns
        """
        cache_key = f"frequent_{min_support}_{min_modalities}"
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        patterns = []
        entity_index = self.mmhg._build_entity_index()
        
        # Pattern 1: Modal Bridge Patterns
        bridges = self.mmhg.find_modal_bridges(min_modalities=min_modalities)
        
        if len(bridges) >= min_support:
            # Group by modality combination
            modality_combinations = defaultdict(list)
            for entity, modalities in bridges.items():
                mod_key = tuple(sorted(modalities))
                modality_combinations[mod_key].append(entity)
            
            # Create patterns for each combination
            for mod_combo, entities in modality_combinations.items():
                if len(entities) >= min_support:
                    pattern = CrossModalPattern(
                        pattern_id=f"bridge_{hash(mod_combo)}",
                        pattern_type="modal_bridge",
                        modalities=list(mod_combo),
                        entities=entities[:max_pattern_size],
                        support=len(entities),
                        confidence=len(entities) / len(entity_index),
                        metadata={
                            'modality_count': len(mod_combo),
                            'entity_count': len(entities)
                        }
                    )
                    patterns.append(pattern)
        
        # Pattern 2: Co-occurrence Patterns
        cooccurrence_patterns = self._mine_cooccurrence_patterns(
            entity_index, min_support
        )
        patterns.extend(cooccurrence_patterns)
        
        # Pattern 3: Activity Concentration Patterns
        concentration_patterns = self._mine_concentration_patterns(
            entity_index, min_support
        )
        patterns.extend(concentration_patterns)
        
        self.pattern_cache[cache_key] = patterns
        logger.info(f"Mined {len(patterns)} frequent patterns")
        
        return patterns
    
    def _mine_cooccurrence_patterns(
        self,
        entity_index: Dict[str, Set[str]],
        min_support: int
    ) -> List[CrossModalPattern]:
        """Mine modality co-occurrence patterns."""
        patterns = []
        modality_pairs = defaultdict(list)
        
        for entity, modalities in entity_index.items():
            if len(modalities) < 2:
                continue
            
            mod_list = sorted(modalities)
            for i, mod1 in enumerate(mod_list):
                for mod2 in mod_list[i+1:]:
                    modality_pairs[(mod1, mod2)].append(entity)
        
        for (mod1, mod2), entities in modality_pairs.items():
            if len(entities) >= min_support:
                pattern = CrossModalPattern(
                    pattern_id=f"cooccur_{mod1}_{mod2}",
                    pattern_type="modality_cooccurrence",
                    modalities=[mod1, mod2],
                    entities=entities,
                    support=len(entities),
                    confidence=1.0,
                    metadata={
                        'pair': (mod1, mod2),
                        'entity_count': len(entities)
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def _mine_concentration_patterns(
        self,
        entity_index: Dict[str, Set[str]],
        min_support: int
    ) -> List[CrossModalPattern]:
        """Identify entities concentrated in specific modalities."""
        patterns = []
        
        # Group entities by their modality signature
        modality_signatures = defaultdict(list)
        for entity, modalities in entity_index.items():
            sig = tuple(sorted(modalities))
            modality_signatures[sig].append(entity)
        
        # Find concentrated patterns
        for signature, entities in modality_signatures.items():
            if len(entities) >= min_support and len(signature) == 1:
                # Entities in only one modality
                pattern = CrossModalPattern(
                    pattern_id=f"concentrated_{signature[0]}",
                    pattern_type="modality_concentration",
                    modalities=list(signature),
                    entities=entities,
                    support=len(entities),
                    confidence=1.0,
                    metadata={
                        'exclusive_modality': signature[0],
                        'entity_count': len(entities)
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def detect_anomalies(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalous cross-modal behavior.
        
        Args:
            method: Anomaly detection method
            contamination: Expected proportion of anomalies
            
        Returns:
            List of detected anomalies
        """
        entity_index = self.mmhg._build_entity_index()
        modalities = self.mmhg.list_modalities()
        
        # Build feature vectors for entities
        features = []
        entity_ids = []
        
        for entity, entity_mods in entity_index.items():
            # Binary features: in modality or not
            feature = [1 if mod in entity_mods else 0 for mod in modalities]
            
            # Add centrality features
            for mod in modalities:
                if mod in entity_mods:
                    hg = self.mmhg.get_modality(mod)
                    cent = self.mmhg._get_entity_connections(hg, entity)
                    feature.append(cent)
                else:
                    feature.append(0)
            
            features.append(feature)
            entity_ids.append(entity)
        
        if not features:
            return []
        
        # Detect anomalies
        anomalies = []
        
        if method == "isolation_forest":
            try:
                features_array = np.array(features)
                iso_forest = IsolationForest(
                    contamination=contamination,
                    random_state=42
                )
                predictions = iso_forest.fit_predict(features_array)
                scores = iso_forest.score_samples(features_array)
                
                for i, (pred, score) in enumerate(zip(predictions, scores)):
                    if pred == -1:  # Anomaly
                        anomalies.append({
                            'entity_id': entity_ids[i],
                            'anomaly_score': -score,
                            'modalities': list(entity_index[entity_ids[i]]),
                            'method': method
                        })
            except Exception as e:
                logger.warning(f"Isolation Forest failed: {e}")
        
        elif method == "statistical":
            # Statistical outlier detection
            features_array = np.array(features)
            
            # Z-score based anomaly detection
            for i, feature in enumerate(features):
                modal_count = sum(feature[:len(modalities)])
                
                # Entities in very few or very many modalities
                if modal_count <= 1 or modal_count >= len(modalities):
                    anomalies.append({
                        'entity_id': entity_ids[i],
                        'anomaly_score': 1.0,
                        'modalities': list(entity_index[entity_ids[i]]),
                        'method': method,
                        'reason': 'extreme_modality_count'
                    })
        
        logger.info(f"Detected {len(anomalies)} anomalies using {method}")
        return anomalies
    
    def infer_implicit_relationships(
        self,
        source_modality: str,
        target_modality: str,
        bridging_modalities: Optional[List[str]] = None,
        min_path_length: int = 2,
        max_path_length: int = 3
    ) -> List[InterModalRelationship]:
        """
        Infer implicit relationships through bridging modalities.
        
        Args:
            source_modality: Source modality
            target_modality: Target modality
            bridging_modalities: Modalities to bridge through
            min_path_length: Minimum path length
            max_path_length: Maximum path length
            
        Returns:
            List of inferred relationships
        """
        cache_key = (source_modality, target_modality)
        if cache_key in self.relationship_cache:
            return self.relationship_cache[cache_key]
        
        relationships = []
        entity_index = self.mmhg._build_entity_index()
        
        # Direct relationships (entities in both modalities)
        direct_relationships = self.mmhg.discover_inter_modal_relationships(
            source_modality, target_modality
        )
        
        for rel in direct_relationships:
            relationships.append(InterModalRelationship(
                source_entity=rel['node_id'],
                target_entity=rel['node_id'],
                source_modality=source_modality,
                target_modality=target_modality,
                relationship_type="direct",
                strength=1.0,
                metadata=rel
            ))
        
        # Implicit relationships through bridging
        if bridging_modalities:
            for bridge_mod in bridging_modalities:
                # Entities in source and bridge
                source_bridge = self._get_entities_in_modalities(
                    entity_index, [source_modality, bridge_mod]
                )
                
                # Entities in bridge and target
                bridge_target = self._get_entities_in_modalities(
                    entity_index, [bridge_mod, target_modality]
                )
                
                # Infer connections through bridge
                for src_entity in source_bridge:
                    for tgt_entity in bridge_target:
                        if src_entity != tgt_entity:
                            relationships.append(InterModalRelationship(
                                source_entity=src_entity,
                                target_entity=tgt_entity,
                                source_modality=source_modality,
                                target_modality=target_modality,
                                relationship_type="bridged",
                                strength=0.5,  # Lower confidence for implicit
                                metadata={
                                    'bridge_modality': bridge_mod,
                                    'path_length': 2
                                }
                            ))
        
        self.relationship_cache[cache_key] = relationships
        logger.info(f"Inferred {len(relationships)} relationships between "
                   f"{source_modality} and {target_modality}")
        
        return relationships
    
    def _get_entities_in_modalities(
        self,
        entity_index: Dict[str, Set[str]],
        modalities: List[str]
    ) -> Set[str]:
        """Get entities that appear in all specified modalities."""
        entities = set()
        for entity, entity_mods in entity_index.items():
            if all(mod in entity_mods for mod in modalities):
                entities.add(entity)
        return entities
    
    def compute_pattern_significance(
        self,
        pattern: CrossModalPattern,
        baseline_support: Optional[int] = None
    ) -> float:
        """
        Compute statistical significance of a pattern.
        
        Args:
            pattern: Pattern to evaluate
            baseline_support: Expected support under null hypothesis
            
        Returns:
            P-value (lower = more significant)
        """
        if baseline_support is None:
            entity_index = self.mmhg._build_entity_index()
            baseline_support = len(entity_index) * 0.05  # 5% baseline
        
        # Binomial test
        try:
            from scipy.stats import binom_test
            p_value = binom_test(
                pattern.support,
                len(self.mmhg._build_entity_index()),
                baseline_support / len(self.mmhg._build_entity_index())
            )
            return p_value
        except:
            # Fallback: simple ratio
            return baseline_support / max(pattern.support, 1)
    
    def rank_patterns_by_interestingness(
        self,
        patterns: List[CrossModalPattern],
        criteria: str = "support"
    ) -> List[CrossModalPattern]:
        """
        Rank patterns by interestingness.
        
        Args:
            patterns: List of patterns to rank
            criteria: Ranking criteria (support, confidence, modalities)
            
        Returns:
            Ranked list of patterns
        """
        if criteria == "support":
            return sorted(patterns, key=lambda p: p.support, reverse=True)
        elif criteria == "confidence":
            return sorted(patterns, key=lambda p: p.confidence, reverse=True)
        elif criteria == "modalities":
            return sorted(patterns, key=lambda p: len(p.modalities), reverse=True)
        elif criteria == "combined":
            # Combined score
            return sorted(
                patterns,
                key=lambda p: p.support * p.confidence * len(p.modalities),
                reverse=True
            )
        else:
            return patterns
    
    def generate_pattern_report(
        self,
        patterns: List[CrossModalPattern]
    ) -> Dict[str, Any]:
        """Generate comprehensive pattern analysis report."""
        if not patterns:
            return {'total_patterns': 0}
        
        # Group by type
        by_type = defaultdict(list)
        for pattern in patterns:
            by_type[pattern.pattern_type].append(pattern)
        
        # Statistics
        total_support = sum(p.support for p in patterns)
        avg_confidence = np.mean([p.confidence for p in patterns])
        avg_modalities = np.mean([len(p.modalities) for p in patterns])
        
        return {
            'total_patterns': len(patterns),
            'patterns_by_type': {k: len(v) for k, v in by_type.items()},
            'total_support': total_support,
            'avg_confidence': float(avg_confidence),
            'avg_modalities': float(avg_modalities),
            'top_patterns': sorted(
                patterns,
                key=lambda p: p.support,
                reverse=True
            )[:10]
        }
