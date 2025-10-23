"""
Matching Manifold
=================

Universal framework for similarity-based pairing and matching.

Applies to:
- Healthcare: patient-trial matching, organ donor matching
- HR: candidate-job matching
- Education: student-course matching
- Dating: profile matching
- Rideshare: driver-passenger matching

Key insights:
- Entities as manifold points
- Metric = compatibility/similarity
- Geodesic distance = match quality
- Curvature = preference heterogeneity
- Clusters = natural groupings
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import numpy as np
    MATCHING_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATCHING_AVAILABLE = False

from ..core.property_manifold import PropertyManifold

logger = logging.getLogger(__name__)


@dataclass
class MatchQuality:
    """Match compatibility score"""
    entity_a: str
    entity_b: str
    geodesic_distance: float
    compatibility_score: float
    match_rank: int


@dataclass
class MatchingCluster:
    """Group of similar entities"""
    cluster_id: int
    members: List[str]
    centroid_properties: Dict[str, float]
    cohesion: float


class MatchingManifold:
    """
    Similarity-based pairing via manifold geometry.
    
    Models entities as points on a manifold where geodesic
    distance measures true compatibility and clusters reveal
    natural groupings.
    
    Examples:
        >>> # Healthcare: patient-trial matching
        >>> manifold = MatchingManifold(
        ...     set_a=patients,
        ...     set_b=clinical_trials,
        ... )
        >>> matches = manifold.find_optimal_matches(trial_id="NCT123")
        >>> 
        >>> # HR: candidate-job matching
        >>> manifold = MatchingManifold(
        ...     set_a=candidates,
        ...     set_b=job_postings,
        ... )
        >>> top_candidates = manifold.rank_matches(job_id="JOB-456")
    """
    
    def __init__(
        self,
        set_a: Dict[str, Dict[str, float]],
        set_b: Optional[Dict[str, Dict[str, float]]] = None,
        weights: Optional[Dict[str, float]] = None,
        bidirectional: bool = True,
    ) -> None:
        if not MATCHING_AVAILABLE:
            raise RuntimeError("NumPy required for MatchingManifold")
        if not set_a:
            raise ValueError("set_a must not be empty")
        
        self.set_a = set_a
        self.set_b = set_b if set_b is not None else set_a
        self.bidirectional = bidirectional
        
        # Combine both sets for unified manifold
        combined = {**set_a, **self.set_b}
        
        self.manifold = PropertyManifold(
            property_vectors=combined,
            property_weights=weights,
        )
        
        self.entities_a = list(set_a.keys())
        self.entities_b = list(self.set_b.keys())
        
        logger.info(
            "MatchingManifold initialized with %d (A) and %d (B) entities",
            len(self.entities_a),
            len(self.entities_b),
        )
    
    # ------------------------------------------------------------------
    # Matching & ranking
    # ------------------------------------------------------------------
    def find_optimal_matches(
        self,
        entity_id: str,
        from_set: str = "a",
        top_k: int = 5,
    ) -> List[MatchQuality]:
        """Find best matches for an entity."""
        target_set = self.entities_b if from_set == "a" else self.entities_a
        
        matches: List[Tuple[str, float]] = []
        for candidate in target_set:
            if candidate == entity_id:
                continue
            
            distance = self.manifold.property_distance(entity_id, candidate)
            matches.append((candidate, distance))
        
        # Sort by distance (lower = better match)
        matches.sort(key=lambda x: x[1])
        
        # Build match quality objects
        results: List[MatchQuality] = []
        for rank, (match_id, distance) in enumerate(matches[:top_k], start=1):
            # Compatibility: inverse of distance, normalized to 0-1
            max_dist = matches[-1][1] if matches else 1.0
            compatibility = 1.0 - (distance / (max_dist + 1e-12))
            
            results.append(
                MatchQuality(
                    entity_a=entity_id,
                    entity_b=match_id,
                    geodesic_distance=distance,
                    compatibility_score=compatibility,
                    match_rank=rank,
                )
            )
        
        return results
    
    def rank_matches(
        self,
        entity_id: str,
        candidates: Optional[List[str]] = None,
    ) -> List[MatchQuality]:
        """Rank specific candidates for an entity."""
        if candidates is None:
            return self.find_optimal_matches(entity_id)
        
        matches: List[Tuple[str, float]] = []
        for candidate in candidates:
            if candidate == entity_id:
                continue
            distance = self.manifold.property_distance(entity_id, candidate)
            matches.append((candidate, distance))
        
        matches.sort(key=lambda x: x[1])
        
        results: List[MatchQuality] = []
        max_dist = matches[-1][1] if matches else 1.0
        for rank, (match_id, distance) in enumerate(matches, start=1):
            compatibility = 1.0 - (distance / (max_dist + 1e-12))
            results.append(
                MatchQuality(
                    entity_a=entity_id,
                    entity_b=match_id,
                    geodesic_distance=distance,
                    compatibility_score=compatibility,
                    match_rank=rank,
                )
            )
        
        return results
    
    # ------------------------------------------------------------------
    # Clustering for group discovery
    # ------------------------------------------------------------------
    def discover_natural_groups(
        self,
        num_clusters: int = 5,
    ) -> List[MatchingCluster]:
        """Discover natural entity groupings via curvature."""
        # Use property manifold's outlier detection as proxy for clustering
        all_entities = self.entities_a + self.entities_b
        
        # Simple k-means-style clustering using geodesic distances
        # Pick initial centroids
        np.random.seed(42)
        centroids = np.random.choice(all_entities, size=min(num_clusters, len(all_entities)), replace=False).tolist()
        
        clusters: Dict[int, List[str]] = {i: [] for i in range(len(centroids))}
        
        # Assign entities to nearest centroid
        for entity in all_entities:
            distances = [
                self.manifold.property_distance(entity, centroid)
                for centroid in centroids
            ]
            nearest = int(np.argmin(distances))
            clusters[nearest].append(entity)
        
        # Build cluster objects
        results: List[MatchingCluster] = []
        for cluster_id, members in clusters.items():
            if not members:
                continue
            
            # Compute centroid properties (average)
            centroid_props = self._compute_centroid(members)
            
            # Cohesion: average intra-cluster distance
            cohesion = self._compute_cohesion(members)
            
            results.append(
                MatchingCluster(
                    cluster_id=cluster_id,
                    members=members,
                    centroid_properties=centroid_props,
                    cohesion=cohesion,
                )
            )
        
        return results
    
    def _compute_centroid(self, members: List[str]) -> Dict[str, float]:
        """Compute average properties for cluster members."""
        if not members:
            return {}
        
        all_props: Dict[str, List[float]] = {}
        for member in members:
            props = self.manifold.property_vectors.get(member, {})
            for prop, value in props.items():
                if prop not in all_props:
                    all_props[prop] = []
                all_props[prop].append(value)
        
        centroid = {prop: float(np.mean(values)) for prop, values in all_props.items()}
        return centroid
    
    def _compute_cohesion(self, members: List[str]) -> float:
        """Compute average intra-cluster distance."""
        if len(members) < 2:
            return 0.0
        
        distances: List[float] = []
        for i, member_a in enumerate(members):
            for member_b in members[i+1:]:
                dist = self.manifold.property_distance(member_a, member_b)
                distances.append(dist)
        
        return float(np.mean(distances)) if distances else 0.0
    
    # ------------------------------------------------------------------
    # Similarity analysis
    # ------------------------------------------------------------------
    def compute_similarity_matrix(
        self,
        entities: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Compute pairwise similarity matrix."""
        if entities is None:
            entities = self.entities_a[:min(50, len(self.entities_a))]  # Limit size
        
        n = len(entities)
        matrix = np.zeros((n, n), dtype=float)
        
        for i, entity_a in enumerate(entities):
            for j, entity_b in enumerate(entities):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    dist = self.manifold.property_distance(entity_a, entity_b)
                    similarity = 1.0 / (1.0 + dist)
                    matrix[i, j] = similarity
        
        return matrix
    
    def summary(self) -> Dict[str, int]:
        """Summary statistics."""
        return {
            "entities_a": len(self.entities_a),
            "entities_b": len(self.entities_b),
            "properties": len(self.manifold.property_list),
        }


def find_best_matches(
    entity_id: str,
    entities: Dict[str, Dict[str, float]],
    candidates: Dict[str, Dict[str, float]],
    top_k: int = 5,
    **kwargs: float,
) -> List[MatchQuality]:
    """Convenience: find best matches for an entity."""
    manifold = MatchingManifold(
        set_a={entity_id: entities[entity_id]},
        set_b=candidates,
        **kwargs,
    )
    return manifold.find_optimal_matches(entity_id, top_k=top_k)
