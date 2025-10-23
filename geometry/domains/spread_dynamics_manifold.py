"""
Spread Dynamics Manifold
========================

Universal framework for contagion/propagation phenomena over networks.

Applies to:
- Epidemics: disease spread through populations
- Social: viral content, influence cascades
- Network: failure propagation, attacks
- Finance: systemic risk contagion
- Biology: trait/gene spread

Key insights:
- Curvature = acceleration of spread
- Geodesics = propagation pathways
- Divergence = outbreak sources/sinks
- Negative curvature = containment zones
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import numpy as np
    SPREAD_AVAILABLE = True
except ImportError:  # pragma: no cover
    SPREAD_AVAILABLE = False

from ..core.riemannian_manifold import RiemannianGraphManifold

logger = logging.getLogger(__name__)


@dataclass
class SpreadHotspot:
    """High-acceleration spread zone"""
    node_id: str
    curvature: float
    spread_rate: float
    severity: str  # 'critical', 'high', 'moderate'


@dataclass
class PropagationPath:
    """Geodesic propagation trajectory"""
    source: str
    path: List[str]
    distance: float
    time_estimate: float


class SpreadDynamicsManifold:
    """
    Universal manifold for contagion/propagation analysis.
    
    Maps any spreading phenomenon to geometric structure where
    curvature reveals acceleration zones and geodesics trace
    optimal propagation paths.
    
    Examples:
        >>> # Epidemic
        >>> manifold = SpreadDynamicsManifold(contact_network, 'infection_rate')
        >>> hotspots = manifold.detect_acceleration_zones()
        >>> 
        >>> # Social virality
        >>> manifold = SpreadDynamicsManifold(social_graph, 'shares')
        >>> influencers = manifold.find_propagation_sources()
    """
    
    def __init__(
        self,
        network: Any,
        spread_attribute: str = "spread_rate",
        time_attribute: Optional[str] = None,
        model: str = "generic",  # 'generic', 'sir', 'sis', 'seir'
    ) -> None:
        if not SPREAD_AVAILABLE:
            raise RuntimeError("NumPy required for SpreadDynamicsManifold")
        
        self.network = network
        self.spread_attribute = spread_attribute
        self.time_attribute = time_attribute
        self.model = model
        
        # Build geometric manifold from network
        self.manifold = RiemannianGraphManifold(
            network,
            property_weights={spread_attribute: 2.0},  # Emphasize spread
            use_properties=True,
        )
        
        self.node_coords = self.manifold.node_coordinates
        self.nodes = list(self.node_coords.keys())
        
        logger.info(
            "SpreadDynamicsManifold initialized with %d nodes (model=%s)",
            len(self.nodes),
            model,
        )
    
    # ------------------------------------------------------------------
    # Spread acceleration detection (curvature)
    # ------------------------------------------------------------------
    def detect_acceleration_zones(
        self,
        threshold: float = 2.0,
    ) -> List[SpreadHotspot]:
        """Find zones where spread is accelerating (high curvature)."""
        curvature_field = self.manifold.compute_curvature_field()
        
        hotspots: List[SpreadHotspot] = []
        curvatures = [c.scalar_curvature for c in curvature_field.values()]
        mean_curv = float(np.mean(curvatures))
        std_curv = float(np.std(curvatures) + 1e-12)
        
        for node_id, curv_data in curvature_field.items():
            z_score = abs(curv_data.scalar_curvature - mean_curv) / std_curv
            if z_score > threshold:
                severity = 'critical' if z_score > 3 else ('high' if z_score > 2 else 'moderate')
                spread_rate = self._get_spread_rate(node_id)
                
                hotspots.append(
                    SpreadHotspot(
                        node_id=node_id,
                        curvature=curv_data.scalar_curvature,
                        spread_rate=spread_rate,
                        severity=severity,
                    )
                )
        
        hotspots.sort(key=lambda h: h.curvature, reverse=True)
        return hotspots
    
    def find_propagation_sources(self, top_k: int = 5) -> List[str]:
        """Identify likely propagation origin points (high positive curvature)."""
        hotspots = self.detect_acceleration_zones()
        sources = [h.node_id for h in hotspots if h.curvature > 0]
        return sources[:top_k]
    
    def find_containment_zones(self, top_k: int = 5) -> List[str]:
        """Identify natural containment areas (negative curvature)."""
        clusters = self.manifold.detect_natural_clusters()
        containment = [c.node_id for c in clusters]
        return containment[:top_k]
    
    # ------------------------------------------------------------------
    # Propagation path analysis (geodesics)
    # ------------------------------------------------------------------
    def compute_propagation_path(
        self,
        source: str,
        target: str,
    ) -> PropagationPath:
        """Compute optimal propagation path from source to target."""
        distance = self.manifold.geodesic_distance(source, target)
        
        # Estimate propagation time if temporal data available
        time_estimate = distance
        if self.time_attribute:
            source_rate = self._get_spread_rate(source)
            if source_rate > 0:
                time_estimate = distance / source_rate
        
        # For now, simple path is [source, target]; full geodesic integration TBD
        path = [source, target]
        
        return PropagationPath(
            source=source,
            path=path,
            distance=distance,
            time_estimate=time_estimate,
        )
    
    def predict_spread_reach(
        self,
        source: str,
        max_distance: float = 1.0,
    ) -> List[Tuple[str, float]]:
        """Predict which nodes will be reached from source within distance."""
        reachable: List[Tuple[str, float]] = []
        
        for node in self.nodes:
            if node == source:
                continue
            dist = self.manifold.geodesic_distance(source, node)
            if dist <= max_distance:
                reachable.append((node, dist))
        
        reachable.sort(key=lambda x: x[1])
        return reachable
    
    # ------------------------------------------------------------------
    # Model-specific dynamics
    # ------------------------------------------------------------------
    def estimate_r0(self) -> float:
        """Estimate basic reproduction number from network geometry."""
        # R0 approximation: mean spread rate * mean degree
        spread_rates = [self._get_spread_rate(n) for n in self.nodes]
        mean_rate = float(np.mean(spread_rates))
        
        # Approximate degree from manifold curvature
        curvatures = [
            self.manifold.compute_curvature_at_node(n).scalar_curvature
            for n in self.nodes
        ]
        mean_curv = abs(np.mean(curvatures))
        
        r0 = mean_rate * (1 + mean_curv)
        return float(r0)
    
    def predict_peak_time(self) -> Optional[float]:
        """Estimate time to epidemic peak using geometric properties."""
        r0 = self.estimate_r0()
        if r0 <= 1:
            return None  # No epidemic
        
        # Simplified peak time: inversely proportional to R0-1
        peak_time = 1.0 / (r0 - 1)
        return float(peak_time)
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_spread_rate(self, node_id: str) -> float:
        """Extract spread rate for a node."""
        if hasattr(self.network, 'nodes'):
            node_data = dict(self.network.nodes(data=True)).get(node_id, {})  # type: ignore
            if isinstance(node_data, dict):
                return float(node_data.get(self.spread_attribute, 1.0))
        return 1.0
    
    def summary(self) -> Dict[str, Any]:
        """Summary statistics."""
        hotspots = self.detect_acceleration_zones()
        r0 = self.estimate_r0()
        
        return {
            "nodes": len(self.nodes),
            "model": self.model,
            "hotspots": len(hotspots),
            "estimated_r0": r0,
            "max_curvature": hotspots[0].curvature if hotspots else 0.0,
        }


def detect_spread_hotspots(
    network: Any,
    spread_attribute: str = "spread_rate",
    top_k: int = 5,
    **kwargs: Any,
) -> List[SpreadHotspot]:
    """Convenience: find top acceleration zones."""
    manifold = SpreadDynamicsManifold(
        network=network,
        spread_attribute=spread_attribute,
        **kwargs,
    )
    return manifold.detect_acceleration_zones()[:top_k]
