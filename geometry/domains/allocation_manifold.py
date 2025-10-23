"""
Allocation Manifold
===================

Universal framework for multi-resource optimization and allocation.

Applies to:
- Healthcare: beds, nurses, equipment, budget
- Cloud: CPU, memory, bandwidth, storage
- Logistics: fleet, warehouses, inventory
- Energy: power grid distribution
- Finance: portfolio allocation

Key insights:
- Coordinates = resource quantities
- Metric = scarcity/cost weights
- Curvature = allocation stress/imbalance
- Geodesics = optimal reallocation paths
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
    ALLOCATION_AVAILABLE = True
except ImportError:  # pragma: no cover
    ALLOCATION_AVAILABLE = False

from ..core.property_manifold import PropertyManifold

logger = logging.getLogger(__name__)


@dataclass
class AllocationStress:
    """Resource allocation stress point"""
    entity_id: str
    stress_level: float
    constrained_resources: List[str]
    severity: str  # 'critical', 'high', 'moderate'


@dataclass
class ReallocationPath:
    """Optimal resource transfer geodesic"""
    from_entity: str
    to_entity: str
    resources: Dict[str, float]
    geodesic_cost: float


class AllocationManifold:
    """
    Multi-resource allocation optimization via manifold geometry.
    
    Treats resource allocation as points on a manifold where
    curvature reveals stress/imbalance and geodesics provide
    optimal reallocation paths.
    
    Examples:
        >>> # Healthcare
        >>> manifold = AllocationManifold(
        ...     allocations={'ICU': {'beds': 10, 'nurses': 15}},
        ...     capacities={'beds': 50, 'nurses': 100},
        ... )
        >>> stress = manifold.detect_stress_points()
        >>> 
        >>> # Cloud
        >>> manifold = AllocationManifold(
        ...     allocations={'pod-1': {'cpu': 8, 'memory': 16}},
        ...     capacities={'cpu': 64, 'memory': 128},
        ... )
        >>> bottlenecks = manifold.find_constrained_resources()
    """
    
    def __init__(
        self,
        allocations: Dict[str, Dict[str, float]],
        capacities: Optional[Dict[str, float]] = None,
        costs: Optional[Dict[str, float]] = None,
        epsilon: float = 1e-9,
    ) -> None:
        if not ALLOCATION_AVAILABLE:
            raise RuntimeError("NumPy required for AllocationManifold")
        if not allocations:
            raise ValueError("allocations must not be empty")
        
        self.allocations = allocations
        self.capacities = capacities or {}
        self.costs = costs or {}
        self.epsilon = epsilon
        
        # Build property manifold from allocation vectors
        self.manifold = PropertyManifold(
            property_vectors=allocations,
            property_weights=self._compute_scarcity_weights(),
            epsilon=epsilon,
        )
        
        self.entities = list(allocations.keys())
        self.resources = self.manifold.property_list
        
        logger.info(
            "AllocationManifold initialized with %d entities and %d resources",
            len(self.entities),
            len(self.resources),
        )
    
    # ------------------------------------------------------------------
    # Scarcity-aware metric
    # ------------------------------------------------------------------
    def _compute_scarcity_weights(self) -> Dict[str, float]:
        """Compute resource weights based on scarcity."""
        weights: Dict[str, float] = {}
        
        for resource in self._get_all_resources():
            total_allocated = sum(
                alloc.get(resource, 0.0)
                for alloc in self.allocations.values()
            )
            capacity = self.capacities.get(resource, float('inf'))
            
            # Scarcity = allocated / capacity
            if capacity > 0 and capacity != float('inf'):
                scarcity = total_allocated / capacity
                weights[resource] = 1.0 + scarcity  # Higher weight for scarce
            else:
                weights[resource] = 1.0
        
        return weights
    
    def _get_all_resources(self) -> List[str]:
        """Get all unique resource names."""
        resources = set()
        for alloc in self.allocations.values():
            resources.update(alloc.keys())
        if self.capacities:
            resources.update(self.capacities.keys())
        return sorted(resources)
    
    # ------------------------------------------------------------------
    # Stress detection (curvature)
    # ------------------------------------------------------------------
    def detect_stress_points(
        self,
        threshold: float = 2.5,
    ) -> List[AllocationStress]:
        """Find entities under allocation stress (high curvature)."""
        outliers = self.manifold.detect_property_outliers(z_threshold=threshold)
        curvature_data = self.manifold.compute_curvature()
        
        stress_points: List[AllocationStress] = []
        
        for entity_id in outliers:
            alloc = self.allocations[entity_id]
            constrained = self._find_constrained_resources(alloc)
            
            # Estimate stress level from allocation vs capacity
            stress = self._compute_stress_level(alloc)
            severity = 'critical' if stress > 0.9 else ('high' if stress > 0.7 else 'moderate')
            
            stress_points.append(
                AllocationStress(
                    entity_id=entity_id,
                    stress_level=stress,
                    constrained_resources=constrained,
                    severity=severity,
                )
            )
        
        stress_points.sort(key=lambda s: s.stress_level, reverse=True)
        return stress_points
    
    def _find_constrained_resources(self, allocation: Dict[str, float]) -> List[str]:
        """Identify which resources are constrained."""
        constrained: List[str] = []
        
        for resource, amount in allocation.items():
            capacity = self.capacities.get(resource)
            if capacity and amount / capacity > 0.8:  # 80% utilization
                constrained.append(resource)
        
        return constrained
    
    def _compute_stress_level(self, allocation: Dict[str, float]) -> float:
        """Compute overall stress level for an allocation."""
        if not self.capacities:
            return 0.0
        
        utilizations: List[float] = []
        for resource, amount in allocation.items():
            capacity = self.capacities.get(resource, float('inf'))
            if capacity > 0 and capacity != float('inf'):
                utilizations.append(amount / capacity)
        
        if not utilizations:
            return 0.0
        
        return float(np.max(utilizations))
    
    # ------------------------------------------------------------------
    # Reallocation (geodesics)
    # ------------------------------------------------------------------
    def compute_reallocation_path(
        self,
        from_entity: str,
        to_entity: str,
        resources: Optional[Dict[str, float]] = None,
    ) -> ReallocationPath:
        """Compute optimal resource reallocation geodesic."""
        geodesic_cost = self.manifold.property_distance(from_entity, to_entity)
        
        # If no specific resources, use small fraction of from_entity allocation
        if resources is None:
            from_alloc = self.allocations[from_entity]
            resources = {r: 0.1 * v for r, v in from_alloc.items()}
        
        # Adjust cost by resource weights
        weighted_cost = geodesic_cost
        for resource, amount in resources.items():
            weight = self.manifold.property_weights.get(resource, 1.0)
            weighted_cost += weight * amount * 0.01
        
        return ReallocationPath(
            from_entity=from_entity,
            to_entity=to_entity,
            resources=resources,
            geodesic_cost=weighted_cost,
        )
    
    def suggest_optimal_transfers(
        self,
        stressed_entity: str,
        donor_pool: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[ReallocationPath]:
        """Suggest optimal resource transfers to relieve stress."""
        if donor_pool is None:
            donor_pool = [e for e in self.entities if e != stressed_entity]
        
        # Find closest donors by geodesic distance
        transfers: List[Tuple[str, float]] = []
        for donor in donor_pool:
            dist = self.manifold.property_distance(donor, stressed_entity)
            transfers.append((donor, dist))
        
        transfers.sort(key=lambda x: x[1])
        
        # Build reallocation paths
        paths: List[ReallocationPath] = []
        for donor, _ in transfers[:top_k]:
            path = self.compute_reallocation_path(donor, stressed_entity)
            paths.append(path)
        
        return paths
    
    # ------------------------------------------------------------------
    # Resource analysis
    # ------------------------------------------------------------------
    def find_constrained_resources(self) -> List[Tuple[str, float]]:
        """Identify globally constrained resources."""
        constrained: List[Tuple[str, float]] = []
        
        for resource in self.resources:
            total = sum(alloc.get(resource, 0.0) for alloc in self.allocations.values())
            capacity = self.capacities.get(resource, float('inf'))
            
            if capacity > 0 and capacity != float('inf'):
                utilization = total / capacity
                if utilization > 0.7:
                    constrained.append((resource, utilization))
        
        constrained.sort(key=lambda x: x[1], reverse=True)
        return constrained
    
    def compute_allocation_efficiency(self) -> float:
        """Measure overall allocation efficiency (inverse of curvature variance)."""
        curvature = self.manifold.compute_curvature()
        # Lower curvature variance = more balanced allocation
        metric = self.manifold.compute_metric()
        eigenvalues = metric.eigenvalues
        efficiency = 1.0 / (1.0 + float(np.var(eigenvalues)))
        return efficiency
    
    def summary(self) -> Dict[str, float]:
        """Summary statistics."""
        stress_points = self.detect_stress_points()
        efficiency = self.compute_allocation_efficiency()
        constrained = self.find_constrained_resources()
        
        return {
            "entities": len(self.entities),
            "resources": len(self.resources),
            "stress_points": len(stress_points),
            "efficiency": efficiency,
            "constrained_resources": len(constrained),
        }


def find_allocation_stress(
    allocations: Dict[str, Dict[str, float]],
    capacities: Optional[Dict[str, float]] = None,
    top_k: int = 5,
    **kwargs: float,
) -> List[AllocationStress]:
    """Convenience: find top stressed entities."""
    manifold = AllocationManifold(
        allocations=allocations,
        capacities=capacities,
        **kwargs,
    )
    return manifold.detect_stress_points()[:top_k]
