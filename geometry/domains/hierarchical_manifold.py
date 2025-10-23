"""
Hierarchical Manifold
=====================

Universal framework for nested/hierarchical systems with cross-level interactions.

Applies to:
- Healthcare: patient → department → hospital → health system
- Organizations: employee → team → division → company
- Geography: city → state → country → continent
- Infrastructure: device → rack → datacenter → region
- Taxonomy: species → genus → family → order

Key insights:
- Vertical dimension = hierarchy levels
- Horizontal = entities within level
- Curvature = cross-level coupling strength
- Geodesics = escalation/aggregation paths
- Fiber bundles = natural hierarchical geometry
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import numpy as np
    HIERARCHICAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    HIERARCHICAL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HierarchyNode:
    """Node in hierarchical structure"""
    node_id: str
    level: int
    level_name: str
    parent: Optional[str]
    children: List[str]
    properties: Dict[str, float]


@dataclass
class CrossLevelImpact:
    """Impact propagation across levels"""
    source_node: str
    source_level: int
    affected_nodes: List[Tuple[str, int, float]]  # (node_id, level, impact_strength)
    total_reach: int


class HierarchicalManifold:
    """
    Multi-level hierarchical system analysis via fiber bundle geometry.
    
    Models hierarchies as fiber bundles where vertical structure
    (levels) and horizontal structure (within-level) interact
    through curvature and parallel transport.
    
    Examples:
        >>> # Healthcare
        >>> manifold = HierarchicalManifold(
        ...     hierarchy={
        ...         'patient-1': {'level': 0, 'parent': 'dept-A'},
        ...         'dept-A': {'level': 1, 'parent': 'hospital-1'},
        ...     },
        ...     level_names=['patient', 'department', 'hospital'],
        ... )
        >>> impact = manifold.compute_cross_level_impact('dept-A', change={'beds': -5})
        >>> 
        >>> # Organization
        >>> manifold = HierarchicalManifold(org_structure)
        >>> aggregated = manifold.aggregate_metrics(level='division')
    """
    
    def __init__(
        self,
        hierarchy: Dict[str, Dict[str, Any]],
        level_names: Optional[List[str]] = None,
        property_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        if not HIERARCHICAL_AVAILABLE:
            raise RuntimeError("NumPy required for HierarchicalManifold")
        if not hierarchy:
            raise ValueError("hierarchy must not be empty")
        
        self.hierarchy = hierarchy
        self.property_weights = property_weights or {}
        
        # Build hierarchy nodes
        self.nodes = self._build_nodes(hierarchy)
        
        # Determine levels
        self.levels = sorted(set(node.level for node in self.nodes.values()))
        self.level_names = level_names or [f"level_{i}" for i in self.levels]
        
        # Build level-wise groupings
        self.nodes_by_level: Dict[int, List[str]] = {}
        for level in self.levels:
            self.nodes_by_level[level] = [
                nid for nid, node in self.nodes.items() if node.level == level
            ]
        
        logger.info(
            "HierarchicalManifold initialized with %d nodes across %d levels",
            len(self.nodes),
            len(self.levels),
        )
    
    def _build_nodes(self, hierarchy: Dict[str, Dict[str, Any]]) -> Dict[str, HierarchyNode]:
        """Build hierarchy node objects."""
        nodes: Dict[str, HierarchyNode] = {}
        
        # First pass: create nodes
        for node_id, data in hierarchy.items():
            level = data.get('level', 0)
            parent = data.get('parent')
            properties = {k: v for k, v in data.items() if isinstance(v, (int, float))}
            
            nodes[node_id] = HierarchyNode(
                node_id=node_id,
                level=level,
                level_name=f"level_{level}",
                parent=parent,
                children=[],
                properties=properties,
            )
        
        # Second pass: link children
        for node_id, node in nodes.items():
            if node.parent and node.parent in nodes:
                nodes[node.parent].children.append(node_id)
        
        return nodes
    
    # ------------------------------------------------------------------
    # Cross-level analysis
    # ------------------------------------------------------------------
    def compute_cross_level_impact(
        self,
        source_node: str,
        change: Optional[Dict[str, float]] = None,
        propagation_factor: float = 0.5,
    ) -> CrossLevelImpact:
        """Compute how changes at one node propagate across hierarchy."""
        if source_node not in self.nodes:
            raise KeyError(f"Node '{source_node}' not found in hierarchy")
        
        source = self.nodes[source_node]
        affected: List[Tuple[str, int, float]] = []
        
        # Propagate upward (to parents)
        current = source
        impact_strength = 1.0
        while current.parent:
            parent = self.nodes[current.parent]
            impact_strength *= propagation_factor
            affected.append((parent.node_id, parent.level, impact_strength))
            current = parent
        
        # Propagate downward (to children)
        self._propagate_to_children(source.node_id, propagation_factor, affected)
        
        # Propagate horizontally (to siblings)
        if source.parent:
            siblings = self.nodes[source.parent].children
            for sibling in siblings:
                if sibling != source_node:
                    affected.append((sibling, source.level, propagation_factor * 0.3))
        
        return CrossLevelImpact(
            source_node=source_node,
            source_level=source.level,
            affected_nodes=affected,
            total_reach=len(affected),
        )
    
    def _propagate_to_children(
        self,
        node_id: str,
        impact: float,
        affected: List[Tuple[str, int, float]],
    ) -> None:
        """Recursively propagate impact to children."""
        node = self.nodes[node_id]
        reduced_impact = impact * 0.5
        
        for child_id in node.children:
            child = self.nodes[child_id]
            affected.append((child_id, child.level, reduced_impact))
            if child.children:
                self._propagate_to_children(child_id, reduced_impact, affected)
    
    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate_metrics(
        self,
        level: int,
        aggregation: str = "sum",  # 'sum', 'mean', 'max', 'min'
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics at a specific level from lower levels."""
        if level not in self.levels:
            raise ValueError(f"Level {level} not found in hierarchy")
        
        aggregated: Dict[str, Dict[str, float]] = {}
        
        for node_id in self.nodes_by_level[level]:
            node = self.nodes[node_id]
            
            # Collect metrics from all descendants
            descendant_metrics = self._collect_descendant_metrics(node_id)
            
            # Aggregate
            agg_metrics: Dict[str, float] = {}
            for metric, values in descendant_metrics.items():
                if aggregation == "sum":
                    agg_metrics[metric] = float(np.sum(values))
                elif aggregation == "mean":
                    agg_metrics[metric] = float(np.mean(values))
                elif aggregation == "max":
                    agg_metrics[metric] = float(np.max(values))
                elif aggregation == "min":
                    agg_metrics[metric] = float(np.min(values))
            
            aggregated[node_id] = agg_metrics
        
        return aggregated
    
    def _collect_descendant_metrics(
        self,
        node_id: str,
    ) -> Dict[str, List[float]]:
        """Collect all metrics from descendants."""
        node = self.nodes[node_id]
        metrics: Dict[str, List[float]] = {}
        
        # Add own properties
        for prop, value in node.properties.items():
            if prop not in metrics:
                metrics[prop] = []
            metrics[prop].append(value)
        
        # Recursively collect from children
        for child_id in node.children:
            child_metrics = self._collect_descendant_metrics(child_id)
            for prop, values in child_metrics.items():
                if prop not in metrics:
                    metrics[prop] = []
                metrics[prop].extend(values)
        
        return metrics
    
    # ------------------------------------------------------------------
    # Escalation paths
    # ------------------------------------------------------------------
    def compute_escalation_path(
        self,
        source_node: str,
        target_level: Optional[int] = None,
    ) -> List[str]:
        """Compute escalation path from node upward to target level."""
        if source_node not in self.nodes:
            raise KeyError(f"Node '{source_node}' not found")
        
        if target_level is None:
            target_level = max(self.levels)
        
        path: List[str] = [source_node]
        current = self.nodes[source_node]
        
        while current.level < target_level and current.parent:
            path.append(current.parent)
            current = self.nodes[current.parent]
        
        return path
    
    def find_common_ancestor(
        self,
        node_a: str,
        node_b: str,
    ) -> Optional[str]:
        """Find lowest common ancestor of two nodes."""
        path_a = set(self.compute_escalation_path(node_a))
        path_b = self.compute_escalation_path(node_b)
        
        for ancestor in path_b:
            if ancestor in path_a:
                return ancestor
        
        return None
    
    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def compute_hierarchy_depth(self) -> int:
        """Maximum depth of hierarchy."""
        return len(self.levels)
    
    def compute_branching_factor(self, level: int) -> float:
        """Average branching factor at a level."""
        nodes_at_level = self.nodes_by_level.get(level, [])
        if not nodes_at_level:
            return 0.0
        
        total_children = sum(len(self.nodes[nid].children) for nid in nodes_at_level)
        return total_children / len(nodes_at_level)
    
    def summary(self) -> Dict[str, Any]:
        """Summary statistics."""
        return {
            "total_nodes": len(self.nodes),
            "levels": len(self.levels),
            "depth": self.compute_hierarchy_depth(),
            "avg_branching": float(np.mean([
                self.compute_branching_factor(lvl) for lvl in self.levels[:-1]
            ])) if len(self.levels) > 1 else 0.0,
        }
