"""
Process Manifold
================

Universal framework for workflow and pipeline analysis.

Applies to:
- Manufacturing: production pipelines
- Healthcare: patient care pathways
- Software: CI/CD workflows
- Business: approval processes
- Supply chain: order fulfillment

Key insights:
- Nodes = process steps/states
- Edges = state transitions
- Curvature = process complexity/friction
- Geodesics = optimal workflow paths
- Bottlenecks = high transition curvature
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
    PROCESS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROCESS_AVAILABLE = False

from ..core.riemannian_manifold import RiemannianGraphManifold

logger = logging.getLogger(__name__)


@dataclass
class ProcessBottleneck:
    """High-friction process step"""
    step_id: str
    curvature: float
    avg_duration: float
    failure_rate: float
    severity: str  # 'critical', 'high', 'moderate'


@dataclass
class OptimalWorkflow:
    """Streamlined process geodesic"""
    path: List[str]
    total_cost: float
    estimated_duration: float
    efficiency_gain: float


class ProcessManifold:
    """
    Workflow/pipeline analysis via manifold geometry.
    
    Models processes as trajectories on a manifold where
    curvature reveals friction/complexity and geodesics
    show optimal streamlined paths.
    
    Examples:
        >>> # Manufacturing
        >>> manifold = ProcessManifold(production_workflow)
        >>> bottlenecks = manifold.find_bottlenecks()
        >>> optimal = manifold.discover_streamlined_path(start, end)
        >>> 
        >>> # Software CI/CD
        >>> manifold = ProcessManifold(deployment_pipeline)
        >>> failures = manifold.detect_failure_prone_steps()
    """
    
    def __init__(
        self,
        workflow: Any,
        duration_attribute: str = "duration",
        failure_attribute: str = "failure_rate",
        cost_attribute: str = "cost",
    ) -> None:
        if not PROCESS_AVAILABLE:
            raise RuntimeError("NumPy required for ProcessManifold")
        
        self.workflow = workflow
        self.duration_attr = duration_attribute
        self.failure_attr = failure_attribute
        self.cost_attr = cost_attribute
        
        # Build manifold emphasizing process metrics
        self.manifold = RiemannianGraphManifold(
            workflow,
            property_weights={
                duration_attribute: 1.5,
                failure_attribute: 2.0,
                cost_attribute: 1.0,
            },
            use_properties=True,
        )
        
        self.steps = list(self.manifold.node_coordinates.keys())
        logger.info(
            "ProcessManifold initialized with %d process steps",
            len(self.steps),
        )
    
    # ------------------------------------------------------------------
    # Bottleneck detection (high curvature)
    # ------------------------------------------------------------------
    def find_bottlenecks(
        self,
        threshold: float = 2.0,
    ) -> List[ProcessBottleneck]:
        """Identify process bottlenecks (high curvature steps)."""
        curvature_field = self.manifold.compute_curvature_field()
        
        bottlenecks: List[ProcessBottleneck] = []
        curvatures = [c.scalar_curvature for c in curvature_field.values()]
        mean_curv = float(np.mean(curvatures))
        std_curv = float(np.std(curvatures) + 1e-12)
        
        for step_id, curv_data in curvature_field.items():
            z_score = abs(curv_data.scalar_curvature - mean_curv) / std_curv
            if z_score > threshold:
                severity = 'critical' if z_score > 3 else ('high' if z_score > 2 else 'moderate')
                duration = self._get_step_duration(step_id)
                failure_rate = self._get_step_failure_rate(step_id)
                
                bottlenecks.append(
                    ProcessBottleneck(
                        step_id=step_id,
                        curvature=curv_data.scalar_curvature,
                        avg_duration=duration,
                        failure_rate=failure_rate,
                        severity=severity,
                    )
                )
        
        bottlenecks.sort(key=lambda b: b.curvature, reverse=True)
        return bottlenecks
    
    def detect_failure_prone_steps(self, top_k: int = 5) -> List[str]:
        """Identify steps with highest failure rates."""
        failures: List[Tuple[str, float]] = []
        
        for step in self.steps:
            rate = self._get_step_failure_rate(step)
            failures.append((step, rate))
        
        failures.sort(key=lambda x: x[1], reverse=True)
        return [step for step, _ in failures[:top_k]]
    
    # ------------------------------------------------------------------
    # Workflow optimization (geodesics)
    # ------------------------------------------------------------------
    def discover_streamlined_path(
        self,
        start: str,
        end: str,
    ) -> OptimalWorkflow:
        """Find optimal workflow path (geodesic)."""
        # Geodesic distance in manifold
        geodesic_dist = self.manifold.geodesic_distance(start, end)
        
        # Estimate duration and cost
        duration = self._estimate_path_duration(start, end)
        
        # For now, simple path; full geodesic integration TBD
        path = [start, end]
        
        # Compare to current average path cost
        baseline_duration = self._estimate_baseline_duration(start, end)
        efficiency_gain = max(0, (baseline_duration - duration) / baseline_duration)
        
        return OptimalWorkflow(
            path=path,
            total_cost=geodesic_dist,
            estimated_duration=duration,
            efficiency_gain=efficiency_gain,
        )
    
    def suggest_process_improvements(
        self,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Suggest improvements for high-curvature steps."""
        bottlenecks = self.find_bottlenecks()
        
        improvements: List[Dict[str, Any]] = []
        for bottleneck in bottlenecks[:top_k]:
            # Find alternative paths that bypass this step
            alternatives = self._find_bypass_alternatives(bottleneck.step_id)
            
            improvements.append({
                "step": bottleneck.step_id,
                "issue": f"High friction (curvature={bottleneck.curvature:.2f})",
                "alternatives": alternatives,
                "potential_savings": bottleneck.avg_duration * 0.5,
            })
        
        return improvements
    
    def _find_bypass_alternatives(self, step_id: str) -> List[str]:
        """Find steps that could replace the given step."""
        # Find steps with similar function but lower curvature
        target_curv = self.manifold.compute_curvature_at_node(step_id).scalar_curvature
        
        alternatives: List[Tuple[str, float]] = []
        for candidate in self.steps:
            if candidate == step_id:
                continue
            
            cand_curv = self.manifold.compute_curvature_at_node(candidate).scalar_curvature
            if cand_curv < target_curv * 0.7:  # 30% improvement
                dist = self.manifold.geodesic_distance(step_id, candidate)
                alternatives.append((candidate, dist))
        
        alternatives.sort(key=lambda x: x[1])
        return [alt for alt, _ in alternatives[:3]]
    
    # ------------------------------------------------------------------
    # Duration estimation
    # ------------------------------------------------------------------
    def _estimate_path_duration(self, start: str, end: str) -> float:
        """Estimate duration for path."""
        start_dur = self._get_step_duration(start)
        end_dur = self._get_step_duration(end)
        return start_dur + end_dur
    
    def _estimate_baseline_duration(self, start: str, end: str) -> float:
        """Estimate typical duration for this type of path."""
        durations = [self._get_step_duration(s) for s in self.steps]
        return float(np.mean(durations)) * 2
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_step_duration(self, step_id: str) -> float:
        """Extract step duration."""
        if hasattr(self.workflow, 'nodes'):
            node_data = dict(self.workflow.nodes(data=True)).get(step_id, {})  # type: ignore
            if isinstance(node_data, dict):
                return float(node_data.get(self.duration_attr, 1.0))
        return 1.0
    
    def _get_step_failure_rate(self, step_id: str) -> float:
        """Extract step failure rate."""
        if hasattr(self.workflow, 'nodes'):
            node_data = dict(self.workflow.nodes(data=True)).get(step_id, {})  # type: ignore
            if isinstance(node_data, dict):
                return float(node_data.get(self.failure_attr, 0.0))
        return 0.0
    
    def summary(self) -> Dict[str, Any]:
        """Summary statistics."""
        bottlenecks = self.find_bottlenecks()
        failures = self.detect_failure_prone_steps()
        
        return {
            "steps": len(self.steps),
            "bottlenecks": len(bottlenecks),
            "failure_prone_steps": len(failures),
            "max_curvature": bottlenecks[0].curvature if bottlenecks else 0.0,
        }


def find_workflow_bottlenecks(
    workflow: Any,
    top_k: int = 5,
    **kwargs: Any,
) -> List[ProcessBottleneck]:
    """Convenience: find top workflow bottlenecks."""
    manifold = ProcessManifold(workflow=workflow, **kwargs)
    return manifold.find_bottlenecks()[:top_k]
