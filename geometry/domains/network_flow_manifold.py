"""
Network Flow Manifold
=====================

Interprets network flows as vector fields on a manifold.

Key Concepts:
- Nodes embedded via RiemannianGraphManifold coordinates
- Edge flow magnitudes mapped to tangent vectors
- Divergence measures bottlenecks / sources / sinks
- Curl highlights rotational communities / echo chambers

This module provides tools to analyze network flow structure
geometrically without crafting bespoke algorithms.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import numpy as np
    NETWORK_FLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - numpy always expected
    NETWORK_FLOW_AVAILABLE = False

from ..core.riemannian_manifold import RiemannianGraphManifold

logger = logging.getLogger(__name__)


@dataclass
class FlowDivergence:
    node_id: str
    divergence: float


@dataclass
class FlowCurl:
    node_id: str
    curl_magnitude: float


class NetworkFlowManifold:
    """
    Manifold perspective on network flows.

    Args:
        graph: Graph-like object supporting nodes() and edges() iteration.
               Edges can be tuples (u, v, data) where data contains flow.
        flow_attribute: Attribute name storing flow magnitude/direction.
        default_flow: Flow value to use when attribute missing.
        property_weights: Optional property weights for manifold embedding.

    Methods:
        compute_divergence(): identifies sources/sinks/bottlenecks
        compute_curl(): detects rotational communities / echo chambers
        find_bottlenecks(): high absolute divergence nodes
        influence_sources(): positive divergence nodes
        curl_hotspots(): nodes with largest curl magnitude
    """

    def __init__(
        self,
        graph: Any,
        flow_attribute: str = "flow",
        default_flow: float = 1.0,
        property_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        if not NETWORK_FLOW_AVAILABLE:
            raise RuntimeError("NumPy is required for NetworkFlowManifold")

        self.graph = graph
        self.flow_attribute = flow_attribute
        self.default_flow = default_flow

        # Build base manifold for geometric coordinates
        self.manifold = RiemannianGraphManifold(
            graph,
            property_weights=property_weights,
            use_properties=True,
        )
        self.node_coordinates = self.manifold.node_coordinates
        self.node_index = {node: idx for idx, node in enumerate(self.node_coordinates.keys())}

        logger.info(
            "NetworkFlowManifold initialized with %d nodes and %d edges",
            len(self.node_coordinates),
            len(self._edge_list()),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _edge_list(self) -> List[Tuple[str, str, float]]:
        edges: List[Tuple[str, str, float]] = []
        if hasattr(self.graph, "edges"):
            for edge in self.graph.edges(data=True):  # type: ignore[attr-defined]
                u, v = edge[0], edge[1]
                data = edge[2] if len(edge) > 2 else {}
                flow = self._extract_flow(data)
                edges.append((u, v, flow))
        elif hasattr(self.graph, "get_edges"):
            for edge in self.graph.get_edges():  # type: ignore[attr-defined]
                u, v = edge[:2]
                data = edge[2] if len(edge) > 2 else {}
                flow = self._extract_flow(data)
                edges.append((u, v, flow))
        else:
            raise ValueError("Graph object must provide edges() or get_edges()")
        return edges

    def _extract_flow(self, data: Any) -> float:
        if isinstance(data, dict) and self.flow_attribute in data:
            value = data[self.flow_attribute]
            if isinstance(value, (int, float)):
                return float(value)
        return float(self.default_flow)

    # ------------------------------------------------------------------
    # Vector field construction
    # ------------------------------------------------------------------
    def compute_vector_field(self) -> Dict[str, np.ndarray]:
        """Compute vector field at each node from incident flows."""
        dim = self.manifold.embedding_dim
        vectors = {node: np.zeros(dim, dtype=float) for node in self.node_coordinates}

        for u, v, flow in self._edge_list():
            if u not in self.node_coordinates or v not in self.node_coordinates:
                continue
            direction = self.node_coordinates[v] - self.node_coordinates[u]
            norm = np.linalg.norm(direction) + 1e-12
            direction /= norm
            vectors[u] += flow * direction
            vectors[v] -= flow * direction

        return vectors

    # ------------------------------------------------------------------
    # Divergence & curl analytics
    # ------------------------------------------------------------------
    def compute_divergence(self) -> List[FlowDivergence]:
        vectors = self.compute_vector_field()
        divergences: List[FlowDivergence] = []

        for node, vector in vectors.items():
            divergence = float(np.sum(vector))
            divergences.append(FlowDivergence(node_id=node, divergence=divergence))

        divergences.sort(key=lambda d: abs(d.divergence), reverse=True)
        return divergences

    def compute_curl(self) -> List[FlowCurl]:
        vectors = self.compute_vector_field()
        curl_values: List[FlowCurl] = []

        for node, vector in vectors.items():
            curl_mag = float(np.linalg.norm(np.gradient(vector))) if vector.size > 1 else 0.0
            curl_values.append(FlowCurl(node_id=node, curl_magnitude=curl_mag))
        curl_values.sort(key=lambda c: c.curl_magnitude, reverse=True)
        return curl_values

    # ------------------------------------------------------------------
    # High-level insights
    # ------------------------------------------------------------------
    def find_bottlenecks(self, top_k: int = 5) -> List[FlowDivergence]:
        divergences = self.compute_divergence()
        return divergences[:top_k]

    def influence_sources(self, top_k: int = 5) -> List[FlowDivergence]:
        divergences = [d for d in self.compute_divergence() if d.divergence > 0]
        divergences.sort(key=lambda d: d.divergence, reverse=True)
        return divergences[:top_k]

    def influence_sinks(self, top_k: int = 5) -> List[FlowDivergence]:
        divergences = [d for d in self.compute_divergence() if d.divergence < 0]
        divergences.sort(key=lambda d: d.divergence)
        return divergences[:top_k]

    def curl_hotspots(self, top_k: int = 5) -> List[FlowCurl]:
        return self.compute_curl()[:top_k]

    def summary(self) -> Dict[str, Any]:
        divergences = self.compute_divergence()
        curls = self.compute_curl()
        return {
            "nodes": len(self.node_coordinates),
            "edges": len(self._edge_list()),
            "max_divergence": divergences[0].divergence if divergences else 0.0,
            "max_curl": curls[0].curl_magnitude if curls else 0.0,
        }


def find_bottlenecks_geometric(
    graph: Any,
    flow_attribute: str = "flow",
    top_k: int = 5,
    **kwargs: Any,
) -> List[FlowDivergence]:
    """Convenience wrapper returning top bottlenecks via geometric divergence."""

    manifold = NetworkFlowManifold(
        graph=graph,
        flow_attribute=flow_attribute,
        **kwargs,
    )
    return manifold.find_bottlenecks(top_k=top_k)
