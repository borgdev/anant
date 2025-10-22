"""
Domain-Specific Geometric Manifolds
===================================

Specialized manifolds where geometry delivers immediate insight:

- TimeSeriesManifold: Cyclic geodesics, curvature anomalies
- NetworkFlowManifold: Vector fields, divergence/curl analytics
- FinancialManifold: Risk geometry, diversification curvature
- MolecularManifold: Configuration manifolds, reaction paths
- PhaseSpaceManifold: Dynamical stability, chaos detection
- SemanticManifold: Language manifolds, analogies as parallel transport
"""

from .timeseries_manifold import TimeSeriesManifold, detect_cycles_geometric
from .network_flow_manifold import NetworkFlowManifold, find_bottlenecks_geometric
from .financial_manifold import FinancialManifold, compute_risk_geometric
from .molecular_manifold import MolecularManifold, find_strained_conformers_geometric
from .phase_space_manifold import PhaseSpaceManifold
from .semantic_manifold import SemanticManifold

__all__ = [
    # Time Series
    'TimeSeriesManifold',
    'detect_cycles_geometric',

    # Network Flow
    'NetworkFlowManifold',
    'find_bottlenecks_geometric',

    # Finance
    'FinancialManifold',
    'compute_risk_geometric',

    # Molecular
    'MolecularManifold',
    'find_strained_conformers_geometric',

    # Phase Space
    'PhaseSpaceManifold',

    # Semantic
    'SemanticManifold',
]
