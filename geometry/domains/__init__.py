"""
Domain-Specific Geometric Manifolds
===================================

Specialized manifolds for different domains where geometry has natural meaning.

Key Domains:
- TimeSeriesManifold: Cyclic geodesics, anomalies as curvature
- NetworkFlowManifold: Vector fields, influence propagation
- FinancialManifold: Risk geometry, volatility as curvature
- MolecularManifold: Configuration space, reaction paths
- SemanticManifold: Language as manifold, analogies as parallel transport
"""

from .timeseries_manifold import TimeSeriesManifold, detect_cycles_geometric
from .network_flow_manifold import NetworkFlowManifold, find_bottlenecks_geometric
from .financial_manifold import FinancialManifold, compute_risk_geometric

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
]
