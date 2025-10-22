"""
Domain-Specific Geometric Manifolds
===================================

Specialized manifolds where geometry delivers immediate insight.

Domain-Specific:
- TimeSeriesManifold: Cyclic geodesics, curvature anomalies
- NetworkFlowManifold: Vector fields, divergence/curl analytics
- FinancialManifold: Risk geometry, diversification curvature
- MolecularManifold: Configuration manifolds, reaction paths
- PhaseSpaceManifold: Dynamical stability, chaos detection
- SemanticManifold: Language manifolds, analogies as parallel transport

General-Purpose:
- SpreadDynamicsManifold: Universal contagion/propagation (epidemic, viral, cascade)
- AllocationManifold: Multi-resource optimization (healthcare, cloud, logistics)
- ProcessManifold: Workflow/pipeline analysis (manufacturing, software, business)
- MatchingManifold: Similarity-based pairing (trials, HR, education)
- HierarchicalManifold: Multi-level systems (orgs, geography, taxonomy)
"""

from .timeseries_manifold import TimeSeriesManifold, detect_cycles_geometric
from .network_flow_manifold import NetworkFlowManifold, find_bottlenecks_geometric
from .financial_manifold import FinancialManifold, compute_risk_geometric
from .molecular_manifold import MolecularManifold, find_strained_conformers_geometric
from .phase_space_manifold import PhaseSpaceManifold
from .semantic_manifold import SemanticManifold
from .spread_dynamics_manifold import SpreadDynamicsManifold, detect_spread_hotspots
from .allocation_manifold import AllocationManifold, find_allocation_stress
from .process_manifold import ProcessManifold, find_workflow_bottlenecks
from .matching_manifold import MatchingManifold, find_best_matches
from .hierarchical_manifold import HierarchicalManifold

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
    
    # Spread Dynamics (General)
    'SpreadDynamicsManifold',
    'detect_spread_hotspots',
    
    # Allocation (General)
    'AllocationManifold',
    'find_allocation_stress',
    
    # Process (General)
    'ProcessManifold',
    'find_workflow_bottlenecks',
    
    # Matching (General)
    'MatchingManifold',
    'find_best_matches',
    
    # Hierarchical (General)
    'HierarchicalManifold',
]
