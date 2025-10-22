"""
Geometric Analysis Core
=======================

Revolutionary framework: Riemannian geometry for graph analytics.

Core modules:
- riemannian_manifold: Graph as Riemannian manifold
- property_manifold: Property space geometry
- curvature_engine: Curvature computations
- geodesic_solver: Geodesic equations
"""

from .riemannian_manifold import RiemannianGraphManifold
from .property_manifold import PropertyManifold
from .curvature_engine import CurvatureEngine
from .geodesic_solver import GeodesicSolver

__all__ = [
    'RiemannianGraphManifold',
    'PropertyManifold',
    'CurvatureEngine',
    'GeodesicSolver',
]
