"""
Geometry Module
===============

Revolutionary geometric analysis for graph properties and manifolds.

This module provides:
- Riemannian manifold analysis
- Property manifolds for different domains
- Geometric time series analysis
- Domain-specific manifolds (financial, biological, etc.)

Core Features:
- Revolutionary insights through differential geometry
- Natural pattern detection via curvature analysis
- Domain-agnostic geometric transformations
- Advanced manifold operations
"""

# Core geometry components
from .core import RiemannianGraphManifold, PropertyManifold

# Domain-specific manifolds - lazy import to avoid heavy dependencies
def _get_domains():
    """Lazy import of domain-specific manifolds"""
    try:
        from . import domains
        return domains
    except ImportError as e:
        print(f"Warning: Geometry domains not available: {e}")
        return None

def __getattr__(name):
    """Provide lazy access to domains module"""
    if name == 'domains':
        return _get_domains()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'RiemannianGraphManifold',
    'PropertyManifold',
    'domains',
]