"""
Geometric Analysis Core
=======================

Core building blocks for the geometric analytics framework.
"""

from .riemannian_manifold import RiemannianGraphManifold
from .property_manifold import PropertyManifold

__all__ = [
    'RiemannianGraphManifold',
    'PropertyManifold',
]
