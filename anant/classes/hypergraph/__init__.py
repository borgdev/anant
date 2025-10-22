"""
Hypergraph package for Anant Library

Provides modular hypergraph implementation with specialized operation modules.
"""

from .core.hypergraph import Hypergraph
from .core.property_wrapper import PropertyWrapper
from .views.edge_view import EdgeView

__all__ = [
    'Hypergraph',
    'PropertyWrapper', 
    'EdgeView'
]