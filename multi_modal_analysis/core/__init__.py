"""
Multi-Modal Relationship Analysis Core Module
==============================================

Core components for multi-modal hypergraph analysis enabling cross-domain insights.
"""

from .multi_modal_hypergraph import MultiModalHypergraph, ModalityConfig
from .cross_modal_analyzer import (
    CrossModalAnalyzer,
    CrossModalPattern,
    InterModalRelationship
)
from .modal_metrics import (
    ModalMetrics,
    MultiModalCentrality,
    ModalCorrelation
)

__all__ = [
    'MultiModalHypergraph',
    'ModalityConfig',
    'CrossModalAnalyzer',
    'CrossModalPattern',
    'InterModalRelationship',
    'ModalMetrics',
    'MultiModalCentrality',
    'ModalCorrelation',
]

__version__ = '1.0.0'
