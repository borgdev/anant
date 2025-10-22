"""
Cross-Graph Interoperability Module
==================================

Unified interface and conversion utilities for working across different graph types:
- Hypergraph
- KnowledgeGraph  
- HierarchicalKnowledgeGraph
- Metagraph

Provides conversion, migration, unified querying, and graph fusion capabilities.
"""

from .core import CrossGraphManager
from .converters import (
    HypergraphToKGConverter,
    KGToMetagraphMigrator,
    GraphConverter
)
from .unified_query import UnifiedQueryInterface
from .fusion import GraphFusion

__all__ = [
    'CrossGraphManager',
    'HypergraphToKGConverter', 
    'KGToMetagraphMigrator',
    'GraphConverter',
    'UnifiedQueryInterface',
    'GraphFusion'
]