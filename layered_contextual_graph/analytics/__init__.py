"""
LCG Analytics Module
====================

Property-level analytics for LayeredContextualGraph.

Analyzes:
- Properties across layers
- Property-based context derivation
- Index-based analytics
- Tag-based clustering
- Property evolution across hierarchy
"""

from .property_analytics import (
    PropertyAnalytics,
    PropertyBasedContext,
    derive_contexts_from_properties
)

from .index_analytics import (
    IndexAnalytics,
    build_property_indices
)

from .tag_analytics import (
    TagAnalytics,
    cluster_by_tags
)

__all__ = [
    'PropertyAnalytics',
    'PropertyBasedContext',
    'derive_contexts_from_properties',
    'IndexAnalytics',
    'build_property_indices',
    'TagAnalytics',
    'cluster_by_tags',
]
