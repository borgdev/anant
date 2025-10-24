"""
Anant GraphQL Module
===================

GraphQL schema and resolvers for Anant platform API.
"""

try:
    from .anant_graphql_schema import (
        schema,
        Query,
        Mutation,
        Graph,
        Node,
        Edge,
        Hyperedge
    )
    GRAPHQL_AVAILABLE = True
except ImportError:
    GRAPHQL_AVAILABLE = False

__all__ = []
if GRAPHQL_AVAILABLE:
    __all__.extend([
        "schema",
        "Query",
        "Mutation", 
        "Graph",
        "Node",
        "Edge",
        "Hyperedge"
    ])