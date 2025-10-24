"""
Anant Registry Module
====================

Graph registry components for managing graph metadata and Parquet storage.
"""

try:
    from .registry_server import (
        RegistryConfig,
        GraphRegistryEntry,
        GraphStats,
        GraphCreateRequest,
        GraphUpdateRequest,
        GraphQueryRequest,
        app
    )
    from .create_registry_schema import (
        DatabaseConfig,
        create_graph_registry_schema,
        insert_initial_data
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

__all__ = []
if REGISTRY_AVAILABLE:
    __all__.extend([
        "RegistryConfig",
        "GraphRegistryEntry",
        "GraphStats",
        "GraphCreateRequest",
        "GraphUpdateRequest", 
        "GraphQueryRequest",
        "app",
        "DatabaseConfig",
        "create_graph_registry_schema",
        "insert_initial_data"
    ])