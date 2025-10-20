"""
Anant Metagraph - Enterprise Knowledge Management System
========================================================

A comprehensive dual-capability library offering both traditional hypergraph functionality 
and advanced metagraph capabilities with Polars+Parquet backend for enterprise metadata management.

Dual Capability Architecture:
- Traditional Hypergraph: pandas-based hypergraph operations for compatibility
- Advanced Metagraph: Polars+Parquet-based enterprise knowledge management

Core Components:
- Hierarchical Store: Multi-level entity organization with parent-child relationships
- Metadata Store: Rich metadata management with schema validation and search
- Semantic Layer: Embeddings, relationships, and business glossary
- Temporal Layer: Event tracking, patterns analysis, and lineage
- Policy Layer: Governance, access control, and compliance management

Features:
- High-performance analytics with Polars backend
- Enterprise-grade security and compliance
- LLM integration ready (Phase 2)
- Semantic search and relationship discovery
- Temporal analytics and pattern detection
- Policy-driven governance and access control
"""

from pathlib import Path
from .core.metagraph import Metagraph
from .core.hierarchical_store import HierarchicalStore  
from .core.metadata_store import MetadataStore
from .semantic.semantic_layer import SemanticLayer, SemanticRelationship
from .temporal.temporal_layer import TemporalLayer, TemporalEvent, TemporalPattern
from .governance.policy_layer import PolicyEngine, PolicyRule, AccessControl

__version__ = "1.0.0-phase1"
__author__ = "Anant Team"
__description__ = "Enterprise Knowledge Management with Dual Hypergraph+Metagraph Capabilities"

# Main exports for easy access
__all__ = [
    # Core classes
    "Metagraph",
    "HierarchicalStore", 
    "MetadataStore",
    "SemanticLayer",
    "TemporalLayer", 
    "PolicyEngine",
    
    # Data classes
    "SemanticRelationship",
    "TemporalEvent",
    "TemporalPattern", 
    "PolicyRule",
    "AccessControl",
    
    # Convenience functions
    "create_enterprise_metagraph",
    "create_basic_metagraph"
]

def create_enterprise_metagraph(storage_path: str = "./enterprise_metagraph",
                               embedding_dimension: int = 768,
                               compression: str = "zstd",
                               retention_days: int = 2555) -> Metagraph:
    """
    Create a fully-configured enterprise metagraph with all layers enabled.
    
    Args:
        storage_path: Base storage path for all metagraph data
        embedding_dimension: Dimension for semantic embeddings (768 for BERT-like models)
        compression: Parquet compression algorithm (zstd recommended for enterprise)
        retention_days: Temporal data retention period (default: 7 years)
        
    Returns:
        Configured Metagraph instance ready for enterprise use
        
    Example:
        >>> mg = create_enterprise_metagraph("./my_knowledge_graph")
        >>> mg.create_entity("product_1", "product", {"name": "Widget A", "price": 100})
        >>> entities = mg.search_entities("Widget")
    """
    return Metagraph(
        storage_path=Path(storage_path),
        embedding_dimension=embedding_dimension,
        compression=compression,
        retention_days=retention_days
    )

def create_basic_metagraph(storage_path: str = "./basic_metagraph") -> Metagraph:
    """
    Create a basic metagraph configuration for development and testing.
    
    Args:
        storage_path: Base storage path
        
    Returns:
        Basic Metagraph instance
        
    Example:
        >>> mg = create_basic_metagraph()
        >>> mg.create_entity("test_1", "test", {"value": "hello"})
    """
    return Metagraph(storage_path=Path(storage_path))

# Version information
def get_version_info():
    """Get detailed version and capability information."""
    return {
        "version": __version__,
        "phase": "Phase 1 - Core Functionality",
        "capabilities": {
            "hierarchical_organization": True,
            "metadata_management": True, 
            "semantic_relationships": True,
            "temporal_tracking": True,
            "governance_policies": True,
            "llm_integration": False,  # Phase 2
            "advanced_analytics": False,  # Phase 3
            "production_optimization": False  # Phase 4
        },
        "backend": "Polars+Parquet",
        "compression": ["zstd", "lz4", "snappy", "gzip", "brotli"],
        "storage_format": "Parquet with metadata schemas"
    }