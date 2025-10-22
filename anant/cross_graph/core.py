"""
Cross-Graph Core Manager
======================

Central manager for cross-graph operations and interoperability.
"""

from typing import Dict, List, Any, Optional, Union, Type
import logging
from pathlib import Path

from ..classes.hypergraph import Hypergraph
from ..kg.core import KnowledgeGraph
from ..kg.hierarchical import HierarchicalKnowledgeGraph
from ..metagraph.core.metagraph import Metagraph

logger = logging.getLogger(__name__)


class CrossGraphManager:
    """
    Central manager for cross-graph operations
    
    Coordinates conversions, migrations, unified queries, and fusion
    operations across all supported graph types.
    """
    
    def __init__(self):
        """Initialize cross-graph manager"""
        self._converters = {}
        self._query_interface = None
        self._fusion_engine = None
        
        # Register supported graph types
        self.supported_types = {
            'hypergraph': Hypergraph,
            'knowledge_graph': KnowledgeGraph, 
            'hierarchical_kg': HierarchicalKnowledgeGraph,
            'metagraph': Metagraph
        }
        
        logger.info("CrossGraphManager initialized")
    
    def get_graph_type(self, graph: Any) -> Optional[str]:
        """Identify the type of a graph object"""
        for type_name, type_class in self.supported_types.items():
            if isinstance(graph, type_class):
                return type_name
        return None
    
    def convert_graph(self, source_graph: Any, target_type: str, **kwargs) -> Any:
        """Convert a graph from one type to another"""
        source_type = self.get_graph_type(source_graph)
        if not source_type:
            raise ValueError(f"Unsupported source graph type: {type(source_graph)}")
        
        if target_type not in self.supported_types:
            raise ValueError(f"Unsupported target graph type: {target_type}")
        
        if source_type == target_type:
            return source_graph  # No conversion needed
        
        # Load appropriate converter
        converter_key = f"{source_type}_to_{target_type}"
        if converter_key not in self._converters:
            self._load_converter(source_type, target_type)
        
        converter = self._converters[converter_key]
        return converter.convert(source_graph, **kwargs)
    
    def _load_converter(self, source_type: str, target_type: str):
        """Load the appropriate converter for the graph type pair"""
        from .converters import GraphConverter
        
        converter_key = f"{source_type}_to_{target_type}"
        self._converters[converter_key] = GraphConverter(source_type, target_type)
    
    def get_unified_query_interface(self):
        """Get unified query interface for cross-graph queries"""
        if self._query_interface is None:
            from .unified_query import UnifiedQueryInterface
            self._query_interface = UnifiedQueryInterface()
        return self._query_interface
    
    def get_fusion_engine(self):
        """Get graph fusion engine for combining multiple graphs"""
        if self._fusion_engine is None:
            from .fusion import GraphFusion
            self._fusion_engine = GraphFusion()
        return self._fusion_engine
    
    def fuse_graphs(self, graphs: List[Any], fusion_strategy: str = "unified", **kwargs) -> Any:
        """Fuse multiple graphs into a unified structure"""
        fusion_engine = self.get_fusion_engine()
        return fusion_engine.fuse_graphs(graphs, fusion_strategy, **kwargs)
    
    def query_across_graphs(self, graphs: List[Any], query: str, **kwargs) -> Dict[str, Any]:
        """Execute a query across multiple graphs"""
        query_interface = self.get_unified_query_interface()
        return query_interface.query_multiple_graphs(graphs, query, **kwargs)


# Global cross-graph manager instance
cross_graph_manager = CrossGraphManager()