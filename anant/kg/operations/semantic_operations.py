"""
Semantic Operations for Knowledge Graph

Handles semantic aspects of knowledge graphs including:
- Entity type extraction and management
- URI handling and namespace management
- Semantic indexing and caching
- Entity and relationship type queries
"""

from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
import logging

from ...exceptions import KnowledgeGraphError, ValidationError
from ...utils.performance import performance_monitor

logger = logging.getLogger(__name__)


class SemanticOperations:
    """
    Semantic operations for knowledge graph
    
    Provides semantic awareness including entity typing, URI management,
    and namespace handling for knowledge graphs.
    """
    
    def __init__(self, knowledge_graph):
        """
        Initialize SemanticOperations
        
        Parameters
        ----------
        knowledge_graph : KnowledgeGraph
            Parent knowledge graph instance
        """
        if knowledge_graph is None:
            raise KnowledgeGraphError("Knowledge graph instance cannot be None")
        self.kg = knowledge_graph
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Semantic indexes for fast querying
        self._entity_types = defaultdict(set)
        self._relationship_types = defaultdict(set)
        self._uri_to_node = {}
        self._node_to_uri = {}
        
        # Performance caches
        self._query_cache = {}
        self._type_cache = {}
    
    def build_semantic_indexes(self):
        """Build semantic indexes for fast querying"""
        self.logger.info("Building semantic indexes...")
        
        try:
            # Clear existing indexes
            self._entity_types.clear()
            self._relationship_types.clear()
            self._uri_to_node.clear()
            self._node_to_uri.clear()
            self._type_cache.clear()
            
            # Index entity types based on URI patterns
            for node in self.kg.nodes:
                entity_type = self.extract_entity_type(node)
                if entity_type:
                    self._entity_types[entity_type].add(node)
                    self._type_cache[node] = entity_type
                
                # Index URI mappings
                if self.is_uri(node):
                    self._uri_to_node[node] = node
                    self._node_to_uri[node] = node
            
            # Index relationship types
            for edge in self.kg.edges:
                edge_nodes = self.kg.incidences.get_edge_nodes(edge)
                relationship_type = self.extract_relationship_type(edge, edge_nodes)
                if relationship_type:
                    self._relationship_types[relationship_type].add(edge)
            
            self.logger.info(f"Indexed {len(self._entity_types)} entity types, "
                           f"{len(self._relationship_types)} relationship types")
            
        except Exception as e:
            raise KnowledgeGraphError(f"Failed to build semantic indexes: {e}")
    
    def extract_entity_type(self, node: str) -> Optional[str]:
        """
        Extract entity type from URI or node identifier
        
        Parameters
        ----------
        node : str
            Node identifier or URI
            
        Returns
        -------
        Optional[str]
            Extracted entity type or None
        """
        if not isinstance(node, str):
            return None
        
        try:
            # Handle FIBO-style URIs
            if 'ontology/' in node:
                parts = node.split('/')
                if len(parts) >= 2:
                    # Extract class name from URI
                    class_name = parts[-1] or parts[-2]
                    return class_name
            
            # Handle other URI patterns
            if '/' in node or '#' in node:
                separator = '#' if '#' in node else '/'
                return node.split(separator)[-1]
            
            # Handle typed literals or prefixed names
            if ':' in node and not node.startswith('http'):
                return node.split(':')[0]
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error extracting entity type from {node}: {e}")
            return None
    
    def extract_relationship_type(self, edge: str, edge_nodes: List[str]) -> Optional[str]:
        """
        Extract relationship type from edge and its nodes
        
        Parameters
        ----------
        edge : str
            Edge identifier
        edge_nodes : List[str]
            List of nodes in the edge
            
        Returns
        -------
        Optional[str]
            Extracted relationship type or None
        """
        try:
            # Look for relationship indicators in edge nodes
            for node in edge_nodes:
                if any(indicator in node.lower() for indicator in ['has', 'is', 'relates', 'connects']):
                    return self.extract_entity_type(node)
            
            # Default relationship type based on edge structure
            return f"connects_{len(edge_nodes)}_entities"
            
        except Exception as e:
            self.logger.warning(f"Error extracting relationship type from {edge}: {e}")
            return None
    
    def is_uri(self, node: str) -> bool:
        """
        Check if a node is a URI
        
        Parameters
        ----------
        node : str
            Node to check
            
        Returns
        -------
        bool
            True if node is a URI
        """
        if not isinstance(node, str):
            return False
        return node.startswith('http://') or node.startswith('https://') or '/' in node
    
    @performance_monitor("kg_entity_type_query")
    def get_entities_by_type(self, entity_type: str) -> Set[str]:
        """
        Get all entities of a specific type
        
        Parameters
        ----------
        entity_type : str
            The type of entities to retrieve
            
        Returns
        -------
        Set[str]
            Set of entity identifiers
        """
        return self._entity_types.get(entity_type, set())
    
    @performance_monitor("kg_relationship_type_query")
    def get_relationships_by_type(self, relationship_type: str) -> Set[str]:
        """
        Get all relationships of a specific type
        
        Parameters
        ----------
        relationship_type : str
            The type of relationships to retrieve
            
        Returns
        -------
        Set[str]
            Set of edge identifiers
        """
        return self._relationship_types.get(relationship_type, set())
    
    def get_entity_type(self, entity: str) -> Optional[str]:
        """
        Get the type of a specific entity
        
        Parameters
        ----------
        entity : str
            Entity identifier
            
        Returns
        -------
        Optional[str]
            Entity type or None
        """
        return self._type_cache.get(entity)
    
    def get_all_entity_types(self) -> List[str]:
        """
        Get all available entity types
        
        Returns
        -------
        List[str]
            List of all entity types
        """
        return list(self._entity_types.keys())
    
    def get_all_relationship_types(self) -> List[str]:
        """
        Get all available relationship types
        
        Returns
        -------
        List[str]
            List of all relationship types
        """
        return list(self._relationship_types.keys())
    
    def add_namespace(self, prefix: str, uri: str):
        """
        Add a namespace mapping
        
        Parameters
        ----------
        prefix : str
            Namespace prefix
        uri : str
            Full namespace URI
        """
        if not hasattr(self.kg, 'namespaces'):
            self.kg.namespaces = {}
        self.kg.namespaces[prefix] = uri
    
    def expand_uri(self, short_uri: str) -> str:
        """
        Expand a prefixed URI to full URI
        
        Parameters
        ----------
        short_uri : str
            Prefixed URI to expand
            
        Returns
        -------
        str
            Expanded URI
        """
        if ':' in short_uri and not short_uri.startswith('http'):
            prefix, suffix = short_uri.split(':', 1)
            if hasattr(self.kg, 'namespaces') and prefix in self.kg.namespaces:
                return f"{self.kg.namespaces[prefix]}{suffix}"
        return short_uri
    
    def compress_uri(self, full_uri: str) -> str:
        """
        Compress a full URI to prefixed form if possible
        
        Parameters
        ----------
        full_uri : str
            Full URI to compress
            
        Returns
        -------
        str
            Compressed URI or original if no matching namespace
        """
        if hasattr(self.kg, 'namespaces'):
            for prefix, namespace in self.kg.namespaces.items():
                if full_uri.startswith(namespace):
                    return f"{prefix}:{full_uri[len(namespace):]}"
        return full_uri
    
    def clear_cache(self):
        """Clear all semantic caches"""
        self._query_cache.clear()
        self._type_cache.clear()
        self.logger.debug("Semantic caches cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get semantic cache statistics
        
        Returns
        -------
        Dict[str, Any]
            Cache statistics
        """
        return {
            'entity_types_count': len(self._entity_types),
            'relationship_types_count': len(self._relationship_types),
            'uri_mappings_count': len(self._uri_to_node),
            'type_cache_size': len(self._type_cache),
            'query_cache_size': len(self._query_cache)
        }