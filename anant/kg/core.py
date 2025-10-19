"""
Core Knowledge Graph Classes
===========================

High-performance knowledge graph implementation built on top of ANANT hypergraphs
with semantic awareness, ontology support, and advanced analytics capabilities.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import defaultdict
import numpy as np

from ..classes.hypergraph import Hypergraph
from ..utils.performance import performance_monitor, PerformanceProfiler
from ..algorithms.sampling import SmartSampler
from ..utils.extras import safe_import

# Optional dependencies
rdflib = safe_import('rdflib')
networkx = safe_import('networkx')

logger = logging.getLogger(__name__)


class SemanticHypergraph(Hypergraph):
    """
    Enhanced hypergraph with semantic awareness and knowledge graph capabilities
    
    Extends the base Hypergraph class with:
    - Semantic node and edge typing
    - Ontology schema integration
    - URI-based entity identification
    - Namespace management
    """
    
    def __init__(self, data=None, ontology_schema=None, namespaces=None):
        """
        Initialize semantic hypergraph
        
        Args:
            data: Initial hypergraph data
            ontology_schema: Ontology schema information
            namespaces: URI namespace mappings
        """
        super().__init__(data)
        
        self.ontology_schema = ontology_schema or {}
        self.namespaces = namespaces or {}
        
        # Semantic indexes for fast querying
        self._entity_types = defaultdict(set)
        self._relationship_types = defaultdict(set)
        self._uri_to_node = {}
        self._node_to_uri = {}
        
        # Performance caches
        self._query_cache = {}
        self._type_cache = {}
        
        # Build semantic indexes
        if data is not None:
            self._build_semantic_indexes()
    
    def _build_semantic_indexes(self):
        """Build semantic indexes for fast querying"""
        logger.info("Building semantic indexes...")
        
        with PerformanceProfiler("semantic_indexing") as profiler:
            
            profiler.checkpoint("start_indexing")
            
            # Index entity types based on URI patterns
            for node in self.nodes:
                entity_type = self._extract_entity_type(node)
                if entity_type:
                    self._entity_types[entity_type].add(node)
                    self._type_cache[node] = entity_type
                
                # Index URI mappings
                if self._is_uri(node):
                    self._uri_to_node[node] = node
                    self._node_to_uri[node] = node
            
            profiler.checkpoint("entity_indexing_complete")
            
            # Index relationship types
            for edge in self.edges:
                edge_nodes = self.incidences.get_edge_nodes(edge)
                relationship_type = self._extract_relationship_type(edge, edge_nodes)
                if relationship_type:
                    self._relationship_types[relationship_type].add(edge)
            
            profiler.checkpoint("relationship_indexing_complete")
        
        report = profiler.get_report()
        logger.info(f"Semantic indexing completed in {report['total_execution_time']:.2f}s")
        logger.info(f"Indexed {len(self._entity_types)} entity types, {len(self._relationship_types)} relationship types")
    
    def _extract_entity_type(self, node: str) -> Optional[str]:
        """Extract entity type from URI or node identifier"""
        if not isinstance(node, str):
            return None
        
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
    
    def _extract_relationship_type(self, edge: str, edge_nodes: List[str]) -> Optional[str]:
        """Extract relationship type from edge and its nodes"""
        # Look for relationship indicators in edge nodes
        for node in edge_nodes:
            if any(indicator in node.lower() for indicator in ['has', 'is', 'relates', 'connects']):
                return self._extract_entity_type(node)
        
        # Default relationship type based on edge structure
        return f"connects_{len(edge_nodes)}_entities"
    
    def _is_uri(self, node: str) -> bool:
        """Check if a node is a URI"""
        if not isinstance(node, str):
            return False
        return node.startswith('http://') or node.startswith('https://') or '/' in node
    
    @performance_monitor("kg_entity_type_query")
    def get_entities_by_type(self, entity_type: str) -> Set[str]:
        """
        Get all entities of a specific type
        
        Args:
            entity_type: The type of entities to retrieve
            
        Returns:
            Set of entity identifiers
        """
        return self._entity_types.get(entity_type, set())
    
    @performance_monitor("kg_relationship_type_query")
    def get_relationships_by_type(self, relationship_type: str) -> Set[str]:
        """
        Get all relationships of a specific type
        
        Args:
            relationship_type: The type of relationships to retrieve
            
        Returns:
            Set of edge identifiers
        """
        return self._relationship_types.get(relationship_type, set())
    
    def get_entity_type(self, entity: str) -> Optional[str]:
        """Get the type of a specific entity"""
        return self._type_cache.get(entity)
    
    def get_all_entity_types(self) -> List[str]:
        """Get all available entity types"""
        return list(self._entity_types.keys())
    
    def get_all_relationship_types(self) -> List[str]:
        """Get all available relationship types"""
        return list(self._relationship_types.keys())
    
    def add_namespace(self, prefix: str, uri: str):
        """Add a namespace mapping"""
        self.namespaces[prefix] = uri
    
    def expand_uri(self, short_uri: str) -> str:
        """Expand a prefixed URI to full URI"""
        if ':' in short_uri and not short_uri.startswith('http'):
            prefix, suffix = short_uri.split(':', 1)
            if prefix in self.namespaces:
                return f"{self.namespaces[prefix]}{suffix}"
        return short_uri
    
    def compress_uri(self, full_uri: str) -> str:
        """Compress a full URI to prefixed form if possible"""
        for prefix, namespace in self.namespaces.items():
            if full_uri.startswith(namespace):
                return f"{prefix}:{full_uri[len(namespace):]}"
        return full_uri


class KnowledgeGraph(SemanticHypergraph):
    """
    Complete Knowledge Graph implementation with advanced analytics capabilities
    
    Features:
    - Semantic querying with SPARQL-like interface
    - Ontology analysis and validation
    - Entity resolution and linking
    - Path reasoning and inference
    - Embeddings and vector operations
    - Temporal analysis capabilities
    - High-performance caching and indexing
    """
    
    def __init__(self, 
                 data=None, 
                 ontology=None,
                 namespaces=None,
                 performance_config=None):
        """
        Initialize Knowledge Graph
        
        Args:
            data: Initial graph data
            ontology: Ontology schema or ontology graph
            namespaces: Namespace prefix mappings
            performance_config: Performance optimization settings
        """
        super().__init__(data, ontology, namespaces)
        
        # Performance configuration
        self.performance_config = performance_config or {
            'max_query_nodes': 10000,
            'max_reasoning_depth': 5,
            'cache_size': 1000,
            'use_sampling': True,
            'sampling_threshold': 5000
        }
        
        # Initialize components (lazy loading for performance)
        self._query_engine = None
        self._ontology_analyzer = None
        self._entity_resolver = None
        self._path_reasoner = None
        self._validator = None
        self._embedder = None
        
        # Advanced indexes
        self._adjacency_cache = {}
        self._path_cache = {}
        self._similarity_cache = {}
        
        logger.info(f"Knowledge Graph initialized with {len(self.nodes)} nodes and {len(self.edges)} edges")
    
    @property
    def query(self):
        """Lazy-loaded semantic query engine"""
        if self._query_engine is None:
            from .query import SemanticQueryEngine
            self._query_engine = SemanticQueryEngine(self)
        return self._query_engine
    
    @property
    def ontology(self):
        """Lazy-loaded ontology analyzer"""
        if self._ontology_analyzer is None:
            from .ontology import OntologyAnalyzer
            self._ontology_analyzer = OntologyAnalyzer(self)
        return self._ontology_analyzer
    
    @property
    def entities(self):
        """Lazy-loaded entity resolver"""
        if self._entity_resolver is None:
            from .entity import EntityResolver
            self._entity_resolver = EntityResolver(self)
        return self._entity_resolver
    
    @property
    def reasoning(self):
        """Lazy-loaded path reasoner"""
        if self._path_reasoner is None:
            from .reasoning import PathReasoner
            self._path_reasoner = PathReasoner(self)
        return self._path_reasoner
    
    @property
    def validator(self):
        """Lazy-loaded KG validator"""
        if self._validator is None:
            from .validation import KGValidator
            self._validator = KGValidator(self)
        return self._validator
    
    @property
    def embeddings(self):
        """Lazy-loaded embeddings engine"""
        if self._embedder is None:
            from .embeddings import KGEmbedder
            self._embedder = KGEmbedder(self)
        return self._embedder
    
    @performance_monitor("kg_semantic_search")
    def semantic_search(self, 
                       entity_type: Optional[str] = None,
                       relationship_type: Optional[str] = None,
                       pattern: Optional[Dict] = None,
                       limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform semantic search across the knowledge graph
        
        Args:
            entity_type: Filter by entity type
            relationship_type: Filter by relationship type
            pattern: Complex pattern to match
            limit: Maximum number of results
            
        Returns:
            Search results with metadata
        """
        
        # Use sampling for large graphs
        if self.performance_config['use_sampling'] and len(self.nodes) > self.performance_config['sampling_threshold']:
            logger.info("Using sampling for large graph semantic search")
            sampler = SmartSampler(self, strategy='adaptive')
            sample_kg = sampler.adaptive_sample(
                sample_size=self.performance_config['max_query_nodes'],
                algorithm='general'
            )
            # Recursively search on sample
            sample_results = sample_kg.semantic_search(entity_type, relationship_type, pattern, limit)
            return self._extend_search_results(sample_results)
        
        results = {
            'entities': [],
            'relationships': [],
            'patterns': [],
            'metadata': {
                'total_nodes_searched': len(self.nodes),
                'total_edges_searched': len(self.edges),
                'search_time': 0.0
            }
        }
        
        start_time = time.time()
        
        # Search by entity type
        if entity_type:
            entities = self.get_entities_by_type(entity_type)
            if limit:
                entities = set(list(entities)[:limit])
            results['entities'] = list(entities)
        
        # Search by relationship type  
        if relationship_type:
            relationships = self.get_relationships_by_type(relationship_type)
            if limit:
                relationships = set(list(relationships)[:limit])
            results['relationships'] = list(relationships)
        
        # Pattern matching (delegated to query engine)
        if pattern:
            pattern_results = self.query.pattern_match(pattern, limit=limit)
            results['patterns'] = pattern_results
        
        results['metadata']['search_time'] = time.time() - start_time
        
        return results
    
    def _extend_search_results(self, sample_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extend search results from sample to full graph"""
        # For knowledge graphs, we often want to see the sample results
        # rather than extending to the full graph for performance
        sample_results['metadata']['note'] = 'Results from sampled graph for performance'
        return sample_results
    
    @performance_monitor("kg_subgraph_extraction")
    def get_subgraph(self, 
                    entities: List[str], 
                    max_hops: int = 1,
                    include_types: Optional[List[str]] = None) -> 'KnowledgeGraph':
        """
        Extract a subgraph around specified entities
        
        Args:
            entities: Central entities for subgraph
            max_hops: Maximum number of hops from central entities
            include_types: Entity types to include in expansion
            
        Returns:
            New KnowledgeGraph containing the subgraph
        """
        
        subgraph_nodes = set(entities)
        
        # Expand by hops
        current_nodes = set(entities)
        for hop in range(max_hops):
            next_nodes = set()
            
            for node in current_nodes:
                # Get neighbors through hyperedges
                incident_edges = self.incidences.get_node_edges(node)
                
                for edge in incident_edges:
                    edge_nodes = self.incidences.get_edge_nodes(edge)
                    
                    for neighbor in edge_nodes:
                        if neighbor != node:
                            # Filter by entity type if specified
                            if include_types:
                                neighbor_type = self.get_entity_type(neighbor)
                                if neighbor_type not in include_types:
                                    continue
                            
                            next_nodes.add(neighbor)
            
            subgraph_nodes.update(next_nodes)
            current_nodes = next_nodes
            
            if not current_nodes:  # No more expansion possible
                break
        
        # Build subgraph edge dictionary
        subgraph_edges = {}
        for edge in self.edges:
            edge_nodes = self.incidences.get_edge_nodes(edge)
            
            # Keep edge if at least 2 nodes are in subgraph
            relevant_nodes = [n for n in edge_nodes if n in subgraph_nodes]
            if len(relevant_nodes) >= 2:
                subgraph_edges[edge] = relevant_nodes
        
        # Create new KG with same configuration
        subgraph = KnowledgeGraph(
            data=subgraph_edges,
            ontology=self.ontology_schema,
            namespaces=self.namespaces,
            performance_config=self.performance_config
        )
        
        logger.info(f"Created subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges")
        
        return subgraph
    
    @performance_monitor("kg_merge_graphs")
    def merge_knowledge_graphs(self, other_kg: 'KnowledgeGraph') -> 'KnowledgeGraph':
        """
        Merge another knowledge graph into this one
        
        Args:
            other_kg: Knowledge graph to merge
            
        Returns:
            New merged knowledge graph
        """
        
        # Combine edge dictionaries
        merged_edges = self.incidences.to_dict()
        other_edges = other_kg.incidences.to_dict()
        
        # Handle edge ID conflicts
        edge_id_offset = len(merged_edges)
        for edge_id, nodes in other_edges.items():
            new_edge_id = f"merged_{edge_id_offset}_{edge_id}"
            merged_edges[new_edge_id] = nodes
            edge_id_offset += 1
        
        # Combine namespaces
        merged_namespaces = {**self.namespaces, **other_kg.namespaces}
        
        # Combine ontology schemas
        merged_ontology = {**self.ontology_schema, **other_kg.ontology_schema}
        
        # Create merged KG
        merged_kg = KnowledgeGraph(
            data=merged_edges,
            ontology=merged_ontology,
            namespaces=merged_namespaces,
            performance_config=self.performance_config
        )
        
        logger.info(f"Merged KG: {len(merged_kg.nodes)} nodes, {len(merged_kg.edges)} edges")
        
        return merged_kg
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive knowledge graph statistics"""
        
        base_stats = self.incidences.get_statistics()
        
        kg_stats = {
            'basic_stats': base_stats,
            'semantic_stats': {
                'entity_types': len(self._entity_types),
                'relationship_types': len(self._relationship_types),
                'entities_by_type': {etype: len(entities) for etype, entities in self._entity_types.items()},
                'relationships_by_type': {rtype: len(rels) for rtype, rels in self._relationship_types.items()}
            },
            'performance_stats': {
                'query_cache_size': len(self._query_cache),
                'type_cache_size': len(self._type_cache),
                'adjacency_cache_size': len(self._adjacency_cache)
            },
            'ontology_stats': {
                'namespaces': len(self.namespaces),
                'schema_elements': len(self.ontology_schema)
            }
        }
        
        return kg_stats
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"KnowledgeGraph(nodes={stats['basic_stats']['num_nodes']}, "
                f"edges={stats['basic_stats']['num_edges']}, "
                f"entity_types={stats['semantic_stats']['entity_types']}, "
                f"relationship_types={stats['semantic_stats']['relationship_types']})")