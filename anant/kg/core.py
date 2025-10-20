"""
Core Knowledge Graph Classes
===========================

High-performance knowledge graph implementation built on top of ANANT hypergraphs
with semantic awareness, ontology support, and advanced analytics capabilities.
"""

import logging
import time
from datetime import datetime
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
    
    # Semantic wrapper methods for user-friendly API
    def add_node(self, node_id: Any, properties: Optional[Dict] = None, entity_type: Optional[str] = None):
        """
        Add a node with semantic entity type information
        
        Args:
            node_id: Unique identifier for the node
            properties: Additional properties for the node
            entity_type: Semantic type of the entity (Person, Company, etc.)
        """
        # Combine entity_type into properties
        if properties is None:
            properties = {}
        if entity_type is not None:
            properties['entity_type'] = entity_type
        
        # Add to underlying hypergraph
        super().add_node(node_id, properties)
        
        # Update semantic indexes
        if entity_type:
            self._entity_types[entity_type].add(node_id)
            self._type_cache[node_id] = entity_type
    
    def add_entity(self, entity_id: Any, properties: Optional[Dict] = None, entity_type: Optional[str] = None):
        """
        Add an entity to the knowledge graph (alias for add_node).
        
        Args:
            entity_id: Unique identifier for the entity
            properties: Additional properties for the entity
            entity_type: Semantic type of the entity (Person, Company, etc.)
        """
        return self.add_node(entity_id, properties, entity_type)
    
    def add_edge(self, edge_id: Any, node_list: List[Any], weight: float = 1.0, 
                 properties: Optional[Dict] = None, edge_type: Optional[str] = None):
        """
        Add an edge with semantic relationship type information
        
        Args:
            edge_id: Unique identifier for the edge
            node_list: List of nodes to connect
            weight: Edge weight
            properties: Additional properties for the edge
            edge_type: Semantic type of the relationship (knows, works_for, etc.)
        """
        # Combine edge_type into properties
        if properties is None:
            properties = {}
        if edge_type is not None:
            properties['edge_type'] = edge_type
        
        # Add to underlying hypergraph
        super().add_edge(edge_id, node_list, weight, properties)
        
        # Update semantic indexes
        if edge_type:
            self._relationship_types[edge_type].add(edge_id)
    
    def add_relationship(self, source_entity: Any, target_entity: Any, relationship_type: str, 
                        properties: Optional[Dict] = None, weight: float = 1.0):
        """
        Add a relationship between two entities (semantic alias for add_edge).
        
        Args:
            source_entity: Source entity identifier
            target_entity: Target entity identifier  
            relationship_type: Type of relationship
            properties: Additional relationship properties
            weight: Relationship strength/weight
        """
        # Generate relationship ID
        relationship_id = f"{source_entity}_{relationship_type}_{target_entity}"
        
        # Add the edge with semantic type
        return self.add_edge(relationship_id, [source_entity, target_entity], 
                           weight=weight, properties=properties, edge_type=relationship_type)
    
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
    

    
    @performance_monitor("kg_semantic_similarity")
    def semantic_similarity(self, entity1: str, entity2: str, method: str = "jaccard") -> float:
        """
        Calculate semantic similarity between two entities
        
        Args:
            entity1: First entity
            entity2: Second entity
            method: Similarity method ('jaccard', 'cosine', 'structural')
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        
        if entity1 not in self.nodes or entity2 not in self.nodes:
            return 0.0
        
        if entity1 == entity2:
            return 1.0
        
        if method == "jaccard":
            return self._jaccard_similarity(entity1, entity2)
        elif method == "cosine":
            return self._cosine_similarity(entity1, entity2)
        elif method == "structural":
            return self._structural_similarity(entity1, entity2)
        else:
            logger.warning(f"Unknown similarity method {method}, using jaccard")
            return self._jaccard_similarity(entity1, entity2)
    
    def _jaccard_similarity(self, entity1: str, entity2: str) -> float:
        """Calculate Jaccard similarity based on shared neighbors"""
        
        # Get neighbors for both entities
        neighbors1 = set()
        neighbors2 = set()
        
        for edge in self.incidences.get_node_edges(entity1):
            edge_nodes = self.incidences.get_edge_nodes(edge)
            neighbors1.update(n for n in edge_nodes if n != entity1)
        
        for edge in self.incidences.get_node_edges(entity2):
            edge_nodes = self.incidences.get_edge_nodes(edge)
            neighbors2.update(n for n in edge_nodes if n != entity2)
        
        # Calculate Jaccard index
        intersection = len(neighbors1 & neighbors2)
        union = len(neighbors1 | neighbors2)
        
        return intersection / union if union > 0 else 0.0
    
    def _cosine_similarity(self, entity1: str, entity2: str) -> float:
        """Calculate cosine similarity based on edge type vectors"""
        
        # Build edge type vectors for each entity
        vector1 = defaultdict(int)
        vector2 = defaultdict(int)
        
        # Count edge types for entity1
        for edge in self.incidences.get_node_edges(entity1):
            edge_type = self.properties.get_edge_properties(edge).get('edge_type', 'unknown')
            vector1[edge_type] += 1
        
        # Count edge types for entity2
        for edge in self.incidences.get_node_edges(entity2):
            edge_type = self.properties.get_edge_properties(edge).get('edge_type', 'unknown')
            vector2[edge_type] += 1
        
        # Get all edge types
        all_types = set(vector1.keys()) | set(vector2.keys())
        
        if not all_types:
            return 0.0
        
        # Calculate cosine similarity
        dot_product = sum(vector1[t] * vector2[t] for t in all_types)
        
        norm1 = np.sqrt(sum(vector1[t]**2 for t in all_types))
        norm2 = np.sqrt(sum(vector2[t]**2 for t in all_types))
        
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
    
    def _structural_similarity(self, entity1: str, entity2: str) -> float:
        """Calculate structural similarity based on graph position"""
        
        # Get entity types and properties
        type1 = self.get_entity_type(entity1)
        type2 = self.get_entity_type(entity2)
        
        # Type similarity component
        type_sim = 1.0 if type1 == type2 else 0.0
        
        # Degree similarity component  
        degree1 = len(self.incidences.get_node_edges(entity1))
        degree2 = len(self.incidences.get_node_edges(entity2))
        
        max_degree = max(degree1, degree2)
        degree_sim = 1.0 - abs(degree1 - degree2) / max_degree if max_degree > 0 else 1.0
        
        # Neighborhood similarity component
        neighbor_sim = self._jaccard_similarity(entity1, entity2)
        
        # Weighted combination
        return 0.4 * type_sim + 0.3 * degree_sim + 0.3 * neighbor_sim
    
    @performance_monitor("kg_shortest_semantic_path")
    def shortest_semantic_path(self, entity1: str, entity2: str, semantic_constraints: Optional[Dict] = None) -> Optional[List[Tuple[str, str]]]:
        """
        Find shortest semantic path between two entities with optional constraints
        
        Args:
            entity1: Start entity
            entity2: End entity
            semantic_constraints: Optional constraints (entity_types, relationship_types)
            
        Returns:
            List of (entity, relationship_type) tuples representing the path, or None if no path
        """
        
        if entity1 not in self.nodes or entity2 not in self.nodes:
            return None
        
        if entity1 == entity2:
            return [(entity1, 'self')]
        
        # Initialize BFS
        queue = [(entity1, [(entity1, 'start')])]  # (current_entity, path)
        visited = {entity1}
        constraints = semantic_constraints or {}
        
        allowed_entity_types = constraints.get('entity_types')
        allowed_relationship_types = constraints.get('relationship_types')
        
        while queue:
            current_entity, path = queue.pop(0)
            
            # Get all edges from current entity
            incident_edges = self.incidences.get_node_edges(current_entity)
            
            for edge in incident_edges:
                edge_nodes = self.incidences.get_edge_nodes(edge)
                edge_properties = self.properties.get_edge_properties(edge)
                relationship_type = edge_properties.get('edge_type', 'connects')
                
                # Check relationship type constraints
                if allowed_relationship_types and relationship_type not in allowed_relationship_types:
                    continue
                
                for neighbor in edge_nodes:
                    if neighbor == current_entity or neighbor in visited:
                        continue
                    
                    # Check entity type constraints
                    if allowed_entity_types:
                        neighbor_type = self.get_entity_type(neighbor)
                        if neighbor_type and neighbor_type not in allowed_entity_types:
                            continue
                    
                    # Build new path
                    new_path = path + [(neighbor, relationship_type)]
                    
                    # Found target
                    if neighbor == entity2:
                        return new_path[1:]  # Remove start marker
                    
                    # Add to queue for further exploration
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, new_path))
        
        return None  # No path found
    
    @performance_monitor("kg_extract_ontology")
    def extract_ontology(self, include_instances: bool = False) -> Dict[str, Any]:
        """
        Extract ontology schema from the knowledge graph structure
        
        Args:
            include_instances: Whether to include instance counts in the ontology
            
        Returns:
            Dictionary representing the extracted ontology
        """
        
        ontology = {
            'entity_classes': {},
            'relationship_classes': {},
            'class_hierarchy': {},
            'domain_ranges': {},
            'statistics': {},
            'extraction_metadata': {
                'timestamp': time.time(),
                'total_entities': len(self.nodes),
                'total_relationships': len(self.edges)
            }
        }
        
        # Extract entity classes
        for entity_type, entities in self._entity_types.items():
            class_info = {
                'type': entity_type,
                'properties': set(),
                'superclasses': [],
                'subclasses': []
            }
            
            if include_instances:
                class_info['instance_count'] = len(entities)
                class_info['sample_instances'] = list(entities)[:5]  # Sample for reference
            
            # Analyze properties across instances
            for entity in list(entities)[:10]:  # Sample for performance
                props = self.properties.get_node_properties(entity)
                if props:
                    class_info['properties'].update(props.keys())
            
            class_info['properties'] = list(class_info['properties'])
            ontology['entity_classes'][entity_type] = class_info
        
        # Extract relationship classes
        for rel_type, relationships in self._relationship_types.items():
            rel_info = {
                'type': rel_type,
                'domain_entities': set(),
                'range_entities': set(),
                'properties': set()
            }
            
            if include_instances:
                rel_info['instance_count'] = len(relationships)
            
            # Analyze relationship patterns
            for rel in list(relationships)[:10]:  # Sample for performance
                rel_nodes = self.incidences.get_edge_nodes(rel)
                if len(rel_nodes) >= 2:
                    # Identify domain and range
                    source_type = self.get_entity_type(rel_nodes[0])
                    target_type = self.get_entity_type(rel_nodes[-1])
                    
                    if source_type:
                        rel_info['domain_entities'].add(source_type)
                    if target_type:
                        rel_info['range_entities'].add(target_type)
                
                # Get relationship properties
                rel_props = self.properties.get_edge_properties(rel)
                if rel_props:
                    rel_info['properties'].update(rel_props.keys())
            
            # Convert sets to lists for JSON serialization
            rel_info['domain_entities'] = list(rel_info['domain_entities'])
            rel_info['range_entities'] = list(rel_info['range_entities'])
            rel_info['properties'] = list(rel_info['properties'])
            
            ontology['relationship_classes'][rel_type] = rel_info
        
        # Infer class hierarchy based on naming patterns and relationships
        entity_types = list(self._entity_types.keys())
        for entity_type in entity_types:
            # Simple heuristic: if one type name contains another, it might be a subclass
            potential_supertypes = []
            potential_subtypes = []
            
            for other_type in entity_types:
                if other_type != entity_type:
                    if entity_type.lower() in other_type.lower():
                        potential_subtypes.append(other_type)
                    elif other_type.lower() in entity_type.lower():
                        potential_supertypes.append(other_type)
            
            if potential_supertypes or potential_subtypes:
                ontology['class_hierarchy'][entity_type] = {
                    'supertypes': potential_supertypes,
                    'subtypes': potential_subtypes
                }
        
        # Calculate ontology statistics
        ontology['statistics'] = {
            'total_entity_classes': len(ontology['entity_classes']),
            'total_relationship_classes': len(ontology['relationship_classes']),
            'total_hierarchical_relationships': len(ontology['class_hierarchy']),
            'avg_entities_per_class': np.mean([len(entities) for entities in self._entity_types.values()]) if self._entity_types else 0,
            'avg_relationships_per_class': np.mean([len(rels) for rels in self._relationship_types.values()]) if self._relationship_types else 0
        }
        
        return ontology
    
    def add_ontology(self, ontology_data: Dict[str, Any], namespace: Optional[str] = None) -> bool:
        """
        Add ontology schema to the knowledge graph
        
        Args:
            ontology_data: Ontology schema data
            namespace: Optional namespace for the ontology
            
        Returns:
            Success status
        """
        try:
            if namespace:
                self.add_namespace(namespace, ontology_data.get('base_uri', ''))
            
            # Merge with existing ontology schema
            self.ontology_schema.update(ontology_data)
            
            # Add ontology entities and relationships
            if 'classes' in ontology_data:
                for class_name, class_info in ontology_data['classes'].items():
                    self.add_node(class_name, properties=class_info, entity_type='OntologyClass')
            
            if 'properties' in ontology_data:
                for prop_name, prop_info in ontology_data['properties'].items():
                    if 'domain' in prop_info and 'range' in prop_info:
                        rel_id = f"ontology_property_{prop_name}"
                        self.add_edge(rel_id, [prop_info['domain'], prop_info['range']], 
                                    properties=prop_info, edge_type='ontologyProperty')
            
            logger.info(f"Added ontology with {len(ontology_data)} elements")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add ontology: {e}")
            return False
    
    def add_provenance(self, entity_id: str, provenance_data: Dict[str, Any]) -> bool:
        """
        Add provenance information for an entity
        
        Args:
            entity_id: Entity to add provenance for
            provenance_data: Provenance metadata
            
        Returns:
            Success status
        """
        try:
            if entity_id not in self.nodes:
                logger.warning(f"Entity {entity_id} not found")
                return False
            
            # Get existing properties
            existing_props = self.properties.get_node_properties(entity_id)
            
            # Add provenance information
            provenance_info = {
                'provenance_source': provenance_data.get('source', 'unknown'),
                'provenance_timestamp': provenance_data.get('timestamp', str(datetime.now().isoformat())),
                'provenance_method': provenance_data.get('method', 'manual'),
                'provenance_confidence': provenance_data.get('confidence', 1.0),
                'provenance_metadata': provenance_data
            }
            
            # Update properties
            existing_props.update(provenance_info)
            self.properties.set_node_properties(entity_id, existing_props)
            
            logger.info(f"Added provenance for entity {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add provenance: {e}")
            return False
    
    def add_temporal_fact(self, subject: str, predicate: str, object_val: str, 
                         temporal_info: Dict[str, Any]) -> str:
        """
        Add a temporal fact to the knowledge graph
        
        Args:
            subject: Subject entity
            predicate: Relationship type
            object_val: Object entity
            temporal_info: Temporal metadata (start_time, end_time, etc.)
            
        Returns:
            Temporal fact ID
        """
        try:
            fact_id = f"temporal_fact_{len(self.edges)}_{subject}_{predicate}_{object_val}"
            
            # Add temporal relationship
            temporal_properties = {
                'temporal_start': temporal_info.get('start_time'),
                'temporal_end': temporal_info.get('end_time'),
                'temporal_duration': temporal_info.get('duration'),
                'temporal_valid_time': temporal_info.get('valid_time'),
                'temporal_transaction_time': temporal_info.get('transaction_time'),
                'is_temporal_fact': True
            }
            
            self.add_edge(fact_id, [subject, object_val], 
                         properties=temporal_properties, edge_type=predicate)
            
            logger.info(f"Added temporal fact: {fact_id}")
            return fact_id
            
        except Exception as e:
            logger.error(f"Failed to add temporal fact: {e}")
            return ""
    
    def get_entity(self, entity_id: str, include_relationships: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get complete entity information
        
        Args:
            entity_id: Entity identifier
            include_relationships: Whether to include relationship information
            
        Returns:
            Entity data with properties and relationships
        """
        if entity_id not in self.nodes:
            return None
        
        try:
            entity_data = {
                'entity_id': entity_id,
                'entity_type': self.get_entity_type(entity_id),
                'properties': self.properties.get_node_properties(entity_id),
                'relationships': []
            }
            
            if include_relationships:
                # Get all relationships
                incident_edges = self.incidences.get_node_edges(entity_id)
                
                for edge_id in incident_edges:
                    edge_nodes = self.incidences.get_edge_nodes(edge_id)
                    edge_props = self.properties.get_edge_properties(edge_id)
                    
                    relationship = {
                        'edge_id': edge_id,
                        'relationship_type': edge_props.get('edge_type', 'connects'),
                        'connected_entities': [n for n in edge_nodes if n != entity_id],
                        'properties': edge_props
                    }
                    entity_data['relationships'].append(relationship)
            
            return entity_data
            
        except Exception as e:
            logger.error(f"Failed to get entity {entity_id}: {e}")
            return None
    
    def remove_entity(self, entity_id: str, remove_relationships: bool = True) -> bool:
        """
        Remove an entity from the knowledge graph
        
        Args:
            entity_id: Entity to remove
            remove_relationships: Whether to remove associated relationships
            
        Returns:
            Success status
        """
        try:
            if entity_id not in self.nodes:
                return False
            
            if remove_relationships:
                # Remove all edges containing this entity
                incident_edges = list(self.incidences.get_node_edges(entity_id))
                for edge_id in incident_edges:
                    self.remove_edge(edge_id)
            
            # Remove from semantic indexes
            entity_type = self.get_entity_type(entity_id)
            if entity_type and entity_type in self._entity_types:
                self._entity_types[entity_type].discard(entity_id)
            
            if entity_id in self._type_cache:
                del self._type_cache[entity_id]
            
            # Remove node
            self.remove_node(entity_id)
            
            logger.info(f"Removed entity {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove entity {entity_id}: {e}")
            return False
    
    def remove_relationship(self, relationship_id: str) -> bool:
        """
        Remove a relationship from the knowledge graph
        
        Args:
            relationship_id: Relationship edge ID to remove
            
        Returns:
            Success status
        """
        try:
            if relationship_id not in self.edges:
                return False
            
            # Remove from semantic indexes
            edge_props = self.properties.get_edge_properties(relationship_id)
            edge_type = edge_props.get('edge_type')
            
            if edge_type and edge_type in self._relationship_types:
                self._relationship_types[edge_type].discard(relationship_id)
            
            # Remove edge
            self.remove_edge(relationship_id)
            
            logger.info(f"Removed relationship {relationship_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove relationship {relationship_id}: {e}")
            return False
    
    def pagerank(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> Dict[Any, float]:
        """
        Calculate PageRank values for all nodes
        
        Args:
            alpha: Damping parameter
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Dictionary mapping nodes to PageRank values
        """
        nodes = list(self.nodes)
        n = len(nodes)
        
        if n == 0:
            return {}
        
        # Initialize PageRank values
        pagerank_values = {node: 1.0 / n for node in nodes}
        
        # Create adjacency information
        node_to_neighbors = {}
        for node in nodes:
            neighbors = set()
            for edge in self.incidences.get_node_edges(node):
                edge_nodes = self.incidences.get_edge_nodes(edge)
                neighbors.update(n for n in edge_nodes if n != node)
            node_to_neighbors[node] = neighbors
        
        # Power iteration
        for iteration in range(max_iter):
            new_pagerank = {}
            
            for node in nodes:
                # Base value from random jumps
                rank_sum = (1.0 - alpha) / n
                
                # Add rank from incoming links
                for source_node, source_neighbors in node_to_neighbors.items():
                    if node in source_neighbors:
                        out_degree = len(source_neighbors)
                        if out_degree > 0:
                            rank_sum += alpha * pagerank_values[source_node] / out_degree
                
                new_pagerank[node] = rank_sum
            
            # Check convergence
            max_change = max(abs(new_pagerank[node] - pagerank_values[node]) 
                           for node in nodes)
            
            pagerank_values = new_pagerank
            
            if max_change < tol:
                break
        
        return pagerank_values
    
    def extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Extract entities from text using NLP techniques
        
        Args:
            text: Input text for entity extraction
            entity_types: Specific entity types to extract
            
        Returns:
            List of extracted entity dictionaries
        """
        try:
            # Import spaCy if available
            nlp_lib = safe_import('spacy')
            if not nlp_lib:
                logger.warning("spaCy not available, using simple pattern matching")
                return self._simple_entity_extraction(text, entity_types)
            
            # Use spaCy for NER
            try:
                nlp = nlp_lib.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using simple extraction")
                return self._simple_entity_extraction(text, entity_types)
            
            doc = nlp(text)
            entities = []
            
            for ent in doc.ents:
                if entity_types is None or ent.label_ in entity_types:
                    entity_dict = {
                        'text': ent.text,
                        'label': ent.label_,
                        'start_char': ent.start_char,
                        'end_char': ent.end_char,
                        'confidence': 0.8  # Default confidence for spaCy
                    }
                    entities.append(entity_dict)
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def _simple_entity_extraction(self, text: str, entity_types: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Simple pattern-based entity extraction fallback"""
        import re
        
        entities = []
        
        # Simple patterns for common entity types
        patterns = {
            'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'ORG': r'\b[A-Z][a-zA-Z\s]+ (Inc|Corp|Ltd|LLC|Company)\b',
            'GPE': r'\b[A-Z][a-zA-Z]+ (City|State|Country)\b',
            'DATE': r'\b\d{1,2}\/\d{1,2}\/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b'
        }
        
        for entity_type, pattern in patterns.items():
            if entity_types is None or entity_type in entity_types:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity_dict = {
                        'text': match.group(),
                        'label': entity_type,
                        'start_char': match.start(),
                        'end_char': match.end(),
                        'confidence': 0.5  # Lower confidence for simple patterns
                    }
                    entities.append(entity_dict)
        
        return entities
    
    def extract_relations(self, text: str, entity_pairs: Optional[List[Tuple[str, str]]] = None) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities in text
        
        Args:
            text: Input text
            entity_pairs: Specific entity pairs to check for relations
            
        Returns:
            List of extracted relation dictionaries
        """
        try:
            # Simple relation extraction using patterns
            relations = []
            
            # Common relation patterns
            relation_patterns = [
                (r'(\w+) is the (\w+) of (\w+)', 'is_role_of'),
                (r'(\w+) works for (\w+)', 'employed_by'),
                (r'(\w+) founded (\w+)', 'founded'),
                (r'(\w+) acquired (\w+)', 'acquired'),
                (r'(\w+) partnered with (\w+)', 'partnered_with'),
                (r'(\w+) located in (\w+)', 'located_in')
            ]
            
            import re
            for pattern, relation_type in relation_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        relation_dict = {
                            'subject': groups[0],
                            'predicate': relation_type,
                            'object': groups[-1],
                            'text': match.group(),
                            'confidence': 0.6
                        }
                        
                        # Check if this matches requested entity pairs
                        if entity_pairs:
                            subject_object = (groups[0], groups[-1])
                            if subject_object in entity_pairs:
                                relations.append(relation_dict)
                        else:
                            relations.append(relation_dict)
            
            return relations
            
        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            return []
    
    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract RDF-style triples from text
        
        Args:
            text: Input text
            
        Returns:
            List of (subject, predicate, object) triples
        """
        try:
            relations = self.extract_relations(text)
            triples = []
            
            for relation in relations:
                triple = (
                    relation['subject'],
                    relation['predicate'], 
                    relation['object']
                )
                triples.append(triple)
            
            return triples
            
        except Exception as e:
            logger.error(f"Triple extraction failed: {e}")
            return []
    
    def extract_concepts(self, text: str, min_frequency: int = 2) -> List[Dict[str, Any]]:
        """
        Extract key concepts from text
        
        Args:
            text: Input text
            min_frequency: Minimum frequency for concept extraction
            
        Returns:
            List of concept dictionaries
        """
        try:
            # Simple concept extraction using NLP
            nlp_lib = safe_import('spacy')
            if nlp_lib:
                try:
                    nlp = nlp_lib.load("en_core_web_sm")
                    doc = nlp(text)
                    
                    # Extract noun phrases as concepts
                    concepts = {}
                    for chunk in doc.noun_chunks:
                        concept_text = chunk.text.lower().strip()
                        if len(concept_text) > 2:  # Filter out very short concepts
                            concepts[concept_text] = concepts.get(concept_text, 0) + 1
                    
                    # Filter by frequency and convert to list
                    concept_list = []
                    for concept, freq in concepts.items():
                        if freq >= min_frequency:
                            concept_list.append({
                                'concept': concept,
                                'frequency': freq,
                                'type': 'noun_phrase'
                            })
                    
                    # Sort by frequency
                    concept_list.sort(key=lambda x: x['frequency'], reverse=True)
                    return concept_list
                    
                except OSError:
                    pass  # Fall through to simple extraction
            
            # Simple word frequency-based concept extraction
            import re
            from collections import Counter
            
            # Extract words (simple tokenization)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            word_freq = Counter(words)
            
            concepts = []
            for word, freq in word_freq.most_common():
                if freq >= min_frequency:
                    concepts.append({
                        'concept': word,
                        'frequency': freq,
                        'type': 'word'
                    })
            
            return concepts
            
        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            return []
    
    def named_entity_recognition(self, text: str) -> Dict[str, List[str]]:
        """
        Perform named entity recognition on text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping entity types to lists of entities
        """
        try:
            entities = self.extract_entities(text)
            ner_results = defaultdict(list)
            
            for entity in entities:
                entity_type = entity['label']
                entity_text = entity['text']
                ner_results[entity_type].append(entity_text)
            
            return dict(ner_results)
            
        except Exception as e:
            logger.error(f"NER failed: {e}")
            return {}
    
    def entity_linking(self, entities: List[str], knowledge_base: Optional[Dict] = None) -> Dict[str, List[str]]:
        """
        Link entities to knowledge base entries
        
        Args:
            entities: List of entity mentions
            knowledge_base: External knowledge base for linking
            
        Returns:
            Dictionary mapping entities to potential KB links
        """
        try:
            linked_entities = {}
            
            for entity in entities:
                # Check if entity exists in current graph
                candidates = []
                
                # Look for exact matches in current nodes
                if entity in self.nodes:
                    candidates.append(f"internal:{entity}")
                
                # Look for partial matches
                for node in self.nodes:
                    if isinstance(node, str) and entity.lower() in node.lower():
                        candidates.append(f"internal:{node}")
                
                # If external knowledge base provided, check it
                if knowledge_base:
                    for kb_entry, kb_info in knowledge_base.items():
                        if entity.lower() in kb_entry.lower():
                            candidates.append(f"external:{kb_entry}")
                
                linked_entities[entity] = candidates[:5]  # Limit to top 5
            
            return linked_entities
            
        except Exception as e:
            logger.error(f"Entity linking failed: {e}")
            return {}
    
    def entity_embeddings(self, embedding_model: str = 'simple') -> Dict[str, np.ndarray]:
        """
        Generate embeddings for entities in the knowledge graph
        
        Args:
            embedding_model: Type of embedding model to use
            
        Returns:
            Dictionary mapping entities to embedding vectors
        """
        try:
            embeddings = {}
            
            if embedding_model == 'simple':
                # Simple structural embeddings based on connections
                for entity in self.nodes:
                    # Get entity connections
                    connections = self.incidences.get_node_edges(entity)
                    
                    # Create simple feature vector
                    features = [
                        len(connections),  # Degree
                        len([e for e in connections if 'type' in self.properties.get_edge_properties(e)]),  # Typed edges
                        len(self.neighbors(entity)) if hasattr(self, 'neighbors') else 0,  # Neighbors
                    ]
                    
                    # Add entity type information if available
                    entity_props = self.properties.get_node_properties(entity)
                    entity_type = entity_props.get('entity_type', 'unknown') if entity_props else 'unknown'
                    
                    # Simple one-hot encoding for entity types
                    type_features = [1 if entity_type == t else 0 
                                   for t in ['person', 'organization', 'location', 'concept']]
                    features.extend(type_features)
                    
                    embeddings[entity] = np.array(features, dtype=float)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Entity embeddings failed: {e}")
            return {}
    
    def entity_neighborhood(self, entity_id: str, radius: int = 2) -> Dict[str, Any]:
        """
        Get the neighborhood of an entity up to specified radius
        
        Args:
            entity_id: Entity to get neighborhood for
            radius: Maximum radius of neighborhood
            
        Returns:
            Dictionary containing neighborhood information
        """
        try:
            if entity_id not in self.nodes:
                return {}
            
            neighborhood = {
                'center': entity_id,
                'radius': radius,
                'nodes': set([entity_id]),
                'edges': set(),
                'layers': {}
            }
            
            current_layer = {entity_id}
            
            for layer in range(radius):
                next_layer = set()
                layer_edges = set()
                
                for node in current_layer:
                    # Get edges connected to this node
                    node_edges = self.incidences.get_node_edges(node)
                    layer_edges.update(node_edges)
                    
                    # Get neighbors through these edges
                    for edge in node_edges:
                        edge_nodes = self.incidences.get_edge_nodes(edge)
                        next_layer.update(edge_nodes)
                
                # Remove already visited nodes
                next_layer -= neighborhood['nodes']
                
                if not next_layer:
                    break
                
                neighborhood['layers'][layer + 1] = list(next_layer)
                neighborhood['nodes'].update(next_layer)
                neighborhood['edges'].update(layer_edges)
                
                current_layer = next_layer
            
            # Convert sets to lists for JSON serialization
            neighborhood['nodes'] = list(neighborhood['nodes'])
            neighborhood['edges'] = list(neighborhood['edges'])
            
            return neighborhood
            
        except Exception as e:
            logger.error(f"Entity neighborhood extraction failed: {e}")
            return {}
    
    def concept_extraction(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract domain-specific concepts from the knowledge graph
        
        Args:
            domain: Specific domain to focus on (optional)
            
        Returns:
            List of extracted concept dictionaries
        """
        try:
            concepts = []
            
            # Analyze entity types and relationships for concepts
            entity_types = defaultdict(int)
            relationship_types = defaultdict(int)
            
            # Count entity and relationship types
            for node in self.nodes:
                node_props = self.properties.get_node_properties(node)
                if node_props and 'entity_type' in node_props:
                    entity_type = node_props['entity_type']
                    if domain is None or domain.lower() in entity_type.lower():
                        entity_types[entity_type] += 1
            
            for edge in self.edges:
                edge_props = self.properties.get_edge_properties(edge)
                if edge_props and 'relationship_type' in edge_props:
                    rel_type = edge_props['relationship_type']
                    if domain is None or domain.lower() in rel_type.lower():
                        relationship_types[rel_type] += 1
            
            # Create concept entries
            for concept_type, count in entity_types.items():
                concepts.append({
                    'concept': concept_type,
                    'type': 'entity_type',
                    'frequency': count,
                    'domain': domain or 'general'
                })
            
            for concept_type, count in relationship_types.items():
                concepts.append({
                    'concept': concept_type,
                    'type': 'relationship_type',
                    'frequency': count,
                    'domain': domain or 'general'
                })
            
            # Sort by frequency
            concepts.sort(key=lambda x: x['frequency'], reverse=True)
            
            return concepts
            
        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            return []
    
    def class_hierarchy(self) -> Dict[str, List[str]]:
        """
        Extract class hierarchy from the knowledge graph
        
        Returns:
            Dictionary representing the class hierarchy
        """
        try:
            hierarchy = defaultdict(list)
            
            # Look for subclass relationships
            for edge in self.edges:
                edge_props = self.properties.get_edge_properties(edge)
                if edge_props:
                    rel_type = edge_props.get('relationship_type', '')
                    if 'subclass' in rel_type.lower() or 'isa' in rel_type.lower():
                        edge_nodes = self.incidences.get_edge_nodes(edge)
                        if len(edge_nodes) >= 2:
                            # Assume first node is subclass, second is superclass
                            subclass, superclass = edge_nodes[0], edge_nodes[1]
                            hierarchy[superclass].append(subclass)
            
            # Also look for entity type hierarchies
            entity_types = set()
            for node in self.nodes:
                node_props = self.properties.get_node_properties(node)
                if node_props and 'entity_type' in node_props:
                    entity_types.add(node_props['entity_type'])
            
            # Simple heuristic hierarchy for common types
            type_hierarchy = {
                'Thing': ['Person', 'Organization', 'Location', 'Concept'],
                'Person': ['Researcher', 'Author', 'Executive'],
                'Organization': ['Company', 'University', 'Government'],
                'Location': ['City', 'Country', 'Building']
            }
            
            for parent, children in type_hierarchy.items():
                if parent in entity_types:
                    hierarchy[parent].extend([c for c in children if c in entity_types])
            
            return dict(hierarchy)
            
        except Exception as e:
            logger.error(f"Class hierarchy extraction failed: {e}")
            return {}
    
    def infer_relationships(self, entity1: str, entity2: str, max_path_length: int = 3) -> List[Dict[str, Any]]:
        """
        Infer possible relationships between two entities
        
        Args:
            entity1: First entity
            entity2: Second entity  
            max_path_length: Maximum path length to consider
            
        Returns:
            List of inferred relationship dictionaries
        """
        try:
            if entity1 not in self.nodes or entity2 not in self.nodes:
                return []
            
            inferences = []
            
            # Direct relationship check
            for edge in self.edges:
                edge_nodes = self.incidences.get_edge_nodes(edge)
                if entity1 in edge_nodes and entity2 in edge_nodes:
                    edge_props = self.properties.get_edge_properties(edge)
                    inferences.append({
                        'type': 'direct',
                        'relationship': edge_props.get('relationship_type', 'connected'),
                        'path_length': 1,
                        'confidence': 0.9,
                        'path': [entity1, entity2]
                    })
            
            # Indirect relationship inference (simplified)
            # Look for common neighbors
            neighbors1 = set()
            neighbors2 = set()
            
            for edge in self.incidences.get_node_edges(entity1):
                edge_nodes = self.incidences.get_edge_nodes(edge)
                neighbors1.update(n for n in edge_nodes if n != entity1)
            
            for edge in self.incidences.get_node_edges(entity2):
                edge_nodes = self.incidences.get_edge_nodes(edge)
                neighbors2.update(n for n in edge_nodes if n != entity2)
            
            common_neighbors = neighbors1 & neighbors2
            
            for common in common_neighbors:
                inferences.append({
                    'type': 'indirect',
                    'relationship': 'connected_through',
                    'path_length': 2,
                    'confidence': 0.6,
                    'path': [entity1, common, entity2],
                    'mediator': common
                })
            
            return inferences
            
        except Exception as e:
            logger.error(f"Relationship inference failed: {e}")
            return []
    
    def embedding_based_reasoning(self, query_entity: str, reasoning_type: str = 'similarity') -> List[Dict[str, Any]]:
        """
        Perform embedding-based reasoning
        
        Args:
            query_entity: Entity to reason about
            reasoning_type: Type of reasoning ('similarity', 'analogy', 'composition')
            
        Returns:
            List of reasoning results
        """
        try:
            if query_entity not in self.nodes:
                return []
            
            # Get embeddings for all entities
            embeddings = self.entity_embeddings()
            
            if query_entity not in embeddings:
                return []
            
            query_embedding = embeddings[query_entity]
            results = []
            
            if reasoning_type == 'similarity':
                # Find similar entities based on embedding similarity
                similarities = []
                
                for entity, embedding in embeddings.items():
                    if entity != query_entity:
                        # Compute cosine similarity
                        similarity = np.dot(query_embedding, embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-8
                        )
                        similarities.append((entity, similarity))
                
                # Sort by similarity and take top results
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                for entity, sim_score in similarities[:10]:
                    results.append({
                        'entity': entity,
                        'reasoning_type': 'similarity',
                        'score': float(sim_score),
                        'explanation': f"Similar to {query_entity} based on structural properties"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Embedding-based reasoning failed: {e}")
            return []
    
    def temporal_query(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Query temporal facts within a time range
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of temporal facts
        """
        try:
            temporal_facts = []
            
            # Search through edges and nodes for temporal information
            for edge in self.edges:
                edge_props = self.properties.get_edge_properties(edge)
                if edge_props and 'timestamp' in edge_props:
                    timestamp = edge_props['timestamp']
                    
                    # Convert timestamp to datetime if needed
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.fromisoformat(timestamp)
                        except:
                            continue
                    
                    # Check if within range
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue
                    
                    edge_nodes = self.incidences.get_edge_nodes(edge)
                    temporal_facts.append({
                        'edge_id': edge,
                        'nodes': edge_nodes,
                        'timestamp': timestamp,
                        'properties': edge_props
                    })
            
            return temporal_facts
            
        except Exception as e:
            logger.error(f"Temporal query failed: {e}")
            return []
    
    def temporal_reasoning(self, entity: str, reasoning_type: str = 'evolution') -> List[Dict[str, Any]]:
        """
        Perform temporal reasoning about an entity
        
        Args:
            entity: Entity to reason about temporally
            reasoning_type: Type of temporal reasoning
            
        Returns:
            List of temporal reasoning results
        """
        try:
            if entity not in self.nodes:
                return []
            
            results = []
            
            # Get all temporal facts involving this entity
            entity_facts = []
            for edge in self.incidences.get_node_edges(entity):
                edge_props = self.properties.get_edge_properties(edge)
                if edge_props and 'timestamp' in edge_props:
                    entity_facts.append({
                        'edge': edge,
                        'timestamp': edge_props['timestamp'],
                        'properties': edge_props
                    })
            
            # Sort by timestamp
            entity_facts.sort(key=lambda x: x['timestamp'])
            
            if reasoning_type == 'evolution':
                # Analyze how entity evolved over time
                for i in range(len(entity_facts) - 1):
                    current = entity_facts[i]
                    next_fact = entity_facts[i + 1]
                    
                    results.append({
                        'type': 'temporal_transition',
                        'from_state': current['properties'],
                        'to_state': next_fact['properties'],
                        'time_span': next_fact['timestamp'] - current['timestamp'],
                        'entity': entity
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Temporal reasoning failed: {e}")
            return []
    
    def time_based_inference(self, query: str, reference_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Perform time-based inference
        
        Args:
            query: Query string
            reference_time: Reference time for inference
            
        Returns:
            List of inference results
        """
        try:
            if reference_time is None:
                reference_time = datetime.now()
            
            inferences = []
            
            # Simple time-based inference patterns
            patterns = [
                ('before', lambda t, ref: t < ref),
                ('after', lambda t, ref: t > ref),
                ('during', lambda t, ref: abs((t - ref).days) < 30),  # Within 30 days
            ]
            
            for pattern_name, pattern_func in patterns:
                if pattern_name in query.lower():
                    temporal_facts = self.temporal_query()
                    
                    for fact in temporal_facts:
                        fact_time = fact['timestamp']
                        if isinstance(fact_time, str):
                            try:
                                fact_time = datetime.fromisoformat(fact_time)
                            except:
                                continue
                        
                        if pattern_func(fact_time, reference_time):
                            inferences.append({
                                'pattern': pattern_name,
                                'fact': fact,
                                'reference_time': reference_time,
                                'match_reason': f"Fact occurred {pattern_name} reference time"
                            })
            
            return inferences
            
        except Exception as e:
            logger.error(f"Time-based inference failed: {e}")
            return []
    
    def get_ontology(self) -> Dict[str, Any]:
        """Get the ontology schema for this knowledge graph"""
        return getattr(self, 'ontology_schema', {})
    
    def get_relationship(self, subject: str, object_entity: str) -> Optional[Dict[str, Any]]:
        """
        Get relationship between two entities
        
        Args:
            subject: Subject entity
            object_entity: Object entity
            
        Returns:
            Relationship information if exists
        """
        try:
            if subject not in self.nodes or object_entity not in self.nodes:
                return None
            
            # Find edges connecting both entities
            for edge in self.edges:
                edge_nodes = self.incidences.get_edge_nodes(edge)
                if subject in edge_nodes and object_entity in edge_nodes:
                    edge_props = self.properties.get_edge_properties(edge)
                    return {
                        'edge_id': edge,
                        'subject': subject,
                        'object': object_entity,
                        'properties': edge_props
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Get relationship failed: {e}")
            return None
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get schema information for the knowledge graph
        
        Returns:
            Dictionary containing schema information
        """
        try:
            schema = {
                'entity_types': set(),
                'relationship_types': set(),
                'properties': {
                    'node_properties': set(),
                    'edge_properties': set()
                },
                'statistics': {
                    'num_entities': len(self.nodes),
                    'num_relationships': len(self.edges)
                }
            }
            
            # Analyze entity types
            for node in self.nodes:
                node_props = self.properties.get_node_properties(node)
                if node_props:
                    if 'entity_type' in node_props:
                        schema['entity_types'].add(node_props['entity_type'])
                    schema['properties']['node_properties'].update(node_props.keys())
            
            # Analyze relationship types
            for edge in self.edges:
                edge_props = self.properties.get_edge_properties(edge)
                if edge_props:
                    if 'relationship_type' in edge_props:
                        schema['relationship_types'].add(edge_props['relationship_type'])
                    schema['properties']['edge_properties'].update(edge_props.keys())
            
            # Convert sets to lists for JSON serialization
            schema['entity_types'] = list(schema['entity_types'])
            schema['relationship_types'] = list(schema['relationship_types'])
            schema['properties']['node_properties'] = list(schema['properties']['node_properties'])
            schema['properties']['edge_properties'] = list(schema['properties']['edge_properties'])
            
            return schema
            
        except Exception as e:
            logger.error(f"Get schema failed: {e}")
            return {}
    
    def sparql_query(self, query_string: str) -> List[Dict[str, Any]]:
        """
        Execute a SPARQL-like query on the knowledge graph
        
        Args:
            query_string: SPARQL-like query string
            
        Returns:
            Query results
        """
        try:
            # Simple SPARQL-like query parsing and execution
            # This is a very basic implementation
            
            results = []
            
            # Handle SELECT queries
            if 'SELECT' in query_string.upper():
                # Very simple pattern matching for basic queries
                if '?s ?p ?o' in query_string:
                    # Return all triples
                    for edge in self.edges:
                        edge_nodes = self.incidences.get_edge_nodes(edge)
                        edge_props = self.properties.get_edge_properties(edge)
                        
                        if len(edge_nodes) >= 2:
                            result = {
                                's': edge_nodes[0],
                                'p': edge_props.get('relationship_type', 'connected_to'),
                                'o': edge_nodes[1]
                            }
                            results.append(result)
                
                # Handle entity type queries
                elif 'a' in query_string or 'rdf:type' in query_string:
                    for node in self.nodes:
                        node_props = self.properties.get_node_properties(node)
                        if node_props and 'entity_type' in node_props:
                            results.append({
                                's': node,
                                'p': 'rdf:type',
                                'o': node_props['entity_type']
                            })
            
            return results
            
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return []
    
    def get_provenance(self, entity_or_relation: str) -> Optional[Dict[str, Any]]:
        """
        Get provenance information for an entity or relation
        
        Args:
            entity_or_relation: ID of entity or relation
            
        Returns:
            Provenance information if available
        """
        try:
            # Check if it's a node
            if entity_or_relation in self.nodes:
                node_props = self.properties.get_node_properties(entity_or_relation)
                return node_props.get('provenance') if node_props else None
            
            # Check if it's an edge
            if entity_or_relation in self.edges:
                edge_props = self.properties.get_edge_properties(entity_or_relation)
                return edge_props.get('provenance') if edge_props else None
            
            return None
            
        except Exception as e:
            logger.error(f"Get provenance failed: {e}")
            return None
    
    def validate_schema(self, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate the knowledge graph against a schema
        
        Args:
            schema: Schema to validate against (uses internal schema if None)
            
        Returns:
            Validation results
        """
        try:
            if schema is None:
                schema = self.get_schema()
            
            validation_results = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'statistics': {}
            }
            
            # Basic validation checks
            expected_entity_types = set(schema.get('entity_types', []))
            expected_relation_types = set(schema.get('relationship_types', []))
            
            found_entity_types = set()
            found_relation_types = set()
            
            # Check entity types
            for node in self.nodes:
                node_props = self.properties.get_node_properties(node)
                if node_props and 'entity_type' in node_props:
                    entity_type = node_props['entity_type']
                    found_entity_types.add(entity_type)
                    
                    if expected_entity_types and entity_type not in expected_entity_types:
                        validation_results['warnings'].append(
                            f"Unexpected entity type: {entity_type}"
                        )
            
            # Check relationship types
            for edge in self.edges:
                edge_props = self.properties.get_edge_properties(edge)
                if edge_props and 'relationship_type' in edge_props:
                    rel_type = edge_props['relationship_type']
                    found_relation_types.add(rel_type)
                    
                    if expected_relation_types and rel_type not in expected_relation_types:
                        validation_results['warnings'].append(
                            f"Unexpected relationship type: {rel_type}"
                        )
            
            # Check for missing expected types
            missing_entity_types = expected_entity_types - found_entity_types
            missing_relation_types = expected_relation_types - found_relation_types
            
            if missing_entity_types:
                validation_results['warnings'].extend([
                    f"Missing expected entity type: {et}" for et in missing_entity_types
                ])
            
            if missing_relation_types:
                validation_results['warnings'].extend([
                    f"Missing expected relationship type: {rt}" for rt in missing_relation_types
                ])
            
            validation_results['statistics'] = {
                'entity_types_found': len(found_entity_types),
                'relationship_types_found': len(found_relation_types),
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges)
            }
            
            if validation_results['errors']:
                validation_results['valid'] = False
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return {'valid': False, 'errors': [str(e)]}

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"KnowledgeGraph(nodes={stats['basic_stats']['num_nodes']}, "
                f"edges={stats['basic_stats']['num_edges']}, "
                f"entity_types={stats['semantic_stats']['entity_types']}, "
                f"relationship_types={stats['semantic_stats']['relationship_types']})")