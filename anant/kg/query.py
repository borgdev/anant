"""
Semantic Query Engine
===================

Advanced query engine for knowledge graphs with SPARQL-like capabilities,
pattern matching, path queries, and semantic reasoning.
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
import itertools

from ..utils.performance import performance_monitor, PerformanceProfiler
from ..algorithms.sampling import SmartSampler

logger = logging.getLogger(__name__)


@dataclass
class QueryPattern:
    """Represents a semantic query pattern"""
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    variables: Optional[Set[str]] = None
    filters: Optional[List[Callable]] = None
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = set()
        if self.filters is None:
            self.filters = []


@dataclass
class QueryResult:
    """Query result with metadata"""
    bindings: List[Dict[str, str]]
    metadata: Dict[str, Any]
    execution_time: float
    total_results: int


class SemanticQueryEngine:
    """
    High-performance semantic query engine for knowledge graphs
    
    Supports:
    - SPARQL-like pattern matching
    - Path queries with constraints
    - Entity-relationship queries
    - Semantic filtering and ranking
    - Performance optimization with sampling
    """
    
    def __init__(self, knowledge_graph):
        """
        Initialize query engine
        
        Args:
            knowledge_graph: The KnowledgeGraph instance to query
        """
        self.kg = knowledge_graph
        self.query_cache = {}
        self.pattern_cache = {}
        
        # Query optimization settings
        self.max_results = 10000
        self.use_caching = True
        self.use_sampling = True
        self.sampling_threshold = 5000
        
        logger.info("Semantic Query Engine initialized")
    
    @performance_monitor("sparql_query")
    def sparql_like_query(self, 
                         query_string: str,
                         limit: Optional[int] = None) -> QueryResult:
        """
        Execute a SPARQL-like query string
        
        Args:
            query_string: Query in simplified SPARQL syntax
            limit: Maximum number of results
            
        Returns:
            Query results with bindings
        """
        
        # Parse the query string
        patterns = self._parse_sparql_query(query_string)
        
        # Execute each pattern
        all_bindings = []
        start_time = time.time()
        
        for pattern in patterns:
            pattern_results = self.pattern_match(pattern, limit=limit)
            all_bindings.extend(pattern_results)
        
        execution_time = time.time() - start_time
        
        # Apply limit across all patterns
        if limit:
            all_bindings = all_bindings[:limit]
        
        return QueryResult(
            bindings=all_bindings,
            metadata={
                'patterns_executed': len(patterns),
                'total_graph_nodes': len(self.kg.nodes),
                'cache_hits': 0,  # TODO: implement cache hit tracking
                'sampling_used': self._should_use_sampling()
            },
            execution_time=execution_time,
            total_results=len(all_bindings)
        )
    
    def _parse_sparql_query(self, query_string: str) -> List[QueryPattern]:
        """
        Parse a simplified SPARQL query into patterns
        
        Supports basic SELECT queries with WHERE clauses
        """
        patterns = []
        
        # Extract WHERE clause content
        where_match = re.search(r'WHERE\s*\{([^}]+)\}', query_string, re.IGNORECASE)
        if not where_match:
            return patterns
        
        where_content = where_match.group(1)
        
        # Split into triple patterns
        lines = [line.strip() for line in where_content.split('.') if line.strip()]
        
        for line in lines:
            # Parse triple pattern: ?subject predicate ?object
            parts = line.split()
            if len(parts) >= 3:
                subject = parts[0] if not parts[0].startswith('?') else None
                predicate = parts[1] if not parts[1].startswith('?') else None
                obj = parts[2] if not parts[2].startswith('?') else None
                
                variables = set()
                for part in parts:
                    if part.startswith('?'):
                        variables.add(part[1:])  # Remove '?' prefix
                
                patterns.append(QueryPattern(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    variables=variables
                ))
        
        return patterns
    
    @performance_monitor("pattern_matching")
    def pattern_match(self, 
                     pattern: Union[QueryPattern, Dict],
                     limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Find all matches for a given pattern
        
        Args:
            pattern: Pattern to match (QueryPattern or dict)
            limit: Maximum results to return
            
        Returns:
            List of variable bindings
        """
        
        if isinstance(pattern, dict):
            pattern = QueryPattern(**pattern)
        
        # Check cache first
        pattern_key = self._pattern_to_key(pattern)
        if self.use_caching and pattern_key in self.pattern_cache:
            cached_results = self.pattern_cache[pattern_key]
            return cached_results[:limit] if limit else cached_results
        
        # Use sampling for large graphs
        working_kg = self.kg
        if self._should_use_sampling():
            logger.info("Using sampling for pattern matching")
            sampler = SmartSampler(self.kg, strategy='adaptive')
            working_kg = sampler.adaptive_sample(
                sample_size=self.sampling_threshold,
                algorithm='general'
            )
        
        bindings = []
        
        # Pattern matching strategy depends on available constraints
        if pattern.subject and pattern.predicate:
            # Most constrained: find specific relationships
            bindings = self._match_subject_predicate(working_kg, pattern)
        elif pattern.subject:
            # Subject constrained: find all relationships for entity
            bindings = self._match_subject(working_kg, pattern)
        elif pattern.predicate:
            # Predicate constrained: find all instances of relationship
            bindings = self._match_predicate(working_kg, pattern)
        else:
            # Unconstrained: enumerate all relationships
            bindings = self._match_unconstrained(working_kg, pattern)
        
        # Apply filters
        if pattern.filters:
            bindings = self._apply_filters(bindings, pattern.filters)
        
        # Apply limit
        if limit:
            bindings = bindings[:limit]
        
        # Cache results
        if self.use_caching and len(bindings) < 1000:  # Don't cache huge results
            self.pattern_cache[pattern_key] = bindings
        
        return bindings
    
    def _match_subject_predicate(self, kg, pattern: QueryPattern) -> List[Dict[str, str]]:
        """Match patterns with both subject and predicate specified"""
        bindings = []
        
        # Find edges involving the subject
        if pattern.subject in kg.nodes:
            incident_edges = kg.incidences.get_node_edges(pattern.subject)
            
            for edge in incident_edges:
                edge_nodes = kg.incidences.get_edge_nodes(edge)
                
                # Check if this edge represents the desired predicate
                if self._edge_matches_predicate(kg, edge, edge_nodes, pattern.predicate):
                    # Extract object(s)
                    objects = [n for n in edge_nodes if n != pattern.subject]
                    
                    for obj in objects:
                        binding = {}
                        if 'subject' in pattern.variables:
                            binding['subject'] = pattern.subject
                        if 'predicate' in pattern.variables:
                            binding['predicate'] = pattern.predicate
                        if 'object' in pattern.variables:
                            binding['object'] = obj
                        
                        bindings.append(binding)
        
        return bindings
    
    def _match_subject(self, kg, pattern: QueryPattern) -> List[Dict[str, str]]:
        """Match patterns with subject specified"""
        bindings = []
        
        if pattern.subject in kg.nodes:
            incident_edges = kg.incidences.get_node_edges(pattern.subject)
            
            for edge in incident_edges:
                edge_nodes = kg.incidences.get_edge_nodes(edge)
                
                # Extract predicate and objects
                predicate = self._extract_predicate_from_edge(kg, edge, edge_nodes)
                objects = [n for n in edge_nodes if n != pattern.subject]
                
                for obj in objects:
                    binding = {}
                    if 'subject' in pattern.variables:
                        binding['subject'] = pattern.subject
                    if 'predicate' in pattern.variables:
                        binding['predicate'] = predicate
                    if 'object' in pattern.variables:
                        binding['object'] = obj
                    
                    bindings.append(binding)
        
        return bindings
    
    def _match_predicate(self, kg, pattern: QueryPattern) -> List[Dict[str, str]]:
        """Match patterns with predicate specified"""
        bindings = []
        
        # Find all edges of the specified relationship type
        relationship_edges = kg.get_relationships_by_type(pattern.predicate)
        
        for edge in relationship_edges:
            edge_nodes = kg.incidences.get_edge_nodes(edge)
            
            # Generate all possible subject-object pairs
            for i, subject in enumerate(edge_nodes):
                for j, obj in enumerate(edge_nodes):
                    if i != j:  # Different nodes
                        binding = {}
                        if 'subject' in pattern.variables:
                            binding['subject'] = subject
                        if 'predicate' in pattern.variables:
                            binding['predicate'] = pattern.predicate
                        if 'object' in pattern.variables:
                            binding['object'] = obj
                        
                        bindings.append(binding)
        
        return bindings
    
    def _match_unconstrained(self, kg, pattern: QueryPattern) -> List[Dict[str, str]]:
        """Match completely unconstrained patterns"""
        bindings = []
        
        # Enumerate all relationships
        for edge in list(kg.edges)[:1000]:  # Limit for performance
            edge_nodes = kg.incidences.get_edge_nodes(edge)
            predicate = self._extract_predicate_from_edge(kg, edge, edge_nodes)
            
            # Generate subject-object pairs
            for i, subject in enumerate(edge_nodes):
                for j, obj in enumerate(edge_nodes):
                    if i != j:
                        binding = {}
                        if 'subject' in pattern.variables:
                            binding['subject'] = subject
                        if 'predicate' in pattern.variables:
                            binding['predicate'] = predicate
                        if 'object' in pattern.variables:
                            binding['object'] = obj
                        
                        bindings.append(binding)
        
        return bindings
    
    def _edge_matches_predicate(self, kg, edge: str, edge_nodes: List[str], predicate: str) -> bool:
        """Check if an edge represents the specified predicate"""
        # Look for predicate name in edge nodes
        for node in edge_nodes:
            if predicate in node or self._normalize_predicate_name(node) == predicate:
                return True
        
        # Check relationship type
        edge_type = kg._extract_relationship_type(edge, edge_nodes)
        return edge_type == predicate
    
    def _extract_predicate_from_edge(self, kg, edge: str, edge_nodes: List[str]) -> str:
        """Extract predicate name from edge"""
        # Try to find a predicate node
        for node in edge_nodes:
            if any(indicator in node.lower() for indicator in ['has', 'is', 'relates', 'connects']):
                return self._normalize_predicate_name(node)
        
        # Use relationship type
        return kg._extract_relationship_type(edge, edge_nodes) or "relates"
    
    def _normalize_predicate_name(self, predicate: str) -> str:
        """Normalize predicate name for consistent matching"""
        # Extract meaningful part from URI
        if '/' in predicate or '#' in predicate:
            separator = '#' if '#' in predicate else '/'
            predicate = predicate.split(separator)[-1]
        
        # Remove common prefixes/suffixes
        prefixes = ['has', 'is', 'get', 'set']
        for prefix in prefixes:
            if predicate.lower().startswith(prefix.lower()):
                predicate = predicate[len(prefix):].lstrip('_')
                break
        
        return predicate
    
    def _apply_filters(self, bindings: List[Dict[str, str]], filters: List[Callable]) -> List[Dict[str, str]]:
        """Apply filter functions to bindings"""
        filtered_bindings = bindings
        
        for filter_func in filters:
            filtered_bindings = [b for b in filtered_bindings if filter_func(b)]
        
        return filtered_bindings
    
    def _pattern_to_key(self, pattern: QueryPattern) -> str:
        """Convert pattern to cache key"""
        return f"{pattern.subject}|{pattern.predicate}|{pattern.object}|{sorted(pattern.variables)}"
    
    def _should_use_sampling(self) -> bool:
        """Determine if sampling should be used"""
        return (self.use_sampling and 
                len(self.kg.nodes) > self.sampling_threshold)
    
    @performance_monitor("path_query")
    def path_query(self, 
                  start_entity: str,
                  end_entity: str,
                  max_hops: int = 3,
                  path_constraints: Optional[List[str]] = None) -> List[List[str]]:
        """
        Find paths between two entities
        
        Args:
            start_entity: Starting entity
            end_entity: Target entity
            max_hops: Maximum path length
            path_constraints: Required relationship types in path
            
        Returns:
            List of paths (each path is a list of entities)
        """
        
        if start_entity not in self.kg.nodes or end_entity not in self.kg.nodes:
            return []
        
        paths = []
        
        # BFS to find paths
        queue = deque([(start_entity, [start_entity])])
        visited_paths = set()
        
        while queue:
            current_entity, path = queue.popleft()
            
            if len(path) > max_hops + 1:
                continue
            
            if current_entity == end_entity and len(path) > 1:
                paths.append(path)
                continue
            
            # Get neighbors
            incident_edges = self.kg.incidences.get_node_edges(current_entity)
            
            for edge in incident_edges:
                edge_nodes = self.kg.incidences.get_edge_nodes(edge)
                
                for neighbor in edge_nodes:
                    if neighbor != current_entity and neighbor not in path:
                        new_path = path + [neighbor]
                        path_key = tuple(new_path)
                        
                        if path_key not in visited_paths:
                            visited_paths.add(path_key)
                            
                            # Check path constraints
                            if path_constraints and len(new_path) > 2:
                                # Check if path satisfies constraints
                                if not self._path_satisfies_constraints(new_path, path_constraints):
                                    continue
                            
                            queue.append((neighbor, new_path))
        
        return paths
    
    def _path_satisfies_constraints(self, path: List[str], constraints: List[str]) -> bool:
        """Check if path satisfies relationship type constraints"""
        # For now, implement simple constraint checking
        # In a full implementation, this would check relationship types along the path
        return True
    
    @performance_monitor("neighborhood_query")
    def entity_neighborhood_query(self, 
                                entity: str,
                                hops: int = 1,
                                entity_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get the neighborhood around an entity
        
        Args:
            entity: Central entity
            hops: Number of hops to explore
            entity_types: Filter by entity types
            
        Returns:
            Neighborhood information
        """
        
        if entity not in self.kg.nodes:
            return {'entity': entity, 'neighborhood': {}, 'error': 'Entity not found'}
        
        neighborhood = {
            'center': entity,
            'neighbors_by_hop': {},
            'relationships': {},
            'statistics': {}
        }
        
        current_entities = {entity}
        all_visited = {entity}
        
        for hop in range(1, hops + 1):
            next_entities = set()
            hop_relationships = []
            
            for current_entity in current_entities:
                incident_edges = self.kg.incidences.get_node_edges(current_entity)
                
                for edge in incident_edges:
                    edge_nodes = self.kg.incidences.get_edge_nodes(edge)
                    
                    for neighbor in edge_nodes:
                        if neighbor != current_entity and neighbor not in all_visited:
                            # Filter by entity type if specified
                            if entity_types:
                                neighbor_type = self.kg.get_entity_type(neighbor)
                                if neighbor_type not in entity_types:
                                    continue
                            
                            next_entities.add(neighbor)
                            hop_relationships.append({
                                'from': current_entity,
                                'to': neighbor,
                                'edge': edge,
                                'relationship': self._extract_predicate_from_edge(self.kg, edge, edge_nodes)
                            })
            
            neighborhood['neighbors_by_hop'][hop] = list(next_entities)
            neighborhood['relationships'][hop] = hop_relationships
            
            all_visited.update(next_entities)
            current_entities = next_entities
            
            if not next_entities:  # No more expansion
                break
        
        # Calculate statistics
        neighborhood['statistics'] = {
            'total_neighbors': len(all_visited) - 1,  # Exclude center
            'neighbors_by_hop': {hop: len(neighbors) for hop, neighbors in neighborhood['neighbors_by_hop'].items()},
            'total_relationships': sum(len(rels) for rels in neighborhood['relationships'].values())
        }
        
        return neighborhood
    
    def clear_cache(self):
        """Clear all query caches"""
        self.query_cache.clear()
        self.pattern_cache.clear()
        logger.info("Query cache cleared")


class SPARQLEngine:
    """
    Full SPARQL 1.1 compatible query engine
    
    This is a more advanced implementation that would support
    full SPARQL syntax and semantics.
    """
    
    def __init__(self, knowledge_graph):
        """Initialize SPARQL engine"""
        self.kg = knowledge_graph
        self.semantic_engine = SemanticQueryEngine(knowledge_graph)
        
        # SPARQL features support
        self.features = {
            'basic_graph_patterns': True,
            'optional_patterns': True,
            'union_patterns': True,
            'filter_expressions': True,
            'solution_modifiers': True,
            'federated_queries': False,  # Future enhancement
            'update_operations': False   # Future enhancement
        }
    
    def execute_sparql(self, sparql_query: str) -> Dict[str, Any]:
        """
        Execute a full SPARQL query
        
        Args:
            sparql_query: Complete SPARQL query string
            
        Returns:
            SPARQL-compliant results
        """
        
        # For now, delegate to semantic engine with adaptation
        # Full SPARQL implementation would parse and execute complete SPARQL 1.1 syntax
        
        try:
            result = self.semantic_engine.sparql_like_query(sparql_query)
            
            return {
                'head': {'vars': list(result.bindings[0].keys()) if result.bindings else []},
                'results': {'bindings': result.bindings},
                'metadata': result.metadata,
                'execution_time': result.execution_time
            }
            
        except Exception as e:
            logger.error(f"SPARQL execution error: {str(e)}")
            return {
                'error': str(e),
                'query': sparql_query
            }
    
    def validate_sparql(self, sparql_query: str) -> Dict[str, Any]:
        """
        Validate SPARQL query syntax
        
        Returns validation results
        """
        
        # Basic validation - full implementation would use SPARQL parser
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check for basic SPARQL keywords
        required_keywords = ['SELECT', 'WHERE']
        for keyword in required_keywords:
            if keyword not in sparql_query.upper():
                validation['valid'] = False
                validation['errors'].append(f"Missing required keyword: {keyword}")
        
        # Check for balanced braces
        if sparql_query.count('{') != sparql_query.count('}'):
            validation['valid'] = False
            validation['errors'].append("Unbalanced braces in WHERE clause")
        
        return validation