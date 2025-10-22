"""
Query Operations for Knowledge Graph

Handles various query operations including:
- Semantic search and filtering
- Subgraph extraction
- Path finding and traversal
- Statistical queries and analysis
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
import logging
from collections import defaultdict
import heapq

from ...exceptions import KnowledgeGraphError, ValidationError
from ...utils.performance import performance_monitor
from ...algorithms.sampling import SmartSampler

logger = logging.getLogger(__name__)


class QueryOperations:
    """
    Query operations for knowledge graph
    
    Provides comprehensive querying capabilities including semantic search,
    path finding, subgraph extraction, and statistical analysis.
    """
    
    def __init__(self, knowledge_graph):
        """
        Initialize QueryOperations
        
        Parameters
        ----------
        knowledge_graph : KnowledgeGraph
            Parent knowledge graph instance
        """
        if knowledge_graph is None:
            raise KnowledgeGraphError("Knowledge graph instance cannot be None")
        self.kg = knowledge_graph
        self.logger = logger.getChild(self.__class__.__name__)
        
                # Initialize sampler for performance optimization
        self.sampler = None  # Will be initialized when needed if available
    
    @performance_monitor("kg_semantic_search")
    def semantic_search(self, 
                       query: str,
                       entity_types: Optional[List[str]] = None,
                       relationship_types: Optional[List[str]] = None,
                       limit: int = 100,
                       include_similar: bool = True,
                       similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Perform semantic search on the knowledge graph
        
        Parameters
        ----------
        query : str
            Search query
        entity_types : Optional[List[str]]
            Filter by entity types
        relationship_types : Optional[List[str]]
            Filter by relationship types
        limit : int, default 100
            Maximum number of results
        include_similar : bool, default True
            Include semantically similar entities
        similarity_threshold : float, default 0.7
            Minimum similarity score for similar entities
            
        Returns
        -------
        Dict[str, Any]
            Search results with entities, relationships, and metadata
        """
        try:
            results = {
                'entities': [],
                'relationships': [],
                'metadata': {
                    'query': query,
                    'total_entities': 0,
                    'total_relationships': 0,
                    'search_time': 0.0
                }
            }
            
            # Search entities
            matching_entities = self._search_entities(
                query, entity_types, limit, include_similar, similarity_threshold
            )
            results['entities'] = matching_entities
            results['metadata']['total_entities'] = len(matching_entities)
            
            # Search relationships if requested
            if relationship_types:
                matching_relationships = self._search_relationships(
                    query, relationship_types, limit
                )
                results['relationships'] = matching_relationships
                results['metadata']['total_relationships'] = len(matching_relationships)
            
            return results
            
        except Exception as e:
            raise KnowledgeGraphError(f"Semantic search failed: {e}")
    
    def _search_entities(self, query: str, entity_types: Optional[List[str]], 
                        limit: int, include_similar: bool, 
                        similarity_threshold: float) -> List[Dict[str, Any]]:
        """Search for matching entities"""
        matching_entities = []
        query_lower = query.lower()
        
        # Get candidate entities
        candidates = set()
        if entity_types:
            for entity_type in entity_types:
                if hasattr(self.kg, '_semantic_ops'):
                    candidates.update(self.kg._semantic_ops.get_entities_by_type(entity_type))
        else:
            candidates = set(self.kg.nodes)
        
        # Score entities
        entity_scores = []
        for entity in candidates:
            score = self._calculate_entity_relevance(entity, query_lower)
            if score > 0:
                entity_scores.append((score, entity))
        
        # Sort by relevance and limit results
        entity_scores.sort(key=lambda x: x[0], reverse=True)
        top_entities = entity_scores[:limit]
        
        for score, entity in top_entities:
            entity_data = {
                'id': entity,
                'score': score,
                'type': None,
                'properties': {}
            }
            
            # Add entity type if available
            if hasattr(self.kg, '_semantic_ops'):
                entity_data['type'] = self.kg._semantic_ops.get_entity_type(entity)
            
            # Add properties if available
            if hasattr(self.kg.properties, 'get_node_properties'):
                entity_data['properties'] = self.kg.properties.get_node_properties(entity) or {}
            
            matching_entities.append(entity_data)
        
        return matching_entities
    
    def _search_relationships(self, query: str, relationship_types: List[str], 
                            limit: int) -> List[Dict[str, Any]]:
        """Search for matching relationships"""
        matching_relationships = []
        query_lower = query.lower()
        
        # Get candidate relationships
        candidates = set()
        for rel_type in relationship_types:
            if hasattr(self.kg, '_semantic_ops'):
                candidates.update(self.kg._semantic_ops.get_relationships_by_type(rel_type))
        
        # Score relationships
        rel_scores = []
        for edge in candidates:
            score = self._calculate_relationship_relevance(edge, query_lower)
            if score > 0:
                rel_scores.append((score, edge))
        
        # Sort and limit
        rel_scores.sort(key=lambda x: x[0], reverse=True)
        top_relationships = rel_scores[:limit]
        
        for score, edge in top_relationships:
            edge_nodes = self.kg.incidences.get_edge_nodes(edge)
            rel_data = {
                'id': edge,
                'score': score,
                'source': edge_nodes[0] if edge_nodes else None,
                'targets': edge_nodes[1:] if len(edge_nodes) > 1 else [],
                'properties': {}
            }
            
            # Add properties if available
            if hasattr(self.kg.properties, 'get_edge_properties'):
                rel_data['properties'] = self.kg.properties.get_edge_properties(edge) or {}
            
            matching_relationships.append(rel_data)
        
        return matching_relationships
    
    def _calculate_entity_relevance(self, entity: str, query: str) -> float:
        """Calculate relevance score for entity"""
        score = 0.0
        entity_lower = entity.lower()
        
        # Exact match
        if query == entity_lower:
            score += 1.0
        # Substring match
        elif query in entity_lower:
            score += 0.8
        # Word boundary match
        elif any(word in entity_lower for word in query.split()):
            score += 0.6
        
        # Bonus for URI components
        if '/' in entity or '#' in entity:
            separator = '#' if '#' in entity else '/'
            local_name = entity.split(separator)[-1].lower()
            if query in local_name:
                score += 0.4
        
        return score
    
    def _calculate_relationship_relevance(self, edge: str, query: str) -> float:
        """Calculate relevance score for relationship"""
        score = 0.0
        edge_lower = edge.lower()
        
        # Basic string matching
        if query in edge_lower:
            score += 0.5
        
        # Check edge nodes for relevance
        try:
            edge_nodes = self.kg.incidences.get_edge_nodes(edge)
            for node in edge_nodes:
                if query in node.lower():
                    score += 0.3
        except:
            pass
        
        return score
    
    @performance_monitor("kg_subgraph_extraction")
    def get_subgraph(self, 
                    center_entities: List[str],
                    max_depth: int = 2,
                    entity_types: Optional[List[str]] = None,
                    relationship_types: Optional[List[str]] = None,
                    max_nodes: int = 1000,
                    include_properties: bool = True) -> Dict[str, Any]:
        """
        Extract subgraph around center entities
        
        Parameters
        ----------
        center_entities : List[str]
            Center entities for subgraph extraction
        max_depth : int, default 2
            Maximum traversal depth
        entity_types : Optional[List[str]]
            Filter by entity types
        relationship_types : Optional[List[str]]
            Filter by relationship types
        max_nodes : int, default 1000
            Maximum nodes in subgraph
        include_properties : bool, default True
            Include entity and relationship properties
            
        Returns
        -------
        Dict[str, Any]
            Subgraph data with nodes, edges, and metadata
        """
        try:
            # Validate inputs
            if not center_entities:
                raise ValidationError("center_entities cannot be empty")
            
            if max_depth < 1:
                raise ValidationError("max_depth must be at least 1")
            
            # Initialize subgraph
            subgraph_nodes = set(center_entities)
            subgraph_edges = set()
            visited = set()
            
            # BFS traversal
            current_level = set(center_entities)
            actual_depth = 0
            
            for depth in range(max_depth):
                next_level = set()
                actual_depth = depth + 1
                
                for entity in current_level:
                    if entity in visited or len(subgraph_nodes) >= max_nodes:
                        continue
                    
                    visited.add(entity)
                    
                    # Get connected entities
                    connected = self._get_connected_entities(
                        entity, entity_types, relationship_types
                    )
                    
                    for neighbor, edge in connected:
                        if len(subgraph_nodes) < max_nodes:
                            subgraph_nodes.add(neighbor)
                            subgraph_edges.add(edge)
                            next_level.add(neighbor)
                
                current_level = next_level
                if not current_level or len(subgraph_nodes) >= max_nodes:
                    break
            
            # Build result
            result = {
                'nodes': list(subgraph_nodes),
                'edges': list(subgraph_edges),
                'metadata': {
                    'center_entities': center_entities,
                    'max_depth': max_depth,
                    'actual_depth': actual_depth,
                    'node_count': len(subgraph_nodes),
                    'edge_count': len(subgraph_edges)
                }
            }
            
            # Add properties if requested
            if include_properties:
                result['node_properties'] = {}
                result['edge_properties'] = {}
                
                for node in subgraph_nodes:
                    if hasattr(self.kg.properties, 'get_node_properties'):
                        props = self.kg.properties.get_node_properties(node)
                        if props:
                            result['node_properties'][node] = props
                
                for edge in subgraph_edges:
                    if hasattr(self.kg.properties, 'get_edge_properties'):
                        props = self.kg.properties.get_edge_properties(edge)
                        if props:
                            result['edge_properties'][edge] = props
            
            return result
            
        except Exception as e:
            raise KnowledgeGraphError(f"Subgraph extraction failed: {e}")
    
    def _get_connected_entities(self, entity: str, entity_types: Optional[List[str]], 
                              relationship_types: Optional[List[str]]) -> List[Tuple[str, str]]:
        """Get entities connected to given entity"""
        connected = []
        
        # Get edges incident to entity
        if hasattr(self.kg.incidences, 'get_node_edges'):
            incident_edges = self.kg.incidences.get_node_edges(entity)
        else:
            incident_edges = []
        
        for edge in incident_edges:
            # Filter by relationship type if specified
            if relationship_types:
                edge_type = None
                if hasattr(self.kg, '_semantic_ops'):
                    edge_nodes = self.kg.incidences.get_edge_nodes(edge)
                    edge_type = self.kg._semantic_ops.extract_relationship_type(edge, edge_nodes)
                
                if edge_type not in relationship_types:
                    continue
            
            # Get other nodes in the edge
            edge_nodes = self.kg.incidences.get_edge_nodes(edge)
            for node in edge_nodes:
                if node != entity:
                    # Filter by entity type if specified
                    if entity_types:
                        node_type = None
                        if hasattr(self.kg, '_semantic_ops'):
                            node_type = self.kg._semantic_ops.get_entity_type(node)
                        
                        if node_type not in entity_types:
                            continue
                    
                    connected.append((node, edge))
        
        return connected
    
    @performance_monitor("kg_shortest_path")
    def shortest_semantic_path(self, entity1: str, entity2: str, 
                              semantic_constraints: Optional[Dict] = None) -> Optional[List[Tuple[str, str]]]:
        """
        Find shortest semantic path between two entities
        
        Parameters
        ----------
        entity1 : str
            Source entity
        entity2 : str
            Target entity
        semantic_constraints : Optional[Dict]
            Constraints for path traversal
            
        Returns
        -------
        Optional[List[Tuple[str, str]]]
            Path as list of (entity, relationship) tuples
        """
        try:
            if entity1 not in self.kg.nodes or entity2 not in self.kg.nodes:
                return None
            
            if entity1 == entity2:
                return [(entity1, entity1)]
            
            # Dijkstra's algorithm with semantic constraints
            distances: Dict[str, float] = {entity1: 0.0}
            previous: Dict[str, Tuple[str, str]] = {}
            unvisited: List[Tuple[float, str]] = [(0.0, entity1)]
            visited = set()
            
            while unvisited:
                current_distance, current_entity = heapq.heappop(unvisited)
                
                if current_entity in visited:
                    continue
                
                visited.add(current_entity)
                
                if current_entity == entity2:
                    break
                
                # Get neighbors
                neighbors = self._get_connected_entities(
                    current_entity, 
                    semantic_constraints.get('entity_types') if semantic_constraints else None,
                    semantic_constraints.get('relationship_types') if semantic_constraints else None
                )
                
                for neighbor, edge in neighbors:
                    if neighbor in visited:
                        continue
                    
                    # Calculate edge weight (can be customized based on semantics)
                    edge_weight = 1.0
                    if semantic_constraints and 'edge_weights' in semantic_constraints:
                        edge_weight = semantic_constraints['edge_weights'].get(edge, 1.0)
                    
                    new_distance = current_distance + edge_weight
                    
                    if neighbor not in distances or new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = (current_entity, edge)
                        heapq.heappush(unvisited, (new_distance, neighbor))
            
            # Reconstruct path
            if entity2 not in previous and entity2 != entity1:
                return None
            
            path = []
            current = entity2
            
            while current != entity1:
                if current not in previous:
                    return None
                prev_entity, edge = previous[current]
                path.append((current, edge))
                current = prev_entity
            
            path.append((entity1, None))
            path.reverse()
            
            return path
            
        except Exception as e:
            self.logger.warning(f"Error finding shortest semantic path: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge graph statistics
        
        Returns
        -------
        Dict[str, Any]
            Comprehensive statistics
        """
        try:
            stats = {
                'basic': {
                    'total_entities': len(self.kg.nodes),
                    'total_relationships': len(self.kg.edges),
                    'total_incidences': len(self.kg.incidences.data) if hasattr(self.kg.incidences, 'data') else 0
                },
                'semantic': {},
                'connectivity': {}
            }
            
            # Semantic statistics
            if hasattr(self.kg, '_semantic_ops'):
                stats['semantic'] = {
                    'entity_types': len(self.kg._semantic_ops.get_all_entity_types()),
                    'relationship_types': len(self.kg._semantic_ops.get_all_relationship_types()),
                    'entity_type_distribution': {
                        et: len(self.kg._semantic_ops.get_entities_by_type(et))
                        for et in self.kg._semantic_ops.get_all_entity_types()
                    }
                }
            
            # Connectivity statistics
            degrees = []
            for node in self.kg.nodes:
                if hasattr(self.kg.incidences, 'get_node_edges'):
                    degree = len(self.kg.incidences.get_node_edges(node))
                    degrees.append(degree)
            
            if degrees:
                stats['connectivity'] = {
                    'average_degree': sum(degrees) / len(degrees),
                    'max_degree': max(degrees),
                    'min_degree': min(degrees),
                    'degree_distribution': self._calculate_degree_distribution(degrees)
                }
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Error calculating statistics: {e}")
            return {'error': str(e)}
    
    def _calculate_degree_distribution(self, degrees: List[int]) -> Dict[int, int]:
        """Calculate degree distribution"""
        distribution = defaultdict(int)
        for degree in degrees:
            distribution[degree] += 1
        return dict(distribution)