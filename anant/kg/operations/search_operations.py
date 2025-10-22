"""
Search Operations for Hierarchical Knowledge Graph
================================================

This module handles all search and query operations for the hierarchical knowledge graph,
including semantic search, cross-level queries, aggregation, and advanced filtering.

Key Features:
- Semantic search across multiple hierarchy levels
- Cross-level query aggregation and result fusion
- Context-aware search with hierarchical relevance
- Multi-criteria filtering with level constraints
- Search result ranking and relevance scoring
- Query expansion using hierarchical relationships
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
from datetime import datetime
import logging
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class SearchOperations:
    """
    Handles search and query operations for hierarchical knowledge graphs.
    
    This class provides comprehensive search capabilities that leverage the
    hierarchical structure to provide context-aware and multi-level search results.
    
    Features:
    - Semantic search with hierarchical context
    - Cross-level aggregation and filtering
    - Query expansion using parent/child relationships
    - Relevance scoring based on hierarchy position
    - Multi-criteria search with level constraints
    """
    
    def __init__(self, hierarchical_kg):
        """
        Initialize search operations.
        
        Args:
            hierarchical_kg: Reference to parent HierarchicalKnowledgeGraph instance
        """
        self.hkg = hierarchical_kg
    
    def semantic_search(self, 
                       query: str,
                       level_ids: Optional[List[str]] = None,
                       max_results: int = 10,
                       include_cross_level: bool = True) -> List[Dict[str, Any]]:
        """
        Perform semantic search across hierarchical levels.
        
        Args:
            query: Search query string
            level_ids: Specific levels to search (None for all levels)
            max_results: Maximum number of results to return
            include_cross_level: Include cross-level relationship context
            
        Returns:
            List of search results with hierarchical context
        """
        results = []
        
        # Determine levels to search
        search_levels = level_ids or list(self.hkg.levels.keys())
        
        for level_id in search_levels:
            if level_id in self.hkg.level_graphs:
                # Search within level-specific graph
                level_results = self._search_in_level(query, level_id, max_results)
                
                # Add hierarchical context
                for result in level_results:
                    result['level_id'] = level_id
                    result['level_name'] = self.hkg.levels.get(level_id, {}).get('name', level_id)
                    result['hierarchy_context'] = self._get_hierarchy_context(result['entity_id'])
                
                results.extend(level_results)
        
        # Include cross-level relationships if requested
        if include_cross_level:
            cross_level_results = self._search_cross_level_relationships(query)
            results.extend(cross_level_results)
        
        # Rank and limit results
        ranked_results = self._rank_search_results(results, query)
        return ranked_results[:max_results]
    
    def _search_in_level(self, query: str, level_id: str, max_results: int) -> List[Dict[str, Any]]:
        """Search within a specific hierarchical level."""
        if level_id not in self.hkg.level_graphs:
            return []
        
        level_graph = self.hkg.level_graphs[level_id]
        
        # Use knowledge graph's semantic search if available
        if hasattr(level_graph, 'semantic_search'):
            return level_graph.semantic_search(query, max_results)
        
        # Fallback to simple text matching
        results = []
        entities = level_graph.get_all_entities() if hasattr(level_graph, 'get_all_entities') else []
        
        for entity_id in entities:
            entity_data = level_graph.get_entity(entity_id) if hasattr(level_graph, 'get_entity') else {}
            if self._matches_query(entity_data, query):
                results.append({
                    'entity_id': entity_id,
                    'entity_data': entity_data,
                    'relevance_score': self._calculate_relevance(entity_data, query),
                    'match_type': 'text_match'
                })
        
        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:max_results]
    
    def _matches_query(self, entity_data: Dict[str, Any], query: str) -> bool:
        """Check if entity data matches the search query."""
        query_lower = query.lower()
        
        # Search in entity properties
        for key, value in entity_data.get('properties', {}).items():
            if isinstance(value, str) and query_lower in value.lower():
                return True
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str) and query_lower in item.lower():
                        return True
        
        # Search in entity type and ID
        if query_lower in entity_data.get('type', '').lower():
            return True
        if query_lower in entity_data.get('id', '').lower():
            return True
        
        return False
    
    def _calculate_relevance(self, entity_data: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for entity relative to query."""
        score = 0.0
        query_lower = query.lower()
        
        # Exact matches get higher scores
        properties = entity_data.get('properties', {})
        for key, value in properties.items():
            if isinstance(value, str):
                if query_lower == value.lower():
                    score += 1.0
                elif query_lower in value.lower():
                    score += 0.5
        
        # Type and ID matches
        if query_lower == entity_data.get('type', '').lower():
            score += 0.8
        elif query_lower in entity_data.get('type', '').lower():
            score += 0.4
        
        if query_lower == entity_data.get('id', '').lower():
            score += 0.7
        elif query_lower in entity_data.get('id', '').lower():
            score += 0.3
        
        return score
    
    def _search_cross_level_relationships(self, query: str) -> List[Dict[str, Any]]:
        """Search within cross-level relationships."""
        results = []
        
        for relationship in self.hkg.cross_level_relationships:
            if self._matches_query(relationship, query):
                results.append({
                    'entity_id': f"cross_level_{relationship.get('source_entity')}_{relationship.get('target_entity')}",
                    'entity_data': relationship,
                    'relevance_score': self._calculate_relevance(relationship, query),
                    'match_type': 'cross_level_relationship',
                    'level_id': 'cross_level'
                })
        
        return results
    
    def _get_hierarchy_context(self, entity_id: str) -> Dict[str, Any]:
        """Get hierarchical context for an entity."""
        context = {
            'level': self.hkg.entity_levels.get(entity_id),
            'level_order': None,
            'parent_entities': [],
            'child_entities': [],
            'sibling_entities': []
        }
        
        level_id = context['level']
        if level_id:
            context['level_order'] = self.hkg.level_order.get(level_id)
            
            # Get hierarchical relationships via navigation operations
            if hasattr(self.hkg, 'navigation_ops'):
                context['parent_entities'] = self.hkg.navigation_ops.navigate_up(entity_id)
                context['child_entities'] = self.hkg.navigation_ops.navigate_down(entity_id)
        
        return context
    
    def _rank_search_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank search results by relevance and hierarchical context."""
        def ranking_score(result):
            base_score = result.get('relevance_score', 0.0)
            
            # Boost score based on level order (higher levels = more important)
            level_order = result.get('hierarchy_context', {}).get('level_order')
            if level_order is not None:
                # Lower level_order (higher in hierarchy) gets higher boost
                hierarchy_boost = 1.0 / (level_order + 1)
                base_score += hierarchy_boost * 0.2
            
            # Boost exact matches
            if result.get('match_type') == 'exact_match':
                base_score += 0.5
            
            return base_score
        
        return sorted(results, key=ranking_score, reverse=True)
    
    def aggregate_search_results(self,
                               query: str,
                               level_ids: Optional[List[str]] = None,
                               aggregation_type: str = "summary") -> Dict[str, Any]:
        """
        Aggregate search results across multiple levels.
        
        Args:
            query: Search query
            level_ids: Levels to include in aggregation
            aggregation_type: Type of aggregation ("summary", "count", "detailed")
            
        Returns:
            Aggregated results by level and overall statistics
        """
        search_results = self.semantic_search(query, level_ids, max_results=1000)
        
        aggregation = {
            'query': query,
            'total_results': len(search_results),
            'levels_searched': level_ids or list(self.hkg.levels.keys()),
            'results_by_level': defaultdict(list),
            'statistics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Group results by level
        for result in search_results:
            level_id = result.get('level_id', 'unknown')
            aggregation['results_by_level'][level_id].append(result)
        
        # Calculate statistics
        for level_id, level_results in aggregation['results_by_level'].items():
            aggregation['statistics'][level_id] = {
                'count': len(level_results),
                'avg_relevance': sum(r.get('relevance_score', 0) for r in level_results) / len(level_results) if level_results else 0,
                'top_relevance': max(r.get('relevance_score', 0) for r in level_results) if level_results else 0
            }
        
        # Simplify output based on aggregation type
        if aggregation_type == "count":
            return {
                'query': query,
                'total_results': aggregation['total_results'],
                'counts_by_level': {k: v['count'] for k, v in aggregation['statistics'].items()}
            }
        elif aggregation_type == "summary":
            return {
                'query': query,
                'total_results': aggregation['total_results'],
                'statistics': aggregation['statistics'],
                'top_results': search_results[:5]
            }
        
        return aggregation
    
    def query_with_filters(self,
                          query: str,
                          filters: Dict[str, Any],
                          level_constraints: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform filtered search with multiple criteria.
        
        Args:
            query: Base search query
            filters: Entity property filters
            level_constraints: Level-specific constraints
            
        Returns:
            Filtered search results
        """
        # Get initial search results
        initial_results = self.semantic_search(
            query, 
            level_ids=level_constraints.get('level_ids') if level_constraints else None,
            max_results=1000
        )
        
        # Apply property filters
        filtered_results = []
        for result in initial_results:
            if self._passes_filters(result, filters):
                # Apply level constraints
                if level_constraints is None or self._passes_level_constraints(result, level_constraints):
                    filtered_results.append(result)
        
        return filtered_results
    
    def _passes_filters(self, result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if result passes property filters."""
        entity_data = result.get('entity_data', {})
        properties = entity_data.get('properties', {})
        
        for filter_key, filter_value in filters.items():
            if filter_key not in properties:
                return False
            
            prop_value = properties[filter_key]
            
            # Handle different filter types
            if isinstance(filter_value, dict):
                # Range or complex filters
                if 'min' in filter_value or 'max' in filter_value:
                    if not self._check_range_filter(prop_value, filter_value):
                        return False
                elif 'contains' in filter_value:
                    if isinstance(prop_value, str) and filter_value['contains'].lower() not in prop_value.lower():
                        return False
            else:
                # Simple equality filter
                if prop_value != filter_value:
                    return False
        
        return True
    
    def _passes_level_constraints(self, result: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        """Check if result passes level-specific constraints."""
        level_id = result.get('level_id')
        level_order = result.get('hierarchy_context', {}).get('level_order')
        
        # Check level order constraints
        if 'min_level_order' in constraints:
            if level_order is None or level_order < constraints['min_level_order']:
                return False
        
        if 'max_level_order' in constraints:
            if level_order is None or level_order > constraints['max_level_order']:
                return False
        
        # Check specific level inclusion/exclusion
        if 'include_levels' in constraints:
            if level_id not in constraints['include_levels']:
                return False
        
        if 'exclude_levels' in constraints:
            if level_id in constraints['exclude_levels']:
                return False
        
        return True
    
    def _check_range_filter(self, value: Any, range_filter: Dict[str, Any]) -> bool:
        """Check if value passes range filter."""
        try:
            numeric_value = float(value)
            
            if 'min' in range_filter and numeric_value < range_filter['min']:
                return False
            
            if 'max' in range_filter and numeric_value > range_filter['max']:
                return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    def expand_query_with_hierarchy(self,
                                   query: str,
                                   expansion_depth: int = 1) -> List[str]:
        """
        Expand search query using hierarchical relationships.
        
        Args:
            query: Original query
            expansion_depth: How many levels up/down to expand
            
        Returns:
            List of expanded query terms
        """
        expanded_terms = [query]
        
        # Find entities that match the original query
        initial_results = self.semantic_search(query, max_results=5)
        
        for result in initial_results:
            entity_id = result['entity_id']
            
            # Get related entities through hierarchy
            if hasattr(self.hkg, 'navigation_ops'):
                # Add parent entities
                for _ in range(expansion_depth):
                    parents = self.hkg.navigation_ops.get_parent(entity_id)
                    if isinstance(parents, str):
                        parents = [parents]
                    for parent in parents or []:
                        parent_data = self.hkg.knowledge_graph.get_entity(parent)
                        if parent_data:
                            parent_name = parent_data.get('properties', {}).get('name', parent)
                            if parent_name not in expanded_terms:
                                expanded_terms.append(parent_name)
                
                # Add child entities
                children = self.hkg.navigation_ops.get_children(entity_id)
                for child in children or []:
                    child_data = self.hkg.knowledge_graph.get_entity(child)
                    if child_data:
                        child_name = child_data.get('properties', {}).get('name', child)
                        if child_name not in expanded_terms:
                            expanded_terms.append(child_name)
        
        return expanded_terms
    
    def search_statistics(self) -> Dict[str, Any]:
        """Get search-related statistics for the hierarchical knowledge graph."""
        stats = {
            'total_entities': self.hkg.num_nodes(),
            'total_levels': len(self.hkg.levels),
            'entities_by_level': {},
            'cross_level_relationships': len(self.hkg.cross_level_relationships),
            'searchable_properties': set()
        }
        
        # Count entities by level
        for entity_id in self.hkg.nodes():
            level_id = self.hkg.entity_levels.get(entity_id, 'unassigned')
            if level_id not in stats['entities_by_level']:
                stats['entities_by_level'][level_id] = 0
            stats['entities_by_level'][level_id] += 1
            
            # Collect searchable properties
            entity_data = self.hkg.knowledge_graph.get_entity(entity_id)
            if entity_data:
                properties = entity_data.get('properties', {})
                stats['searchable_properties'].update(properties.keys())
        
        stats['searchable_properties'] = list(stats['searchable_properties'])
        
        return stats