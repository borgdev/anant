"""
Indexing Operations for Knowledge Graph

Handles performance optimization through indexing and caching including:
- Semantic index building and maintenance
- Query result caching
- Performance monitoring and optimization
- Index statistics and management
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
import time
import logging

from ...exceptions import KnowledgeGraphError
from ...utils.performance import performance_monitor, PerformanceProfiler

logger = logging.getLogger(__name__)


class IndexingOperations:
    """
    Indexing operations for knowledge graph
    
    Provides high-performance indexing and caching capabilities
    for fast semantic queries and operations.
    """
    
    def __init__(self, knowledge_graph):
        """
        Initialize IndexingOperations
        
        Parameters
        ----------
        knowledge_graph : KnowledgeGraph
            Parent knowledge graph instance
        """
        if knowledge_graph is None:
            raise KnowledgeGraphError("Knowledge graph instance cannot be None")
        self.kg = knowledge_graph
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Performance indexes
        self._entity_property_index = defaultdict(dict)
        self._relationship_index = defaultdict(set)
        self._path_cache = {}
        self._similarity_cache = {}
        
        # Index statistics
        self._index_stats = {
            'build_time': 0.0,
            'last_rebuild': None,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    @performance_monitor("kg_build_indexes")
    def build_all_indexes(self):
        """Build all performance indexes"""
        self.logger.info("Building all performance indexes...")
        
        try:
            start_time = time.time()
            
            # Build entity property index
            self._build_entity_property_index()
                
            # Build relationship index
            self._build_relationship_index()
                
            # Build semantic indexes through semantic operations
            if hasattr(self.kg, '_semantic_ops'):
                self.kg._semantic_ops.build_semantic_indexes()
                
            # Update statistics
            end_time = time.time()
            build_time = end_time - start_time
            self._index_stats['build_time'] = build_time
            self._index_stats['last_rebuild'] = end_time
                
            self.logger.info(f"All indexes built in {build_time:.2f}s")
            
            return {
                'build_time': build_time,
                'entity_property_count': len(self._entity_property_index),
                'relationship_count': len(self._relationship_index),
                'cache_stats': self._index_stats
            }
                
        except Exception as e:
            raise KnowledgeGraphError(f"Failed to build indexes: {e}")
    
    def _build_entity_property_index(self):
        """Build index for entity properties"""
        self._entity_property_index.clear()
        
        for node in self.kg.nodes:
            if hasattr(self.kg.properties, 'get_node_properties'):
                node_props = self.kg.properties.get_node_properties(node)
                if node_props:
                    for prop_name, prop_value in node_props.items():
                        if prop_name not in self._entity_property_index:
                            self._entity_property_index[prop_name] = defaultdict(set)
                        self._entity_property_index[prop_name][prop_value].add(node)
    
    def _build_relationship_index(self):
        """Build index for relationships"""
        self._relationship_index.clear()
        
        for edge in self.kg.edges:
            edge_nodes = self.kg.incidences.get_edge_nodes(edge)
            
            # Index by source-target pairs
            for i, source in enumerate(edge_nodes):
                for j, target in enumerate(edge_nodes):
                    if i != j:
                        self._relationship_index[source].add((target, edge))
    
    @performance_monitor("kg_property_query")
    def find_entities_by_property(self, property_name: str, property_value: Any) -> Set[str]:
        """
        Find entities with specific property value
        
        Parameters
        ----------
        property_name : str
            Name of property to search
        property_value : Any
            Value to match
            
        Returns
        -------
        Set[str]
            Set of matching entities
        """
        cache_key = f"prop_{property_name}_{property_value}"
        
        if cache_key in self._path_cache:
            self._index_stats['cache_hits'] += 1
            return self._path_cache[cache_key]
        
        self._index_stats['cache_misses'] += 1
        
        result = self._entity_property_index.get(property_name, {}).get(property_value, set())
        self._path_cache[cache_key] = result
        
        return result
    
    @performance_monitor("kg_relationship_query")
    def find_related_entities(self, entity: str, max_distance: int = 1) -> Dict[str, List[Tuple[str, str]]]:
        """
        Find entities related to given entity
        
        Parameters
        ----------
        entity : str
            Source entity
        max_distance : int, default 1
            Maximum relationship distance
            
        Returns
        -------
        Dict[str, List[Tuple[str, str]]]
            Dictionary mapping distance to list of (target_entity, relationship_id) tuples
        """
        cache_key = f"related_{entity}_{max_distance}"
        
        if cache_key in self._path_cache:
            self._index_stats['cache_hits'] += 1
            return self._path_cache[cache_key]
        
        self._index_stats['cache_misses'] += 1
        
        result = {}
        visited = set()
        current_level = {entity}
        
        for distance in range(1, max_distance + 1):
            next_level = set()
            level_results = []
            
            for current_entity in current_level:
                if current_entity in visited:
                    continue
                visited.add(current_entity)
                
                # Get direct relationships
                direct_relations = self._relationship_index.get(current_entity, set())
                for target, edge_id in direct_relations:
                    if target not in visited:
                        level_results.append((target, edge_id))
                        next_level.add(target)
            
            if level_results:
                result[f"distance_{distance}"] = level_results
            
            current_level = next_level
            if not current_level:
                break
        
        self._path_cache[cache_key] = result
        return result
    
    def cache_similarity_result(self, entity1: str, entity2: str, similarity: float, method: str):
        """
        Cache similarity calculation result
        
        Parameters
        ----------
        entity1 : str
            First entity
        entity2 : str  
            Second entity
        similarity : float
            Calculated similarity score
        method : str
            Similarity calculation method
        """
        cache_key = f"sim_{method}_{min(entity1, entity2)}_{max(entity1, entity2)}"
        self._similarity_cache[cache_key] = {
            'similarity': similarity,
            'timestamp': time.time()
        }
    
    def get_cached_similarity(self, entity1: str, entity2: str, method: str, 
                            max_age: float = 3600.0) -> Optional[float]:
        """
        Get cached similarity result
        
        Parameters
        ----------
        entity1 : str
            First entity
        entity2 : str
            Second entity
        method : str
            Similarity calculation method
        max_age : float, default 3600.0
            Maximum cache age in seconds
            
        Returns
        -------
        Optional[float]
            Cached similarity score or None
        """
        cache_key = f"sim_{method}_{min(entity1, entity2)}_{max(entity1, entity2)}"
        
        if cache_key in self._similarity_cache:
            cached = self._similarity_cache[cache_key]
            age = time.time() - cached['timestamp']
            
            if age <= max_age:
                self._index_stats['cache_hits'] += 1
                return cached['similarity']
            else:
                # Remove expired entry
                del self._similarity_cache[cache_key]
        
        self._index_stats['cache_misses'] += 1
        return None
    
    def clear_all_caches(self):
        """Clear all performance caches"""
        self._path_cache.clear()
        self._similarity_cache.clear()
        
        # Clear semantic caches if available
        if hasattr(self.kg, '_semantic_ops'):
            self.kg._semantic_ops.clear_cache()
        
        self.logger.info("All caches cleared")
    
    def optimize_indexes(self):
        """Optimize indexes for better performance"""
        # Remove expired similarity cache entries
        current_time = time.time()
        expired_keys = []
        
        for key, cached in self._similarity_cache.items():
            if current_time - cached['timestamp'] > 3600:  # 1 hour
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._similarity_cache[key]
        
        # Compact path cache if too large
        if len(self._path_cache) > 10000:
            # Keep only the 5000 most recent entries
            sorted_items = sorted(
                self._path_cache.items(),
                key=lambda x: getattr(x[1], 'timestamp', 0),
                reverse=True
            )[:5000]
            self._path_cache = dict(sorted_items)
        
        self.logger.info("Indexes optimized")
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive index statistics
        
        Returns
        -------
        Dict[str, Any]
            Index statistics
        """
        cache_hit_rate = 0.0
        total_requests = self._index_stats['cache_hits'] + self._index_stats['cache_misses']
        if total_requests > 0:
            cache_hit_rate = self._index_stats['cache_hits'] / total_requests
        
        return {
            'build_time': self._index_stats['build_time'],
            'last_rebuild': self._index_stats['last_rebuild'],
            'cache_hit_rate': f"{cache_hit_rate:.2%}",
            'cache_hits': self._index_stats['cache_hits'],
            'cache_misses': self._index_stats['cache_misses'],
            'entity_property_index_size': len(self._entity_property_index),
            'relationship_index_size': len(self._relationship_index),
            'path_cache_size': len(self._path_cache),
            'similarity_cache_size': len(self._similarity_cache),
            'memory_usage': {
                'entity_properties': sum(len(props) for props in self._entity_property_index.values()),
                'relationships': sum(len(rels) for rels in self._relationship_index.values()),
                'path_cache_entries': len(self._path_cache),
                'similarity_cache_entries': len(self._similarity_cache)
            }
        }
    
    def rebuild_if_stale(self, max_age: float = 1800.0) -> bool:
        """
        Rebuild indexes if they are stale
        
        Parameters
        ----------
        max_age : float, default 1800.0
            Maximum age in seconds before rebuilding
            
        Returns
        -------
        bool
            True if indexes were rebuilt
        """
        if self._index_stats['last_rebuild'] is None:
            self.build_all_indexes()
            return True
        
        age = time.time() - self._index_stats['last_rebuild']
        if age > max_age:
            self.build_all_indexes()
            return True
        
        return False