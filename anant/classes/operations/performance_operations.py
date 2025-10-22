"""
Performance Operations for Hypergraphs
======================================

Performance optimization operations including:
- Index building and maintenance
- Cache management and invalidation
- Performance monitoring and statistics
- Memory optimization
"""

from typing import Dict, List, Optional, Any, Union
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class PerformanceOperations:
    """
    Handles performance optimization operations for hypergraphs.
    
    This class provides methods for caching, indexing, and performance
    monitoring to ensure optimal operation at scale.
    """
    
    def __init__(self, hypergraph):
        """
        Initialize performance operations.
        
        Args:
            hypergraph: Reference to parent Hypergraph instance
        """
        self.hg = hypergraph
        self.logger = logger.getChild(self.__class__.__name__)
    
    def _build_performance_indexes(self):
        """
        Build high-performance indexes for O(1) lookups.
        
        Creates adjacency lists and degree caches for billion-scale optimization.
        """
        if self.hg._indexes_built and not self.hg._dirty:
            return
        
        self.logger.debug("Building performance indexes")
        
        # Clear existing indexes
        self.hg._node_to_edges.clear()
        self.hg._edge_to_nodes.clear()
        self.hg._node_degrees.clear()
        self.hg._edge_sizes.clear()
        
        if self.hg.incidences.data.is_empty():
            self.hg._indexes_built = True
            self.hg._dirty = False
            return
        
        # Build node-to-edges mapping
        node_edge_pairs = self.hg.incidences.data.select(['node_id', 'edge_id']).unique()
        
        for row in node_edge_pairs.iter_rows():
            node_id, edge_id = row
            self.hg._node_to_edges[node_id].add(edge_id)
            self.hg._edge_to_nodes[edge_id].add(node_id)
        
        # Build degree caches
        for node_id, edges in self.hg._node_to_edges.items():
            self.hg._node_degrees[node_id] = len(edges)
        
        for edge_id, nodes in self.hg._edge_to_nodes.items():
            self.hg._edge_sizes[edge_id] = len(nodes)
        
        # Cache node and edge sets
        self.hg._nodes_cache = set(self.hg._node_to_edges.keys())
        self.hg._edges_cache = set(self.hg._edge_to_nodes.keys())
        
        # Mark as built and clean
        self.hg._indexes_built = True
        self.hg._dirty = False
        self.hg._performance_stats['index_builds'] += 1
        
        self.logger.info(
            f"Built performance indexes: {len(self.hg._nodes_cache)} nodes, "
            f"{len(self.hg._edges_cache)} edges"
        )
    
    def _invalidate_cache(self):
        """
        Invalidate all caches when the graph structure changes.
        """
        self.hg._cache.clear()
        self.hg._nodes_cache = None
        self.hg._edges_cache = None
        self.hg._node_degrees.clear()
        self.hg._edge_sizes.clear()
        self.hg._node_to_edges.clear()
        self.hg._edge_to_nodes.clear()
        self.hg._indexes_built = False
        self.hg._dirty = True
        
        self.logger.debug("Invalidated performance caches")
    
    @classmethod
    def clear_global_cache(cls):
        """
        Clear any global/class-level caches.
        
        This is useful for memory management in long-running applications.
        """
        # If there were any global caches, clear them here
        logger.info("Cleared global performance caches")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        stats = self.hg._performance_stats.copy()
        
        # Add current state information
        stats.update({
            'indexes_built': self.hg._indexes_built,
            'is_dirty': self.hg._dirty,
            'nodes_cached': self.hg._nodes_cache is not None,
            'edges_cached': self.hg._edges_cache is not None,
            'node_degree_cache_size': len(self.hg._node_degrees),
            'edge_size_cache_size': len(self.hg._edge_sizes),
            'node_to_edges_cache_size': len(self.hg._node_to_edges),
            'edge_to_nodes_cache_size': len(self.hg._edge_to_nodes),
            'cache_hit_rate': (
                stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
                if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0.0
            )
        })
        
        return stats
    
    def print_performance_report(self):
        """
        Print a detailed performance report to the console.
        """
        stats = self.get_performance_stats()
        
        print(f"\n{'='*50}")
        print(f"Performance Report for {self.hg.name}")
        print(f"{'='*50}")
        
        print(f"Graph Size:")
        print(f"  - Nodes: {self.hg.num_nodes()}")
        print(f"  - Edges: {self.hg.num_edges()}")
        print(f"  - Incidences: {self.hg.num_incidences()}")
        
        print(f"\nCache Status:")
        print(f"  - Indexes Built: {stats['indexes_built']}")
        print(f"  - Is Dirty: {stats['is_dirty']}")
        print(f"  - Nodes Cached: {stats['nodes_cached']}")
        print(f"  - Edges Cached: {stats['edges_cached']}")
        
        print(f"\nCache Performance:")
        print(f"  - Cache Hits: {stats['cache_hits']}")
        print(f"  - Cache Misses: {stats['cache_misses']}")
        print(f"  - Hit Rate: {stats['cache_hit_rate']:.2%}")
        
        print(f"\nCache Sizes:")
        print(f"  - Node Degrees: {stats['node_degree_cache_size']}")
        print(f"  - Edge Sizes: {stats['edge_size_cache_size']}")
        print(f"  - Node→Edges: {stats['node_to_edges_cache_size']}")
        print(f"  - Edge→Nodes: {stats['edge_to_nodes_cache_size']}")
        
        print(f"\nOperations:")
        print(f"  - Index Builds: {stats['index_builds']}")
        print(f"  - Batch Operations: {stats['batch_operations']}")
        
        print(f"{'='*50}\n")
    
    def get_multiple_node_degrees(self, node_ids: List[Any]) -> Dict[Any, int]:
        """
        Get degrees for multiple nodes efficiently.
        
        Args:
            node_ids: List of node identifiers
            
        Returns:
            Dictionary mapping node IDs to their degrees
        """
        if not self.hg._indexes_built:
            self._build_performance_indexes()
        
        self.hg._performance_stats['batch_operations'] += 1
        
        return {
            node_id: self.hg._node_degrees.get(node_id, 0)
            for node_id in node_ids
        }
    
    def get_multiple_edge_nodes(self, edge_ids: List[Any]) -> Dict[Any, List[Any]]:
        """
        Get nodes for multiple edges efficiently.
        
        Args:
            edge_ids: List of edge identifiers
            
        Returns:
            Dictionary mapping edge IDs to their node lists
        """
        if not self.hg._indexes_built:
            self._build_performance_indexes()
        
        self.hg._performance_stats['batch_operations'] += 1
        
        return {
            edge_id: list(self.hg._edge_to_nodes.get(edge_id, set()))
            for edge_id in edge_ids
        }
    
    def get_multiple_node_edges(self, node_ids: List[Any]) -> Dict[Any, List[Any]]:
        """
        Get edges for multiple nodes efficiently.
        
        Args:
            node_ids: List of node identifiers
            
        Returns:
            Dictionary mapping node IDs to their edge lists
        """
        if not self.hg._indexes_built:
            self._build_performance_indexes()
        
        self.hg._performance_stats['batch_operations'] += 1
        
        return {
            node_id: list(self.hg._node_to_edges.get(node_id, set()))
            for node_id in node_ids
        }
    
    def optimize_memory_usage(self):
        """
        Optimize memory usage by cleaning up unnecessary caches and data structures.
        """
        # Clear instance-specific cache
        self.hg._cache.clear()
        
        # If indexes are clean, keep them; otherwise, clear for rebuild
        if not self.hg._dirty:
            self.logger.debug("Memory optimization: keeping clean indexes")
        else:
            self._invalidate_cache()
            self.logger.debug("Memory optimization: cleared dirty indexes")
        
        # Reset performance counters
        self.hg._performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'index_builds': self.hg._performance_stats.get('index_builds', 0),
            'batch_operations': 0
        }
        
        self.logger.info("Optimized memory usage")
    
    def precompute_common_queries(self):
        """
        Precompute results for common queries to improve performance.
        """
        if not self.hg._indexes_built:
            self._build_performance_indexes()
        
        # Precompute neighbor sets for all nodes
        neighbor_cache = {}
        for node_id in self.hg._nodes_cache or set():
            neighbors = set()
            for edge_id in self.hg._node_to_edges.get(node_id, set()):
                edge_nodes = self.hg._edge_to_nodes.get(edge_id, set())
                neighbors.update(node for node in edge_nodes if node != node_id)
            neighbor_cache[node_id] = neighbors
        
        # Store in instance cache
        self.hg._cache['neighbors'] = neighbor_cache
        
        self.logger.info(f"Precomputed neighbor sets for {len(neighbor_cache)} nodes")
    
    def get_memory_usage_estimate(self) -> Dict[str, int]:
        """
        Estimate memory usage of various data structures.
        
        Returns:
            Dictionary with estimated memory usage in bytes
        """
        import sys
        
        usage = {}
        
        # Incidence data (approximate)
        if not self.hg.incidences.data.is_empty():
            usage['incidence_data'] = self.hg.incidences.data.estimated_size()
        else:
            usage['incidence_data'] = 0
        
        # Index structures
        usage['node_to_edges'] = sum(
            sys.getsizeof(node_id) + sys.getsizeof(edges) + sum(sys.getsizeof(e) for e in edges)
            for node_id, edges in self.hg._node_to_edges.items()
        )
        
        usage['edge_to_nodes'] = sum(
            sys.getsizeof(edge_id) + sys.getsizeof(nodes) + sum(sys.getsizeof(n) for n in nodes)
            for edge_id, nodes in self.hg._edge_to_nodes.items()
        )
        
        usage['degree_caches'] = (
            sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in self.hg._node_degrees.items()) +
            sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in self.hg._edge_sizes.items())
        )
        
        # Node and edge caches
        if self.hg._nodes_cache:
            usage['nodes_cache'] = sys.getsizeof(self.hg._nodes_cache) + sum(
                sys.getsizeof(node) for node in self.hg._nodes_cache
            )
        else:
            usage['nodes_cache'] = 0
        
        if self.hg._edges_cache:
            usage['edges_cache'] = sys.getsizeof(self.hg._edges_cache) + sum(
                sys.getsizeof(edge) for edge in self.hg._edges_cache
            )
        else:
            usage['edges_cache'] = 0
        
        # Properties
        usage['properties'] = sys.getsizeof(self.hg.properties)
        
        # Total
        usage['total_estimated'] = sum(usage.values())
        
        return usage