"""
Performance operations for Hypergraph

Handles performance optimization including indexing, caching, batch operations,
and performance analytics.
"""

from typing import Dict, List, Any, Optional
import time
import polars as pl
from collections import defaultdict
from ....exceptions import HypergraphError, IncidenceError


class PerformanceOperations:
    """
    Performance optimization operations for hypergraph
    
    Handles high-performance indexing, caching, batch operations,
    and performance monitoring for billion-scale hypergraph operations.
    """
    
    def __init__(self, hypergraph):
        """
        Initialize PerformanceOperations
        
        Parameters
        ----------
        hypergraph : Hypergraph
            Parent hypergraph instance
        """
        if hypergraph is None:
            raise HypergraphError("Hypergraph instance cannot be None")
        self.hypergraph = hypergraph
    
    def build_performance_indexes(self) -> None:
        """
        Build high-performance indexes for O(1) operations
        
        Optimized for billion-scale hypergraphs with vectorized Polars operations.
        Creates adjacency lists, degree caches, and node/edge sets for fast lookups.
        
        Raises
        ------
        HypergraphError
            If index building fails
        IncidenceError
            If incidence data is corrupted
        """
        if self.hypergraph._indexes_built and not self.hypergraph._dirty:
            return
        
        try:
            start_time = time.time()
            
            # Clear existing indexes
            self.hypergraph._node_to_edges.clear()
            self.hypergraph._edge_to_nodes.clear()
            self.hypergraph._node_degrees.clear()
            self.hypergraph._edge_sizes.clear()
            self.hypergraph._nodes_cache = None
            self.hypergraph._edges_cache = None
            
            if self.hypergraph.incidences.data.is_empty():
                self.hypergraph._indexes_built = True
                self.hypergraph._dirty = False
                return
            
            # Build adjacency lists using vectorized Polars operations
            data = self.hypergraph.incidences.data
            
            # Get unique nodes and edges
            nodes_df = data.select('node_id').unique()
            edges_df = data.select('edge_id').unique()
            
            self.hypergraph._nodes_cache = set(nodes_df.to_series().to_list())
            self.hypergraph._edges_cache = set(edges_df.to_series().to_list())
            
            # Build node→edges mapping (vectorized)
            node_edge_groups = (
                data
                .group_by('node_id')
                .agg(pl.col('edge_id').unique().alias('edges'))
            )
            
            for row in node_edge_groups.iter_rows():
                node_id, edges = row
                edge_set = set(edges)
                self.hypergraph._node_to_edges[node_id] = edge_set
                self.hypergraph._node_degrees[node_id] = len(edge_set)
            
            # Build edge→nodes mapping (vectorized)
            edge_node_groups = (
                data
                .group_by('edge_id')
                .agg(pl.col('node_id').unique().alias('nodes'))
            )
            
            for row in edge_node_groups.iter_rows():
                edge_id, nodes = row
                node_set = set(nodes)
                self.hypergraph._edge_to_nodes[edge_id] = node_set
                self.hypergraph._edge_sizes[edge_id] = len(node_set)
            
            self.hypergraph._indexes_built = True
            self.hypergraph._dirty = False
            self.hypergraph._performance_stats['index_builds'] += 1
            
            build_time = time.time() - start_time
            if len(self.hypergraph._nodes_cache) > 100:  # Only print for larger graphs
                print(f"🚀 Performance indexes built in {build_time:.3f}s for {len(self.hypergraph._nodes_cache):,} nodes and {len(self.hypergraph._edges_cache):,} edges")
                
        except Exception as e:
            raise HypergraphError(f"Failed to build performance indexes: {e}")
    
    def invalidate_cache(self) -> None:
        """
        Invalidate all cached data to force rebuild
        
        Clears all cached properties and marks performance indexes as dirty.
        Should be called after any structural modifications to the hypergraph.
        """
        try:
            self.hypergraph._cache.clear()
            self.hypergraph._dirty = True
            self.hypergraph._indexes_built = False
        except Exception as e:
            raise HypergraphError(f"Failed to invalidate cache: {e}")
    
    @classmethod
    def clear_global_cache(cls) -> None:
        """
        Clear any global caches to ensure fresh state
        
        Clears global caches from decorators and other shared state.
        Useful for testing and ensuring clean state between operations.
        """
        try:
            # Clear any global cache from decorators if imported
            try:
                from ....utils.decorators import _cache, _cache_lock
                with _cache_lock:
                    _cache.clear()
            except ImportError:
                pass
        except Exception as e:
            raise HypergraphError(f"Failed to clear global cache: {e}")
    
    def get_multiple_node_degrees(self, node_ids: List[Any]) -> Dict[Any, int]:
        """
        Get degrees for multiple nodes in one optimized operation
        
        Parameters
        ----------
        node_ids : List[Any]
            List of node identifiers
            
        Returns
        -------
        Dict[Any, int]
            Dictionary mapping node IDs to their degrees
            
        Raises
        ------
        HypergraphError
            If batch operation fails
        """
        try:
            self.build_performance_indexes()
            self.hypergraph._performance_stats['batch_operations'] += 1
            
            return {node_id: self.hypergraph._node_degrees.get(node_id, 0) 
                   for node_id in node_ids}
        except Exception as e:
            raise HypergraphError(f"Failed to get multiple node degrees: {e}")
    
    def get_multiple_edge_nodes(self, edge_ids: List[Any]) -> Dict[Any, List[Any]]:
        """
        Get nodes for multiple edges in one optimized operation
        
        Parameters
        ----------
        edge_ids : List[Any]
            List of edge identifiers
            
        Returns
        -------
        Dict[Any, List[Any]]
            Dictionary mapping edge IDs to lists of their nodes
            
        Raises
        ------
        HypergraphError
            If batch operation fails
        """
        try:
            self.build_performance_indexes()
            self.hypergraph._performance_stats['batch_operations'] += 1
            
            return {edge_id: list(self.hypergraph._edge_to_nodes.get(edge_id, set())) 
                   for edge_id in edge_ids}
        except Exception as e:
            raise HypergraphError(f"Failed to get multiple edge nodes: {e}")
    
    def get_multiple_node_edges(self, node_ids: List[Any]) -> Dict[Any, List[Any]]:
        """
        Get edges for multiple nodes in one optimized operation
        
        Parameters
        ----------
        node_ids : List[Any]
            List of node identifiers
            
        Returns
        -------
        Dict[Any, List[Any]]
            Dictionary mapping node IDs to lists of their edges
            
        Raises
        ------
        HypergraphError
            If batch operation fails
        """
        try:
            self.build_performance_indexes()
            self.hypergraph._performance_stats['batch_operations'] += 1
            
            return {node_id: list(self.hypergraph._node_to_edges.get(node_id, set())) 
                   for node_id in node_ids}
        except Exception as e:
            raise HypergraphError(f"Failed to get multiple node edges: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get detailed performance statistics
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing comprehensive performance metrics including:
            - Cache hit rate and statistics  
            - Batch operation counts
            - Index build information
            - Memory usage breakdown
            
        Raises
        ------
        HypergraphError
            If statistics calculation fails
        """
        try:
            cache_hit_rate = 0
            total_queries = (self.hypergraph._performance_stats['cache_hits'] + 
                           self.hypergraph._performance_stats['cache_misses'])
            if total_queries > 0:
                cache_hit_rate = self.hypergraph._performance_stats['cache_hits'] / total_queries * 100
            
            return {
                'cache_hit_rate': f"{cache_hit_rate:.1f}%",
                'cache_hits': self.hypergraph._performance_stats['cache_hits'],
                'cache_misses': self.hypergraph._performance_stats['cache_misses'],
                'batch_operations': self.hypergraph._performance_stats['batch_operations'],
                'index_builds': self.hypergraph._performance_stats['index_builds'],
                'indexes_built': self.hypergraph._indexes_built,
                'memory_usage': {
                    'nodes': len(self.hypergraph._nodes_cache or set()),
                    'edges': len(self.hypergraph._edges_cache or set()),
                    'node_degrees': len(self.hypergraph._node_degrees),
                    'edge_sizes': len(self.hypergraph._edge_sizes),
                    'adjacency_entries': sum(len(edges) for edges in self.hypergraph._node_to_edges.values())
                }
            }
        except Exception as e:
            raise HypergraphError(f"Failed to get performance statistics: {e}")
    
    def print_performance_report(self) -> None:
        """
        Print detailed performance analysis
        
        Displays comprehensive performance metrics including cache statistics,
        memory usage, and optimization recommendations.
        
        Raises
        ------
        HypergraphError
            If report generation fails
        """
        try:
            stats = self.get_performance_stats()
            
            print("\n📊 ANANT Hypergraph Performance Report")
            print("=" * 45)
            print(f"Cache Hit Rate: {stats['cache_hit_rate']}")
            print(f"Cache Hits: {stats['cache_hits']:,}")
            print(f"Cache Misses: {stats['cache_misses']:,}")
            print(f"Batch Operations: {stats['batch_operations']:,}")
            print(f"Index Builds: {stats['index_builds']:,}")
            print(f"Indexes Built: {'✅ Yes' if stats['indexes_built'] else '❌ No'}")
            
            print(f"\n💾 Memory Usage:")
            mem = stats['memory_usage']
            print(f"  Nodes: {mem['nodes']:,}")
            print(f"  Edges: {mem['edges']:,}")
            print(f"  Degree Cache: {mem['node_degrees']:,}")
            print(f"  Size Cache: {mem['edge_sizes']:,}")
            print(f"  Adjacency Entries: {mem['adjacency_entries']:,}")
            
            # Performance recommendations
            print(f"\n🚀 Performance Status:")
            hit_rate_str = stats['cache_hit_rate'].replace('%', '')
            hit_rate_num = float(hit_rate_str) if hit_rate_str else 0
            
            if hit_rate_num > 95:
                print("  ✅ Excellent cache performance!")
            elif hit_rate_num > 80:
                print("  ⚠️ Good cache performance")
            else:
                print("  🔄 Building indexes for optimal performance...")
                
        except Exception as e:
            raise HypergraphError(f"Failed to print performance report: {e}")
    
    def get_node_degree(self, node_id: Any) -> int:
        """
        Get node degree in O(1) time using pre-computed cache
        
        Parameters
        ----------
        node_id : Any
            The node identifier
            
        Returns
        -------
        int
            Degree of the node (number of incident edges)
            
        Raises
        ------
        HypergraphError
            If degree calculation fails
        """
        try:
            self.build_performance_indexes()
            self.hypergraph._performance_stats['cache_hits'] += 1
            return self.hypergraph._node_degrees.get(node_id, 0)
        except Exception as e:
            self.hypergraph._performance_stats['cache_misses'] += 1
            raise HypergraphError(f"Failed to get node degree for {node_id}: {e}")
    
    def get_edge_size(self, edge_id: Any) -> int:
        """
        Get edge size in O(1) time using pre-computed cache
        
        Parameters
        ----------
        edge_id : Any
            The edge identifier
            
        Returns
        -------
        int
            Size of the edge (number of incident nodes)
            
        Raises
        ------
        HypergraphError
            If edge size calculation fails
        """
        try:
            self.build_performance_indexes()
            self.hypergraph._performance_stats['cache_hits'] += 1
            return self.hypergraph._edge_sizes.get(edge_id, 0)
        except Exception as e:
            self.hypergraph._performance_stats['cache_misses'] += 1
            raise HypergraphError(f"Failed to get edge size for {edge_id}: {e}")
    
    def optimize_for_queries(self) -> None:
        """
        Optimize hypergraph structure for query performance
        
        Builds all performance indexes and prepares the hypergraph
        for optimal query performance. Should be called before
        running many queries on a large hypergraph.
        """
        try:
            self.build_performance_indexes()
            print(f"✅ Hypergraph optimized for queries: {self.hypergraph.num_nodes:,} nodes, {self.hypergraph.num_edges:,} edges")
        except Exception as e:
            raise HypergraphError(f"Failed to optimize for queries: {e}")
    
    def reset_performance_stats(self) -> None:
        """
        Reset all performance statistics to zero
        
        Useful for benchmarking specific operations or resetting
        counters after optimization.
        """
        try:
            self.hypergraph._performance_stats = {
                'cache_hits': 0,
                'cache_misses': 0,
                'index_builds': 0,
                'batch_operations': 0
            }
        except Exception as e:
            raise HypergraphError(f"Failed to reset performance stats: {e}")
    
    def get_memory_usage_mb(self) -> float:
        """
        Estimate memory usage in megabytes
        
        Returns
        -------
        float
            Estimated memory usage in MB
        """
        try:
            # Rough estimation based on data structures
            stats = self.get_performance_stats()
            mem = stats['memory_usage']
            
            # Estimate bytes (very rough approximation)
            node_bytes = mem['nodes'] * 100  # ~100 bytes per node on average
            edge_bytes = mem['edges'] * 100  # ~100 bytes per edge on average
            cache_bytes = mem['adjacency_entries'] * 50  # ~50 bytes per adjacency entry
            
            total_bytes = node_bytes + edge_bytes + cache_bytes
            return total_bytes / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            raise HypergraphError(f"Failed to calculate memory usage: {e}")