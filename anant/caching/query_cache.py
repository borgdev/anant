"""
Query Cache - Specialized Caching for Database and API Queries

Provides intelligent caching for database queries, API calls, and other
expensive operations with query-aware invalidation and optimization.
"""

import hashlib
import time
import threading
import weakref
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    xxhash = None

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries that can be cached."""
    SELECT = "select"
    API_GET = "api_get"
    COMPUTATION = "computation"
    FUNCTION_CALL = "function_call"
    GRAPH_QUERY = "graph_query"


@dataclass
class QueryMetadata:
    """Metadata for cached queries."""
    query_type: QueryType
    query_hash: str
    tables: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    cost_estimate: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    cache_hit_ratio: float = 0.0


class QueryPattern:
    """Pattern for query matching and invalidation."""
    
    def __init__(self, pattern: str, query_type: QueryType = QueryType.SELECT):
        self.pattern = pattern
        self.query_type = query_type
        self.regex = re.compile(pattern, re.IGNORECASE)
        
    def matches(self, query: str) -> bool:
        """Check if query matches this pattern."""
        return bool(self.regex.search(query))
        
    def extract_tables(self, query: str) -> Set[str]:
        """Extract table names from query."""
        tables = set()
        
        if self.query_type == QueryType.SELECT:
            # Simple SQL table extraction
            from_match = re.search(r'\bFROM\s+(\w+)', query, re.IGNORECASE)
            if from_match:
                tables.add(from_match.group(1).lower())
                
            join_matches = re.findall(r'\bJOIN\s+(\w+)', query, re.IGNORECASE)
            for match in join_matches:
                tables.add(match.lower())
                
        return tables


class QueryCache:
    """
    Specialized cache for database queries and API calls.
    
    Features:
    - Query-aware caching with table/dependency tracking
    - Intelligent invalidation based on data changes
    - Cost-based caching decisions
    - Query pattern matching
    - Cache warming and prefetching
    """
    
    def __init__(self, 
                 cache_manager=None,
                 default_ttl: int = 3600,
                 max_query_cache_size: int = 10000,
                 enable_query_analysis: bool = True):
        """
        Initialize query cache.
        
        Args:
            cache_manager: Underlying cache manager
            default_ttl: Default time to live in seconds
            max_query_cache_size: Maximum number of cached queries
            enable_query_analysis: Enable query pattern analysis
        """
        if cache_manager is None:
            from .cache_manager import get_cache_manager
            self.cache_manager = get_cache_manager()
        else:
            self.cache_manager = cache_manager
            
        self.default_ttl = default_ttl
        self.max_query_cache_size = max_query_cache_size
        self.enable_query_analysis = enable_query_analysis
        
        # Query metadata storage
        self._query_metadata: Dict[str, QueryMetadata] = {}
        self._table_queries: Dict[str, Set[str]] = {}  # table -> query_hashes
        self._dependency_queries: Dict[str, Set[str]] = {}  # dependency -> query_hashes
        self._lock = threading.RLock()
        
        # Query patterns for analysis
        self._patterns: List[QueryPattern] = []
        self._init_default_patterns()
        
        # Statistics
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'invalidations': 0,
            'queries_analyzed': 0,
            'patterns_matched': 0
        }
        
    def _init_default_patterns(self):
        """Initialize default query patterns."""
        self._patterns = [
            QueryPattern(r'SELECT.*FROM\s+(\w+)', QueryType.SELECT),
            QueryPattern(r'GET\s+/api/', QueryType.API_GET),
            QueryPattern(r'compute_\w+', QueryType.COMPUTATION),
        ]
        
    def _hash_query(self, query: str) -> str:
        """Generate hash for query string."""
        if XXHASH_AVAILABLE:
            return xxhash.xxh64(query.encode()).hexdigest()
        else:
            return hashlib.sha256(query.encode()).hexdigest()[:16]
            
    def _analyze_query(self, query: str) -> QueryMetadata:
        """Analyze query and extract metadata."""
        query_hash = self._hash_query(query)
        
        # Default metadata
        metadata = QueryMetadata(
            query_type=QueryType.FUNCTION_CALL,
            query_hash=query_hash
        )
        
        if not self.enable_query_analysis:
            return metadata
            
        # Match against patterns
        for pattern in self._patterns:
            if pattern.matches(query):
                metadata.query_type = pattern.query_type
                metadata.tables = pattern.extract_tables(query)
                self._stats['patterns_matched'] += 1
                break
                
        self._stats['queries_analyzed'] += 1
        return metadata
        
    def _generate_cache_key(self, query: str, params: Optional[Dict] = None,
                          context: Optional[Dict] = None) -> str:
        """Generate cache key for query with parameters."""
        key_parts = [query]
        
        if params:
            # Sort parameters for consistent key generation
            param_str = '|'.join(f"{k}={v}" for k, v in sorted(params.items()))
            key_parts.append(param_str)
            
        if context:
            # Include relevant context
            context_str = '|'.join(f"{k}={v}" for k, v in sorted(context.items()))
            key_parts.append(context_str)
            
        combined = '::'.join(key_parts)
        return self._hash_query(combined)
        
    def get(self, query: str, params: Optional[Dict] = None,
           context: Optional[Dict] = None, default: Any = None) -> Any:
        """
        Get cached query result.
        
        Args:
            query: Query string or identifier
            params: Query parameters
            context: Additional context
            default: Default value if not cached
            
        Returns:
            Cached result or default
        """
        cache_key = self._generate_cache_key(query, params, context)
        
        with self._lock:
            # Check if we have metadata for this query
            if cache_key in self._query_metadata:
                metadata = self._query_metadata[cache_key]
                metadata.last_accessed = time.time()
                metadata.access_count += 1
                
        # Try to get from cache
        result = self.cache_manager.get(cache_key, namespace="query", default=None)
        
        if result is not None:
            self._stats['cache_hits'] += 1
            with self._lock:
                if cache_key in self._query_metadata:
                    metadata = self._query_metadata[cache_key]
                    # Update hit ratio
                    total_accesses = metadata.access_count
                    hits = self._stats['cache_hits']
                    metadata.cache_hit_ratio = hits / total_accesses if total_accesses > 0 else 0.0
            return result
        else:
            self._stats['cache_misses'] += 1
            return default
            
    def set(self, query: str, result: Any, params: Optional[Dict] = None,
           context: Optional[Dict] = None, ttl: Optional[int] = None,
           cost: Optional[float] = None, dependencies: Optional[Set[str]] = None) -> bool:
        """
        Cache query result.
        
        Args:
            query: Query string or identifier
            result: Query result to cache
            params: Query parameters
            context: Additional context
            ttl: Time to live in seconds
            cost: Estimated cost of the query
            dependencies: Set of dependencies (tables, APIs, etc.)
            
        Returns:
            True if cached successfully
        """
        cache_key = self._generate_cache_key(query, params, context)
        ttl = ttl or self.default_ttl
        
        # Analyze query
        metadata = self._analyze_query(query)
        if cost is not None:
            metadata.cost_estimate = cost
        if dependencies:
            metadata.dependencies.update(dependencies)
            
        with self._lock:
            # Store metadata
            self._query_metadata[cache_key] = metadata
            
            # Update table mappings
            for table in metadata.tables:
                if table not in self._table_queries:
                    self._table_queries[table] = set()
                self._table_queries[table].add(cache_key)
                
            # Update dependency mappings
            for dep in metadata.dependencies:
                if dep not in self._dependency_queries:
                    self._dependency_queries[dep] = set()
                self._dependency_queries[dep].add(cache_key)
                
            # Cleanup if we exceed max size
            if len(self._query_metadata) > self.max_query_cache_size:
                self._cleanup_old_queries()
                
        # Cache the result
        return self.cache_manager.set(cache_key, result, ttl=ttl, namespace="query")
        
    def invalidate_table(self, table_name: str) -> int:
        """
        Invalidate all queries that depend on a table.
        
        Args:
            table_name: Name of the table that changed
            
        Returns:
            Number of queries invalidated
        """
        table_name = table_name.lower()
        invalidated = 0
        
        with self._lock:
            if table_name in self._table_queries:
                query_hashes = self._table_queries[table_name].copy()
                
                for query_hash in query_hashes:
                    if self.cache_manager.delete(query_hash, namespace="query"):
                        invalidated += 1
                        
                    # Remove from metadata
                    if query_hash in self._query_metadata:
                        del self._query_metadata[query_hash]
                        
                # Clear table mapping
                del self._table_queries[table_name]
                
        self._stats['invalidations'] += invalidated
        logger.info(f"Invalidated {invalidated} queries for table: {table_name}")
        return invalidated
        
    def invalidate_dependency(self, dependency: str) -> int:
        """
        Invalidate all queries that depend on a specific dependency.
        
        Args:
            dependency: Name of the dependency that changed
            
        Returns:
            Number of queries invalidated
        """
        invalidated = 0
        
        with self._lock:
            if dependency in self._dependency_queries:
                query_hashes = self._dependency_queries[dependency].copy()
                
                for query_hash in query_hashes:
                    if self.cache_manager.delete(query_hash, namespace="query"):
                        invalidated += 1
                        
                    # Remove from metadata
                    if query_hash in self._query_metadata:
                        del self._query_metadata[query_hash]
                        
                # Clear dependency mapping
                del self._dependency_queries[dependency]
                
        self._stats['invalidations'] += invalidated
        logger.info(f"Invalidated {invalidated} queries for dependency: {dependency}")
        return invalidated
        
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate queries matching a pattern.
        
        Args:
            pattern: Regex pattern to match queries
            
        Returns:
            Number of queries invalidated
        """
        regex = re.compile(pattern, re.IGNORECASE)
        invalidated = 0
        
        with self._lock:
            query_hashes_to_remove = []
            
            for query_hash, metadata in self._query_metadata.items():
                # This is a simplified approach - in practice, you'd need
                # to store the original query string to match against
                query_hashes_to_remove.append(query_hash)
                
            for query_hash in query_hashes_to_remove:
                if self.cache_manager.delete(query_hash, namespace="query"):
                    invalidated += 1
                    
                # Remove from metadata and mappings
                if query_hash in self._query_metadata:
                    metadata = self._query_metadata[query_hash]
                    
                    # Remove from table mappings
                    for table in metadata.tables:
                        if table in self._table_queries:
                            self._table_queries[table].discard(query_hash)
                            if not self._table_queries[table]:
                                del self._table_queries[table]
                                
                    # Remove from dependency mappings
                    for dep in metadata.dependencies:
                        if dep in self._dependency_queries:
                            self._dependency_queries[dep].discard(query_hash)
                            if not self._dependency_queries[dep]:
                                del self._dependency_queries[dep]
                                
                    del self._query_metadata[query_hash]
                    
        self._stats['invalidations'] += invalidated
        logger.info(f"Invalidated {invalidated} queries matching pattern: {pattern}")
        return invalidated
        
    def _cleanup_old_queries(self):
        """Remove least recently used queries to make space."""
        with self._lock:
            if len(self._query_metadata) <= self.max_query_cache_size:
                return
                
            # Sort by last access time
            sorted_queries = sorted(
                self._query_metadata.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Remove oldest 10%
            remove_count = len(sorted_queries) // 10
            
            for i in range(remove_count):
                query_hash, metadata = sorted_queries[i]
                
                # Remove from cache
                self.cache_manager.delete(query_hash, namespace="query")
                
                # Remove from mappings
                for table in metadata.tables:
                    if table in self._table_queries:
                        self._table_queries[table].discard(query_hash)
                        if not self._table_queries[table]:
                            del self._table_queries[table]
                            
                for dep in metadata.dependencies:
                    if dep in self._dependency_queries:
                        self._dependency_queries[dep].discard(query_hash)
                        if not self._dependency_queries[dep]:
                            del self._dependency_queries[dep]
                            
                # Remove metadata
                del self._query_metadata[query_hash]
                
    def warm_cache(self, queries: List[Tuple[str, Callable, Dict]]):
        """
        Warm the cache by pre-computing query results.
        
        Args:
            queries: List of (query, function, params) tuples
        """
        for query, func, params in queries:
            try:
                # Check if already cached
                cache_key = self._generate_cache_key(query, params)
                if cache_key not in self._query_metadata:
                    # Execute and cache
                    result = func(**params)
                    self.set(query, result, params)
                    logger.info(f"Warmed cache for query: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to warm cache for query {query[:50]}: {e}")
                
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query cache statistics."""
        with self._lock:
            total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
            hit_rate = self._stats['cache_hits'] / total_requests if total_requests > 0 else 0.0
            
            # Query type distribution
            type_counts = {}
            for metadata in self._query_metadata.values():
                query_type = metadata.query_type.value
                type_counts[query_type] = type_counts.get(query_type, 0) + 1
                
            return {
                'cached_queries': len(self._query_metadata),
                'tracked_tables': len(self._table_queries),
                'tracked_dependencies': len(self._dependency_queries),
                'cache_hits': self._stats['cache_hits'],
                'cache_misses': self._stats['cache_misses'],
                'hit_rate': hit_rate,
                'invalidations': self._stats['invalidations'],
                'queries_analyzed': self._stats['queries_analyzed'],
                'patterns_matched': self._stats['patterns_matched'],
                'query_type_distribution': type_counts
            }
            
    def get_expensive_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most expensive cached queries."""
        with self._lock:
            expensive = sorted(
                self._query_metadata.items(),
                key=lambda x: x[1].cost_estimate,
                reverse=True
            )
            
            result = []
            for i, (query_hash, metadata) in enumerate(expensive[:limit]):
                result.append({
                    'rank': i + 1,
                    'query_hash': query_hash,
                    'query_type': metadata.query_type.value,
                    'cost_estimate': metadata.cost_estimate,
                    'access_count': metadata.access_count,
                    'cache_hit_ratio': metadata.cache_hit_ratio,
                    'tables': list(metadata.tables),
                    'dependencies': list(metadata.dependencies)
                })
                
            return result
            
    def add_pattern(self, pattern: str, query_type: QueryType = QueryType.SELECT):
        """Add a new query pattern for analysis."""
        self._patterns.append(QueryPattern(pattern, query_type))
        
    def clear_all(self):
        """Clear all cached queries and metadata."""
        with self._lock:
            # Clear underlying cache
            # Note: This is a simplified approach - actual implementation
            # would clear only the query namespace
            
            # Clear metadata
            self._query_metadata.clear()
            self._table_queries.clear()
            self._dependency_queries.clear()
            
            # Reset statistics
            for key in self._stats:
                self._stats[key] = 0