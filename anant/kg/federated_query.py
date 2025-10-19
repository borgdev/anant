"""
Federated Query Engine for Knowledge Graphs
===========================================

Advanced federated query system that enables cross-database querying,
query decomposition, result merging, and distributed execution across
multiple knowledge graph sources.
"""

import logging
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

from ..utils.performance import performance_monitor, PerformanceProfiler
from ..utils.extras import safe_import
from .query_optimization import QueryOptimizer, OptimizationResult

# Optional dependencies
aiohttp = safe_import('aiohttp')
requests = safe_import('requests')

logger = logging.getLogger(__name__)


class FederationProtocol(Enum):
    """Supported federation protocols"""
    SPARQL_ENDPOINT = "sparql_endpoint"
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    NATIVE_KG = "native_kg"
    DATABASE_DIRECT = "database_direct"


class QueryType(Enum):
    """Types of federated queries"""
    UNION = "union"          # Combine results from multiple sources
    JOIN = "join"            # Join results across sources
    NESTED = "nested"        # Nested queries with dependencies
    DISTRIBUTED = "distributed"  # Distributed computation


@dataclass
class DataSource:
    """Configuration for a federated data source"""
    source_id: str
    name: str
    protocol: FederationProtocol
    endpoint_url: str
    authentication: Optional[Dict[str, Any]] = None
    capabilities: Optional[Set[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    health_status: str = "unknown"
    response_time_ms: float = 0.0
    priority: int = 1  # Higher priority = preferred source


@dataclass
class QueryFragment:
    """A fragment of a federated query targeting a specific source"""
    fragment_id: str
    source_id: str
    query: str
    query_type: str
    dependencies: List[str] = None
    estimated_cost: float = 0.0
    estimated_cardinality: int = 0
    timeout_seconds: int = 30


@dataclass
class FragmentResult:
    """Result from executing a query fragment"""
    fragment_id: str
    source_id: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    row_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FederatedQueryPlan:
    """Execution plan for a federated query"""
    plan_id: str
    fragments: List[QueryFragment]
    execution_order: List[str]
    merge_strategy: str
    estimated_total_cost: float
    parallelizable_groups: List[List[str]]
    optimization_hints: List[str]


@dataclass
class FederatedQueryResult:
    """Result of federated query execution"""
    query_id: str
    success: bool
    data: Optional[Any] = None
    fragment_results: List[FragmentResult] = None
    total_execution_time: float = 0.0
    total_rows: int = 0
    sources_used: List[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class FederatedQueryEngine:
    """
    Advanced Federated Query Engine
    
    Provides:
    - Cross-database query decomposition and execution
    - Intelligent source selection and load balancing
    - Result merging and deduplication
    - Distributed execution with fault tolerance
    - Query optimization across federation boundaries
    - Authentication and security management
    - Caching and performance optimization
    """
    
    def __init__(self, knowledge_graph=None):
        """
        Initialize federated query engine
        
        Args:
            knowledge_graph: Primary knowledge graph instance
        """
        self.kg = knowledge_graph
        
        # Data sources registry
        self.data_sources: Dict[str, DataSource] = {}
        self.source_capabilities: Dict[str, Set[str]] = {}
        
        # Query planning and optimization
        self.query_optimizer = QueryOptimizer(knowledge_graph) if knowledge_graph else None
        self.plan_cache: Dict[str, FederatedQueryPlan] = {}
        self.result_cache: Dict[str, FederatedQueryResult] = {}
        
        # Execution management
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.active_queries: Dict[str, Dict[str, Any]] = {}
        
        # Performance and monitoring
        self.execution_stats = defaultdict(list)
        self.source_health = {}
        
        # Configuration
        self.config = {
            'max_parallel_fragments': 5,
            'default_timeout': 30,
            'retry_attempts': 2,
            'enable_caching': True,
            'cache_ttl_seconds': 300,
            'health_check_interval': 60,
            'load_balancing': True,
            'auto_failover': True
        }
        
        # Start background health monitoring
        self._start_health_monitoring()
        
        logger.info("Federated Query Engine initialized")
    
    def register_data_source(self, source: DataSource) -> None:
        """
        Register a new data source for federation
        
        Args:
            source: DataSource configuration
        """
        
        logger.info(f"Registering data source: {source.name} ({source.protocol.value})")
        
        # Validate source configuration
        self._validate_source_config(source)
        
        # Store source
        self.data_sources[source.source_id] = source
        
        # Initialize capabilities
        if source.capabilities:
            self.source_capabilities[source.source_id] = source.capabilities
        else:
            # Auto-detect capabilities
            self.source_capabilities[source.source_id] = self._detect_source_capabilities(source)
        
        # Initial health check
        self._check_source_health(source.source_id)
        
        logger.info(f"Registered source {source.source_id} with capabilities: {self.source_capabilities[source.source_id]}")
    
    def _validate_source_config(self, source: DataSource) -> None:
        """Validate data source configuration"""
        
        if not source.source_id:
            raise ValueError("Data source must have a source_id")
        
        if not source.endpoint_url:
            raise ValueError("Data source must have an endpoint_url")
        
        if source.protocol not in FederationProtocol:
            raise ValueError(f"Unsupported protocol: {source.protocol}")
    
    def _detect_source_capabilities(self, source: DataSource) -> Set[str]:
        """Auto-detect capabilities of a data source"""
        
        capabilities = set()
        
        if source.protocol == FederationProtocol.SPARQL_ENDPOINT:
            capabilities.update(['sparql', 'rdf', 'inference', 'full_text_search'])
        
        elif source.protocol == FederationProtocol.REST_API:
            capabilities.update(['json', 'pagination', 'filtering'])
        
        elif source.protocol == FederationProtocol.GRAPHQL:
            capabilities.update(['graphql', 'typed_queries', 'introspection'])
        
        elif source.protocol == FederationProtocol.NATIVE_KG:
            capabilities.update(['semantic_search', 'reasoning', 'ontology', 'embeddings'])
        
        elif source.protocol == FederationProtocol.DATABASE_DIRECT:
            capabilities.update(['sql', 'transactions', 'joins', 'aggregation'])
        
        return capabilities
    
    @performance_monitor("federated_query_execution")
    def execute_federated_query(self, 
                               query: str, 
                               query_type: str = 'sparql',
                               sources: Optional[List[str]] = None,
                               optimization_level: int = 2) -> FederatedQueryResult:
        """
        Execute a federated query across multiple data sources
        
        Args:
            query: Query string to execute
            query_type: Type of query ('sparql', 'pattern', 'natural')
            sources: Specific sources to target (None = auto-select)
            optimization_level: Optimization level (0=none, 1=basic, 2=full)
            
        Returns:
            FederatedQueryResult with aggregated results
        """
        
        query_id = hashlib.md5(f"{query}_{sources}_{time.time()}".encode()).hexdigest()[:12]
        
        logger.info(f"Executing federated query {query_id}: {query[:100]}...")
        
        start_time = time.time()
        
        with PerformanceProfiler("federated_query") as profiler:
            
            profiler.checkpoint("query_analysis")
            
            # Analyze query and select sources
            if sources is None:
                sources = self._select_optimal_sources(query, query_type)
            
            # Validate selected sources
            available_sources = [s for s in sources if s in self.data_sources and self._is_source_healthy(s)]
            if not available_sources:
                return FederatedQueryResult(
                    query_id=query_id,
                    success=False,
                    error="No healthy data sources available",
                    total_execution_time=time.time() - start_time
                )
            
            profiler.checkpoint("query_decomposition")
            
            # Decompose query into fragments
            fragments = self._decompose_query(query, query_type, available_sources)
            
            profiler.checkpoint("query_planning")
            
            # Create execution plan
            execution_plan = self._create_execution_plan(fragments, optimization_level)
            
            profiler.checkpoint("query_optimization")
            
            # Optimize fragments if query optimizer is available
            if self.query_optimizer and optimization_level > 0:
                execution_plan = self._optimize_execution_plan(execution_plan)
            
            profiler.checkpoint("query_execution")
            
            # Execute plan
            fragment_results = self._execute_plan(execution_plan)
            
            profiler.checkpoint("result_merging")
            
            # Merge results
            merged_result = self._merge_fragment_results(fragment_results, execution_plan)
            
            profiler.checkpoint("query_complete")
        
        total_time = time.time() - start_time
        
        # Create final result
        result = FederatedQueryResult(
            query_id=query_id,
            success=merged_result is not None,
            data=merged_result,
            fragment_results=fragment_results,
            total_execution_time=total_time,
            total_rows=sum(fr.row_count for fr in fragment_results if fr.success),
            sources_used=available_sources,
            metadata={
                'fragments_executed': len(fragments),
                'parallel_groups': len(execution_plan.parallelizable_groups),
                'optimization_level': optimization_level,
                'profiler_report': profiler.get_report()
            }
        )
        
        # Cache result if enabled
        if self.config['enable_caching']:
            cache_key = self._get_cache_key(query, sources)
            self.result_cache[cache_key] = result
        
        # Update execution statistics
        self._update_execution_stats(result)
        
        logger.info(f"Federated query {query_id} completed in {total_time:.3f}s with {result.total_rows} rows")
        
        return result
    
    def _select_optimal_sources(self, query: str, query_type: str) -> List[str]:
        """Select optimal data sources for a query"""
        
        # Analyze query requirements
        required_capabilities = self._analyze_query_requirements(query, query_type)
        
        # Score sources based on capabilities, health, and performance
        source_scores = {}
        
        for source_id, source in self.data_sources.items():
            if not self._is_source_healthy(source_id):
                continue
            
            score = 0.0
            
            # Capability match score
            source_caps = self.source_capabilities.get(source_id, set())
            capability_overlap = len(required_capabilities.intersection(source_caps))
            score += capability_overlap * 10
            
            # Health and performance score
            health_data = self.source_health.get(source_id, {})
            if health_data.get('status') == 'healthy':
                score += 5
            
            response_time = health_data.get('response_time_ms', 1000)
            score += max(0, (1000 - response_time) / 100)  # Favor faster sources
            
            # Priority boost
            score += source.priority * 2
            
            source_scores[source_id] = score
        
        # Select top sources
        sorted_sources = sorted(source_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 3 sources or all if fewer available
        selected = [source_id for source_id, score in sorted_sources[:3] if score > 0]
        
        logger.info(f"Selected sources for query: {selected}")
        
        return selected
    
    def _analyze_query_requirements(self, query: str, query_type: str) -> Set[str]:
        """Analyze what capabilities a query requires"""
        
        requirements = set()
        
        query_lower = query.lower()
        
        # SPARQL patterns
        if any(keyword in query_lower for keyword in ['select', 'where', 'filter', 'optional']):
            requirements.add('sparql')
        
        # RDF patterns
        if any(pattern in query_lower for pattern in ['rdf:type', 'rdfs:', 'owl:']):
            requirements.add('rdf')
        
        # Full-text search
        if any(keyword in query_lower for keyword in ['contains', 'regex', 'like']):
            requirements.add('full_text_search')
        
        # Reasoning
        if any(keyword in query_lower for keyword in ['subclass', 'instanceof', 'infer']):
            requirements.add('inference')
        
        # Aggregation
        if any(keyword in query_lower for keyword in ['count', 'sum', 'avg', 'group by']):
            requirements.add('aggregation')
        
        # Geospatial
        if any(keyword in query_lower for keyword in ['geo:', 'spatial', 'distance']):
            requirements.add('geospatial')
        
        return requirements
    
    def _decompose_query(self, query: str, query_type: str, sources: List[str]) -> List[QueryFragment]:
        """Decompose a federated query into source-specific fragments"""
        
        fragments = []
        
        if query_type == 'sparql':
            fragments = self._decompose_sparql_query(query, sources)
        elif query_type == 'pattern':
            fragments = self._decompose_pattern_query(query, sources)
        elif query_type == 'natural':
            fragments = self._decompose_natural_query(query, sources)
        else:
            # Default: create identical fragments for all sources
            for i, source_id in enumerate(sources):
                fragments.append(QueryFragment(
                    fragment_id=f"frag_{i}_{source_id}",
                    source_id=source_id,
                    query=query,
                    query_type=query_type
                ))
        
        return fragments
    
    def _decompose_sparql_query(self, query: str, sources: List[str]) -> List[QueryFragment]:
        """Decompose SPARQL query into source-specific fragments"""
        
        fragments = []
        
        # Parse SPARQL query structure
        query_structure = self._parse_sparql_structure(query)
        
        # Create fragments based on query patterns
        for i, source_id in enumerate(sources):
            source_caps = self.source_capabilities.get(source_id, set())
            
            # Adapt query based on source capabilities
            adapted_query = self._adapt_query_for_source(query, query_structure, source_caps)
            
            # If can't adapt, use original query (let the execution method handle it)
            if adapted_query is None:
                adapted_query = query
            
            fragment = QueryFragment(
                fragment_id=f"sparql_{i}_{source_id}",
                source_id=source_id,
                query=adapted_query,
                query_type='sparql',
                estimated_cost=self._estimate_fragment_cost(adapted_query, source_id)
            )
            fragments.append(fragment)
        
        return fragments
    
    def _decompose_pattern_query(self, query: str, sources: List[str]) -> List[QueryFragment]:
        """Decompose pattern-based query"""
        
        fragments = []
        
        # Parse pattern query
        try:
            pattern = json.loads(query)
        except:
            pattern = {'raw_query': query}
        
        for i, source_id in enumerate(sources):
            # Convert pattern to source-specific format
            source_query = self._convert_pattern_for_source(pattern, source_id)
            
            fragment = QueryFragment(
                fragment_id=f"pattern_{i}_{source_id}",
                source_id=source_id,
                query=source_query,
                query_type='pattern'
            )
            fragments.append(fragment)
        
        return fragments
    
    def _decompose_natural_query(self, query: str, sources: List[str]) -> List[QueryFragment]:
        """Decompose natural language query"""
        
        fragments = []
        
        for i, source_id in enumerate(sources):
            source_caps = self.source_capabilities.get(source_id, set())
            
            # Convert natural language to appropriate query format
            if 'sparql' in source_caps:
                converted_query = self._natural_to_sparql(query)
                query_type = 'sparql'
            elif 'graphql' in source_caps:
                converted_query = self._natural_to_graphql(query)
                query_type = 'graphql'
            else:
                converted_query = query  # Pass through as-is
                query_type = 'natural'
            
            fragment = QueryFragment(
                fragment_id=f"natural_{i}_{source_id}",
                source_id=source_id,
                query=converted_query,
                query_type=query_type
            )
            fragments.append(fragment)
        
        return fragments
    
    def _parse_sparql_structure(self, query: str) -> Dict[str, Any]:
        """Parse SPARQL query structure for decomposition"""
        
        structure = {
            'select_vars': [],
            'where_patterns': [],
            'filters': [],
            'order_by': [],
            'limit': None,
            'union_parts': []
        }
        
        # Clean up the query
        query = ' '.join(query.split())  # Normalize whitespace
        
        # Extract SELECT variables
        import re
        select_match = re.search(r'SELECT\s+(.*?)\s+WHERE', query, re.IGNORECASE)
        if select_match:
            vars_part = select_match.group(1)
            variables = re.findall(r'\?\w+', vars_part)
            structure['select_vars'] = variables
        
        # Extract WHERE clause content
        where_match = re.search(r'WHERE\s*\{(.*?)\}', query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_content = where_match.group(1).strip()
            
            # Split WHERE content into lines and process
            lines = [line.strip() for line in where_content.split('.') if line.strip()]
            
            for line in lines:
                line = line.strip()
                if line and not line.lower().startswith('filter'):
                    # Check if it's a triple pattern (contains variables)
                    if '?' in line:
                        structure['where_patterns'].append(line)
        
        # Extract FILTER clauses
        filter_matches = re.findall(r'FILTER\s*\([^)]+\)', query, re.IGNORECASE)
        structure['filters'] = filter_matches
        
        # Extract ORDER BY
        order_match = re.search(r'ORDER\s+BY\s+[\w?]+', query, re.IGNORECASE)
        if order_match:
            structure['order_by'] = [order_match.group(0)]
        
        # Extract LIMIT
        limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
        if limit_match:
            structure['limit'] = int(limit_match.group(1))
        
        return structure
    
    def _adapt_query_for_source(self, query: str, structure: Dict[str, Any], capabilities: Set[str]) -> Optional[str]:
        """Adapt query for specific source capabilities"""
        
        # If source doesn't support SPARQL, try to convert
        if 'sparql' not in capabilities:
            if 'graphql' in capabilities:
                return self._sparql_to_graphql(query, structure)
            elif 'sql' in capabilities:
                return self._sparql_to_sql(query, structure)
            else:
                return None  # Can't adapt
        
        # Apply capability-specific optimizations
        adapted_query = query
        
        # Remove unsupported features
        if 'inference' not in capabilities:
            # Remove inference-specific patterns
            adapted_query = adapted_query.replace('rdfs:subClassOf*', 'rdfs:subClassOf')
        
        if 'full_text_search' not in capabilities:
            # Remove full-text search functions
            adapted_query = adapted_query.replace('CONTAINS(', 'REGEX(')
        
        return adapted_query
    
    def _convert_pattern_for_source(self, pattern: Dict[str, Any], source_id: str) -> str:
        """Convert pattern query for specific source"""
        
        source_caps = self.source_capabilities.get(source_id, set())
        
        if 'sparql' in source_caps:
            # Convert to SPARQL
            if 'subject' in pattern and 'predicate' in pattern and 'object' in pattern:
                return f"SELECT * WHERE {{ {pattern['subject']} {pattern['predicate']} {pattern['object']} }}"
        
        elif 'graphql' in source_caps:
            # Convert to GraphQL
            return json.dumps(pattern)
        
        # Default: return as JSON
        return json.dumps(pattern)
    
    def _natural_to_sparql(self, query: str) -> str:
        """Convert natural language to SPARQL (simplified)"""
        
        # This is a placeholder for natural language processing
        # In a real implementation, this would use NLP models
        
        query_lower = query.lower()
        
        if 'find' in query_lower and 'person' in query_lower:
            return "SELECT ?person WHERE { ?person rdf:type Person }"
        
        elif 'count' in query_lower:
            return "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }"
        
        else:
            # Default query
            return "SELECT * WHERE { ?s ?p ?o } LIMIT 100"
    
    def _natural_to_graphql(self, query: str) -> str:
        """Convert natural language to GraphQL (simplified)"""
        
        # Placeholder implementation
        return "{ entities { id, type, properties } }"
    
    def _sparql_to_graphql(self, query: str, structure: Dict[str, Any]) -> str:
        """Convert SPARQL to GraphQL"""
        
        # Simplified conversion
        variables = structure.get('select_vars', ['id'])
        
        fields = []
        for var in variables:
            clean_var = var.replace('?', '')
            fields.append(clean_var)
        
        return f"{{ entities {{ {', '.join(fields)} }} }}"
    
    def _sparql_to_sql(self, query: str, structure: Dict[str, Any]) -> str:
        """Convert SPARQL to SQL"""
        
        # Very simplified conversion
        variables = structure.get('select_vars', ['*'])
        
        select_clause = ', '.join(var.replace('?', '') for var in variables)
        
        return f"SELECT {select_clause} FROM triples"
    
    def _estimate_fragment_cost(self, query: str, source_id: str) -> float:
        """Estimate execution cost for a query fragment"""
        
        # Base cost
        cost = 1.0
        
        # Query complexity factors
        if 'JOIN' in query.upper() or 'join' in query.lower():
            cost *= 2.0
        
        if 'UNION' in query.upper() or 'union' in query.lower():
            cost *= 1.5
        
        if 'FILTER' in query.upper() or 'filter' in query.lower():
            cost *= 1.3
        
        # Source-specific factors
        health = self.source_health.get(source_id, {})
        response_time = health.get('response_time_ms', 1000)
        cost *= (response_time / 100)  # Scale by response time
        
        return cost
    
    def _create_execution_plan(self, fragments: List[QueryFragment], optimization_level: int) -> FederatedQueryPlan:
        """Create execution plan for query fragments"""
        
        plan_id = hashlib.md5(f"{[f.fragment_id for f in fragments]}_{time.time()}".encode()).hexdigest()[:12]
        
        # Determine execution order based on dependencies
        execution_order = []
        parallelizable_groups = []
        
        if optimization_level == 0:
            # Sequential execution
            execution_order = [f.fragment_id for f in fragments]
            parallelizable_groups = [[f.fragment_id] for f in fragments]
        
        else:
            # Parallel execution where possible
            independent_fragments = [f for f in fragments if not f.dependencies]
            dependent_fragments = [f for f in fragments if f.dependencies]
            
            if independent_fragments:
                # All independent fragments can run in parallel
                group = [f.fragment_id for f in independent_fragments]
                parallelizable_groups.append(group)
                execution_order.extend(group)
            
            # Add dependent fragments (simplified - would need dependency analysis)
            for fragment in dependent_fragments:
                parallelizable_groups.append([fragment.fragment_id])
                execution_order.append(fragment.fragment_id)
        
        # Determine merge strategy
        merge_strategy = 'union'  # Default
        if len(fragments) > 1 and optimization_level > 1:
            merge_strategy = 'smart_merge'  # Deduplication and intelligent merging
        
        # Calculate total estimated cost
        total_cost = sum(f.estimated_cost for f in fragments)
        
        plan = FederatedQueryPlan(
            plan_id=plan_id,
            fragments=fragments,
            execution_order=execution_order,
            merge_strategy=merge_strategy,
            estimated_total_cost=total_cost,
            parallelizable_groups=parallelizable_groups,
            optimization_hints=[
                f"parallel_groups_{len(parallelizable_groups)}",
                f"merge_strategy_{merge_strategy}",
                f"optimization_level_{optimization_level}"
            ]
        )
        
        return plan
    
    def _optimize_execution_plan(self, plan: FederatedQueryPlan) -> FederatedQueryPlan:
        """Optimize execution plan using query optimizer"""
        
        # This could be enhanced to use the query optimizer
        # For now, apply simple optimizations
        
        optimized_plan = copy.deepcopy(plan)
        
        # Reorder fragments by estimated cost (cheapest first)
        if len(plan.fragments) > 1:
            sorted_fragments = sorted(plan.fragments, key=lambda f: f.estimated_cost)
            optimized_plan.fragments = sorted_fragments
            
            # Update execution order
            independent_fragments = [f for f in sorted_fragments if not f.dependencies]
            optimized_plan.execution_order = [f.fragment_id for f in independent_fragments]
            
            optimized_plan.optimization_hints.append("cost_based_reordering")
        
        return optimized_plan
    
    def _execute_plan(self, plan: FederatedQueryPlan) -> List[FragmentResult]:
        """Execute the federated query plan"""
        
        results = []
        
        # Execute parallelizable groups
        for group in plan.parallelizable_groups:
            group_fragments = [f for f in plan.fragments if f.fragment_id in group]
            
            if len(group_fragments) == 1:
                # Single fragment execution
                result = self._execute_fragment(group_fragments[0])
                results.append(result)
            
            else:
                # Parallel execution
                group_results = self._execute_fragments_parallel(group_fragments)
                results.extend(group_results)
        
        return results
    
    def _execute_fragment(self, fragment: QueryFragment) -> FragmentResult:
        """Execute a single query fragment"""
        
        start_time = time.time()
        
        try:
            source = self.data_sources[fragment.source_id]
            
            logger.debug(f"Executing fragment {fragment.fragment_id} on {source.name}")
            
            # Execute based on protocol
            if source.protocol == FederationProtocol.SPARQL_ENDPOINT:
                data = self._execute_sparql_endpoint(fragment, source)
            
            elif source.protocol == FederationProtocol.REST_API:
                data = self._execute_rest_api(fragment, source)
            
            elif source.protocol == FederationProtocol.GRAPHQL:
                data = self._execute_graphql(fragment, source)
            
            elif source.protocol == FederationProtocol.NATIVE_KG:
                data = self._execute_native_kg(fragment, source)
            
            elif source.protocol == FederationProtocol.DATABASE_DIRECT:
                data = self._execute_database_direct(fragment, source)
            
            else:
                raise ValueError(f"Unsupported protocol: {source.protocol}")
            
            execution_time = time.time() - start_time
            
            result = FragmentResult(
                fragment_id=fragment.fragment_id,
                source_id=fragment.source_id,
                success=True,
                data=data,
                execution_time=execution_time,
                row_count=len(data) if isinstance(data, list) else 1,
                metadata={'protocol': source.protocol.value}
            )
            
            logger.debug(f"Fragment {fragment.fragment_id} completed in {execution_time:.3f}s with {result.row_count} rows")
            
            return result
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(f"Fragment {fragment.fragment_id} failed: {str(e)}")
            
            return FragmentResult(
                fragment_id=fragment.fragment_id,
                source_id=fragment.source_id,
                success=False,
                error=str(e),
                execution_time=execution_time,
                row_count=0
            )
    
    def _execute_fragments_parallel(self, fragments: List[QueryFragment]) -> List[FragmentResult]:
        """Execute multiple fragments in parallel"""
        
        results = []
        
        # Submit all fragments for parallel execution
        future_to_fragment = {}
        for fragment in fragments:
            future = self.executor.submit(self._execute_fragment, fragment)
            future_to_fragment[future] = fragment
        
        # Collect results as they complete
        for future in as_completed(future_to_fragment.keys(), timeout=max(f.timeout_seconds for f in fragments)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                fragment = future_to_fragment[future]
                error_result = FragmentResult(
                    fragment_id=fragment.fragment_id,
                    source_id=fragment.source_id,
                    success=False,
                    error=f"Parallel execution error: {str(e)}",
                    execution_time=0.0,
                    row_count=0
                )
                results.append(error_result)
        
        return results
    
    def _execute_sparql_endpoint(self, fragment: QueryFragment, source: DataSource) -> List[Dict[str, Any]]:
        """Execute query against SPARQL endpoint"""
        
        # This is a placeholder - would use actual SPARQL client
        if requests:
            response = requests.post(
                source.endpoint_url,
                data={'query': fragment.query},
                headers={'Accept': 'application/json'},
                timeout=fragment.timeout_seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                # Extract bindings from SPARQL JSON format
                if 'results' in result and 'bindings' in result['results']:
                    return result['results']['bindings']
                else:
                    return [result]
            else:
                raise Exception(f"SPARQL endpoint error: {response.status_code}")
        
        else:
            # Fallback simulation
            return [{'result': f'simulated_sparql_{fragment.fragment_id}'}]
    
    def _execute_rest_api(self, fragment: QueryFragment, source: DataSource) -> List[Dict[str, Any]]:
        """Execute query against REST API"""
        
        # Placeholder implementation
        if requests:
            response = requests.get(
                f"{source.endpoint_url}?q={fragment.query}",
                timeout=fragment.timeout_seconds
            )
            
            if response.status_code == 200:
                data = response.json()
                return data if isinstance(data, list) else [data]
            else:
                raise Exception(f"REST API error: {response.status_code}")
        
        else:
            return [{'result': f'simulated_rest_{fragment.fragment_id}'}]
    
    def _execute_graphql(self, fragment: QueryFragment, source: DataSource) -> List[Dict[str, Any]]:
        """Execute GraphQL query"""
        
        # Placeholder implementation
        if requests:
            response = requests.post(
                source.endpoint_url,
                json={'query': fragment.query},
                timeout=fragment.timeout_seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'data' in result:
                    return [result['data']]
                else:
                    return [result]
            else:
                raise Exception(f"GraphQL error: {response.status_code}")
        
        else:
            return [{'result': f'simulated_graphql_{fragment.fragment_id}'}]
    
    def _execute_native_kg(self, fragment: QueryFragment, source: DataSource) -> List[Dict[str, Any]]:
        """Execute query against native knowledge graph"""
        
        # Placeholder - would integrate with actual KG
        if self.kg:
            # Use the local knowledge graph
            try:
                results = self.kg.semantic_search(pattern={'raw': fragment.query})
                return [results]
            except:
                return [{'result': f'native_kg_{fragment.fragment_id}'}]
        
        else:
            return [{'result': f'simulated_native_{fragment.fragment_id}'}]
    
    def _execute_database_direct(self, fragment: QueryFragment, source: DataSource) -> List[Dict[str, Any]]:
        """Execute query directly against database"""
        
        # Placeholder for database connection
        return [{'result': f'simulated_db_{fragment.fragment_id}'}]
    
    def _merge_fragment_results(self, 
                              fragment_results: List[FragmentResult], 
                              plan: FederatedQueryPlan) -> Optional[Any]:
        """Merge results from multiple fragments"""
        
        successful_results = [fr for fr in fragment_results if fr.success]
        
        if not successful_results:
            return None
        
        if plan.merge_strategy == 'union':
            # Simple union of all results
            merged_data = []
            for result in successful_results:
                if isinstance(result.data, list):
                    merged_data.extend(result.data)
                else:
                    merged_data.append(result.data)
            
            return merged_data
        
        elif plan.merge_strategy == 'smart_merge':
            # Intelligent merging with deduplication
            return self._smart_merge_results(successful_results)
        
        else:
            # Default: return first successful result
            return successful_results[0].data
    
    def _smart_merge_results(self, results: List[FragmentResult]) -> List[Dict[str, Any]]:
        """Intelligently merge and deduplicate results"""
        
        merged_data = []
        seen_items = set()
        
        for result in results:
            if isinstance(result.data, list):
                for item in result.data:
                    # Create hash for deduplication
                    item_hash = hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()
                    
                    if item_hash not in seen_items:
                        seen_items.add(item_hash)
                        # Add source information
                        enhanced_item = dict(item)
                        enhanced_item['_source'] = result.source_id
                        merged_data.append(enhanced_item)
            
            else:
                # Single item
                if result.data is not None:
                    item_hash = hashlib.md5(json.dumps(result.data, sort_keys=True).encode()).hexdigest()
                    if item_hash not in seen_items:
                        seen_items.add(item_hash)
                        enhanced_item = dict(result.data) if isinstance(result.data, dict) else {'data': result.data}
                        enhanced_item['_source'] = result.source_id
                        merged_data.append(enhanced_item)
        
        return merged_data
    
    def _start_health_monitoring(self):
        """Start background health monitoring for data sources"""
        
        def health_monitor():
            while True:
                try:
                    for source_id in self.data_sources.keys():
                        self._check_source_health(source_id)
                    
                    time.sleep(self.config['health_check_interval'])
                
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    time.sleep(30)  # Wait before retrying
        
        # Run in background thread
        health_thread = threading.Thread(target=health_monitor, daemon=True)
        health_thread.start()
    
    def _check_source_health(self, source_id: str) -> None:
        """Check health of a specific data source"""
        
        if source_id not in self.data_sources:
            return
        
        source = self.data_sources[source_id]
        
        start_time = time.time()
        
        try:
            # Simple health check - try to connect
            if requests:
                response = requests.get(source.endpoint_url, timeout=5)
                status = 'healthy' if response.status_code < 400 else 'degraded'
            else:
                status = 'healthy'  # Assume healthy if no requests library
            
            response_time = (time.time() - start_time) * 1000  # milliseconds
            
            self.source_health[source_id] = {
                'status': status,
                'response_time_ms': response_time,
                'last_checked': time.time()
            }
            
        except Exception as e:
            self.source_health[source_id] = {
                'status': 'unhealthy',
                'error': str(e),
                'response_time_ms': 0,
                'last_checked': time.time()
            }
    
    def _is_source_healthy(self, source_id: str) -> bool:
        """Check if a data source is healthy"""
        
        health = self.source_health.get(source_id, {})
        return health.get('status') in ['healthy', 'degraded']
    
    def _get_cache_key(self, query: str, sources: Optional[List[str]]) -> str:
        """Generate cache key for query result"""
        
        cache_data = {
            'query': query,
            'sources': sorted(sources) if sources else None
        }
        
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    def _update_execution_stats(self, result: FederatedQueryResult) -> None:
        """Update execution statistics"""
        
        stats = {
            'execution_time': result.total_execution_time,
            'total_rows': result.total_rows,
            'success': result.success,
            'sources_used': len(result.sources_used or []),
            'timestamp': time.time()
        }
        
        self.execution_stats['query_history'].append(stats)
        
        # Keep only recent stats (last 1000 queries)
        if len(self.execution_stats['query_history']) > 1000:
            self.execution_stats['query_history'] = self.execution_stats['query_history'][-1000:]
    
    def get_federation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive federation statistics"""
        
        stats = {
            'sources': {
                'total_registered': len(self.data_sources),
                'healthy_sources': len([s for s in self.data_sources.keys() if self._is_source_healthy(s)]),
                'source_details': {}
            },
            'execution': {
                'total_queries': len(self.execution_stats.get('query_history', [])),
                'cache_hits': len(self.result_cache),
                'average_execution_time': 0.0,
                'success_rate': 0.0
            },
            'performance': {
                'active_queries': len(self.active_queries),
                'plan_cache_size': len(self.plan_cache),
                'result_cache_size': len(self.result_cache)
            }
        }
        
        # Source details
        for source_id, source in self.data_sources.items():
            health = self.source_health.get(source_id, {})
            stats['sources']['source_details'][source_id] = {
                'name': source.name,
                'protocol': source.protocol.value,
                'status': health.get('status', 'unknown'),
                'response_time_ms': health.get('response_time_ms', 0),
                'capabilities': list(self.source_capabilities.get(source_id, []))
            }
        
        # Execution statistics
        query_history = self.execution_stats.get('query_history', [])
        if query_history:
            stats['execution']['average_execution_time'] = sum(q['execution_time'] for q in query_history) / len(query_history)
            stats['execution']['success_rate'] = sum(1 for q in query_history if q['success']) / len(query_history)
        
        return stats
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)