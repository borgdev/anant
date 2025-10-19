"""
Query Optimization Engine for Knowledge Graphs
==============================================

Advanced query optimization with cost-based optimization, adaptive indexing,
query pattern learning, and execution plan optimization.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import defaultdict, Counter
from dataclasses import dataclass
import json
import hashlib
from enum import Enum

from ..utils.performance import performance_monitor, PerformanceProfiler
from ..utils.extras import safe_import

# Optional dependencies
sklearn = safe_import('sklearn')

logger = logging.getLogger(__name__)


class JoinType(Enum):
    """Types of join operations"""
    NESTED_LOOP = "nested_loop"
    HASH_JOIN = "hash_join"
    MERGE_JOIN = "merge_join"
    INDEX_JOIN = "index_join"


class IndexType(Enum):
    """Types of indexes"""
    BTREE = "btree"
    HASH = "hash"
    BITMAP = "bitmap"
    GRAPH = "graph"


@dataclass
class QueryStatistics:
    """Statistics for query optimization"""
    total_nodes: int
    total_edges: int
    node_selectivity: Dict[str, float]
    edge_selectivity: Dict[str, float]
    join_cardinality: Dict[str, int]
    index_statistics: Dict[str, Dict[str, Any]]


@dataclass
class ExecutionPlan:
    """Query execution plan"""
    plan_id: str
    operations: List[Dict[str, Any]]
    estimated_cost: float
    estimated_cardinality: int
    optimization_hints: List[str]
    indexes_used: List[str]


@dataclass
class OptimizationResult:
    """Result of query optimization"""
    original_query: str
    optimized_query: str
    execution_plan: ExecutionPlan
    optimization_time: float
    estimated_speedup: float
    statistics_used: QueryStatistics


class QueryOptimizer:
    """
    Advanced Query Optimization Engine
    
    Provides:
    - Cost-based query optimization
    - Adaptive indexing and statistics collection
    - Query pattern learning and caching
    - Join reordering and predicate pushdown
    - Cardinality estimation
    - Index recommendation
    """
    
    def __init__(self, knowledge_graph):
        """
        Initialize query optimizer
        
        Args:
            knowledge_graph: KnowledgeGraph instance
        """
        self.kg = knowledge_graph
        
        # Statistics and cost models
        self.statistics = None
        self.cost_model = CostModel()
        self.cardinality_estimator = CardinalityEstimator()
        
        # Indexes and caching
        self.indexes = {}
        self.query_cache = {}
        self.pattern_cache = {}
        
        # Adaptive learning
        self.query_history = []
        self.performance_feedback = defaultdict(list)
        
        # Configuration
        self.config = {
            'enable_cost_based_optimization': True,
            'enable_adaptive_indexing': True,
            'enable_query_caching': True,
            'max_cache_size': 1000,
            'statistics_update_threshold': 100,
            'join_reordering_threshold': 3
        }
        
        logger.info("Query Optimizer initialized")
    
    @performance_monitor("query_optimization")
    def optimize_query(self, query: str, query_type: str = 'sparql') -> OptimizationResult:
        """
        Optimize a query using cost-based optimization
        
        Args:
            query: Query string to optimize
            query_type: Type of query ('sparql', 'pattern', 'graph')
            
        Returns:
            OptimizationResult with optimized query and execution plan
        """
        
        start_time = time.time()
        
        logger.info(f"Optimizing {query_type} query...")
        
        with PerformanceProfiler("query_optimization") as profiler:
            
            profiler.checkpoint("query_analysis")
            
            # Parse and analyze query
            query_structure = self._parse_query(query, query_type)
            
            profiler.checkpoint("statistics_collection")
            
            # Collect/update statistics if needed
            if not self.statistics or self._should_update_statistics():
                self.statistics = self._collect_statistics()
            
            profiler.checkpoint("plan_generation")
            
            # Generate multiple execution plans
            candidate_plans = self._generate_execution_plans(query_structure)
            
            profiler.checkpoint("cost_estimation")
            
            # Estimate costs for each plan
            for plan in candidate_plans:
                plan.estimated_cost = self._estimate_plan_cost(plan)
                plan.estimated_cardinality = self._estimate_plan_cardinality(plan)
            
            profiler.checkpoint("plan_selection")
            
            # Select best plan
            best_plan = min(candidate_plans, key=lambda p: p.estimated_cost)
            
            profiler.checkpoint("query_rewriting")
            
            # Generate optimized query
            optimized_query = self._generate_optimized_query(query_structure, best_plan)
            
            profiler.checkpoint("optimization_complete")
        
        optimization_time = time.time() - start_time
        
        # Calculate estimated speedup
        baseline_cost = candidate_plans[0].estimated_cost if candidate_plans else 1.0
        estimated_speedup = baseline_cost / best_plan.estimated_cost if best_plan.estimated_cost > 0 else 1.0
        
        result = OptimizationResult(
            original_query=query,
            optimized_query=optimized_query,
            execution_plan=best_plan,
            optimization_time=optimization_time,
            estimated_speedup=estimated_speedup,
            statistics_used=self.statistics
        )
        
        # Cache result for future use
        if self.config['enable_query_caching']:
            query_hash = self._hash_query(query)
            self.query_cache[query_hash] = result
        
        # Record query for adaptive learning
        self.query_history.append({
            'query': query,
            'optimization_time': optimization_time,
            'estimated_cost': best_plan.estimated_cost,
            'timestamp': time.time()
        })
        
        logger.info(f"Query optimization completed in {optimization_time:.3f}s (estimated {estimated_speedup:.2f}x speedup)")
        
        return result
    
    def _parse_query(self, query: str, query_type: str) -> Dict[str, Any]:
        """Parse query into structured representation"""
        
        query_structure = {
            'type': query_type,
            'original': query,
            'patterns': [],
            'filters': [],
            'joins': [],
            'projections': [],
            'ordering': [],
            'limits': None
        }
        
        if query_type == 'sparql':
            query_structure = self._parse_sparql_query(query)
        elif query_type == 'pattern':
            query_structure = self._parse_pattern_query(query)
        elif query_type == 'graph':
            query_structure = self._parse_graph_query(query)
        
        return query_structure
    
    def _parse_sparql_query(self, query: str) -> Dict[str, Any]:
        """Parse SPARQL-like query"""
        
        # Simple SPARQL parsing (can be extended with full SPARQL parser)
        lines = [line.strip() for line in query.strip().split('\n') if line.strip()]
        
        structure = {
            'type': 'sparql',
            'original': query,
            'patterns': [],
            'filters': [],
            'joins': [],
            'projections': [],
            'ordering': [],
            'limits': None
        }
        
        current_section = None
        
        for line in lines:
            line_lower = line.lower()
            
            # Handle ORDER BY and LIMIT first to prevent them being processed as patterns
            if 'order by' in line_lower:
                # Handle case insensitive order by parsing - also reset section
                current_section = 'order'
                order_idx = line_lower.find('order by')
                if order_idx != -1:
                    order_part = line[order_idx + 8:].strip()
                    order_vars = [var.strip() for var in order_part.split()]
                    structure['ordering'] = order_vars
            
            elif line_lower.startswith('limit'):
                # Handle case insensitive limit parsing - also reset section
                current_section = 'limit'
                limit_idx = line_lower.find('limit')
                if limit_idx != -1:
                    limit_part = line[limit_idx + 5:].strip()
                    try:
                        limit_val = int(limit_part)
                        structure['limits'] = limit_val
                    except ValueError:
                        pass  # Ignore invalid limit values
            
            elif line_lower.startswith('select'):
                current_section = 'select'
                # Extract variables - handle case insensitive split
                if 'select' in line_lower:
                    select_idx = line_lower.find('select')
                    select_part = line[select_idx + 6:]  # Skip 'select'
                    if 'where' in select_part.lower():
                        where_idx = select_part.lower().find('where')
                        variables_part = select_part[:where_idx]
                        # If WHERE is on same line, set section to where
                        current_section = 'where'
                    else:
                        variables_part = select_part
                    variables = [var.strip() for var in variables_part.split()]
                    structure['projections'] = [var for var in variables if var.startswith('?')]
            
            elif line_lower.startswith('where') or current_section == 'where':
                if line_lower.startswith('where'):
                    current_section = 'where'
                
                # Parse triple patterns - skip braces
                clean_line = line.replace('{', '').replace('}', '').strip()
                if clean_line and clean_line.lower() != 'where' and ('?' in clean_line or any(keyword in clean_line.lower() for keyword in ['rdf:', 'foaf:', 'type', 'name'])):
                    pattern = self._parse_triple_pattern(clean_line)
                    if pattern:
                        structure['patterns'].append(pattern)
            
            elif line_lower.startswith('filter'):
                # Handle case insensitive filter parsing
                filter_idx = line_lower.find('filter')
                if filter_idx != -1:
                    filter_expr = line[filter_idx + 6:].strip()
                    structure['filters'].append(filter_expr)
        
        return structure
    
    def _parse_triple_pattern(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a triple pattern from SPARQL"""
        
        # Simple triple pattern parsing
        line = line.strip().rstrip('.')
        
        # Remove common SPARQL syntax
        for char in ['{', '}']:
            line = line.replace(char, '')
        
        parts = line.split()
        
        if len(parts) >= 3:
            return {
                'subject': parts[0],
                'predicate': parts[1],
                'object': ' '.join(parts[2:]),
                'variables': [p for p in parts if p.startswith('?')]
            }
        
        return None
    
    def _parse_pattern_query(self, query: str) -> Dict[str, Any]:
        """Parse pattern-based query"""
        
        # Assume query is JSON-like pattern
        try:
            import json
            pattern = json.loads(query)
            
            return {
                'type': 'pattern',
                'original': query,
                'patterns': [pattern] if isinstance(pattern, dict) else pattern,
                'filters': [],
                'joins': [],
                'projections': pattern.get('variables', []) if isinstance(pattern, dict) else [],
                'ordering': [],
                'limits': pattern.get('limit') if isinstance(pattern, dict) else None
            }
        except:
            # Fallback parsing
            return {
                'type': 'pattern',
                'original': query,
                'patterns': [{'raw': query}],
                'filters': [],
                'joins': [],
                'projections': [],
                'ordering': [],
                'limits': None
            }
    
    def _parse_graph_query(self, query: str) -> Dict[str, Any]:
        """Parse graph traversal query"""
        
        return {
            'type': 'graph',
            'original': query,
            'patterns': [{'traversal': query}],
            'filters': [],
            'joins': [],
            'projections': [],
            'ordering': [],
            'limits': None
        }
    
    def _collect_statistics(self) -> QueryStatistics:
        """Collect comprehensive statistics for optimization"""
        
        logger.info("Collecting query statistics...")
        
        start_time = time.time()
        
        # Basic graph statistics
        total_nodes = self.kg.num_nodes
        total_edges = self.kg.num_edges
        
        # Node selectivity (frequency distribution)
        node_types = defaultdict(int)
        node_degrees = {}
        
        for node in self.kg.nodes:
            # Get node type
            try:
                node_type = self.kg.properties.get_node_property(node, 'entity_type')
                if node_type:
                    node_types[node_type] += 1
            except:
                node_types['unknown'] += 1
            
            # Get node degree
            node_degrees[node] = self.kg.get_node_degree(node)
        
        # Calculate node selectivity
        node_selectivity = {}
        for node_type, count in node_types.items():
            node_selectivity[node_type] = count / total_nodes if total_nodes > 0 else 0.0
        
        # Edge selectivity (relation frequency)
        relation_types = defaultdict(int)
        edge_cardinalities = {}
        
        for edge_id in self.kg.edges:
            try:
                relation_type = self.kg.properties.get_edge_property(edge_id, 'relation_type')
                if relation_type:
                    relation_types[relation_type] += 1
                
                edge_nodes = self.kg.get_edge_nodes(edge_id)
                edge_cardinalities[edge_id] = len(edge_nodes)
            except:
                relation_types['unknown'] += 1
        
        # Calculate edge selectivity
        edge_selectivity = {}
        for relation_type, count in relation_types.items():
            edge_selectivity[relation_type] = count / total_edges if total_edges > 0 else 0.0
        
        # Join cardinality estimates
        join_cardinality = {}
        
        # Sample join cardinalities
        for relation_type in relation_types.keys():
            # Estimate average join cardinality
            sample_edges = [eid for eid in self.kg.edges 
                          if self.kg.properties.get_edge_property(eid, 'relation_type') == relation_type][:100]
            
            if sample_edges:
                avg_cardinality = sum(len(self.kg.get_edge_nodes(eid)) for eid in sample_edges) / len(sample_edges)
                join_cardinality[relation_type] = int(avg_cardinality)
        
        # Index statistics
        index_statistics = {}
        for index_name, index_data in self.indexes.items():
            index_statistics[index_name] = {
                'type': index_data.get('type', 'unknown'),
                'size': index_data.get('size', 0),
                'selectivity': index_data.get('selectivity', 1.0),
                'last_updated': index_data.get('last_updated', time.time())
            }
        
        statistics = QueryStatistics(
            total_nodes=total_nodes,
            total_edges=total_edges,
            node_selectivity=node_selectivity,
            edge_selectivity=edge_selectivity,
            join_cardinality=join_cardinality,
            index_statistics=index_statistics
        )
        
        collection_time = time.time() - start_time
        logger.info(f"Statistics collected in {collection_time:.3f}s")
        
        return statistics
    
    def _should_update_statistics(self) -> bool:
        """Determine if statistics should be updated"""
        
        if not hasattr(self, '_last_stats_update'):
            self._last_stats_update = 0
            self._stats_update_counter = 0
        
        self._stats_update_counter += 1
        
        # Update based on threshold or time
        time_since_update = time.time() - self._last_stats_update
        
        return (self._stats_update_counter >= self.config['statistics_update_threshold'] or
                time_since_update > 3600)  # Update every hour
    
    def _generate_execution_plans(self, query_structure: Dict[str, Any]) -> List[ExecutionPlan]:
        """Generate multiple execution plans for cost comparison"""
        
        plans = []
        
        # Plan 1: Baseline (no optimization)
        baseline_plan = ExecutionPlan(
            plan_id="baseline",
            operations=self._generate_baseline_operations(query_structure),
            estimated_cost=0.0,
            estimated_cardinality=0,
            optimization_hints=[],
            indexes_used=[]
        )
        plans.append(baseline_plan)
        
        # Plan 2: With join reordering
        if len(query_structure.get('patterns', [])) >= self.config['join_reordering_threshold']:
            reordered_plan = ExecutionPlan(
                plan_id="join_reordered",
                operations=self._generate_join_reordered_operations(query_structure),
                estimated_cost=0.0,
                estimated_cardinality=0,
                optimization_hints=["join_reordering"],
                indexes_used=[]
            )
            plans.append(reordered_plan)
        
        # Plan 3: With predicate pushdown
        if query_structure.get('filters'):
            pushdown_plan = ExecutionPlan(
                plan_id="predicate_pushdown",
                operations=self._generate_pushdown_operations(query_structure),
                estimated_cost=0.0,
                estimated_cardinality=0,
                optimization_hints=["predicate_pushdown"],
                indexes_used=[]
            )
            plans.append(pushdown_plan)
        
        # Plan 4: With index usage
        if self.indexes:
            index_plan = ExecutionPlan(
                plan_id="index_optimized",
                operations=self._generate_index_operations(query_structure),
                estimated_cost=0.0,
                estimated_cardinality=0,
                optimization_hints=["index_usage"],
                indexes_used=list(self.indexes.keys())
            )
            plans.append(index_plan)
        
        return plans
    
    def _generate_baseline_operations(self, query_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate baseline execution operations"""
        
        operations = []
        
        # Sequential pattern matching
        for i, pattern in enumerate(query_structure.get('patterns', [])):
            operations.append({
                'type': 'pattern_match',
                'pattern': pattern,
                'method': 'sequential_scan',
                'order': i
            })
        
        # Apply filters
        for filter_expr in query_structure.get('filters', []):
            operations.append({
                'type': 'filter',
                'expression': filter_expr,
                'method': 'post_filter'
            })
        
        # Join operations
        if len(query_structure.get('patterns', [])) > 1:
            operations.append({
                'type': 'join',
                'method': JoinType.NESTED_LOOP.value,
                'patterns': query_structure['patterns']
            })
        
        # Projection
        if query_structure.get('projections'):
            operations.append({
                'type': 'projection',
                'variables': query_structure['projections']
            })
        
        return operations
    
    def _generate_join_reordered_operations(self, query_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate operations with optimized join order"""
        
        operations = []
        patterns = query_structure.get('patterns', [])
        
        if len(patterns) <= 1:
            return self._generate_baseline_operations(query_structure)
        
        # Reorder joins based on selectivity
        reordered_patterns = self._optimize_join_order(patterns)
        
        # Generate operations with reordered patterns
        for i, pattern in enumerate(reordered_patterns):
            operations.append({
                'type': 'pattern_match',
                'pattern': pattern,
                'method': 'selective_scan',
                'order': i
            })
        
        # Use hash joins for better performance
        operations.append({
            'type': 'join',
            'method': JoinType.HASH_JOIN.value,
            'patterns': reordered_patterns
        })
        
        # Apply other operations
        for filter_expr in query_structure.get('filters', []):
            operations.append({
                'type': 'filter',
                'expression': filter_expr,
                'method': 'early_filter'
            })
        
        if query_structure.get('projections'):
            operations.append({
                'type': 'projection',
                'variables': query_structure['projections']
            })
        
        return operations
    
    def _generate_pushdown_operations(self, query_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate operations with predicate pushdown"""
        
        operations = []
        patterns = query_structure.get('patterns', [])
        filters = query_structure.get('filters', [])
        
        # Push filters down to pattern matching level
        for i, pattern in enumerate(patterns):
            applicable_filters = self._find_applicable_filters(pattern, filters)
            
            operations.append({
                'type': 'pattern_match',
                'pattern': pattern,
                'method': 'filtered_scan',
                'filters': applicable_filters,
                'order': i
            })
        
        # Remaining filters
        remaining_filters = [f for f in filters if not self._is_filter_pushed_down(f, patterns)]
        for filter_expr in remaining_filters:
            operations.append({
                'type': 'filter',
                'expression': filter_expr,
                'method': 'post_filter'
            })
        
        # Join operations
        if len(patterns) > 1:
            operations.append({
                'type': 'join',
                'method': JoinType.MERGE_JOIN.value,
                'patterns': patterns
            })
        
        if query_structure.get('projections'):
            operations.append({
                'type': 'projection',
                'variables': query_structure['projections']
            })
        
        return operations
    
    def _generate_index_operations(self, query_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate operations using available indexes"""
        
        operations = []
        patterns = query_structure.get('patterns', [])
        
        for i, pattern in enumerate(patterns):
            # Find best index for pattern
            best_index = self._find_best_index(pattern)
            
            if best_index:
                operations.append({
                    'type': 'pattern_match',
                    'pattern': pattern,
                    'method': 'index_scan',
                    'index': best_index,
                    'order': i
                })
            else:
                operations.append({
                    'type': 'pattern_match',
                    'pattern': pattern,
                    'method': 'sequential_scan',
                    'order': i
                })
        
        # Use index joins when possible
        join_method = JoinType.INDEX_JOIN.value if any(op.get('index') for op in operations) else JoinType.HASH_JOIN.value
        
        if len(patterns) > 1:
            operations.append({
                'type': 'join',
                'method': join_method,
                'patterns': patterns
            })
        
        # Other operations
        for filter_expr in query_structure.get('filters', []):
            operations.append({
                'type': 'filter',
                'expression': filter_expr,
                'method': 'indexed_filter' if self.indexes else 'post_filter'
            })
        
        if query_structure.get('projections'):
            operations.append({
                'type': 'projection',
                'variables': query_structure['projections']
            })
        
        return operations
    
    def _optimize_join_order(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize join order based on selectivity estimates"""
        
        if not self.statistics:
            return patterns
        
        # Calculate selectivity for each pattern
        pattern_selectivities = []
        
        for pattern in patterns:
            selectivity = self._estimate_pattern_selectivity(pattern)
            pattern_selectivities.append((pattern, selectivity))
        
        # Sort by selectivity (most selective first)
        pattern_selectivities.sort(key=lambda x: x[1])
        
        return [pattern for pattern, _ in pattern_selectivities]
    
    def _estimate_pattern_selectivity(self, pattern: Dict[str, Any]) -> float:
        """Estimate selectivity of a pattern"""
        
        if not self.statistics:
            return 1.0
        
        # Simple selectivity estimation
        selectivity = 1.0
        
        # Check subject selectivity
        subject = pattern.get('subject', '')
        if subject and not subject.startswith('?'):
            selectivity *= 0.01  # Specific entity is highly selective
        
        # Check predicate selectivity
        predicate = pattern.get('predicate', '')
        if predicate in self.statistics.edge_selectivity:
            selectivity *= self.statistics.edge_selectivity[predicate]
        elif predicate == 'rdf:type':
            selectivity *= 0.1  # Type constraints are moderately selective
        
        # Check object selectivity
        obj = pattern.get('object', '')
        if obj and not obj.startswith('?'):
            # Specific literal values are highly selective
            if obj.isdigit():
                selectivity *= 0.00001  # Numbers are extremely selective
            elif obj.startswith('"'):
                selectivity *= 0.0001  # Quoted strings are very selective  
            else:
                selectivity *= 0.001   # Other constants are selective
        
        return max(selectivity, 0.000001)  # Minimum selectivity
    
    def _find_applicable_filters(self, pattern: Dict[str, Any], filters: List[str]) -> List[str]:
        """Find filters that can be applied to a specific pattern"""
        
        applicable = []
        pattern_variables = set(pattern.get('variables', []))
        
        for filter_expr in filters:
            # Simple check if filter variables overlap with pattern variables
            filter_variables = set(var for var in filter_expr.split() if var.startswith('?'))
            
            if filter_variables.intersection(pattern_variables):
                applicable.append(filter_expr)
        
        return applicable
    
    def _is_filter_pushed_down(self, filter_expr: str, patterns: List[Dict[str, Any]]) -> bool:
        """Check if a filter has been pushed down to pattern level"""
        
        filter_variables = set(var for var in filter_expr.split() if var.startswith('?'))
        
        for pattern in patterns:
            pattern_variables = set(pattern.get('variables', []))
            if filter_variables.intersection(pattern_variables):
                return True
        
        return False
    
    def _find_best_index(self, pattern: Dict[str, Any]) -> Optional[str]:
        """Find the best index for a pattern"""
        
        if not self.indexes:
            return None
        
        best_index = None
        best_selectivity = 1.0
        
        for index_name, index_data in self.indexes.items():
            # Check if index is applicable to pattern
            if self._is_index_applicable(pattern, index_data):
                selectivity = index_data.get('selectivity', 1.0)
                if selectivity < best_selectivity:
                    best_selectivity = selectivity
                    best_index = index_name
        
        return best_index
    
    def _is_index_applicable(self, pattern: Dict[str, Any], index_data: Dict[str, Any]) -> bool:
        """Check if an index is applicable to a pattern"""
        
        index_field = index_data.get('field', '')
        
        # Check if pattern can use this index
        if index_field == 'entity_type':
            # Can use entity_type index for rdf:type patterns
            predicate = pattern.get('predicate', '')
            return predicate == 'rdf:type' or 'type' in predicate.lower()
        
        elif index_field == 'relation_type':
            # Can use relation_type index for any predicate
            return 'predicate' in pattern
        
        else:
            # General field matching
            index_fields = set(index_data.get('fields', []))
            pattern_fields = set(pattern.keys())
            return bool(index_fields.intersection(pattern_fields))
    
    def _estimate_plan_cost(self, plan: ExecutionPlan) -> float:
        """Estimate the cost of an execution plan"""
        
        total_cost = 0.0
        
        for operation in plan.operations:
            op_type = operation.get('type', '')
            
            if op_type == 'pattern_match':
                cost = self._estimate_pattern_match_cost(operation)
            elif op_type == 'join':
                cost = self._estimate_join_cost(operation)
            elif op_type == 'filter':
                cost = self._estimate_filter_cost(operation)
            elif op_type == 'projection':
                cost = self._estimate_projection_cost(operation)
            else:
                cost = 1.0  # Default cost
            
            total_cost += cost
        
        return total_cost
    
    def _estimate_pattern_match_cost(self, operation: Dict[str, Any]) -> float:
        """Estimate cost of pattern matching operation"""
        
        method = operation.get('method', 'sequential_scan')
        pattern = operation.get('pattern', {})
        
        if method == 'sequential_scan':
            # Cost proportional to number of edges
            return self.statistics.total_edges * 1.0 if self.statistics else 1000.0
        
        elif method == 'index_scan':
            # Much lower cost with index
            selectivity = self._estimate_pattern_selectivity(pattern)
            return self.statistics.total_edges * selectivity * 0.1 if self.statistics else 100.0
        
        elif method == 'selective_scan':
            # Medium cost with selectivity optimization
            selectivity = self._estimate_pattern_selectivity(pattern)
            return self.statistics.total_edges * selectivity * 0.5 if self.statistics else 500.0
        
        else:
            return 100.0
    
    def _estimate_join_cost(self, operation: Dict[str, Any]) -> float:
        """Estimate cost of join operation"""
        
        method = operation.get('method', JoinType.NESTED_LOOP.value)
        patterns = operation.get('patterns', [])
        
        # Estimate cardinalities
        left_cardinality = 1000  # Default estimate
        right_cardinality = 1000
        
        if len(patterns) >= 2:
            left_selectivity = self._estimate_pattern_selectivity(patterns[0])
            right_selectivity = self._estimate_pattern_selectivity(patterns[1])
            
            if self.statistics:
                left_cardinality = self.statistics.total_edges * left_selectivity
                right_cardinality = self.statistics.total_edges * right_selectivity
        
        # Cost based on join method
        if method == JoinType.NESTED_LOOP.value:
            return left_cardinality * right_cardinality
        elif method == JoinType.HASH_JOIN.value:
            return left_cardinality + right_cardinality
        elif method == JoinType.MERGE_JOIN.value:
            return left_cardinality * np.log(left_cardinality) + right_cardinality * np.log(right_cardinality)
        elif method == JoinType.INDEX_JOIN.value:
            return left_cardinality * np.log(right_cardinality)
        else:
            return left_cardinality + right_cardinality
    
    def _estimate_filter_cost(self, operation: Dict[str, Any]) -> float:
        """Estimate cost of filter operation"""
        
        method = operation.get('method', 'post_filter')
        
        if method == 'early_filter':
            return 10.0  # Low cost when applied early
        elif method == 'indexed_filter':
            return 5.0   # Very low cost with index
        else:
            return 50.0  # Higher cost for post-processing
    
    def _estimate_projection_cost(self, operation: Dict[str, Any]) -> float:
        """Estimate cost of projection operation"""
        
        variables = operation.get('variables', [])
        return len(variables) * 1.0  # Linear cost in number of variables
    
    def _estimate_plan_cardinality(self, plan: ExecutionPlan) -> int:
        """Estimate the output cardinality of a plan"""
        
        # Simple cardinality estimation
        cardinality = 1000  # Default
        
        pattern_ops = [op for op in plan.operations if op.get('type') == 'pattern_match']
        
        if pattern_ops and self.statistics:
            total_selectivity = 1.0
            
            for op in pattern_ops:
                pattern_selectivity = self._estimate_pattern_selectivity(op.get('pattern', {}))
                total_selectivity *= pattern_selectivity
            
            cardinality = int(self.statistics.total_edges * total_selectivity)
        
        return max(cardinality, 1)
    
    def _generate_optimized_query(self, query_structure: Dict[str, Any], plan: ExecutionPlan) -> str:
        """Generate optimized query string from execution plan"""
        
        # For now, return a representation of the optimized structure
        optimization_notes = []
        
        if "join_reordering" in plan.optimization_hints:
            optimization_notes.append("/* JOIN REORDERED */")
        
        if "predicate_pushdown" in plan.optimization_hints:
            optimization_notes.append("/* PREDICATE PUSHDOWN */")
        
        if "index_usage" in plan.optimization_hints:
            optimization_notes.append(f"/* USING INDEXES: {', '.join(plan.indexes_used)} */")
        
        # Reconstruct query with optimization hints
        optimized = query_structure['original']
        
        if optimization_notes:
            optimized = '\n'.join(optimization_notes) + '\n' + optimized
        
        return optimized
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query caching"""
        return hashlib.md5(query.encode()).hexdigest()
    
    @performance_monitor("index_creation")
    def create_adaptive_index(self, field: str, index_type: IndexType = IndexType.BTREE) -> str:
        """
        Create an adaptive index based on query patterns
        
        Args:
            field: Field to index on
            index_type: Type of index to create
            
        Returns:
            Index name/ID
        """
        
        index_name = f"{field}_{index_type.value}_{int(time.time())}"
        
        logger.info(f"Creating {index_type.value} index on {field}...")
        
        # Create index structure (simplified)
        index_data = {
            'name': index_name,
            'field': field,
            'type': index_type.value,
            'created': time.time(),
            'last_updated': time.time(),
            'size': 0,
            'selectivity': 1.0,
            'fields': [field]
        }
        
        # Build index based on field
        if field == 'entity_type':
            index_data['data'] = self._build_entity_type_index()
        elif field == 'relation_type':
            index_data['data'] = self._build_relation_type_index()
        else:
            index_data['data'] = {}
        
        index_data['size'] = len(index_data['data'])
        # Calculate selectivity based on the distribution of data
        if index_data['size'] > 0:
            # For indexes, selectivity should be better than 1.0
            # Use the inverse of the number of distinct values
            index_data['selectivity'] = 1.0 / max(index_data['size'], 1) * 0.1  # 10% of sequential scan
        else:
            index_data['selectivity'] = 1.0
        
        self.indexes[index_name] = index_data
        
        logger.info(f"Index {index_name} created with {index_data['size']} entries")
        
        return index_name
    
    def _build_entity_type_index(self) -> Dict[str, List[str]]:
        """Build index for entity types"""
        
        index = defaultdict(list)
        
        for node in self.kg.nodes:
            try:
                entity_type = self.kg.properties.get_node_property(node, 'entity_type')
                if entity_type:
                    index[entity_type].append(node)
            except:
                index['unknown'].append(node)
        
        return dict(index)
    
    def _build_relation_type_index(self) -> Dict[str, List[str]]:
        """Build index for relation types"""
        
        index = defaultdict(list)
        
        for edge_id in self.kg.edges:
            try:
                relation_type = self.kg.properties.get_edge_property(edge_id, 'relation_type')
                if relation_type:
                    index[relation_type].append(edge_id)
            except:
                index['unknown'].append(edge_id)
        
        return dict(index)
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for query optimization"""
        
        recommendations = []
        
        # Analyze query patterns
        if self.query_history:
            # Find common query patterns
            common_patterns = self._analyze_query_patterns()
            
            for pattern, frequency in common_patterns.items():
                if frequency > 5:  # Threshold for recommendation
                    recommendations.append({
                        'type': 'index_recommendation',
                        'pattern': pattern,
                        'frequency': frequency,
                        'recommendation': f"Consider creating index for pattern: {pattern}"
                    })
        
        # Check for missing indexes
        if not self.indexes:
            recommendations.append({
                'type': 'general_recommendation',
                'recommendation': "Consider creating indexes on frequently queried fields (entity_type, relation_type)"
            })
        
        # Performance-based recommendations
        slow_queries = [q for q in self.query_history if q.get('optimization_time', 0) > 1.0]
        
        if slow_queries:
            recommendations.append({
                'type': 'performance_recommendation',
                'recommendation': f"Found {len(slow_queries)} slow queries. Consider query restructuring or additional indexes."
            })
        
        return recommendations
    
    def _analyze_query_patterns(self) -> Dict[str, int]:
        """Analyze common patterns in query history"""
        
        patterns = Counter()
        
        for query_record in self.query_history:
            query = query_record.get('query', '')
            
            # Extract simple patterns (this can be enhanced)
            if 'entity_type' in query.lower():
                patterns['entity_type_filter'] += 1
            
            if 'relation_type' in query.lower() or any(rel in query.lower() for rel in ['worksfor', 'knows', 'hasage']):
                patterns['relation_filter'] += 1
            
            if '?' in query and 'where' in query.lower():
                patterns['variable_binding'] += 1
        
        return dict(patterns)


class CostModel:
    """Cost model for query operations"""
    
    def __init__(self):
        self.base_costs = {
            'sequential_scan': 1.0,
            'index_scan': 0.1,
            'nested_loop_join': 1.0,
            'hash_join': 0.5,
            'merge_join': 0.7,
            'filter': 0.1,
            'projection': 0.05
        }
    
    def get_cost(self, operation: str, cardinality: int = 1000) -> float:
        """Get estimated cost for operation"""
        base_cost = self.base_costs.get(operation, 1.0)
        return base_cost * cardinality


class CardinalityEstimator:
    """Cardinality estimation for query planning"""
    
    def __init__(self):
        self.histograms = {}
        self.statistics = {}
    
    def estimate_cardinality(self, operation: Dict[str, Any], statistics: QueryStatistics) -> int:
        """Estimate output cardinality of an operation"""
        
        op_type = operation.get('type', '')
        
        if op_type == 'pattern_match':
            return self._estimate_pattern_cardinality(operation, statistics)
        elif op_type == 'join':
            return self._estimate_join_cardinality(operation, statistics)
        elif op_type == 'filter':
            return self._estimate_filter_cardinality(operation, statistics)
        else:
            return 1000  # Default estimate
    
    def _estimate_pattern_cardinality(self, operation: Dict[str, Any], statistics: QueryStatistics) -> int:
        """Estimate cardinality for pattern matching"""
        
        pattern = operation.get('pattern', {})
        
        # Base cardinality
        base_cardinality = statistics.total_edges
        
        # Apply selectivity based on pattern specificity
        selectivity = 1.0
        
        if pattern.get('subject') and not pattern.get('subject', '').startswith('?'):
            selectivity *= 0.1
        
        if pattern.get('predicate') and pattern.get('predicate') in statistics.edge_selectivity:
            selectivity *= statistics.edge_selectivity[pattern.get('predicate')]
        
        if pattern.get('object') and not pattern.get('object', '').startswith('?'):
            selectivity *= 0.1
        
        return max(int(base_cardinality * selectivity), 1)
    
    def _estimate_join_cardinality(self, operation: Dict[str, Any], statistics: QueryStatistics) -> int:
        """Estimate cardinality for join operations"""
        
        patterns = operation.get('patterns', [])
        
        if len(patterns) < 2:
            return statistics.total_edges
        
        # Simple join cardinality estimation
        left_selectivity = 0.1  # Assume 10% selectivity
        right_selectivity = 0.1
        
        # Cross product reduced by join selectivity
        estimated_cardinality = statistics.total_edges * left_selectivity * right_selectivity * 10
        
        return max(int(estimated_cardinality), 1)
    
    def _estimate_filter_cardinality(self, operation: Dict[str, Any], statistics: QueryStatistics) -> int:
        """Estimate cardinality after filtering"""
        
        # Assume filters reduce cardinality by 50%
        return max(statistics.total_edges // 2, 1)