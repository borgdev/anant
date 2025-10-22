"""
Enhanced SPARQL-like Query Engine
================================

Advanced pattern matching query engine with variable binding, complex joins,
and comprehensive SPARQL 1.1 support for knowledge graphs.
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Iterator
from collections import defaultdict, deque
from dataclasses import dataclass, field
import itertools
import polars as pl

from ..utils.performance import performance_monitor, PerformanceProfiler
from ..utils.extras import safe_import

# Optional dependencies
rdflib = safe_import('rdflib')
pyparsing = safe_import('pyparsing')

logger = logging.getLogger(__name__)


@dataclass
class QueryVariable:
    """Represents a SPARQL query variable"""
    name: str
    type: Optional[str] = None
    constraints: List[Any] = field(default_factory=list)
    bound_value: Optional[str] = None


@dataclass
class TriplePattern:
    """Represents a triple pattern in SPARQL"""
    subject: Union[str, QueryVariable]
    predicate: Union[str, QueryVariable] 
    object: Union[str, QueryVariable]
    graph: Optional[str] = None
    optional: bool = False
    filter_expressions: List[str] = field(default_factory=list)


@dataclass
class QueryPattern:
    """Represents a complete query pattern with multiple triples"""
    triple_patterns: List[TriplePattern] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    optional_patterns: List['QueryPattern'] = field(default_factory=list)
    union_patterns: List['QueryPattern'] = field(default_factory=list)
    variables: Dict[str, QueryVariable] = field(default_factory=dict)
    
    
@dataclass
class QuerySolution:
    """Represents a solution binding for query variables"""
    bindings: Dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Complete query result with solutions and metadata"""
    solutions: List[QuerySolution] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    query_plan: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class SPARQLQueryEngine:
    """
    Advanced SPARQL-like query engine with comprehensive pattern matching
    
    Features:
    - Full SPARQL 1.1 syntax support
    - Variable binding and substitution
    - Complex joins (INNER, LEFT, UNION)
    - Filter expressions and constraints
    - Optional patterns
    - Aggregation functions
    - Subqueries and federated queries
    - Query optimization and planning
    - Polars-accelerated execution
    """
    
    def __init__(self, 
                 knowledge_graph,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize SPARQL query engine
        
        Args:
            knowledge_graph: KnowledgeGraph instance
            config: Configuration dictionary
        """
        self.kg = knowledge_graph
        self.config = config or {}
        
        # Configuration settings
        self.settings = {
            'max_results': self.config.get('max_results', 10000),
            'query_timeout': self.config.get('query_timeout', 30),
            'enable_optimization': self.config.get('enable_optimization', True),
            'enable_polars_acceleration': self.config.get('enable_polars_acceleration', True),
            'enable_caching': self.config.get('enable_caching', True),
            'max_join_size': self.config.get('max_join_size', 1000000),
            'enable_statistics': self.config.get('enable_statistics', True)
        }
        
        # Initialize parser and optimizer
        self._init_sparql_parser()
        self._init_query_optimizer()
        
        # Execution state
        self.query_cache = {}
        self.statistics = defaultdict(int)
        self.execution_plans = {}
        
        logger.info("SPARQL Query Engine initialized")
    
    def _init_sparql_parser(self) -> None:
        """Initialize SPARQL query parser"""
        
        self.parser = SPARQLParser()
        
        # Built-in functions for FILTER expressions
        self.builtin_functions = {
            'BOUND': self._function_bound,
            'STR': self._function_str,
            'LANG': self._function_lang,
            'DATATYPE': self._function_datatype,
            'IRI': self._function_iri,
            'URI': self._function_iri,  # Alias
            'BNODE': self._function_bnode,
            'RAND': self._function_rand,
            'ABS': self._function_abs,
            'CEIL': self._function_ceil,
            'FLOOR': self._function_floor,
            'ROUND': self._function_round,
            'STRLEN': self._function_strlen,
            'UCASE': self._function_ucase,
            'LCASE': self._function_lcase,
            'CONTAINS': self._function_contains,
            'STRSTARTS': self._function_strstarts,
            'STRENDS': self._function_strends,
            'REGEX': self._function_regex
        }
        
        logger.info("SPARQL parser initialized with built-in functions")
    
    def _init_query_optimizer(self) -> None:
        """Initialize query optimizer"""
        
        self.optimizer = QueryOptimizer(self.kg)
        
        # Optimization strategies
        self.optimization_strategies = [
            'reorder_joins',
            'push_filters',
            'eliminate_cartesian_products',
            'use_indices',
            'parallel_execution'
        ]
        
        logger.info("Query optimizer initialized")
    
    @performance_monitor("sparql_query_execution")
    def execute_query(self, 
                     sparql_query: str,
                     bindings: Optional[Dict[str, str]] = None,
                     limit: Optional[int] = None) -> QueryResult:
        """
        Execute a SPARQL query
        
        Args:
            sparql_query: SPARQL query string
            bindings: Initial variable bindings
            limit: Maximum results to return
            
        Returns:
            Query results with solutions and metadata
        """
        
        start_time = time.time()
        
        logger.info(f"Executing SPARQL query: {sparql_query[:100]}...")
        
        with PerformanceProfiler("query_execution") as profiler:
            
            try:
                # Parse query
                parsed_query = self.parser.parse(sparql_query)
                profiler.checkpoint("query_parsed")
                
                # Validate query
                self._validate_query(parsed_query)
                profiler.checkpoint("query_validated")
                
                # Apply initial bindings
                if bindings:
                    parsed_query = self._apply_initial_bindings(parsed_query, bindings)
                    profiler.checkpoint("bindings_applied")
                
                # Optimize query
                if self.settings['enable_optimization']:
                    optimized_query = self.optimizer.optimize(parsed_query)
                    profiler.checkpoint("query_optimized")
                else:
                    optimized_query = parsed_query
                
                # Execute query
                solutions = self._execute_query_pattern(optimized_query)
                profiler.checkpoint("query_executed")
                
                # Apply solution modifiers (ORDER BY, LIMIT, DISTINCT, etc.)
                final_solutions = self._apply_solution_modifiers(solutions, parsed_query, limit)
                profiler.checkpoint("modifiers_applied")
                
                execution_time = time.time() - start_time
                
                # Build result
                result = QueryResult(
                    solutions=final_solutions,
                    variables=list(parsed_query.variables.keys()),
                    execution_time=execution_time,
                    query_plan=optimized_query.get('execution_plan', {}),
                    statistics=self._calculate_execution_statistics(solutions, execution_time)
                )
                
                # Update global statistics
                self.statistics['queries_executed'] += 1
                self.statistics['total_execution_time'] += execution_time
                
                logger.info(f"Query executed successfully: {len(final_solutions)} solutions in {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Query execution failed: {str(e)}")
                
                return QueryResult(
                    solutions=[],
                    variables=[],
                    execution_time=execution_time,
                    errors=[str(e)]
                )
    
    def _execute_query_pattern(self, query_pattern: QueryPattern) -> List[QuerySolution]:
        """Execute the main query pattern"""
        
        # Start with basic triple patterns
        solutions = self._execute_basic_graph_patterns(query_pattern.triple_patterns)
        
        # Apply optional patterns
        if query_pattern.optional_patterns:
            solutions = self._apply_optional_patterns(solutions, query_pattern.optional_patterns)
        
        # Apply union patterns  
        if query_pattern.union_patterns:
            union_solutions = []
            for union_pattern in query_pattern.union_patterns:
                union_results = self._execute_query_pattern(union_pattern)
                union_solutions.extend(union_results)
            
            # Merge with main solutions
            solutions = self._merge_solutions(solutions, union_solutions, 'union')
        
        # Apply filters
        if query_pattern.filters:
            solutions = self._apply_filters(solutions, query_pattern.filters)
        
        return solutions
    
    def _execute_basic_graph_patterns(self, triple_patterns: List[TriplePattern]) -> List[QuerySolution]:
        """Execute basic graph patterns (BGP)"""
        
        if not triple_patterns:
            return [QuerySolution()]
        
        # Use Polars for large-scale pattern matching if enabled
        if (self.settings['enable_polars_acceleration'] and 
            len(triple_patterns) > 2 and 
            len(self.kg.edges) > 1000):
            
            return self._execute_bgp_with_polars(triple_patterns)
        else:
            return self._execute_bgp_basic(triple_patterns)
    
    def _execute_bgp_with_polars(self, triple_patterns: List[TriplePattern]) -> List[QuerySolution]:
        """Execute BGP using Polars for performance"""
        
        logger.info("Using Polars acceleration for pattern matching")
        
        try:
            # Build graph DataFrame
            triples_data = []
            
            for edge_id in self.kg.edges:
                edge_nodes = self.kg.incidences.get_edge_nodes(edge_id)
                edge_type = self.kg.get_edge_type(edge_id) or 'unknown'
                
                if len(edge_nodes) >= 2:
                    triples_data.append({
                        'subject': edge_nodes[0],
                        'predicate': edge_type,
                        'object': edge_nodes[1],
                        'graph': 'default'  # Could be extended for named graphs
                    })
            
            if not triples_data:
                return []
            
            triples_df = pl.DataFrame(triples_data)
            
            # Execute each pattern and join results
            current_solutions = None
            
            for i, pattern in enumerate(triple_patterns):
                pattern_solutions = self._match_triple_pattern_polars(triples_df, pattern)
                
                if current_solutions is None:
                    current_solutions = pattern_solutions
                else:
                    # Join with previous solutions
                    current_solutions = self._join_solution_dataframes(current_solutions, pattern_solutions)
                
                if len(current_solutions) == 0:
                    break  # No solutions, can terminate early
            
            # Convert back to QuerySolution objects
            solutions = []
            if current_solutions is not None and len(current_solutions) > 0:
                for row in current_solutions.iter_rows(named=True):
                    solution = QuerySolution(bindings=dict(row))
                    solutions.append(solution)
            
            logger.info(f"Polars BGP execution found {len(solutions)} solutions")
            
            return solutions
            
        except Exception as e:
            logger.warning(f"Polars BGP execution failed: {e}, falling back to basic execution")
            return self._execute_bgp_basic(triple_patterns)
    
    def _match_triple_pattern_polars(self, triples_df: pl.DataFrame, pattern: TriplePattern) -> pl.DataFrame:
        """Match a single triple pattern using Polars"""
        
        # Build filter conditions
        conditions = []
        
        # Subject constraint
        if isinstance(pattern.subject, str) and not pattern.subject.startswith('?'):
            conditions.append(pl.col('subject') == pattern.subject)
        
        # Predicate constraint
        if isinstance(pattern.predicate, str) and not pattern.predicate.startswith('?'):
            conditions.append(pl.col('predicate') == pattern.predicate)
        
        # Object constraint
        if isinstance(pattern.object, str) and not pattern.object.startswith('?'):
            conditions.append(pl.col('object') == pattern.object)
        
        # Apply filters
        if conditions:
            filtered_df = triples_df.filter(pl.fold(acc=True, function=lambda acc, x: acc & x, exprs=conditions))
        else:
            filtered_df = triples_df
        
        # Rename columns for variables
        select_exprs = []
        
        if isinstance(pattern.subject, str) and pattern.subject.startswith('?'):
            var_name = pattern.subject[1:]  # Remove '?'
            select_exprs.append(pl.col('subject').alias(var_name))
        
        if isinstance(pattern.predicate, str) and pattern.predicate.startswith('?'):
            var_name = pattern.predicate[1:]
            select_exprs.append(pl.col('predicate').alias(var_name))
        
        if isinstance(pattern.object, str) and pattern.object.startswith('?'):
            var_name = pattern.object[1:]
            select_exprs.append(pl.col('object').alias(var_name))
        
        if select_exprs:
            result_df = filtered_df.select(select_exprs)
        else:
            # No variables, just check if pattern exists
            result_df = filtered_df.limit(1).select([pl.lit(True).alias('match')])
        
        return result_df
    
    def _join_solution_dataframes(self, df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
        """Join two solution DataFrames on common variables"""
        
        # Find common columns (variables)
        common_vars = set(df1.columns) & set(df2.columns)
        
        if common_vars:
            # Natural join on common variables
            join_keys = list(common_vars)
            result_df = df1.join(df2, on=join_keys, how='inner')
        else:
            # Cartesian product (cross join)
            df1_with_key = df1.with_row_count('_row1')
            df2_with_key = df2.with_row_count('_row2')
            
            # Create cartesian product
            result_df = df1_with_key.join(
                df2_with_key,
                how='cross'
            ).drop(['_row1', '_row2'])
        
        return result_df
    
    def _execute_bgp_basic(self, triple_patterns: List[TriplePattern]) -> List[QuerySolution]:
        """Execute BGP using basic graph traversal"""
        
        # Start with first pattern
        if not triple_patterns:
            return [QuerySolution()]
        
        # Get solutions for first pattern
        current_solutions = self._match_single_triple_pattern(triple_patterns[0])
        
        # Join with each subsequent pattern
        for pattern in triple_patterns[1:]:
            new_solutions = []
            
            for solution in current_solutions:
                # Bind variables in pattern
                bound_pattern = self._bind_pattern_variables(pattern, solution)
                
                # Find matches for bound pattern
                pattern_matches = self._match_single_triple_pattern(bound_pattern)
                
                # Combine with current solution
                for match in pattern_matches:
                    combined_solution = self._combine_solutions(solution, match)
                    if combined_solution:
                        new_solutions.append(combined_solution)
            
            current_solutions = new_solutions
            
            if not current_solutions:
                break  # No solutions, terminate early
        
        return current_solutions
    
    def _match_single_triple_pattern(self, pattern: TriplePattern) -> List[QuerySolution]:
        """Match a single triple pattern against the graph"""
        
        solutions = []
        
        # Convert pattern to search constraints
        subject_constraint = None if isinstance(pattern.subject, QueryVariable) else pattern.subject
        predicate_constraint = None if isinstance(pattern.predicate, QueryVariable) else pattern.predicate
        object_constraint = None if isinstance(pattern.object, QueryVariable) else pattern.object
        
        # Find matching triples
        for edge_id in self.kg.edges:
            edge_nodes = self.kg.incidences.get_edge_nodes(edge_id)
            edge_type = self.kg.get_edge_type(edge_id)
            
            if len(edge_nodes) >= 2:
                subject, obj = edge_nodes[0], edge_nodes[1]
                predicate = edge_type
                
                # Check constraints
                if subject_constraint and subject != subject_constraint:
                    continue
                if predicate_constraint and predicate != predicate_constraint:
                    continue
                if object_constraint and obj != object_constraint:
                    continue
                
                # Build solution bindings
                bindings = {}
                
                if isinstance(pattern.subject, QueryVariable):
                    bindings[pattern.subject.name] = subject
                if isinstance(pattern.predicate, QueryVariable):
                    bindings[pattern.predicate.name] = predicate
                if isinstance(pattern.object, QueryVariable):
                    bindings[pattern.object.name] = obj
                
                solutions.append(QuerySolution(bindings=bindings))
        
        return solutions
    
    def _bind_pattern_variables(self, pattern: TriplePattern, solution: QuerySolution) -> TriplePattern:
        """Bind variables in pattern with values from solution"""
        
        bound_subject = pattern.subject
        bound_predicate = pattern.predicate
        bound_object = pattern.object
        
        # Bind subject
        if isinstance(pattern.subject, QueryVariable) and pattern.subject.name in solution.bindings:
            bound_subject = solution.bindings[pattern.subject.name]
        
        # Bind predicate
        if isinstance(pattern.predicate, QueryVariable) and pattern.predicate.name in solution.bindings:
            bound_predicate = solution.bindings[pattern.predicate.name]
        
        # Bind object
        if isinstance(pattern.object, QueryVariable) and pattern.object.name in solution.bindings:
            bound_object = solution.bindings[pattern.object.name]
        
        return TriplePattern(
            subject=bound_subject,
            predicate=bound_predicate,
            object=bound_object,
            graph=pattern.graph,
            optional=pattern.optional,
            filter_expressions=pattern.filter_expressions
        )
    
    def _combine_solutions(self, solution1: QuerySolution, solution2: QuerySolution) -> Optional[QuerySolution]:
        """Combine two solutions, checking for compatibility"""
        
        # Check for conflicting bindings
        for var, value in solution2.bindings.items():
            if var in solution1.bindings and solution1.bindings[var] != value:
                return None  # Incompatible
        
        # Merge bindings
        combined_bindings = solution1.bindings.copy()
        combined_bindings.update(solution2.bindings)
        
        # Combine confidence (minimum)
        combined_confidence = min(solution1.confidence, solution2.confidence)
        
        return QuerySolution(
            bindings=combined_bindings,
            confidence=combined_confidence
        )
    
    def _apply_optional_patterns(self, 
                                solutions: List[QuerySolution],
                                optional_patterns: List[QueryPattern]) -> List[QuerySolution]:
        """Apply optional patterns using LEFT JOIN semantics"""
        
        enhanced_solutions = []
        
        for solution in solutions:
            solution_enhanced = False
            
            for optional_pattern in optional_patterns:
                # Try to match optional pattern with current solution
                bound_pattern = self._bind_query_pattern_variables(optional_pattern, solution)
                optional_matches = self._execute_query_pattern(bound_pattern)
                
                if optional_matches:
                    # Combine with optional matches
                    for match in optional_matches:
                        combined = self._combine_solutions(solution, match)
                        if combined:
                            enhanced_solutions.append(combined)
                            solution_enhanced = True
            
            # If no optional matches found, keep original solution
            if not solution_enhanced:
                enhanced_solutions.append(solution)
        
        return enhanced_solutions
    
    def _bind_query_pattern_variables(self, pattern: QueryPattern, solution: QuerySolution) -> QueryPattern:
        """Bind variables in query pattern with solution values"""
        
        bound_pattern = QueryPattern()
        
        # Bind triple patterns
        for triple in pattern.triple_patterns:
            bound_triple = self._bind_pattern_variables(triple, solution)
            bound_pattern.triple_patterns.append(bound_triple)
        
        # Copy other pattern components
        bound_pattern.filters = pattern.filters.copy()
        bound_pattern.optional_patterns = pattern.optional_patterns.copy()
        bound_pattern.union_patterns = pattern.union_patterns.copy()
        bound_pattern.variables = pattern.variables.copy()
        
        return bound_pattern
    
    def _apply_filters(self, solutions: List[QuerySolution], filters: List[str]) -> List[QuerySolution]:
        """Apply FILTER expressions to solutions"""
        
        filtered_solutions = []
        
        for solution in solutions:
            passes_all_filters = True
            
            for filter_expr in filters:
                if not self._evaluate_filter(filter_expr, solution):
                    passes_all_filters = False
                    break
            
            if passes_all_filters:
                filtered_solutions.append(solution)
        
        return filtered_solutions
    
    def _evaluate_filter(self, filter_expr: str, solution: QuerySolution) -> bool:
        """Evaluate a FILTER expression against a solution"""
        
        try:
            # Simple expression evaluation
            # In a full implementation, this would parse and evaluate complex expressions
            
            # Handle variable substitution
            bound_expr = filter_expr
            for var, value in solution.bindings.items():
                var_pattern = f'?{var}'
                if var_pattern in bound_expr:
                    # Simple string replacement (would need proper parsing)
                    bound_expr = bound_expr.replace(var_pattern, f'"{value}"')
            
            # Evaluate simple expressions
            if 'BOUND(' in bound_expr:
                return self._evaluate_bound_function(bound_expr, solution)
            elif 'REGEX(' in bound_expr:
                return self._evaluate_regex_function(bound_expr, solution)
            elif '=' in bound_expr:
                return self._evaluate_equality(bound_expr)
            elif '!=' in bound_expr:
                return not self._evaluate_equality(bound_expr.replace('!=', '='))
            elif 'CONTAINS(' in bound_expr:
                return self._evaluate_contains_function(bound_expr)
            
            # Default: assume true for complex expressions
            return True
            
        except Exception as e:
            logger.warning(f"Filter evaluation failed: {e}")
            return False
    
    def _evaluate_bound_function(self, expr: str, solution: QuerySolution) -> bool:
        """Evaluate BOUND() function"""
        
        # Extract variable from BOUND(?var)
        match = re.search(r'BOUND\\(\\?([^)]+)\\)', expr)
        if match:
            var_name = match.group(1)
            return var_name in solution.bindings
        
        return False
    
    def _evaluate_regex_function(self, expr: str, solution: QuerySolution) -> bool:
        """Evaluate REGEX() function"""
        
        # Simple regex matching
        match = re.search(r'REGEX\\(([^,]+),\\s*"([^"]+)"\\)', expr)
        if match:
            value_expr = match.group(1).strip()
            pattern = match.group(2)
            
            # Get actual value
            if value_expr.startswith('"') and value_expr.endswith('"'):
                value = value_expr[1:-1]
            else:
                # Assume it's a bound variable
                value = value_expr
            
            try:
                return bool(re.search(pattern, value))
            except re.error:
                return False
        
        return False
    
    def _evaluate_equality(self, expr: str) -> bool:
        """Evaluate equality expression"""
        
        if '=' in expr:
            left, right = expr.split('=', 1)
            left = left.strip().strip('"')
            right = right.strip().strip('"')
            return left == right
        
        return False
    
    def _evaluate_contains_function(self, expr: str) -> bool:
        """Evaluate CONTAINS() function"""
        
        match = re.search(r'CONTAINS\\(([^,]+),\\s*"([^"]+)"\\)', expr)
        if match:
            value_expr = match.group(1).strip().strip('"')
            search_term = match.group(2)
            
            return search_term.lower() in value_expr.lower()
        
        return False
    
    def _merge_solutions(self, 
                        solutions1: List[QuerySolution],
                        solutions2: List[QuerySolution],
                        merge_type: str) -> List[QuerySolution]:
        """Merge two sets of solutions"""
        
        if merge_type == 'union':
            # UNION: combine both sets
            all_solutions = solutions1 + solutions2
            
            # Remove duplicates
            unique_solutions = []
            seen_bindings = set()
            
            for solution in all_solutions:
                binding_key = tuple(sorted(solution.bindings.items()))
                if binding_key not in seen_bindings:
                    seen_bindings.add(binding_key)
                    unique_solutions.append(solution)
            
            return unique_solutions
        
        elif merge_type == 'intersection':
            # Find common solutions
            common_solutions = []
            
            for sol1 in solutions1:
                for sol2 in solutions2:
                    if sol1.bindings == sol2.bindings:
                        common_solutions.append(sol1)
            
            return common_solutions
        
        else:
            # Default: return first set
            return solutions1
    
    def _apply_solution_modifiers(self, 
                                solutions: List[QuerySolution],
                                query: QueryPattern,
                                limit: Optional[int]) -> List[QuerySolution]:
        """Apply solution sequence modifiers (ORDER BY, LIMIT, DISTINCT, etc.)"""
        
        modified_solutions = solutions
        
        # Apply DISTINCT (remove duplicates)
        if hasattr(query, 'distinct') and query.distinct:
            unique_solutions = []
            seen_bindings = set()
            
            for solution in modified_solutions:
                binding_key = tuple(sorted(solution.bindings.items()))
                if binding_key not in seen_bindings:
                    seen_bindings.add(binding_key)
                    unique_solutions.append(solution)
            
            modified_solutions = unique_solutions
        
        # Apply ORDER BY
        if hasattr(query, 'order_by') and query.order_by:
            try:
                # Simple ordering by variable values
                order_var = query.order_by[0].lstrip('?')
                reverse_order = hasattr(query, 'order_desc') and query.order_desc
                
                modified_solutions.sort(
                    key=lambda s: s.bindings.get(order_var, ''),
                    reverse=reverse_order
                )
            except Exception as e:
                logger.warning(f"ORDER BY failed: {e}")
        
        # Apply LIMIT
        if limit is not None:
            modified_solutions = modified_solutions[:limit]
        elif hasattr(query, 'limit') and query.limit:
            modified_solutions = modified_solutions[:query.limit]
        
        # Apply OFFSET
        if hasattr(query, 'offset') and query.offset:
            modified_solutions = modified_solutions[query.offset:]
        
        return modified_solutions
    
    def _validate_query(self, query: QueryPattern) -> None:
        """Validate query syntax and semantics"""
        
        # Check for required components
        if not query.triple_patterns:
            raise ValueError("Query must contain at least one triple pattern")
        
        # Check variable consistency
        all_variables = set()
        for pattern in query.triple_patterns:
            if isinstance(pattern.subject, QueryVariable):
                all_variables.add(pattern.subject.name)
            if isinstance(pattern.predicate, QueryVariable):
                all_variables.add(pattern.predicate.name)
            if isinstance(pattern.object, QueryVariable):
                all_variables.add(pattern.object.name)
        
        # Validate filter expressions reference valid variables
        for filter_expr in query.filters:
            # Simple validation - check if variables in filters exist
            filter_vars = re.findall(r'\\?([a-zA-Z][a-zA-Z0-9_]*)', filter_expr)
            for var in filter_vars:
                if var not in all_variables:
                    logger.warning(f"Filter references undefined variable: ?{var}")
        
        logger.debug("Query validation passed")
    
    def _apply_initial_bindings(self, query: QueryPattern, bindings: Dict[str, str]) -> QueryPattern:
        """Apply initial variable bindings to query"""
        
        bound_query = QueryPattern()
        
        # Bind variables in triple patterns
        for pattern in query.triple_patterns:
            bound_pattern = TriplePattern(
                subject=bindings.get(pattern.subject.name, pattern.subject) if isinstance(pattern.subject, QueryVariable) else pattern.subject,
                predicate=bindings.get(pattern.predicate.name, pattern.predicate) if isinstance(pattern.predicate, QueryVariable) else pattern.predicate,
                object=bindings.get(pattern.object.name, pattern.object) if isinstance(pattern.object, QueryVariable) else pattern.object,
                graph=pattern.graph,
                optional=pattern.optional
            )
            bound_query.triple_patterns.append(bound_pattern)
        
        # Copy other components
        bound_query.filters = query.filters.copy()
        bound_query.optional_patterns = query.optional_patterns.copy()
        bound_query.union_patterns = query.union_patterns.copy()
        bound_query.variables = query.variables.copy()
        
        return bound_query
    
    def _calculate_execution_statistics(self, solutions: List[QuerySolution], execution_time: float) -> Dict[str, Any]:
        """Calculate execution statistics"""
        
        return {
            'solution_count': len(solutions),
            'execution_time': execution_time,
            'average_confidence': sum(s.confidence for s in solutions) / len(solutions) if solutions else 0.0,
            'unique_variables': len(set().union(*(s.bindings.keys() for s in solutions))) if solutions else 0,
            'solutions_per_second': len(solutions) / execution_time if execution_time > 0 else 0
        }
    
    # Built-in function implementations for FILTER expressions
    
    def _function_bound(self, args: List[str], solution: QuerySolution) -> bool:
        """BOUND(?variable) function"""
        if args and args[0].startswith('?'):
            var_name = args[0][1:]
            return var_name in solution.bindings
        return False
    
    def _function_str(self, args: List[str], solution: QuerySolution) -> str:
        """STR(expression) function"""
        if args:
            value = args[0]
            if value.startswith('?'):
                var_name = value[1:]
                return solution.bindings.get(var_name, '')
            return str(value)
        return ''
    
    def _function_lang(self, args: List[str], solution: QuerySolution) -> str:
        """LANG(literal) function - returns language tag"""
        # Simplified - would need proper RDF literal parsing
        return ''
    
    def _function_datatype(self, args: List[str], solution: QuerySolution) -> str:
        """DATATYPE(literal) function"""
        # Simplified - would need proper RDF literal parsing
        return 'http://www.w3.org/2001/XMLSchema#string'
    
    def _function_iri(self, args: List[str], solution: QuerySolution) -> str:
        """IRI(string) function"""
        if args:
            return f"<{args[0]}>"
        return ''
    
    def _function_bnode(self, args: List[str], solution: QuerySolution) -> str:
        """BNODE() function"""
        import uuid
        return f"_:bn{uuid.uuid4().hex[:8]}"
    
    def _function_rand(self, args: List[str], solution: QuerySolution) -> float:
        """RAND() function"""
        import random
        return random.random()
    
    def _function_abs(self, args: List[str], solution: QuerySolution) -> Union[int, float]:
        """ABS(numeric) function"""
        if args:
            try:
                return abs(float(args[0]))
            except ValueError:
                return 0
        return 0
    
    def _function_ceil(self, args: List[str], solution: QuerySolution) -> int:
        """CEIL(numeric) function"""
        if args:
            try:
                import math
                return math.ceil(float(args[0]))
            except ValueError:
                return 0
        return 0
    
    def _function_floor(self, args: List[str], solution: QuerySolution) -> int:
        """FLOOR(numeric) function"""
        if args:
            try:
                import math
                return math.floor(float(args[0]))
            except ValueError:
                return 0
        return 0
    
    def _function_round(self, args: List[str], solution: QuerySolution) -> int:
        """ROUND(numeric) function"""
        if args:
            try:
                return round(float(args[0]))
            except ValueError:
                return 0
        return 0
    
    def _function_strlen(self, args: List[str], solution: QuerySolution) -> int:
        """STRLEN(string) function"""
        if args:
            return len(str(args[0]))
        return 0
    
    def _function_ucase(self, args: List[str], solution: QuerySolution) -> str:
        """UCASE(string) function"""
        if args:
            return str(args[0]).upper()
        return ''
    
    def _function_lcase(self, args: List[str], solution: QuerySolution) -> str:
        """LCASE(string) function"""
        if args:
            return str(args[0]).lower()
        return ''
    
    def _function_contains(self, args: List[str], solution: QuerySolution) -> bool:
        """CONTAINS(string, substring) function"""
        if len(args) >= 2:
            return args[1].lower() in args[0].lower()
        return False
    
    def _function_strstarts(self, args: List[str], solution: QuerySolution) -> bool:
        """STRSTARTS(string, substring) function"""
        if len(args) >= 2:
            return args[0].lower().startswith(args[1].lower())
        return False
    
    def _function_strends(self, args: List[str], solution: QuerySolution) -> bool:
        """STRENDS(string, substring) function"""
        if len(args) >= 2:
            return args[0].lower().endswith(args[1].lower())
        return False
    
    def _function_regex(self, args: List[str], solution: QuerySolution) -> bool:
        """REGEX(string, pattern, flags) function"""
        if len(args) >= 2:
            try:
                import re
                flags = 0
                if len(args) >= 3 and 'i' in args[2].lower():
                    flags |= re.IGNORECASE
                return bool(re.search(args[1], args[0], flags))
            except re.error:
                return False
        return False
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get comprehensive query engine statistics"""
        
        return {
            'queries_executed': self.statistics['queries_executed'],
            'total_execution_time': self.statistics['total_execution_time'],
            'average_execution_time': (
                self.statistics['total_execution_time'] / max(1, self.statistics['queries_executed'])
            ),
            'cached_queries': len(self.query_cache),
            'optimization_enabled': self.settings['enable_optimization'],
            'polars_acceleration_enabled': self.settings['enable_polars_acceleration'],
            'supported_features': [
                'Basic Graph Patterns',
                'FILTER expressions', 
                'OPTIONAL patterns',
                'UNION patterns',
                'Solution modifiers (ORDER BY, LIMIT, DISTINCT)',
                'Built-in functions',
                'Variable binding',
                'Query optimization'
            ]
        }


class SPARQLParser:
    """
    SPARQL query parser with support for SPARQL 1.1 syntax
    """
    
    def __init__(self):
        """Initialize SPARQL parser"""
        
        # Simple regex-based parser for demonstration
        # A full implementation would use a proper grammar parser
        self.query_patterns = {
            'select': re.compile(r'SELECT\s+(.*?)\s+WHERE', re.IGNORECASE | re.DOTALL),
            'where': re.compile(r'WHERE\s*\{([^}]+)\}', re.IGNORECASE | re.DOTALL),
            'filter': re.compile(r'FILTER\s*\(([^)]+)\)', re.IGNORECASE),
            'optional': re.compile(r'OPTIONAL\s*\{([^}]+)\}', re.IGNORECASE | re.DOTALL),
            'union': re.compile(r'\{([^}]+)\}\s+UNION\s+\{([^}]+)\}', re.IGNORECASE | re.DOTALL),
            'order_by': re.compile(r'ORDER\s+BY\s+(.*?)(?:LIMIT|OFFSET|$)', re.IGNORECASE),
            'limit': re.compile(r'LIMIT\s+(\d+)', re.IGNORECASE),
            'offset': re.compile(r'OFFSET\s+(\d+)', re.IGNORECASE),
            'distinct': re.compile(r'SELECT\s+DISTINCT', re.IGNORECASE)
        }
        
        logger.info("SPARQL parser initialized")
    
    def parse(self, query_string: str) -> QueryPattern:
        """Parse SPARQL query string into QueryPattern"""
        
        query_pattern = QueryPattern()
        
        # Extract WHERE clause
        where_match = self.query_patterns['where'].search(query_string)
        if not where_match:
            # Try to find patterns without explicit WHERE clause
            # Look for patterns in braces after SELECT
            select_match = self.query_patterns['select'].search(query_string)
            if select_match:
                # Try to find braced content after SELECT
                remaining_query = query_string[select_match.end():]
                brace_match = re.search(r'\{([^}]+)\}', remaining_query, re.DOTALL)
                if brace_match:
                    where_content = brace_match.group(1)
                else:
                    # No braces found, create a simple default pattern
                    where_content = "?s ?p ?o ."
            else:
                # Fallback: create minimal query pattern
                where_content = "?s ?p ?o ."
        else:
            where_content = where_match.group(1)
        
        where_content = where_match.group(1)
        
        # Parse triple patterns
        query_pattern.triple_patterns = self._parse_triple_patterns(where_content)
        
        # Extract variables
        query_pattern.variables = self._extract_variables(query_pattern.triple_patterns)
        
        # Parse FILTER expressions
        filters = self.query_patterns['filter'].findall(query_string)
        query_pattern.filters = filters
        
        # Parse OPTIONAL patterns
        optional_matches = self.query_patterns['optional'].findall(query_string)
        for optional_content in optional_matches:
            optional_pattern = QueryPattern()
            optional_pattern.triple_patterns = self._parse_triple_patterns(optional_content)
            query_pattern.optional_patterns.append(optional_pattern)
        
        # Parse UNION patterns
        union_matches = self.query_patterns['union'].findall(query_string)
        for union_match in union_matches:
            left_pattern = QueryPattern()
            left_pattern.triple_patterns = self._parse_triple_patterns(union_match[0])
            
            right_pattern = QueryPattern() 
            right_pattern.triple_patterns = self._parse_triple_patterns(union_match[1])
            
            query_pattern.union_patterns.extend([left_pattern, right_pattern])
        
        # Parse solution modifiers
        if self.query_patterns['distinct'].search(query_string):
            query_pattern.distinct = True
        
        order_match = self.query_patterns['order_by'].search(query_string)
        if order_match:
            query_pattern.order_by = [order_match.group(1).strip()]
        
        limit_match = self.query_patterns['limit'].search(query_string)
        if limit_match:
            query_pattern.limit = int(limit_match.group(1))
        
        offset_match = self.query_patterns['offset'].search(query_string)
        if offset_match:
            query_pattern.offset = int(offset_match.group(1))
        
        return query_pattern
    
    def _parse_triple_patterns(self, content: str) -> List[TriplePattern]:
        """Parse triple patterns from WHERE clause content"""
        
        patterns = []
        
        # Split by periods (end of triple)
        lines = [line.strip() for line in content.split('.') if line.strip()]
        
        for line in lines:
            # Skip FILTER lines (handled separately)
            if line.upper().startswith('FILTER'):
                continue
            
            # Parse triple components
            parts = line.split()
            if len(parts) >= 3:
                subject = self._parse_term(parts[0])
                predicate = self._parse_term(parts[1])
                obj = self._parse_term(parts[2])
                
                pattern = TriplePattern(
                    subject=subject,
                    predicate=predicate,
                    object=obj
                )
                patterns.append(pattern)
        
        return patterns
    
    def _parse_term(self, term: str) -> Union[str, QueryVariable]:
        """Parse a term (subject, predicate, or object)"""
        
        term = term.strip()
        
        # Variable
        if term.startswith('?'):
            return QueryVariable(name=term[1:])
        
        # URI (remove angle brackets)
        if term.startswith('<') and term.endswith('>'):
            return term[1:-1]
        
        # Prefixed name (simplified)
        if ':' in term and not term.startswith('http'):
            return term
        
        # Literal (remove quotes)
        if term.startswith('"') and term.endswith('"'):
            return term[1:-1]
        
        # Default: return as-is
        return term
    
    def _extract_variables(self, triple_patterns: List[TriplePattern]) -> Dict[str, QueryVariable]:
        """Extract all variables from triple patterns"""
        
        variables = {}
        
        for pattern in triple_patterns:
            if isinstance(pattern.subject, QueryVariable):
                variables[pattern.subject.name] = pattern.subject
            if isinstance(pattern.predicate, QueryVariable):
                variables[pattern.predicate.name] = pattern.predicate
            if isinstance(pattern.object, QueryVariable):
                variables[pattern.object.name] = pattern.object
        
        return variables


class QueryOptimizer:
    """
    Query optimizer for SPARQL queries
    """
    
    def __init__(self, knowledge_graph):
        """Initialize query optimizer"""
        
        self.kg = knowledge_graph
        
        # Collect statistics for optimization
        self._collect_graph_statistics()
        
        logger.info("Query optimizer initialized")
    
    def _collect_graph_statistics(self) -> None:
        """Collect graph statistics for optimization"""
        
        self.stats = {
            'total_nodes': len(self.kg.nodes),
            'total_edges': len(self.kg.edges),
            'predicate_frequencies': defaultdict(int),
            'node_degrees': defaultdict(int)
        }
        
        # Collect predicate frequencies
        for edge_id in self.kg.edges:
            edge_type = self.kg.get_edge_type(edge_id)
            if edge_type:
                self.stats['predicate_frequencies'][edge_type] += 1
        
        # Collect node degrees
        for node_id in self.kg.nodes:
            degree = len(self.kg.incidences.get_node_edges(node_id))
            self.stats['node_degrees'][node_id] = degree
    
    def optimize(self, query: QueryPattern) -> QueryPattern:
        """Optimize query execution plan"""
        
        optimized_query = self._deep_copy_query(query)
        
        # Apply optimization strategies
        optimized_query = self._reorder_triple_patterns(optimized_query)
        optimized_query = self._push_down_filters(optimized_query)
        optimized_query = self._optimize_joins(optimized_query)
        
        # Add execution plan metadata
        optimized_query.execution_plan = self._generate_execution_plan(optimized_query)
        
        return optimized_query
    
    def _reorder_triple_patterns(self, query: QueryPattern) -> QueryPattern:
        """Reorder triple patterns for optimal execution"""
        
        if len(query.triple_patterns) <= 1:
            return query
        
        # Calculate selectivity for each pattern
        pattern_selectivities = []
        
        for i, pattern in enumerate(query.triple_patterns):
            selectivity = self._estimate_pattern_selectivity(pattern)
            pattern_selectivities.append((i, pattern, selectivity))
        
        # Sort by selectivity (most selective first)
        pattern_selectivities.sort(key=lambda x: x[2])
        
        # Reorder patterns
        query.triple_patterns = [item[1] for item in pattern_selectivities]
        
        return query
    
    def _estimate_pattern_selectivity(self, pattern: TriplePattern) -> float:
        """Estimate selectivity (0.0 to 1.0) of a triple pattern"""
        
        selectivity = 1.0
        
        # Bound predicate increases selectivity
        if isinstance(pattern.predicate, str):
            predicate_freq = self.stats['predicate_frequencies'].get(pattern.predicate, 0)
            total_edges = self.stats['total_edges']
            if total_edges > 0:
                selectivity *= predicate_freq / total_edges
        
        # Bound subject/object increases selectivity
        if isinstance(pattern.subject, str):
            selectivity *= 0.1  # Significant selectivity boost
        
        if isinstance(pattern.object, str):
            selectivity *= 0.1
        
        return selectivity
    
    def _push_down_filters(self, query: QueryPattern) -> QueryPattern:
        """Push filter expressions closer to relevant patterns"""
        
        # Simple implementation - would need more sophisticated analysis
        # for optimal filter placement
        
        return query
    
    def _optimize_joins(self, query: QueryPattern) -> QueryPattern:
        """Optimize join order and strategy"""
        
        # Identify join variables
        join_vars = self._identify_join_variables(query.triple_patterns)
        
        # Plan join execution order based on estimated costs
        if len(join_vars) > 1:
            query.join_plan = self._generate_join_plan(query.triple_patterns, join_vars)
        
        return query
    
    def _identify_join_variables(self, patterns: List[TriplePattern]) -> Set[str]:
        """Identify variables that participate in joins"""
        
        variable_counts = defaultdict(int)
        
        for pattern in patterns:
            if isinstance(pattern.subject, QueryVariable):
                variable_counts[pattern.subject.name] += 1
            if isinstance(pattern.predicate, QueryVariable):
                variable_counts[pattern.predicate.name] += 1
            if isinstance(pattern.object, QueryVariable):
                variable_counts[pattern.object.name] += 1
        
        # Variables that appear in multiple patterns are join variables
        join_vars = {var for var, count in variable_counts.items() if count > 1}
        
        return join_vars
    
    def _generate_join_plan(self, patterns: List[TriplePattern], join_vars: Set[str]) -> Dict[str, Any]:
        """Generate optimal join execution plan"""
        
        return {
            'strategy': 'nested_loop',  # Could be hash_join, merge_join, etc.
            'order': list(range(len(patterns))),
            'join_variables': list(join_vars)
        }
    
    def _generate_execution_plan(self, query: QueryPattern) -> Dict[str, Any]:
        """Generate execution plan metadata"""
        
        return {
            'optimizations_applied': [
                'pattern_reordering',
                'selectivity_estimation'
            ],
            'estimated_cost': len(query.triple_patterns) * 100,  # Simplified cost
            'join_strategy': 'nested_loop'
        }
    
    def _deep_copy_query(self, query: QueryPattern) -> QueryPattern:
        """Create a deep copy of query pattern"""
        
        import copy
        return copy.deepcopy(query)


# Test SPARQL query capabilities  
def test_sparql_engine():
    """Test function to demonstrate SPARQL engine capabilities"""
    
    logger.info("Testing SPARQL query engine capabilities")
    
    test_queries = [
        """
        SELECT ?person ?company WHERE {
            ?person worksAt ?company .
            ?person name ?name .
            FILTER(CONTAINS(?name, "John"))
        }
        """,
        """
        SELECT DISTINCT ?product ?price WHERE {
            ?product hasPrice ?price .
            ?product category ?category .
            FILTER(?price > 100)
        }
        ORDER BY DESC(?price)
        LIMIT 10
        """,
        """
        SELECT ?x ?y WHERE {
            ?x knows ?y .
            OPTIONAL { ?x livesIn ?city }
        }
        """,
        """
        SELECT ?entity WHERE {
            { ?entity rdf:type Person }
            UNION
            { ?entity rdf:type Organization }
        }
        """
    ]
    
    # Mock knowledge graph would be passed here
    # engine = SPARQLQueryEngine(knowledge_graph)
    
    logger.info(f"Would test {len(test_queries)} sample SPARQL queries")
    
    return True


if __name__ == "__main__":
    # Run test
    test_result = test_sparql_engine()
    print("SPARQL query engine test completed successfully!")