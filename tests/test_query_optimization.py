"""
Test Query Optimization Engine
==============================

Comprehensive tests for the query optimization system including:
- Cost-based optimization
- Adaptive indexing
- Query pattern learning
- Execution plan generation
"""

import unittest
import time
from unittest.mock import Mock, patch
import numpy as np

from anant.kg.query_optimization import (
    QueryOptimizer, OptimizationResult, ExecutionPlan, QueryStatistics,
    JoinType, IndexType, CostModel, CardinalityEstimator
)


class TestQueryOptimizer(unittest.TestCase):
    """Test the Query Optimization Engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        
        # Mock knowledge graph
        self.mock_kg = Mock()
        self.mock_kg.num_nodes = 1000
        self.mock_kg.num_edges = 5000
        self.mock_kg.nodes = [f'node_{i}' for i in range(1000)]
        self.mock_kg.edges = [f'edge_{i}' for i in range(5000)]
        
        # Mock properties
        self.mock_kg.properties = Mock()
        self.mock_kg.properties.get_node_property.return_value = 'Person'
        self.mock_kg.properties.get_edge_property.return_value = 'knows'
        
        # Mock methods
        self.mock_kg.get_node_degree.return_value = 5
        self.mock_kg.get_edge_nodes.return_value = ['node_1', 'node_2']
        
        # Create optimizer
        self.optimizer = QueryOptimizer(self.mock_kg)
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        
        self.assertEqual(self.optimizer.kg, self.mock_kg)
        self.assertIsNotNone(self.optimizer.cost_model)
        self.assertIsNotNone(self.optimizer.cardinality_estimator)
        self.assertEqual(self.optimizer.indexes, {})
        self.assertEqual(self.optimizer.query_cache, {})
        self.assertTrue(self.optimizer.config['enable_cost_based_optimization'])
    
    def test_sparql_query_parsing(self):
        """Test SPARQL query parsing"""
        
        query = """
        SELECT ?person ?name WHERE {
            ?person rdf:type Person .
            ?person foaf:name ?name .
        }
        ORDER BY ?name
        LIMIT 10
        """
        
        structure = self.optimizer._parse_query(query, 'sparql')
        
        self.assertEqual(structure['type'], 'sparql')
        self.assertEqual(len(structure['patterns']), 2)
        self.assertIn('?person', structure['projections'])
        self.assertIn('?name', structure['projections'])
        self.assertEqual(structure['limits'], 10)
    
    def test_pattern_query_parsing(self):
        """Test pattern-based query parsing"""
        
        query = '{"subject": "?person", "predicate": "knows", "object": "?friend"}'
        
        structure = self.optimizer._parse_query(query, 'pattern')
        
        self.assertEqual(structure['type'], 'pattern')
        self.assertIsInstance(structure['patterns'], list)
    
    def test_statistics_collection(self):
        """Test statistics collection"""
        
        statistics = self.optimizer._collect_statistics()
        
        self.assertIsInstance(statistics, QueryStatistics)
        self.assertEqual(statistics.total_nodes, 1000)
        self.assertEqual(statistics.total_edges, 5000)
        self.assertIsInstance(statistics.node_selectivity, dict)
        self.assertIsInstance(statistics.edge_selectivity, dict)
    
    def test_execution_plan_generation(self):
        """Test execution plan generation"""
        
        query_structure = {
            'type': 'sparql',
            'patterns': [
                {'subject': '?person', 'predicate': 'rdf:type', 'object': 'Person'},
                {'subject': '?person', 'predicate': 'knows', 'object': '?friend'}
            ],
            'filters': ['?person != ?friend'],
            'projections': ['?person', '?friend'],
            'ordering': [],
            'limits': None
        }
        
        plans = self.optimizer._generate_execution_plans(query_structure)
        
        self.assertGreater(len(plans), 0)
        self.assertIsInstance(plans[0], ExecutionPlan)
        
        # Check baseline plan
        baseline_plan = plans[0]
        self.assertEqual(baseline_plan.plan_id, 'baseline')
        self.assertGreater(len(baseline_plan.operations), 0)
    
    def test_cost_estimation(self):
        """Test cost estimation for execution plans"""
        
        plan = ExecutionPlan(
            plan_id="test_plan",
            operations=[
                {'type': 'pattern_match', 'method': 'sequential_scan'},
                {'type': 'join', 'method': JoinType.NESTED_LOOP.value, 'patterns': [{}, {}]},
                {'type': 'filter', 'method': 'post_filter'}
            ],
            estimated_cost=0.0,
            estimated_cardinality=0,
            optimization_hints=[],
            indexes_used=[]
        )
        
        # Update statistics first
        self.optimizer.statistics = self.optimizer._collect_statistics()
        
        cost = self.optimizer._estimate_plan_cost(plan)
        
        self.assertGreater(cost, 0.0)
        self.assertIsInstance(cost, float)
    
    def test_join_order_optimization(self):
        """Test join order optimization"""
        
        patterns = [
            {'subject': '?person', 'predicate': 'rdf:type', 'object': 'Person'},
            {'subject': '?person', 'predicate': 'hasAge', 'object': '25'},  # More selective
            {'subject': '?person', 'predicate': 'knows', 'object': '?friend'}
        ]
        
        # Update statistics
        self.optimizer.statistics = self.optimizer._collect_statistics()
        
        reordered = self.optimizer._optimize_join_order(patterns)
        
        self.assertEqual(len(reordered), 3)
        # The more selective pattern should come first
        self.assertEqual(reordered[0]['object'], '25')
    
    def test_adaptive_indexing(self):
        """Test adaptive index creation"""
        
        index_name = self.optimizer.create_adaptive_index('entity_type', IndexType.BTREE)
        
        self.assertIn(index_name, self.optimizer.indexes)
        
        index_data = self.optimizer.indexes[index_name]
        self.assertEqual(index_data['field'], 'entity_type')
        self.assertEqual(index_data['type'], 'btree')
        self.assertGreater(index_data['size'], 0)
    
    def test_query_optimization_full_flow(self):
        """Test complete query optimization flow"""
        
        query = """
        SELECT ?person ?friend WHERE {
            ?person rdf:type Person .
            ?person knows ?friend .
            ?friend rdf:type Person .
        }
        """
        
        result = self.optimizer.optimize_query(query, 'sparql')
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.original_query, query)
        self.assertIsNotNone(result.optimized_query)
        self.assertIsInstance(result.execution_plan, ExecutionPlan)
        self.assertGreater(result.optimization_time, 0.0)
        self.assertGreater(result.estimated_speedup, 0.0)
    
    def test_query_caching(self):
        """Test query result caching"""
        
        self.optimizer.config['enable_query_caching'] = True
        
        query = "SELECT ?x WHERE { ?x rdf:type Person }"
        
        # First optimization
        result1 = self.optimizer.optimize_query(query, 'sparql')
        
        # Check cache
        query_hash = self.optimizer._hash_query(query)
        self.assertIn(query_hash, self.optimizer.query_cache)
        
        # Cached result should be the same
        cached_result = self.optimizer.query_cache[query_hash]
        self.assertEqual(cached_result.original_query, result1.original_query)
    
    def test_predicate_pushdown(self):
        """Test predicate pushdown optimization"""
        
        query_structure = {
            'type': 'sparql',
            'patterns': [
                {'subject': '?person', 'predicate': 'rdf:type', 'object': 'Person', 'variables': ['?person']},
                {'subject': '?person', 'predicate': 'hasAge', 'object': '?age', 'variables': ['?person', '?age']}
            ],
            'filters': ['?age > 18'],
            'projections': ['?person'],
            'ordering': [],
            'limits': None
        }
        
        operations = self.optimizer._generate_pushdown_operations(query_structure)
        
        # Check that filter is pushed down to pattern level
        pattern_ops = [op for op in operations if op['type'] == 'pattern_match']
        self.assertTrue(any('filters' in op for op in pattern_ops))
    
    def test_index_usage_optimization(self):
        """Test optimization with available indexes"""
        
        # Create some indexes
        self.optimizer.create_adaptive_index('entity_type', IndexType.BTREE)
        self.optimizer.create_adaptive_index('relation_type', IndexType.HASH)
        
        query_structure = {
            'type': 'sparql',
            'patterns': [
                {'subject': '?person', 'predicate': 'rdf:type', 'object': 'Person'},
                {'subject': '?person', 'predicate': 'knows', 'object': '?friend'}
            ],
            'filters': [],
            'projections': ['?person'],
            'ordering': [],
            'limits': None
        }
        
        operations = self.optimizer._generate_index_operations(query_structure)
        
        # Check that index scans are used
        pattern_ops = [op for op in operations if op['type'] == 'pattern_match']
        self.assertTrue(any(op.get('method') == 'index_scan' for op in pattern_ops))
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations"""
        
        # Simulate query history
        self.optimizer.query_history = [
            {'query': 'SELECT * WHERE { ?x entity_type Person }'},
            {'query': 'SELECT * WHERE { ?x entity_type Person }'},
            {'query': 'SELECT * WHERE { ?x entity_type Person }'},
            {'query': 'SELECT * WHERE { ?x entity_type Company }'},
            {'query': 'SELECT * WHERE { ?x relation_type knows ?y }', 'optimization_time': 2.0},
        ] * 2  # Repeat to get higher frequency
        
        recommendations = self.optimizer.get_optimization_recommendations()
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Should recommend indexes for common patterns
        rec_types = [rec['type'] for rec in recommendations]
        self.assertIn('index_recommendation', rec_types)


class TestCostModel(unittest.TestCase):
    """Test the cost model"""
    
    def setUp(self):
        self.cost_model = CostModel()
    
    def test_cost_calculation(self):
        """Test cost calculation for different operations"""
        
        # Sequential scan should be expensive
        seq_cost = self.cost_model.get_cost('sequential_scan', 1000)
        self.assertEqual(seq_cost, 1000.0)
        
        # Index scan should be cheaper
        idx_cost = self.cost_model.get_cost('index_scan', 1000)
        self.assertEqual(idx_cost, 100.0)
        
        # Hash join should be cheaper than nested loop
        hash_cost = self.cost_model.get_cost('hash_join', 1000)
        nested_cost = self.cost_model.get_cost('nested_loop_join', 1000)
        self.assertLess(hash_cost, nested_cost)


class TestCardinalityEstimator(unittest.TestCase):
    """Test the cardinality estimator"""
    
    def setUp(self):
        self.estimator = CardinalityEstimator()
        
        # Mock statistics
        self.statistics = QueryStatistics(
            total_nodes=1000,
            total_edges=5000,
            node_selectivity={'Person': 0.3, 'Company': 0.1},
            edge_selectivity={'knows': 0.2, 'worksFor': 0.1},
            join_cardinality={'knows': 5, 'worksFor': 2},
            index_statistics={}
        )
    
    def test_pattern_cardinality_estimation(self):
        """Test cardinality estimation for pattern matching"""
        
        operation = {
            'type': 'pattern_match',
            'pattern': {
                'subject': '?person',
                'predicate': 'knows',
                'object': '?friend'
            }
        }
        
        cardinality = self.estimator.estimate_cardinality(operation, self.statistics)
        
        self.assertGreater(cardinality, 0)
        self.assertIsInstance(cardinality, int)
    
    def test_join_cardinality_estimation(self):
        """Test cardinality estimation for joins"""
        
        operation = {
            'type': 'join',
            'patterns': [
                {'subject': '?person', 'predicate': 'rdf:type', 'object': 'Person'},
                {'subject': '?person', 'predicate': 'knows', 'object': '?friend'}
            ]
        }
        
        cardinality = self.estimator.estimate_cardinality(operation, self.statistics)
        
        self.assertGreater(cardinality, 0)
        self.assertIsInstance(cardinality, int)
    
    def test_filter_cardinality_estimation(self):
        """Test cardinality estimation for filters"""
        
        operation = {
            'type': 'filter',
            'expression': '?age > 18'
        }
        
        cardinality = self.estimator.estimate_cardinality(operation, self.statistics)
        
        self.assertGreater(cardinality, 0)
        self.assertLess(cardinality, self.statistics.total_edges)


class TestQueryOptimizationIntegration(unittest.TestCase):
    """Integration tests for query optimization"""
    
    def setUp(self):
        """Set up integration test environment"""
        
        # More realistic mock knowledge graph
        self.mock_kg = Mock()
        self.mock_kg.num_nodes = 10000
        self.mock_kg.num_edges = 50000
        
        # Create realistic data distributions
        nodes = []
        for i in range(10000):
            if i < 3000:
                nodes.append(f'person_{i}')
            elif i < 4000:
                nodes.append(f'company_{i}')
            else:
                nodes.append(f'location_{i}')
        
        self.mock_kg.nodes = nodes
        self.mock_kg.edges = [f'edge_{i}' for i in range(50000)]
        
        # Mock properties with realistic distributions
        def mock_node_property(node, prop):
            if prop == 'entity_type':
                if 'person' in node:
                    return 'Person'
                elif 'company' in node:
                    return 'Company'
                else:
                    return 'Location'
            return None
        
        def mock_edge_property(edge, prop):
            if prop == 'relation_type':
                edge_num = int(edge.split('_')[1])
                if edge_num % 5 == 0:
                    return 'knows'
                elif edge_num % 5 == 1:
                    return 'worksFor'
                elif edge_num % 5 == 2:
                    return 'locatedIn'
                else:
                    return 'hasProperty'
            return None
        
        self.mock_kg.properties = Mock()
        self.mock_kg.properties.get_node_property.side_effect = mock_node_property
        self.mock_kg.properties.get_edge_property.side_effect = mock_edge_property
        
        self.mock_kg.get_node_degree.return_value = 10
        self.mock_kg.get_edge_nodes.return_value = ['node_1', 'node_2']
        
        self.optimizer = QueryOptimizer(self.mock_kg)
    
    def test_complex_query_optimization(self):
        """Test optimization of a complex query"""
        
        complex_query = """
        SELECT ?person ?company ?location WHERE {
            ?person rdf:type Person .
            ?person worksFor ?company .
            ?company rdf:type Company .
            ?company locatedIn ?location .
            ?location rdf:type Location .
        }
        ORDER BY ?person
        LIMIT 100
        """
        
        result = self.optimizer.optimize_query(complex_query, 'sparql')
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertGreater(result.estimated_speedup, 1.0)
        
        # Should have multiple operations in the plan
        self.assertGreaterEqual(len(result.execution_plan.operations), 3)
    
    def test_optimization_with_indexes(self):
        """Test optimization improvements with indexes"""
        
        query = """
        SELECT ?person WHERE {
            ?person rdf:type Person .
            ?person hasAge ?age .
        }
        """
        
        # Optimize without indexes
        result_no_index = self.optimizer.optimize_query(query, 'sparql')
        
        # Create relevant indexes
        self.optimizer.create_adaptive_index('entity_type', IndexType.BTREE)
        self.optimizer.create_adaptive_index('relation_type', IndexType.HASH)
        
        # Optimize with indexes
        result_with_index = self.optimizer.optimize_query(query, 'sparql')
        
        # Should have better estimated cost with indexes
        self.assertLessEqual(
            result_with_index.execution_plan.estimated_cost,
            result_no_index.execution_plan.estimated_cost
        )
    
    def test_adaptive_optimization_learning(self):
        """Test that optimizer learns from query patterns"""
        
        # Submit multiple similar queries
        queries = [
            "SELECT ?p WHERE { ?p rdf:type Person }",
            "SELECT ?person WHERE { ?person rdf:type Person . ?person hasAge 25 }",
            "SELECT * WHERE { ?x rdf:type Person . ?x hasName ?name }",
            "SELECT ?p ?age WHERE { ?p rdf:type Person . ?p hasAge ?age }"
        ]
        
        # Process queries multiple times
        for _ in range(3):
            for query in queries:
                self.optimizer.optimize_query(query, 'sparql')
        
        # Check that patterns were learned
        self.assertGreater(len(self.optimizer.query_history), 0)
        
        # Get recommendations
        recommendations = self.optimizer.get_optimization_recommendations()
        self.assertGreater(len(recommendations), 0)
        
        # Should recommend entity_type index
        rec_texts = [rec.get('recommendation', '') for rec in recommendations]
        self.assertTrue(any('entity_type' in text for text in rec_texts))
    
    def test_performance_monitoring_integration(self):
        """Test integration with performance monitoring"""
        
        query = "SELECT ?x ?y WHERE { ?x knows ?y }"
        
        # The optimization should be monitored and timing captured
        result = self.optimizer.optimize_query(query, 'sparql')
        
        # Check that timing information is captured
        self.assertGreater(result.optimization_time, 0.0)
        self.assertIsInstance(result, OptimizationResult)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)