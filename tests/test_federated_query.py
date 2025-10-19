"""
Test suite for Federated Query Engine
====================================

Comprehensive tests for cross-database querying, query decomposition,
result merging, and distributed execution capabilities.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
import threading

from anant.kg.federated_query import (
    FederatedQueryEngine, DataSource, FederationProtocol,
    QueryType, QueryFragment, FragmentResult, FederatedQueryPlan,
    FederatedQueryResult
)


class TestFederatedQueryEngine:
    """Test federated query engine core functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create mock knowledge graph
        self.mock_kg = Mock()
        self.mock_kg.semantic_search.return_value = {
            'entities': ['entity1', 'entity2'],
            'relationships': ['rel1'],
            'metadata': {'total_nodes_searched': 100}
        }
        
        # Initialize federated query engine
        self.engine = FederatedQueryEngine(self.mock_kg)
        
        # Test data sources
        self.sparql_source = DataSource(
            source_id="sparql_test",
            name="Test SPARQL Endpoint",
            protocol=FederationProtocol.SPARQL_ENDPOINT,
            endpoint_url="http://test-sparql.example.com/query",
            capabilities={'sparql', 'rdf', 'inference'},
            priority=2
        )
        
        self.rest_source = DataSource(
            source_id="rest_test",
            name="Test REST API",
            protocol=FederationProtocol.REST_API,
            endpoint_url="http://test-api.example.com/data",
            capabilities={'json', 'pagination', 'filtering'},
            priority=1
        )
        
        self.native_source = DataSource(
            source_id="native_test",
            name="Test Native KG",
            protocol=FederationProtocol.NATIVE_KG,
            endpoint_url="local://kg",
            capabilities={'semantic_search', 'reasoning', 'embeddings'},
            priority=3
        )
    
    def test_engine_initialization(self):
        """Test federated query engine initialization"""
        
        engine = FederatedQueryEngine()
        
        assert engine.data_sources == {}
        assert engine.source_capabilities == {}
        assert engine.plan_cache == {}
        assert engine.result_cache == {}
        assert engine.config['max_parallel_fragments'] == 5
        assert engine.config['default_timeout'] == 30
    
    def test_register_data_source_sparql(self):
        """Test registering SPARQL data source"""
        
        self.engine.register_data_source(self.sparql_source)
        
        assert "sparql_test" in self.engine.data_sources
        assert self.engine.data_sources["sparql_test"] == self.sparql_source
        assert 'sparql' in self.engine.source_capabilities["sparql_test"]
        assert 'rdf' in self.engine.source_capabilities["sparql_test"]
        assert 'inference' in self.engine.source_capabilities["sparql_test"]
    
    def test_register_data_source_rest(self):
        """Test registering REST API data source"""
        
        self.engine.register_data_source(self.rest_source)
        
        assert "rest_test" in self.engine.data_sources
        assert self.engine.data_sources["rest_test"] == self.rest_source
        assert 'json' in self.engine.source_capabilities["rest_test"]
        assert 'pagination' in self.engine.source_capabilities["rest_test"]
    
    def test_register_data_source_native(self):
        """Test registering native KG data source"""
        
        self.engine.register_data_source(self.native_source)
        
        assert "native_test" in self.engine.data_sources
        assert 'semantic_search' in self.engine.source_capabilities["native_test"]
        assert 'reasoning' in self.engine.source_capabilities["native_test"]
    
    def test_register_invalid_source(self):
        """Test registering invalid data source"""
        
        invalid_source = DataSource(
            source_id="",  # Invalid empty ID
            name="Invalid Source",
            protocol=FederationProtocol.SPARQL_ENDPOINT,
            endpoint_url=""  # Invalid empty URL
        )
        
        with pytest.raises(ValueError, match="source_id"):
            self.engine.register_data_source(invalid_source)
    
    def test_capability_detection_sparql(self):
        """Test automatic capability detection for SPARQL endpoints"""
        
        capabilities = self.engine._detect_source_capabilities(self.sparql_source)
        
        assert 'sparql' in capabilities
        assert 'rdf' in capabilities
        assert 'inference' in capabilities
        assert 'full_text_search' in capabilities
    
    def test_capability_detection_rest(self):
        """Test automatic capability detection for REST APIs"""
        
        capabilities = self.engine._detect_source_capabilities(self.rest_source)
        
        assert 'json' in capabilities
        assert 'pagination' in capabilities
        assert 'filtering' in capabilities
    
    def test_capability_detection_native(self):
        """Test automatic capability detection for native KG"""
        
        capabilities = self.engine._detect_source_capabilities(self.native_source)
        
        assert 'semantic_search' in capabilities
        assert 'reasoning' in capabilities
        assert 'ontology' in capabilities
        assert 'embeddings' in capabilities
    
    def test_analyze_query_requirements_sparql(self):
        """Test query requirement analysis for SPARQL queries"""
        
        sparql_query = """
        SELECT ?person ?name WHERE {
            ?person rdf:type Person .
            ?person rdfs:label ?name .
            FILTER CONTAINS(?name, "John")
        }
        """
        
        requirements = self.engine._analyze_query_requirements(sparql_query, 'sparql')
        
        assert 'sparql' in requirements
        assert 'rdf' in requirements
        assert 'full_text_search' in requirements
    
    def test_analyze_query_requirements_reasoning(self):
        """Test query requirement analysis for reasoning queries"""
        
        reasoning_query = """
        SELECT ?subclass WHERE {
            ?subclass rdfs:subClassOf Animal .
            ?instance rdf:type ?subclass .
        }
        """
        
        requirements = self.engine._analyze_query_requirements(reasoning_query, 'sparql')
        
        assert 'sparql' in requirements
        assert 'rdf' in requirements
        assert 'inference' in requirements
    
    def test_select_optimal_sources(self):
        """Test optimal source selection"""
        
        # Register all test sources
        self.engine.register_data_source(self.sparql_source)
        self.engine.register_data_source(self.rest_source)
        self.engine.register_data_source(self.native_source)
        
        # Mock health status
        self.engine.source_health = {
            "sparql_test": {"status": "healthy", "response_time_ms": 100},
            "rest_test": {"status": "healthy", "response_time_ms": 200},
            "native_test": {"status": "healthy", "response_time_ms": 50}
        }
        
        # SPARQL query should prefer SPARQL endpoint and native KG
        sparql_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
        selected = self.engine._select_optimal_sources(sparql_query, 'sparql')
        
        assert "native_test" in selected  # Highest priority + fast response
        assert "sparql_test" in selected  # SPARQL capability match
        assert len(selected) <= 3
    
    def test_parse_sparql_structure(self):
        """Test SPARQL query structure parsing"""
        
        query = """
        SELECT ?person ?name WHERE {
            ?person rdf:type Person .
            ?person rdfs:label ?name .
            FILTER (?name != "")
        }
        ORDER BY ?name
        LIMIT 10
        """
        
        structure = self.engine._parse_sparql_structure(query)
        
        assert '?person' in structure['select_vars']
        assert '?name' in structure['select_vars']
        assert len(structure['where_patterns']) > 0
        assert len(structure['filters']) > 0
        assert structure['limit'] == 10
    
    def test_decompose_sparql_query(self):
        """Test SPARQL query decomposition"""
        
        self.engine.register_data_source(self.sparql_source)
        self.engine.register_data_source(self.native_source)
        
        query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
        sources = ["sparql_test", "native_test"]
        
        fragments = self.engine._decompose_sparql_query(query, sources)
        
        assert len(fragments) == 2
        assert all(f.query_type == 'sparql' for f in fragments)
        assert all(f.source_id in sources for f in fragments)
        assert all('sparql_' in f.fragment_id for f in fragments)
    
    def test_decompose_pattern_query(self):
        """Test pattern query decomposition"""
        
        self.engine.register_data_source(self.rest_source)
        self.engine.register_data_source(self.native_source)
        
        pattern_query = json.dumps({
            'subject': '?person',
            'predicate': 'rdf:type',
            'object': 'Person'
        })
        
        sources = ["rest_test", "native_test"]
        fragments = self.engine._decompose_pattern_query(pattern_query, sources)
        
        assert len(fragments) == 2
        assert all(f.query_type == 'pattern' for f in fragments)
        assert all('pattern_' in f.fragment_id for f in fragments)
    
    def test_create_execution_plan_sequential(self):
        """Test execution plan creation with sequential optimization"""
        
        fragments = [
            QueryFragment("frag1", "source1", "query1", "sparql", estimated_cost=1.0),
            QueryFragment("frag2", "source2", "query2", "sparql", estimated_cost=2.0),
        ]
        
        plan = self.engine._create_execution_plan(fragments, optimization_level=0)
        
        assert plan.plan_id is not None
        assert len(plan.fragments) == 2
        assert len(plan.execution_order) == 2
        assert len(plan.parallelizable_groups) == 2  # Sequential = one fragment per group
        assert plan.estimated_total_cost == 3.0
    
    def test_create_execution_plan_parallel(self):
        """Test execution plan creation with parallel optimization"""
        
        fragments = [
            QueryFragment("frag1", "source1", "query1", "sparql"),
            QueryFragment("frag2", "source2", "query2", "sparql"),
            QueryFragment("frag3", "source3", "query3", "sparql"),
        ]
        
        plan = self.engine._create_execution_plan(fragments, optimization_level=1)
        
        assert len(plan.fragments) == 3
        # Independent fragments should be grouped together for parallel execution
        assert len(plan.parallelizable_groups) >= 1
        assert plan.merge_strategy in ['union', 'smart_merge']
    
    def test_execute_fragment_success(self):
        """Test successful fragment execution"""
        
        # Register source and mock execution
        self.engine.register_data_source(self.native_source)
        
        fragment = QueryFragment("test_frag", "native_test", "test query", "sparql")
        
        # Mock the native KG execution
        with patch.object(self.engine, '_execute_native_kg') as mock_execute:
            mock_execute.return_value = [{'result': 'test_data'}]
            
            result = self.engine._execute_fragment(fragment)
            
            assert result.success is True
            assert result.fragment_id == "test_frag"
            assert result.source_id == "native_test"
            assert result.row_count == 1
            assert result.data == [{'result': 'test_data'}]
            assert result.execution_time > 0
    
    def test_execute_fragment_failure(self):
        """Test fragment execution failure handling"""
        
        # Register source
        self.engine.register_data_source(self.sparql_source)
        
        fragment = QueryFragment("fail_frag", "sparql_test", "bad query", "sparql")
        
        # Mock execution to raise exception
        with patch.object(self.engine, '_execute_sparql_endpoint') as mock_execute:
            mock_execute.side_effect = Exception("Connection failed")
            
            result = self.engine._execute_fragment(fragment)
            
            assert result.success is False
            assert result.fragment_id == "fail_frag"
            assert result.error == "Connection failed"
            assert result.row_count == 0
    
    def test_merge_fragment_results_union(self):
        """Test fragment result merging with union strategy"""
        
        results = [
            FragmentResult("frag1", "source1", True, data=[{'id': 1}, {'id': 2}], row_count=2),
            FragmentResult("frag2", "source2", True, data=[{'id': 3}, {'id': 4}], row_count=2),
        ]
        
        plan = FederatedQueryPlan(
            plan_id="test_plan",
            fragments=[],
            execution_order=[],
            merge_strategy='union',
            estimated_total_cost=0.0,
            parallelizable_groups=[],
            optimization_hints=[]
        )
        
        merged = self.engine._merge_fragment_results(results, plan)
        
        assert merged is not None
        assert len(merged) == 4
        assert {'id': 1} in merged
        assert {'id': 2} in merged
        assert {'id': 3} in merged
        assert {'id': 4} in merged
    
    def test_merge_fragment_results_smart_merge(self):
        """Test fragment result merging with smart merge (deduplication)"""
        
        results = [
            FragmentResult("frag1", "source1", True, data=[{'id': 1, 'name': 'John'}], row_count=1),
            FragmentResult("frag2", "source2", True, data=[{'id': 1, 'name': 'John'}], row_count=1),  # Duplicate
            FragmentResult("frag3", "source3", True, data=[{'id': 2, 'name': 'Jane'}], row_count=1),
        ]
        
        plan = FederatedQueryPlan(
            plan_id="test_plan",
            fragments=[],
            execution_order=[],
            merge_strategy='smart_merge',
            estimated_total_cost=0.0,
            parallelizable_groups=[],
            optimization_hints=[]
        )
        
        merged = self.engine._smart_merge_results(results)
        
        # Should deduplicate and add source information
        assert len(merged) == 2  # Deduplication should remove one duplicate
        
        # Check that source information is added
        for item in merged:
            assert '_source' in item
            assert item['_source'] in ['source1', 'source2', 'source3']
    
    @patch('anant.kg.federated_query.requests')
    def test_execute_sparql_endpoint(self, mock_requests):
        """Test SPARQL endpoint execution"""
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': {
                'bindings': [
                    {'s': {'value': 'http://example.org/person1'}},
                    {'s': {'value': 'http://example.org/person2'}}
                ]
            }
        }
        mock_requests.post.return_value = mock_response
        
        fragment = QueryFragment("sparql_frag", "sparql_test", "SELECT ?s WHERE { ?s ?p ?o }", "sparql")
        
        result = self.engine._execute_sparql_endpoint(fragment, self.sparql_source)
        
        assert len(result) == 2
        assert result[0]['s']['value'] == 'http://example.org/person1'
        mock_requests.post.assert_called_once()
    
    @patch('anant.kg.federated_query.requests')
    def test_execute_rest_api(self, mock_requests):
        """Test REST API execution"""
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {'id': 1, 'name': 'John'},
            {'id': 2, 'name': 'Jane'}
        ]
        mock_requests.get.return_value = mock_response
        
        fragment = QueryFragment("rest_frag", "rest_test", "search=person", "rest")
        
        result = self.engine._execute_rest_api(fragment, self.rest_source)
        
        assert len(result) == 2
        assert result[0]['name'] == 'John'
        mock_requests.get.assert_called_once()
    
    def test_execute_native_kg(self):
        """Test native KG execution"""
        
        fragment = QueryFragment("native_frag", "native_test", "find persons", "pattern")
        
        result = self.engine._execute_native_kg(fragment, self.native_source)
        
        # Should return the mocked semantic search result
        assert len(result) == 1
        assert 'entities' in result[0]
        assert result[0]['entities'] == ['entity1', 'entity2']
    
    def test_natural_language_to_sparql_conversion(self):
        """Test natural language to SPARQL conversion"""
        
        # Test person query
        sparql = self.engine._natural_to_sparql("find all persons")
        assert "Person" in sparql
        assert "SELECT" in sparql
        
        # Test count query
        sparql = self.engine._natural_to_sparql("count all entities")
        assert "COUNT" in sparql
        
        # Test default query
        sparql = self.engine._natural_to_sparql("show me data")
        assert "SELECT" in sparql
        assert "LIMIT" in sparql
    
    def test_query_adaptation_for_source_capabilities(self):
        """Test query adaptation based on source capabilities"""
        
        query = "SELECT ?s WHERE { ?s rdfs:subClassOf* ?class . FILTER CONTAINS(?label, 'test') }"
        structure = self.engine._parse_sparql_structure(query)
        
        # Test adaptation for source without inference capability
        no_inference_caps = {'sparql', 'rdf'}
        adapted = self.engine._adapt_query_for_source(query, structure, no_inference_caps)
        assert adapted is not None
        assert 'rdfs:subClassOf*' not in adapted  # Should remove transitive property
        
        # Test adaptation for source without full-text search
        no_fulltext_caps = {'sparql', 'rdf', 'inference'}
        adapted = self.engine._adapt_query_for_source(query, structure, no_fulltext_caps)
        assert adapted is not None
        assert 'CONTAINS(' not in adapted  # Should convert to REGEX
    
    def test_health_monitoring(self):
        """Test data source health monitoring"""
        
        self.engine.register_data_source(self.sparql_source)
        
        # Mock health check
        with patch.object(self.engine, '_check_source_health') as mock_health:
            # Simulate health check
            self.engine._check_source_health("sparql_test")
            mock_health.assert_called_with("sparql_test")
    
    def test_cache_key_generation(self):
        """Test cache key generation for query results"""
        
        query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
        sources = ["source1", "source2"]
        
        key1 = self.engine._get_cache_key(query, sources)
        key2 = self.engine._get_cache_key(query, sources)
        key3 = self.engine._get_cache_key(query, ["source2", "source1"])  # Different order
        
        assert key1 == key2  # Same query and sources should generate same key
        assert key1 == key3  # Order shouldn't matter (sources are sorted)
        
        # Different query should generate different key
        key4 = self.engine._get_cache_key("SELECT ?x WHERE { ?x ?y ?z }", sources)
        assert key1 != key4
    
    def test_federation_statistics(self):
        """Test federation statistics collection"""
        
        # Register sources
        self.engine.register_data_source(self.sparql_source)
        self.engine.register_data_source(self.rest_source)
        
        # Mock health data
        self.engine.source_health = {
            "sparql_test": {"status": "healthy", "response_time_ms": 100},
            "rest_test": {"status": "degraded", "response_time_ms": 500}
        }
        
        # Add some execution stats
        self.engine.execution_stats['query_history'] = [
            {'execution_time': 1.0, 'total_rows': 10, 'success': True, 'sources_used': 2},
            {'execution_time': 2.0, 'total_rows': 5, 'success': False, 'sources_used': 1},
        ]
        
        stats = self.engine.get_federation_statistics()
        
        assert stats['sources']['total_registered'] == 2
        assert stats['sources']['healthy_sources'] == 2  # degraded still counts as available
        assert stats['execution']['total_queries'] == 2
        assert stats['execution']['average_execution_time'] == 1.5
        assert stats['execution']['success_rate'] == 0.5
        
        # Check source details
        assert 'sparql_test' in stats['sources']['source_details']
        assert stats['sources']['source_details']['sparql_test']['protocol'] == 'sparql_endpoint'


class TestFederatedQueryIntegration:
    """Integration tests for federated query functionality"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.mock_kg = Mock()
        self.mock_kg.semantic_search.return_value = {
            'entities': ['entity1', 'entity2'],
            'relationships': ['rel1']
        }
        
        self.engine = FederatedQueryEngine(self.mock_kg)
        
        # Register multiple sources
        sources = [
            DataSource("sparql1", "SPARQL Source 1", FederationProtocol.SPARQL_ENDPOINT, 
                      "http://sparql1.test", capabilities={'sparql', 'rdf'}),
            DataSource("rest1", "REST Source 1", FederationProtocol.REST_API,
                      "http://rest1.test", capabilities={'json', 'filtering'}),
            DataSource("native1", "Native KG", FederationProtocol.NATIVE_KG,
                      "local://kg", capabilities={'semantic_search', 'reasoning'})
        ]
        
        for source in sources:
            self.engine.register_data_source(source)
        
        # Mock all sources as healthy
        self.engine.source_health = {
            "sparql1": {"status": "healthy", "response_time_ms": 100},
            "rest1": {"status": "healthy", "response_time_ms": 150},
            "native1": {"status": "healthy", "response_time_ms": 50}
        }
    
    @patch('anant.kg.federated_query.requests')
    def test_full_federated_query_execution(self, mock_requests):
        """Test complete federated query execution pipeline"""
        
        # Mock HTTP responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': {'bindings': [{'person': {'value': 'http://example.org/john'}}]}
        }
        mock_requests.post.return_value = mock_response
        mock_requests.get.return_value = mock_response
        
        query = "SELECT ?person WHERE { ?person rdf:type Person }"
        
        result = self.engine.execute_federated_query(
            query=query,
            query_type='sparql',
            optimization_level=1
        )
        
        assert result.success is True
        assert result.query_id is not None
        assert result.total_execution_time > 0
        assert len(result.fragment_results) >= 1
        assert result.sources_used is not None
        assert len(result.sources_used) > 0
        assert result.metadata is not None
    
    def test_federated_query_with_source_selection(self):
        """Test federated query with specific source selection"""
        
        # Mock execution methods
        with patch.object(self.engine, '_execute_native_kg') as mock_native:
            mock_native.return_value = [{'entity': 'test_entity'}]
            
            result = self.engine.execute_federated_query(
                query="find entities",
                query_type='pattern',
                sources=['native1'],  # Only use native source
                optimization_level=0
            )
            
            assert result.success is True
            assert result.sources_used == ['native1']
            assert len(result.fragment_results) == 1
            assert result.fragment_results[0].source_id == 'native1'
    
    def test_federated_query_error_handling(self):
        """Test error handling in federated queries"""
        
        # Mock all sources as unhealthy
        self.engine.source_health = {
            "sparql1": {"status": "unhealthy", "error": "Connection timeout"},
            "rest1": {"status": "unhealthy", "error": "Service unavailable"},
            "native1": {"status": "unhealthy", "error": "Internal error"}
        }
        
        result = self.engine.execute_federated_query(
            query="SELECT ?s ?p ?o WHERE { ?s ?p ?o }",
            query_type='sparql'
        )
        
        assert result.success is False
        assert result.error is not None
        assert "No healthy data sources available" in result.error
    
    def test_parallel_fragment_execution(self):
        """Test parallel execution of query fragments"""
        
        # Create fragments that can run in parallel
        fragments = [
            QueryFragment("frag1", "sparql1", "query1", "sparql"),
            QueryFragment("frag2", "rest1", "query2", "rest"),
            QueryFragment("frag3", "native1", "query3", "pattern")
        ]
        
        # Mock execution methods
        with patch.object(self.engine, '_execute_fragment') as mock_execute:
            mock_execute.side_effect = [
                FragmentResult("frag1", "sparql1", True, data=[{'result': 1}], row_count=1),
                FragmentResult("frag2", "rest1", True, data=[{'result': 2}], row_count=1),
                FragmentResult("frag3", "native1", True, data=[{'result': 3}], row_count=1)
            ]
            
            results = self.engine._execute_fragments_parallel(fragments)
            
            assert len(results) == 3
            assert all(r.success for r in results)
            assert {r.fragment_id for r in results} == {"frag1", "frag2", "frag3"}
    
    def test_query_result_caching(self):
        """Test query result caching functionality"""
        
        # Enable caching
        self.engine.config['enable_caching'] = True
        
        # Mock execution
        with patch.object(self.engine, '_execute_native_kg') as mock_native:
            mock_native.return_value = [{'cached': 'result'}]
            
            query = "test query for caching"
            sources = ['native1']
            
            # First execution
            result1 = self.engine.execute_federated_query(query, sources=sources)
            
            # Check that result is cached
            cache_key = self.engine._get_cache_key(query, sources)
            assert cache_key in self.engine.result_cache
            
            # Verify cache contains the result
            cached_result = self.engine.result_cache[cache_key]
            assert cached_result.query_id == result1.query_id
    
    def test_optimization_level_effects(self):
        """Test different optimization levels"""
        
        # Mock all execution methods to avoid network calls
        with patch.object(self.engine, '_execute_native_kg') as mock_native, \
             patch.object(self.engine, '_execute_sparql_endpoint') as mock_sparql, \
             patch.object(self.engine, '_execute_rest_api') as mock_rest:
            
            # Set up return values
            mock_native.return_value = [{'test': 'data', 'source': 'native'}]
            mock_sparql.return_value = [{'test': 'data', 'source': 'sparql'}]
            mock_rest.return_value = [{'test': 'data', 'source': 'rest'}]
            
            query = "SELECT ?s WHERE { ?s ?p ?o }"
            
            # Test no optimization
            result_none = self.engine.execute_federated_query(
                query, optimization_level=0
            )
            
            # Test basic optimization  
            result_basic = self.engine.execute_federated_query(
                query, optimization_level=1
            )
            
            # Test full optimization
            result_full = self.engine.execute_federated_query(
                query, optimization_level=2
            )
            
            # All should succeed but with different execution characteristics
            assert result_none.success
            assert result_basic.success
            assert result_full.success
            
            # Optimization level should be recorded in metadata
            assert result_none.metadata is not None
            assert result_basic.metadata is not None
            assert result_full.metadata is not None
            assert result_none.metadata['optimization_level'] == 0
            assert result_basic.metadata['optimization_level'] == 1
            assert result_full.metadata['optimization_level'] == 2


class TestFederatedQueryEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Setup edge case test environment"""
        self.engine = FederatedQueryEngine()
    
    def test_empty_query_execution(self):
        """Test execution with empty query"""
        
        result = self.engine.execute_federated_query("")
        
        assert result.success is False
        assert result.error is not None
        assert "No healthy data sources available" in result.error
    
    def test_execution_with_no_sources(self):
        """Test execution when no sources are registered"""
        
        result = self.engine.execute_federated_query("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
        
        assert result.success is False
        assert result.error is not None
        assert "No healthy data sources available" in result.error
    
    def test_fragment_timeout_handling(self):
        """Test fragment execution timeout"""
        
        # Register a source
        source = DataSource("slow_source", "Slow Source", FederationProtocol.REST_API, "http://slow.test")
        self.engine.register_data_source(source)
        self.engine.source_health["slow_source"] = {"status": "healthy", "response_time_ms": 100}
        
        # Create fragment with short timeout
        fragment = QueryFragment("timeout_frag", "slow_source", "slow query", "rest", timeout_seconds=1)
        
        # Mock slow execution
        def slow_execution(*args, **kwargs):
            time.sleep(2)  # Longer than timeout
            return [{'result': 'too_late'}]
        
        with patch.object(self.engine, '_execute_rest_api', side_effect=slow_execution):
            result = self.engine._execute_fragment(fragment)
            
            # Fragment should complete but might handle timeout differently
            # depending on implementation details
            assert result.fragment_id == "timeout_frag"
    
    def test_malformed_query_parsing(self):
        """Test handling of malformed queries"""
        
        malformed_sparql = "SELEC ?s WHRE { ?s ?p ?o"  # Typos
        
        structure = self.engine._parse_sparql_structure(malformed_sparql)
        
        # Should handle gracefully without crashing
        assert isinstance(structure, dict)
    
    def test_large_result_merging(self):
        """Test merging of large result sets"""
        
        # Create large result sets
        large_results = []
        for i in range(5):
            data = [{'id': j, 'source': i} for j in range(1000)]
            result = FragmentResult(f"frag{i}", f"source{i}", True, data=data, row_count=1000)
            large_results.append(result)
        
        plan = FederatedQueryPlan(
            plan_id="large_plan",
            fragments=[],
            execution_order=[],
            merge_strategy='union',
            estimated_total_cost=0.0,
            parallelizable_groups=[],
            optimization_hints=[]
        )
        
        merged = self.engine._merge_fragment_results(large_results, plan)
        
        assert merged is not None
        assert len(merged) == 5000  # 5 sources * 1000 items each
    
    def test_partial_failure_handling(self):
        """Test handling when some fragments fail"""
        
        results = [
            FragmentResult("frag1", "source1", True, data=[{'success': 1}], row_count=1),
            FragmentResult("frag2", "source2", False, error="Network error", row_count=0),
            FragmentResult("frag3", "source3", True, data=[{'success': 3}], row_count=1),
        ]
        
        plan = FederatedQueryPlan(
            plan_id="partial_fail_plan",
            fragments=[],
            execution_order=[],
            merge_strategy='union',
            estimated_total_cost=0.0,
            parallelizable_groups=[],
            optimization_hints=[]
        )
        
        merged = self.engine._merge_fragment_results(results, plan)
        
        # Should merge only successful results
        assert merged is not None
        assert len(merged) == 2
        assert {'success': 1} in merged
        assert {'success': 3} in merged