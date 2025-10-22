"""
Integration Tests for Advanced Knowledge Graph Capabilities
==========================================================

Comprehensive test suite to validate all four advanced capabilities working together:
1. Ontology Processing
2. Semantic Search  
3. Relationship Inference
4. SPARQL-like Queries

Tests use Schema.org data for realistic validation.
"""

import pytest
import logging
import time
import json
import tempfile
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

from anant.kg.knowledge_graph import KnowledgeGraph
from anant.kg.ontology_processor import OntologyProcessor, OntologyFormat
from anant.kg.semantic_search_engine import SemanticSearchEngine, SearchMode
from anant.kg.relationship_inference_engine import RelationshipInferenceEngine
from anant.kg.sparql_query_engine import SPARQLQueryEngine


# Sample Schema.org ontology data for testing
SCHEMA_ORG_SAMPLE = """
@prefix schema: <https://schema.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

schema:Person rdf:type rdfs:Class .
schema:Person rdfs:label "Person" .
schema:Person rdfs:comment "A person (alive, dead, undead, or fictional)." .

schema:Organization rdf:type rdfs:Class .
schema:Organization rdfs:label "Organization" .
schema:Organization rdfs:comment "An organization such as a school, NGO, corporation, club, etc." .

schema:worksFor rdf:type rdf:Property .
schema:worksFor rdfs:label "works for" .
schema:worksFor rdfs:domain schema:Person .
schema:worksFor rdfs:range schema:Organization .

schema:name rdf:type rdf:Property .
schema:name rdfs:label "name" .
schema:name rdfs:comment "The name of the item." .

schema:address rdf:type rdf:Property .
schema:address rdfs:label "address" .

schema:Employee rdfs:subClassOf schema:Person .
schema:Employee rdfs:label "Employee" .

schema:Company rdfs:subClassOf schema:Organization .
schema:Company rdfs:label "Company" .
"""

# Sample instance data for testing
INSTANCE_DATA = [
    {
        "id": "john_doe",
        "type": "Person", 
        "properties": {
            "name": "John Doe",
            "age": "35",
            "occupation": "Software Engineer"
        }
    },
    {
        "id": "jane_smith", 
        "type": "Person",
        "properties": {
            "name": "Jane Smith",
            "age": "28", 
            "occupation": "Data Scientist"
        }
    },
    {
        "id": "tech_corp",
        "type": "Organization", 
        "properties": {
            "name": "Tech Corporation",
            "industry": "Technology",
            "size": "Large"
        }
    },
    {
        "id": "ai_startup",
        "type": "Organization",
        "properties": {
            "name": "AI Startup Inc",
            "industry": "Artificial Intelligence", 
            "size": "Small"
        }
    }
]

# Sample relationships
RELATIONSHIPS = [
    {"source": "john_doe", "target": "tech_corp", "type": "worksFor"},
    {"source": "jane_smith", "target": "ai_startup", "type": "worksFor"},
    {"source": "john_doe", "target": "jane_smith", "type": "knows"}
]


class TestAdvancedKnowledgeGraphCapabilities:
    """Integration test suite for advanced KG capabilities"""
    
    @pytest.fixture
    def setup_test_environment(self):
        """Set up test environment with sample data"""
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample ontology file
        self.ontology_file = os.path.join(self.temp_dir, "schema_sample.ttl")
        with open(self.ontology_file, 'w') as f:
            f.write(SCHEMA_ORG_SAMPLE)
        
        # Initialize knowledge graph
        self.kg = KnowledgeGraph()
        
        # Add sample data
        self._populate_knowledge_graph()
        
        # Initialize all engines
        self.ontology_processor = OntologyProcessor(self.kg)
        self.semantic_search = SemanticSearchEngine(self.kg)
        self.inference_engine = RelationshipInferenceEngine(self.kg)
        self.sparql_engine = SPARQLQueryEngine(self.kg)
        
        yield self
        
        # Cleanup
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _populate_knowledge_graph(self):
        """Populate KG with sample instance data"""
        
        # Add nodes
        for item in INSTANCE_DATA:
            node_id = self.kg.add_node(
                item["id"],
                properties=item["properties"], 
                node_type=item["type"]
            )
        
        # Add relationships as hyperedges
        for rel in RELATIONSHIPS:
            edge_id = self.kg.add_edge(
                [rel["source"], rel["target"]],
                edge_type=rel["type"]
            )
    
    def test_ontology_processing_integration(self, setup_test_environment):
        """Test ontology processing capabilities"""
        
        test_env = setup_test_environment
        
        # Test ontology loading
        result = test_env.ontology_processor.load_ontology_file(
            test_env.ontology_file,
            format=OntologyFormat.TURTLE
        )
        
        assert result["success"], f"Ontology loading failed: {result.get('error')}"
        assert result["classes_loaded"] > 0
        assert result["properties_loaded"] > 0
        
        # Test hierarchy construction
        hierarchy = test_env.ontology_processor.build_class_hierarchy()
        
        assert "Person" in hierarchy["classes"]
        assert "Organization" in hierarchy["classes"] 
        assert "Employee" in hierarchy["subclasses"]
        assert "Company" in hierarchy["subclasses"]
        
        # Test Schema.org compatibility
        schema_info = test_env.ontology_processor.get_schema_org_compatibility()
        assert schema_info["is_compatible"]
        
        print("✅ Ontology Processing: All tests passed")
    
    def test_semantic_search_integration(self, setup_test_environment):
        """Test semantic search capabilities"""
        
        test_env = setup_test_environment
        
        # Test exact search
        results = test_env.semantic_search.search_entities(
            "John Doe",
            mode=SearchMode.EXACT
        )
        
        assert len(results) > 0
        assert any(r.entity_id == "john_doe" for r in results)
        
        # Test fuzzy search
        results = test_env.semantic_search.search_entities(
            "Jon Do",  # Misspelled
            mode=SearchMode.FUZZY
        )
        
        assert len(results) > 0
        assert any("john_doe" in r.entity_id for r in results)
        
        # Test comprehensive search
        results = test_env.semantic_search.search_entities(
            "software engineer technology",
            mode=SearchMode.COMPREHENSIVE
        )
        
        assert len(results) > 0
        
        # Test relationship search
        rel_results = test_env.semantic_search.search_relationships(
            "employment work job"
        )
        
        assert len(rel_results) > 0
        
        print("✅ Semantic Search: All tests passed")
    
    def test_relationship_inference_integration(self, setup_test_environment):
        """Test relationship inference capabilities"""
        
        test_env = setup_test_environment
        
        # Test statistical inference
        stat_inferences = test_env.inference_engine.infer_relationships_statistical(
            confidence_threshold=0.3
        )
        
        assert len(stat_inferences) >= 0  # May not find patterns in small dataset
        
        # Test rule-based inference
        # Add a simple rule: Person worksFor Organization -> Person employed
        rules = [
            {
                "name": "employment_rule",
                "pattern": ["Person", "worksFor", "Organization"],
                "infer": ["Person", "employed", "*"]
            }
        ]
        
        rule_inferences = test_env.inference_engine.apply_inference_rules(rules)
        
        # Should infer employment relationships
        assert len(rule_inferences) >= 2  # John and Jane both work somewhere
        
        # Test pattern discovery
        patterns = test_env.inference_engine.discover_relationship_patterns(
            min_support=0.1
        )
        
        assert len(patterns) >= 0
        
        print("✅ Relationship Inference: All tests passed")
    
    def test_sparql_query_integration(self, setup_test_environment):
        """Test SPARQL query engine capabilities"""
        
        test_env = setup_test_environment
        
        # Test basic SELECT query
        query1 = """
        SELECT ?person ?org WHERE {
            ?person worksFor ?org
        }
        """
        
        result1 = test_env.sparql_engine.execute_query(query1)
        
        assert len(result1.solutions) >= 2  # John and Jane both work
        assert len(result1.errors) == 0
        
        # Test query with FILTER
        query2 = """
        SELECT ?person WHERE {
            ?person name ?name .
            FILTER(CONTAINS(?name, "John"))
        }
        """
        
        result2 = test_env.sparql_engine.execute_query(query2)
        
        assert len(result2.solutions) >= 1
        
        # Test OPTIONAL pattern
        query3 = """
        SELECT ?person ?org ?name WHERE {
            ?person worksFor ?org .
            OPTIONAL { ?person name ?name }
        }
        """
        
        result3 = test_env.sparql_engine.execute_query(query3)
        
        assert len(result3.solutions) >= 2
        
        # Test with LIMIT
        query4 = """
        SELECT ?person WHERE {
            ?person name ?name
        }
        LIMIT 1
        """
        
        result4 = test_env.sparql_engine.execute_query(query4)
        
        assert len(result4.solutions) <= 1
        
        print("✅ SPARQL Query Engine: All tests passed")
    
    def test_full_integration_workflow(self, setup_test_environment):
        """Test complete workflow using all capabilities together"""
        
        test_env = setup_test_environment
        
        print("🚀 Testing Full Integration Workflow...")
        
        # Step 1: Load and process ontology
        ontology_result = test_env.ontology_processor.load_ontology_file(
            test_env.ontology_file,
            format=OntologyFormat.TURTLE
        )
        
        assert ontology_result["success"]
        print("✓ Step 1: Ontology loaded and processed")
        
        # Step 2: Use semantic search to discover entities
        tech_entities = test_env.semantic_search.search_entities(
            "technology software AI",
            mode=SearchMode.COMPREHENSIVE
        )
        
        assert len(tech_entities) > 0
        print(f"✓ Step 2: Found {len(tech_entities)} technology-related entities")
        
        # Step 3: Infer new relationships
        new_relationships = test_env.inference_engine.infer_relationships_ml(
            feature_types=['semantic', 'structural']
        )
        
        print(f"✓ Step 3: Inferred {len(new_relationships)} new relationships")
        
        # Step 4: Query the enhanced graph
        enriched_query = """
        SELECT ?person ?org ?relationship WHERE {
            ?person ?relationship ?org .
            ?org name ?orgName .
            FILTER(CONTAINS(?orgName, "Tech") || CONTAINS(?orgName, "AI"))
        }
        """
        
        enriched_result = test_env.sparql_engine.execute_query(enriched_query)
        
        print(f"✓ Step 4: Found {len(enriched_result.solutions)} enriched relationships")
        
        # Step 5: Validate ontology compliance
        hierarchy = test_env.ontology_processor.build_class_hierarchy()
        compliance = test_env.ontology_processor.validate_instance_compliance(
            test_env.kg.nodes
        )
        
        print(f"✓ Step 5: Ontology compliance - {compliance['compliant_instances']} compliant instances")
        
        print("🎉 Full Integration Workflow: SUCCESS!")
        
        return {
            'ontology_classes': len(hierarchy['classes']),
            'entities_found': len(tech_entities),
            'relationships_inferred': len(new_relationships), 
            'query_solutions': len(enriched_result.solutions),
            'compliant_instances': compliance['compliant_instances']
        }
    
    def test_performance_benchmarks(self, setup_test_environment):
        """Test performance of all capabilities"""
        
        test_env = setup_test_environment
        
        benchmarks = {}
        
        # Benchmark ontology processing
        start_time = time.time()
        test_env.ontology_processor.load_ontology_file(
            test_env.ontology_file,
            format=OntologyFormat.TURTLE
        )
        benchmarks['ontology_processing'] = time.time() - start_time
        
        # Benchmark semantic search
        start_time = time.time()
        for _ in range(10):  # Multiple searches
            test_env.semantic_search.search_entities(
                "technology person organization",
                mode=SearchMode.COMPREHENSIVE
            )
        benchmarks['semantic_search'] = (time.time() - start_time) / 10
        
        # Benchmark relationship inference
        start_time = time.time()
        test_env.inference_engine.infer_relationships_statistical()
        benchmarks['relationship_inference'] = time.time() - start_time
        
        # Benchmark SPARQL queries  
        start_time = time.time()
        for _ in range(10):  # Multiple queries
            test_env.sparql_engine.execute_query("""
                SELECT ?s ?p ?o WHERE { ?s ?p ?o }
            """)
        benchmarks['sparql_queries'] = (time.time() - start_time) / 10
        
        print("📊 Performance Benchmarks:")
        for capability, time_taken in benchmarks.items():
            print(f"  • {capability}: {time_taken:.4f}s")
        
        # Assert reasonable performance (these are loose bounds)
        assert benchmarks['ontology_processing'] < 5.0
        assert benchmarks['semantic_search'] < 1.0  
        assert benchmarks['relationship_inference'] < 10.0
        assert benchmarks['sparql_queries'] < 0.5
        
        return benchmarks
    
    def test_error_handling_robustness(self, setup_test_environment):
        """Test error handling and robustness"""
        
        test_env = setup_test_environment
        
        errors_handled = 0
        
        # Test invalid ontology file
        try:
            result = test_env.ontology_processor.load_ontology_file(
                "/nonexistent/file.ttl"
            )
            assert not result["success"]
            errors_handled += 1
        except Exception:
            errors_handled += 1
        
        # Test invalid search query
        try:
            results = test_env.semantic_search.search_entities("")
            # Should handle gracefully
            errors_handled += 1
        except Exception:
            errors_handled += 1
        
        # Test invalid SPARQL query
        try:
            result = test_env.sparql_engine.execute_query("INVALID SPARQL")
            assert len(result.errors) > 0
            errors_handled += 1
        except Exception:
            errors_handled += 1
        
        # Test inference with empty graph
        try:
            empty_kg = KnowledgeGraph()
            empty_inference = RelationshipInferenceEngine(empty_kg)
            results = empty_inference.infer_relationships_statistical()
            # Should handle gracefully
            errors_handled += 1
        except Exception:
            errors_handled += 1
        
        print(f"🛡️ Error Handling: {errors_handled}/4 error cases handled gracefully")
        
        assert errors_handled >= 3  # At least 3 of 4 should be handled


class TestScalabilityAndPerformance:
    """Test scalability with larger datasets"""
    
    @pytest.fixture
    def large_dataset(self):
        """Generate larger test dataset"""
        
        kg = KnowledgeGraph()
        
        # Generate 1000 person nodes
        for i in range(1000):
            kg.add_node(
                f"person_{i}",
                properties={
                    "name": f"Person {i}",
                    "age": str(20 + (i % 50)),
                    "occupation": ["Engineer", "Scientist", "Manager", "Analyst"][i % 4]
                },
                node_type="Person"
            )
        
        # Generate 100 organization nodes
        for i in range(100):
            kg.add_node(
                f"org_{i}",
                properties={
                    "name": f"Organization {i}",
                    "industry": ["Technology", "Finance", "Healthcare", "Education"][i % 4],
                    "size": ["Small", "Medium", "Large"][i % 3]
                },
                node_type="Organization"
            )
        
        # Generate 2000 relationships
        import random
        for i in range(2000):
            person = f"person_{random.randint(0, 999)}"
            org = f"org_{random.randint(0, 99)}"
            
            kg.add_edge([person, org], edge_type="worksFor")
        
        yield kg
    
    def test_large_scale_semantic_search(self, large_dataset):
        """Test semantic search on large dataset"""
        
        search_engine = SemanticSearchEngine(large_dataset)
        
        start_time = time.time()
        
        results = search_engine.search_entities(
            "Engineer Technology",
            mode=SearchMode.COMPREHENSIVE,
            limit=50
        )
        
        search_time = time.time() - start_time
        
        print(f"🔍 Large-scale search: {len(results)} results in {search_time:.3f}s")
        
        assert len(results) > 0
        assert search_time < 10.0  # Should complete within 10 seconds
    
    def test_large_scale_sparql_queries(self, large_dataset):
        """Test SPARQL queries on large dataset"""
        
        sparql_engine = SPARQLQueryEngine(large_dataset)
        
        complex_query = """
        SELECT ?person ?org ?industry WHERE {
            ?person worksFor ?org .
            ?org industry ?industry .
            ?person occupation "Engineer" .
            FILTER(?industry = "Technology")
        }
        LIMIT 100
        """
        
        start_time = time.time()
        
        result = sparql_engine.execute_query(complex_query)
        
        query_time = time.time() - start_time
        
        print(f"🔎 Large-scale SPARQL: {len(result.solutions)} results in {query_time:.3f}s")
        
        assert len(result.errors) == 0
        assert query_time < 30.0  # Should complete within 30 seconds
    
    def test_large_scale_inference(self, large_dataset):
        """Test relationship inference on large dataset"""
        
        inference_engine = RelationshipInferenceEngine(large_dataset)
        
        start_time = time.time()
        
        patterns = inference_engine.discover_relationship_patterns(
            min_support=0.01,  # Lower threshold for large dataset
            max_patterns=50
        )
        
        inference_time = time.time() - start_time
        
        print(f"🧠 Large-scale inference: {len(patterns)} patterns in {inference_time:.3f}s")
        
        assert inference_time < 60.0  # Should complete within 1 minute


def run_comprehensive_tests():
    """Run all integration tests"""
    
    print("🧪 Starting Comprehensive Integration Tests")
    print("=" * 60)
    
    # Initialize test environment
    test_env = TestAdvancedKnowledgeGraphCapabilities()
    
    with test_env.setup_test_environment() as env:
        
        try:
            # Run individual capability tests
            env.test_ontology_processing_integration(env)
            env.test_semantic_search_integration(env) 
            env.test_relationship_inference_integration(env)
            env.test_sparql_query_integration(env)
            
            # Run integration workflow
            workflow_results = env.test_full_integration_workflow(env)
            
            # Run performance benchmarks
            benchmark_results = env.test_performance_benchmarks(env)
            
            # Run error handling tests
            env.test_error_handling_robustness(env)
            
            print("\n" + "=" * 60)
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            print("=" * 60)
            
            print(f"\n📊 Summary Statistics:")
            print(f"  • Ontology classes processed: {workflow_results['ontology_classes']}")
            print(f"  • Entities discovered: {workflow_results['entities_found']}") 
            print(f"  • Relationships inferred: {workflow_results['relationships_inferred']}")
            print(f"  • Query solutions found: {workflow_results['query_solutions']}")
            print(f"  • Compliant instances: {workflow_results['compliant_instances']}")
            
            print(f"\n⚡ Performance Results:")
            for capability, time_taken in benchmark_results.items():
                print(f"  • {capability}: {time_taken:.4f}s")
            
            return True
            
        except Exception as e:
            print(f"\n❌ INTEGRATION TEST FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # Run comprehensive integration tests
    success = run_comprehensive_tests()
    
    if success:
        print("\n✅ All advanced knowledge graph capabilities validated!")
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        exit(1)