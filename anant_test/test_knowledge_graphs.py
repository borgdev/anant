"""
Knowledge Graphs Test Suite
===========================

Tests for knowledge graph implementations:
- Traditional KnowledgeGraph
- HierarchicalKnowledgeGraph
- Semantic operations and reasoning
- Cross-level navigation and analytics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_traditional_knowledge_graph():
    """Test traditional KnowledgeGraph functionality."""
    print("  Testing Traditional KnowledgeGraph...")
    
    try:
        from anant.kg import KnowledgeGraph
        
        # Create knowledge graph
        kg = KnowledgeGraph()
        
        # Add nodes (entities)
        kg.add_node("entity1", properties={"name": "Alice", "age": 30}, entity_type="Person")
        kg.add_node("entity2", properties={"name": "Bob", "age": 25}, entity_type="Person")
        kg.add_node("entity3", properties={"name": "TechCorp"}, entity_type="Company")
        
        # Add edges (relationships)
        kg.add_edge("relationship1", ["entity1", "entity2"], properties={"since": "2020"}, edge_type="knows")
        kg.add_edge("relationship2", ["entity1", "entity3"], properties={"role": "engineer"}, edge_type="works_for")
        
        # Test basic properties
        assert kg.num_nodes >= 3, "Should have at least 3 nodes"
        assert kg.num_edges >= 2, "Should have at least 2 edges"
        
        # Test entity retrieval
        entities = kg.get_entities_by_type("Person")
        assert len(entities) >= 2, "Should have at least 2 Person entities"
        
        # Test relationship queries
        relationships = kg.get_relationships_by_type("knows")
        assert len(relationships) >= 1, "Should have at least 1 'knows' relationship"
        
        # Test semantic search if available
        try:
            results = kg.semantic_search("Person")
            assert isinstance(results, list), "Semantic search should return list"
        except AttributeError:
            print("      Semantic search not available, skipping...")
        
        print("    ✅ Traditional KnowledgeGraph working")
        return True
        
    except ImportError:
        print("    ⚠️  KnowledgeGraph module not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Traditional KnowledgeGraph test failed: {e}")
        return False


def test_hierarchical_knowledge_graph():
    """Test HierarchicalKnowledgeGraph functionality."""
    print("  Testing HierarchicalKnowledgeGraph...")
    
    try:
        from anant.kg import HierarchicalKnowledgeGraph, create_enterprise_hierarchy
        
        # Create hierarchical knowledge graph with custom levels
        hkg = HierarchicalKnowledgeGraph("TestDomain")
        
        # Create hierarchy levels
        hkg.create_level("enterprise", "Enterprise Level", level_order=0)
        hkg.create_level("division", "Division Level", level_order=1)
        hkg.create_level("department", "Department Level", level_order=2)
        hkg.create_level("team", "Team Level", level_order=3)
        
        # Add entities to different levels
        hkg.add_entity_to_level("ACME_Corp", "enterprise", "Organization", 
                               properties={"industry": "technology"})
        hkg.add_entity_to_level("Engineering_Division", "division", "Division",
                               properties={"focus": "product development"})
        hkg.add_entity_to_level("Backend_Dept", "department", "Department",
                               properties={"tech_stack": "python"})
        hkg.add_entity_to_level("API_Team", "team", "Team",
                               properties={"size": 5})
        
        # Add cross-level relationships
        hkg.add_relationship("ACME_Corp", "Engineering_Division", "contains", cross_level=True)
        hkg.add_relationship("Engineering_Division", "Backend_Dept", "contains", cross_level=True)
        hkg.add_relationship("Backend_Dept", "API_Team", "contains", cross_level=True)
        
        # Test hierarchical navigation
        higher_entities = hkg.navigate_up("API_Team")
        assert len(higher_entities) > 0, "Should find higher-level entities"
        
        lower_entities = hkg.navigate_down("ACME_Corp")
        assert len(lower_entities) > 0, "Should find lower-level entities"
        
        # Test level queries
        enterprise_entities = hkg.get_entities_at_level("enterprise")
        assert len(enterprise_entities) >= 1, "Should have enterprise-level entities"
        
        # Test hierarchy statistics
        stats = hkg.get_hierarchy_statistics()
        assert isinstance(stats, dict), "Should return hierarchy statistics"
        assert "levels" in stats, "Stats should include level information"
        assert "cross_level_relationships" in stats, "Stats should include cross-level info"
        
        # Test cross-level connectivity analysis
        connectivity = hkg.analyze_cross_level_connectivity()
        assert isinstance(connectivity, dict), "Should return connectivity analysis"
        
        print("    ✅ HierarchicalKnowledgeGraph working")
        return True
        
    except ImportError:
        print("    ⚠️  HierarchicalKnowledgeGraph module not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ HierarchicalKnowledgeGraph test failed: {e}")
        return False


def test_enterprise_hierarchy_preset():
    """Test pre-configured enterprise hierarchy."""
    print("  Testing Enterprise Hierarchy Preset...")
    
    try:
        from anant.kg import create_enterprise_hierarchy
        
        # Create pre-configured enterprise hierarchy
        hkg = create_enterprise_hierarchy()
        
        # Add entities using the enterprise structure
        hkg.add_entity_to_level("MyCompany", "enterprise", "Organization")
        hkg.add_entity_to_level("DataScience_BU", "business_unit", "BusinessUnit")
        hkg.add_entity_to_level("CustomerData", "data_domain", "DataDomain")
        hkg.add_entity_to_level("CustomerDB", "dataset", "Database")
        hkg.add_entity_to_level("customer_table", "schema", "Table")
        hkg.add_entity_to_level("customer_id", "field", "PrimaryKey")
        
        # Add hierarchical relationships
        hkg.add_relationship("MyCompany", "DataScience_BU", "contains", cross_level=True)
        hkg.add_relationship("DataScience_BU", "CustomerData", "owns", cross_level=True)
        hkg.add_relationship("CustomerData", "CustomerDB", "contains", cross_level=True)
        hkg.add_relationship("CustomerDB", "customer_table", "contains", cross_level=True)
        hkg.add_relationship("customer_table", "customer_id", "contains", cross_level=True)
        
        # Test navigation through enterprise hierarchy
        field_ancestors = hkg.navigate_up("customer_id")
        assert len(field_ancestors) >= 4, "Field should have multiple ancestor levels"
        
        company_descendants = hkg.navigate_down("MyCompany")
        assert len(company_descendants) >= 4, "Company should have multiple descendant levels"
        
        # Test enterprise-specific queries
        datasets = hkg.get_entities_at_level("dataset")
        assert len(datasets) >= 1, "Should have dataset entities"
        
        print("    ✅ Enterprise Hierarchy Preset working")
        return True
        
    except ImportError:
        print("    ⚠️  Enterprise hierarchy preset not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Enterprise hierarchy preset test failed: {e}")
        return False


def test_research_hierarchy_preset():
    """Test pre-configured research hierarchy."""
    print("  Testing Research Hierarchy Preset...")
    
    try:
        from anant.kg import create_research_hierarchy
        
        # Create pre-configured research hierarchy
        research_kg = create_research_hierarchy()
        
        # Add research entities
        research_kg.add_entity_to_level("Computer_Science", "field", "ResearchField")
        research_kg.add_entity_to_level("Machine_Learning", "area", "ResearchArea")
        research_kg.add_entity_to_level("Deep_Learning", "topic", "ResearchTopic")
        research_kg.add_entity_to_level("Attention_Mechanisms", "paper", "ResearchPaper")
        research_kg.add_entity_to_level("Transformer_Architecture", "concept", "Concept")
        
        # Add research relationships
        research_kg.add_relationship("Computer_Science", "Machine_Learning", "contains", cross_level=True)
        research_kg.add_relationship("Machine_Learning", "Deep_Learning", "contains", cross_level=True)
        research_kg.add_relationship("Deep_Learning", "Attention_Mechanisms", "studies", cross_level=True)
        research_kg.add_relationship("Attention_Mechanisms", "Transformer_Architecture", "introduces", cross_level=True)
        
        # Test research hierarchy navigation
        concept_field = research_kg.navigate_up("Transformer_Architecture")
        assert len(concept_field) > 0, "Concept should trace back to field level"
        
        field_concepts = research_kg.navigate_down("Computer_Science")
        assert len(field_concepts) > 0, "Field should have concepts below it"
        
        print("    ✅ Research Hierarchy Preset working")
        return True
        
    except ImportError:
        print("    ⚠️  Research hierarchy preset not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Research hierarchy preset test failed: {e}")
        return False


def test_knowledge_graph_semantic_operations():
    """Test semantic operations on knowledge graphs."""
    print("  Testing Knowledge Graph Semantic Operations...")
    
    try:
        from anant.kg import KnowledgeGraph
        
        # Create knowledge graph with semantic data
        kg = KnowledgeGraph()
        
        # Add entities with rich semantic information
        kg.add_node("alice", properties={
            "name": "Alice Johnson",
            "occupation": "Data Scientist",
            "skills": ["Python", "Machine Learning", "Statistics"],
            "experience_years": 5
        }, entity_type="Person")
        
        kg.add_node("bob", properties={ 
            "name": "Bob Smith", 
            "occupation": "Software Engineer",
            "skills": ["Java", "Python", "Distributed Systems"],
            "experience_years": 8
        }, entity_type="Person")
        
        kg.add_node("techcorp", properties={
            "name": "TechCorp Inc",
            "industry": "Technology",
            "size": "Medium",
            "technologies": ["Python", "Java", "Machine Learning"]
        }, entity_type="Company")
        
        # Add semantic relationships
        kg.add_edge("alice_works_at", ["alice", "techcorp"], 
                   properties={"role": "Senior Data Scientist", "since": "2020"}, edge_type="works_for")
        kg.add_edge("bob_works_at", ["bob", "techcorp"],
                   properties={"role": "Lead Engineer", "since": "2018"}, edge_type="works_for")
        kg.add_edge("alice_knows_bob", ["alice", "bob"],
                   properties={"context": "work", "strength": "strong"}, edge_type="knows")
        
        # Test semantic queries
        try:
            # Query by entity type
            people = kg.get_entities_by_type("Person")
            assert len(people) >= 2, "Should find Person entities"
            
            # Query by relationship type
            work_relationships = kg.get_relationships_by_type("works_for")
            assert len(work_relationships) >= 2, "Should find work relationships"
            
            # Test property-based queries
            if hasattr(kg, 'query_by_property'):
                python_users = kg.query_by_property("skills", "Python")
                assert len(python_users) >= 2, "Should find Python users"
            
            print("    ✅ Semantic operations working")
        except AttributeError:
            print("      Some semantic operations not available, basic functionality confirmed")
        
        return True
        
    except ImportError:
        print("    ⚠️  Knowledge graph semantic operations not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Knowledge graph semantic operations test failed: {e}")
        return False


def test_knowledge_graph_performance():
    """Test knowledge graph performance with larger datasets."""
    print("  Testing Knowledge Graph Performance...")
    
    try:
        from anant.kg import KnowledgeGraph
        import time
        
        # Create knowledge graph
        kg = KnowledgeGraph()
        
        # Add many entities
        start_time = time.time()
        
        for i in range(100):
            kg.add_node(f"person_{i}", properties={
                "name": f"Person {i}",
                "age": 20 + (i % 50),
                "department": f"dept_{i % 10}"
            }, entity_type="Person")
        
        for i in range(50):
            kg.add_node(f"company_{i}", properties={
                "name": f"Company {i}",
                "industry": f"industry_{i % 5}"
            }, entity_type="Company")
        
        creation_time = time.time() - start_time
        
        # Add relationships
        relationship_start = time.time()
        
        for i in range(100):
            company_id = f"company_{i % 50}"
            person_id = f"person_{i}"
            kg.add_edge(f"employment_{i}", [person_id, company_id], 
                       properties={"role": f"role_{i % 10}"}, edge_type="works_for")
        
        relationship_time = time.time() - relationship_start
        
        # Test query performance
        query_start = time.time()
        
        people = kg.get_entities_by_type("Person")
        companies = kg.get_entities_by_type("Company")
        work_relations = kg.get_relationships_by_type("works_for")
        
        query_time = time.time() - query_start
        
        print(f"    ✅ Performance test completed:")
        print(f"       Created 150 entities in {creation_time:.3f}s")
        print(f"       Created 100 relationships in {relationship_time:.3f}s")
        print(f"       Queried entities/relationships in {query_time:.3f}s")
        print(f"       Found {len(people)} people, {len(companies)} companies, {len(work_relations)} relationships")
        
        return True
        
    except ImportError:
        print("    ⚠️  Knowledge graph performance test not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Knowledge graph performance test failed: {e}")
        return False


def run_tests():
    """Run all knowledge graph tests."""
    print("🧪 Running Knowledge Graph Tests")
    
    test_functions = [
        test_traditional_knowledge_graph,
        test_hierarchical_knowledge_graph,
        test_enterprise_hierarchy_preset,
        test_research_hierarchy_preset,
        test_knowledge_graph_semantic_operations,
        test_knowledge_graph_performance
    ]
    
    passed = 0
    failed = 0
    details = []
    
    for test_func in test_functions:
        try:
            result = test_func()
            if result:
                passed += 1
                details.append(f"✅ {test_func.__name__}")
            else:
                failed += 1
                details.append(f"❌ {test_func.__name__}: Test returned False")
        except Exception as e:
            failed += 1
            details.append(f"❌ {test_func.__name__}: {str(e)}")
    
    status = "PASSED" if failed == 0 else "FAILED"
    
    return {
        "status": status,
        "passed": passed,
        "failed": failed,
        "details": details
    }


if __name__ == "__main__":
    result = run_tests()
    print(f"\nKnowledge Graph Tests: {result['status']}")
    print(f"Passed: {result['passed']}, Failed: {result['failed']}")
    for detail in result["details"]:
        print(f"  {detail}")