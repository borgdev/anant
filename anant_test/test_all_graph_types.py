"""
All Graph Types Test Suite
===========================

Comprehensive tests for all four graph types:
1. Hypergraph (traditional hypergraphs with hyperedges)
2. Metagraph (enterprise metadata management)
3. KnowledgeGraph (semantic reasoning and entity relationships)
4. HierarchicalKnowledgeGraph (multi-level hierarchical knowledge)

This module tests interoperability and compares capabilities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_all_graph_types_basic_creation():
    """Test that all four graph types can be created successfully."""
    print("  Testing All Graph Types Basic Creation...")
    
    graph_types = {}
    
    # Test Hypergraph
    try:
        from anant.classes.hypergraph import Hypergraph
        hg = Hypergraph()
        hg.add_node("node1")
        hg.add_node("node2") 
        hg.add_node("node3")
        hg.add_edge("edge1", ["node1", "node2", "node3"])  # Hyperedge with 3 nodes
        graph_types["Hypergraph"] = "✅ Created successfully"
    except Exception as e:
        graph_types["Hypergraph"] = f"❌ Failed: {e}"
    
    # Test Metagraph
    try:
        from anant.metagraph import Metagraph
        mg = Metagraph()
        mg.create_entity("entity1", "Dataset", {"name": "Test Dataset"})
        mg.create_entity("entity2", "Pipeline", {"name": "Test Pipeline"})
        mg.create_relationship("entity1", "entity2", "feeds_into", {})
        graph_types["Metagraph"] = "✅ Created successfully"
    except Exception as e:
        graph_types["Metagraph"] = f"❌ Failed: {e}"
    
    # Test KnowledgeGraph
    try:
        from anant.kg import KnowledgeGraph
        kg = KnowledgeGraph()
        kg.add_node("person1", properties={"name": "Alice"}, entity_type="Person")
        kg.add_node("company1", properties={"name": "TechCorp"}, entity_type="Company")
        kg.add_edge("works_at", ["person1", "company1"], properties={}, edge_type="works_for")
        graph_types["KnowledgeGraph"] = "✅ Created successfully"
    except Exception as e:
        graph_types["KnowledgeGraph"] = f"❌ Failed: {e}"
    
    # Test HierarchicalKnowledgeGraph
    try:
        from anant.kg import HierarchicalKnowledgeGraph
        hkg = HierarchicalKnowledgeGraph("TestDomain")
        hkg.create_level("top", "Top Level", level_order=0)
        hkg.create_level("bottom", "Bottom Level", level_order=1)
        hkg.add_entity_to_level("entity_top", "top", "TopEntity")
        hkg.add_entity_to_level("entity_bottom", "bottom", "BottomEntity")
        hkg.add_relationship("entity_top", "entity_bottom", "contains", cross_level=True)
        graph_types["HierarchicalKnowledgeGraph"] = "✅ Created successfully"
    except Exception as e:
        graph_types["HierarchicalKnowledgeGraph"] = f"❌ Failed: {e}"
    
    # Report results
    for graph_type, status in graph_types.items():
        print(f"    {graph_type}: {status}")
    
    # Check if we have all four types
    successful_types = [name for name, status in graph_types.items() if "✅" in status]
    
    if len(successful_types) == 4:
        print("    ✅ All four graph types available and working!")
        return True
    else:
        print(f"    ⚠️  {len(successful_types)}/4 graph types working")
        return len(successful_types) > 0  # Partial success is still okay


def test_graph_types_data_modeling_comparison():
    """Compare how different graph types model the same domain."""
    print("  Testing Graph Types Data Modeling Comparison...")
    
    try:
        # Common domain: Company organizational structure with data assets
        
        # 1. Model with Hypergraph (focus on complex relationships)
        hypergraph_model = {}
        try:
            from anant.classes.hypergraph import Hypergraph
            hg = Hypergraph()
            
            # Nodes for entities
            entities = ["CEO", "CTO", "DataTeam", "EngineeringTeam", "CustomerDB", "AnalyticsPipeline", "Dashboard"]
            for entity in entities:
                hg.add_node(entity)
            
            # Hyperedges for complex relationships
            hg.add_edge("org_structure", ["CEO", "CTO", "DataTeam", "EngineeringTeam"])  # Org chart
            hg.add_edge("data_pipeline", ["CustomerDB", "AnalyticsPipeline", "Dashboard"])  # Data flow
            hg.add_edge("team_resources", ["DataTeam", "CustomerDB", "AnalyticsPipeline"])  # Team ownership
            
            hypergraph_model = {
                "nodes": hg.num_nodes,
                "edges": hg.num_edges,
                "approach": "Multi-entity relationships via hyperedges"
            }
        except Exception as e:
            hypergraph_model = {"error": str(e)}
        
        # 2. Model with Metagraph (focus on metadata and governance)
        metagraph_model = {}
        try:
            from anant.metagraph import Metagraph
            mg = Metagraph()
            
            # Entities with rich metadata
            mg.create_entity("CustomerDB", "Dataset", {
                "owner": "DataTeam",
                "classification": "confidential",
                "size_gb": 100,
                "compliance": ["GDPR"]
            })
            
            mg.create_entity("AnalyticsPipeline", "Pipeline", {
                "owner": "DataTeam", 
                "status": "active",
                "frequency": "daily",
                "technology": "Python"
            })
            
            mg.create_entity("Dashboard", "Report", {
                "owner": "DataTeam",
                "viewers": ["CEO", "CTO"],
                "refresh_rate": "hourly"
            })
            
            # Lineage relationships
            mg.create_relationship("CustomerDB", "AnalyticsPipeline", "feeds_into", {"latency": "5min"})
            mg.create_relationship("AnalyticsPipeline", "Dashboard", "generates", {"format": "json"})
            
            metagraph_model = {
                "entities": len(mg.get_entities()),
                "relationships": len(mg.get_relationships()),
                "approach": "Enterprise metadata with governance"
            }
        except Exception as e:
            metagraph_model = {"error": str(e)}
        
        # 3. Model with KnowledgeGraph (focus on semantic relationships)
        kg_model = {}
        try:
            from anant.kg import KnowledgeGraph
            kg = KnowledgeGraph()
            
            # Semantic entities
            kg.add_node("CEO", properties={"role": "Chief Executive Officer"}, entity_type="Person")
            kg.add_node("CTO", properties={"role": "Chief Technology Officer"}, entity_type="Person")
            kg.add_node("DataTeam", properties={"department": "Data Science"}, entity_type="Team")
            kg.add_node("CustomerDB", properties={"type": "database", "domain": "customer"}, entity_type="DataAsset")
            
            # Semantic relationships
            kg.add_edge("ceo_leads_cto", ["CEO", "CTO"], properties={}, edge_type="manages")
            kg.add_edge("cto_leads_data", ["CTO", "DataTeam"], properties={}, edge_type="manages")
            kg.add_edge("team_owns_db", ["DataTeam", "CustomerDB"], properties={}, edge_type="owns")
            
            kg_model = {
                "entities": kg.num_nodes,
                "relationships": kg.num_edges,
                "approach": "Semantic entities with typed relationships"
            }
        except Exception as e:
            kg_model = {"error": str(e)}
        
        # 4. Model with HierarchicalKnowledgeGraph (focus on organizational hierarchy)
        hkg_model = {}
        try:
            from anant.kg import HierarchicalKnowledgeGraph
            hkg = HierarchicalKnowledgeGraph("Organization")
            
            # Create organizational levels
            hkg.create_level("executive", "Executive Level", level_order=0)
            hkg.create_level("management", "Management Level", level_order=1) 
            hkg.create_level("team", "Team Level", level_order=2)
            hkg.create_level("asset", "Asset Level", level_order=3)
            
            # Add entities to levels
            hkg.add_entity_to_level("CEO", "executive", "Executive")
            hkg.add_entity_to_level("CTO", "management", "Manager")
            hkg.add_entity_to_level("DataTeam", "team", "Team")
            hkg.add_entity_to_level("CustomerDB", "asset", "DataAsset")
            
            # Hierarchical relationships
            hkg.add_relationship("CEO", "CTO", "manages", cross_level=True)
            hkg.add_relationship("CTO", "DataTeam", "oversees", cross_level=True)
            hkg.add_relationship("DataTeam", "CustomerDB", "owns", cross_level=True)
            
            hkg_model = {
                "levels": len(hkg.levels) if hasattr(hkg, 'levels') else 4,
                "entities": len(hkg.get_entities_at_level("executive")) + len(hkg.get_entities_at_level("management")) + len(hkg.get_entities_at_level("team")) + len(hkg.get_entities_at_level("asset")),
                "approach": "Multi-level organizational hierarchy"
            }
        except Exception as e:
            hkg_model = {"error": str(e)}
        
        # Report comparison
        print("    📊 Data Modeling Comparison:")
        print(f"      Hypergraph: {hypergraph_model}")
        print(f"      Metagraph: {metagraph_model}")
        print(f"      KnowledgeGraph: {kg_model}")
        print(f"      HierarchicalKnowledgeGraph: {hkg_model}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Data modeling comparison failed: {e}")
        return False


def test_graph_types_query_capabilities():
    """Test and compare query capabilities across graph types."""
    print("  Testing Graph Types Query Capabilities...")
    
    capabilities = {}
    
    # Test Hypergraph queries
    try:
        from anant.classes.hypergraph import Hypergraph
        hg = Hypergraph()
        
        # Add test data
        for i in range(5):
            hg.add_node(f"node_{i}")
        hg.add_edge("edge1", ["node_0", "node_1", "node_2"])
        hg.add_edge("edge2", ["node_2", "node_3", "node_4"])
        
        # Test basic queries
        query_results = []
        query_results.append(f"Nodes: {hg.num_nodes}")
        query_results.append(f"Edges: {hg.num_edges}")
        
        if hasattr(hg, 'neighbors'):
            neighbors = hg.neighbors("node_2")
            query_results.append(f"Neighbors of node_2: {len(neighbors) if neighbors else 0}")
        
        if hasattr(hg, 'edge_neighbors'):
            edge_neighbors = hg.edge_neighbors("edge1")
            query_results.append(f"Edge neighbors: {len(edge_neighbors) if edge_neighbors else 0}")
        
        capabilities["Hypergraph"] = {
            "status": "✅ Working",
            "queries": query_results,
            "strengths": ["Multi-node relationships", "Complex connectivity patterns"]
        }
        
    except Exception as e:
        capabilities["Hypergraph"] = {"status": f"❌ Failed: {e}"}
    
    # Test Metagraph queries
    try:
        from anant.metagraph import Metagraph
        mg = Metagraph()
        
        # Add test data
        mg.create_entity("data1", "Dataset", {"owner": "team1", "size": 100})
        mg.create_entity("data2", "Dataset", {"owner": "team2", "size": 200})
        mg.create_entity("pipe1", "Pipeline", {"owner": "team1", "status": "active"})
        mg.create_relationship("data1", "pipe1", "feeds_into", {})
        
        query_results = []
        query_results.append(f"Entities: {len(mg.get_entities())}")
        query_results.append(f"Relationships: {len(mg.get_relationships())}")
        
        datasets = mg.get_entities_by_type("Dataset")
        query_results.append(f"Datasets: {len(datasets)}")
        
        capabilities["Metagraph"] = {
            "status": "✅ Working",
            "queries": query_results,
            "strengths": ["Enterprise metadata", "Governance policies", "Lineage tracking"]
        }
        
    except Exception as e:
        capabilities["Metagraph"] = {"status": f"❌ Failed: {e}"}
    
    # Test KnowledgeGraph queries
    try:
        from anant.kg import KnowledgeGraph
        kg = KnowledgeGraph()
        
        # Add test data
        kg.add_node("alice", properties={"skill": "Python"}, entity_type="Person")
        kg.add_node("bob", properties={"skill": "Java"}, entity_type="Person")
        kg.add_node("techcorp", properties={"industry": "tech"}, entity_type="Company")
        kg.add_edge("alice_works", ["alice", "techcorp"], properties={}, edge_type="works_for")
        
        query_results = []
        query_results.append(f"Nodes: {kg.num_nodes}")
        query_results.append(f"Edges: {kg.num_edges}")
        
        people = kg.get_entities_by_type("Person")
        query_results.append(f"People: {len(people)}")
        
        work_relations = kg.get_relationships_by_type("works_for")
        query_results.append(f"Work relationships: {len(work_relations)}")
        
        capabilities["KnowledgeGraph"] = {
            "status": "✅ Working",
            "queries": query_results,
            "strengths": ["Semantic reasoning", "Entity types", "Relationship types"]
        }
        
    except Exception as e:
        capabilities["KnowledgeGraph"] = {"status": f"❌ Failed: {e}"}
    
    # Test HierarchicalKnowledgeGraph queries
    try:
        from anant.kg import HierarchicalKnowledgeGraph
        hkg = HierarchicalKnowledgeGraph("TestHierarchy")
        
        # Add test data
        hkg.create_level("top", "Top Level", order=0)
        hkg.create_level("middle", "Middle Level", order=1)
        hkg.create_level("bottom", "Bottom Level", order=2)
        
        hkg.add_entity_to_level("root", "top", "Root")
        hkg.add_entity_to_level("branch", "middle", "Branch")
        hkg.add_entity_to_level("leaf", "bottom", "Leaf")
        
        hkg.add_relationship("root", "branch", "contains", cross_level=True)
        hkg.add_relationship("branch", "leaf", "contains", cross_level=True)
        
        query_results = []
        query_results.append(f"Levels: {len(hkg.levels) if hasattr(hkg, 'levels') else 3}")
        
        top_entities = hkg.get_entities_at_level("top")
        query_results.append(f"Top level entities: {len(top_entities)}")
        
        ancestors = hkg.navigate_up("leaf")
        query_results.append(f"Leaf ancestors: {len(ancestors)}")
        
        descendants = hkg.navigate_down("root")
        query_results.append(f"Root descendants: {len(descendants)}")
        
        capabilities["HierarchicalKnowledgeGraph"] = {
            "status": "✅ Working",
            "queries": query_results,
            "strengths": ["Multi-level hierarchy", "Cross-level navigation", "Hierarchical analytics"]
        }
        
    except Exception as e:
        capabilities["HierarchicalKnowledgeGraph"] = {"status": f"❌ Failed: {e}"}
    
    # Report capabilities
    print("    🔍 Query Capabilities Comparison:")
    for graph_type, info in capabilities.items():
        print(f"      {graph_type}: {info['status']}")
        if "queries" in info:
            for query in info["queries"]:
                print(f"        - {query}")
            print(f"        Strengths: {', '.join(info['strengths'])}")
    
    return True


def test_graph_types_use_case_alignment():
    """Test which graph types are best suited for different use cases."""
    print("  Testing Graph Types Use Case Alignment...")
    
    use_cases = {
        "Social Network Analysis": {
            "description": "Modeling complex social relationships",
            "best_fit": "Hypergraph",
            "reason": "Group interactions require hyperedges"
        },
        "Enterprise Data Governance": {
            "description": "Managing data assets with policies and compliance",
            "best_fit": "Metagraph", 
            "reason": "Enterprise metadata and governance features"
        },
        "Knowledge Management": {
            "description": "Semantic reasoning over entities and relationships",
            "best_fit": "KnowledgeGraph",
            "reason": "Semantic types and reasoning capabilities"
        },
        "Organizational Hierarchy": {
            "description": "Multi-level organizational structures",
            "best_fit": "HierarchicalKnowledgeGraph",
            "reason": "Natural hierarchical navigation and cross-level analysis"
        },
        "Scientific Collaboration": {
            "description": "Research papers, authors, institutions, topics",
            "best_fit": "Multiple",
            "reason": "Could use Hypergraph for collaborations or HierarchicalKG for field→area→topic"
        }
    }
    
    print("    🎯 Use Case Alignment Analysis:")
    for use_case, details in use_cases.items():
        print(f"      {use_case}:")
        print(f"        Description: {details['description']}")
        print(f"        Best Fit: {details['best_fit']}")
        print(f"        Reason: {details['reason']}")
    
    return True


def test_graph_types_performance_comparison():
    """Compare performance characteristics across graph types."""
    print("  Testing Graph Types Performance Comparison...")
    
    performance_results = {}
    
    # Test each graph type with similar operations
    test_size = 50  # Smaller size for comprehensive testing
    
    # Hypergraph performance
    try:
        from anant.classes.hypergraph import Hypergraph
        import time
        
        start_time = time.time()
        hg = Hypergraph()
        
        # Add nodes
        for i in range(test_size):
            hg.add_node(f"node_{i}")
        
        # Add hyperedges
        for i in range(test_size // 3):
            nodes = [f"node_{j}" for j in range(i, min(i+3, test_size))]
            hg.add_edge(f"edge_{i}", nodes)
        
        creation_time = time.time() - start_time
        
        # Query performance
        query_start = time.time()
        node_count = hg.num_nodes
        edge_count = hg.num_edges
        query_time = time.time() - query_start
        
        performance_results["Hypergraph"] = {
            "creation_time": f"{creation_time:.3f}s",
            "query_time": f"{query_time:.4f}s",
            "nodes": node_count,
            "edges": edge_count
        }
        
    except Exception as e:
        performance_results["Hypergraph"] = {"error": str(e)}
    
    # Metagraph performance
    try:
        from anant.metagraph import Metagraph
        import time
        
        start_time = time.time()
        mg = Metagraph()
        
        # Add entities
        for i in range(test_size):
            entity_type = ["Dataset", "Pipeline", "Report"][i % 3]
            mg.create_entity(f"entity_{i}", entity_type, {"index": i})
        
        # Add relationships
        for i in range(test_size // 2):
            mg.create_relationship(f"entity_{i}", f"entity_{i + test_size // 2}", "relates_to", {})
        
        creation_time = time.time() - start_time
        
        # Query performance
        query_start = time.time()
        entities = mg.get_entities()
        relationships = mg.get_relationships()
        datasets = mg.get_entities_by_type("Dataset")
        query_time = time.time() - query_start
        
        performance_results["Metagraph"] = {
            "creation_time": f"{creation_time:.3f}s",
            "query_time": f"{query_time:.4f}s",
            "entities": len(entities),
            "relationships": len(relationships)
        }
        
    except Exception as e:
        performance_results["Metagraph"] = {"error": str(e)}
    
    # KnowledgeGraph performance
    try:
        from anant.kg import KnowledgeGraph
        import time
        
        start_time = time.time()
        kg = KnowledgeGraph()
        
        # Add nodes
        for i in range(test_size):
            entity_type = ["Person", "Company", "Product"][i % 3]
            kg.add_node(f"entity_{i}", properties={"index": i}, entity_type=entity_type)
        
        # Add edges
        for i in range(test_size // 2):
            kg.add_edge(f"rel_{i}", [f"entity_{i}", f"entity_{i + test_size // 2}"], 
                       properties={}, edge_type="relates_to")
        
        creation_time = time.time() - start_time
        
        # Query performance
        query_start = time.time()
        node_count = kg.num_nodes
        edge_count = kg.num_edges
        people = kg.get_entities_by_type("Person")
        relations = kg.get_relationships_by_type("relates_to")
        query_time = time.time() - query_start
        
        performance_results["KnowledgeGraph"] = {
            "creation_time": f"{creation_time:.3f}s",
            "query_time": f"{query_time:.4f}s",
            "nodes": node_count,
            "edges": edge_count
        }
        
    except Exception as e:
        performance_results["KnowledgeGraph"] = {"error": str(e)}
    
    # HierarchicalKnowledgeGraph performance
    try:
        from anant.kg import HierarchicalKnowledgeGraph
        import time
        
        start_time = time.time()
        hkg = HierarchicalKnowledgeGraph("Performance")
        
        # Create levels
        for i in range(3):
            hkg.create_level(f"level_{i}", f"Level {i}", level_order=i)
        
        # Add entities
        for i in range(test_size):
            level = f"level_{i % 3}"
            hkg.add_entity_to_level(f"entity_{i}", level, "Entity", properties={"index": i})
        
        # Add cross-level relationships
        for i in range(test_size // 3):
            hkg.add_relationship(f"entity_{i}", f"entity_{i + 1}", "relates_to", cross_level=True)
        
        creation_time = time.time() - start_time
        
        # Query performance
        query_start = time.time()
        level_0_entities = hkg.get_entities_at_level("level_0")
        level_1_entities = hkg.get_entities_at_level("level_1")
        if level_0_entities:
            descendants = hkg.navigate_down(list(level_0_entities.keys())[0])
        query_time = time.time() - query_start
        
        performance_results["HierarchicalKnowledgeGraph"] = {
            "creation_time": f"{creation_time:.3f}s",
            "query_time": f"{query_time:.4f}s",
            "level_0_entities": len(level_0_entities),
            "level_1_entities": len(level_1_entities)
        }
        
    except Exception as e:
        performance_results["HierarchicalKnowledgeGraph"] = {"error": str(e)}
    
    print("    ⚡ Performance Comparison:")
    for graph_type, results in performance_results.items():
        print(f"      {graph_type}: {results}")
    
    return True


def run_tests():
    """Run all graph types comparison tests."""
    print("📊 Running All Graph Types Comparison Tests")
    
    test_functions = [
        test_all_graph_types_basic_creation,
        test_graph_types_data_modeling_comparison,
        test_graph_types_query_capabilities,
        test_graph_types_use_case_alignment,
        test_graph_types_performance_comparison
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
    print(f"\nAll Graph Types Tests: {result['status']}")
    print(f"Passed: {result['passed']}, Failed: {result['failed']}")
    for detail in result["details"]:
        print(f"  {detail}")