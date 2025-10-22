#!/usr/bin/env python3
"""
Test Script for Refactored Knowledge Graph
==========================================

Quick test to verify the modular Knowledge Graph implementation works correctly.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/amansingh/dev/ai/anant')

try:
    from anant.kg.core import KnowledgeGraph
    print("✓ Successfully imported refactored KnowledgeGraph")
except ImportError as e:
    print(f"✗ Failed to import KnowledgeGraph: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic knowledge graph operations"""
    print("\n=== Testing Basic Functionality ===")
    
    try:
        # Create knowledge graph
        kg = KnowledgeGraph(name="test_kg", enable_nlp=False, enable_reasoning=False)
        print(f"✓ Created KnowledgeGraph: {kg}")
        
        # Add nodes with semantic types
        kg.add_node("alice", {"name": "Alice Smith", "age": 30}, node_type="Person")
        kg.add_node("bob", {"name": "Bob Jones", "age": 25}, node_type="Person")  
        kg.add_node("acme", {"name": "Acme Corp", "founded": 2000}, node_type="Organization")
        print("✓ Added 3 nodes with semantic types")
        
        # Add relationships
        kg.add_edge(("alice", "acme"), {"role": "employee"}, edge_type="works_for")
        kg.add_edge(("bob", "acme"), {"role": "manager"}, edge_type="works_for") 
        kg.add_edge(("alice", "bob"), {"relationship": "colleague"}, edge_type="knows")
        print("✓ Added 3 relationships with semantic types")
        
        # Test statistics
        stats = kg.get_statistics()
        print(f"✓ Statistics: {stats['node_count']} nodes, {stats['edge_count']} edges")
        
        # Test semantic metadata
        alice_type = kg.get_node_type("alice")
        works_for_type = kg.get_edge_type(("alice", "acme"))
        print(f"✓ Semantic metadata: alice is {alice_type}, alice->acme is {works_for_type}")
        
        # Test summary
        print(f"✓ Summary:\n{kg.summary()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_operations_modules():
    """Test that operations modules are properly initialized"""
    print("\n=== Testing Operations Modules ===")
    
    try:
        kg = KnowledgeGraph(name="ops_test")
        
        # Check module availability
        modules = {
            'semantic': kg.semantic,
            'indexing': kg.indexing, 
            'query': kg.query,
            'ontology': kg.ontology,
            'nlp': kg.nlp,
            'reasoning': kg.reasoning
        }
        
        active_modules = []
        for name, module in modules.items():
            if module is not None:
                active_modules.append(name)
                print(f"✓ {name.capitalize()} operations module: Active")
            else:
                print(f"- {name.capitalize()} operations module: Disabled")
        
        print(f"✓ Total active modules: {len(active_modules)}/6")
        
        # Test basic operations
        kg.add_node("test", {"type": "test"})
        entity_type = kg.extract_entity_type("test")
        print(f"✓ Semantic operation works: extracted type for 'test'")
        
        # Test indexing
        indexes = kg.build_all_indexes()
        print(f"✓ Indexing operation works: built indexes")
        
        return True
        
    except Exception as e:
        print(f"✗ Operations modules test failed: {e}")
        return False

def test_ontology_operations():
    """Test ontology operations"""
    print("\n=== Testing Ontology Operations ===")
    
    try:
        kg = KnowledgeGraph(name="onto_test")
        
        # Define classes
        kg.define_class("foaf:Person")
        kg.define_class("foaf:Organization") 
        kg.define_class("Employee", parent_classes=["foaf:Person"])
        print("✓ Defined ontology classes")
        
        # Get class hierarchy
        hierarchy = kg.get_class_hierarchy()
        print(f"✓ Retrieved class hierarchy with {len(hierarchy.get('roots', []))} root classes")
        
        # Add namespace
        kg.add_namespace("test", "http://test.example.org/")
        print("✓ Added custom namespace")
        
        return True
        
    except Exception as e:
        print(f"✗ Ontology operations test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Knowledge Graph Refactoring Test Suite")
    print("=" * 40)
    
    tests = [
        test_basic_functionality,
        test_operations_modules, 
        test_ontology_operations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Refactoring successful!")
        return 0
    else:
        print("⚠️  Some tests failed. Review implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())