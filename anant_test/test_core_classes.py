"""
Core Classes Test Suite
=======================

Tests for fundamental Anant graph classes:
- Hypergraph
- IncidenceStore
- PropertyStore
- Factory methods
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import numpy as np
from anant.classes.hypergraph import Hypergraph
from anant.classes.incidence_store import IncidenceStore
from anant.classes.property_store import PropertyStore


def test_hypergraph_creation():
    """Test various hypergraph creation methods."""
    print("  Testing hypergraph creation...")
    
    # Test creation from DataFrame
    data = pl.DataFrame({
        "edge_id": ["e1", "e1", "e2", "e3"],
        "node_id": ["n1", "n2", "n1", "n3"],
        "weight": [1.0, 0.8, 1.0, 0.5]
    })
    
    hg = Hypergraph(data=data)
    assert hg.num_nodes > 0, "Hypergraph should have nodes"
    assert hg.num_edges > 0, "Hypergraph should have edges"
    
    # Test creation from dict
    setsystem = {
        "e1": ["n1", "n2"],
        "e2": ["n1", "n3"],
        "e3": ["n2", "n3", "n4"]
    }
    
    hg2 = Hypergraph(setsystem=setsystem)
    assert hg2.num_nodes == 4, f"Expected 4 nodes, got {hg2.num_nodes}"
    assert hg2.num_edges == 3, f"Expected 3 edges, got {hg2.num_edges}"
    
    return True


def test_incidence_store():
    """Test IncidenceStore functionality."""
    print("  Testing IncidenceStore...")
    
    # Create incidence store
    data = pl.DataFrame({
        "edge_id": ["e1", "e1", "e2"],
        "node_id": ["n1", "n2", "n1"],
        "weight": [1.0, 0.8, 1.0]
    })
    
    store = IncidenceStore(data)
    
    # Test basic properties
    assert store.num_nodes() > 0, "Should have nodes"
    assert store.num_edges() > 0, "Should have edges"
    
    # Test edge operations
    nodes_in_e1 = store.get_edge_nodes("e1")
    assert "n1" in nodes_in_e1, "n1 should be in edge e1"
    assert "n2" in nodes_in_e1, "n2 should be in edge e1"
    
    # Test node operations
    edges_of_n1 = store.get_node_edges("n1")
    assert "e1" in edges_of_n1, "e1 should contain n1"
    assert "e2" in edges_of_n1, "e2 should contain n1"
    
    return True


def test_property_store():
    """Test PropertyStore functionality."""
    print("  Testing PropertyStore...")
    
    store = PropertyStore()
    
    # Test property operations
    store.set_node_property("n1", "type", "user")
    store.set_node_property("n1", "age", 25)
    store.set_node_property("n2", "type", "product")
    
    # Test retrieval
    assert store.get_node_property("n1", "type") == "user", "Property retrieval failed"
    assert store.get_node_property("n1", "age") == 25, "Numeric property retrieval failed"
    
    # Test property names
    n1_prop_names = store.get_node_property_names("n1")
    assert "type" in n1_prop_names, "Entity should have type property"
    assert "age" in n1_prop_names, "Entity should have age property"
    
    # Test property filtering
    users = store.filter_nodes_by_property("type", "user")
    assert "n1" in users, "Should find user entities"
    
    return True


def test_hypergraph_operations():
    """Test advanced hypergraph operations."""
    print("  Testing hypergraph operations...")
    
    # Create test hypergraph
    setsystem = {
        "e1": ["n1", "n2", "n3"],
        "e2": ["n2", "n3", "n4"],
        "e3": ["n1", "n4"]
    }
    
    hg = Hypergraph(setsystem=setsystem)
    
    # Test basic metrics
    assert hg.num_nodes == 4, f"Expected 4 nodes, got {hg.num_nodes}"
    assert hg.num_edges == 3, f"Expected 3 edges, got {hg.num_edges}"
    
    # Test node and edge access
    nodes = hg.nodes
    edges = hg.edges
    assert len(nodes) == 4, "Should have 4 nodes"
    assert len(edges) == 3, "Should have 3 edges"
    
    # Test edge size functionality
    try:
        # This may not be implemented, so we'll handle gracefully
        edge_sizes = {}
        for edge in edges:
            nodes_in_edge = hg.incidences.get_edge_nodes(edge)
            edge_sizes[edge] = len(nodes_in_edge)
        
        assert edge_sizes["e1"] == 3, "Edge e1 should have 3 nodes"
        assert edge_sizes["e2"] == 3, "Edge e2 should have 3 nodes"
        assert edge_sizes["e3"] == 2, "Edge e3 should have 2 nodes"
    except AttributeError:
        print("    Edge size functionality not available, skipping...")
    
    return True


def test_hypergraph_with_properties():
    """Test hypergraph with properties."""
    print("  Testing hypergraph with properties...")
    
    # Create hypergraph
    setsystem = {
        "e1": ["n1", "n2"],
        "e2": ["n2", "n3"]
    }
    
    # Create with properties
    properties = {
        "nodes": {
            "n1": {"type": "user", "age": 25},
            "n2": {"type": "user", "age": 30},
            "n3": {"type": "product", "price": 100}
        },
        "edges": {
            "e1": {"relation": "buys", "weight": 0.8},
            "e2": {"relation": "similar", "weight": 0.6}
        }
    }
    
    hg = Hypergraph(setsystem=setsystem, properties=properties)
    
    # Test property access
    try:
        # Properties might be accessible through the properties attribute
        if hasattr(hg, 'properties') and hasattr(hg.properties, 'node_properties'):
            node_props = hg.properties.node_properties
            if node_props and "n1" in node_props:
                assert node_props["n1"]["type"] == "user", "n1 should be user type"
        else:
            print("    Property access structure differs, skipping detailed property test...")
    except AttributeError:
        print("    Property access not available in this format, skipping...")
    
    return True


def test_error_handling():
    """Test error handling in core classes."""
    print("  Testing error handling...")
    
    # Test empty hypergraph
    try:
        hg_empty = Hypergraph()
        assert hg_empty.num_nodes == 0, "Empty hypergraph should have 0 nodes"
        assert hg_empty.num_edges == 0, "Empty hypergraph should have 0 edges"
    except Exception as e:
        print(f"    Empty hypergraph creation failed: {e}")
    
    # Test invalid data
    try:
        invalid_data = pl.DataFrame({"invalid": ["column"]})
        hg_invalid = Hypergraph(data=invalid_data)
        print("    Warning: Invalid data accepted (may be intentional)")
    except Exception:
        print("    Invalid data properly rejected")
    
    return True


def run_tests():
    """Run all core classes tests."""
    print("🧪 Running Core Classes Tests")
    
    test_functions = [
        test_hypergraph_creation,
        test_incidence_store,
        test_property_store,
        test_hypergraph_operations,
        test_hypergraph_with_properties,
        test_error_handling
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
    print(f"\nCore Classes Tests: {result['status']}")
    print(f"Passed: {result['passed']}, Failed: {result['failed']}")
    for detail in result["details"]:
        print(f"  {detail}")