#!/usr/bin/env python3
"""Test the corrected property format"""

import polars as pl
from anant.classes.hypergraph import Hypergraph
from anant.classes.incidence_store import IncidenceStore


def test_correct_property_format():
    """Test that properties are handled with the correct format"""
    
    # Create test hypergraph
    test_df = pl.DataFrame([
        {"edge_id": "E1", "node_id": "A", "weight": 1.0},
        {"edge_id": "E1", "node_id": "B", "weight": 1.0},
    ])
    
    # Convert to IncidenceStore
    incidence_store = IncidenceStore(test_df)
    hg = Hypergraph(incidence_store, name="test_hypergraph")
    
    # Test correct format: add properties to nodes
    test_nodes = list(hg.nodes)[:2]
    print(f"test_nodes: {test_nodes}")
    
    # Add properties using the property store
    for node in test_nodes:
        hg.properties.set_node_property(node, "test_prop", 1.0)
    
    print("✓ Properties added with correct format")
    
    # Test retrieval
    if test_nodes:
        node_props = hg.properties.get_node_properties(test_nodes[0])
        print(f"Retrieved properties: {node_props}")
        
        has_test_prop = "test_prop" in node_props if isinstance(node_props, dict) else False
        print(f"Has test_prop: {has_test_prop}")
        
        if has_test_prop:
            print(f"test_prop value: {node_props['test_prop']}")
            print("✅ Property validation would PASS")
            assert node_props['test_prop'] == 1.0
        else:
            print("❌ Property validation would FAIL")
            assert False, "Property not found"
    
    assert len(test_nodes) >= 2, "Should have at least 2 nodes"