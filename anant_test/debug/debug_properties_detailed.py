#!/usr/bin/env python3
"""Debug property store in detail"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anant'))

import polars as pl
from anant.classes.hypergraph import Hypergraph

# Create test hypergraph
test_df = pl.DataFrame([
    {"edges": "E1", "nodes": "A", "weight": 1.0},
    {"edges": "E1", "nodes": "B", "weight": 1.0},
])

hg = Hypergraph(test_df, name="test_hypergraph")
print(f"Hypergraph nodes: {hg.nodes}")

# Test property operations step by step
test_node = "A"
print(f"Test node: {test_node}")

# Check property store directly
print(f"Property store before: {hg._node_properties}")

# Add single property using direct method
try:
    hg._node_properties.set_property(test_node, "test_prop", 1.0)
    print("✓ Direct property set successful")
    
    # Try to retrieve directly
    direct_result = hg._node_properties.get_properties(test_node)
    print(f"Direct retrieval: {direct_result}")
    
except Exception as e:
    print(f"✗ Direct property operations failed: {e}")
    import traceback
    traceback.print_exc()

# Test through hypergraph interface
try:
    hg.add_node_properties({"test_prop2": {test_node: 2.0}})
    print("✓ Hypergraph add_node_properties successful")
    
    # Retrieve through hypergraph interface
    hg_result = hg.get_node_properties(test_node)
    print(f"Hypergraph retrieval: {hg_result}")
    
except Exception as e:
    print(f"✗ Hypergraph property operations failed: {e}")
    import traceback
    traceback.print_exc()