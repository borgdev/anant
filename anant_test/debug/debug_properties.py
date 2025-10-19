#!/usr/bin/env python3
"""Debug property integration"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anant'))

import polars as pl
from anant.classes.hypergraph import Hypergraph

# Create test hypergraph
test_df = pl.DataFrame([
    {"edges": "E1", "nodes": "A", "weight": 1.0},
    {"edges": "E1", "nodes": "B", "weight": 1.0},
    {"edges": "E2", "nodes": "B", "weight": 1.0},
    {"edges": "E2", "nodes": "C", "weight": 1.0},
])

hg = Hypergraph(test_df, name="test_hypergraph")
print(f"Hypergraph nodes: {hg.nodes}")

# Test property operations
test_nodes = list(hg.nodes)[:3]
print(f"Test nodes: {test_nodes}")

# Add properties
try:
    hg.add_node_properties({"test_prop": {node: 1.0 for node in test_nodes}})
    print("✓ Properties added successfully")
except Exception as e:
    print(f"✗ Adding properties failed: {e}")
    import traceback
    traceback.print_exc()

# Test retrieval
try:
    if test_nodes:
        node_prop = hg.get_node_properties(test_nodes[0])
        print(f"Retrieved property: {node_prop}")
        if not node_prop:
            print("✗ Property retrieval returned empty result")
        else:
            print("✓ Property retrieval successful")
except Exception as e:
    print(f"✗ Property retrieval failed: {e}")
    import traceback
    traceback.print_exc()