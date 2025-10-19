#!/usr/bin/env python3
"""Debug the exact property issue in validation"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anant'))

import polars as pl
from anant.classes.hypergraph import Hypergraph

# Create the exact same hypergraph as in the test
test_df = pl.DataFrame([
    {"edges": "E1", "nodes": "A", "weight": 1.0},
    {"edges": "E1", "nodes": "B", "weight": 1.0},
    {"edges": "E2", "nodes": "B", "weight": 1.0},
    {"edges": "E2", "nodes": "C", "weight": 1.0},
    {"edges": "E3", "nodes": "C", "weight": 1.0},
    {"edges": "E3", "nodes": "D", "weight": 1.0},
])

hg = Hypergraph(test_df, name="test_hypergraph")

# Replicate the exact validation logic
print("=== Replicating Validation Logic ===")
test_nodes = list(hg.nodes)[:3]
print(f"test_nodes: {test_nodes}")

# Add test properties (exactly as in validation)
hg.add_node_properties({"test_prop": {node: 1.0 for node in test_nodes}})
print("✓ Properties added")

# Test individual node property retrieval (exactly as in validation)
if test_nodes:
    node_prop = hg.get_node_properties(test_nodes[0])
    print(f"node_prop for '{test_nodes[0]}': {node_prop}")
    
    # Check the validation condition
    is_dict = isinstance(node_prop, dict)
    has_test_prop = "test_prop" in node_prop if is_dict else False
    
    print(f"isinstance(node_prop, dict): {is_dict}")
    print(f"'test_prop' in node_prop: {has_test_prop}")
    
    # This is the exact condition in the validation
    condition_result = not isinstance(node_prop, dict) or "test_prop" not in node_prop
    print(f"Validation condition (should be False): {condition_result}")
    
    if condition_result:
        print("❌ Property storage/retrieval failed (according to validation logic)")
    else:
        print("✅ Property storage/retrieval succeeded")

# Let's also check what keys are in the node_prop
if test_nodes and isinstance(node_prop, dict):
    print(f"Keys in node_prop: {list(node_prop.keys())}")
    if "test_prop" in node_prop:
        print(f"test_prop value: {node_prop['test_prop']}")