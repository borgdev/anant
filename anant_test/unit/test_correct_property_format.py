#!/usr/bin/env python3
"""Test the corrected property format"""

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

# Test correct format: {node_id: {prop_name: value}}
test_nodes = list(hg.nodes)[:2]
print(f"test_nodes: {test_nodes}")

test_properties = {node: {"test_prop": 1.0} for node in test_nodes}
print(f"test_properties: {test_properties}")

hg.add_node_properties(test_properties)
print("✓ Properties added with correct format")

# Test retrieval
node_prop = hg.get_node_properties(test_nodes[0])
print(f"Retrieved properties: {node_prop}")

has_test_prop = "test_prop" in node_prop if isinstance(node_prop, dict) else False
print(f"Has test_prop: {has_test_prop}")

if has_test_prop:
    print(f"test_prop value: {node_prop['test_prop']}")
    print("✅ Property validation would PASS")