#!/usr/bin/env python3
"""Debug JSON I/O issue"""

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
print(f"Original: {hg.num_nodes} nodes, {hg.num_edges} edges")
print(f"Nodes: {hg.nodes}")
print(f"Edges: {hg.edges}")

# Test the fallback JSON I/O from validation framework
try:
    from anant.validation import HypergraphIO
    
    # Test to_json
    json_data = HypergraphIO.to_json(hg)
    print(f"JSON data: {json_data}")
    
    # Test from_json 
    reconstructed_hg = HypergraphIO.from_json(json_data)
    print(f"Reconstructed: {reconstructed_hg.num_nodes} nodes, {reconstructed_hg.num_edges} edges")
    print(f"Reconstructed nodes: {reconstructed_hg.nodes}")
    print(f"Reconstructed edges: {reconstructed_hg.edges}")
    
except Exception as e:
    print(f"JSON I/O failed: {e}")
    import traceback
    traceback.print_exc()