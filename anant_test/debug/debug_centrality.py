#!/usr/bin/env python3
"""Debug centrality analysis"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anant'))

import polars as pl
from anant.classes.hypergraph import Hypergraph
from anant.analysis.centrality import degree_centrality

# Create test hypergraph
test_df = pl.DataFrame([
    {"edges": "E1", "nodes": "A", "weight": 1.0},
    {"edges": "E1", "nodes": "B", "weight": 1.0},
    {"edges": "E2", "nodes": "B", "weight": 1.0},
    {"edges": "E2", "nodes": "C", "weight": 1.0},
    {"edges": "E3", "nodes": "C", "weight": 1.0},
    {"edges": "E3", "nodes": "D", "weight": 1.0},
])

hg = Hypergraph(test_df, name="test_hypergraph")
print(f"Hypergraph nodes: {hg.num_nodes}")
print(f"Nodes: {hg.nodes}")

centralities = degree_centrality(hg)
print(f"Centrality result type: {type(centralities)}")
print(f"Centrality result: {centralities}")
print(f"Centrality length: {len(centralities) if isinstance(centralities, dict) else 'N/A'}")