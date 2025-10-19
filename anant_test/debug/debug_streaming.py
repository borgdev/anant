#!/usr/bin/env python3
"""Debug streaming DataFrame issues"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anant'))

import polars as pl
from anant.classes.hypergraph import Hypergraph

# Create initial hypergraph
initial_df = pl.DataFrame([
    {"edges": "E1", "nodes": "A", "weight": 1.0},
    {"edges": "E1", "nodes": "B", "weight": 1.0},
    {"edges": "E2", "nodes": "B", "weight": 1.0},
    {"edges": "E2", "nodes": "C", "weight": 1.0},
])

print("Initial DataFrame:")
print(initial_df)
print("\nColumns:", initial_df.columns)
print("Shape:", initial_df.shape)

# Create hypergraph
hg = Hypergraph(initial_df)
print("\nHypergraph incidence store data:")
print(hg._incidence_store.data)
print("Columns:", hg._incidence_store.data.columns)
print("Shape:", hg._incidence_store.data.shape)

# Create new edge data
new_incidences = [
    {"edges": "E3", "nodes": "C", "weight": 1.0},
    {"edges": "E3", "nodes": "D", "weight": 1.0}
]

new_df = pl.DataFrame(new_incidences)
print("\nNew DataFrame:")
print(new_df)
print("Columns:", new_df.columns)
print("Shape:", new_df.shape)

# Test concatenation
try:
    combined_df = pl.concat([hg._incidence_store.data, new_df])
    print("\nCombined DataFrame:")
    print(combined_df)
    print("SUCCESS!")
except Exception as e:
    print(f"\nERROR: {e}")