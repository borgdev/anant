#!/usr/bin/env python3
"""
Debug script to understand DataFrame structure from Hypergraph
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anant'))

from anant.classes import Hypergraph
from anant.factory import SetSystemFactory
from typing import Dict, Iterable, cast


def debug_hypergraph_dataframes():
    """Debug what DataFrames look like"""
    print("Creating test hypergraph...")
    
    # Create test data
    edge_data = {
        "edge1": ["node1", "node2", "node3"],
        "edge2": ["node2", "node4"],
        "edge3": ["node1", "node4", "node5"],
        "edge4": ["node3", "node5", "node6"]
    }
    
    # Create hypergraph
    incidence_df = SetSystemFactory.from_dict_of_iterables(cast(Dict[str, Iterable], edge_data))
    hg = Hypergraph(incidence_df)
    
    print(f"Created hypergraph: {hg}")
    print(f"Nodes: {hg.nodes}")
    print(f"Edges: {hg.edges}")
    print(f"Num incidences: {hg.num_incidences}")
    
    # Check DataFrames
    print("\n=== Nodes DataFrame ===")
    nodes_df = hg.to_dataframe("nodes")
    print(f"Shape: {nodes_df.shape}")
    print(f"Columns: {nodes_df.columns}")
    print(f"Schema: {nodes_df.schema}")
    print(nodes_df.head())
    
    print("\n=== Edges DataFrame ===")
    edges_df = hg.to_dataframe("edges")
    print(f"Shape: {edges_df.shape}")
    print(f"Columns: {edges_df.columns}")
    print(f"Schema: {edges_df.schema}")
    print(edges_df.head())
    
    print("\n=== Incidences DataFrame ===")
    incidences_df = hg.to_dataframe("incidences")
    print(f"Shape: {incidences_df.shape}")
    print(f"Columns: {incidences_df.columns}")
    print(f"Schema: {incidences_df.schema}")
    print(incidences_df.head())
    
    # Add properties and check again
    print("\n=== Adding Properties ===")
    node_props = {
        "node1": {"type": "source", "weight": 1.5},
        "node2": {"type": "intermediate", "weight": 2.0},
        "node3": {"type": "intermediate", "weight": 1.8},
        "node4": {"type": "sink", "weight": 2.5},
        "node5": {"type": "sink", "weight": 2.2},
        "node6": {"type": "sink", "weight": 1.9}
    }
    hg.add_node_properties(node_props)
    
    edge_props = {
        "edge1": {"category": "A", "strength": 0.8},
        "edge2": {"category": "B", "strength": 0.6},
        "edge3": {"category": "A", "strength": 0.9},
        "edge4": {"category": "C", "strength": 0.7}
    }
    hg.add_edge_properties(edge_props)
    
    print("\n=== Nodes DataFrame (with properties) ===")
    nodes_df = hg.to_dataframe("nodes")
    print(f"Shape: {nodes_df.shape}")
    print(f"Columns: {nodes_df.columns}")
    print(f"Schema: {nodes_df.schema}")
    print(nodes_df.head())
    
    print("\n=== Edges DataFrame (with properties) ===")
    edges_df = hg.to_dataframe("edges")
    print(f"Shape: {edges_df.shape}")
    print(f"Columns: {edges_df.columns}")
    print(f"Schema: {edges_df.schema}")
    print(edges_df.head())


if __name__ == "__main__":
    debug_hypergraph_dataframes()