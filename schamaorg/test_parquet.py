#!/usr/bin/env python3
import sys
import json
sys.path.insert(0, '/home/amansingh/dev/ai/anant')

import anant
from anant import Hypergraph
import polars as pl

print("🚀 Testing parquet persistence in schamaorg folder")

# Initialize ANANT
anant.setup()

# Create simple hypergraph
hg = Hypergraph()
hg.add_node("Person", properties={'type': 'class'})
hg.add_node("Organization", properties={'type': 'class'})
hg.add_edge("rel1", ["Person", "Organization"], properties={'type': 'memberOf'})

print(f"Created hypergraph: {hg.num_nodes} nodes, {hg.num_edges} edges")

# Save to parquet in current directory (schamaorg)
nodes_data = [{"node_id": str(node)} for node in hg.nodes]
nodes_df = pl.DataFrame(nodes_data)
nodes_df.write_parquet("schema_org_nodes.parquet")

edges_data = []
for edge_id in hg.edges:
    edge_nodes = hg.get_edge_nodes(edge_id)
    edges_data.append({"edge_id": str(edge_id), "nodes": json.dumps(list(edge_nodes))})
edges_df = pl.DataFrame(edges_data)
edges_df.write_parquet("schema_org_edges.parquet")

print("✅ Saved parquet files to schamaorg folder!")

# Test loading
nodes_df2 = pl.read_parquet("schema_org_nodes.parquet")
edges_df2 = pl.read_parquet("schema_org_edges.parquet")

print(f"✅ Loaded: {len(nodes_df2)} nodes, {len(edges_df2)} edges from parquet")
print("🎯 Parquet persistence in schamaorg folder working!")
