#!/usr/bin/env python3
"""
FIBO Metagraph Loader

Load and analyze the saved FIBO metagraphs from parquet files.
"""

import sys
import os
from pathlib import Path
import json

# Add ANANT to path
sys.path.insert(0, '/home/amansingh/dev/ai/anant')

import anant
import polars as pl
from anant import Hypergraph
from anant.io import AnantIO

def load_fibo_metagraph():
    """Load the primary FIBO metagraph from parquet files"""
    
    print("🏦 Loading FIBO Primary Metagraph")
    print("=" * 40)
    
    # Initialize ANANT
    anant.setup()
    
    # Paths
    fibo_dir = Path("/home/amansingh/dev/ai/anant/prod.jsonld/fibo_metagraphs")
    
    if not fibo_dir.exists():
        print(f"❌ FIBO metagraphs directory not found: {fibo_dir}")
        return None
    
    try:
        # Load the primary metagraph
        hg = AnantIO.load_hypergraph_parquet(
            str(fibo_dir / "fibo_primary_metagraph")
        )
        
        print(f"✅ Loaded FIBO Primary Metagraph:")
        print(f"   📊 Nodes: {hg.num_nodes:,}")
        print(f"   📊 Edges: {hg.num_edges:,}")
        
        # Sample some nodes
        print(f"\n🔍 Sample FIBO Classes:")
        node_count = 0
        for node in hg.nodes:
            if node_count >= 5:
                break
            print(f"   • {node}")
            node_count += 1
        
        # Sample some edges
        print(f"\n🔗 Sample FIBO Relationships:")
        edge_count = 0
        for edge in hg.edges:
            if edge_count >= 5:
                break
            edge_nodes = hg.get_edge_nodes(edge)
            print(f"   • Edge {edge}: {len(edge_nodes)} nodes")
            edge_count += 1
        
        # Show metadata
        metadata_file = fibo_dir / "fibo_primary_metagraph_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"\n📄 Metadata:")
            print(f"   📅 Created: {metadata.get('created', 'unknown')}")
            print(f"   🔢 Version: {metadata.get('version', 'unknown')}")
        
        return hg
        
    except Exception as e:
        print(f"❌ Error loading FIBO metagraph: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_fibo_domains():
    """Analyze FIBO domain coverage"""
    
    print(f"\n🎯 FIBO Domain Analysis")
    print("=" * 40)
    
    fibo_dir = Path("/home/amansingh/dev/ai/anant/prod.jsonld/fibo_metagraphs")
    domains = ['actus', 'be', 'bp', 'cae', 'der', 'fbc', 'fnd', 'ind', 'loan', 'md', 'sec']
    
    for domain in domains:
        metadata_file = fibo_dir / f"fibo_{domain}_subgraph_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            nodes = metadata.get('nodes', 0)
            edges = metadata.get('edges', 0)
            print(f"   📁 {domain.upper()}: {nodes:,} nodes, {edges:,} edges")

if __name__ == "__main__":
    # Load primary metagraph
    hg = load_fibo_metagraph()
    
    if hg:
        # Analyze domains
        analyze_fibo_domains()
        
        print(f"\n🎉 FIBO metagraph successfully loaded and ready for analysis!")
        print(f"💡 Use the returned hypergraph object for further analysis")
    else:
        print(f"❌ Failed to load FIBO metagraph")
        sys.exit(1)