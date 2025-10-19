#!/usr/bin/env python3
"""
Quick FIBO Loader - Test Script

Simple script to test loading and basic processing of FIBO JSON-LD files.
"""

import sys
import os
import json
from pathlib import Path

# Add ANANT to path
sys.path.insert(0, '/home/amansingh/dev/ai/anant')

import anant
import polars as pl
from anant import Hypergraph

try:
    import rdflib
    from rdflib import Graph
    from rdflib.namespace import RDF, RDFS, OWL
    print("✅ All libraries loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def quick_fibo_test():
    """Quick test of FIBO loading capabilities"""
    
    print("🏦 Quick FIBO Test")
    print("=" * 40)
    
    # Initialize ANANT
    anant.setup()
    
    # Test with MetadataFIBO.jsonld first
    metadata_file = "/home/amansingh/dev/ai/anant/prod.jsonld/fibo/ontology/master/2025Q3/MetadataFIBO.jsonld"
    
    if not os.path.exists(metadata_file):
        print(f"❌ Metadata file not found: {metadata_file}")
        return False
    
    print(f"📂 Loading: {metadata_file}")
    
    try:
        # Load RDF
        g = Graph()
        g.parse(metadata_file, format='json-ld')
        print(f"✅ Loaded {len(g):,} triples from metadata")
        
        # Create simple hypergraph
        hg = Hypergraph()
        
        # Add a few sample nodes and edges
        classes_found = 0
        for s, p, o in g.triples((None, RDF.type, OWL.Ontology)):
            ontology_name = str(s).split('/')[-2] if '/' in str(s) else str(s)
            hg.add_node(ontology_name, properties={
                'type': 'ontology',
                'uri': str(s)
            })
            classes_found += 1
            if classes_found >= 5:  # Limit for quick test
                break
        
        print(f"✅ Created hypergraph: {hg.num_nodes} nodes, {hg.num_edges} edges")
        
        # Save to parquet for testing
        output_dir = Path("/home/amansingh/dev/ai/anant/prod.jsonld")
        
        nodes_data = [{"node_id": str(node)} for node in hg.nodes]
        if nodes_data:
            nodes_df = pl.DataFrame(nodes_data)
            nodes_file = output_dir / "fibo_test_nodes.parquet"
            nodes_df.write_parquet(nodes_file)
            print(f"✅ Saved test parquet: {nodes_file}")
        
        print("🎯 Quick FIBO test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_fibo_test()
    sys.exit(0 if success else 1)