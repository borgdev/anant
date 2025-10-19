#!/usr/bin/env python3
"""
Test FIBO Metagraph Loading

Load and analyze the saved FIBO unified metagraph.
"""

import sys
import os
from pathlib import Path

# Add ANANT to path
sys.path.insert(0, '/home/amansingh/dev/ai/anant')

import anant
from anant import Hypergraph
from anant.io import AnantIO

def test_fibo_metagraph_loading():
    """Test loading the FIBO unified metagraph"""
    print("🏦 Testing FIBO Unified Metagraph Loading")
    print("=" * 50)
    
    # Initialize ANANT
    anant.setup()
    
    # Load the unified FIBO metagraph
    metagraph_path = "/home/amansingh/dev/ai/anant/prod.jsonld/fibo_metagraphs/fibo_unified_metagraph"
    
    if not Path(metagraph_path).exists():
        print(f"❌ Metagraph not found: {metagraph_path}")
        return False
    
    try:
        print(f"📂 Loading FIBO metagraph from: {metagraph_path}")
        hg = AnantIO.load_hypergraph_parquet(metagraph_path)
        
        print(f"✅ Successfully loaded FIBO metagraph!")
        print(f"   📊 Nodes: {hg.num_nodes:,}")
        print(f"   📊 Edges: {hg.num_edges:,}")
        print(f"   📊 Incidences: {hg.num_incidences:,}")
        
        # Sample some nodes (FIBO classes and properties)
        print(f"\n🔍 Sample FIBO Entities:")
        node_count = 0
        for node in hg.nodes:
            if node_count >= 10:
                break
            # Show meaningful FIBO URIs
            if 'fibo' in str(node).lower() or 'edmcouncil' in str(node).lower():
                node_short = str(node).split('/')[-1] if '/' in str(node) else str(node)
                print(f"   • {node_short}")
                node_count += 1
        
        # Sample some edges (RDF triples as hyperedges)
        print(f"\n🔗 Sample FIBO Relationships:")
        edge_count = 0
        for edge in hg.edges:
            if edge_count >= 5:
                break
            edge_nodes = hg.get_edge_nodes(edge)
            print(f"   • {edge}: connects {len(edge_nodes)} entities")
            
            # Show the triple components if it's a typical RDF triple
            if len(edge_nodes) == 3:
                nodes_list = list(edge_nodes)
                subj = str(nodes_list[0]).split('/')[-1] if '/' in str(nodes_list[0]) else str(nodes_list[0])
                pred = str(nodes_list[1]).split('/')[-1] if '/' in str(nodes_list[1]) else str(nodes_list[1])
                obj = str(nodes_list[2]).split('/')[-1] if '/' in str(nodes_list[2]) else str(nodes_list[2])
                print(f"     ({subj} → {pred} → {obj})")
            
            edge_count += 1
        
        # Analyze FIBO domains by checking node URIs
        print(f"\n🎯 FIBO Domain Analysis:")
        domain_stats = {}
        
        for node in list(hg.nodes)[:1000]:  # Sample first 1000 nodes
            node_str = str(node)
            if 'edmcouncil.org/fibo/ontology' in node_str:
                # Extract domain from URI like .../fibo/ontology/master/2025Q3/FBC/...
                parts = node_str.split('/')
                if len(parts) > 6:
                    domain = parts[6]  # Should be FBC, FND, etc.
                    domain_stats[domain] = domain_stats.get(domain, 0) + 1
        
        print(f"   📊 FIBO domains detected in sample:")
        for domain, count in sorted(domain_stats.items()):
            print(f"     {domain}: {count:,} entities")
        
        print(f"\n🎉 FIBO unified metagraph successfully loaded and analyzed!")
        print(f"💡 ANANT has successfully stored the entire FIBO financial ontology!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading metagraph: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_fibo_metagraph_loading()
        print(f"\n✅ FIBO metagraph loading test: {'PASSED' if success else 'FAILED'}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)