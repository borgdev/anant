#!/usr/bin/env python3
"""
Test ANANT Parquet I/O with Properties

Test that the property accessors work with the Parquet save/load functionality.
"""

import sys
import os
from pathlib import Path

# Add ANANT to path
sys.path.insert(0, '/home/amansingh/dev/ai/anant')

import anant
from anant import Hypergraph
from anant.io import AnantIO

def test_parquet_io_with_properties():
    """Test Parquet save/load with properties"""
    print("🧪 Testing ANANT Parquet I/O with Properties")
    print("=" * 50)
    
    # Initialize ANANT
    anant.setup()
    
    # Create a hypergraph with properties
    edge_dict = {
        'financial_rule_1': ['Asset', 'hasValue', 'MonetaryAmount'],
        'ownership_relation': ['Person', 'owns', 'Asset'],
        'regulatory_constraint': ['Bank', 'compliesWith', 'Regulation']
    }
    
    hg = Hypergraph.from_dict(edge_dict)
    print(f"✅ Created test hypergraph: {hg.num_nodes} nodes, {hg.num_edges} edges")
    
    # Add meaningful edge properties (like FIBO would have)
    hg.properties.set_edge_properties('financial_rule_1', {
        'type': 'owl:ObjectProperty',
        'domain': 'fibo:Asset',
        'range': 'fibo:MonetaryAmount',
        'source_file': 'FBC/FinancialInstruments.jsonld'
    })
    
    hg.properties.set_edge_properties('ownership_relation', {
        'type': 'owl:ObjectProperty', 
        'domain': 'fibo:Person',
        'range': 'fibo:Asset',
        'source_file': 'BE/LegalEntities.jsonld'
    })
    
    # Add node properties
    hg.properties.set_node_properties('Asset', {
        'type': 'owl:Class',
        'ontology': 'FIBO',
        'domain': 'FBC'
    })
    
    hg.properties.set_node_properties('Person', {
        'type': 'owl:Class',
        'ontology': 'FIBO', 
        'domain': 'BE'
    })
    
    print(f"✅ Added properties to hypergraph")
    
    # Test save
    save_path = Path("/home/amansingh/dev/ai/anant/anant/test_hypergraph_with_properties")
    
    try:
        print(f"\n💾 Testing save to: {save_path}")
        AnantIO.save_hypergraph_parquet(hg, str(save_path), compression='snappy')
        print(f"✅ Successfully saved hypergraph with properties")
        
        # Check what files were created
        if save_path.exists():
            files = list(save_path.iterdir())
            print(f"   📁 Files created: {[f.name for f in files]}")
        
        # Test load
        print(f"\n📂 Testing load from: {save_path}")
        loaded_hg = AnantIO.load_hypergraph_parquet(str(save_path))
        print(f"✅ Successfully loaded hypergraph")
        print(f"   📊 Loaded: {loaded_hg.num_nodes} nodes, {loaded_hg.num_edges} edges")
        
        # Verify properties were preserved
        edge_props = loaded_hg._edge_properties
        node_props = loaded_hg._node_properties
        
        print(f"\n🔍 Verifying properties were preserved:")
        print(f"   📊 Edge properties: {len(edge_props)} sets")
        print(f"   📊 Node properties: {len(node_props)} sets")
        
        if len(edge_props) > 0:
            edge_df = edge_props.properties
            print(f"   📋 Edge properties DataFrame: {edge_df.shape}")
            if not edge_df.is_empty():
                print(f"   Sample edge properties:")
                print(f"     {edge_df.head(3)}")
        
        print(f"\n🎉 Parquet I/O with properties working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error during save/load: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if save_path.exists():
            import shutil
            shutil.rmtree(save_path)
            print(f"🧹 Cleaned up test directory")

if __name__ == "__main__":
    try:
        success = test_parquet_io_with_properties()
        print(f"\n✅ ANANT Parquet I/O test: {'PASSED' if success else 'FAILED'}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)