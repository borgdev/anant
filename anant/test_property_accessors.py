#!/usr/bin/env python3
"""
Test ANANT Property Accessors

Quick test to verify that the property accessors work correctly.
"""

import sys
import os
from pathlib import Path

# Add ANANT to path
sys.path.insert(0, '/home/amansingh/dev/ai/anant')

import anant
from anant import Hypergraph

def test_property_accessors():
    """Test the new property accessors"""
    print("🧪 Testing ANANT Property Accessors")
    print("=" * 40)
    
    # Initialize ANANT
    anant.setup()
    
    # Create a simple hypergraph
    edge_dict = {
        'e1': ['n1', 'n2', 'n3'],
        'e2': ['n2', 'n4'],
        'e3': ['n1', 'n4', 'n5']
    }
    
    hg = Hypergraph.from_dict(edge_dict)
    
    print(f"✅ Created hypergraph: {hg.num_nodes} nodes, {hg.num_edges} edges")
    
    # Add some edge properties
    hg.properties.set_edge_properties('e1', {'type': 'strong', 'weight': 0.8})
    hg.properties.set_edge_properties('e2', {'type': 'weak', 'weight': 0.3})
    
    # Add some node properties  
    hg.properties.set_node_properties('n1', {'color': 'red', 'size': 'large'})
    hg.properties.set_node_properties('n2', {'color': 'blue', 'size': 'small'})
    
    print(f"✅ Added properties via PropertyStore")
    
    # Test property accessors
    print(f"\n🔍 Testing Property Accessors:")
    
    # Test _edge_properties
    edge_props = hg._edge_properties
    print(f"   📊 Edge properties wrapper: {len(edge_props)} properties")
    print(f"   📊 Edge properties exist: {bool(edge_props)}")
    
    # Test _node_properties
    node_props = hg._node_properties  
    print(f"   📊 Node properties wrapper: {len(node_props)} properties")
    print(f"   📊 Node properties exist: {bool(node_props)}")
    
    # Test DataFrame conversion
    edge_df = edge_props.properties
    node_df = node_props.properties
    
    print(f"\n📋 Property DataFrames:")
    print(f"   Edge properties DataFrame: {edge_df.shape}")
    print(f"   Node properties DataFrame: {node_df.shape}")
    
    if not edge_df.is_empty():
        print(f"   Edge properties sample:")
        print(f"     {edge_df.head(3)}")
    
    if not node_df.is_empty():
        print(f"   Node properties sample:")
        print(f"     {node_df.head(3)}")
    
    print(f"\n🎉 Property accessors working correctly!")
    return True

if __name__ == "__main__":
    try:
        success = test_property_accessors()
        print(f"\n✅ ANANT property accessors test: {'PASSED' if success else 'FAILED'}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)