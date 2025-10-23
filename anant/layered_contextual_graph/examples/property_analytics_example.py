"""
Property-Level Analytics Example
================================

Demonstrates property-level analytics in LCG using PropertyStore.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import polars as pl
except ImportError:
    print("⚠️  Polars required")
    sys.exit(1)

try:
    from anant.classes.hypergraph.core.hypergraph import Hypergraph as AnantHypergraph
    ANANT_AVAILABLE = True
except ImportError:
    ANANT_AVAILABLE = False
    print("⚠️  Anant required for full demo")
    sys.exit(1)

from anant.layered_contextual_graph.core import LayeredContextualGraph, LayerType
from anant.layered_contextual_graph.analytics import (
    PropertyAnalytics,
    IndexAnalytics,
    TagAnalytics,
    derive_contexts_from_properties,
    cluster_by_tags
)


def main():
    print("\n" + "="*70)
    print("🔍 PROPERTY-LEVEL ANALYTICS DEMO")
    print("="*70)
    
    # Create LCG
    lcg = LayeredContextualGraph(name="property_demo")
    
    # Create layers with properties
    data = pl.DataFrame([{'edge_id': 'e1', 'node_id': 'n1', 'weight': 1.0}])
    hg = AnantHypergraph(data=data, name="physical")
    
    # Add properties
    hg.properties.set_node_property('n1', 'category', 'sensor')
    hg.properties.set_node_property('n1', 'tags', ['iot', 'monitoring'])
    hg.properties.set_node_property('n1', 'priority', 'high')
    
    lcg.add_layer("physical", hg, LayerType.PHYSICAL, level=0)
    
    print("✅ Created LCG with properties")
    
    # Property Analytics
    print("\n📊 Property Analytics:")
    pa = PropertyAnalytics(lcg)
    summary = pa.get_property_summary()
    print(f"   Layers: {summary['total_layers']}")
    
    # Derive contexts
    print("\n🎯 Deriving Contexts from Properties:")
    contexts = derive_contexts_from_properties(lcg, auto_apply=True)
    print(f"   ✅ Derived {len(contexts)} contexts")
    
    # Index Analytics
    print("\n🔎 Index-Based Queries:")
    ia = IndexAnalytics(lcg)
    index = ia.build_cross_layer_index('category')
    print(f"   ✅ Built index: {len(index)} unique values")
    
    # Tag Analytics
    print("\n🏷️  Tag-Based Clustering:")
    ta = TagAnalytics(lcg)
    clusters = ta.cluster_by_tags(min_cluster_size=1)
    print(f"   ✅ Found {len(clusters)} tag clusters")
    
    print("\n✅ Property analytics complete!")
    print("\n💡 LCG can analyze:")
    print("   • Properties across layers")
    print("   • Automatic context derivation")
    print("   • Fast property-based indices")
    print("   • Tag-based clustering")
    print()


if __name__ == "__main__":
    main()
