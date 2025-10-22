"""
Anant Integration Example
==========================

Demonstrates how MultiModalHypergraph extends Anant's core Hypergraph class.
Shows integration with Anant's full ecosystem.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import polars as pl
    import numpy as np
except ImportError:
    print("⚠️  Required dependencies not installed. Please install: polars numpy")
    sys.exit(1)

# Import Anant core (if available)
try:
    from anant.classes.hypergraph import Hypergraph as AnantHypergraph
    ANANT_AVAILABLE = True
    print("✅ Anant core library available")
except ImportError:
    ANANT_AVAILABLE = False
    print("⚠️  Anant core library not available")
    # Use mock for demo
    class AnantHypergraph:
        def __init__(self, data=None, **kwargs):
            self.data = data
            if data is not None and 'nodes' in data.columns:
                self._nodes = set(data['nodes'].unique())
            else:
                self._nodes = set()
        def nodes(self):
            return self._nodes

# Import multi-modal components
from core.multi_modal_hypergraph import MultiModalHypergraph


def create_anant_hypergraphs():
    """Create Anant Hypergraph instances for different modalities"""
    
    print("\n📦 Creating Anant Hypergraphs...")
    
    # E-commerce modalities (using Anant's required column names: node_id, edge_id)
    purchases = pl.DataFrame([
        {'edge_id': 'p1', 'node_id': 'customer_1', 'weight': 100.0, 'role': 'buyer'},
        {'edge_id': 'p1', 'node_id': 'product_A', 'weight': 1, 'role': 'product'},
        {'edge_id': 'p2', 'node_id': 'customer_2', 'weight': 50.0, 'role': 'buyer'},
        {'edge_id': 'p2', 'node_id': 'product_B', 'weight': 1, 'role': 'product'},
        {'edge_id': 'p3', 'node_id': 'customer_1', 'weight': 75.0, 'role': 'buyer'},
        {'edge_id': 'p3', 'node_id': 'product_C', 'weight': 1, 'role': 'product'},
    ])
    
    reviews = pl.DataFrame([
        {'edge_id': 'r1', 'node_id': 'customer_1', 'weight': 5, 'role': 'reviewer'},
        {'edge_id': 'r1', 'node_id': 'product_A', 'weight': 5, 'role': 'reviewed'},
        {'edge_id': 'r2', 'node_id': 'customer_2', 'weight': 4, 'role': 'reviewer'},
        {'edge_id': 'r2', 'node_id': 'product_B', 'weight': 4, 'role': 'reviewed'},
    ])
    
    wishlists = pl.DataFrame([
        {'edge_id': 'w1', 'node_id': 'customer_2', 'weight': 1, 'role': 'wisher'},
        {'edge_id': 'w1', 'node_id': 'product_C', 'weight': 1, 'role': 'wished'},
        {'edge_id': 'w2', 'node_id': 'customer_1', 'weight': 1, 'role': 'wisher'},
        {'edge_id': 'w2', 'node_id': 'product_D', 'weight': 1, 'role': 'wished'},
    ])
    
    # Create Anant Hypergraphs
    if ANANT_AVAILABLE:
        # Use actual Anant Hypergraph with full functionality
        purchase_hg = AnantHypergraph(data=purchases, name="purchases")
        review_hg = AnantHypergraph(data=reviews, name="reviews")
        wishlist_hg = AnantHypergraph(data=wishlists, name="wishlists")
        
        print("   ✅ Created Anant Hypergraph instances with full functionality")
    else:
        # Use mock hypergraphs
        purchase_hg = AnantHypergraph(data=purchases)
        review_hg = AnantHypergraph(data=reviews)
        wishlist_hg = AnantHypergraph(data=wishlists)
        
        print("   ✅ Created mock hypergraph instances (Anant not available)")
    
    return purchase_hg, review_hg, wishlist_hg


def demonstrate_inheritance():
    """Demonstrate that MultiModalHypergraph extends Anant Hypergraph"""
    
    print("\n" + "="*70)
    print("🔗 Demonstrating Anant Integration")
    print("="*70)
    
    # Create multi-modal hypergraph
    print("\n1. Creating MultiModalHypergraph (extends Anant Hypergraph)...")
    mmhg = MultiModalHypergraph(name="ecommerce_multimodal")
    
    # Verify inheritance
    print(f"\n   Instance type: {type(mmhg).__name__}")
    print(f"   Base classes: {[c.__name__ for c in type(mmhg).__mro__]}")
    
    if ANANT_AVAILABLE:
        print(f"   ✅ Inherits from: AnantHypergraph")
        print(f"   ✅ Has all Anant functionality plus multi-modal capabilities")
    else:
        print(f"   ⚠️  Running in standalone mode")
    
    # Create Anant hypergraphs
    purchase_hg, review_hg, wishlist_hg = create_anant_hypergraphs()
    
    # Add modalities
    print("\n2. Adding modalities (Anant Hypergraph instances)...")
    mmhg.add_modality("purchases", purchase_hg, weight=2.0)
    mmhg.add_modality("reviews", review_hg, weight=1.5)
    mmhg.add_modality("wishlists", wishlist_hg, weight=1.0)
    
    print("   ✅ Added 3 modalities")
    
    # Use multi-modal capabilities
    print("\n3. Using multi-modal analysis capabilities...")
    
    # Summary
    summary = mmhg.generate_summary()
    print(f"\n   📊 Summary:")
    print(f"      Name: {summary['name']}")
    print(f"      Modalities: {summary['num_modalities']}")
    print(f"      Entities: {summary['total_unique_entities']}")
    
    # Find cross-modal entities
    bridges = mmhg.find_modal_bridges(min_modalities=2)
    print(f"\n   🌉 Modal Bridges:")
    print(f"      Entities in 2+ modalities: {len(bridges)}")
    for entity, mods in list(bridges.items())[:3]:
        print(f"         {entity}: {', '.join(mods)}")
    
    # Cross-modal patterns
    patterns = mmhg.detect_cross_modal_patterns(min_support=1)
    print(f"\n   🔎 Cross-Modal Patterns:")
    print(f"      Patterns detected: {len(patterns)}")
    for pattern in patterns[:2]:
        print(f"         {pattern['type']}: {pattern['support']} instances")
    
    # Modal correlation
    corr = mmhg.compute_modal_correlation("purchases", "reviews")
    print(f"\n   📈 Modal Correlation:")
    print(f"      Purchases ↔ Reviews: {corr:.3f}")
    
    # Centrality (multi-modal)
    centrality = mmhg.compute_cross_modal_centrality(
        "customer_1",
        metric="degree",
        aggregation="weighted_average"
    )
    print(f"\n   ⭐ Cross-Modal Centrality:")
    print(f"      customer_1: {centrality['aggregated']:.2f}")
    print(f"         By modality:")
    for mod, score in centrality['per_modality'].items():
        print(f"            {mod}: {score:.1f}")
    
    print("\n" + "="*70)
    print("✅ Integration Demonstration Complete!")
    print("="*70)
    
    return mmhg


def demonstrate_anant_features():
    """Demonstrate inherited Anant features (if available)"""
    
    if not ANANT_AVAILABLE:
        print("\n⚠️  Anant not available - skipping Anant-specific features")
        return
    
    print("\n" + "="*70)
    print("🎯 Demonstrating Inherited Anant Features")
    print("="*70)
    
    # Create multi-modal hypergraph with base data
    print("\n1. Creating MultiModalHypergraph with base hypergraph data...")
    
    base_data = pl.DataFrame([
        {'edge_id': 'e1', 'node_id': 'n1', 'weight': 1.0},
        {'edge_id': 'e1', 'node_id': 'n2', 'weight': 1.0},
        {'edge_id': 'e2', 'node_id': 'n2', 'weight': 1.0},
        {'edge_id': 'e2', 'node_id': 'n3', 'weight': 1.0},
    ])
    
    mmhg = MultiModalHypergraph(
        name="hybrid_multimodal",
        data=base_data
    )
    
    print("   ✅ Created with base hypergraph data")
    
    # Access Anant features
    print("\n2. Accessing inherited Anant Hypergraph features...")
    
    if hasattr(mmhg, 'incidences'):
        print(f"   ✅ Has incidence store: {type(mmhg.incidences).__name__}")
    
    if hasattr(mmhg, 'properties'):
        print(f"   ✅ Has property store: {type(mmhg.properties).__name__}")
    
    if hasattr(mmhg, '_core_ops'):
        print(f"   ✅ Has core operations module")
    
    if hasattr(mmhg, '_algorithm_ops'):
        print(f"   ✅ Has algorithm operations module")
    
    print("\n3. MultiModalHypergraph = Anant Hypergraph + Multi-Modal Features")
    print("   Base features:")
    print("      ✓ Incidence management")
    print("      ✓ Property management")
    print("      ✓ Core operations")
    print("      ✓ Algorithm operations")
    print("   ")
    print("   Multi-modal features:")
    print("      ✓ Modality management")
    print("      ✓ Cross-modal pattern detection")
    print("      ✓ Inter-modal relationships")
    print("      ✓ Modal correlation analysis")
    print("      ✓ Aggregate centrality metrics")


def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("🎯 Anant Integration Example")
    print("="*70)
    print("\nDemonstrating how MultiModalHypergraph extends Anant's Hypergraph")
    print("This shows full integration with the Anant ecosystem\n")
    
    # Demonstrate inheritance
    mmhg = demonstrate_inheritance()
    
    # Demonstrate Anant features (if available)
    demonstrate_anant_features()
    
    print("\n" + "="*70)
    print("💡 Key Points")
    print("="*70)
    print("\n✅ MultiModalHypergraph extends Anant's core Hypergraph class")
    print("✅ Inherits all Anant functionality (when available)")
    print("✅ Adds multi-modal specific capabilities on top")
    print("✅ Works in standalone mode when Anant not available")
    print("✅ Seamlessly integrates with Anant ecosystem")
    
    print("\n📚 Integration Benefits:")
    print("   • Access to all Anant's graph algorithms")
    print("   • Use Anant's property management system")
    print("   • Leverage Anant's I/O operations")
    print("   • Combine with Anant's visualization tools")
    print("   • Full compatibility with Anant data structures\n")


if __name__ == "__main__":
    main()
