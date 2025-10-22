"""
E-Commerce Multi-Modal Analysis Demo
====================================

Demonstrates multi-modal relationship analysis for e-commerce customer behavior.

Modalities:
- Purchases: Customer-product purchase relationships
- Reviews: Customer-product review relationships
- Wishlists: Customer-product wishlist relationships
- Returns: Customer-product return relationships

Cross-Modal Insights:
- Customers who review without purchasing
- Products frequently wishlisted but rarely bought
- Review patterns vs purchase patterns
- Customer segmentation across modalities
"""

import sys
from pathlib import Path
import time
from datetime import datetime, timedelta
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import polars as pl
    import numpy as np
except ImportError:
    print("⚠️  Required dependencies not installed. Please install: polars numpy")
    sys.exit(1)

# Import Anant core (try both absolute and relative imports)
try:
    from anant import Hypergraph
    ANANT_AVAILABLE = True
except ImportError:
    ANANT_AVAILABLE = False
    print("⚠️  Anant core library not found. Using mock hypergraphs.")

from core.multi_modal_hypergraph import MultiModalHypergraph


class MockHypergraph:
    """Mock hypergraph for demo when Anant not available"""
    def __init__(self, data):
        self.data = data
        self.incidences = type('obj', (object,), {'data': data})()
        if 'nodes' in data.columns:
            self._nodes = set(data['nodes'].unique())
        else:
            self._nodes = set()
    
    def nodes(self):
        return self._nodes


def generate_ecommerce_data(
    num_customers: int = 500,
    num_products: int = 200,
    num_purchases: int = 2000,
    num_reviews: int = 800,
    num_wishlists: int = 1500,
    num_returns: int = 300
):
    """Generate synthetic e-commerce data for demo"""
    
    print("📊 Generating E-Commerce Data...")
    print(f"   Customers: {num_customers}")
    print(f"   Products: {num_products}")
    
    # Generate customer and product IDs
    customers = [f"customer_{i:04d}" for i in range(num_customers)]
    products = [f"product_{i:04d}" for i in range(num_products)]
    
    # Generate purchases (edge-node incidences)
    purchases = []
    for i in range(num_purchases):
        purchase_id = f"purchase_{i:05d}"
        customer = random.choice(customers)
        product = random.choice(products)
        
        # Customer node
        purchases.append({
            'edges': purchase_id,
            'nodes': customer,
            'weight': random.uniform(10, 500),  # Purchase amount
            'role': 'customer',
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 365))
        })
        
        # Product node
        purchases.append({
            'edges': purchase_id,
            'nodes': product,
            'weight': random.randint(1, 5),  # Quantity
            'role': 'product',
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 365))
        })
    
    # Generate reviews
    reviews = []
    for i in range(num_reviews):
        review_id = f"review_{i:05d}"
        customer = random.choice(customers)
        product = random.choice(products)
        
        reviews.append({
            'edges': review_id,
            'nodes': customer,
            'weight': random.randint(1, 5),  # Star rating
            'role': 'reviewer',
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 365))
        })
        
        reviews.append({
            'edges': review_id,
            'nodes': product,
            'weight': random.randint(1, 5),
            'role': 'reviewed_product',
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 365))
        })
    
    # Generate wishlists
    wishlists = []
    for i in range(num_wishlists):
        wishlist_id = f"wishlist_{i:05d}"
        customer = random.choice(customers)
        product = random.choice(products)
        
        wishlists.append({
            'edges': wishlist_id,
            'nodes': customer,
            'weight': 1.0,
            'role': 'wisher',
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 365))
        })
        
        wishlists.append({
            'edges': wishlist_id,
            'nodes': product,
            'weight': 1.0,
            'role': 'wished_product',
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 365))
        })
    
    # Generate returns
    returns = []
    for i in range(num_returns):
        return_id = f"return_{i:05d}"
        customer = random.choice(customers)
        product = random.choice(products)
        
        returns.append({
            'edges': return_id,
            'nodes': customer,
            'weight': random.uniform(10, 500),
            'role': 'returner',
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 180))
        })
        
        returns.append({
            'edges': return_id,
            'nodes': product,
            'weight': random.randint(1, 3),
            'role': 'returned_product',
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 180))
        })
    
    return {
        'purchases': pl.DataFrame(purchases),
        'reviews': pl.DataFrame(reviews),
        'wishlists': pl.DataFrame(wishlists),
        'returns': pl.DataFrame(returns)
    }


def demo_basic_multimodal_construction():
    """Demo 1: Basic multi-modal hypergraph construction"""
    
    print("\n" + "="*70)
    print("🔍 Demo 1: Basic Multi-Modal Hypergraph Construction")
    print("="*70)
    
    # Generate data
    data = generate_ecommerce_data()
    
    # Create hypergraphs for each modality
    print("\n📦 Creating Hypergraphs for Each Modality...")
    if ANANT_AVAILABLE:
        purchase_hg = Hypergraph(data['purchases'])
        review_hg = Hypergraph(data['reviews'])
        wishlist_hg = Hypergraph(data['wishlists'])
        return_hg = Hypergraph(data['returns'])
    else:
        purchase_hg = MockHypergraph(data['purchases'])
        review_hg = MockHypergraph(data['reviews'])
        wishlist_hg = MockHypergraph(data['wishlists'])
        return_hg = MockHypergraph(data['returns'])
    
    print("   ✅ Purchases hypergraph created")
    print("   ✅ Reviews hypergraph created")
    print("   ✅ Wishlists hypergraph created")
    print("   ✅ Returns hypergraph created")
    
    # Create multi-modal hypergraph
    print("\n🔗 Constructing Multi-Modal Hypergraph...")
    mmhg = MultiModalHypergraph(name="ecommerce_customer_behavior")
    
    # Add modalities with different weights
    mmhg.add_modality(
        "purchases",
        purchase_hg,
        weight=3.0,  # Highest weight - most important signal
        description="Customer purchase transactions"
    )
    
    mmhg.add_modality(
        "reviews",
        review_hg,
        weight=2.0,  # Important but less than purchases
        description="Customer product reviews"
    )
    
    mmhg.add_modality(
        "wishlists",
        wishlist_hg,
        weight=1.0,  # Intent signal
        description="Customer wishlist additions"
    )
    
    mmhg.add_modality(
        "returns",
        return_hg,
        weight=1.5,  # Negative signal
        description="Product returns"
    )
    
    print("   ✅ Multi-modal hypergraph constructed")
    
    # Generate summary
    print("\n📊 Multi-Modal Hypergraph Summary:")
    summary = mmhg.generate_summary()
    print(f"   Name: {summary['name']}")
    print(f"   Number of Modalities: {summary['num_modalities']}")
    print(f"   Total Unique Entities: {summary['total_unique_entities']}")
    print(f"   Avg Modalities per Entity: {summary['avg_modalities_per_entity']:.2f}")
    
    print("\n   Modality Details:")
    for mod_name, stats in summary['modality_stats'].items():
        print(f"      {mod_name}: {stats['num_nodes']} nodes, weight={stats['weight']}")
    
    print("\n   Modal Participation Distribution:")
    for num_modalities, count in sorted(summary['modal_participation_distribution'].items()):
        print(f"      {count} entities in {num_modalities} modality(ies)")
    
    return mmhg


def demo_modal_bridges(mmhg):
    """Demo 2: Finding entities that bridge multiple modalities"""
    
    print("\n" + "="*70)
    print("🔍 Demo 2: Finding Modal Bridges")
    print("="*70)
    
    print("\n🌉 Finding entities active across multiple modalities...")
    
    # Find entities in 2+ modalities
    start_time = time.time()
    bridges_2plus = mmhg.find_modal_bridges(min_modalities=2)
    elapsed = time.time() - start_time
    
    print(f"\n📊 Entities in 2+ Modalities:")
    print(f"   Found: {len(bridges_2plus)} entities")
    print(f"   Execution Time: {elapsed:.3f}s")
    
    # Sample a few
    sample_bridges = list(bridges_2plus.items())[:5]
    print("\n   Sample Bridges:")
    for entity_id, modalities in sample_bridges:
        print(f"      {entity_id}: {', '.join(modalities)}")
    
    # Find entities in 3+ modalities (highly engaged)
    bridges_3plus = mmhg.find_modal_bridges(min_modalities=3)
    print(f"\n📊 Highly Engaged Entities (3+ Modalities):")
    print(f"   Found: {len(bridges_3plus)} entities")
    
    # Find entities in all 4 modalities (super engaged)
    bridges_4 = mmhg.find_modal_bridges(min_modalities=4)
    print(f"\n📊 Super Engaged Entities (All 4 Modalities):")
    print(f"   Found: {len(bridges_4)} entities")
    
    if bridges_4:
        print("\n   Sample Super Engaged Customers:")
        for entity_id in list(bridges_4.keys())[:3]:
            print(f"      {entity_id}: purchases, reviews, wishlists, returns")


def demo_cross_modal_patterns(mmhg):
    """Demo 3: Detecting cross-modal patterns"""
    
    print("\n" + "="*70)
    print("🔍 Demo 3: Cross-Modal Pattern Detection")
    print("="*70)
    
    print("\n🔎 Detecting patterns across modalities...")
    
    start_time = time.time()
    patterns = mmhg.detect_cross_modal_patterns(min_support=5)
    elapsed = time.time() - start_time
    
    print(f"\n📊 Pattern Detection Results:")
    print(f"   Patterns Found: {len(patterns)}")
    print(f"   Execution Time: {elapsed:.3f}s")
    
    print("\n   Detected Patterns:")
    for i, pattern in enumerate(patterns, 1):
        print(f"\n   Pattern {i}: {pattern['type']}")
        print(f"      Description: {pattern['description']}")
        print(f"      Support: {pattern['support']}")
        if 'modalities' in pattern:
            print(f"      Modalities: {', '.join(pattern['modalities'])}")


def demo_cross_modal_centrality(mmhg):
    """Demo 4: Computing cross-modal centrality"""
    
    print("\n" + "="*70)
    print("🔍 Demo 4: Cross-Modal Centrality Analysis")
    print("="*70)
    
    # Get sample entities that bridge modalities
    bridges = mmhg.find_modal_bridges(min_modalities=2)
    sample_entities = list(bridges.keys())[:5]
    
    print("\n📊 Computing Cross-Modal Centrality...")
    
    for entity_id in sample_entities:
        centrality = mmhg.compute_cross_modal_centrality(
            node_id=entity_id,
            metric="degree",
            aggregation="weighted_average"
        )
        
        print(f"\n   Entity: {entity_id}")
        print(f"      Aggregated Centrality: {centrality['aggregated']:.2f}")
        print(f"      Per-Modality Scores:")
        for mod, score in centrality['per_modality'].items():
            weight = mmhg.modality_configs[mod].weight
            print(f"         {mod}: {score:.2f} (weight={weight})")


def demo_inter_modal_relationships(mmhg):
    """Demo 5: Discovering inter-modal relationships"""
    
    print("\n" + "="*70)
    print("🔍 Demo 5: Inter-Modal Relationship Discovery")
    print("="*70)
    
    # Find relationships between purchases and reviews
    print("\n🔗 Finding relationships between Purchases and Reviews...")
    
    start_time = time.time()
    relationships = mmhg.discover_inter_modal_relationships(
        source_modality="purchases",
        target_modality="reviews"
    )
    elapsed = time.time() - start_time
    
    print(f"\n📊 Purchase-Review Relationships:")
    print(f"   Found: {len(relationships)} entities active in both")
    print(f"   Execution Time: {elapsed:.3f}s")
    
    # Sample relationships
    sample_rels = relationships[:5]
    print("\n   Sample Relationships:")
    for rel in sample_rels:
        print(f"      {rel['node_id']}: "
              f"purchases={rel['source_degree']}, reviews={rel['target_degree']}")
    
    # Find relationships between wishlists and purchases
    print("\n🔗 Finding relationships between Wishlists and Purchases...")
    
    wishlist_purchase_rels = mmhg.discover_inter_modal_relationships(
        source_modality="wishlists",
        target_modality="purchases"
    )
    
    print(f"\n📊 Wishlist-Purchase Conversion:")
    print(f"   Entities in both: {len(wishlist_purchase_rels)}")
    
    # Entities only in wishlists (intent but no purchase)
    wishlist_only = mmhg.find_modal_bridges(min_modalities=1)
    # Filter to only wishlist
    # This is a simplified version - full implementation would need more logic
    print(f"   Note: Full 'wishlist-only' analysis requires additional filtering")


def demo_modal_correlation(mmhg):
    """Demo 6: Computing modal correlations"""
    
    print("\n" + "="*70)
    print("🔍 Demo 6: Modal Correlation Analysis")
    print("="*70)
    
    print("\n📊 Computing Correlations Between Modalities...")
    
    modalities = mmhg.list_modalities()
    
    # Compute pairwise correlations
    correlations = []
    for i, mod1 in enumerate(modalities):
        for mod2 in modalities[i+1:]:
            corr = mmhg.compute_modal_correlation(mod1, mod2, method="jaccard")
            correlations.append((mod1, mod2, corr))
    
    # Sort by correlation
    correlations.sort(key=lambda x: x[2], reverse=True)
    
    print("\n   Modality Correlations (Jaccard):")
    for mod1, mod2, corr in correlations:
        print(f"      {mod1} ↔ {mod2}: {corr:.3f}")
    
    # Interpret correlations
    print("\n   Interpretation:")
    highest = correlations[0]
    print(f"      Highest correlation: {highest[0]} ↔ {highest[1]} ({highest[2]:.3f})")
    print(f"      This suggests strong overlap in entities between these modalities")


def demo_business_insights(mmhg):
    """Demo 7: Deriving business insights"""
    
    print("\n" + "="*70)
    print("🔍 Demo 7: Business Insights from Multi-Modal Analysis")
    print("="*70)
    
    print("\n💡 Deriving Actionable Business Insights...")
    
    # Insight 1: Customer engagement levels
    bridges = mmhg.find_modal_bridges(min_modalities=1)
    bridges_2plus = mmhg.find_modal_bridges(min_modalities=2)
    bridges_3plus = mmhg.find_modal_bridges(min_modalities=3)
    bridges_4 = mmhg.find_modal_bridges(min_modalities=4)
    
    print("\n   Customer Engagement Segmentation:")
    print(f"      Total Customers: {len(bridges)}")
    print(f"      Low Engagement (1 modality): {len(bridges) - len(bridges_2plus)}")
    print(f"      Medium Engagement (2 modalities): {len(bridges_2plus) - len(bridges_3plus)}")
    print(f"      High Engagement (3 modalities): {len(bridges_3plus) - len(bridges_4)}")
    print(f"      Super Engaged (4 modalities): {len(bridges_4)}")
    
    # Insight 2: Modal correlations indicate behavior patterns
    purchase_review_corr = mmhg.compute_modal_correlation("purchases", "reviews")
    wishlist_purchase_corr = mmhg.compute_modal_correlation("wishlists", "purchases")
    
    print("\n   Behavioral Insights:")
    print(f"      Purchase-Review Correlation: {purchase_review_corr:.3f}")
    if purchase_review_corr < 0.3:
        print("      → Action: Incentivize reviews after purchase")
    
    print(f"      Wishlist-Purchase Conversion: {wishlist_purchase_corr:.3f}")
    if wishlist_purchase_corr < 0.4:
        print("      → Action: Send promotions for wishlisted items")
    
    # Insight 3: Return patterns
    purchase_return_corr = mmhg.compute_modal_correlation("purchases", "returns")
    print(f"      Purchase-Return Correlation: {purchase_return_corr:.3f}")
    if purchase_return_corr > 0.2:
        print("      → Warning: High return rate - investigate product quality")


def main():
    """Run all demos"""
    
    print("\n" + "="*70)
    print("🎯 E-Commerce Multi-Modal Analysis Demo")
    print("="*70)
    print("\nThis demo shows how to analyze customer behavior across multiple")
    print("relationship types (modalities) to gain cross-domain insights.\n")
    
    # Run demos
    mmhg = demo_basic_multimodal_construction()
    demo_modal_bridges(mmhg)
    demo_cross_modal_patterns(mmhg)
    demo_cross_modal_centrality(mmhg)
    demo_inter_modal_relationships(mmhg)
    demo_modal_correlation(mmhg)
    demo_business_insights(mmhg)
    
    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70)
    print("\n💡 Key Takeaways:")
    print("   • Multi-modal analysis reveals insights not visible in single modalities")
    print("   • Modal bridges identify most engaged entities")
    print("   • Cross-modal patterns detect behavioral trends")
    print("   • Inter-modal relationships enable conversion analysis")
    print("   • Modal correlations guide business strategy\n")


if __name__ == "__main__":
    main()
