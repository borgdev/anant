"""
Simple Multi-Modal Analysis Example
====================================

Quick start guide for multi-modal relationship analysis.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import polars as pl
except ImportError:
    print("Please install polars: pip install polars")
    sys.exit(1)

# Import multi-modal components
from core.multi_modal_hypergraph import MultiModalHypergraph

# Try to import Anant core
try:
    from anant import Hypergraph
    ANANT_AVAILABLE = True
except ImportError:
    ANANT_AVAILABLE = False
    # Use mock for demo
    class Hypergraph:
        def __init__(self, data):
            self.data = data
            self.incidences = type('obj', (object,), {'data': data})()
            if 'nodes' in data.columns:
                self._nodes = set(data['nodes'].unique())
        def nodes(self):
            return self._nodes


def main():
    """Simple multi-modal analysis example"""
    
    print("\n" + "="*60)
    print("Simple Multi-Modal Analysis Example")
    print("="*60 + "\n")
    
    # Step 1: Create sample data for different modalities
    print("Step 1: Creating sample data...")
    
    # Social network: Friendship relationships
    friendships = pl.DataFrame([
        {"edges": "f1", "nodes": "Alice", "weight": 1.0},
        {"edges": "f1", "nodes": "Bob", "weight": 1.0},
        {"edges": "f2", "nodes": "Bob", "weight": 1.0},
        {"edges": "f2", "nodes": "Charlie", "weight": 1.0},
        {"edges": "f3", "nodes": "Alice", "weight": 1.0},
        {"edges": "f3", "nodes": "Charlie", "weight": 1.0},
    ])
    
    # Collaboration: Work collaborations
    collaborations = pl.DataFrame([
        {"edges": "c1", "nodes": "Alice", "weight": 1.0},
        {"edges": "c1", "nodes": "Bob", "weight": 1.0},
        {"edges": "c2", "nodes": "Alice", "weight": 1.0},
        {"edges": "c2", "nodes": "David", "weight": 1.0},
    ])
    
    # Communication: Message exchanges
    communications = pl.DataFrame([
        {"edges": "m1", "nodes": "Alice", "weight": 1.0},
        {"edges": "m1", "nodes": "Bob", "weight": 1.0},
        {"edges": "m2", "nodes": "Bob", "weight": 1.0},
        {"edges": "m2", "nodes": "Charlie", "weight": 1.0},
        {"edges": "m3", "nodes": "Charlie", "weight": 1.0},
        {"edges": "m3", "nodes": "David", "weight": 1.0},
    ])
    
    print("   ✓ Created friendship data")
    print("   ✓ Created collaboration data")
    print("   ✓ Created communication data\n")
    
    # Step 2: Create hypergraphs for each modality
    print("Step 2: Creating hypergraphs...")
    
    friendship_hg = Hypergraph(friendships)
    collaboration_hg = Hypergraph(collaborations)
    communication_hg = Hypergraph(communications)
    
    print("   ✓ Friendship hypergraph created")
    print("   ✓ Collaboration hypergraph created")
    print("   ✓ Communication hypergraph created\n")
    
    # Step 3: Create multi-modal hypergraph
    print("Step 3: Building multi-modal hypergraph...")
    
    mmhg = MultiModalHypergraph(name="social_network")
    
    mmhg.add_modality("friendships", friendship_hg, weight=1.0)
    mmhg.add_modality("collaborations", collaboration_hg, weight=2.0)
    mmhg.add_modality("communications", communication_hg, weight=1.5)
    
    print("   ✓ Multi-modal hypergraph created\n")
    
    # Step 4: Analyze the multi-modal structure
    print("Step 4: Analyzing multi-modal structure...\n")
    
    # Get summary
    summary = mmhg.generate_summary()
    print(f"   Total Modalities: {summary['num_modalities']}")
    print(f"   Unique People: {summary['total_unique_entities']}")
    print(f"   Avg Modalities per Person: {summary['avg_modalities_per_entity']:.2f}\n")
    
    # Step 5: Find modal bridges
    print("Step 5: Finding people active in multiple modalities...\n")
    
    bridges = mmhg.find_modal_bridges(min_modalities=2)
    
    print(f"   People in 2+ modalities: {len(bridges)}")
    for person, modalities in bridges.items():
        print(f"      {person}: {', '.join(modalities)}")
    print()
    
    # Step 6: Compute cross-modal centrality
    print("Step 6: Computing cross-modal centrality...\n")
    
    for person in ["Alice", "Bob", "Charlie"]:
        centrality = mmhg.compute_cross_modal_centrality(
            node_id=person,
            metric="degree",
            aggregation="weighted_average"
        )
        
        print(f"   {person}:")
        print(f"      Overall Centrality: {centrality['aggregated']:.2f}")
        print(f"      By Modality:")
        for mod, score in centrality['per_modality'].items():
            print(f"         {mod}: {score:.0f}")
        print()
    
    # Step 7: Discover inter-modal relationships
    print("Step 7: Discovering inter-modal relationships...\n")
    
    relationships = mmhg.discover_inter_modal_relationships(
        source_modality="friendships",
        target_modality="collaborations"
    )
    
    print(f"   People who are both friends AND collaborators: {len(relationships)}")
    for rel in relationships:
        print(f"      {rel['node_id']}")
    print()
    
    # Step 8: Compute modal correlation
    print("Step 8: Computing modal correlation...\n")
    
    corr = mmhg.compute_modal_correlation(
        modality_a="friendships",
        modality_b="collaborations",
        method="jaccard"
    )
    
    print(f"   Friendship-Collaboration Overlap: {corr:.2%}")
    print(f"   Interpretation: {corr*100:.0f}% of people appear in both modalities\n")
    
    # Step 9: Detect cross-modal patterns
    print("Step 9: Detecting cross-modal patterns...\n")
    
    patterns = mmhg.detect_cross_modal_patterns(min_support=1)
    
    print(f"   Patterns Found: {len(patterns)}")
    for i, pattern in enumerate(patterns, 1):
        print(f"\n   Pattern {i}:")
        print(f"      Type: {pattern['type']}")
        print(f"      Description: {pattern['description']}")
        print(f"      Support: {pattern['support']}")
    print()
    
    # Summary
    print("="*60)
    print("Analysis Complete!")
    print("="*60 + "\n")
    
    print("Key Insights:")
    print("   • Alice is highly connected across all modalities")
    print("   • Friendship and collaboration networks overlap")
    print("   • Multi-modal analysis reveals hidden patterns")
    print("   • Cross-domain insights enable better understanding\n")


if __name__ == "__main__":
    main()
