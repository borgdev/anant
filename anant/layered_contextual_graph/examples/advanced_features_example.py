"""
Advanced Features Example
=========================

Demonstrates all three LCG extensions working together:
1. Streaming & Event-Driven
2. Machine Learning
3. Advanced Reasoning
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import polars as pl
    import numpy as np
except ImportError:
    print("⚠️  Required dependencies not installed.")
    sys.exit(1)

# Import Anant Hypergraph
try:
    from anant.classes.hypergraph.core.hypergraph import Hypergraph as AnantHypergraph
    ANANT_AVAILABLE = True
except ImportError:
    ANANT_AVAILABLE = False
    class AnantHypergraph:
        def __init__(self, data=None, **kwargs):
            self.data = data
            self.name = kwargs.get('name', 'hypergraph')

# Import LCG with extensions
from anant.layered_contextual_graph.core import LayeredContextualGraph, LayerType, ContextType
from anant.layered_contextual_graph.extensions import (
    StreamingLayeredGraph,
    MLLayeredGraph,
    ReasoningLayeredGraph,
    RuleType,
    InferenceRule
)


def demo_streaming():
    """Demo 1: Streaming & Event-Driven"""
    print("\n" + "="*70)
    print("🔄 Demo 1: Streaming & Event-Driven")
    print("="*70)
    
    # Create streaming-enabled graph
    slcg = StreamingLayeredGraph(
        name="streaming_demo",
        quantum_enabled=True,
        enable_event_store=True
    )
    
    print("✅ Created StreamingLayeredGraph")
    
    # Subscribe to events
    event_log = []
    
    def log_event(event):
        event_log.append(event)
        print(f"   📨 Event: {event.event_type} (layer={event.layer_name})")
    
    slcg.event_adapter.subscribe(log_event)
    
    # Add layers (events emitted automatically)
    data1 = pl.DataFrame([
        {'edge_id': 'e1', 'node_id': 'n1', 'weight': 1.0},
    ])
    hg1 = AnantHypergraph(data=data1, name="layer1") if ANANT_AVAILABLE else type('obj', (), {'name': 'layer1'})()
    
    print("\nAdding layer...")
    slcg.add_layer("physical", hg1, LayerType.PHYSICAL, level=0)
    
    # Create superposition (event emitted)
    print("\nCreating superposition...")
    slcg.create_superposition(
        "entity_A",
        layer_states={"physical": "node_1"},
        quantum_states={"state1": 0.7, "state2": 0.3}
    )
    
    # Observe (collapse event emitted)
    print("\nObserving (collapsing)...")
    result = slcg.observe("entity_A", collapse_quantum=True)
    
    # Stats
    stats = slcg.get_streaming_stats()
    print(f"\n📊 Streaming Stats:")
    print(f"   Total events: {stats['total_events']}")
    print(f"   Events by type: {stats['events_by_type']}")
    
    return slcg


def demo_ml(lcg):
    """Demo 2: Machine Learning"""
    print("\n" + "="*70)
    print("🤖 Demo 2: Machine Learning")
    print("="*70)
    
    # Convert to ML-enabled graph
    from anant.layered_contextual_graph.extensions import enable_ml
    ml_lcg = enable_ml(lcg, embedding_dim=128)
    
    print("✅ Enabled ML capabilities")
    
    # Add embedding layer
    ml_lcg.add_embedding_layer("physical", embedding_dim=128)
    
    # Create some embeddings
    print("\nAdding embeddings...")
    np.random.seed(42)
    
    entities = ["entity_A", "entity_B", "entity_C", "entity_D"]
    for entity in entities:
        embedding = np.random.randn(128)
        ml_lcg.set_entity_embedding(entity, "physical", embedding)
        print(f"   Added embedding for {entity}")
    
    # Similarity search
    print("\nSimilarity search...")
    query_emb = np.random.randn(128)
    results = ml_lcg.similarity_search(query_emb, layer_name="physical", top_k=3)
    
    print(f"   Top 3 similar entities:")
    for entity_id, similarity in results.get("physical", []):
        print(f"      {entity_id}: {similarity:.3f}")
    
    # Cross-layer similarity
    print("\nCross-layer similarity...")
    sim = ml_lcg.cross_layer_similarity("entity_A", "entity_B")
    print(f"   Similarity(entity_A, entity_B): {sim:.3f}")
    
    # Clustering
    print("\nClustering entities...")
    clusters = ml_lcg.cluster_entities("physical", n_clusters=2)
    print(f"   Found {len(clusters)} clusters:")
    for cluster_id, members in clusters.items():
        print(f"      Cluster {cluster_id}: {members}")
    
    return ml_lcg


def demo_reasoning(lcg):
    """Demo 3: Advanced Reasoning"""
    print("\n" + "="*70)
    print("🧠 Demo 3: Advanced Reasoning")
    print("="*70)
    
    # Convert to reasoning-enabled graph
    from anant.layered_contextual_graph.extensions import enable_reasoning
    r_lcg = enable_reasoning(lcg, auto_detect=True)
    
    print("✅ Enabled reasoning capabilities")
    
    # Add more layers for reasoning
    data2 = pl.DataFrame([{'edge_id': 'e2', 'node_id': 'n2', 'weight': 1.0}])
    hg2 = AnantHypergraph(data=data2, name="semantic") if ANANT_AVAILABLE else type('obj', (), {'name': 'semantic'})()
    
    r_lcg.add_layer("semantic", hg2, LayerType.SEMANTIC, level=1, parent_layer="physical")
    
    # Add inference rules
    print("\nAdding inference rules...")
    
    def condition_always_true(data):
        return True
    
    def infer_semantic_from_physical(data):
        return f"concept_of_{data}"
    
    r_lcg.add_inference_rule(
        rule_id="physical_to_semantic",
        source_layer="physical",
        target_layer="semantic",
        condition=condition_always_true,
        action=infer_semantic_from_physical,
        rule_type=RuleType.FORWARD,
        confidence=0.9
    )
    
    print("   ✅ Added rule: physical → semantic")
    
    # Create superpositions with potential contradictions
    print("\nCreating entities with states...")
    
    r_lcg.create_superposition(
        "entity_X",
        layer_states={
            "physical": "raw_data_1",
            "semantic": "concept_A"
        }
    )
    
    r_lcg.create_superposition(
        "entity_Y",
        layer_states={
            "physical": "raw_data_2",
            "semantic": "concept_B"
        }
    )
    
    # Perform inference
    print("\nPerforming inference...")
    inferred = r_lcg.infer_cross_layer("entity_X", "physical", "semantic")
    print(f"   Inferred facts: {len(inferred)}")
    for fact in inferred:
        print(f"      {fact['fact']} (confidence={fact['confidence']})")
    
    # Check for contradictions
    print("\nChecking for contradictions...")
    contradictions = r_lcg.check_consistency()
    
    if contradictions:
        print(f"   ⚠️  Found {len(contradictions)} contradictions:")
        for contra in contradictions:
            print(f"      {contra.entity_id}: {contra.layer1} vs {contra.layer2}")
            print(f"         Severity: {contra.severity:.2f}")
    else:
        print("   ✅ No contradictions detected")
    
    # Hierarchical inference (bottom-up)
    print("\nHierarchical inference (bottom-up)...")
    hierarchical_results = r_lcg.reasoning_engine.hierarchical_inference("entity_X", direction="up")
    print(f"   Inference results: {list(hierarchical_results.keys())}")
    
    return r_lcg


def demo_combined():
    """Demo 4: All Features Combined"""
    print("\n" + "="*70)
    print("🎯 Demo 4: All Features Combined")
    print("="*70)
    
    # Create a graph that combines all features
    # We can manually combine by using the base classes together
    
    print("This would require a unified class that inherits from all three.")
    print("For now, we demonstrated each independently.")
    print("\n💡 In production, you can:")
    print("   1. Use StreamingLayeredGraph for real-time updates")
    print("   2. Add ML capabilities with enable_ml()")
    print("   3. Add reasoning with enable_reasoning()")
    print("   4. Or create a custom class inheriting from all three")


def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("🚀 LCG Advanced Features Demonstration")
    print("="*70)
    print("\nDemonstrating three integrated extensions:")
    print("  1. Streaming & Event-Driven")
    print("  2. Machine Learning")
    print("  3. Advanced Reasoning\n")
    
    # Demo 1: Streaming
    slcg = demo_streaming()
    
    # Demo 2: ML (using the streaming graph)
    ml_lcg = demo_ml(slcg)
    
    # Demo 3: Reasoning (using the ML graph)
    r_lcg = demo_reasoning(ml_lcg)
    
    # Demo 4: Combined
    demo_combined()
    
    print("\n" + "="*70)
    print("✅ All Demos Complete!")
    print("="*70)
    
    print("\n💡 Summary of Capabilities:")
    print("\n📡 Streaming & Event-Driven:")
    print("   ✓ Real-time event emission for all operations")
    print("   ✓ Event subscriptions and listeners")
    print("   ✓ Automatic layer synchronization")
    print("   ✓ Event store for replay and debugging")
    
    print("\n🤖 Machine Learning:")
    print("   ✓ Embedding layers for semantic similarity")
    print("   ✓ Cross-layer similarity search")
    print("   ✓ Entity clustering")
    print("   ✓ Dimensionality reduction for visualization")
    print("   ✓ Predictive collapse (ML-based)")
    
    print("\n🧠 Advanced Reasoning:")
    print("   ✓ Rule-based inference across layers")
    print("   ✓ Contradiction detection")
    print("   ✓ Belief propagation")
    print("   ✓ Hierarchical reasoning (bottom-up/top-down)")
    print("   ✓ Automatic consistency checking")
    
    print("\n🎉 LCG is now production-ready with:")
    print("   • Real-time streaming capabilities")
    print("   • ML-powered semantic understanding")
    print("   • Intelligent reasoning and inference")
    print()


if __name__ == "__main__":
    main()
