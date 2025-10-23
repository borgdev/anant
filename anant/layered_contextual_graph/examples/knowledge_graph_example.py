"""
Layered Contextual Knowledge Graph Example
==========================================

Demonstrates how to build a multi-layered contextual knowledge graph
with quantum superposition capabilities.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

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
    # Use mock
    class AnantHypergraph:
        def __init__(self, data=None, **kwargs):
            self.data = data
            self.name = kwargs.get('name', 'hypergraph')

# Import Layered Contextual Graph
from anant.layered_contextual_graph.core import (
    LayeredContextualGraph,
    LayerType,
    ContextType,
    Context
)


def create_physical_layer():
    """Create physical/data layer (Level 0)"""
    print("Creating physical layer...")
    
    # Raw data: entities and their physical connections
    data = pl.DataFrame([
        {'edge_id': 'doc1', 'node_id': 'word_machine', 'weight': 1.0},
        {'edge_id': 'doc1', 'node_id': 'word_learning', 'weight': 1.0},
        {'edge_id': 'doc2', 'node_id': 'word_deep', 'weight': 1.0},
        {'edge_id': 'doc2', 'node_id': 'word_learning', 'weight': 1.0},
        {'edge_id': 'doc3', 'node_id': 'word_neural', 'weight': 1.0},
        {'edge_id': 'doc3', 'node_id': 'word_network', 'weight': 1.0},
    ])
    
    if ANANT_AVAILABLE:
        hg = AnantHypergraph(data=data, name="physical_layer")
    else:
        hg = AnantHypergraph(data=data, name="physical_layer")
    
    print(f"   ✅ Physical layer created")
    return hg


def create_semantic_layer():
    """Create semantic/meaning layer (Level 1)"""
    print("Creating semantic layer...")
    
    # Semantic relationships: concepts and their meanings
    data = pl.DataFrame([
        {'edge_id': 'concept1', 'node_id': 'concept_AI', 'weight': 1.0},
        {'edge_id': 'concept1', 'node_id': 'concept_ML', 'weight': 1.0},
        {'edge_id': 'concept2', 'node_id': 'concept_ML', 'weight': 1.0},
        {'edge_id': 'concept2', 'node_id': 'concept_DL', 'weight': 1.0},
        {'edge_id': 'concept3', 'node_id': 'concept_DL', 'weight': 1.0},
        {'edge_id': 'concept3', 'node_id': 'concept_NN', 'weight': 1.0},
    ])
    
    if ANANT_AVAILABLE:
        hg = AnantHypergraph(data=data, name="semantic_layer")
    else:
        hg = AnantHypergraph(data=data, name="semantic_layer")
    
    print(f"   ✅ Semantic layer created")
    return hg


def create_conceptual_layer():
    """Create conceptual/abstract layer (Level 2)"""
    print("Creating conceptual layer...")
    
    # Abstract concepts and philosophical relationships
    data = pl.DataFrame([
        {'edge_id': 'abstract1', 'node_id': 'abstract_Intelligence', 'weight': 1.0},
        {'edge_id': 'abstract1', 'node_id': 'abstract_Computation', 'weight': 1.0},
        {'edge_id': 'abstract2', 'node_id': 'abstract_Learning', 'weight': 1.0},
        {'edge_id': 'abstract2', 'node_id': 'abstract_Adaptation', 'weight': 1.0},
    ])
    
    if ANANT_AVAILABLE:
        hg = AnantHypergraph(data=data, name="conceptual_layer")
    else:
        hg = AnantHypergraph(data=data, name="conceptual_layer")
    
    print(f"   ✅ Conceptual layer created")
    return hg


def demo_layered_contextual_graph():
    """Main demonstration"""
    
    print("\n" + "="*70)
    print("🌐 Layered Contextual Knowledge Graph Demo")
    print("="*70)
    print("\nDemonstrating quantum-inspired multi-layered graph with contexts\n")
    
    # Create layered contextual graph
    print("1. Creating Layered Contextual Graph...")
    lcg = LayeredContextualGraph(
        name="knowledge_graph",
        quantum_enabled=True
    )
    print(f"   ✅ Graph created (Anant: {ANANT_AVAILABLE})")
    print(f"   ✅ Quantum features: {'enabled' if lcg.quantum_enabled else 'disabled'}")
    
    # Create layers
    print("\n2. Adding Hierarchical Layers...")
    
    physical_hg = create_physical_layer()
    semantic_hg = create_semantic_layer()
    conceptual_hg = create_conceptual_layer()
    
    lcg.add_layer(
        "physical",
        physical_hg,
        LayerType.PHYSICAL,
        level=0,
        weight=1.0,
        quantum_enabled=True
    )
    
    lcg.add_layer(
        "semantic",
        semantic_hg,
        LayerType.SEMANTIC,
        level=1,
        parent_layer="physical",
        weight=2.0,
        quantum_enabled=True
    )
    
    lcg.add_layer(
        "conceptual",
        conceptual_hg,
        LayerType.CONCEPTUAL,
        level=2,
        parent_layer="semantic",
        weight=1.5,
        quantum_enabled=True
    )
    
    print(f"   ✅ Added 3 hierarchical layers")
    
    # Add contexts
    print("\n3. Adding Contextual Information...")
    
    # Temporal context
    lcg.add_context(
        "recent",
        ContextType.TEMPORAL,
        parameters={'recency_weight': 0.8},
        applicable_layers={"physical", "semantic"},
        priority=1,
        temporal_range=(
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
    )
    
    # Domain context
    lcg.add_context(
        "ai_domain",
        ContextType.DOMAIN,
        parameters={'domain': 'artificial_intelligence'},
        applicable_layers={"semantic", "conceptual"},
        priority=2
    )
    
    print(f"   ✅ Added 2 contexts (temporal, domain)")
    
    # Create superpositions
    print("\n4. Creating Quantum Superpositions...")
    
    # Entity exists in multiple layers simultaneously
    superposition = lcg.create_superposition(
        "entity_ML",
        layer_states={
            "physical": "word_learning",
            "semantic": "concept_ML",
            "conceptual": "abstract_Learning"
        },
        quantum_states={
            "data_level": 0.3,
            "concept_level": 0.5,
            "abstract_level": 0.2
        }
    )
    
    print(f"   ✅ Created superposition for 'entity_ML'")
    print(f"   ✅ Quantum states: {len(superposition.quantum_state.states)}")
    
    # Entangle related entities
    print("\n5. Creating Quantum Entanglement...")
    
    lcg.create_superposition("entity_AI", quantum_states={"state1": 0.6, "state2": 0.4})
    lcg.entangle_entities("entity_ML", "entity_AI", correlation_strength=0.9)
    
    print(f"   ✅ Entangled 'entity_ML' and 'entity_AI'")
    
    # Query across layers
    print("\n6. Querying Across Layers...")
    
    results = lcg.query_across_layers(
        "entity_ML",
        layers=["physical", "semantic", "conceptual"],
        context="ai_domain"
    )
    
    print(f"   Results across layers:")
    for layer_name, result in results.items():
        print(f"      • {layer_name}: {result['state']} (level={result['level']})")
    
    # Observe (quantum collapse)
    print("\n7. Quantum Observation (Collapse)...")
    
    print(f"   Before observation:")
    print(f"      Coherence: {lcg.get_quantum_coherence('entity_ML'):.3f}")
    
    observed_state = lcg.observe("entity_ML", layer="semantic", collapse_quantum=True)
    
    print(f"   After observation:")
    print(f"      Collapsed to: {observed_state}")
    print(f"      Coherence: {lcg.get_quantum_coherence('entity_ML'):.3f}")
    
    # Propagate up hierarchy
    print("\n8. Hierarchical Propagation...")
    
    up_results = lcg.propagate_up("entity_ML", from_layer="physical", to_level=2)
    print(f"   Bottom-up propagation:")
    for layer, state in up_results.items():
        print(f"      {layer}: {state}")
    
    # Generate summary
    print("\n9. Graph Summary...")
    summary = lcg.generate_summary()
    
    print(f"   Name: {summary['name']}")
    print(f"   Layers: {summary['num_layers']}")
    print(f"   Contexts: {summary['num_contexts']}")
    print(f"   Superpositions: {summary['num_superpositions']}")
    print(f"   Quantum states: {summary['num_quantum_states']}")
    print(f"   Quantum enabled: {summary['quantum_enabled']}")
    
    print(f"\n   Layers by level:")
    for level, count in summary['layers_by_level'].items():
        print(f"      Level {level}: {count} layer(s)")
    
    # Quantum-ready features
    print("\n10. Quantum-Ready Features (Future DB)...")
    
    from anant.layered_contextual_graph.quantum import (
        QuantumReadyInterface,
        prepare_for_quantum_db
    )
    
    qi = QuantumReadyInterface(lcg.name)
    
    # Create entanglement circuit
    circuit = qi.create_entanglement_circuit("entity_ML", "entity_AI")
    print(f"   ✅ Entanglement circuit created: {circuit.name}")
    print(f"   ✅ Qubits: {circuit.qubits}, Gates: {len(circuit.gates)}")
    
    # Export for quantum execution
    qiskit_code = circuit.to_qiskit()
    print(f"\n   Generated Qiskit code:")
    for line in qiskit_code.split('\n')[:5]:
        print(f"      {line}")
    print(f"      ...")
    
    # Quantum advantage estimate
    advantage = qi.get_quantum_advantage_estimate("graph_search", size=1000)
    print(f"\n   Quantum advantage estimate:")
    print(f"      Operation: {advantage['operation']}")
    print(f"      Classical: O({advantage['classical_complexity']:.0f})")
    print(f"      Quantum: O({advantage['quantum_complexity']:.0f})")
    print(f"      Speedup: {advantage['speedup_factor']:.1f}x")
    print(f"      Quantum recommended: {advantage['quantum_recommended']}")
    
    print("\n" + "="*70)
    print("✅ Layered Contextual Graph Demo Complete!")
    print("="*70)
    
    print("\n💡 Key Features Demonstrated:")
    print("   ✓ Hierarchical layered architecture")
    print("   ✓ Contextual awareness (temporal, domain)")
    print("   ✓ Quantum superposition of entity states")
    print("   ✓ Quantum entanglement between entities")
    print("   ✓ Cross-layer querying and reasoning")
    print("   ✓ Quantum observation and collapse")
    print("   ✓ Hierarchical propagation (up/down)")
    print("   ✓ Quantum-ready for future quantum DB")
    print("   ✓ Extends Anant's core Hypergraph")


if __name__ == "__main__":
    demo_layered_contextual_graph()
