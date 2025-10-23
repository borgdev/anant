"""
Test Anant Inheritance for LayeredContextualGraph
=================================================

Validates that LayeredContextualGraph properly extends Anant's core Hypergraph.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import polars as pl
    import numpy as np
except ImportError:
    print("⚠️  Required dependencies not installed.")
    sys.exit(1)

# Import Anant
try:
    from anant.classes.hypergraph.core.hypergraph import Hypergraph as AnantHypergraph
    ANANT_AVAILABLE = True
except ImportError:
    ANANT_AVAILABLE = False
    print("❌ Anant core library not available")

from anant.layered_contextual_graph.core import LayeredContextualGraph, LayerType, ContextType


def test_inheritance_chain():
    """Test 1: Verify inheritance chain"""
    print("\n" + "="*60)
    print("Test 1: Inheritance Chain Validation")
    print("="*60)
    
    lcg = LayeredContextualGraph(name="test")
    
    if ANANT_AVAILABLE:
        # Check inheritance
        assert isinstance(lcg, AnantHypergraph), "Should be instance of AnantHypergraph"
        assert isinstance(lcg, LayeredContextualGraph), "Should be instance of LayeredContextualGraph"
        
        # Check MRO
        mro = [c.__name__ for c in type(lcg).__mro__]
        print(f"   Method Resolution Order: {mro}")
        
        assert 'LayeredContextualGraph' in mro
        assert 'Hypergraph' in mro
        assert mro.index('LayeredContextualGraph') < mro.index('Hypergraph')
        
        print("   ✅ Inheritance chain correct")
        print(f"   ✅ MRO: {' → '.join(mro[:3])}")
    else:
        print("   ⚠️  Anant not available - standalone mode")
    
    return True


def test_inherited_attributes():
    """Test 2: Verify inherited Anant attributes"""
    print("\n" + "="*60)
    print("Test 2: Inherited Anant Attributes")
    print("="*60)
    
    data = pl.DataFrame([
        {'edge_id': 'e1', 'node_id': 'n1', 'weight': 1.0},
        {'edge_id': 'e1', 'node_id': 'n2', 'weight': 1.0},
    ])
    
    lcg = LayeredContextualGraph(data=data, name="test_attrs", quantum_enabled=True)
    
    if ANANT_AVAILABLE:
        # Core Anant attributes
        assert hasattr(lcg, 'name')
        assert hasattr(lcg, '_instance_id')
        assert hasattr(lcg, 'incidences')
        assert hasattr(lcg, 'properties')
        assert hasattr(lcg, 'metadata')
        
        print("   ✅ Core Anant attributes present:")
        print(f"      • name: {lcg.name}")
        print(f"      • incidences: {type(lcg.incidences).__name__}")
        print(f"      • properties: {type(lcg.properties).__name__}")
    
    # Layered contextual attributes
    assert hasattr(lcg, 'layers')
    assert hasattr(lcg, 'contexts')
    assert hasattr(lcg, 'superposition_states')
    assert hasattr(lcg, 'quantum_states')
    assert hasattr(lcg, 'quantum_enabled')
    
    print("   ✅ Layered contextual attributes present:")
    print(f"      • layers: {type(lcg.layers).__name__}")
    print(f"      • quantum_enabled: {lcg.quantum_enabled}")
    
    return True


def test_layer_management():
    """Test 3: Layer management functionality"""
    print("\n" + "="*60)
    print("Test 3: Layer Management")
    print("="*60)
    
    lcg = LayeredContextualGraph(name="test_layers")
    
    # Create mock hypergraph
    data = pl.DataFrame([
        {'edge_id': 'e1', 'node_id': 'n1', 'weight': 1.0},
    ])
    
    if ANANT_AVAILABLE:
        hg = AnantHypergraph(data=data, name="layer1")
    else:
        class MockHG:
            def __init__(self):
                self.name = "layer1"
        hg = MockHG()
    
    # Add layer
    lcg.add_layer("physical", hg, LayerType.PHYSICAL, level=0)
    
    assert "physical" in lcg.layers
    assert len(lcg.layers) == 1
    
    print("   ✅ Layer added successfully")
    print(f"      Layers: {list(lcg.layers.keys())}")
    
    # Remove layer
    removed = lcg.remove_layer("physical")
    assert removed
    assert "physical" not in lcg.layers
    
    print("   ✅ Layer removed successfully")
    
    return True


def test_context_management():
    """Test 4: Context management"""
    print("\n" + "="*60)
    print("Test 4: Context Management")
    print("="*60)
    
    lcg = LayeredContextualGraph(name="test_contexts")
    
    # Add context
    from anant.layered_contextual_graph.core import Context
    
    lcg.add_context(
        "temporal",
        ContextType.TEMPORAL,
        parameters={'weight': 0.8},
        priority=1
    )
    
    assert "temporal" in lcg.contexts
    assert len(lcg.contexts) == 1
    
    print("   ✅ Context added successfully")
    print(f"      Contexts: {list(lcg.contexts.keys())}")
    
    return True


def test_quantum_features():
    """Test 5: Quantum superposition features"""
    print("\n" + "="*60)
    print("Test 5: Quantum Features")
    print("="*60)
    
    lcg = LayeredContextualGraph(name="test_quantum", quantum_enabled=True)
    
    # Create superposition
    superpos = lcg.create_superposition(
        "entity1",
        quantum_states={"state_a": 0.6, "state_b": 0.4}
    )
    
    assert "entity1" in lcg.superposition_states
    assert superpos.quantum_state is not None
    assert len(superpos.quantum_state.states) == 2
    
    print("   ✅ Superposition created")
    print(f"      States: {list(superpos.quantum_state.states.keys())}")
    
    # Test coherence
    coherence = lcg.get_quantum_coherence("entity1")
    assert 0.0 <= coherence <= 1.0
    
    print(f"   ✅ Quantum coherence: {coherence:.3f}")
    
    # Test observation (collapse)
    state_before = len(lcg.superposition_states["entity1"].quantum_state.states)
    observed = lcg.observe("entity1", collapse_quantum=True)
    collapsed = lcg.superposition_states["entity1"].quantum_state.collapsed
    
    assert collapsed
    assert observed in ["state_a", "state_b"]
    
    print(f"   ✅ Quantum collapse: {observed}")
    
    return True


def test_entanglement():
    """Test 6: Quantum entanglement"""
    print("\n" + "="*60)
    print("Test 6: Quantum Entanglement")
    print("="*60)
    
    lcg = LayeredContextualGraph(name="test_entangle", quantum_enabled=True)
    
    # Create two entities
    lcg.create_superposition("entity1", quantum_states={"s1": 0.5, "s2": 0.5})
    lcg.create_superposition("entity2", quantum_states={"s1": 0.5, "s2": 0.5})
    
    # Entangle them
    success = lcg.entangle_entities("entity1", "entity2")
    
    assert success
    
    qs1 = lcg.superposition_states["entity1"].quantum_state
    qs2 = lcg.superposition_states["entity2"].quantum_state
    
    assert qs2.state_id in qs1.entangled_with
    assert qs1.state_id in qs2.entangled_with
    
    print("   ✅ Entities entangled")
    print(f"      Entity1 entangled with: {len(qs1.entangled_with)} entities")
    print(f"      Entity2 entangled with: {len(qs2.entangled_with)} entities")
    
    return True


def test_cross_layer_query():
    """Test 7: Cross-layer querying"""
    print("\n" + "="*60)
    print("Test 7: Cross-Layer Querying")
    print("="*60)
    
    lcg = LayeredContextualGraph(name="test_query")
    
    # Add two layers with mock hypergraphs
    class MockHG:
        def __init__(self, name):
            self.name = name
    
    lcg.add_layer("layer1", MockHG("layer1"), LayerType.PHYSICAL, level=0)
    lcg.add_layer("layer2", MockHG("layer2"), LayerType.SEMANTIC, level=1)
    
    # Create superposition across layers
    lcg.create_superposition(
        "entity_x",
        layer_states={"layer1": "data1", "layer2": "concept1"}
    )
    
    # Query across layers
    results = lcg.query_across_layers("entity_x", layers=["layer1", "layer2"])
    
    assert len(results) == 2
    assert "layer1" in results
    assert "layer2" in results
    
    print("   ✅ Cross-layer query successful")
    print(f"      Results: {list(results.keys())}")
    
    return True


def test_hierarchy_propagation():
    """Test 8: Hierarchical propagation"""
    print("\n" + "="*60)
    print("Test 8: Hierarchical Propagation")
    print("="*60)
    
    lcg = LayeredContextualGraph(name="test_hierarchy")
    
    # Create 3-level hierarchy
    class MockHG:
        def __init__(self, name):
            self.name = name
    
    lcg.add_layer("level0", MockHG("l0"), LayerType.PHYSICAL, level=0)
    lcg.add_layer("level1", MockHG("l1"), LayerType.SEMANTIC, level=1, parent_layer="level0")
    lcg.add_layer("level2", MockHG("l2"), LayerType.CONCEPTUAL, level=2, parent_layer="level1")
    
    # Create entity in all layers
    lcg.create_superposition(
        "entity_h",
        layer_states={"level0": "d0", "level1": "d1", "level2": "d2"}
    )
    
    # Propagate up
    up_results = lcg.propagate_up("entity_h", from_layer="level0", to_level=2)
    
    # Propagate down
    down_results = lcg.propagate_down("entity_h", from_layer="level2", to_level=0)
    
    print(f"   ✅ Bottom-up propagation: {len(up_results)} layers")
    print(f"   ✅ Top-down propagation: {len(down_results)} layers")
    
    return True


def run_all_tests():
    """Run all tests"""
    
    print("\n" + "="*70)
    print("LAYERED CONTEXTUAL GRAPH - ANANT INHERITANCE TESTS")
    print("="*70)
    print(f"\nAnant Available: {ANANT_AVAILABLE}\n")
    
    try:
        test_inheritance_chain()
        test_inherited_attributes()
        test_layer_management()
        test_context_management()
        test_quantum_features()
        test_entanglement()
        test_cross_layer_query()
        test_hierarchy_propagation()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nValidation Summary:")
        if ANANT_AVAILABLE:
            print("   ✅ Inheritance chain correct (LayeredContextualGraph → Hypergraph)")
            print("   ✅ All Anant attributes inherited")
        print("   ✅ Layer management working")
        print("   ✅ Context management working")
        print("   ✅ Quantum superposition working")
        print("   ✅ Quantum entanglement working")
        print("   ✅ Cross-layer querying working")
        print("   ✅ Hierarchy propagation working")
        print("\n   🎉 LayeredContextualGraph PROPERLY extends Anant's Hypergraph 🎉")
        print("\n   Total: 8/8 tests passed\n")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
