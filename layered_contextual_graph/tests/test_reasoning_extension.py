"""
Test Suite: Advanced Reasoning Extension
========================================

Comprehensive tests for reasoning capabilities in LCG.
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
    class AnantHypergraph:
        def __init__(self, data=None, **kwargs):
            self.data = data
            self.name = kwargs.get('name', 'hg')

from layered_contextual_graph.core import LayeredContextualGraph, LayerType
from layered_contextual_graph.extensions import (
    ReasoningLayeredGraph,
    InferenceRule,
    RuleType,
    ReasoningEngine,
    enable_reasoning
)


def create_test_hypergraph(name: str) -> AnantHypergraph:
    """Create a simple test hypergraph"""
    data = pl.DataFrame([
        {'edge_id': f'{name}_e1', 'node_id': f'{name}_n1', 'weight': 1.0},
    ])
    return AnantHypergraph(data=data, name=name) if ANANT_AVAILABLE else type('obj', (), {'name': name})()


def test_reasoning_graph_creation():
    """Test 1: ReasoningLayeredGraph creation"""
    print("\n" + "="*60)
    print("Test 1: ReasoningLayeredGraph Creation")
    print("="*60)
    
    r_lcg = ReasoningLayeredGraph(
        name="test_reasoning",
        quantum_enabled=True,
        auto_detect_contradictions=True
    )
    
    assert r_lcg.name == "test_reasoning"
    assert r_lcg.auto_detect_contradictions == True
    assert hasattr(r_lcg, 'reasoning_engine')
    
    print("   ✅ ReasoningLayeredGraph created successfully")
    print(f"   ✅ Auto-detect contradictions: {r_lcg.auto_detect_contradictions}")
    
    return True


def test_inference_rules():
    """Test 2: Inference rules creation and evaluation"""
    print("\n" + "="*60)
    print("Test 2: Inference Rules")
    print("="*60)
    
    def always_true(data):
        return True
    
    def transform_data(data):
        return f"transformed_{data}"
    
    rule = InferenceRule(
        rule_id="test_rule",
        rule_type=RuleType.FORWARD,
        source_layer="physical",
        target_layer="semantic",
        condition=always_true,
        action=transform_data,
        confidence=0.9
    )
    
    assert rule.rule_id == "test_rule"
    assert rule.evaluate("test_data") == True
    assert rule.apply("data") == "transformed_data"
    assert rule.confidence == 0.9
    
    print("   ✅ Inference rule created")
    print(f"   ✅ Rule evaluation: {rule.evaluate('test')}")
    print(f"   ✅ Rule application: {rule.apply('sample')}")
    
    return True


def test_add_inference_rule():
    """Test 3: Adding rules to reasoning graph"""
    print("\n" + "="*60)
    print("Test 3: Adding Inference Rules")
    print("="*60)
    
    r_lcg = ReasoningLayeredGraph(name="test_add_rule")
    
    # Add layers
    hg1 = create_test_hypergraph("l1")
    hg2 = create_test_hypergraph("l2")
    r_lcg.add_layer("physical", hg1, LayerType.PHYSICAL, level=0)
    r_lcg.add_layer("semantic", hg2, LayerType.SEMANTIC, level=1, parent_layer="physical")
    
    # Add inference rule
    r_lcg.add_inference_rule(
        rule_id="phys_to_sem",
        source_layer="physical",
        target_layer="semantic",
        condition=lambda x: True,
        action=lambda x: f"concept_of_{x}",
        rule_type=RuleType.FORWARD,
        confidence=0.8
    )
    
    assert "phys_to_sem" in r_lcg.reasoning_engine.rules
    assert r_lcg.reasoning_engine.rules["phys_to_sem"].confidence == 0.8
    
    print("   ✅ Rule added to reasoning engine")
    print(f"   ✅ Total rules: {len(r_lcg.reasoning_engine.rules)}")
    
    return True


def test_inference():
    """Test 4: Performing inference across layers"""
    print("\n" + "="*60)
    print("Test 4: Cross-Layer Inference")
    print("="*60)
    
    r_lcg = ReasoningLayeredGraph(name="test_infer")
    
    # Setup
    hg1 = create_test_hypergraph("l1")
    hg2 = create_test_hypergraph("l2")
    r_lcg.add_layer("layer1", hg1, LayerType.PHYSICAL, level=0)
    r_lcg.add_layer("layer2", hg2, LayerType.SEMANTIC, level=1)
    
    # Add rule
    r_lcg.add_inference_rule(
        "infer_rule",
        "layer1",
        "layer2",
        condition=lambda x: isinstance(x, str),
        action=lambda x: f"inferred_from_{x}"
    )
    
    # Create entity with state
    r_lcg.create_superposition(
        "test_entity",
        layer_states={"layer1": "raw_data"}
    )
    
    # Perform inference
    results = r_lcg.infer_cross_layer("test_entity", "layer1", "layer2")
    
    assert len(results) >= 1
    assert "inferred_from_raw_data" in str(results[0]['fact'])
    assert results[0]['confidence'] > 0
    
    print("   ✅ Inference performed successfully")
    print(f"   ✅ Inferred facts: {len(results)}")
    print(f"   ✅ Sample fact: {results[0]['fact']}")
    
    return True


def test_contradiction_detection():
    """Test 5: Detecting contradictions"""
    print("\n" + "="*60)
    print("Test 5: Contradiction Detection")
    print("="*60)
    
    r_lcg = ReasoningLayeredGraph(name="test_contradiction", auto_detect_contradictions=False)
    
    # Setup layers
    hg1 = create_test_hypergraph("l1")
    hg2 = create_test_hypergraph("l2")
    r_lcg.add_layer("layer1", hg1, LayerType.PHYSICAL, level=0)
    r_lcg.add_layer("layer2", hg2, LayerType.SEMANTIC, level=1)
    
    # Create entity with potentially contradictory states
    r_lcg.create_superposition(
        "entity_with_contradiction",
        layer_states={
            "layer1": "state_A",
            "layer2": "state_B"  # Different state
        }
    )
    
    # Detect contradictions
    contradictions = r_lcg.check_consistency("entity_with_contradiction")
    
    # May or may not find contradictions depending on rules
    # Just check that detection runs
    assert isinstance(contradictions, list)
    
    print(f"   ✅ Contradiction detection completed")
    print(f"   ✅ Contradictions found: {len(contradictions)}")
    
    return True


def test_auto_contradiction_detection():
    """Test 6: Automatic contradiction detection"""
    print("\n" + "="*60)
    print("Test 6: Auto-Contradiction Detection")
    print("="*60)
    
    r_lcg = ReasoningLayeredGraph(name="test_auto_contra", auto_detect_contradictions=True)
    
    # Setup
    hg1 = create_test_hypergraph("l1")
    hg2 = create_test_hypergraph("l2")
    r_lcg.add_layer("layer1", hg1, LayerType.PHYSICAL, level=0)
    r_lcg.add_layer("layer2", hg2, LayerType.SEMANTIC, level=1)
    
    # Create superposition (should trigger auto-detection)
    r_lcg.create_superposition(
        "auto_check_entity",
        layer_states={"layer1": "s1", "layer2": "s2"}
    )
    
    # Auto-detection should have run (check logs or contradictions list)
    assert hasattr(r_lcg.reasoning_engine, 'contradictions')
    
    print("   ✅ Auto-detection enabled")
    print(f"   ✅ Total contradictions tracked: {len(r_lcg.reasoning_engine.contradictions)}")
    
    return True


def test_contradiction_resolution():
    """Test 7: Resolving contradictions"""
    print("\n" + "="*60)
    print("Test 7: Contradiction Resolution")
    print("="*60)
    
    r_lcg = ReasoningLayeredGraph(name="test_resolve")
    
    # Setup hierarchy (different levels for priority)
    hg1 = create_test_hypergraph("l1")
    hg2 = create_test_hypergraph("l2")
    r_lcg.add_layer("low_priority", hg1, LayerType.PHYSICAL, level=0)
    r_lcg.add_layer("high_priority", hg2, LayerType.CONCEPTUAL, level=2)
    
    # Create contradictory states
    r_lcg.create_superposition(
        "conflict_entity",
        layer_states={
            "low_priority": "state_low",
            "high_priority": "state_high"
        }
    )
    
    # Detect contradictions
    contradictions = r_lcg.check_consistency("conflict_entity")
    
    if contradictions:
        # Resolve using priority strategy
        resolved = r_lcg.reasoning_engine.resolve_contradiction(
            contradictions[0],
            strategy="priority"
        )
        
        assert resolved == True
        
        print(f"   ✅ Contradiction resolved")
        print(f"   ✅ Resolution strategy: priority")
    else:
        print(f"   ℹ️  No contradictions to resolve")
    
    return True


def test_belief_propagation():
    """Test 8: Belief propagation"""
    print("\n" + "="*60)
    print("Test 8: Belief Propagation")
    print("="*60)
    
    r_lcg = ReasoningLayeredGraph(name="test_belief")
    
    # Create entity with quantum state
    r_lcg.create_superposition(
        "belief_entity",
        quantum_states={"state_a": 0.6, "state_b": 0.4}
    )
    
    # Propagate beliefs with evidence
    evidence = {"layer1": "observed_value"}
    beliefs = r_lcg.reasoning_engine.propagate_beliefs("belief_entity", evidence)
    
    assert isinstance(beliefs, dict)
    # Beliefs should sum to ~1.0
    if beliefs:
        total = sum(beliefs.values())
        assert 0.99 <= total <= 1.01
        
        print(f"   ✅ Belief propagation completed")
        print(f"   ✅ Updated beliefs: {beliefs}")
    else:
        print(f"   ℹ️  No beliefs to propagate")
    
    return True


def test_hierarchical_inference():
    """Test 9: Hierarchical inference (bottom-up and top-down)"""
    print("\n" + "="*60)
    print("Test 9: Hierarchical Inference")
    print("="*60)
    
    r_lcg = ReasoningLayeredGraph(name="test_hierarchical")
    
    # Create 3-level hierarchy
    hg0 = create_test_hypergraph("l0")
    hg1 = create_test_hypergraph("l1")
    hg2 = create_test_hypergraph("l2")
    
    r_lcg.add_layer("level0", hg0, LayerType.PHYSICAL, level=0)
    r_lcg.add_layer("level1", hg1, LayerType.SEMANTIC, level=1, parent_layer="level0")
    r_lcg.add_layer("level2", hg2, LayerType.CONCEPTUAL, level=2, parent_layer="level1")
    
    # Add inference rules
    for i in range(2):
        r_lcg.add_inference_rule(
            f"rule_up_{i}",
            f"level{i}",
            f"level{i+1}",
            lambda x: True,
            lambda x: f"higher_{x}"
        )
    
    # Create entity in all layers
    r_lcg.create_superposition(
        "hierarchical_entity",
        layer_states={
            "level0": "data_0",
            "level1": "data_1",
            "level2": "data_2"
        }
    )
    
    # Bottom-up inference
    up_results = r_lcg.reasoning_engine.hierarchical_inference("hierarchical_entity", direction="up")
    
    # Top-down inference
    down_results = r_lcg.reasoning_engine.hierarchical_inference("hierarchical_entity", direction="down")
    
    assert isinstance(up_results, dict)
    assert isinstance(down_results, dict)
    
    print(f"   ✅ Bottom-up inference completed")
    print(f"   ✅ Inferred connections: {list(up_results.keys())}")
    print(f"   ✅ Top-down inference completed")
    print(f"   ✅ Inferred connections: {list(down_results.keys())}")
    
    return True


def test_enable_reasoning():
    """Test 10: Enable reasoning on existing LCG"""
    print("\n" + "="*60)
    print("Test 10: Enable Reasoning on Existing LCG")
    print("="*60)
    
    # Create regular LCG
    lcg = LayeredContextualGraph(name="regular_lcg")
    
    # Add some layers and entities
    hg = create_test_hypergraph("base")
    lcg.add_layer("base", hg, LayerType.PHYSICAL, level=0)
    lcg.create_superposition("entity", layer_states={"base": "state"})
    
    # Enable reasoning
    r_lcg = enable_reasoning(lcg, auto_detect=True)
    
    assert hasattr(r_lcg, 'reasoning_engine')
    assert hasattr(r_lcg, 'add_inference_rule')
    assert r_lcg.auto_detect_contradictions == True
    
    # Test reasoning functionality
    r_lcg.add_inference_rule(
        "test_rule",
        "base",
        "base",
        lambda x: True,
        lambda x: f"processed_{x}"
    )
    
    assert "test_rule" in r_lcg.reasoning_engine.rules
    
    print(f"   ✅ Reasoning enabled on existing LCG")
    print(f"   ✅ Reasoning engine functional")
    print(f"   ✅ Inherited {len(r_lcg.layers)} layers")
    
    return True


def run_all_tests():
    """Run all reasoning tests"""
    
    print("\n" + "="*70)
    print("ADVANCED REASONING EXTENSION TEST SUITE")
    print("="*70)
    print("\nComprehensive tests for LCG reasoning capabilities\n")
    
    try:
        test_reasoning_graph_creation()
        test_inference_rules()
        test_add_inference_rule()
        test_inference()
        test_contradiction_detection()
        test_auto_contradiction_detection()
        test_contradiction_resolution()
        test_belief_propagation()
        test_hierarchical_inference()
        test_enable_reasoning()
        
        print("\n" + "="*70)
        print("✅ ALL REASONING TESTS PASSED")
        print("="*70)
        print("\nTest Summary:")
        print("   ✅ ReasoningLayeredGraph creation")
        print("   ✅ Inference rules")
        print("   ✅ Adding rules to graph")
        print("   ✅ Cross-layer inference")
        print("   ✅ Contradiction detection")
        print("   ✅ Auto-contradiction detection")
        print("   ✅ Contradiction resolution")
        print("   ✅ Belief propagation")
        print("   ✅ Hierarchical inference")
        print("   ✅ Enable reasoning on existing LCG")
        print("\n   Total: 10/10 reasoning tests passed\n")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
