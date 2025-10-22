"""
Test Anant Inheritance for MultiModalHypergraph
================================================

Validates that MultiModalHypergraph properly extends Anant's core Hypergraph class.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import polars as pl
except ImportError:
    print("⚠️  Required dependencies not installed.")
    sys.exit(1)

# Import Anant core
try:
    from anant.classes.hypergraph.core.hypergraph import Hypergraph as AnantHypergraph
    from anant.classes.incidence_store import IncidenceStore
    from anant.classes.property_store import PropertyStore
    ANANT_AVAILABLE = True
except ImportError:
    ANANT_AVAILABLE = False
    print("❌ Anant core library not available")
    sys.exit(1)

from core.multi_modal_hypergraph import MultiModalHypergraph


def test_inheritance_chain():
    """Test 1: Verify inheritance chain"""
    print("\n" + "="*60)
    print("Test 1: Inheritance Chain Validation")
    print("="*60)
    
    mmhg = MultiModalHypergraph(name="test")
    
    # Check inheritance
    assert isinstance(mmhg, AnantHypergraph), "Should be instance of AnantHypergraph"
    assert isinstance(mmhg, MultiModalHypergraph), "Should be instance of MultiModalHypergraph"
    
    # Check MRO (Method Resolution Order)
    mro = [c.__name__ for c in type(mmhg).__mro__]
    print(f"   Method Resolution Order: {mro}")
    
    assert 'MultiModalHypergraph' in mro, "Should have MultiModalHypergraph in MRO"
    assert 'Hypergraph' in mro, "Should have Hypergraph in MRO"
    assert mro.index('MultiModalHypergraph') < mro.index('Hypergraph'), \
        "MultiModalHypergraph should come before Hypergraph in MRO"
    
    print("   ✅ Inheritance chain correct")
    print(f"   ✅ MRO: {' → '.join(mro[:3])}")
    
    return True


def test_inherited_attributes():
    """Test 2: Verify inherited Anant attributes"""
    print("\n" + "="*60)
    print("Test 2: Inherited Anant Attributes")
    print("="*60)
    
    # Create with base data
    data = pl.DataFrame([
        {'edge_id': 'e1', 'node_id': 'n1', 'weight': 1.0},
        {'edge_id': 'e1', 'node_id': 'n2', 'weight': 1.0},
    ])
    
    mmhg = MultiModalHypergraph(data=data, name="test_attrs")
    
    # Core Anant attributes
    assert hasattr(mmhg, 'name'), "Should have 'name' attribute"
    assert hasattr(mmhg, '_instance_id'), "Should have '_instance_id' attribute"
    assert hasattr(mmhg, 'incidences'), "Should have 'incidences' attribute"
    assert hasattr(mmhg, 'properties'), "Should have 'properties' attribute"
    assert hasattr(mmhg, 'metadata'), "Should have 'metadata' attribute"
    assert hasattr(mmhg, '_cache'), "Should have '_cache' attribute"
    
    print("   ✅ Core attributes present:")
    print(f"      • name: {mmhg.name}")
    print(f"      • instance_id: {mmhg._instance_id[:8]}...")
    print(f"      • incidences: {type(mmhg.incidences).__name__}")
    print(f"      • properties: {type(mmhg.properties).__name__}")
    
    # Verify types
    assert isinstance(mmhg.incidences, IncidenceStore), \
        "incidences should be IncidenceStore instance"
    assert isinstance(mmhg.properties, PropertyStore), \
        "properties should be PropertyStore instance"
    
    print("   ✅ All inherited attributes correct")
    
    return True


def test_inherited_operations():
    """Test 3: Verify inherited Anant operation modules"""
    print("\n" + "="*60)
    print("Test 3: Inherited Operation Modules")
    print("="*60)
    
    mmhg = MultiModalHypergraph(name="test_ops")
    
    # Operation modules
    operation_modules = [
        ('_core_ops', 'CoreOperations'),
        ('_performance_ops', 'PerformanceOperations'),
        ('_io_ops', 'IOOperations'),
        ('_algorithm_ops', 'AlgorithmOperations'),
        ('_visualization_ops', 'VisualizationOperations'),
        ('_set_ops', 'SetOperations'),
        ('_advanced_ops', 'AdvancedOperations'),
    ]
    
    print("   Checking operation modules:")
    for attr_name, module_name in operation_modules:
        assert hasattr(mmhg, attr_name), f"Should have '{attr_name}' module"
        print(f"      ✅ {attr_name} ({module_name})")
    
    print("   ✅ All operation modules present")
    
    return True


def test_inherited_indexes():
    """Test 4: Verify inherited performance indexes"""
    print("\n" + "="*60)
    print("Test 4: Inherited Performance Structures")
    print("="*60)
    
    data = pl.DataFrame([
        {'edge_id': 'e1', 'node_id': 'n1', 'weight': 1.0},
        {'edge_id': 'e1', 'node_id': 'n2', 'weight': 1.0},
        {'edge_id': 'e2', 'node_id': 'n2', 'weight': 1.0},
        {'edge_id': 'e2', 'node_id': 'n3', 'weight': 1.0},
    ])
    
    mmhg = MultiModalHypergraph(data=data, name="test_indexes")
    
    # Performance structures
    assert hasattr(mmhg, '_indexes_built'), "Should have '_indexes_built' flag"
    assert hasattr(mmhg, '_dirty'), "Should have '_dirty' flag"
    assert hasattr(mmhg, '_node_to_edges'), "Should have '_node_to_edges' index"
    assert hasattr(mmhg, '_edge_to_nodes'), "Should have '_edge_to_nodes' index"
    assert hasattr(mmhg, '_node_degrees'), "Should have '_node_degrees' cache"
    assert hasattr(mmhg, '_edge_sizes'), "Should have '_edge_sizes' cache"
    assert hasattr(mmhg, '_performance_stats'), "Should have '_performance_stats'"
    
    print("   ✅ Performance structures present:")
    print(f"      • Indexes built: {mmhg._indexes_built}")
    print(f"      • Node-to-edges index: {len(mmhg._node_to_edges)} entries")
    print(f"      • Edge-to-nodes index: {len(mmhg._edge_to_nodes)} entries")
    print(f"      • Performance stats: {list(mmhg._performance_stats.keys())}")
    
    print("   ✅ All performance structures correct")
    
    return True


def test_multimodal_extensions():
    """Test 5: Verify multi-modal specific attributes"""
    print("\n" + "="*60)
    print("Test 5: Multi-Modal Extensions")
    print("="*60)
    
    mmhg = MultiModalHypergraph(name="test_multimodal")
    
    # Multi-modal specific attributes
    assert hasattr(mmhg, 'modalities'), "Should have 'modalities' dict"
    assert hasattr(mmhg, 'modality_configs'), "Should have 'modality_configs' dict"
    assert hasattr(mmhg, 'cross_modal_cache'), "Should have 'cross_modal_cache' dict"
    assert hasattr(mmhg, '_entity_index'), "Should have '_entity_index'"
    
    print("   ✅ Multi-modal attributes present:")
    print(f"      • modalities: {type(mmhg.modalities).__name__}")
    print(f"      • modality_configs: {type(mmhg.modality_configs).__name__}")
    print(f"      • cross_modal_cache: {type(mmhg.cross_modal_cache).__name__}")
    
    # Multi-modal specific methods
    assert hasattr(mmhg, 'add_modality'), "Should have 'add_modality' method"
    assert hasattr(mmhg, 'remove_modality'), "Should have 'remove_modality' method"
    assert hasattr(mmhg, 'find_modal_bridges'), "Should have 'find_modal_bridges' method"
    assert hasattr(mmhg, 'detect_cross_modal_patterns'), "Should have pattern detection"
    assert hasattr(mmhg, 'compute_cross_modal_centrality'), "Should have centrality"
    
    print("   ✅ Multi-modal methods present:")
    print(f"      • add_modality")
    print(f"      • find_modal_bridges")
    print(f"      • detect_cross_modal_patterns")
    print(f"      • compute_cross_modal_centrality")
    
    print("   ✅ All multi-modal extensions correct")
    
    return True


def test_full_integration():
    """Test 6: Full integration test"""
    print("\n" + "="*60)
    print("Test 6: Full Integration Test")
    print("="*60)
    
    # Create base MultiModalHypergraph
    base_data = pl.DataFrame([
        {'edge_id': 'base_e1', 'node_id': 'base_n1', 'weight': 1.0},
        {'edge_id': 'base_e1', 'node_id': 'base_n2', 'weight': 1.0},
    ])
    
    mmhg = MultiModalHypergraph(data=base_data, name="integration_test")
    
    print("   1. Base hypergraph created")
    assert not mmhg.incidences.data.is_empty(), "Should have base data"
    
    # Create modality hypergraphs
    mod1_data = pl.DataFrame([
        {'edge_id': 'm1_e1', 'node_id': 'customer_1', 'weight': 1.0},
        {'edge_id': 'm1_e1', 'node_id': 'product_A', 'weight': 1.0},
    ])
    mod1_hg = AnantHypergraph(data=mod1_data, name="modality1")
    
    print("   2. Modality hypergraph created")
    
    # Add modality
    mmhg.add_modality("purchases", mod1_hg, weight=2.0)
    assert "purchases" in mmhg.modalities, "Should have purchases modality"
    print("   3. Modality added successfully")
    
    # Use multi-modal features
    summary = mmhg.generate_summary()
    assert summary['num_modalities'] == 1, "Should have 1 modality"
    print(f"   4. Summary generated: {summary['num_modalities']} modality(ies)")
    
    print("   ✅ Full integration working correctly")
    
    return True


def run_all_tests():
    """Run all inheritance validation tests"""
    
    if not ANANT_AVAILABLE:
        print("❌ Cannot run tests - Anant not available")
        return False
    
    print("\n" + "="*70)
    print("ANANT INHERITANCE VALIDATION TEST SUITE")
    print("="*70)
    print("\nValidating MultiModalHypergraph extends Anant's core Hypergraph\n")
    
    try:
        test_inheritance_chain()
        test_inherited_attributes()
        test_inherited_operations()
        test_inherited_indexes()
        test_multimodal_extensions()
        test_full_integration()
        
        print("\n" + "="*70)
        print("✅ ALL INHERITANCE TESTS PASSED")
        print("="*70)
        print("\nValidation Summary:")
        print("   ✅ Inheritance chain correct (MultiModalHypergraph → Hypergraph)")
        print("   ✅ All Anant attributes inherited")
        print("   ✅ All Anant operation modules present")
        print("   ✅ All Anant performance structures inherited")
        print("   ✅ Multi-modal extensions working")
        print("   ✅ Full integration functional")
        print("\n   🎉 MultiModalHypergraph PROPERLY extends Anant's Hypergraph 🎉")
        print("\n   Total: 6/6 inheritance tests passed\n")
        
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
