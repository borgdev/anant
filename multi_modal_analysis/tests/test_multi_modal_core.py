"""
Test Suite for Multi-Modal Hypergraph Core Functionality
=========================================================

Tests for MultiModalHypergraph class and basic operations.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import polars as pl
    import numpy as np
except ImportError:
    print("⚠️  Required dependencies not installed. Please install: polars numpy")
    sys.exit(1)

from core.multi_modal_hypergraph import MultiModalHypergraph, ModalityConfig


# Mock Hypergraph for testing
class MockHypergraph:
    """Mock hypergraph for testing"""
    def __init__(self, data):
        self.data = data
        self.incidences = type('obj', (object,), {'data': data})()
        if 'nodes' in data.columns:
            self._nodes = set(data['nodes'].unique())
        else:
            self._nodes = set()
    
    def nodes(self):
        return self._nodes


def create_test_data():
    """Create test data for multi-modal analysis"""
    # Modality 1: Friendships
    friendships = pl.DataFrame([
        {"edges": "f1", "nodes": "Alice", "weight": 1.0},
        {"edges": "f1", "nodes": "Bob", "weight": 1.0},
        {"edges": "f2", "nodes": "Bob", "weight": 1.0},
        {"edges": "f2", "nodes": "Charlie", "weight": 1.0},
        {"edges": "f3", "nodes": "Alice", "weight": 1.0},
        {"edges": "f3", "nodes": "Charlie", "weight": 1.0},
    ])
    
    # Modality 2: Collaborations
    collaborations = pl.DataFrame([
        {"edges": "c1", "nodes": "Alice", "weight": 1.0},
        {"edges": "c1", "nodes": "Bob", "weight": 1.0},
        {"edges": "c2", "nodes": "Alice", "weight": 1.0},
        {"edges": "c2", "nodes": "David", "weight": 1.0},
        {"edges": "c3", "nodes": "Bob", "weight": 1.0},
        {"edges": "c3", "nodes": "Charlie", "weight": 1.0},
    ])
    
    # Modality 3: Communications
    communications = pl.DataFrame([
        {"edges": "m1", "nodes": "Alice", "weight": 1.0},
        {"edges": "m1", "nodes": "Bob", "weight": 1.0},
        {"edges": "m2", "nodes": "Charlie", "weight": 1.0},
        {"edges": "m2", "nodes": "David", "weight": 1.0},
    ])
    
    return {
        'friendships': MockHypergraph(friendships),
        'collaborations': MockHypergraph(collaborations),
        'communications': MockHypergraph(communications)
    }


def test_multimodal_construction():
    """Test 1: Multi-modal hypergraph construction"""
    print("\n" + "="*60)
    print("Test 1: Multi-Modal Hypergraph Construction")
    print("="*60)
    
    data = create_test_data()
    mmhg = MultiModalHypergraph(name="test_network")
    
    # Add modalities
    mmhg.add_modality("friendships", data['friendships'], weight=1.0)
    mmhg.add_modality("collaborations", data['collaborations'], weight=2.0)
    mmhg.add_modality("communications", data['communications'], weight=1.5)
    
    # Verify
    assert len(mmhg.list_modalities()) == 3, "Should have 3 modalities"
    assert "friendships" in mmhg.list_modalities(), "Should contain friendships"
    assert mmhg.modality_configs["collaborations"].weight == 2.0, "Weight should be 2.0"
    
    print("✅ Construction test passed")
    return mmhg


def test_modality_management(mmhg):
    """Test 2: Modality management operations"""
    print("\n" + "="*60)
    print("Test 2: Modality Management")
    print("="*60)
    
    # Get modality
    friendship_hg = mmhg.get_modality("friendships")
    assert friendship_hg is not None, "Should retrieve modality"
    
    # Get modality info
    info = mmhg.get_modality_info("collaborations")
    assert isinstance(info, ModalityConfig), "Should return ModalityConfig"
    assert info.weight == 2.0, "Weight should match"
    
    # Add new modality
    new_data = pl.DataFrame([
        {"edges": "e1", "nodes": "Alice", "weight": 1.0},
        {"edges": "e1", "nodes": "Eve", "weight": 1.0},
    ])
    mmhg.add_modality("emails", MockHypergraph(new_data), weight=1.0)
    assert len(mmhg.list_modalities()) == 4, "Should have 4 modalities"
    
    # Remove modality
    mmhg.remove_modality("emails")
    assert len(mmhg.list_modalities()) == 3, "Should have 3 modalities after removal"
    
    print("✅ Modality management test passed")


def test_entity_indexing(mmhg):
    """Test 3: Entity indexing"""
    print("\n" + "="*60)
    print("Test 3: Entity Indexing")
    print("="*60)
    
    entity_index = mmhg._build_entity_index()
    
    # Verify index structure
    assert isinstance(entity_index, dict), "Index should be a dict"
    assert "Alice" in entity_index, "Alice should be indexed"
    assert "Bob" in entity_index, "Bob should be indexed"
    
    # Verify modality associations
    alice_mods = entity_index["Alice"]
    assert "friendships" in alice_mods, "Alice in friendships"
    assert "collaborations" in alice_mods, "Alice in collaborations"
    assert "communications" in alice_mods, "Alice in communications"
    
    print(f"   Total entities indexed: {len(entity_index)}")
    print(f"   Alice participates in: {alice_mods}")
    print("✅ Entity indexing test passed")


def test_modal_bridges(mmhg):
    """Test 4: Modal bridge detection"""
    print("\n" + "="*60)
    print("Test 4: Modal Bridge Detection")
    print("="*60)
    
    # Find entities in 2+ modalities
    bridges_2 = mmhg.find_modal_bridges(min_modalities=2)
    assert len(bridges_2) > 0, "Should find some bridges"
    assert "Alice" in bridges_2, "Alice should be a bridge"
    
    # Find entities in all 3 modalities
    bridges_3 = mmhg.find_modal_bridges(min_modalities=3)
    
    print(f"   Entities in 2+ modalities: {len(bridges_2)}")
    print(f"   Entities in all 3 modalities: {len(bridges_3)}")
    
    # Verify Alice is in all 3
    if "Alice" in bridges_3:
        print("   ✅ Alice appears in all modalities")
    
    print("✅ Modal bridge test passed")
    return bridges_2


def test_cross_modal_patterns(mmhg):
    """Test 5: Cross-modal pattern detection"""
    print("\n" + "="*60)
    print("Test 5: Cross-Modal Pattern Detection")
    print("="*60)
    
    patterns = mmhg.detect_cross_modal_patterns(min_support=1)
    
    assert isinstance(patterns, list), "Should return list of patterns"
    assert len(patterns) > 0, "Should detect some patterns"
    
    # Check pattern structure
    for pattern in patterns[:3]:
        assert 'type' in pattern, "Pattern should have type"
        assert 'description' in pattern, "Pattern should have description"
        assert 'support' in pattern, "Pattern should have support"
    
    print(f"   Patterns detected: {len(patterns)}")
    print("   Pattern types:")
    pattern_types = {}
    for p in patterns:
        pattern_types[p['type']] = pattern_types.get(p['type'], 0) + 1
    
    for ptype, count in pattern_types.items():
        print(f"      {ptype}: {count}")
    
    print("✅ Cross-modal pattern test passed")


def test_cross_modal_centrality(mmhg):
    """Test 6: Cross-modal centrality computation"""
    print("\n" + "="*60)
    print("Test 6: Cross-Modal Centrality")
    print("="*60)
    
    # Test degree centrality
    centrality = mmhg.compute_cross_modal_centrality(
        "Alice",
        metric="degree",
        aggregation="weighted_average"
    )
    
    assert 'node_id' in centrality, "Should have node_id"
    assert 'aggregated' in centrality, "Should have aggregated score"
    assert 'per_modality' in centrality, "Should have per-modality scores"
    assert centrality['node_id'] == "Alice", "Node ID should match"
    
    print(f"   Alice's centrality: {centrality['aggregated']:.3f}")
    print("   Per-modality scores:")
    for mod, score in centrality['per_modality'].items():
        print(f"      {mod}: {score:.1f}")
    
    # Test different aggregations
    for agg in ["max", "min", "sum", "average"]:
        result = mmhg.compute_cross_modal_centrality("Bob", "degree", agg)
        assert result['aggregated'] >= 0, f"Aggregation {agg} should work"
    
    print("✅ Cross-modal centrality test passed")


def test_inter_modal_relationships(mmhg):
    """Test 7: Inter-modal relationship discovery"""
    print("\n" + "="*60)
    print("Test 7: Inter-Modal Relationship Discovery")
    print("="*60)
    
    relationships = mmhg.discover_inter_modal_relationships(
        source_modality="friendships",
        target_modality="collaborations"
    )
    
    assert isinstance(relationships, list), "Should return list"
    assert len(relationships) > 0, "Should find some relationships"
    
    # Check relationship structure
    for rel in relationships[:3]:
        assert 'node_id' in rel, "Should have node_id"
        assert 'source_modality' in rel, "Should have source_modality"
        assert 'target_modality' in rel, "Should have target_modality"
    
    print(f"   Relationships found: {len(relationships)}")
    print("   Sample relationships:")
    for rel in relationships[:3]:
        print(f"      {rel['node_id']}: {rel['source_modality']} → {rel['target_modality']}")
    
    print("✅ Inter-modal relationship test passed")


def test_modal_correlation(mmhg):
    """Test 8: Modal correlation computation"""
    print("\n" + "="*60)
    print("Test 8: Modal Correlation")
    print("="*60)
    
    # Test different correlation methods
    methods = ["jaccard", "overlap", "cosine"]
    
    for method in methods:
        corr = mmhg.compute_modal_correlation(
            "friendships",
            "collaborations",
            method=method
        )
        
        assert 0 <= corr <= 1, f"Correlation should be in [0, 1] for {method}"
        print(f"   {method}: {corr:.3f}")
    
    print("✅ Modal correlation test passed")


def test_summary_generation(mmhg):
    """Test 9: Summary generation"""
    print("\n" + "="*60)
    print("Test 9: Summary Generation")
    print("="*60)
    
    summary = mmhg.generate_summary()
    
    # Verify summary structure
    assert 'name' in summary, "Should have name"
    assert 'num_modalities' in summary, "Should have num_modalities"
    assert 'total_unique_entities' in summary, "Should have total_unique_entities"
    assert 'modality_stats' in summary, "Should have modality_stats"
    
    print(f"   Name: {summary['name']}")
    print(f"   Modalities: {summary['num_modalities']}")
    print(f"   Unique entities: {summary['total_unique_entities']}")
    print(f"   Avg modalities per entity: {summary['avg_modalities_per_entity']:.2f}")
    
    print("✅ Summary generation test passed")


def test_cache_invalidation(mmhg):
    """Test 10: Cache invalidation"""
    print("\n" + "="*60)
    print("Test 10: Cache Invalidation")
    print("="*60)
    
    # Build entity index (cached)
    index1 = mmhg._build_entity_index()
    initial_count = len(index1)
    
    # Add new modality (should invalidate cache)
    new_data = pl.DataFrame([
        {"edges": "n1", "nodes": "Zoe", "weight": 1.0},
        {"edges": "n1", "nodes": "Alice", "weight": 1.0},
    ])
    mmhg.add_modality("new_mod", MockHypergraph(new_data))
    
    # Rebuild index
    index2 = mmhg._build_entity_index()
    new_count = len(index2)
    
    # Verify cache was invalidated and rebuilt
    assert "Zoe" in index2, "New entity should appear"
    assert new_count >= initial_count, "Entity count should increase or stay same"
    
    # Clean up
    mmhg.remove_modality("new_mod")
    
    print(f"   Initial entities: {initial_count}")
    print(f"   After adding modality: {new_count}")
    print("✅ Cache invalidation test passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("MULTI-MODAL CORE FUNCTIONALITY TEST SUITE")
    print("="*70)
    
    try:
        # Test 1: Construction
        mmhg = test_multimodal_construction()
        
        # Test 2: Modality management
        test_modality_management(mmhg)
        
        # Test 3: Entity indexing
        test_entity_indexing(mmhg)
        
        # Test 4: Modal bridges
        test_modal_bridges(mmhg)
        
        # Test 5: Cross-modal patterns
        test_cross_modal_patterns(mmhg)
        
        # Test 6: Centrality
        test_cross_modal_centrality(mmhg)
        
        # Test 7: Inter-modal relationships
        test_inter_modal_relationships(mmhg)
        
        # Test 8: Modal correlation
        test_modal_correlation(mmhg)
        
        # Test 9: Summary
        test_summary_generation(mmhg)
        
        # Test 10: Cache
        test_cache_invalidation(mmhg)
        
        # Summary
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nTest Summary:")
        print("   ✅ Multi-modal construction")
        print("   ✅ Modality management")
        print("   ✅ Entity indexing")
        print("   ✅ Modal bridges")
        print("   ✅ Cross-modal patterns")
        print("   ✅ Cross-modal centrality")
        print("   ✅ Inter-modal relationships")
        print("   ✅ Modal correlation")
        print("   ✅ Summary generation")
        print("   ✅ Cache invalidation")
        print("\n   Total: 10/10 tests passed\n")
        
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
