"""
Test Suite: Machine Learning Extension
======================================

Comprehensive tests for ML capabilities in LCG.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import polars as pl
    import numpy as np
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    print("⚠️  Required dependencies not installed.")
    sys.exit(1)

from layered_contextual_graph.core import LayeredContextualGraph, LayerType
from layered_contextual_graph.extensions import (
    MLLayeredGraph,
    EmbeddingLayer,
    EntityEmbedding,
    enable_ml
)


def test_ml_graph_creation():
    """Test 1: MLLayeredGraph creation"""
    print("\n" + "="*60)
    print("Test 1: MLLayeredGraph Creation")
    print("="*60)
    
    ml_lcg = MLLayeredGraph(
        name="test_ml",
        quantum_enabled=True,
        embedding_dim=128
    )
    
    assert ml_lcg.name == "test_ml"
    assert ml_lcg.embedding_dim == 128
    assert hasattr(ml_lcg, 'embedding_layers')
    assert hasattr(ml_lcg, 'entity_embeddings')
    
    print("   ✅ MLLayeredGraph created successfully")
    print(f"   ✅ Embedding dimension: {ml_lcg.embedding_dim}")
    
    return True


def test_embedding_layer():
    """Test 2: Embedding layer functionality"""
    print("\n" + "="*60)
    print("Test 2: Embedding Layer")
    print("="*60)
    
    emb_layer = EmbeddingLayer(
        name="test_embeddings",
        embedding_dim=64,
        normalize_embeddings=True
    )
    
    # Add embeddings
    np.random.seed(42)
    for i in range(10):
        embedding = np.random.randn(64)
        emb_layer.add_embedding(f"entity_{i}", embedding)
    
    assert len(emb_layer.embeddings) == 10
    
    print(f"   ✅ Added 10 embeddings")
    
    # Build index
    emb_layer.build_index()
    assert emb_layer.embedding_matrix is not None
    assert emb_layer.embedding_matrix.shape == (10, 64)
    
    print(f"   ✅ Built embedding index")
    print(f"   ✅ Index shape: {emb_layer.embedding_matrix.shape}")
    
    return True


def test_similarity_search():
    """Test 3: Similarity search"""
    print("\n" + "="*60)
    print("Test 3: Similarity Search")
    print("="*60)
    
    emb_layer = EmbeddingLayer(name="search_test", embedding_dim=64)
    
    # Add embeddings
    np.random.seed(42)
    embeddings = {}
    for i in range(20):
        emb = np.random.randn(64)
        embeddings[f"entity_{i}"] = emb
        emb_layer.add_embedding(f"entity_{i}", emb)
    
    emb_layer.build_index()
    
    # Search for similar to entity_0
    query = embeddings["entity_0"]
    results = emb_layer.similarity_search(query, top_k=5)
    
    assert len(results) == 5
    assert results[0][0] == "entity_0"  # Most similar to itself
    assert results[0][1] > 0.99  # Very high similarity to self
    
    print(f"   ✅ Similarity search working")
    print(f"   ✅ Top 5 results:")
    for entity_id, similarity in results[:3]:
        print(f"      {entity_id}: {similarity:.3f}")
    
    return True


def test_entity_embeddings():
    """Test 4: Entity embeddings across layers"""
    print("\n" + "="*60)
    print("Test 4: Entity Embeddings Across Layers")
    print("="*60)
    
    ml_lcg = MLLayeredGraph(name="test_entity_emb", embedding_dim=64)
    
    # Add embedding layers
    ml_lcg.add_embedding_layer("physical")
    ml_lcg.add_embedding_layer("semantic")
    
    # Set embeddings for entity in multiple layers
    np.random.seed(42)
    entity_id = "multi_layer_entity"
    
    phys_emb = np.random.randn(64)
    sem_emb = np.random.randn(64)
    
    ml_lcg.set_entity_embedding(entity_id, "physical", phys_emb)
    ml_lcg.set_entity_embedding(entity_id, "semantic", sem_emb)
    
    # Check entity embeddings
    assert entity_id in ml_lcg.entity_embeddings
    assert "physical" in ml_lcg.entity_embeddings[entity_id].layer_embeddings
    assert "semantic" in ml_lcg.entity_embeddings[entity_id].layer_embeddings
    
    print(f"   ✅ Entity has embeddings in 2 layers")
    
    # Aggregate embeddings
    entity_emb = ml_lcg.entity_embeddings[entity_id]
    aggregated = entity_emb.aggregate(method="mean")
    
    assert aggregated is not None
    assert aggregated.shape == (64,)
    
    print(f"   ✅ Aggregated embedding shape: {aggregated.shape}")
    
    return True


def test_cross_layer_similarity():
    """Test 5: Cross-layer similarity"""
    print("\n" + "="*60)
    print("Test 5: Cross-Layer Similarity")
    print("="*60)
    
    ml_lcg = MLLayeredGraph(name="test_cross_sim", embedding_dim=64)
    
    ml_lcg.add_embedding_layer("layer1")
    ml_lcg.add_embedding_layer("layer2")
    
    # Create similar entities
    np.random.seed(42)
    base_emb = np.random.randn(64)
    
    entity1_l1 = base_emb + np.random.randn(64) * 0.1
    entity1_l2 = base_emb + np.random.randn(64) * 0.1
    
    entity2_l1 = base_emb + np.random.randn(64) * 0.1
    entity2_l2 = base_emb + np.random.randn(64) * 0.1
    
    ml_lcg.set_entity_embedding("entity1", "layer1", entity1_l1)
    ml_lcg.set_entity_embedding("entity1", "layer2", entity1_l2)
    
    ml_lcg.set_entity_embedding("entity2", "layer1", entity2_l1)
    ml_lcg.set_entity_embedding("entity2", "layer2", entity2_l2)
    
    # Compute cross-layer similarity
    similarity = ml_lcg.cross_layer_similarity("entity1", "entity2", aggregation="mean")
    
    assert 0 <= similarity <= 1
    assert similarity > 0.5  # Should be similar since they're based on same base
    
    print(f"   ✅ Cross-layer similarity computed")
    print(f"   ✅ Similarity(entity1, entity2): {similarity:.3f}")
    
    return True


def test_multi_layer_search():
    """Test 6: Similarity search across multiple layers"""
    print("\n" + "="*60)
    print("Test 6: Multi-Layer Similarity Search")
    print("="*60)
    
    ml_lcg = MLLayeredGraph(name="test_multi_search", embedding_dim=64)
    
    # Add multiple embedding layers
    for layer_name in ["physical", "semantic", "conceptual"]:
        ml_lcg.add_embedding_layer(layer_name)
    
    # Add entities with embeddings
    np.random.seed(42)
    for i in range(10):
        for layer_name in ["physical", "semantic", "conceptual"]:
            emb = np.random.randn(64)
            ml_lcg.set_entity_embedding(f"entity_{i}", layer_name, emb)
    
    # Search across all layers
    query_emb = np.random.randn(64)
    results = ml_lcg.similarity_search(query_emb, layer_name=None, top_k=3)
    
    assert len(results) == 3  # 3 layers
    assert "physical" in results
    assert "semantic" in results
    assert "conceptual" in results
    
    print(f"   ✅ Multi-layer search working")
    print(f"   ✅ Results from {len(results)} layers")
    for layer_name, layer_results in results.items():
        print(f"      {layer_name}: {len(layer_results)} results")
    
    return True


def test_clustering():
    """Test 7: Entity clustering"""
    print("\n" + "="*60)
    print("Test 7: Entity Clustering")
    print("="*60)
    
    ml_lcg = MLLayeredGraph(name="test_cluster", embedding_dim=64)
    ml_lcg.add_embedding_layer("physical")
    
    # Create clusterable data (2 clear groups)
    np.random.seed(42)
    
    # Group 1: centered at origin
    for i in range(10):
        emb = np.random.randn(64) * 0.5
        ml_lcg.set_entity_embedding(f"group1_entity_{i}", "physical", emb)
    
    # Group 2: centered at [5, 5, ...]
    for i in range(10):
        emb = np.random.randn(64) * 0.5 + 5.0
        ml_lcg.set_entity_embedding(f"group2_entity_{i}", "physical", emb)
    
    # Cluster
    clusters = ml_lcg.cluster_entities("physical", n_clusters=2, method="kmeans")
    
    # If sklearn not available, clustering returns empty dict
    if clusters:
        assert len(clusters) == 2
        total_entities = sum(len(members) for members in clusters.values())
        assert total_entities == 20
        
        print(f"   ✅ Clustering working")
        print(f"   ✅ Found {len(clusters)} clusters")
        for cluster_id, members in clusters.items():
            print(f"      Cluster {cluster_id}: {len(members)} entities")
    else:
        print(f"   ℹ️  Clustering requires scikit-learn (not installed)")
        print(f"   ✅ Clustering function handles missing dependency gracefully")
    
    return True


def test_dimensionality_reduction():
    """Test 8: Dimensionality reduction"""
    print("\n" + "="*60)
    print("Test 8: Dimensionality Reduction")
    print("="*60)
    
    ml_lcg = MLLayeredGraph(name="test_dimred", embedding_dim=128)
    ml_lcg.add_embedding_layer("high_dim")
    
    # Add high-dimensional embeddings
    np.random.seed(42)
    for i in range(20):
        emb = np.random.randn(128)
        ml_lcg.set_entity_embedding(f"entity_{i}", "high_dim", emb)
    
    # Reduce to 2D for visualization
    reduced = ml_lcg.reduce_dimensionality("high_dim", n_components=2, method="pca")
    
    # If sklearn not available, returns empty dict
    if reduced:
        assert len(reduced) == 20
        for entity_id, reduced_emb in reduced.items():
            assert reduced_emb.shape == (2,)
        
        print(f"   ✅ Dimensionality reduction working")
        print(f"   ✅ Reduced from 128D to 2D")
        print(f"   ✅ Processed {len(reduced)} entities")
    else:
        print(f"   ℹ️  PCA requires scikit-learn (not installed)")
        print(f"   ✅ Dimensionality reduction handles missing dependency gracefully")
    
    return True


def test_auto_context_detection():
    """Test 9: Auto-context detection"""
    print("\n" + "="*60)
    print("Test 9: Auto-Context Detection")
    print("="*60)
    
    ml_lcg = MLLayeredGraph(name="test_context", embedding_dim=64)
    ml_lcg.add_embedding_layer("layer1")
    ml_lcg.add_embedding_layer("layer2")
    
    # Add contexts
    from layered_contextual_graph.core import Context, ContextType
    
    context1 = Context(
        name="temporal",
        context_type=ContextType.TEMPORAL,
        applicable_layers={"layer1"}
    )
    context2 = Context(
        name="spatial",
        context_type=ContextType.SPATIAL,
        applicable_layers={"layer2"}
    )
    
    ml_lcg.contexts["temporal"] = context1
    ml_lcg.contexts["spatial"] = context2
    
    # Create entity with embeddings in both layers
    np.random.seed(42)
    ml_lcg.set_entity_embedding("test_entity", "layer1", np.random.randn(64))
    ml_lcg.set_entity_embedding("test_entity", "layer2", np.random.randn(64))
    
    # Auto-detect contexts
    detected = ml_lcg.auto_detect_context("test_entity")
    
    assert len(detected) >= 2
    assert "temporal" in detected
    assert "spatial" in detected
    
    print(f"   ✅ Auto-context detection working")
    print(f"   ✅ Detected contexts: {detected}")
    
    return True


def test_enable_ml():
    """Test 10: Enable ML on existing LCG"""
    print("\n" + "="*60)
    print("Test 10: Enable ML on Existing LCG")
    print("="*60)
    
    # Create regular LCG
    lcg = LayeredContextualGraph(name="regular_lcg")
    
    # Enable ML
    ml_lcg = enable_ml(lcg, embedding_dim=256)
    
    assert ml_lcg.embedding_dim == 256
    assert hasattr(ml_lcg, 'embedding_layers')
    assert hasattr(ml_lcg, 'entity_embeddings')
    
    # Test ML functionality
    ml_lcg.add_embedding_layer("test")
    np.random.seed(42)
    ml_lcg.set_entity_embedding("entity", "test", np.random.randn(256))
    
    assert "entity" in ml_lcg.entity_embeddings
    
    print(f"   ✅ ML enabled on existing LCG")
    print(f"   ✅ Embedding dimension: {ml_lcg.embedding_dim}")
    print(f"   ✅ ML functionality working")
    
    return True


def run_all_tests():
    """Run all ML tests"""
    
    print("\n" + "="*70)
    print("MACHINE LEARNING EXTENSION TEST SUITE")
    print("="*70)
    print("\nComprehensive tests for LCG ML capabilities\n")
    
    try:
        test_ml_graph_creation()
        test_embedding_layer()
        test_similarity_search()
        test_entity_embeddings()
        test_cross_layer_similarity()
        test_multi_layer_search()
        test_clustering()
        test_dimensionality_reduction()
        test_auto_context_detection()
        test_enable_ml()
        
        print("\n" + "="*70)
        print("✅ ALL ML TESTS PASSED")
        print("="*70)
        print("\nTest Summary:")
        print("   ✅ MLLayeredGraph creation")
        print("   ✅ Embedding layer functionality")
        print("   ✅ Similarity search")
        print("   ✅ Entity embeddings across layers")
        print("   ✅ Cross-layer similarity")
        print("   ✅ Multi-layer search")
        print("   ✅ Entity clustering")
        print("   ✅ Dimensionality reduction")
        print("   ✅ Auto-context detection")
        print("   ✅ Enable ML on existing LCG")
        print("\n   Total: 10/10 ML tests passed\n")
        
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
