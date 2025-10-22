"""
Test Suite for Cross-Modal Analysis
====================================

Tests for CrossModalAnalyzer and ModalMetrics.
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

from core.multi_modal_hypergraph import MultiModalHypergraph
from core.cross_modal_analyzer import CrossModalAnalyzer, CrossModalPattern, InterModalRelationship
from core.modal_metrics import ModalMetrics, MultiModalCentrality, ModalCorrelation


# Mock Hypergraph for testing
class MockHypergraph:
    """Mock hypergraph for testing"""
    def __init__(self, data):
        self.data = data
        self.incidences = type('obj', (object,), {'data': data})()
        if 'nodes' in data.columns:
            self._nodes = set(data['nodes'].unique())
    
    def nodes(self):
        return self._nodes


def create_test_mmhg():
    """Create test multi-modal hypergraph"""
    # Create test data
    friendships = pl.DataFrame([
        {"edges": "f1", "nodes": "Alice", "weight": 1.0},
        {"edges": "f1", "nodes": "Bob", "weight": 1.0},
        {"edges": "f2", "nodes": "Bob", "weight": 1.0},
        {"edges": "f2", "nodes": "Charlie", "weight": 1.0},
        {"edges": "f3", "nodes": "Alice", "weight": 1.0},
        {"edges": "f3", "nodes": "Charlie", "weight": 1.0},
        {"edges": "f4", "nodes": "David", "weight": 1.0},
        {"edges": "f4", "nodes": "Eve", "weight": 1.0},
    ])
    
    collaborations = pl.DataFrame([
        {"edges": "c1", "nodes": "Alice", "weight": 1.0},
        {"edges": "c1", "nodes": "Bob", "weight": 1.0},
        {"edges": "c2", "nodes": "Alice", "weight": 1.0},
        {"edges": "c2", "nodes": "David", "weight": 1.0},
        {"edges": "c3", "nodes": "Bob", "weight": 1.0},
        {"edges": "c3", "nodes": "Charlie", "weight": 1.0},
    ])
    
    communications = pl.DataFrame([
        {"edges": "m1", "nodes": "Alice", "weight": 1.0},
        {"edges": "m1", "nodes": "Bob", "weight": 1.0},
        {"edges": "m2", "nodes": "Charlie", "weight": 1.0},
        {"edges": "m2", "nodes": "David", "weight": 1.0},
        {"edges": "m3", "nodes": "Eve", "weight": 1.0},
        {"edges": "m3", "nodes": "Alice", "weight": 1.0},
    ])
    
    # Create multi-modal hypergraph
    mmhg = MultiModalHypergraph(name="test_network")
    mmhg.add_modality("friendships", MockHypergraph(friendships), weight=1.0)
    mmhg.add_modality("collaborations", MockHypergraph(collaborations), weight=2.0)
    mmhg.add_modality("communications", MockHypergraph(communications), weight=1.5)
    
    return mmhg


def test_cross_modal_analyzer_init():
    """Test 1: CrossModalAnalyzer initialization"""
    print("\n" + "="*60)
    print("Test 1: CrossModalAnalyzer Initialization")
    print("="*60)
    
    mmhg = create_test_mmhg()
    analyzer = CrossModalAnalyzer(mmhg)
    
    assert analyzer.mmhg == mmhg, "Should store MMHG reference"
    assert isinstance(analyzer.pattern_cache, dict), "Should have pattern cache"
    assert isinstance(analyzer.relationship_cache, dict), "Should have relationship cache"
    
    print("✅ Analyzer initialization test passed")
    return mmhg, analyzer


def test_frequent_pattern_mining(mmhg, analyzer):
    """Test 2: Frequent pattern mining"""
    print("\n" + "="*60)
    print("Test 2: Frequent Pattern Mining")
    print("="*60)
    
    patterns = analyzer.mine_frequent_patterns(min_support=1, min_modalities=2)
    
    assert isinstance(patterns, list), "Should return list of patterns"
    
    if patterns:
        # Check first pattern
        pattern = patterns[0]
        assert isinstance(pattern, CrossModalPattern), "Should be CrossModalPattern"
        assert hasattr(pattern, 'pattern_id'), "Should have pattern_id"
        assert hasattr(pattern, 'pattern_type'), "Should have pattern_type"
        assert hasattr(pattern, 'modalities'), "Should have modalities"
        assert hasattr(pattern, 'support'), "Should have support"
        assert hasattr(pattern, 'confidence'), "Should have confidence"
    
    print(f"   Patterns mined: {len(patterns)}")
    if patterns:
        print(f"   Sample pattern: {patterns[0]}")
    
    print("✅ Pattern mining test passed")
    return patterns


def test_anomaly_detection(mmhg, analyzer):
    """Test 3: Anomaly detection"""
    print("\n" + "="*60)
    print("Test 3: Anomaly Detection")
    print("="*60)
    
    # Test statistical method
    anomalies_stat = analyzer.detect_anomalies(method="statistical", contamination=0.2)
    
    assert isinstance(anomalies_stat, list), "Should return list"
    print(f"   Statistical anomalies: {len(anomalies_stat)}")
    
    # Test isolation forest if sklearn available
    try:
        anomalies_iso = analyzer.detect_anomalies(method="isolation_forest", contamination=0.2)
        assert isinstance(anomalies_iso, list), "Should return list"
        print(f"   Isolation Forest anomalies: {len(anomalies_iso)}")
    except:
        print("   Isolation Forest not available (sklearn required)")
    
    # Check anomaly structure
    if anomalies_stat:
        anomaly = anomalies_stat[0]
        assert 'entity_id' in anomaly, "Should have entity_id"
        assert 'anomaly_score' in anomaly, "Should have anomaly_score"
        assert 'method' in anomaly, "Should have method"
    
    print("✅ Anomaly detection test passed")


def test_implicit_relationship_inference(mmhg, analyzer):
    """Test 4: Implicit relationship inference"""
    print("\n" + "="*60)
    print("Test 4: Implicit Relationship Inference")
    print("="*60)
    
    relationships = analyzer.infer_implicit_relationships(
        source_modality="friendships",
        target_modality="collaborations",
        bridging_modalities=["communications"]
    )
    
    assert isinstance(relationships, list), "Should return list"
    
    if relationships:
        rel = relationships[0]
        assert isinstance(rel, InterModalRelationship), "Should be InterModalRelationship"
        assert hasattr(rel, 'source_entity'), "Should have source_entity"
        assert hasattr(rel, 'target_entity'), "Should have target_entity"
        assert hasattr(rel, 'source_modality'), "Should have source_modality"
        assert hasattr(rel, 'target_modality'), "Should have target_modality"
        assert hasattr(rel, 'strength'), "Should have strength"
    
    print(f"   Relationships inferred: {len(relationships)}")
    if relationships:
        print(f"   Sample: {relationships[0]}")
    
    print("✅ Relationship inference test passed")


def test_pattern_significance(mmhg, analyzer):
    """Test 5: Pattern significance"""
    print("\n" + "="*60)
    print("Test 5: Pattern Significance")
    print("="*60)
    
    patterns = analyzer.mine_frequent_patterns(min_support=1)
    
    if patterns:
        pattern = patterns[0]
        significance = analyzer.compute_pattern_significance(pattern)
        
        assert isinstance(significance, float), "Should return float"
        assert 0 <= significance <= 1, "Should be in [0, 1]"
        
        print(f"   Pattern: {pattern.pattern_type}")
        print(f"   Significance: {significance:.4f}")
    
    print("✅ Pattern significance test passed")


def test_pattern_ranking(mmhg, analyzer):
    """Test 6: Pattern ranking"""
    print("\n" + "="*60)
    print("Test 6: Pattern Ranking")
    print("="*60)
    
    patterns = analyzer.mine_frequent_patterns(min_support=1)
    
    if patterns:
        # Test different ranking criteria
        for criteria in ["support", "confidence", "modalities", "combined"]:
            ranked = analyzer.rank_patterns_by_interestingness(patterns, criteria=criteria)
            assert isinstance(ranked, list), f"Should return list for {criteria}"
            assert len(ranked) == len(patterns), "Length should match"
            print(f"   ✅ Ranking by {criteria}")
    
    print("✅ Pattern ranking test passed")


def test_pattern_report(mmhg, analyzer):
    """Test 7: Pattern report generation"""
    print("\n" + "="*60)
    print("Test 7: Pattern Report Generation")
    print("="*60)
    
    patterns = analyzer.mine_frequent_patterns(min_support=1)
    report = analyzer.generate_pattern_report(patterns)
    
    assert isinstance(report, dict), "Should return dict"
    assert 'total_patterns' in report, "Should have total_patterns"
    
    if patterns:
        assert 'patterns_by_type' in report, "Should have patterns_by_type"
        assert 'total_support' in report, "Should have total_support"
        assert 'avg_confidence' in report, "Should have avg_confidence"
        
        print(f"   Total patterns: {report['total_patterns']}")
        print(f"   Total support: {report['total_support']}")
        print(f"   Avg confidence: {report['avg_confidence']:.3f}")
    
    print("✅ Pattern report test passed")


def test_modal_metrics_init():
    """Test 8: ModalMetrics initialization"""
    print("\n" + "="*60)
    print("Test 8: ModalMetrics Initialization")
    print("="*60)
    
    mmhg = create_test_mmhg()
    metrics = ModalMetrics(mmhg)
    
    assert metrics.mmhg == mmhg, "Should store MMHG reference"
    assert isinstance(metrics.centrality_cache, dict), "Should have centrality cache"
    
    print("✅ Metrics initialization test passed")
    return mmhg, metrics


def test_degree_centrality(mmhg, metrics):
    """Test 9: Degree centrality"""
    print("\n" + "="*60)
    print("Test 9: Degree Centrality")
    print("="*60)
    
    # Test for Alice in friendships
    degree = metrics.compute_degree_centrality("friendships", "Alice", normalized=True)
    
    assert isinstance(degree, float), "Should return float"
    assert degree >= 0, "Should be non-negative"
    
    print(f"   Alice's degree centrality (normalized): {degree:.3f}")
    
    # Test without normalization
    degree_unnorm = metrics.compute_degree_centrality("friendships", "Alice", normalized=False)
    print(f"   Alice's degree centrality (raw): {degree_unnorm:.0f}")
    
    print("✅ Degree centrality test passed")


def test_betweenness_centrality(mmhg, metrics):
    """Test 10: Betweenness centrality"""
    print("\n" + "="*60)
    print("Test 10: Betweenness Centrality")
    print("="*60)
    
    betweenness = metrics.compute_betweenness_centrality("friendships", "Bob")
    
    assert isinstance(betweenness, float), "Should return float"
    assert betweenness >= 0, "Should be non-negative"
    
    print(f"   Bob's betweenness centrality: {betweenness:.3f}")
    
    print("✅ Betweenness centrality test passed")


def test_closeness_centrality(mmhg, metrics):
    """Test 11: Closeness centrality"""
    print("\n" + "="*60)
    print("Test 11: Closeness Centrality")
    print("="*60)
    
    closeness = metrics.compute_closeness_centrality("friendships", "Alice")
    
    assert isinstance(closeness, float), "Should return float"
    assert closeness >= 0, "Should be non-negative"
    
    print(f"   Alice's closeness centrality: {closeness:.3f}")
    
    print("✅ Closeness centrality test passed")


def test_eigenvector_centrality(mmhg, metrics):
    """Test 12: Eigenvector centrality"""
    print("\n" + "="*60)
    print("Test 12: Eigenvector Centrality")
    print("="*60)
    
    eigenvector = metrics.compute_eigenvector_centrality("friendships", "Alice")
    
    assert isinstance(eigenvector, float), "Should return float"
    assert eigenvector >= 0, "Should be non-negative"
    
    print(f"   Alice's eigenvector centrality: {eigenvector:.3f}")
    
    print("✅ Eigenvector centrality test passed")


def test_batch_centrality(mmhg, metrics):
    """Test 13: Batch centrality computation"""
    print("\n" + "="*60)
    print("Test 13: Batch Centrality Computation")
    print("="*60)
    
    entities = ["Alice", "Bob", "Charlie"]
    results = metrics.compute_multi_modal_centrality_batch(
        entities,
        metric="degree",
        aggregation="weighted_average"
    )
    
    assert isinstance(results, list), "Should return list"
    assert len(results) == len(entities), "Length should match"
    
    for result in results:
        assert isinstance(result, MultiModalCentrality), "Should be MultiModalCentrality"
        assert result.rank is not None, "Should have rank"
        assert result.percentile is not None, "Should have percentile"
    
    print(f"   Computed centrality for {len(results)} entities")
    print("   Rankings:")
    for result in results:
        print(f"      {result.entity_id}: rank={result.rank}, score={result.aggregated:.3f}")
    
    print("✅ Batch centrality test passed")


def test_correlation_matrix(mmhg, metrics):
    """Test 14: Correlation matrix"""
    print("\n" + "="*60)
    print("Test 14: Correlation Matrix")
    print("="*60)
    
    correlations = metrics.compute_correlation_matrix(method="jaccard")
    
    assert isinstance(correlations, dict), "Should return dict"
    
    for key, corr in correlations.items():
        assert isinstance(corr, ModalCorrelation), "Should be ModalCorrelation"
        assert hasattr(corr, 'modality_a'), "Should have modality_a"
        assert hasattr(corr, 'modality_b'), "Should have modality_b"
        assert hasattr(corr, 'correlation_value'), "Should have correlation_value"
        assert hasattr(corr, 'shared_entities'), "Should have shared_entities"
    
    print(f"   Computed {len(correlations)} correlations")
    for key, corr in list(correlations.items())[:3]:
        print(f"   {corr.modality_a} ↔ {corr.modality_b}: {corr.correlation_value:.3f}")
    
    print("✅ Correlation matrix test passed")


def test_modal_diversity(mmhg, metrics):
    """Test 15: Modal diversity"""
    print("\n" + "="*60)
    print("Test 15: Modal Diversity")
    print("="*60)
    
    diversity = metrics.compute_modal_diversity("Alice")
    
    assert isinstance(diversity, float), "Should return float"
    assert 0 <= diversity <= 1, "Should be in [0, 1]"
    
    print(f"   Alice's modal diversity: {diversity:.3f}")
    
    # Test for entity in single modality
    diversity_eve = metrics.compute_modal_diversity("Eve")
    print(f"   Eve's modal diversity: {diversity_eve:.3f}")
    
    print("✅ Modal diversity test passed")


def test_metrics_report(mmhg, metrics):
    """Test 16: Metrics report"""
    print("\n" + "="*60)
    print("Test 16: Metrics Report")
    print("="*60)
    
    report = metrics.generate_metrics_report()
    
    assert isinstance(report, dict), "Should return dict"
    assert 'total_entities' in report, "Should have total_entities"
    assert 'total_modalities' in report, "Should have total_modalities"
    
    print(f"   Total entities: {report['total_entities']}")
    print(f"   Total modalities: {report['total_modalities']}")
    
    if 'avg_correlation' in report:
        print(f"   Avg correlation: {report['avg_correlation']:.3f}")
    
    print("✅ Metrics report test passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("CROSS-MODAL ANALYSIS TEST SUITE")
    print("="*70)
    
    try:
        # CrossModalAnalyzer tests
        mmhg1, analyzer = test_cross_modal_analyzer_init()
        patterns = test_frequent_pattern_mining(mmhg1, analyzer)
        test_anomaly_detection(mmhg1, analyzer)
        test_implicit_relationship_inference(mmhg1, analyzer)
        test_pattern_significance(mmhg1, analyzer)
        test_pattern_ranking(mmhg1, analyzer)
        test_pattern_report(mmhg1, analyzer)
        
        # ModalMetrics tests
        mmhg2, metrics = test_modal_metrics_init()
        test_degree_centrality(mmhg2, metrics)
        test_betweenness_centrality(mmhg2, metrics)
        test_closeness_centrality(mmhg2, metrics)
        test_eigenvector_centrality(mmhg2, metrics)
        test_batch_centrality(mmhg2, metrics)
        test_correlation_matrix(mmhg2, metrics)
        test_modal_diversity(mmhg2, metrics)
        test_metrics_report(mmhg2, metrics)
        
        # Summary
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nTest Summary:")
        print("   CrossModalAnalyzer Tests:")
        print("      ✅ Analyzer initialization")
        print("      ✅ Frequent pattern mining")
        print("      ✅ Anomaly detection")
        print("      ✅ Relationship inference")
        print("      ✅ Pattern significance")
        print("      ✅ Pattern ranking")
        print("      ✅ Pattern report")
        print("\n   ModalMetrics Tests:")
        print("      ✅ Metrics initialization")
        print("      ✅ Degree centrality")
        print("      ✅ Betweenness centrality")
        print("      ✅ Closeness centrality")
        print("      ✅ Eigenvector centrality")
        print("      ✅ Batch centrality")
        print("      ✅ Correlation matrix")
        print("      ✅ Modal diversity")
        print("      ✅ Metrics report")
        print("\n   Total: 16/16 tests passed\n")
        
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
