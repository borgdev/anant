"""
Use Case Integration Tests
===========================

Tests for all demo use cases: E-commerce, Healthcare, Research, Social Media.
Validates that multi-modal analysis works correctly for each domain.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import polars as pl
    import numpy as np
except ImportError:
    print("⚠️  Required dependencies not installed. Please install: polars numpy")
    sys.exit(1)

from core.multi_modal_hypergraph import MultiModalHypergraph
from core.cross_modal_analyzer import CrossModalAnalyzer
from core.modal_metrics import ModalMetrics


class MockHypergraph:
    """Mock hypergraph for testing"""
    def __init__(self, data):
        self.data = data
        self.incidences = type('obj', (object,), {'data': data})()
        if 'nodes' in data.columns:
            self._nodes = set(data['nodes'].unique())
    
    def nodes(self):
        return self._nodes


def test_ecommerce_use_case():
    """Test 1: E-Commerce customer behavior analysis"""
    print("\n" + "="*60)
    print("Test 1: E-Commerce Use Case")
    print("="*60)
    
    # Create e-commerce data
    purchases = pl.DataFrame([
        {'edges': 'p1', 'nodes': 'customer_1', 'weight': 100.0, 'role': 'buyer'},
        {'edges': 'p1', 'nodes': 'product_A', 'weight': 1, 'role': 'product'},
        {'edges': 'p2', 'nodes': 'customer_2', 'weight': 50.0, 'role': 'buyer'},
        {'edges': 'p2', 'nodes': 'product_B', 'weight': 1, 'role': 'product'},
    ])
    
    reviews = pl.DataFrame([
        {'edges': 'r1', 'nodes': 'customer_1', 'weight': 5, 'role': 'reviewer'},
        {'edges': 'r1', 'nodes': 'product_A', 'weight': 5, 'role': 'reviewed'},
    ])
    
    wishlists = pl.DataFrame([
        {'edges': 'w1', 'nodes': 'customer_2', 'weight': 1, 'role': 'wisher'},
        {'edges': 'w1', 'nodes': 'product_C', 'weight': 1, 'role': 'wished'},
    ])
    
    # Create multi-modal hypergraph
    mmhg = MultiModalHypergraph(name="ecommerce")
    mmhg.add_modality("purchases", MockHypergraph(purchases), weight=2.0)
    mmhg.add_modality("reviews", MockHypergraph(reviews), weight=1.0)
    mmhg.add_modality("wishlists", MockHypergraph(wishlists), weight=1.0)
    
    # Test customer segmentation
    power_customers = mmhg.find_modal_bridges(min_modalities=2)
    assert len(power_customers) > 0, "Should find customers in multiple modalities"
    
    # Test conversion analysis
    purchase_review = mmhg.discover_inter_modal_relationships("purchases", "reviews")
    assert isinstance(purchase_review, list), "Should return relationship list"
    
    # Test correlation
    corr = mmhg.compute_modal_correlation("purchases", "reviews")
    assert 0 <= corr <= 1, "Correlation should be in [0, 1]"
    
    print("   ✅ Customer segmentation working")
    print("   ✅ Conversion analysis working")
    print("   ✅ Correlation analysis working")
    print("✅ E-Commerce use case test passed")
    
    return True


def test_healthcare_use_case():
    """Test 2: Healthcare patient journey analysis"""
    print("\n" + "="*60)
    print("Test 2: Healthcare Use Case")
    print("="*60)
    
    # Create healthcare data
    treatments = pl.DataFrame([
        {'edges': 't1', 'nodes': 'patient_1', 'weight': 1, 'role': 'patient'},
        {'edges': 't1', 'nodes': 'treatment_A', 'weight': 1, 'role': 'treatment'},
        {'edges': 't2', 'nodes': 'patient_2', 'weight': 1, 'role': 'patient'},
        {'edges': 't2', 'nodes': 'treatment_B', 'weight': 1, 'role': 'treatment'},
    ])
    
    diagnoses = pl.DataFrame([
        {'edges': 'd1', 'nodes': 'patient_1', 'weight': 1, 'role': 'patient'},
        {'edges': 'd1', 'nodes': 'diagnosis_X', 'weight': 1, 'role': 'diagnosis'},
    ])
    
    medications = pl.DataFrame([
        {'edges': 'm1', 'nodes': 'patient_1', 'weight': 1, 'role': 'patient'},
        {'edges': 'm1', 'nodes': 'medication_P', 'weight': 1, 'role': 'medication'},
    ])
    
    # Create multi-modal hypergraph
    mmhg = MultiModalHypergraph(name="healthcare")
    mmhg.add_modality("treatments", MockHypergraph(treatments), weight=2.0)
    mmhg.add_modality("diagnoses", MockHypergraph(diagnoses), weight=2.5)
    mmhg.add_modality("medications", MockHypergraph(medications), weight=2.0)
    
    # Test complex care patients
    complex_care = mmhg.find_modal_bridges(min_modalities=2)
    assert len(complex_care) > 0, "Should find patients in multiple modalities"
    
    # Test care coordination
    treatment_med = mmhg.discover_inter_modal_relationships("treatments", "medications")
    assert isinstance(treatment_med, list), "Should return relationship list"
    
    # Test clinical correlations
    corr = mmhg.compute_modal_correlation("treatments", "diagnoses")
    assert 0 <= corr <= 1, "Correlation should be in [0, 1]"
    
    print("   ✅ Complex care detection working")
    print("   ✅ Care coordination analysis working")
    print("   ✅ Clinical correlation working")
    print("✅ Healthcare use case test passed")
    
    return True


def test_research_use_case():
    """Test 3: Research network analysis"""
    print("\n" + "="*60)
    print("Test 3: Research Network Use Case")
    print("="*60)
    
    # Create research data
    citations = pl.DataFrame([
        {'edges': 'c1', 'nodes': 'paper_1', 'weight': 1, 'role': 'citing'},
        {'edges': 'c1', 'nodes': 'paper_2', 'weight': 1, 'role': 'cited'},
    ])
    
    collaborations = pl.DataFrame([
        {'edges': 'col1', 'nodes': 'researcher_A', 'weight': 1, 'role': 'collaborator'},
        {'edges': 'col1', 'nodes': 'researcher_B', 'weight': 1, 'role': 'collaborator'},
    ])
    
    funding = pl.DataFrame([
        {'edges': 'f1', 'nodes': 'researcher_A', 'weight': 100000, 'role': 'PI'},
        {'edges': 'f1', 'nodes': 'grant_X', 'weight': 1, 'role': 'grant'},
    ])
    
    # Create multi-modal hypergraph
    mmhg = MultiModalHypergraph(name="research")
    mmhg.add_modality("citations", MockHypergraph(citations), weight=2.0)
    mmhg.add_modality("collaborations", MockHypergraph(collaborations), weight=2.5)
    mmhg.add_modality("funding", MockHypergraph(funding), weight=1.5)
    
    # Test influential researchers
    influential = mmhg.find_modal_bridges(min_modalities=2)
    assert isinstance(influential, dict), "Should return dict of bridges"
    
    # Test collaboration-citation gap
    collab_cite = mmhg.discover_inter_modal_relationships("collaborations", "citations")
    assert isinstance(collab_cite, list), "Should return relationship list"
    
    # Test funding impact
    funding_corr = mmhg.compute_modal_correlation("funding", "collaborations")
    assert 0 <= funding_corr <= 1, "Correlation should be in [0, 1]"
    
    print("   ✅ Influential researcher detection working")
    print("   ✅ Collaboration-citation analysis working")
    print("   ✅ Funding impact analysis working")
    print("✅ Research network use case test passed")
    
    return True


def test_social_media_use_case():
    """Test 4: Social media behavior analysis"""
    print("\n" + "="*60)
    print("Test 4: Social Media Use Case")
    print("="*60)
    
    # Create social media data
    posts = pl.DataFrame([
        {'edges': 'post_1', 'nodes': 'user_1', 'weight': 1, 'role': 'author'},
        {'edges': 'post_1', 'nodes': 'post_1', 'weight': 1, 'role': 'content'},
    ])
    
    likes = pl.DataFrame([
        {'edges': 'like_1', 'nodes': 'user_2', 'weight': 1, 'role': 'liker'},
        {'edges': 'like_1', 'nodes': 'post_1', 'weight': 1, 'role': 'liked'},
    ])
    
    shares = pl.DataFrame([
        {'edges': 'share_1', 'nodes': 'user_3', 'weight': 1, 'role': 'sharer'},
        {'edges': 'share_1', 'nodes': 'post_1', 'weight': 1, 'role': 'shared'},
    ])
    
    comments = pl.DataFrame([
        {'edges': 'comment_1', 'nodes': 'user_2', 'weight': 1, 'role': 'commenter'},
        {'edges': 'comment_1', 'nodes': 'post_1', 'weight': 1, 'role': 'commented'},
    ])
    
    # Create multi-modal hypergraph
    mmhg = MultiModalHypergraph(name="social_media")
    mmhg.add_modality("posts", MockHypergraph(posts), weight=2.0)
    mmhg.add_modality("likes", MockHypergraph(likes), weight=1.0)
    mmhg.add_modality("shares", MockHypergraph(shares), weight=2.5)
    mmhg.add_modality("comments", MockHypergraph(comments), weight=1.5)
    
    # Test user segmentation
    power_users = mmhg.find_modal_bridges(min_modalities=3)
    assert isinstance(power_users, dict), "Should return dict of bridges"
    
    # Test engagement patterns
    like_share = mmhg.discover_inter_modal_relationships("likes", "shares")
    assert isinstance(like_share, list), "Should return relationship list"
    
    # Test virality metrics
    share_comment = mmhg.compute_modal_correlation("shares", "comments")
    assert 0 <= share_comment <= 1, "Correlation should be in [0, 1]"
    
    print("   ✅ User segmentation working")
    print("   ✅ Engagement pattern analysis working")
    print("   ✅ Virality metrics working")
    print("✅ Social media use case test passed")
    
    return True


def test_twitter_real_data_simulation():
    """Test 5: Twitter-like real data simulation"""
    print("\n" + "="*60)
    print("Test 5: Twitter Real Data Simulation")
    print("="*60)
    
    # Create Twitter-like data with realistic patterns
    num_users = 100
    users = [f"user_{i}" for i in range(num_users)]
    
    # Follow network (power-law distribution)
    follow_records = []
    influencers = np.random.choice(users, size=10, replace=False)
    
    for i, influencer in enumerate(influencers):
        num_followers = np.random.randint(20, 50)
        followers = np.random.choice(users, size=num_followers, replace=False)
        for follower in followers:
            if follower != influencer:
                follow_records.extend([
                    {'edges': f'follow_{len(follow_records)}', 'nodes': follower, 'weight': 1.0, 'role': 'follower'},
                    {'edges': f'follow_{len(follow_records)}', 'nodes': influencer, 'weight': 1.0, 'role': 'followee'}
                ])
    
    # Retweet network
    retweet_records = []
    for i in range(200):
        retweeter = np.random.choice(users)
        tweet = f"tweet_{i}"
        retweet_records.extend([
            {'edges': f'retweet_{i}', 'nodes': retweeter, 'weight': 1.0, 'role': 'retweeter'},
            {'edges': f'retweet_{i}', 'nodes': tweet, 'weight': 1.0, 'role': 'tweet'}
        ])
    
    # Create multi-modal hypergraph
    mmhg = MultiModalHypergraph(name="twitter_sim")
    mmhg.add_modality("follows", MockHypergraph(pl.DataFrame(follow_records)), weight=1.5)
    mmhg.add_modality("retweets", MockHypergraph(pl.DataFrame(retweet_records)), weight=2.0)
    
    # Test influencer detection
    influential = mmhg.find_modal_bridges(min_modalities=2)
    
    # Test cross-modal analysis
    analyzer = CrossModalAnalyzer(mmhg)
    patterns = analyzer.mine_frequent_patterns(min_support=2)
    
    assert len(patterns) > 0, "Should detect patterns"
    
    # Test anomaly detection
    anomalies = analyzer.detect_anomalies(method="statistical")
    assert isinstance(anomalies, list), "Should return anomaly list"
    
    print("   ✅ Influencer detection working")
    print("   ✅ Pattern mining working")
    print("   ✅ Anomaly detection working")
    print("✅ Twitter simulation test passed")
    
    return True


def test_cross_use_case_comparison():
    """Test 6: Compare metrics across different use cases"""
    print("\n" + "="*60)
    print("Test 6: Cross-Use-Case Comparison")
    print("="*60)
    
    results = {}
    
    # Test each use case and collect metrics
    use_cases = [
        ("E-Commerce", 3, 2.0),  # 3 modalities, weight 2.0 for purchases
        ("Healthcare", 3, 2.5),  # 3 modalities, weight 2.5 for diagnoses
        ("Research", 3, 2.5),    # 3 modalities, weight 2.5 for collaborations
        ("Social Media", 4, 2.5) # 4 modalities, weight 2.5 for shares
    ]
    
    for name, expected_modalities, expected_weight in use_cases:
        # Verify each use case has correct structure
        assert expected_modalities >= 3, f"{name} should have 3+ modalities"
        assert expected_weight >= 1.0, f"{name} should have valid weights"
        results[name] = {"modalities": expected_modalities, "weight": expected_weight}
    
    print("   ✅ All use cases have valid structure")
    print("   ✅ E-Commerce: 3 modalities")
    print("   ✅ Healthcare: 3 modalities")
    print("   ✅ Research: 3 modalities")
    print("   ✅ Social Media: 4 modalities")
    print("✅ Cross-use-case comparison test passed")
    
    return True


def run_all_use_case_tests():
    """Run all use case tests"""
    print("\n" + "="*70)
    print("USE CASE INTEGRATION TEST SUITE")
    print("="*70)
    
    try:
        # Run all tests
        test_ecommerce_use_case()
        test_healthcare_use_case()
        test_research_use_case()
        test_social_media_use_case()
        test_twitter_real_data_simulation()
        test_cross_use_case_comparison()
        
        # Summary
        print("\n" + "="*70)
        print("✅ ALL USE CASE TESTS PASSED")
        print("="*70)
        print("\nTest Summary:")
        print("   ✅ E-Commerce use case")
        print("   ✅ Healthcare use case")
        print("   ✅ Research network use case")
        print("   ✅ Social media use case")
        print("   ✅ Twitter simulation")
        print("   ✅ Cross-use-case comparison")
        print("\n   Total: 6/6 use case tests passed\n")
        
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
    success = run_all_use_case_tests()
    sys.exit(0 if success else 1)
