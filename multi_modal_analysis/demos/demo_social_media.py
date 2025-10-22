"""
Social Media Multi-Modal Analysis Demo
======================================

Demonstrates multi-modal relationship analysis for social media behavior.

Modalities:
- Posts: User-post relationships
- Likes: User-like relationships
- Shares: User-share relationships
- Comments: User-comment relationships
- Follows: User-follow relationships

Cross-Modal Insights:
- Engagement patterns across content types
- Influence propagation mechanisms
- Community formation dynamics
- Content virality factors
"""

import sys
from pathlib import Path
import random
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import polars as pl
    import numpy as np
except ImportError:
    print("⚠️  Required dependencies not installed. Please install: polars numpy")
    sys.exit(1)

from core.multi_modal_hypergraph import MultiModalHypergraph


class MockHypergraph:
    """Mock hypergraph for demo"""
    def __init__(self, data):
        self.data = data
        self.incidences = type('obj', (object,), {'data': data})()
        if 'nodes' in data.columns:
            self._nodes = set(data['nodes'].unique())
    
    def nodes(self):
        return self._nodes


def generate_social_media_data(
    num_users: int = 400,
    num_posts: int = 1000
):
    """Generate synthetic social media data"""
    
    print("📊 Generating Social Media Data...")
    print(f"   Users: {num_users}")
    print(f"   Posts: {num_posts}")
    
    users = [f"user_{i:04d}" for i in range(num_users)]
    posts = [f"post_{i:05d}" for i in range(num_posts)]
    
    # Generate posts
    post_records = []
    for post in posts:
        author = random.choice(users)
        post_records.extend([
            {'edges': post, 'nodes': author, 'weight': 1.0, 'role': 'author'},
            {'edges': post, 'nodes': post, 'weight': 1.0, 'role': 'content'}
        ])
    
    # Generate likes
    like_records = []
    for i in range(num_posts * 5):  # ~5 likes per post
        like_id = f"like_{i:06d}"
        user = random.choice(users)
        post = random.choice(posts)
        
        like_records.extend([
            {'edges': like_id, 'nodes': user, 'weight': 1.0, 'role': 'liker'},
            {'edges': like_id, 'nodes': post, 'weight': 1.0, 'role': 'liked_post'}
        ])
    
    # Generate shares
    share_records = []
    for i in range(num_posts * 2):  # ~2 shares per post
        share_id = f"share_{i:06d}"
        user = random.choice(users)
        post = random.choice(posts)
        
        share_records.extend([
            {'edges': share_id, 'nodes': user, 'weight': 1.0, 'role': 'sharer'},
            {'edges': share_id, 'nodes': post, 'weight': 1.0, 'role': 'shared_post'}
        ])
    
    # Generate comments
    comment_records = []
    for i in range(num_posts * 3):  # ~3 comments per post
        comment_id = f"comment_{i:06d}"
        user = random.choice(users)
        post = random.choice(posts)
        
        comment_records.extend([
            {'edges': comment_id, 'nodes': user, 'weight': 1.0, 'role': 'commenter'},
            {'edges': comment_id, 'nodes': post, 'weight': 1.0, 'role': 'commented_post'}
        ])
    
    # Generate follows
    follow_records = []
    for i in range(num_users * 10):  # ~10 follows per user
        follow_id = f"follow_{i:06d}"
        follower = random.choice(users)
        followee = random.choice(users)
        
        if follower != followee:
            follow_records.extend([
                {'edges': follow_id, 'nodes': follower, 'weight': 1.0, 'role': 'follower'},
                {'edges': follow_id, 'nodes': followee, 'weight': 1.0, 'role': 'followee'}
            ])
    
    return {
        'posts': pl.DataFrame(post_records),
        'likes': pl.DataFrame(like_records),
        'shares': pl.DataFrame(share_records),
        'comments': pl.DataFrame(comment_records),
        'follows': pl.DataFrame(follow_records)
    }


def demo_social_media():
    """Demo: Social media multi-modal analysis"""
    
    print("\n" + "="*70)
    print("📱 Social Media Multi-Modal Analysis Demo")
    print("="*70)
    
    # Generate data
    data = generate_social_media_data()
    
    # Create hypergraphs
    print("\n📦 Creating Social Media Hypergraphs...")
    post_hg = MockHypergraph(data['posts'])
    like_hg = MockHypergraph(data['likes'])
    share_hg = MockHypergraph(data['shares'])
    comment_hg = MockHypergraph(data['comments'])
    follow_hg = MockHypergraph(data['follows'])
    
    # Build multi-modal hypergraph
    print("\n🔗 Constructing Multi-Modal Social Network...")
    mmhg = MultiModalHypergraph(name="social_network")
    
    mmhg.add_modality("posts", post_hg, weight=2.0,
                     description="Content creation")
    mmhg.add_modality("likes", like_hg, weight=1.0,
                     description="Like interactions")
    mmhg.add_modality("shares", share_hg, weight=2.5,
                     description="Share/repost interactions")
    mmhg.add_modality("comments", comment_hg, weight=1.5,
                     description="Comment interactions")
    mmhg.add_modality("follows", follow_hg, weight=1.2,
                     description="Follow relationships")
    
    # Summary
    summary = mmhg.generate_summary()
    print(f"\n📊 Social Network Summary:")
    print(f"   Total Users: {summary['total_unique_entities']}")
    print(f"   Modalities: {summary['num_modalities']}")
    print(f"   Avg Modalities per User: {summary['avg_modalities_per_entity']:.2f}")
    
    # User segmentation
    print("\n👥 User Segmentation by Engagement...")
    
    lurkers = mmhg.find_modal_bridges(min_modalities=1)
    casual = mmhg.find_modal_bridges(min_modalities=2)
    engaged = mmhg.find_modal_bridges(min_modalities=3)
    power_users = mmhg.find_modal_bridges(min_modalities=4)
    
    print(f"   Lurkers (1 modality): {len(lurkers) - len(casual)}")
    print(f"   Casual (2 modalities): {len(casual) - len(engaged)}")
    print(f"   Engaged (3 modalities): {len(engaged) - len(power_users)}")
    print(f"   Power Users (4+ modalities): {len(power_users)}")
    
    # Engagement analysis
    print("\n📈 Engagement Pattern Analysis...")
    
    # Who likes but doesn't share?
    like_share = mmhg.discover_inter_modal_relationships("likes", "shares")
    print(f"   Users who both like AND share: {len(like_share)}")
    
    # Who comments vs just likes?
    like_comment = mmhg.discover_inter_modal_relationships("likes", "comments")
    print(f"   Users who like AND comment: {len(like_comment)}")
    
    # Content virality
    print("\n🔥 Content Virality Analysis...")
    
    post_share_corr = mmhg.compute_modal_correlation("posts", "shares")
    post_comment_corr = mmhg.compute_modal_correlation("posts", "comments")
    share_comment_corr = mmhg.compute_modal_correlation("shares", "comments")
    
    print(f"   Post-Share correlation: {post_share_corr:.3f}")
    print(f"   Post-Comment correlation: {post_comment_corr:.3f}")
    print(f"   Share-Comment correlation: {share_comment_corr:.3f}")
    
    # Influence analysis
    print("\n⭐ Influence Analysis (Top Power Users)...")
    
    sample_users = list(power_users.keys())[:5]
    for user in sample_users:
        centrality = mmhg.compute_cross_modal_centrality(
            user,
            metric="degree",
            aggregation="weighted_average"
        )
        print(f"\n   {user}:")
        print(f"      Influence Score: {centrality['aggregated']:.2f}")
        modalities = list(power_users[user])
        print(f"      Active in: {', '.join(modalities)}")
    
    # Cross-modal patterns
    print("\n🔎 Detecting Engagement Patterns...")
    patterns = mmhg.detect_cross_modal_patterns(min_support=10)
    print(f"   Engagement patterns detected: {len(patterns)}")
    
    for i, pattern in enumerate(patterns[:3], 1):
        print(f"\n   Pattern {i}:")
        print(f"      Type: {pattern['type']}")
        print(f"      Description: {pattern['description']}")
        print(f"      Support: {pattern['support']}")
    
    # Social insights
    print("\n" + "="*70)
    print("💡 Social Media Insights")
    print("="*70)
    
    print(f"\n1. User Segmentation:")
    total_users = len(lurkers)
    print(f"   • Power users: {len(power_users)/total_users:.1%} of base")
    print(f"   • Engaged users: {len(engaged)/total_users:.1%} of base")
    print(f"   • Target power users for brand partnerships")
    
    print(f"\n2. Engagement Funnel:")
    if post_share_corr < 0.2:
        print(f"   • Low share rate ({post_share_corr:.1%})")
        print(f"   • Action: Incentivize sharing (e.g., contests)")
    
    if post_comment_corr < 0.3:
        print(f"   • Low comment rate ({post_comment_corr:.1%})")
        print(f"   • Action: Ask questions to drive discussion")
    
    print(f"\n3. Content Strategy:")
    if share_comment_corr > 0.5:
        print(f"   • High share-comment correlation")
        print(f"   • Viral content drives both shares and discussion")
    else:
        print(f"   • Different content drives shares vs comments")
        print(f"   • Diversify content strategy")
    
    print("\n" + "="*70)
    print("✅ Social Media Demo Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_social_media()
