"""
Research Network Multi-Modal Analysis Demo
==========================================

Demonstrates multi-modal relationship analysis for academic research networks.

Modalities:
- Citations: Paper citation relationships
- Collaborations: Author collaboration relationships
- Funding: Researcher-grant relationships
- Publications: Author-publication relationships

Cross-Modal Insights:
- Collaboration patterns vs citation patterns
- Funding impact on collaboration
- Influential researchers across modalities
- Cross-institutional research clusters
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


def generate_research_data(
    num_researchers: int = 200,
    num_papers: int = 500,
    num_grants: int = 50
):
    """Generate synthetic research network data"""
    
    print("📊 Generating Research Network Data...")
    print(f"   Researchers: {num_researchers}")
    print(f"   Papers: {num_papers}")
    print(f"   Grants: {num_grants}")
    
    researchers = [f"researcher_{i:04d}" for i in range(num_researchers)]
    papers = [f"paper_{i:04d}" for i in range(num_papers)]
    grants = [f"grant_{i:03d}" for i in range(num_grants)]
    
    # Generate citations
    citation_records = []
    for i in range(num_papers * 3):  # ~3 citations per paper
        citation_id = f"citation_{i:05d}"
        citing_paper = random.choice(papers)
        cited_paper = random.choice(papers)
        
        if citing_paper != cited_paper:
            citation_records.extend([
                {'edges': citation_id, 'nodes': citing_paper, 'weight': 1.0, 'role': 'citing'},
                {'edges': citation_id, 'nodes': cited_paper, 'weight': 1.0, 'role': 'cited'}
            ])
    
    # Generate collaborations
    collaboration_records = []
    for i in range(num_researchers * 4):  # ~4 collaborations per researcher
        collab_id = f"collaboration_{i:05d}"
        researcher1 = random.choice(researchers)
        researcher2 = random.choice(researchers)
        
        if researcher1 != researcher2:
            collaboration_records.extend([
                {'edges': collab_id, 'nodes': researcher1, 'weight': 1.0, 'role': 'collaborator'},
                {'edges': collab_id, 'nodes': researcher2, 'weight': 1.0, 'role': 'collaborator'}
            ])
    
    # Generate funding
    funding_records = []
    for i in range(num_researchers):  # ~1 grant per researcher
        funding_id = f"funding_{i:05d}"
        researcher = random.choice(researchers)
        grant = random.choice(grants)
        
        funding_records.extend([
            {'edges': funding_id, 'nodes': researcher, 'weight': random.uniform(10000, 500000), 'role': 'PI'},
            {'edges': funding_id, 'nodes': grant, 'weight': 1.0, 'role': 'grant'}
        ])
    
    # Generate publications (authorship)
    publication_records = []
    for i in range(num_papers):
        paper = papers[i]
        num_authors = random.randint(1, 5)
        selected_authors = random.sample(researchers, num_authors)
        
        for author in selected_authors:
            publication_records.extend([
                {'edges': paper, 'nodes': author, 'weight': 1.0, 'role': 'author'},
                {'edges': paper, 'nodes': paper, 'weight': 1.0, 'role': 'paper'}
            ])
    
    return {
        'citations': pl.DataFrame(citation_records),
        'collaborations': pl.DataFrame(collaboration_records),
        'funding': pl.DataFrame(funding_records),
        'publications': pl.DataFrame(publication_records)
    }


def demo_research_network():
    """Demo: Research network multi-modal analysis"""
    
    print("\n" + "="*70)
    print("🎓 Research Network Multi-Modal Analysis Demo")
    print("="*70)
    
    # Generate data
    data = generate_research_data()
    
    # Create hypergraphs
    print("\n📦 Creating Research Network Hypergraphs...")
    citation_hg = MockHypergraph(data['citations'])
    collab_hg = MockHypergraph(data['collaborations'])
    funding_hg = MockHypergraph(data['funding'])
    pub_hg = MockHypergraph(data['publications'])
    
    # Build multi-modal hypergraph
    print("\n🔗 Constructing Multi-Modal Research Network...")
    mmhg = MultiModalHypergraph(name="research_network")
    
    mmhg.add_modality("citations", citation_hg, weight=2.0,
                     description="Paper citation network")
    mmhg.add_modality("collaborations", collab_hg, weight=2.5,
                     description="Researcher collaboration network")
    mmhg.add_modality("funding", funding_hg, weight=1.5,
                     description="Research funding network")
    mmhg.add_modality("publications", pub_hg, weight=2.0,
                     description="Publication authorship network")
    
    # Summary
    summary = mmhg.generate_summary()
    print(f"\n📊 Research Network Summary:")
    print(f"   Total Entities: {summary['total_unique_entities']}")
    print(f"   Modalities: {summary['num_modalities']}")
    print(f"   Avg Modalities per Entity: {summary['avg_modalities_per_entity']:.2f}")
    
    # Find influential researchers (in multiple modalities)
    print("\n🔍 Finding Influential Researchers...")
    influential = mmhg.find_modal_bridges(min_modalities=3)
    print(f"   Researchers in 3+ modalities: {len(influential)}")
    
    # Collaboration-citation gap analysis
    print("\n🔗 Collaboration-Citation Analysis...")
    collab_cite = mmhg.discover_inter_modal_relationships(
        "collaborations", "citations"
    )
    print(f"   Researchers who collaborate and cite: {len(collab_cite)}")
    
    # Funding impact
    print("\n💰 Funding Impact Analysis...")
    funding_collab_corr = mmhg.compute_modal_correlation("funding", "collaborations")
    funding_pub_corr = mmhg.compute_modal_correlation("funding", "publications")
    
    print(f"   Funding-Collaboration correlation: {funding_collab_corr:.3f}")
    print(f"   Funding-Publication correlation: {funding_pub_corr:.3f}")
    
    # Cross-modal patterns
    print("\n🔎 Detecting Research Patterns...")
    patterns = mmhg.detect_cross_modal_patterns(min_support=5)
    print(f"   Research patterns detected: {len(patterns)}")
    
    for i, pattern in enumerate(patterns[:3], 1):
        print(f"\n   Pattern {i}:")
        print(f"      Type: {pattern['type']}")
        print(f"      Description: {pattern['description']}")
        print(f"      Support: {pattern['support']}")
    
    # Compute researcher centrality
    print("\n⭐ Top Researchers by Cross-Modal Centrality...")
    sample_researchers = list(influential.keys())[:5]
    
    for researcher in sample_researchers:
        centrality = mmhg.compute_cross_modal_centrality(
            researcher,
            metric="degree",
            aggregation="weighted_average"
        )
        print(f"\n   {researcher}:")
        print(f"      Overall Score: {centrality['aggregated']:.2f}")
        for mod, score in centrality['per_modality'].items():
            print(f"         {mod}: {score:.1f}")
    
    # Research insights
    print("\n" + "="*70)
    print("💡 Research Insights")
    print("="*70)
    
    print(f"\n1. Researcher Engagement:")
    print(f"   • {len(influential)} researchers active across multiple areas")
    print(f"   • These are potential research leaders")
    
    print(f"\n2. Collaboration Patterns:")
    collab_cite_corr = mmhg.compute_modal_correlation("collaborations", "citations")
    if collab_cite_corr < 0.3:
        print(f"   • Low collaboration-citation overlap ({collab_cite_corr:.1%})")
        print(f"   • Researchers collaborate but may not cite each other")
    else:
        print(f"   • Good collaboration-citation alignment ({collab_cite_corr:.1%})")
    
    print(f"\n3. Funding Impact:")
    if funding_collab_corr > 0.4:
        print(f"   • Funding drives collaboration ({funding_collab_corr:.1%})")
    if funding_pub_corr > 0.4:
        print(f"   • Funding drives publications ({funding_pub_corr:.1%})")
    
    print("\n" + "="*70)
    print("✅ Research Network Demo Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_research_network()
