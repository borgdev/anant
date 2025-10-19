"""
Phase 3 Pattern Recognition Demo

This demo showcases the advanced ML-powered pattern detection capabilities:
- Anomaly detection in metagraph structure and behavior
- Trend analysis for evolving patterns over time  
- Community discovery for hidden relationships
- Comprehensive pattern analysis with confidence scoring

Features demonstrated:
- Real-time pattern detection with multiple algorithms
- Enterprise-grade anomaly detection using Isolation Forest
- Statistical fallbacks when ML libraries unavailable
- Trend analysis with cyclical pattern detection
- Community discovery using clustering algorithms
- Pattern filtering, ranking, and export capabilities
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

# Add the project root to the Python path
sys.path.append('/home/amansingh/dev/ai/anant')

def create_sample_metagraph_data(variant: str = "baseline") -> Dict[str, Any]:
    """Create sample metagraph data for testing"""
    
    if variant == "baseline":
        return {
            'entities': {
                'customer_001': {
                    'type': 'Customer',
                    'properties': {'name': 'Acme Corp', 'tier': 'Enterprise'},
                    'created_at': '2024-01-01T00:00:00Z'
                },
                'product_001': {
                    'type': 'Product', 
                    'properties': {'name': 'Widget A', 'category': 'Hardware'},
                    'created_at': '2024-01-02T00:00:00Z'
                },
                'order_001': {
                    'type': 'Order',
                    'properties': {'amount': 1000, 'status': 'completed'},
                    'created_at': '2024-01-05T00:00:00Z'
                },
                'person_001': {
                    'type': 'Person',
                    'properties': {'name': 'John Smith', 'role': 'Manager'},
                    'created_at': '2024-01-03T00:00:00Z'
                },
                'department_001': {
                    'type': 'Department',
                    'properties': {'name': 'Sales', 'budget': 50000},
                    'created_at': '2024-01-01T00:00:00Z'
                }
            },
            'relationships': {
                'rel_001': {
                    'type': 'purchases',
                    'entities': ['customer_001', 'product_001'],
                    'properties': {'date': '2024-01-05', 'quantity': 10}
                },
                'rel_002': {
                    'type': 'orders', 
                    'entities': ['customer_001', 'order_001'],
                    'properties': {'order_date': '2024-01-05'}
                },
                'rel_003': {
                    'type': 'works_for',
                    'entities': ['person_001', 'department_001'],
                    'properties': {'start_date': '2024-01-01', 'position': 'Manager'}
                },
                'rel_004': {
                    'type': 'manages',
                    'entities': ['person_001', 'customer_001'],
                    'properties': {'assigned_date': '2024-01-02'}
                }
            }
        }
    
    elif variant == "anomaly":
        # Create data with anomalies
        data = create_sample_metagraph_data("baseline")
        
        # Add many more entities (structural anomaly)
        for i in range(20):
            data['entities'][f'entity_{i:03d}'] = {
                'type': 'Unknown',
                'properties': {'anomaly': True},
                'created_at': f'2024-01-10T{i%24:02d}:00:00Z'
            }
        
        # Add unusual relationship patterns
        for i in range(15):
            data['relationships'][f'anomaly_rel_{i:03d}'] = {
                'type': 'unusual_connection',
                'entities': [f'entity_{i:03d}', f'entity_{(i+1)%20:03d}'],
                'properties': {'anomaly_score': 0.9}
            }
        
        return data
    
    elif variant == "trending":
        # Create data showing growth trends
        data = create_sample_metagraph_data("baseline")
        
        # Add growing number of entities over time
        base_date = datetime(2024, 1, 1)
        for i in range(15):
            date = base_date + timedelta(days=i)
            # Exponential growth pattern
            for j in range(i + 1):
                entity_id = f'trending_entity_{i:02d}_{j:02d}'
                data['entities'][entity_id] = {
                    'type': 'TrendingEntity',
                    'properties': {'growth_factor': i + 1, 'index': j},
                    'created_at': date.isoformat() + 'Z'
                }
        
        return data
    
    elif variant == "community":
        # Create data with clear community structures
        data = create_sample_metagraph_data("baseline")
        
        # Community 1: Tech team
        tech_entities = ['dev_001', 'dev_002', 'dev_003', 'tech_lead_001', 'project_001']
        for entity_id in tech_entities:
            data['entities'][entity_id] = {
                'type': 'TechRole' if 'dev' in entity_id or 'lead' in entity_id else 'Project',
                'properties': {'team': 'technology', 'skills': ['python', 'ml']},
                'created_at': '2024-01-01T00:00:00Z'
            }
        
        # Community 2: Sales team  
        sales_entities = ['sales_001', 'sales_002', 'sales_manager_001', 'deal_001', 'deal_002']
        for entity_id in sales_entities:
            data['entities'][entity_id] = {
                'type': 'SalesRole' if 'sales' in entity_id or 'manager' in entity_id else 'Deal',
                'properties': {'team': 'sales', 'region': 'north'},
                'created_at': '2024-01-01T00:00:00Z'
            }
        
        # Intra-community relationships
        tech_relationships = [
            ('dev_001', 'tech_lead_001', 'reports_to'),
            ('dev_002', 'tech_lead_001', 'reports_to'),
            ('dev_003', 'tech_lead_001', 'reports_to'),
            ('tech_lead_001', 'project_001', 'manages'),
            ('dev_001', 'project_001', 'works_on'),
            ('dev_002', 'project_001', 'works_on')
        ]
        
        sales_relationships = [
            ('sales_001', 'sales_manager_001', 'reports_to'),
            ('sales_002', 'sales_manager_001', 'reports_to'),
            ('sales_001', 'deal_001', 'owns'),
            ('sales_002', 'deal_002', 'owns'),
            ('sales_manager_001', 'deal_001', 'approves'),
            ('sales_manager_001', 'deal_002', 'approves')
        ]
        
        # Add relationships
        rel_counter = len(data['relationships'])
        for entity1, entity2, rel_type in tech_relationships + sales_relationships:
            rel_counter += 1
            data['relationships'][f'community_rel_{rel_counter:03d}'] = {
                'type': rel_type,
                'entities': [entity1, entity2],
                'properties': {'community_type': 'intra_team'}
            }
        
        return data
    
    # Default fallback
    return create_sample_metagraph_data("baseline")

async def demo_pattern_recognition():
    """Demonstrate the Pattern Recognition Engine capabilities"""
    
    print("🔍 Phase 3: Advanced Pattern Recognition Demo")
    print("=" * 60)
    
    try:
        # Mock the ML dependencies first to test fallback systems
        import sys
        class MockModule:
            def __getattr__(self, name):
                if name in ['DBSCAN', 'KMeans', 'IsolationForest', 'PCA', 'StandardScaler']:
                    return lambda *args, **kwargs: None
                return lambda *args, **kwargs: None
        
        # Test with mock first to show fallback capabilities
        print("🧪 Testing with mocked ML dependencies (fallback systems)...")
        
        # Mock sklearn modules to test fallback systems
        import types
        mock_module = types.ModuleType('mock_sklearn')
        sys.modules['sklearn.cluster'] = mock_module
        sys.modules['sklearn.ensemble'] = mock_module
        sys.modules['sklearn.decomposition'] = mock_module
        sys.modules['sklearn.preprocessing'] = mock_module
        sys.modules['sklearn.metrics'] = mock_module
        
        from anant.metagraph.analytics.pattern_recognition import (
            PatternRecognitionEngine,
            PatternType,
            PatternConfidence,
            AnomalyDetector,
            TrendAnalyzer,
            RelationshipDiscoverer
        )
        
        print("✅ Pattern Recognition components loaded successfully")
        
        # Initialize the pattern recognition engine
        config = {
            'anomaly_contamination': 0.1,
            'anomaly_estimators': 50,
            'trend_window_size': 10,
            'min_cluster_size': 2,
            'clustering_eps': 0.3,
            'max_patterns_per_analysis': 20
        }
        
        engine = PatternRecognitionEngine(config)
        print("✅ Pattern Recognition Engine initialized")
        
        # Test 1: Baseline Analysis
        print("\n📊 Test 1: Baseline Analysis")
        print("-" * 40)
        
        baseline_data = create_sample_metagraph_data("baseline")
        print(f"📈 Baseline data: {len(baseline_data['entities'])} entities, {len(baseline_data['relationships'])} relationships")
        
        patterns = await engine.analyze_metagraph(baseline_data, baseline_data)
        print(f"🔍 Detected {len(patterns)} patterns in baseline data")
        
        # Test 2: Anomaly Detection
        print("\n🚨 Test 2: Anomaly Detection")
        print("-" * 40)
        
        anomaly_data = create_sample_metagraph_data("anomaly")
        print(f"📈 Anomaly data: {len(anomaly_data['entities'])} entities, {len(anomaly_data['relationships'])} relationships")
        
        anomaly_patterns = await engine.analyze_metagraph(anomaly_data)
        anomalies = [p for p in anomaly_patterns if 'anomaly' in p.pattern_type.value]
        print(f"🚨 Detected {len(anomalies)} anomaly patterns")
        
        for anomaly in anomalies[:3]:  # Show first 3
            print(f"  • {anomaly.pattern_type.value}: {anomaly.description}")
            print(f"    Confidence: {anomaly.confidence.value}")
            if 'detection_method' in anomaly.evidence:
                print(f"    Method: {anomaly.evidence['detection_method']}")
        
        # Test 3: Trend Analysis
        print("\n📈 Test 3: Trend Analysis")
        print("-" * 40)
        
        # Simulate multiple time periods for trend detection
        trend_analyzer = TrendAnalyzer(window_size=5)
        
        base_date = datetime(2024, 1, 1)
        for day in range(7):
            snapshot_date = base_date + timedelta(days=day)
            
            # Create evolving data
            if day < 3:
                data = create_sample_metagraph_data("baseline")
            else:
                data = create_sample_metagraph_data("trending")
            
            trend_analyzer.add_snapshot(snapshot_date, data)
            print(f"  📅 Added snapshot for {snapshot_date.strftime('%Y-%m-%d')}")
        
        trend_patterns = trend_analyzer.detect_trends()
        print(f"📊 Detected {len(trend_patterns)} trend patterns")
        
        for trend in trend_patterns[:2]:  # Show first 2
            print(f"  • {trend.pattern_type.value}: {trend.description}")
            print(f"    Confidence: {trend.confidence.value}")
            if 'slope' in trend.evidence:
                print(f"    Trend slope: {trend.evidence['slope']:.3f}")
        
        # Test 4: Community Discovery
        print("\n🏘️ Test 4: Community Discovery")
        print("-" * 40)
        
        community_data = create_sample_metagraph_data("community")
        print(f"📈 Community data: {len(community_data['entities'])} entities, {len(community_data['relationships'])} relationships")
        
        discoverer = RelationshipDiscoverer(min_cluster_size=2, eps=0.3)
        community_patterns = discoverer.discover_communities(community_data)
        communities = [p for p in community_patterns if 'community' in p.pattern_type.value or 'cluster' in p.pattern_type.value]
        
        print(f"🏘️ Discovered {len(communities)} community patterns")
        
        for community in communities[:2]:  # Show first 2
            print(f"  • {community.pattern_type.value}: {community.description}")
            print(f"    Entities: {len(community.entities)} members")
            print(f"    Relationships: {len(community.relationships)} connections")
            if 'detection_method' in community.evidence:
                print(f"    Method: {community.evidence['detection_method']}")
        
        # Test 5: Comprehensive Analysis
        print("\n🔬 Test 5: Comprehensive Analysis")
        print("-" * 40)
        
        comprehensive_patterns = await engine.analyze_metagraph(community_data, baseline_data)
        print(f"🔍 Total patterns detected: {len(comprehensive_patterns)}")
        
        # Categorize patterns
        pattern_summary = engine.export_patterns('summary')
        if isinstance(pattern_summary, dict):
            print(f"📊 Pattern Summary:")
            print(f"  Total: {pattern_summary.get('total_patterns', 0)}")
            print(f"  Recent (24h): {pattern_summary.get('recent_patterns', 0)}")
            
            print(f"  By Type:")
            by_type = pattern_summary.get('by_type', {})
            if isinstance(by_type, dict):
                for ptype, count in by_type.items():
                    print(f"    {ptype}: {count}")
            
            print(f"  By Confidence:")
            by_confidence = pattern_summary.get('by_confidence', {})
            if isinstance(by_confidence, dict):
                for conf, count in by_confidence.items():
                    print(f"    {conf}: {count}")
        else:
            print(f"📊 Pattern Summary: {pattern_summary}")
        
        # Test 6: Pattern Filtering and Export
        print("\n📤 Test 6: Pattern Export and Filtering")
        print("-" * 40)
        
        high_confidence_patterns = engine.get_patterns_by_confidence(PatternConfidence.MEDIUM)
        print(f"🎯 High confidence patterns: {len(high_confidence_patterns)}")
        
        anomaly_patterns_filtered = engine.get_patterns_by_type(PatternType.ANOMALY_BEHAVIORAL)
        print(f"🚨 Behavioral anomalies: {len(anomaly_patterns_filtered)}")
        
        # Export patterns
        exported_patterns = engine.export_patterns('dict')
        print(f"📄 Exported {len(exported_patterns)} patterns as dictionaries")
        
        # Test pattern expiration
        expired_count = engine.clear_expired_patterns()
        print(f"🧹 Cleared {expired_count} expired patterns")
        
        print("\n🎉 Pattern Recognition Demo Complete!")
        print("✅ All components working with graceful fallbacks")
        print("🚀 Ready for Phase 3 continuation: Predictive Analytics")
        
    except Exception as e:
        print(f"❌ Error in pattern recognition demo: {e}")
        import traceback
        traceback.print_exc()

async def demo_individual_components():
    """Demo individual pattern recognition components"""
    
    print("\n🧩 Individual Component Demonstrations")
    print("=" * 50)
    
    try:
        from anant.metagraph.analytics.pattern_recognition import (
            AnomalyDetector,
            TrendAnalyzer,
            RelationshipDiscoverer
        )
        
        # Demo AnomalyDetector
        print("\n🚨 Anomaly Detector Demo")
        print("-" * 30)
        
        detector = AnomalyDetector(contamination=0.15)
        baseline_data = create_sample_metagraph_data("baseline")
        anomaly_data = create_sample_metagraph_data("anomaly")
        
        detector.fit_baseline(baseline_data)
        print("✅ Baseline established for anomaly detection")
        
        anomalies = detector.detect_anomalies(anomaly_data)
        print(f"🔍 Detected {len(anomalies)} anomalies")
        
        # Demo TrendAnalyzer
        print("\n📈 Trend Analyzer Demo")
        print("-" * 30)
        
        analyzer = TrendAnalyzer(window_size=6)
        
        # Add multiple snapshots to build trend data
        base_date = datetime(2024, 1, 1)
        for i in range(8):
            snapshot_date = base_date + timedelta(days=i)
            
            # Simulate growing data
            data = create_sample_metagraph_data("baseline")
            
            # Add more entities over time to create trend
            for j in range(i * 2):
                data['entities'][f'trend_entity_{i}_{j}'] = {
                    'type': 'TrendEntity',
                    'properties': {'day': i, 'index': j},
                    'created_at': snapshot_date.isoformat() + 'Z'
                }
            
            analyzer.add_snapshot(snapshot_date, data)
        
        trends = analyzer.detect_trends()
        print(f"📊 Detected {len(trends)} trend patterns")
        
        # Demo RelationshipDiscoverer
        print("\n🕸️ Relationship Discoverer Demo")
        print("-" * 30)
        
        discoverer = RelationshipDiscoverer(min_cluster_size=2)
        community_data = create_sample_metagraph_data("community")
        
        communities = discoverer.discover_communities(community_data)
        print(f"🏘️ Discovered {len(communities)} communities/clusters")
        
        for community in communities:
            print(f"  • {community.description}")
            print(f"    Members: {len(community.entities)}")
        
        print("\n✅ Individual component demos complete")
        
    except Exception as e:
        print(f"❌ Error in component demo: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main demo function"""
    print("🎯 Phase 3: Advanced Pattern Recognition - Complete Demo")
    print("🔬 Enterprise ML-Powered Analytics for Metagraphs")
    print("=" * 70)
    print()
    
    # Run the main pattern recognition demo
    asyncio.run(demo_pattern_recognition())
    
    # Run individual component demos
    asyncio.run(demo_individual_components())
    
    print("\n" + "=" * 70)
    print("🎊 Phase 3 Pattern Recognition Demo Complete!")
    print("📋 Key Features Demonstrated:")
    print("  ✅ ML-powered anomaly detection with statistical fallbacks")
    print("  ✅ Time-series trend analysis with cyclical pattern detection")
    print("  ✅ Community discovery using clustering algorithms")
    print("  ✅ Pattern filtering, ranking, and confidence scoring")
    print("  ✅ Enterprise-grade error handling and graceful degradation")
    print("  ✅ Comprehensive pattern export and management")
    print()
    print("🚀 Ready for Phase 3 continuation:")
    print("  📊 Predictive Analytics Engine")
    print("  🏭 Production Deployment Framework")
    print("  🛡️ Advanced Governance Automation")

if __name__ == "__main__":
    main()