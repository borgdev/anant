#!/usr/bin/env python3
"""
Anant Metagraph Phase 1 - Comprehensive Demonstration
=====================================================

This script demonstrates the complete Phase 1 functionality of the Anant Metagraph
enterprise knowledge management system, showcasing all core layers and their integration.

Features Demonstrated:
1. Core metagraph creation and entity management
2. Hierarchical organization with multi-level navigation
3. Rich metadata storage with schema validation
4. Semantic relationships and business glossary
5. Temporal event tracking and pattern analysis
6. Governance policies and access control
7. Cross-layer analytics and comprehensive search

Architecture:
- Polars+Parquet backend for high performance
- ZSTD compression for enterprise storage efficiency  
- Instance-specific state management (no global pollution)
- Type-safe operations with comprehensive error handling
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Add the anant package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the metagraph components
from anant.metagraph import (
    create_enterprise_metagraph, 
    create_basic_metagraph,
    get_version_info
)

def demonstrate_phase1_functionality():
    """Comprehensive demonstration of Phase 1 capabilities."""
    
    print("=" * 70)
    print("Anant Metagraph Phase 1 - Enterprise Knowledge Management")
    print("=" * 70)
    
    # Display version information
    version_info = get_version_info()
    print(f"Version: {version_info['version']}")
    print(f"Phase: {version_info['phase']}")
    print(f"Backend: {version_info['backend']}")
    print(f"Storage: {version_info['storage_format']}")
    print()
    
    # Create enterprise metagraph
    print("🚀 Creating Enterprise Metagraph...")
    storage_path = "./demo_enterprise_metagraph"
    mg = create_enterprise_metagraph(
        storage_path=storage_path,
        embedding_dimension=384,  # Smaller for demo
        compression="zstd",
        retention_days=365
    )
    print(f"✅ Enterprise metagraph created at: {storage_path}")
    print()
    
    # Demonstrate entity creation and hierarchical organization
    print("📊 Demonstrating Hierarchical Organization...")
    
    # Create top-level business domains
    mg.create_entity(
        entity_id="business_domain_sales",
        entity_type="domain",
        properties={
            "name": "Sales Domain",
            "description": "All sales-related data and processes",
            "owner": "sales_team",
            "criticality": "high"
        },
        level=0,
        classification="internal",
        created_by="admin"
    )
    
    mg.create_entity(
        entity_id="business_domain_marketing", 
        entity_type="domain",
        properties={
            "name": "Marketing Domain",
            "description": "Marketing campaigns and customer analytics",
            "owner": "marketing_team",
            "criticality": "medium"
        },
        level=0,
        classification="internal",
        created_by="admin"
    )
    
    # Create systems under domains
    mg.create_entity(
        entity_id="system_crm",
        entity_type="system",
        properties={
            "name": "CRM System",
            "technology": "Salesforce", 
            "version": "v2.1",
            "environment": "production"
        },
        level=1,
        parent_id="business_domain_sales",
        classification="confidential",
        created_by="admin"
    )
    
    mg.create_entity(
        entity_id="system_marketing_automation",
        entity_type="system", 
        properties={
            "name": "Marketing Automation",
            "technology": "HubSpot",
            "version": "v3.5",
            "environment": "production"
        },
        level=1,
        parent_id="business_domain_marketing",
        classification="internal",
        created_by="admin"
    )
    
    # Create datasets under systems
    mg.create_entity(
        entity_id="dataset_customer_profiles",
        entity_type="dataset",
        properties={
            "name": "Customer Profiles",
            "table_name": "customers",
            "row_count": 50000,
            "schema_version": "v1.2",
            "last_updated": datetime.now().isoformat(),
            "contains_pii": True
        },
        level=2,
        parent_id="system_crm",
        classification="restricted",
        created_by="data_engineer"
    )
    
    mg.create_entity(
        entity_id="dataset_campaign_metrics",
        entity_type="dataset",
        properties={
            "name": "Campaign Performance Metrics",
            "table_name": "campaign_stats", 
            "row_count": 25000,
            "schema_version": "v2.0",
            "last_updated": datetime.now().isoformat(),
            "contains_pii": False
        },
        level=2,
        parent_id="system_marketing_automation",
        classification="internal",
        created_by="marketing_analyst"
    )
    
    print("✅ Created hierarchical entity structure:")
    print("   📁 Business Domains (Level 0)")
    print("     📂 Systems (Level 1)")
    print("       📄 Datasets (Level 2)")
    print()
    
    # Demonstrate semantic relationships
    print("🔗 Demonstrating Semantic Relationships...")
    
    # Add semantic relationship between customer data and campaigns
    relationship_id = mg.add_relationship(
        source_id="dataset_customer_profiles",
        target_id="dataset_campaign_metrics", 
        relationship_type="semantic",
        strength=0.8,
        metadata={
            "relationship_description": "Customer profiles used for campaign targeting",
            "data_flow": "customers -> segmentation -> campaign_targeting",
            "business_impact": "high"
        },
        created_by="data_analyst"
    )
    print(f"✅ Created semantic relationship: {relationship_id}")
    
    # Add business terms to glossary
    print("📖 Building Business Glossary...")
    
    mg.semantic.add_business_term(
        term="Customer Lifetime Value",
        definition="The predicted revenue that a customer will generate during their lifetime",
        context="Used in marketing and sales analytics",
        domain="sales",
        synonyms=["CLV", "CLTV", "LTV"],
        related_terms=["Customer Acquisition Cost", "Churn Rate"],
        business_rules={
            "calculation": "sum(revenue) - sum(costs) over customer_lifetime",
            "frequency": "monthly",
            "owner": "finance_team"
        },
        usage_examples=[
            "High CLV customers should receive premium support",
            "CLV guides marketing spend allocation"
        ],
        data_sources=["dataset_customer_profiles", "dataset_campaign_metrics"],
        created_by="business_analyst"
    )
    
    mg.semantic.add_business_term(
        term="Conversion Rate",
        definition="Percentage of prospects who become paying customers",
        context="Marketing campaign effectiveness measurement",
        domain="marketing",
        synonyms=["CVR", "Conversion Percentage"],
        related_terms=["Click-through Rate", "Customer Acquisition Cost"],
        business_rules={
            "calculation": "(conversions / total_visitors) * 100",
            "benchmark": "> 2.5% for industry",
            "reporting": "weekly"
        },
        created_by="marketing_analyst"
    )
    
    print("✅ Added business terms to glossary")
    print()
    
    # Demonstrate temporal tracking
    print("⏰ Demonstrating Temporal Tracking...")
    
    # Simulate some data access and update events
    for i in range(5):
        mg.temporal.record_event(
            entity_id="dataset_customer_profiles",
            operation="access",
            details={
                "access_type": "read",
                "query_type": "analytics",
                "rows_accessed": np.random.randint(1000, 5000)
            },
            user_id="analyst_user",
            session_id=f"session_{i}",
            source_system="analytics_platform"
        )
    
    # Update entity and track changes
    mg.update_entity(
        entity_id="dataset_customer_profiles",
        properties={
            "row_count": 51000,  # Updated count
            "last_updated": datetime.now().isoformat(),
            "data_quality_score": 0.95
        },
        updated_by="data_engineer"
    )
    
    print("✅ Recorded temporal events and entity updates")
    
    # Analyze temporal patterns
    print("🔍 Analyzing Temporal Patterns...")
    patterns = mg.analyze_temporal_patterns(
        entity_ids=["dataset_customer_profiles", "dataset_campaign_metrics"],
        days_back=7
    )
    print(f"✅ Discovered {patterns['patterns_discovered']} temporal patterns")
    print()
    
    # Demonstrate governance and policies
    print("🔒 Demonstrating Governance Policies...")
    
    # Create access control policy
    access_policy_id = mg.create_policy(
        name="PII Data Access Control",
        policy_type="access", 
        rules={
            "classification_requirements": "confidential",
            "allowed_roles": ["data_engineer", "privacy_officer"],
            "time_restrictions": {
                "business_hours_only": True
            },
            "audit_required": True
        },
        description="Restricts access to personally identifiable information",
        created_by="security_admin"
    )
    
    # Create data quality policy
    quality_policy_id = mg.create_policy(
        name="Data Quality Standards",
        policy_type="data_quality",
        rules={
            "checks": {
                "completeness": {"threshold": 0.95},
                "validity": {"threshold": 0.98},
                "consistency": {"threshold": 0.92}
            },
            "monitoring": "continuous"
        },
        description="Ensures data meets quality standards",
        created_by="data_quality_admin"
    )
    
    print(f"✅ Created access policy: {access_policy_id}")
    print(f"✅ Created quality policy: {quality_policy_id}")
    
    # Check access permissions
    access_check = mg.check_entity_access(
        user_id="analyst_user",
        entity_id="dataset_customer_profiles", 
        access_level="read"
    )
    print(f"🔐 Access check result: {access_check['decision']}")
    print()
    
    # Demonstrate comprehensive search
    print("🔍 Demonstrating Comprehensive Search...")
    
    # Search across all entities
    search_results = mg.search_entities(
        query="customer",
        entity_types=["dataset", "system"],
        levels=[1, 2],
        limit=10
    )
    
    print(f"✅ Found {len(search_results)} entities matching 'customer'")
    for result in search_results:
        print(f"   📄 {result['entity_id']}: {result['metadata']['properties'].get('name', 'Unknown')}")
    print()
    
    # Demonstrate entity relationships
    print("🌐 Demonstrating Entity Relationships...")
    
    related_entities = mg.get_related_entities(
        entity_id="dataset_customer_profiles",
        max_depth=2,
        min_strength=0.1
    )
    
    print("✅ Related entities analysis:")
    print(f"   🔗 Semantic relationships: {len(related_entities['semantic_relationships'])}")
    print(f"   📁 Hierarchical parent: {related_entities['hierarchical_relationships']['parent']}")
    print(f"   👥 Siblings: {len(related_entities['hierarchical_relationships']['siblings'])}")
    print(f"   🎯 Similar entities: {len(related_entities['similar_entities'])}")
    print()
    
    # Get comprehensive statistics
    print("📈 Comprehensive System Statistics...")
    stats = mg.get_comprehensive_stats()
    
    print("✅ System Statistics:")
    print(f"   📊 Total entities: {stats['cross_layer']['total_entities']}")
    print(f"   🗄️  Metadata records: {stats['metadata']['total_records']}")
    print(f"   🔗 Relationships: {stats['semantic']['relationships_count']}")
    print(f"   ⏰ Temporal events: {stats['temporal']['events_count']}")
    print(f"   🔒 Active policies: {stats['governance']['active_policies_count']}")
    print(f"   📚 Business terms: {stats['semantic']['business_terms_count']}")
    print()
    
    # Demonstrate business glossary search
    print("📖 Demonstrating Business Glossary Search...")
    
    glossary_results = mg.semantic.search_business_terms(
        query="customer",
        include_synonyms=True
    )
    
    print(f"✅ Found {len(glossary_results)} business terms related to 'customer'")
    for term in glossary_results:
        print(f"   📝 {term['term']}: {term['definition'][:100]}...")
    print()
    
    # Save all data
    print("💾 Saving All Data...")
    mg.save_all()
    print("✅ All layers saved to persistent storage")
    print()
    
    # Display final summary
    print("🎉 Phase 1 Demonstration Complete!")
    print("=" * 50)
    print("Successfully demonstrated:")
    print("✅ Hierarchical entity organization")
    print("✅ Rich metadata management") 
    print("✅ Semantic relationships and glossary")
    print("✅ Temporal event tracking and patterns")
    print("✅ Governance policies and access control")
    print("✅ Cross-layer search and analytics")
    print("✅ Enterprise-grade data management")
    print()
    print(f"📂 All data stored in: {storage_path}")
    print("🚀 Ready for Phase 2: LLM Integration and Advanced Analytics")
    print("=" * 70)
    
    return mg, stats

def demonstrate_dual_capability():
    """Demonstrate the dual capability architecture."""
    print("\n" + "=" * 70)
    print("Dual Capability Architecture Demonstration")
    print("=" * 70)
    
    # Traditional usage (basic metagraph)
    print("📊 Basic Metagraph (Development/Testing)...")
    basic_mg = create_basic_metagraph("./demo_basic_metagraph") 
    basic_mg.create_entity("test_entity", "test", {"value": "hello world"})
    basic_stats = basic_mg.get_comprehensive_stats()
    print(f"✅ Basic metagraph entities: {basic_stats['cross_layer']['total_entities']}")
    
    # Enterprise usage (full metagraph) 
    print("🏢 Enterprise Metagraph (Production)...")
    enterprise_mg = create_enterprise_metagraph("./demo_enterprise_full")
    enterprise_mg.create_entity("prod_entity", "production", {"environment": "live"})
    enterprise_stats = enterprise_mg.get_comprehensive_stats()
    print(f"✅ Enterprise metagraph entities: {enterprise_stats['cross_layer']['total_entities']}")
    
    print("✅ Dual capability architecture working correctly!")
    print("   - Users can choose basic OR enterprise functionality")
    print("   - No interference between instances")
    print("   - Polars+Parquet backend for metadata when needed")

if __name__ == "__main__":
    try:
        # Run Phase 1 demonstration
        metagraph, statistics = demonstrate_phase1_functionality()
        
        # Run dual capability demonstration
        demonstrate_dual_capability()
        
        print("\n🎯 All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)