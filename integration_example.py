#!/usr/bin/env python3
"""
Anant Integration Example
========================

Demonstrates how to use the anant_integration ecosystem
to build enterprise-grade knowledge graph applications.

This example shows:
1. Configuration management
2. Integration registration
3. Lifecycle management
4. Database integration
5. API endpoints
6. Monitoring setup
"""

import asyncio
import logging
from pathlib import Path

# Import core Anant library
import sys
sys.path.insert(0, str(Path(__file__).parent))

from anant.kg.core import KnowledgeGraph
from anant.classes.hypergraph import Hypergraph

# Import integration framework
from anant_integration import IntegrationManager, load_config
from anant_integration.database.postgresql import PostgreSQLIntegration
from anant_integration.core.base import IntegrationConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("anant.example")


async def main():
    """Main example demonstrating Anant integration ecosystem"""
    
    print("🚀 ANANT INTEGRATION ECOSYSTEM EXAMPLE")
    print("=" * 60)
    
    # 1. Load configuration
    print("\n📁 Loading Configuration...")
    try:
        config = load_config('anant_integration/anant_integration.yaml', 'development')
        print(f"✅ Configuration loaded for environment: {config.get('global.environment')}")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return
    
    # 2. Initialize integration manager
    print("\n🔧 Initializing Integration Manager...")
    manager = IntegrationManager()
    
    # 3. Register integrations
    print("\n📦 Registering Integrations...")
    
    # Register PostgreSQL integration
    pg_config = IntegrationConfig(
        name="postgresql",
        enabled=config.get('database.postgresql.enabled', False),
        config=config.get('database.postgresql', {}),
        dependencies=[]
    )
    
    if pg_config.enabled:
        manager.register_integration(PostgreSQLIntegration, pg_config)
        print("✅ PostgreSQL integration registered")
    else:
        print("⚠️  PostgreSQL integration disabled in config")
    
    # 4. Start all integrations
    print("\n🚀 Starting Integrations...")
    success = await manager.start_all()
    
    if not success:
        print("❌ Failed to start integrations")
        return
    
    print("✅ All integrations started successfully")
    
    # 5. List running integrations
    print("\n📊 Integration Status:")
    integrations = manager.list_integrations()
    for name, status in integrations.items():
        emoji = "✅" if status['running'] else "❌"
        print(f"  {emoji} {name}: Running={status['running']}, Ready={status['ready']}")
    
    # 6. Health check
    print("\n🏥 Health Check:")
    health = await manager.health_check_all()
    for name, status in health.items():
        emoji = "✅" if status['status'] == 'healthy' else "❌"
        print(f"  {emoji} {name}: {status['status']}")
    
    # 7. Demonstrate knowledge graph operations
    print("\n🧠 Knowledge Graph Operations:")
    
    # Create a knowledge graph
    kg = KnowledgeGraph("example_graph")
    
    # Add some data
    kg.add_node("alice", {"type": "person", "age": 30})
    kg.add_node("bob", {"type": "person", "age": 25})
    kg.add_node("company_x", {"type": "organization"})
    
    kg.add_edge(("alice", "company_x"), edge_type="works_at")
    kg.add_edge(("bob", "company_x"), edge_type="works_at")
    kg.add_edge(("alice", "bob"), edge_type="knows")
    
    print(f"✅ Created knowledge graph with {len(kg.nodes)} nodes and {len(kg.edges)} edges")
    
    # 8. Save to database (if PostgreSQL is available)
    pg_integration = manager.get_integration("postgresql")
    if pg_integration and pg_integration.is_ready:
        print("\n💾 Saving to Database:")
        try:
            success = await pg_integration.save_knowledge_graph(kg)
            if success:
                print("✅ Knowledge graph saved to PostgreSQL")
            else:
                print("❌ Failed to save knowledge graph")
        except Exception as e:
            print(f"❌ Database operation failed: {e}")
    
    # 9. Get metrics
    print("\n📈 Integration Metrics:")
    metrics = manager.get_metrics()
    for name, metric_data in metrics.items():
        if 'error' not in metric_data:
            print(f"  📊 {name}:")
            for key, value in metric_data.items():
                print(f"    {key}: {value}")
    
    # 10. Demonstrate graceful shutdown
    print("\n🛑 Shutting Down...")
    shutdown_success = await manager.stop_all()
    
    if shutdown_success:
        print("✅ All integrations shut down gracefully")
    else:
        print("❌ Some integrations failed to shut down properly")
    
    print("\n🎯 Example completed successfully!")
    print("=" * 60)
    print("📚 Next Steps:")
    print("  1. Configure your database credentials in anant_integration.yaml")
    print("  2. Install database dependencies (asyncpg for PostgreSQL)")
    print("  3. Enable additional integrations (API, monitoring, etc.)")
    print("  4. Build your enterprise knowledge graph application!")


def demo_integration_architecture():
    """Show the integration architecture"""
    print("\n🏗️  ANANT INTEGRATION ARCHITECTURE")
    print("=" * 60)
    
    architecture = {
        "Core Anant Library": [
            "✅ Hypergraph (refactored - 85% reduction)",
            "✅ KnowledgeGraph (refactored - 80% reduction)", 
            "✅ HierarchicalKG (refactored - 77% reduction)",
            "✅ NaturalLanguage (refactored - 78% reduction)"
        ],
        "Integration Ecosystem": [
            "📦 Database: PostgreSQL, Neo4j, MongoDB, Redis, Elasticsearch", 
            "🔄 ETL: File processors, API extractors, streaming pipelines",
            "🔐 Security: Authentication, authorization, encryption, audit",
            "🌐 API: REST, GraphQL, gRPC, WebSocket endpoints",
            "⚙️  Config: Environment management, secrets, validation",
            "📊 Monitoring: Metrics, logging, alerting, health checks",
            "🚀 Deployment: Docker, Kubernetes, auto-scaling",
            "⚡ Streaming: Kafka, real-time processing, event sourcing"
        ],
        "Enterprise Features": [
            "🏢 Production-ready patterns and best practices",
            "🔧 Modular, extensible architecture", 
            "📈 Performance monitoring and optimization",
            "🛡️  Enterprise security and compliance",
            "🌍 Multi-environment deployment support",
            "🔄 Graceful startup/shutdown and error handling"
        ]
    }
    
    for category, features in architecture.items():
        print(f"\n{category}:")
        for feature in features:
            print(f"  {feature}")
    
    print(f"\n🎊 BENEFITS:")
    print(f"  💡 Core library stays clean and focused")
    print(f"  🔧 Integrations are optional and modular")
    print(f"  🚀 Enterprise-ready without bloating core")
    print(f"  📦 Easy to extend with new integrations")
    print(f"  🧪 Each component can be tested independently")


if __name__ == "__main__":
    # Show architecture overview
    demo_integration_architecture()
    
    # Run main example
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        logger.exception("Example failed with exception")