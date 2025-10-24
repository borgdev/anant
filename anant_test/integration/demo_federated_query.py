#!/usr/bin/env python3
"""
Federated Query Engine Demo
===========================

Demonstration of advanced federated querying capabilities across multiple
data sources with intelligent optimization and result merging.
"""

import json
import time
from anant.kg.core import KnowledgeGraph
from anant.kg.federated_query import (
    FederatedQueryEngine, DataSource, FederationProtocol
)


def create_demo_knowledge_graph():
    """Create a demo knowledge graph for testing"""
    
    # Sample knowledge graph data
    kg_data = {
        'person_1': ['http://example.org/person/1', 'http://example.org/type/Person'],
        'person_2': ['http://example.org/person/2', 'http://example.org/type/Person'],
        'works_at_1': ['http://example.org/person/1', 'http://example.org/relation/worksAt', 'http://example.org/org/1'],
        'works_at_2': ['http://example.org/person/2', 'http://example.org/relation/worksAt', 'http://example.org/org/2'],
        'org_1': ['http://example.org/org/1', 'http://example.org/type/Organization'],
        'org_2': ['http://example.org/org/2', 'http://example.org/type/Organization'],
    }
    
    kg = KnowledgeGraph(kg_data)
    return kg


def setup_demo_data_sources(engine):
    """Set up demo data sources for federation"""
    
    # SPARQL endpoint (simulated)
    sparql_source = DataSource(
        source_id="demo_sparql",
        name="Demo SPARQL Endpoint",
        protocol=FederationProtocol.SPARQL_ENDPOINT,
        endpoint_url="http://demo-sparql.example.com/query",
        capabilities={'sparql', 'rdf', 'inference'},
        priority=2
    )
    
    # REST API source (simulated)
    rest_source = DataSource(
        source_id="demo_rest",
        name="Demo REST API",
        protocol=FederationProtocol.REST_API,
        endpoint_url="http://demo-api.example.com/data",
        capabilities={'json', 'pagination', 'filtering'},
        priority=1
    )
    
    # Native KG source
    native_source = DataSource(
        source_id="demo_native",
        name="Demo Native Knowledge Graph",
        protocol=FederationProtocol.NATIVE_KG,
        endpoint_url="local://kg",
        capabilities={'semantic_search', 'reasoning', 'embeddings'},
        priority=3
    )
    
    # GraphQL source (simulated)
    graphql_source = DataSource(
        source_id="demo_graphql",
        name="Demo GraphQL Endpoint",
        protocol=FederationProtocol.GRAPHQL,
        endpoint_url="http://demo-graphql.example.com/graphql",
        capabilities={'graphql', 'typed_queries', 'introspection'},
        priority=2
    )
    
    # Register all sources
    for source in [sparql_source, rest_source, native_source, graphql_source]:
        engine.register_data_source(source)
    
    print(f"✅ Registered {len(engine.data_sources)} data sources for federation")


def demo_sparql_federated_query(engine):
    """Demonstrate federated SPARQL query execution"""
    
    print("\n🔍 Demo 1: Federated SPARQL Query")
    print("=" * 50)
    
    sparql_query = """
    SELECT ?person ?name WHERE {
        ?person rdf:type Person .
        ?person rdfs:label ?name .
        FILTER (?name != "")
    }
    ORDER BY ?name
    LIMIT 10
    """
    
    print("Query:")
    print(sparql_query)
    
    start_time = time.time()
    result = engine.execute_federated_query(
        query=sparql_query,
        query_type='sparql',
        optimization_level=2
    )
    execution_time = time.time() - start_time
    
    print(f"\n📊 Results:")
    print(f"   Success: {result.success}")
    print(f"   Execution Time: {execution_time:.3f}s")
    print(f"   Sources Used: {len(result.sources_used or [])}")
    print(f"   Fragments Executed: {len(result.fragment_results or [])}")
    print(f"   Total Rows: {result.total_rows}")
    
    if result.metadata:
        print(f"   Optimization Level: {result.metadata.get('optimization_level', 'N/A')}")
        print(f"   Parallel Groups: {result.metadata.get('parallel_groups', 'N/A')}")


def demo_pattern_federated_query(engine):
    """Demonstrate federated pattern-based query"""
    
    print("\n🔍 Demo 2: Federated Pattern Query")
    print("=" * 50)
    
    pattern_query = {
        'subject': '?person',
        'predicate': 'worksAt',
        'object': '?organization',
        'constraints': {
            'person_type': 'Person',
            'org_type': 'Organization'
        }
    }
    
    print("Pattern Query:")
    print(json.dumps(pattern_query, indent=2))
    
    start_time = time.time()
    result = engine.execute_federated_query(
        query=json.dumps(pattern_query),
        query_type='pattern',
        optimization_level=1
    )
    execution_time = time.time() - start_time
    
    print(f"\n📊 Results:")
    print(f"   Success: {result.success}")
    print(f"   Execution Time: {execution_time:.3f}s")
    print(f"   Sources Used: {result.sources_used}")
    print(f"   Data Retrieved: {bool(result.data)}")


def demo_natural_language_query(engine):
    """Demonstrate natural language federated query"""
    
    print("\n🔍 Demo 3: Natural Language Federated Query")
    print("=" * 50)
    
    nl_query = "Find all persons who work at organizations"
    
    print(f"Natural Language Query: '{nl_query}'")
    
    start_time = time.time()
    result = engine.execute_federated_query(
        query=nl_query,
        query_type='natural',
        optimization_level=1
    )
    execution_time = time.time() - start_time
    
    print(f"\n📊 Results:")
    print(f"   Success: {result.success}")
    print(f"   Execution Time: {execution_time:.3f}s")
    print(f"   Query Conversion: Automatic NL → Query Language")
    print(f"   Sources Targeted: {len(result.sources_used or [])}")


def demo_source_selection_optimization(engine):
    """Demonstrate intelligent source selection"""
    
    print("\n🔍 Demo 4: Source Selection & Optimization")
    print("=" * 50)
    
    # Query with specific requirements
    complex_query = """
    SELECT ?entity ?type WHERE {
        ?entity rdf:type ?type .
        ?entity rdfs:subClassOf* Animal .
        FILTER CONTAINS(?label, "mammal")
    }
    """
    
    print("Complex Query with Requirements:")
    print("- RDF/RDFS support")
    print("- Inference capabilities (subClassOf*)")
    print("- Full-text search (CONTAINS)")
    
    # Analyze requirements
    requirements = engine._analyze_query_requirements(complex_query, 'sparql')
    print(f"\n🔍 Detected Requirements: {requirements}")
    
    # Select optimal sources
    optimal_sources = engine._select_optimal_sources(complex_query, 'sparql')
    print(f"📍 Selected Sources: {optimal_sources}")
    
    # Show source capabilities
    print("\n📋 Source Capabilities:")
    for source_id in optimal_sources:
        capabilities = engine.source_capabilities.get(source_id, set())
        source = engine.data_sources.get(source_id)
        if source:
            print(f"   {source.name}: {capabilities}")


def demo_federation_statistics(engine):
    """Show federation statistics and health"""
    
    print("\n📈 Federation Statistics")
    print("=" * 50)
    
    stats = engine.get_federation_statistics()
    
    print(f"🗄️  Data Sources:")
    print(f"   Total Registered: {stats['sources']['total_registered']}")
    print(f"   Healthy Sources: {stats['sources']['healthy_sources']}")
    
    print(f"\n⚡ Execution Stats:")
    print(f"   Total Queries: {stats['execution']['total_queries']}")
    print(f"   Cache Hits: {stats['execution']['cache_hits']}")
    print(f"   Average Exec Time: {stats['execution']['average_execution_time']:.3f}s")
    print(f"   Success Rate: {stats['execution']['success_rate']:.1%}")
    
    print(f"\n🔧 Performance:")
    print(f"   Active Queries: {stats['performance']['active_queries']}")
    print(f"   Plan Cache Size: {stats['performance']['plan_cache_size']}")
    print(f"   Result Cache Size: {stats['performance']['result_cache_size']}")
    
    print(f"\n🌐 Source Health:")
    for source_id, details in stats['sources']['source_details'].items():
        status_emoji = "✅" if details['status'] == 'healthy' else "⚠️" if details['status'] == 'degraded' else "❌"
        print(f"   {status_emoji} {details['name']}: {details['status']} ({details['response_time_ms']:.0f}ms)")


def main():
    """Run federated query engine demo"""
    
    print("🚀 ANANT Federated Query Engine Demo")
    print("=" * 60)
    
    # Create demo knowledge graph
    print("📊 Setting up demo environment...")
    kg = create_demo_knowledge_graph()
    
    # Initialize federated query engine
    engine = FederatedQueryEngine(kg)
    
    # Set up demo data sources
    setup_demo_data_sources(engine)
    
    # Run demos
    try:
        demo_sparql_federated_query(engine)
        demo_pattern_federated_query(engine)
        demo_natural_language_query(engine)
        demo_source_selection_optimization(engine)
        demo_federation_statistics(engine)
        
        print(f"\n✅ Demo completed successfully!")
        print(f"📚 Federated Query Engine provides:")
        print(f"   • Cross-database query execution")
        print(f"   • Intelligent source selection")
        print(f"   • Parallel fragment execution")
        print(f"   • Query optimization integration")
        print(f"   • Result merging and deduplication")
        print(f"   • Health monitoring and failover")
        print(f"   • Multiple protocol support")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()