#!/usr/bin/env python3
"""
Natural Language Interface Demo
==============================

This script demonstrates the capabilities of the Natural Language Interface
for knowledge graphs, including:
- Natural language query processing
- Intent recognition and entity extraction  
- Query translation to formal languages
- Multi-turn conversation support
- Context management and pronoun resolution
"""

import json
from datetime import datetime
from anant.kg import KnowledgeGraph
from anant.kg.natural_language import (
    NaturalLanguageInterface,
    Intent,
    QueryType,
    ConfidenceLevel
)
from anant.kg.federated_query import FederatedQueryEngine


def create_sample_data():
    """Create sample knowledge graph data for demonstration"""
    
    sample_kg = KnowledgeGraph()
    
    # Add sample entities and relationships
    sample_triples = [
        ("John Smith", "works_at", "Microsoft Corporation"),
        ("John Smith", "position", "Software Engineer"),
        ("John Smith", "age", "30"),
        ("Microsoft Corporation", "type", "Technology Company"),
        ("Microsoft Corporation", "founded", "1975"),
        ("Jane Doe", "works_at", "Google Inc"),
        ("Jane Doe", "position", "Product Manager"),
        ("Jane Doe", "age", "28"),
        ("Google Inc", "type", "Technology Company"),
        ("Alice Johnson", "works_at", "Microsoft Corporation"),
        ("Alice Johnson", "position", "Data Scientist"),
        ("Bob Wilson", "works_at", "Apple Inc"),
        ("Bob Wilson", "position", "iOS Developer")
    ]
    
    # In a real implementation, we'd load these into the KG
    # For demo purposes, we'll simulate federated query responses
    
    return sample_kg


def create_mock_federated_engine():
    """Create a mock federated query engine for demo purposes"""
    
    from unittest.mock import Mock
    from anant.kg.federated_query import FederatedQueryResult
    
    mock_engine = Mock()
    
    # Define response patterns based on query content
    def mock_execute_federated_query(query, **kwargs):
        query_lower = query.lower()
        
        if 'count' in query_lower:
            if 'people' in query_lower or 'person' in query_lower:
                return FederatedQueryResult(
                    query_id="count_people",
                    success=True,
                    data=[{'count': 5}],
                    total_rows=1,
                    sources_used=['sample_kg']
                )
            elif 'company' in query_lower or 'organization' in query_lower:
                return FederatedQueryResult(
                    query_id="count_companies",
                    success=True,
                    data=[{'count': 3}],
                    total_rows=1,
                    sources_used=['sample_kg']
                )
        
        elif 'john smith' in query_lower:
            return FederatedQueryResult(
                query_id="find_john",
                success=True,
                data=[{
                    'entity': 'http://example.org/john_smith',
                    'name': 'John Smith',
                    'position': 'Software Engineer',
                    'company': 'Microsoft Corporation',
                    'age': '30'
                }],
                total_rows=1,
                sources_used=['sample_kg']
            )
        
        elif 'microsoft' in query_lower and ('employee' in query_lower or 'work' in query_lower):
            return FederatedQueryResult(
                query_id="microsoft_employees",
                success=True,
                data=[
                    {
                        'entity': 'http://example.org/john_smith',
                        'name': 'John Smith',
                        'position': 'Software Engineer'
                    },
                    {
                        'entity': 'http://example.org/alice_johnson', 
                        'name': 'Alice Johnson',
                        'position': 'Data Scientist'
                    }
                ],
                total_rows=2,
                sources_used=['sample_kg']
            )
        
        elif 'people' in query_lower or 'person' in query_lower:
            return FederatedQueryResult(
                query_id="find_people",
                success=True,
                data=[
                    {'entity': 'http://example.org/john_smith', 'name': 'John Smith'},
                    {'entity': 'http://example.org/jane_doe', 'name': 'Jane Doe'},
                    {'entity': 'http://example.org/alice_johnson', 'name': 'Alice Johnson'},
                    {'entity': 'http://example.org/bob_wilson', 'name': 'Bob Wilson'}
                ],
                total_rows=4,
                sources_used=['sample_kg']
            )
        
        elif 'company' in query_lower or 'organization' in query_lower:
            return FederatedQueryResult(
                query_id="find_companies",
                success=True,
                data=[
                    {'entity': 'http://example.org/microsoft', 'name': 'Microsoft Corporation'},
                    {'entity': 'http://example.org/google', 'name': 'Google Inc'},
                    {'entity': 'http://example.org/apple', 'name': 'Apple Inc'}
                ],
                total_rows=3,
                sources_used=['sample_kg']
            )
        
        else:
            # Generic response for other queries
            return FederatedQueryResult(
                query_id="generic_query",
                success=True,
                data=[{'message': 'No specific results available'}],
                total_rows=0,
                sources_used=['sample_kg']
            )
    
    mock_engine.execute_federated_query.side_effect = mock_execute_federated_query
    return mock_engine


def demo_basic_query_processing():
    """Demonstrate basic natural language query processing"""
    
    print("=" * 60)
    print("BASIC QUERY PROCESSING DEMO")
    print("=" * 60)
    
    # Create knowledge graph and federated engine
    kg = create_sample_data()
    federated_engine = create_mock_federated_engine()
    
    # Create natural language interface
    nl_interface = NaturalLanguageInterface(
        knowledge_graph=kg,
        federated_engine=federated_engine,
        use_transformers=False  # Disable transformers for demo
    )
    
    # Test queries
    test_queries = [
        "Find all people in the system",
        "How many employees work at Microsoft?",
        "Who is John Smith?",
        "List all companies",
        "Count the total number of people"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: \"{query}\"")
        print("-" * 40)
        
        result = nl_interface.process_query(query)
        
        print(f"Intent: {result.interpretation.intent.value}")
        print(f"Query Type: {result.interpretation.query_type.value}")
        print(f"Confidence: {result.interpretation.confidence:.2f} ({result.metadata['confidence_level'].value})")
        print(f"Formal Query: {result.formal_query[:100]}{'...' if len(result.formal_query) > 100 else ''}")
        print(f"Response: {result.response_text}")
        
        if result.execution_result:
            print(f"Results Count: {result.execution_result.total_rows}")
        
        if result.suggestions:
            print(f"Suggestions: {', '.join(result.suggestions[:2])}")
    
    return nl_interface


def demo_conversation_context():
    """Demonstrate multi-turn conversation with context"""
    
    print("\n\n" + "=" * 60)
    print("CONVERSATION CONTEXT DEMO")
    print("=" * 60)
    
    # Create interface (reuse from previous demo or create new)
    kg = create_sample_data()
    federated_engine = create_mock_federated_engine()
    
    nl_interface = NaturalLanguageInterface(
        knowledge_graph=kg,
        federated_engine=federated_engine,
        use_transformers=False
    )
    
    conversation_id = "demo_conversation"
    
    # Multi-turn conversation
    conversation = [
        "Find John Smith",
        "What company does he work for?",
        "Who else works there?",
        "How many people work at Microsoft?"
    ]
    
    print(f"\n💬 Starting conversation (ID: {conversation_id})")
    print("-" * 40)
    
    for i, query in enumerate(conversation, 1):
        print(f"\n[Turn {i}] User: \"{query}\"")
        
        result = nl_interface.process_query(
            query, 
            conversation_id=conversation_id,
            user_id="demo_user"
        )
        
        print(f"[Turn {i}] Assistant: {result.response_text}")
        
        if result.interpretation.confidence < 0.5:
            print(f"[Turn {i}] Low confidence. Suggestions: {', '.join(result.suggestions[:2])}")
        
        # Show context evolution
        if conversation_id in nl_interface.conversations:
            context = nl_interface.conversations[conversation_id]
            print(f"[Turn {i}] Context: {len(context.previous_queries)} previous queries, {len(context.entity_context)} entities")
    
    return nl_interface


def demo_entity_extraction():
    """Demonstrate entity extraction and relation detection"""
    
    print("\n\n" + "=" * 60)
    print("ENTITY EXTRACTION DEMO")
    print("=" * 60)
    
    kg = create_sample_data()
    federated_engine = create_mock_federated_engine()
    
    nl_interface = NaturalLanguageInterface(
        knowledge_graph=kg,
        federated_engine=federated_engine,
        use_transformers=False
    )
    
    # Complex queries with entities and relations
    complex_queries = [
        "Find people who work at Microsoft Corporation in Seattle",
        "Show me employees with age greater than 25",
        "List software engineers at technology companies",
        "Who are the managers working with John Smith?"
    ]
    
    for query in complex_queries:
        print(f"\n🔍 Analyzing: \"{query}\"")
        print("-" * 40)
        
        result = nl_interface.process_query(query)
        
        interpretation = result.interpretation
        
        print(f"Entities found: {len(interpretation.entities)}")
        for entity in interpretation.entities:
            print(f"  - {entity.text} ({entity.entity_type}) [confidence: {entity.confidence:.2f}]")
        
        print(f"Relations found: {len(interpretation.relations)}")
        for relation in interpretation.relations:
            print(f"  - {relation.text} ({relation.relation_type}) [confidence: {relation.confidence:.2f}]")
        
        if interpretation.constraints:
            print(f"Constraints: {interpretation.constraints}")
        
        if interpretation.temporal_info:
            print(f"Temporal info: {interpretation.temporal_info}")
        
        print(f"Explanation: {interpretation.explanation}")


def demo_query_types():
    """Demonstrate different query types and their handling"""
    
    print("\n\n" + "=" * 60)
    print("QUERY TYPES DEMO")
    print("=" * 60)
    
    kg = create_sample_data()
    federated_engine = create_mock_federated_engine()
    
    nl_interface = NaturalLanguageInterface(
        knowledge_graph=kg,
        federated_engine=federated_engine,
        use_transformers=False
    )
    
    # Queries of different types
    query_examples = {
        "Entity Search": [
            "Find people",
            "Show me all companies"
        ],
        "Aggregation": [
            "How many people are there?",
            "Count employees at Microsoft"
        ],
        "Boolean": [
            "Is John Smith a manager?",
            "Does Microsoft have offices in Seattle?"
        ],
        "Comparison": [
            "Compare Microsoft and Google",
            "Who is older, John or Jane?"
        ],
        "Temporal": [
            "When was Microsoft founded?",
            "Show me events from last year"
        ],
        "Relationship": [
            "Who works with John Smith?",
            "What companies are related to technology?"
        ]
    }
    
    for category, queries in query_examples.items():
        print(f"\n📊 {category} Queries:")
        print("-" * 30)
        
        for query in queries:
            result = nl_interface.process_query(query)
            
            print(f"  Query: \"{query}\"")
            print(f"  Type: {result.interpretation.query_type.value}")
            print(f"  Intent: {result.interpretation.intent.value}")
            print(f"  Confidence: {result.interpretation.confidence:.2f}")
            print()


def demo_performance_stats():
    """Demonstrate performance monitoring and statistics"""
    
    print("\n\n" + "=" * 60)
    print("PERFORMANCE STATISTICS DEMO")
    print("=" * 60)
    
    kg = create_sample_data()
    federated_engine = create_mock_federated_engine()
    
    nl_interface = NaturalLanguageInterface(
        knowledge_graph=kg,
        federated_engine=federated_engine,
        use_transformers=False
    )
    
    # Process several queries to generate statistics
    test_queries = [
        "Find all people",
        "Count employees", 
        "Who is John Smith?",
        "Show me companies",
        "How many people work at Microsoft?",
        "List software engineers",
        "Find managers",
        "Show technology companies"
    ]
    
    print("Processing queries to generate statistics...")
    for query in test_queries:
        nl_interface.process_query(query)
    
    # Get and display statistics
    stats = nl_interface.get_interface_statistics()
    
    print(f"\n📈 Processing Statistics:")
    print("-" * 25)
    print(f"Total queries processed: {stats['processing_stats']['queries_processed']}")
    print(f"Successful interpretations: {stats['processing_stats']['successful_interpretations']}")
    print(f"Average confidence: {stats['processing_stats']['average_confidence']:.3f}")
    print(f"Cache hits: {stats['processing_stats']['cache_hits']}")
    print(f"Cache hit rate: {stats['cache_stats']['cache_hit_rate']:.2%}")
    
    print(f"\n🎯 Intent Distribution:")
    print("-" * 20)
    for intent, count in stats['processing_stats']['intent_distribution'].items():
        if count > 0:
            print(f"  {intent}: {count}")
    
    print(f"\n🔍 Query Type Distribution:")
    print("-" * 25)
    for query_type, count in stats['processing_stats']['query_type_distribution'].items():
        if count > 0:
            print(f"  {query_type}: {count}")
    
    print(f"\n💬 Conversation Statistics:")
    print("-" * 25)
    print(f"Active conversations: {stats['conversation_stats']['active_conversations']}")
    print(f"Total context entities: {stats['conversation_stats']['total_context_entities']}")
    
    # Show recent performance profile
    print(f"\n⏱️  Recent Query Performance:")
    print("-" * 27)
    print(f"Query: \"{test_queries[-1]}\"")
    result = nl_interface.process_query("Sample performance query")
    if 'profiler_report' in result.metadata:
        profile = result.metadata['profiler_report']
        print(f"Total time: {profile['total_execution_time']:.4f}s")
        print(f"Slowest checkpoint: {profile['checkpoint_analysis']['slowest_checkpoint']}")
        print(f"Fastest checkpoint: {profile['checkpoint_analysis']['fastest_checkpoint']}")


def main():
    """Run all demonstration scenarios"""
    
    print("🤖 Natural Language Interface for Knowledge Graphs")
    print("=" * 60)
    print("This demo showcases advanced natural language processing")
    print("capabilities for knowledge graph querying and interaction.")
    print("=" * 60)
    
    try:
        # Run all demos
        nl_interface = demo_basic_query_processing()
        demo_conversation_context()
        demo_entity_extraction()
        demo_query_types()
        demo_performance_stats()
        
        print("\n\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        print("✅ Basic query processing with intent classification")
        print("✅ Multi-turn conversation with context management")
        print("✅ Entity extraction and relation detection")
        print("✅ Multiple query types (search, count, boolean, etc.)")
        print("✅ Performance monitoring and statistics")
        print("\n🎯 The Natural Language Interface successfully demonstrated:")
        print("   • Robust natural language understanding")
        print("   • Flexible query translation to formal languages")
        print("   • Context-aware conversation management")
        print("   • Comprehensive performance monitoring")
        print("   • Integration with federated query execution")
        
        print(f"\n📊 Final Interface State:")
        print(f"   • Total queries processed: {nl_interface.stats['queries_processed']}")
        print(f"   • Cache entries: {len(nl_interface.interpretation_cache)}")
        print(f"   • Active conversations: {len(nl_interface.conversations)}")
        print(f"   • Success rate: {nl_interface.stats['successful_interpretations'] / max(1, nl_interface.stats['queries_processed']) * 100:.1f}%")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✨ Natural Language Interface demo completed successfully!")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)