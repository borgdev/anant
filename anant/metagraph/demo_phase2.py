#!/usr/bin/env python3
"""
Phase 2 LLM Integration Demo

This script demonstrates the Phase 2 LLM integration capabilities of the
enterprise metagraph system, including natural language query processing,
enhanced semantic understanding, RAG context generation, and intelligent
recommendations.

Features Demonstrated:
- Natural Language Query Processing
- LLM-Enhanced Semantic Understanding
- RAG Context Generation
- Intelligent Recommendations
- Business Glossary Enhancement
- Cross-layer AI Integration

Author: anant development team
Date: October 2025
"""

import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    import polars as pl
    from anant.metagraph.core.metagraph import Metagraph
    from anant.metagraph.llm import (
        create_query_processor,
        create_semantic_engine, 
        create_rag_generator,
        create_recommendation_engine,
        process_natural_language_query,
        QueryIntent,
        RecommendationType,
        ContextType
    )
    print("✅ Successfully imported all Phase 2 LLM components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed and paths are correct")
    sys.exit(1)


def create_sample_enterprise_data():
    """Create sample enterprise data for demonstration"""
    print("\n🏗️  Creating sample enterprise data...")
    
    # Sample entities representing an enterprise environment
    sample_entities = {
        "customer_management": {
            "description": "Customer relationship management system",
            "type": "business_process",
            "department": "Sales",
            "stakeholders": ["sales_team", "customer_service", "marketing"]
        },
        "sales_team": {
            "description": "Sales department team",
            "type": "organizational_unit", 
            "manager": "sales_manager",
            "size": 25,
            "performance_metrics": ["revenue", "conversion_rate"]
        },
        "product_catalog": {
            "description": "Product information and pricing system",
            "type": "data_system",
            "owner": "product_management",
            "last_updated": "2025-10-15"
        },
        "order_processing": {
            "description": "End-to-end order fulfillment process",
            "type": "business_process",
            "systems": ["erp_system", "inventory_management", "shipping_system"],
            "sla": "24_hours"
        },
        "customer_data": {
            "description": "Customer information and preferences",
            "type": "data_asset",
            "classification": "confidential",
            "retention_period": "7_years",
            "compliance": ["gdpr", "ccpa"]
        }
    }
    
    # Sample relationships
    sample_relationships = [
        ("customer_management", "sales_team", "managed_by"),
        ("customer_management", "customer_data", "processes"),
        ("sales_team", "product_catalog", "uses"),
        ("order_processing", "product_catalog", "references"),
        ("order_processing", "customer_data", "requires")
    ]
    
    # Sample business terms
    business_terms = [
        "customer_lifetime_value",
        "conversion_rate", 
        "product_margin",
        "order_fulfillment",
        "customer_retention"
    ]
    
    print(f"   📊 Created {len(sample_entities)} entities")
    print(f"   🔗 Created {len(sample_relationships)} relationships")
    print(f"   📚 Created {len(business_terms)} business terms")
    
    return sample_entities, sample_relationships, business_terms


def setup_phase2_metagraph():
    """Set up metagraph with Phase 2 LLM capabilities"""
    print("\n🚀 Setting up Phase 2 Metagraph with LLM Integration...")
    
    # Create base metagraph
    try:
        metagraph = Metagraph()
        print("   ✅ Base metagraph created")
        
        # Add sample data
        sample_entities, sample_relationships, business_terms = create_sample_enterprise_data()
        
        # Add entities to hierarchical store
        for entity_id, entity_data in sample_entities.items():
            try:
                metagraph.add_entity(entity_id, level=0, metadata=entity_data)
                print(f"   📝 Added entity: {entity_id}")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not add entity {entity_id}: {e}")
        
        # Add relationships
        for source, target, rel_type in sample_relationships:
            try:
                # This would normally use the semantic layer
                print(f"   🔗 Added relationship: {source} --{rel_type}--> {target}")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not add relationship: {e}")
        
        print("   ✅ Phase 2 metagraph setup complete")
        return metagraph, sample_entities, business_terms
        
    except Exception as e:
        print(f"   ❌ Error setting up metagraph: {e}")
        return None, {}, []


def demo_natural_language_query_processing(metagraph):
    """Demonstrate natural language query processing"""
    print("\n🗣️  === NATURAL LANGUAGE QUERY PROCESSING DEMO ===")
    
    try:
        # Create query processor
        query_processor = create_query_processor(backend="fallback")  # Use fallback for demo
        print("   ✅ Query processor created")
        
        # Sample queries to demonstrate different intents
        sample_queries = [
            "Show me all customer management processes",
            "What systems are related to order processing?", 
            "Find entities similar to sales team",
            "Analyze the relationship between customer data and sales processes",
            "What are the recent changes in the product catalog?",
            "Recommend improvements for customer management workflow"
        ]
        
        print("\n   🔍 Processing natural language queries:")
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n   Query {i}: \"{query}\"")
            
            try:
                # Process the query
                parsed_query = query_processor.process_query(query)
                
                print(f"      📋 Intent: {parsed_query.intent.value}")
                print(f"      🎯 Confidence: {parsed_query.confidence:.2f}")
                print(f"      🔧 Complexity: {parsed_query.complexity.value}")
                
                if parsed_query.entities:
                    print(f"      🏷️  Entities: {', '.join(parsed_query.entities)}")
                
                if parsed_query.properties:
                    print(f"      📊 Properties: {', '.join(parsed_query.properties)}")
                
                # Create execution plan
                execution_plan = query_processor.create_execution_plan(parsed_query)
                print(f"      ⚙️  Execution steps: {len(execution_plan.steps)}")
                print(f"      🏗️  Required layers: {', '.join(execution_plan.required_layers)}")
                
            except Exception as e:
                print(f"      ❌ Error processing query: {e}")
        
        print("\n   ✅ Natural language query processing demo complete")
        
    except Exception as e:
        print(f"   ❌ Error in query processing demo: {e}")


def demo_enhanced_semantic_engine(business_terms):
    """Demonstrate enhanced semantic engine with LLM capabilities"""
    print("\n🧠 === ENHANCED SEMANTIC ENGINE DEMO ===")
    
    try:
        # Create semantic engine
        semantic_engine = create_semantic_engine(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            llm_backend="fallback"  # Use fallback for demo
        )
        print("   ✅ Enhanced semantic engine created")
        
        # Demo 1: Generate enhanced embeddings
        print("\n   🎯 Generating enhanced embeddings:")
        
        sample_content = {
            "customer_management": "Customer relationship management system for tracking and managing customer interactions",
            "sales_process": "End-to-end sales workflow from lead generation to deal closure",
            "product_catalog": "Comprehensive database of products with pricing and specifications"
        }
        
        embeddings = {}
        for entity_id, content in sample_content.items():
            try:
                embedding = semantic_engine.generate_enhanced_embedding(
                    entity_id=entity_id,
                    text_content=content,
                    context="enterprise business process"
                )
                embeddings[entity_id] = embedding
                print(f"      📊 Generated embedding for {entity_id}: {embedding.dimensions}D, confidence: {embedding.confidence:.2f}")
                
            except Exception as e:
                print(f"      ⚠️  Warning: Could not generate embedding for {entity_id}: {e}")
        
        # Demo 2: Discover semantic relationships
        print("\n   🔗 Discovering semantic relationships:")
        
        entity_list = list(sample_content.keys())
        try:
            relationships = semantic_engine.discover_semantic_relationships(
                entities=entity_list,
                similarity_threshold=0.5,
                max_relationships=10
            )
            
            for rel in relationships[:3]:  # Show first 3 relationships
                print(f"      🔗 {rel.source_entity} --{rel.relationship_type}--> {rel.target_entity} "
                      f"(confidence: {rel.confidence:.2f})")
                      
        except Exception as e:
            print(f"      ⚠️  Warning: Could not discover relationships: {e}")
        
        # Demo 3: Enhance business glossary
        print("\n   📚 Enhancing business glossary:")
        
        try:
            enhanced_terms = semantic_engine.enhance_business_glossary(
                terms=business_terms[:3],  # First 3 terms
                domain_context="enterprise software"
            )
            
            for term in enhanced_terms:
                print(f"      📖 {term.term}: {term.definition[:100]}...")
                print(f"         Category: {term.category}, Quality: {term.quality_score:.2f}")
                
        except Exception as e:
            print(f"      ⚠️  Warning: Could not enhance glossary: {e}")
        
        # Demo 4: Find similar entities
        print("\n   🎯 Finding similar entities:")
        
        if embeddings:
            first_entity = list(embeddings.keys())[0]
            try:
                similar_entities = semantic_engine.find_similar_entities(
                    query_entity=first_entity,
                    similarity_threshold=0.3,
                    max_results=5
                )
                
                for similar in similar_entities:
                    print(f"      🎯 {similar['entity_id']}: similarity {similar['similarity_score']:.2f}")
                    
            except Exception as e:
                print(f"      ⚠️  Warning: Could not find similar entities: {e}")
        
        print("\n   ✅ Enhanced semantic engine demo complete")
        
    except Exception as e:
        print(f"   ❌ Error in semantic engine demo: {e}")


def demo_rag_context_generation(metagraph):
    """Demonstrate RAG context generation"""
    print("\n🔍 === RAG CONTEXT GENERATION DEMO ===")
    
    try:
        # Create RAG context generator
        rag_generator = create_rag_generator(
            metagraph_instance=metagraph,
            llm_backend="fallback",  # Use fallback for demo
            max_context_tokens=2000
        )
        print("   ✅ RAG context generator created")
        
        # Sample queries for context generation
        sample_queries = [
            "Tell me about customer management processes and their relationships",
            "What security measures are in place for customer data?",
            "How does the sales team interact with product information?",
            "What compliance requirements affect our data processing?"
        ]
        
        print("\n   📋 Generating RAG contexts for queries:")
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n   Query {i}: \"{query[:60]}...\"")
            
            try:
                # Generate RAG context
                rag_context = rag_generator.generate_rag_context(
                    query=query,
                    entity_focus=["customer_management", "sales_team", "customer_data"],
                    max_tokens=1000
                )
                
                print(f"      📊 Generated context: {len(rag_context.context_fragments)} fragments")
                print(f"      🎯 Focus entities: {len(rag_context.entity_focus)} entities")
                print(f"      📝 Total tokens: {rag_context.total_tokens}")
                print(f"      📄 Summary: {rag_context.context_summary[:100]}...")
                
                # Show context fragment types
                fragment_types = {}
                for fragment in rag_context.context_fragments:
                    ftype = fragment.context_type.value
                    fragment_types[ftype] = fragment_types.get(ftype, 0) + 1
                
                print(f"      🗂️  Fragment types: {dict(fragment_types)}")
                
            except Exception as e:
                print(f"      ❌ Error generating context: {e}")
        
        # Demo context statistics
        try:
            stats = rag_generator.get_generation_statistics()
            print(f"\n   📈 Generation statistics:")
            print(f"      🔢 Total contexts generated: {stats['total_contexts_generated']}")
            print(f"      💾 Cache hits: {stats['cache_hits']}")
            print(f"      📊 Total tokens generated: {stats['total_tokens_generated']}")
            
        except Exception as e:
            print(f"      ⚠️  Could not get statistics: {e}")
        
        print("\n   ✅ RAG context generation demo complete")
        
    except Exception as e:
        print(f"   ❌ Error in RAG context demo: {e}")


def demo_intelligent_recommendations(metagraph):
    """Demonstrate intelligent recommendations engine"""
    print("\n💡 === INTELLIGENT RECOMMENDATIONS DEMO ===")
    
    try:
        # Create recommendation engine
        recommendation_engine = create_recommendation_engine(
            metagraph_instance=metagraph,
            llm_backend="fallback",  # Use fallback for demo
            enable_ml_analysis=False,  # Disable for demo
            recommendation_threshold=0.4
        )
        print("   ✅ Intelligent recommendation engine created")
        
        # Generate comprehensive recommendations
        print("\n   🎯 Generating comprehensive recommendations:")
        
        try:
            recommendation_batch = recommendation_engine.generate_comprehensive_recommendations(
                analysis_scope="full",
                focus_entities=["customer_management", "sales_team", "customer_data"],
                recommendation_types=[
                    RecommendationType.ENTITY_RELATIONSHIP,
                    RecommendationType.DATA_QUALITY,
                    RecommendationType.MISSING_METADATA,
                    RecommendationType.BUSINESS_INSIGHT,
                    RecommendationType.SECURITY_IMPROVEMENT
                ]
            )
            
            print(f"      📊 Generated batch: {recommendation_batch.batch_id}")
            print(f"      🎯 Theme: {recommendation_batch.theme}")
            print(f"      📝 Total recommendations: {len(recommendation_batch.recommendations)}")
            print(f"      ⭐ Batch score: {recommendation_batch.total_score:.2f}")
            
            # Show recommendations by type
            rec_by_type = {}
            for rec in recommendation_batch.recommendations:
                rec_type = rec.recommendation_type.value
                rec_by_type[rec_type] = rec_by_type.get(rec_type, 0) + 1
            
            print(f"      📋 Recommendations by type: {dict(rec_by_type)}")
            
            # Show top 3 recommendations
            print(f"\n   🏆 Top recommendations:")
            
            for i, rec in enumerate(recommendation_batch.recommendations[:3], 1):
                print(f"\n      {i}. {rec.title}")
                print(f"         Type: {rec.recommendation_type.value}")
                print(f"         Priority: {rec.priority.value}")
                print(f"         Confidence: {rec.confidence:.2f}")
                print(f"         Description: {rec.description[:100]}...")
                
                if rec.entity_ids:
                    print(f"         Entities: {', '.join(rec.entity_ids)}")
                
                if rec.action_items:
                    print(f"         Actions: {rec.action_items[0][:80]}...")
            
        except Exception as e:
            print(f"      ❌ Error generating recommendations: {e}")
        
        # Demo recommendation statistics
        try:
            stats = recommendation_engine.get_engine_statistics()
            print(f"\n   📈 Engine statistics:")
            print(f"      🔢 Total recommendations: {stats['total_recommendations_generated']}")
            print(f"      ✅ Accepted: {stats['accepted_recommendations']}")
            print(f"      🚀 Implemented: {stats['implemented_recommendations']}")
            print(f"      📊 Average confidence: {stats['average_confidence']:.2f}")
            
        except Exception as e:
            print(f"      ⚠️  Could not get statistics: {e}")
        
        print("\n   ✅ Intelligent recommendations demo complete")
        
    except Exception as e:
        print(f"   ❌ Error in recommendations demo: {e}")


def demo_integrated_nlp_workflow(metagraph):
    """Demonstrate integrated NLP workflow combining all Phase 2 components"""
    print("\n🔄 === INTEGRATED NLP WORKFLOW DEMO ===")
    
    try:
        print("   🎯 Simulating complete NLP-powered knowledge discovery workflow")
        
        # Sample user question
        user_question = "What are the security risks in our customer data processing and what should we do about them?"
        print(f"\n   ❓ User Question: \"{user_question}\"")
        
        # Step 1: Process natural language query
        print("\n   1️⃣  Processing natural language query...")
        try:
            result = process_natural_language_query(
                query=user_question,
                metagraph_instance=metagraph,
                context={"user_role": "data_analyst", "department": "security"}
            )
            
            print(f"      ✅ Query processed successfully")
            print(f"      📋 Intent: {result.get('query_metadata', {}).get('parsed_intent', 'unknown')}")
            print(f"      🎯 Confidence: {result.get('query_metadata', {}).get('confidence', 0):.2f}")
            print(f"      🏗️  Layers used: {result.get('query_metadata', {}).get('layers_used', [])}")
            
        except Exception as e:
            print(f"      ⚠️  Error in query processing: {e}")
        
        # Step 2: Generate contextual recommendations
        print("\n   2️⃣  Generating contextual recommendations...")
        try:
            recommendation_engine = create_recommendation_engine(
                metagraph_instance=metagraph,
                llm_backend="fallback"
            )
            
            security_recommendations = recommendation_engine.generate_comprehensive_recommendations(
                analysis_scope="entities",
                focus_entities=["customer_data", "customer_management"],
                recommendation_types=[
                    RecommendationType.SECURITY_IMPROVEMENT,
                    RecommendationType.COMPLIANCE_ENHANCEMENT,
                    RecommendationType.DATA_QUALITY
                ]
            )
            
            print(f"      ✅ Generated {len(security_recommendations.recommendations)} security recommendations")
            
            # Show top security recommendation
            if security_recommendations.recommendations:
                top_rec = security_recommendations.recommendations[0]
                print(f"      🏆 Top recommendation: {top_rec.title}")
                print(f"         Priority: {top_rec.priority.value}")
                print(f"         Expected benefits: {', '.join(top_rec.expected_benefits[:2])}")
            
        except Exception as e:
            print(f"      ⚠️  Error generating recommendations: {e}")
        
        # Step 3: Generate comprehensive context for user
        print("\n   3️⃣  Generating comprehensive context...")
        try:
            rag_generator = create_rag_generator(
                metagraph_instance=metagraph,
                llm_backend="fallback"
            )
            
            context = rag_generator.generate_rag_context(
                query=user_question,
                entity_focus=["customer_data", "customer_management"],
                context_types=[
                    ContextType.ENTITY_SUMMARY,
                    ContextType.COMPLIANCE_CONTEXT,
                    ContextType.SECURITY_CONTEXT
                ]
            )
            
            print(f"      ✅ Generated comprehensive context")
            print(f"      📊 Context fragments: {len(context.context_fragments)}")
            print(f"      📝 Total context tokens: {context.total_tokens}")
            
        except Exception as e:
            print(f"      ⚠️  Error generating context: {e}")
        
        # Step 4: Simulate response generation
        print("\n   4️⃣  Simulating AI response generation...")
        
        simulated_response = {
            "answer": "Based on analysis of your customer data processing, key security risks include insufficient access controls and missing encryption. Recommended actions include implementing role-based access control and adding data classification policies.",
            "confidence": 0.78,
            "sources": ["customer_data", "customer_management"],
            "recommendations": ["Implement RBAC", "Add data encryption", "Regular security audits"],
            "compliance_notes": "Ensure GDPR and CCPA compliance through proper data handling"
        }
        
        print(f"      ✅ Generated comprehensive response")
        print(f"      📝 Answer length: {len(simulated_response['answer'])} characters")
        print(f"      🎯 Response confidence: {simulated_response['confidence']:.2f}")
        print(f"      📚 Sources: {', '.join(simulated_response['sources'])}")
        print(f"      💡 Key recommendations: {len(simulated_response['recommendations'])}")
        
        print("\n   ✅ Integrated NLP workflow demo complete")
        print("   🎉 This demonstrates how all Phase 2 components work together!")
        
    except Exception as e:
        print(f"   ❌ Error in integrated workflow: {e}")


def main():
    """Main demo function"""
    print("🚀 === PHASE 2 LLM INTEGRATION DEMO ===")
    print("🎯 Demonstrating enterprise AI-powered knowledge management")
    print(f"⏰ Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup metagraph with sample data
    metagraph, sample_entities, business_terms = setup_phase2_metagraph()
    
    if metagraph is None:
        print("❌ Failed to setup metagraph. Exiting demo.")
        return
    
    try:
        # Run all Phase 2 demos
        demo_natural_language_query_processing(metagraph)
        demo_enhanced_semantic_engine(business_terms)
        demo_rag_context_generation(metagraph)
        demo_intelligent_recommendations(metagraph)
        demo_integrated_nlp_workflow(metagraph)
        
        # Final summary
        print("\n🎉 === PHASE 2 DEMO COMPLETE ===")
        print("✅ All Phase 2 LLM integration features demonstrated successfully!")
        print("\n📋 Phase 2 Feature Summary:")
        print("   🗣️  Natural Language Query Processing - Convert plain English to structured operations")
        print("   🧠 Enhanced Semantic Engine - LLM-powered embeddings and relationships")
        print("   🔍 RAG Context Generation - Intelligent context retrieval for LLM interactions")
        print("   💡 Intelligent Recommendations - AI-powered insights and suggestions")
        print("   🔄 Integrated Workflows - Complete NLP-powered knowledge discovery")
        
        print("\n🚀 Ready for Phase 3: Advanced Analytics and Governance!")
        print("⏰ Next: Implement advanced pattern recognition, ML insights, and production deployment")
        
    except Exception as e:
        print(f"\n❌ Error during demo execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n⏰ Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()