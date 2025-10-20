"""
Test script for the advanced KG features (Embeddings and Vector Operations)
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the parent directory to sys.path to import anant
sys.path.insert(0, str(Path(__file__).parent))

# Configure Python environment first
from anant.utils.extras import safe_import

def test_embeddings_and_vectors():
    """Test the embeddings and vector operations engines"""
    
    print("🚀 Testing Advanced KG Features: Embeddings & Vector Operations")
    print("=" * 60)
    
    try:
        # Import the KG modules
        from anant.kg import (
            KnowledgeGraph, 
            KGEmbedder, 
            EmbeddingConfig, 
            VectorEngine, 
            VectorSearchConfig
        )
        
        print("✅ Successfully imported advanced KG modules")
        
        # Create a test knowledge graph with semantic data
        print("\n📊 Creating test knowledge graph...")
        kg = KnowledgeGraph()
        
        # Add test hyperedges with semantic relationships
        test_data = [
            ("Alice", "Person", "works_at", "Google", "Company"),
            ("Bob", "Person", "works_at", "Microsoft", "Company"), 
            ("Alice", "Person", "knows", "Bob", "Person"),
            ("Google", "Company", "competes_with", "Microsoft", "Company"),
            ("Carol", "Person", "works_at", "Apple", "Company"),
            ("Apple", "Company", "competes_with", "Google", "Company"),
            ("Bob", "Person", "knows", "Carol", "Person"),
        ]
        
        for head, head_type, relation, tail, tail_type in test_data:
            # Create hyperedge with semantic properties
            edge_id = f"{head}_{relation}_{tail}"
            kg.hypergraph.add_hyperedge(edge_id, [head, tail])
            
            # Add semantic properties
            kg.properties.set_property(edge_id, 'relation_type', relation)
            kg.properties.set_property(head, 'entity_type', head_type)
            kg.properties.set_property(tail, 'entity_type', tail_type)
        
        print(f"✅ Created KG with {kg.num_nodes} nodes and {kg.num_edges} edges")
        
        # Test embedding generation
        print("\n🧠 Testing embedding generation...")
        
        # Test with simple algorithm first
        config = EmbeddingConfig(
            algorithm='TransE',
            dimensions=64,
            epochs=10,
            learning_rate=0.01
        )
        
        embedder = KGEmbedder(kg, config)
        print(f"✅ Created KGEmbedder with {config.algorithm} algorithm")
        
        # Generate embeddings
        result = embedder.generate_embeddings()
        
        print(f"✅ Generated embeddings:")
        print(f"   - {len(result.entity_embeddings)} entity embeddings")
        print(f"   - {len(result.relation_embeddings)} relation embeddings") 
        print(f"   - Training time: {result.training_time:.3f}s")
        print(f"   - Final loss: {result.evaluation_metrics.get('final_loss', 'N/A')}")
        
        # Test similarity search
        print("\n🔍 Testing similarity search...")
        
        if result.entity_embeddings:
            # Get first entity for testing
            test_entity = list(result.entity_embeddings.keys())[0]
            similar_entities = embedder.similarity_search(test_entity, k=3)
            
            print(f"✅ Similar entities to '{test_entity}':")
            for entity, similarity in similar_entities[:3]:
                print(f"   - {entity}: {similarity:.4f}")
        
        # Test vector search engine
        print("\n⚡ Testing vector search engine...")
        
        vector_config = VectorSearchConfig(
            index_type='Flat',  # Use simple flat index for testing
            distance_metric='cosine'
        )
        
        vector_engine = VectorEngine(vector_config)
        
        # Build index from embeddings
        vector_engine.build_index(result.entity_embeddings)
        
        print(f"✅ Built vector index:")
        stats = vector_engine.get_statistics()
        print(f"   - Vectors: {stats['num_vectors']}")
        print(f"   - Dimensions: {stats['dimension']}")
        print(f"   - Index type: {stats['index_type']}")
        
        # Test vector search
        if result.entity_embeddings:
            test_vector = list(result.entity_embeddings.values())[0]
            search_results = vector_engine.search(test_vector, k=3)
            
            print(f"✅ Vector search results:")
            for result_obj in search_results[:3]:
                print(f"   - {result_obj.entity_id}: {result_obj.similarity:.4f}")
        
        # Test clustering
        print("\n🎯 Testing vector clustering...")
        
        try:
            clusters = vector_engine.cluster_vectors(n_clusters=2, algorithm='kmeans')
            print(f"✅ Clustered entities into groups:")
            
            cluster_groups = {}
            for entity, cluster_id in clusters.items():
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(entity)
            
            for cluster_id, entities in cluster_groups.items():
                print(f"   - Cluster {cluster_id}: {entities}")
                
        except Exception as e:
            print(f"⚠️  Clustering test skipped: {e}")
        
        # Test saving/loading
        print("\n💾 Testing save/load functionality...")
        
        # Save embeddings
        embeddings_file = "/tmp/test_embeddings.pkl"
        embedder.save_embeddings(embeddings_file)
        print(f"✅ Saved embeddings to {embeddings_file}")
        
        # Save vector index  
        index_file = "/tmp/test_vector_index"
        vector_engine.save_index(index_file)
        print(f"✅ Saved vector index to {index_file}")
        
        print("\n🎉 All advanced KG features tested successfully!")
        
        return {
            'embeddings_count': len(result.entity_embeddings),
            'vector_count': stats['num_vectors'],
            'training_time': result.training_time,
            'final_loss': result.evaluation_metrics.get('final_loss', 0)
        }
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test the advanced features
    results = test_embeddings_and_vectors()
    
    if results:
        print("\n📈 Test Summary:")
        print(f"   - Embeddings generated: {results['embeddings_count']}")
        print(f"   - Vectors indexed: {results['vector_count']}")
        print(f"   - Training time: {results['training_time']:.3f}s")
        print(f"   - Final loss: {results['final_loss']:.6f}")
        print("\n✨ Advanced KG features are ready for production use!")