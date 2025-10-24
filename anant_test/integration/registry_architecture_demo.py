#!/usr/bin/env python3
"""
Anant Graph Registry Architecture Demo
=====================================

Demonstrates the new registry-focused architecture where:
- PostgreSQL serves as a lightweight graph registry/catalog
- Actual graph data is stored in Parquet files using Anant's native storage
- Ray provides distributed computing capabilities
- Redis handles caching and session management

This aligns with Anant's native Parquet+Polars architecture.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid4, UUID
import logging

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import polars as pl
    import asyncpg
    import redis.asyncio as redis
    from anant.hypergraph.core import Hypergraph
    from anant.metagraph.core.metagraph import Metagraph
    ANANT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Some dependencies not available: {e}")
    ANANT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphRegistryDemo:
    """Demonstrates the Graph Registry architecture"""
    
    def __init__(self):
        self.postgres_url = "postgresql://postgres:postgres@localhost:5432/anant_registry"
        self.redis_url = "redis://localhost:6379/0" 
        self.parquet_base_path = "./demo_parquet_data"
        
        # Ensure directories exist
        Path(self.parquet_base_path).mkdir(parents=True, exist_ok=True)
        
    async def setup_demo_environment(self):
        """Set up the demo environment"""
        print("🏗️ Setting up Anant Graph Registry Demo Environment...")
        
        try:
            # Test PostgreSQL connection
            conn = await asyncpg.connect(self.postgres_url)
            
            # Create a simple demo user
            user_id = uuid4()
            await conn.execute("""
                INSERT INTO users (id, username, email, password_hash, full_name)
                VALUES ($1, 'demo_user', 'demo@anant.ai', 'demo_hash', 'Demo User')
                ON CONFLICT (username) DO UPDATE SET updated_at = NOW()
            """, user_id)
            
            await conn.close()
            print("✅ PostgreSQL registry connection verified")
            
            # Test Redis connection
            redis_client = redis.from_url(self.redis_url)
            await redis_client.ping()
            await redis_client.set("demo:test", "registry_demo")
            await redis_client.close()
            print("✅ Redis caching connection verified")
            
            return user_id
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return None
    
    async def create_sample_graphs(self, user_id: UUID):
        """Create sample graphs using Anant's native Parquet storage"""
        print("\n📊 Creating sample graphs with Parquet storage...")
        
        graphs_created = []
        
        try:
            conn = await asyncpg.connect(self.postgres_url)
            
            # 1. Create Hypergraph with Parquet storage
            if ANANT_AVAILABLE:
                hypergraph = Hypergraph()
                
                # Add some nodes and hyperedges
                hypergraph.add_node("user_1", properties={"name": "Alice", "type": "person"})
                hypergraph.add_node("user_2", properties={"name": "Bob", "type": "person"})
                hypergraph.add_node("project_1", properties={"name": "AI Research", "type": "project"})
                hypergraph.add_hyperedge("collab_1", ["user_1", "user_2", "project_1"], 
                                       properties={"type": "collaboration"})
                
                # Save to Parquet using Anant's native approach
                hypergraph_path = f"{self.parquet_base_path}/demo_hypergraph.parquet"
                
                # Create DataFrame representation for Parquet storage
                nodes_data = []
                for node_id in hypergraph.nodes:
                    node_data = hypergraph.get_node_data(node_id) or {}
                    nodes_data.append({
                        "id": node_id,
                        "type": "node",
                        "properties": json.dumps(node_data)
                    })
                
                edges_data = []
                for edge_id in hypergraph.hyperedges:
                    edge_data = hypergraph.get_hyperedge_data(edge_id) or {}
                    nodes = list(hypergraph.get_hyperedge_nodes(edge_id))
                    edges_data.append({
                        "id": edge_id,
                        "type": "hyperedge", 
                        "nodes": json.dumps(nodes),
                        "properties": json.dumps(edge_data)
                    })
                
                # Combine and save as Parquet
                all_data = nodes_data + edges_data
                if all_data:
                    df = pl.DataFrame(all_data)
                    df.write_parquet(hypergraph_path)
                    print(f"  📁 Hypergraph saved to Parquet: {hypergraph_path}")
                
                # Register in PostgreSQL catalog
                graph_id = await self.register_graph_in_catalog(
                    conn, user_id, "demo_hypergraph", "Demo Hypergraph",
                    "Sample hypergraph demonstrating Parquet storage",
                    "hypergraph", hypergraph_path, len(nodes_data), len(edges_data)
                )
                graphs_created.append(("hypergraph", graph_id, hypergraph_path))
            
            # 2. Create Metagraph with Parquet storage
            if ANANT_AVAILABLE:
                metagraph = Metagraph()
                
                # Add entities and relationships
                metagraph.create_entity({
                    "id": "concept_1",
                    "type": "concept",
                    "name": "Machine Learning",
                    "description": "AI technique for pattern recognition"
                })
                
                metagraph.create_entity({
                    "id": "concept_2", 
                    "type": "concept",
                    "name": "Deep Learning",
                    "description": "Neural network-based ML approach"
                })
                
                # Create relationship
                metagraph.create_relationship("concept_1", "concept_2", "subsumes", {
                    "strength": 0.9,
                    "context": "academic"
                })
                
                # Save metagraph data to Parquet
                metagraph_path = f"{self.parquet_base_path}/demo_metagraph.parquet"
                
                # Export entities and relationships  
                entities = metagraph.get_all_entities()
                relationships = metagraph.get_all_relationships()
                
                metagraph_data = []
                for entity_id, entity_data in entities.items():
                    metagraph_data.append({
                        "id": entity_id,
                        "type": "entity",
                        "data": json.dumps(entity_data)
                    })
                
                for rel in relationships:
                    metagraph_data.append({
                        "id": f"rel_{rel['source']}_{rel['target']}",
                        "type": "relationship",
                        "source": rel["source"],
                        "target": rel["target"],
                        "relation_type": rel["relation_type"],
                        "data": json.dumps(rel.get("properties", {}))
                    })
                
                if metagraph_data:
                    df = pl.DataFrame(metagraph_data)
                    df.write_parquet(metagraph_path)
                    print(f"  📁 Metagraph saved to Parquet: {metagraph_path}")
                
                # Register in catalog
                graph_id = await self.register_graph_in_catalog(
                    conn, user_id, "demo_metagraph", "Demo Metagraph",
                    "Sample metagraph with entities and relationships", 
                    "metagraph", metagraph_path, len([d for d in metagraph_data if d["type"] == "entity"]),
                    len([d for d in metagraph_data if d["type"] == "relationship"])
                )
                graphs_created.append(("metagraph", graph_id, metagraph_path))
            
            # 3. Create a Knowledge Graph entry (even without full implementation)
            kg_path = f"{self.parquet_base_path}/demo_knowledge_graph.parquet"
            sample_kg_data = [
                {"id": "entity_1", "type": "node", "label": "Person", "properties": json.dumps({"name": "Alice"})},
                {"id": "entity_2", "type": "node", "label": "Organization", "properties": json.dumps({"name": "Anant AI"})},
                {"id": "rel_1", "type": "edge", "source": "entity_1", "target": "entity_2", "relation": "works_at"}
            ]
            
            df = pl.DataFrame(sample_kg_data)
            df.write_parquet(kg_path)
            print(f"  📁 Knowledge Graph saved to Parquet: {kg_path}")
            
            graph_id = await self.register_graph_in_catalog(
                conn, user_id, "demo_knowledge_graph", "Demo Knowledge Graph",
                "Sample knowledge graph with semantic relationships",
                "knowledge_graph", kg_path, 2, 1
            )
            graphs_created.append(("knowledge_graph", graph_id, kg_path))
            
            await conn.close()
            print(f"✅ Created {len(graphs_created)} graphs with Parquet storage")
            return graphs_created
            
        except Exception as e:
            print(f"❌ Graph creation failed: {e}")
            return graphs_created
    
    async def register_graph_in_catalog(self, conn, user_id: UUID, name: str, 
                                      display_name: str, description: str,
                                      graph_type: str, storage_path: str,
                                      node_count: int, edge_count: int) -> UUID:
        """Register a graph in the PostgreSQL catalog"""
        graph_id = uuid4()
        
        await conn.execute("""
            INSERT INTO graph_registry (
                id, name, display_name, description, graph_type, owner_id,
                storage_path, node_count, edge_count, file_size_bytes,
                tags, properties, last_computed_stats
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """, 
        graph_id, name, display_name, description, graph_type, user_id,
        storage_path, node_count, edge_count, Path(storage_path).stat().st_size if Path(storage_path).exists() else 0,
        ["demo", "sample"], {"format": "parquet", "engine": "polars"}, datetime.utcnow())
        
        return graph_id
    
    async def demonstrate_registry_queries(self):
        """Demonstrate querying the graph registry"""
        print("\n🔍 Demonstrating Graph Registry Queries...")
        
        try:
            conn = await asyncpg.connect(self.postgres_url)
            
            # 1. List all graphs
            graphs = await conn.fetch("""
                SELECT id, name, display_name, graph_type, node_count, edge_count, 
                       file_size_bytes, created_at, storage_path
                FROM graph_registry 
                WHERE status = 'active'
                ORDER BY created_at DESC
            """)
            
            print(f"\n📋 Registry contains {len(graphs)} graphs:")
            for graph in graphs:
                print(f"  🔸 {graph['name']} ({graph['graph_type']})")
                print(f"    - Nodes: {graph['node_count']}, Edges: {graph['edge_count']}")
                print(f"    - Size: {graph['file_size_bytes']:,} bytes")
                print(f"    - Storage: {graph['storage_path']}")
                print(f"    - Created: {graph['created_at']}")
            
            # 2. Query by graph type
            hypergraphs = await conn.fetch("""
                SELECT name, node_count, edge_count 
                FROM graph_registry 
                WHERE graph_type = 'hypergraph' AND status = 'active'
            """)
            
            print(f"\n📊 Found {len(hypergraphs)} hypergraphs:")
            for hg in hypergraphs:
                print(f"  🕸️  {hg['name']}: {hg['node_count']} nodes, {hg['edge_count']} edges")
            
            # 3. Storage statistics
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_graphs,
                    COUNT(DISTINCT graph_type) as graph_types,
                    SUM(node_count) as total_nodes,
                    SUM(edge_count) as total_edges,
                    SUM(file_size_bytes) as total_storage_bytes
                FROM graph_registry 
                WHERE status = 'active'
            """)
            
            print(f"\n📈 Registry Statistics:")
            print(f"  📊 Total Graphs: {stats['total_graphs']}")
            print(f"  🎯 Graph Types: {stats['graph_types']}")
            print(f"  🔗 Total Nodes: {stats['total_nodes']:,}")
            print(f"  ↔️  Total Edges: {stats['total_edges']:,}")
            print(f"  💾 Total Storage: {stats['total_storage_bytes']:,} bytes")
            
            await conn.close()
            
        except Exception as e:
            print(f"❌ Registry query failed: {e}")
    
    async def demonstrate_parquet_operations(self, graphs_created):
        """Demonstrate direct Parquet operations using Polars"""
        print("\n⚡ Demonstrating High-Performance Parquet Operations...")
        
        for graph_type, graph_id, storage_path in graphs_created:
            if not Path(storage_path).exists():
                continue
                
            print(f"\n📁 Analyzing {graph_type}: {storage_path}")
            
            try:
                # Read with Polars for high performance
                df = pl.read_parquet(storage_path)
                
                print(f"  📊 Shape: {df.shape}")
                print(f"  🏷️  Columns: {df.columns}")
                
                # Analyze by type
                if "type" in df.columns:
                    type_counts = df.group_by("type").len().sort("len", descending=True)
                    print(f"  📋 Type distribution:")
                    for row in type_counts.iter_rows(named=True):
                        print(f"    - {row['type']}: {row['len']}")
                
                # Show memory usage
                memory_mb = df.estimated_size("mb")
                print(f"  💾 Memory usage: {memory_mb:.2f} MB")
                
                # Demonstrate lazy evaluation (Polars strength)
                lazy_df = pl.scan_parquet(storage_path)
                result = (lazy_df
                         .filter(pl.col("type") == "node") 
                         .select(["id", "type"])
                         .collect())
                
                if len(result) > 0:
                    print(f"  ⚡ Lazy query result: {len(result)} nodes found")
                
            except Exception as e:
                print(f"    ❌ Parquet analysis failed: {e}")
    
    async def cleanup_demo_data(self):
        """Clean up demo data"""
        print("\n🧹 Cleaning up demo data...")
        
        try:
            # Remove Parquet files
            for parquet_file in Path(self.parquet_base_path).glob("*.parquet"):
                parquet_file.unlink()
            
            # Clean registry entries
            conn = await asyncpg.connect(self.postgres_url)
            await conn.execute("DELETE FROM graph_registry WHERE name LIKE 'demo_%'")
            await conn.close()
            
            print("✅ Demo cleanup completed")
            
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")

async def main():
    """Run the complete demo"""
    print("🚀 Anant Graph Registry Architecture Demo")
    print("=" * 60)
    print("🎯 Demonstrating PostgreSQL as lightweight graph registry")
    print("📁 With Anant's native Parquet+Polars storage architecture")
    print("=" * 60)
    
    demo = GraphRegistryDemo()
    
    # Setup environment
    user_id = await demo.setup_demo_environment()
    if not user_id:
        print("❌ Demo environment setup failed")
        return False
    
    # Create sample graphs
    graphs_created = await demo.create_sample_graphs(user_id)
    if not graphs_created:
        print("❌ Graph creation failed")
        return False
    
    # Demonstrate registry queries
    await demo.demonstrate_registry_queries()
    
    # Demonstrate Parquet operations
    await demo.demonstrate_parquet_operations(graphs_created)
    
    # Optional cleanup
    cleanup = input("\n🤔 Clean up demo data? (y/N): ").lower().strip()
    if cleanup == 'y':
        await demo.cleanup_demo_data()
    
    print("\n🎉 Anant Graph Registry Demo completed successfully!")
    print("\n📋 Key Architecture Benefits:")
    print("  ✅ PostgreSQL optimized for registry/metadata only") 
    print("  ✅ Parquet files provide high-performance graph storage")
    print("  ✅ Polars enables fast analytical operations")
    print("  ✅ Registry provides discovery, access control, and analytics")
    print("  ✅ Aligns perfectly with Anant's native storage approach")
    
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)