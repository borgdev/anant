#!/usr/bin/env python3
"""
Anant Graph Registry API Server
===============================

Lightweight FastAPI server for managing the Anant graph registry.
Provides REST API for graph catalog operations while delegating 
actual graph data operations to Anant's native Parquet storage.

Architecture:
- PostgreSQL: Registry metadata, user management, access control
- Parquet Files: Actual graph data via Anant's native storage
- Redis: Caching, session management
- Ray: Distributed computing for graph operations
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from uuid import UUID, uuid4

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# FastAPI and dependencies
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Database and caching
import asyncpg
import redis.asyncio as redis
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

# Anant imports
try:
    from anant.hypergraph.core import Hypergraph
    from anant.metagraph.core.metagraph import Metagraph
    from anant.knowledge_graph.core import KnowledgeGraph
    from anant.io.parquet_io import ParquetIO
    import polars as pl
    import ray
except ImportError as e:
    print(f"⚠️  Anant modules not fully available: {e}")
    print("🔄 Running in registry-only mode")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class RegistryConfig:
    postgres_url = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/anant_registry")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    ray_address = os.getenv("RAY_ADDRESS", "ray://localhost:10001")
    parquet_base_path = os.getenv("PARQUET_BASE_PATH", "./parquet_data")
    jwt_secret = os.getenv("JWT_SECRET", "anant_registry_secret_key")
    debug = os.getenv("DEBUG", "false").lower() == "true"

config = RegistryConfig()

# Pydantic models
class GraphRegistryEntry(BaseModel):
    id: Optional[UUID] = None
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    graph_type: str = "hypergraph"
    owner_id: UUID
    status: str = "active"
    version: str = "1.0.0"
    is_public: bool = False
    storage_path: str
    metadata_path: Optional[str] = None
    tags: List[str] = []
    categories: List[str] = []
    properties: Dict[str, Any] = {}

class GraphStats(BaseModel):
    node_count: int = 0
    edge_count: int = 0
    hyperedge_count: int = 0
    file_size_bytes: int = 0
    last_computed_stats: Optional[datetime] = None

class GraphCreateRequest(BaseModel):
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    graph_type: str = "hypergraph"
    is_public: bool = False
    tags: List[str] = []
    properties: Dict[str, Any] = {}

class GraphUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None
    properties: Optional[Dict[str, Any]] = None

class GraphQueryRequest(BaseModel):
    operation: str  # read, analyze, compute, etc.
    parameters: Dict[str, Any] = {}

# FastAPI app initialization
app = FastAPI(
    title="Anant Graph Registry API",
    description="Lightweight graph registry for Anant's Parquet-native architecture",
    version="1.0.0",
    debug=config.debug
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global connections
pg_pool = None
redis_client = None
ray_initialized = False

# Security
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Extract user from JWT token (simplified for demo)"""
    # In production, implement proper JWT validation
    return {"id": "00000000-0000-0000-0000-000000000001", "username": "demo_user"}

# Database operations
class GraphRegistryDB:
    """Database operations for the graph registry"""
    
    @staticmethod
    async def create_graph(graph_data: GraphRegistryEntry) -> UUID:
        """Register a new graph in the catalog"""
        graph_id = uuid4()
        
        async with pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO graph_registry (
                    id, name, display_name, description, graph_type, owner_id,
                    status, version, is_public, storage_path, metadata_path,
                    tags, categories, properties
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """, 
            graph_id, graph_data.name, graph_data.display_name, graph_data.description,
            graph_data.graph_type, graph_data.owner_id, graph_data.status, graph_data.version,
            graph_data.is_public, graph_data.storage_path, graph_data.metadata_path,
            graph_data.tags, graph_data.categories, graph_data.properties)
        
        return graph_id
    
    @staticmethod
    async def get_graph(graph_id: UUID) -> Optional[Dict[str, Any]]:
        """Get graph registry entry"""
        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM graph_registry WHERE id = $1
            """, graph_id)
            
            return dict(row) if row else None
    
    @staticmethod
    async def list_graphs(owner_id: Optional[UUID] = None, 
                         graph_type: Optional[str] = None,
                         is_public: Optional[bool] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """List graphs with optional filters"""
        conditions = ["status = 'active'"]
        params = []
        param_count = 0
        
        if owner_id:
            param_count += 1
            conditions.append(f"owner_id = ${param_count}")
            params.append(owner_id)
            
        if graph_type:
            param_count += 1
            conditions.append(f"graph_type = ${param_count}")
            params.append(graph_type)
            
        if is_public is not None:
            param_count += 1
            conditions.append(f"is_public = ${param_count}")
            params.append(is_public)
        
        param_count += 1
        params.append(limit)
        
        query = f"""
            SELECT * FROM graph_registry 
            WHERE {' AND '.join(conditions)}
            ORDER BY updated_at DESC
            LIMIT ${param_count}
        """
        
        async with pg_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    @staticmethod
    async def update_graph_stats(graph_id: UUID, stats: GraphStats):
        """Update graph statistics"""
        async with pg_pool.acquire() as conn:
            await conn.execute("""
                UPDATE graph_registry SET
                    node_count = $1,
                    edge_count = $2,
                    hyperedge_count = $3,
                    file_size_bytes = $4,
                    last_computed_stats = $5,
                    updated_at = NOW()
                WHERE id = $6
            """, 
            stats.node_count, stats.edge_count, stats.hyperedge_count,
            stats.file_size_bytes, datetime.utcnow(), graph_id)
    
    @staticmethod
    async def record_query(graph_id: UUID, user_id: UUID, operation: str, 
                          execution_time_ms: float, success: bool):
        """Record query for analytics"""
        async with pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO query_history 
                (id, graph_id, user_id, operation_type, execution_time_ms, success, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
            """, uuid4(), graph_id, user_id, operation, execution_time_ms, success)

# Graph operations using Anant's native storage
class GraphOperations:
    """High-level graph operations using Anant's Parquet storage"""
    
    @staticmethod
    async def compute_graph_stats(storage_path: str) -> GraphStats:
        """Compute graph statistics from Parquet files"""
        try:
            # Use Anant's ParquetIO to read graph data
            parquet_io = ParquetIO()
            
            # Read basic statistics (this would use actual Anant methods)
            if Path(storage_path).exists():
                # Read with Polars for high performance
                df = pl.read_parquet(storage_path)
                
                # Compute basic stats (simplified example)
                stats = GraphStats(
                    node_count=len(df.filter(pl.col("type") == "node")) if "type" in df.columns else 0,
                    edge_count=len(df.filter(pl.col("type") == "edge")) if "type" in df.columns else 0,
                    file_size_bytes=Path(storage_path).stat().st_size,
                    last_computed_stats=datetime.utcnow()
                )
            else:
                stats = GraphStats()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error computing graph stats: {e}")
            return GraphStats()
    
    @staticmethod 
    async def execute_graph_query(storage_path: str, operation: str, 
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute graph query using Ray distributed computing"""
        try:
            if ray_initialized:
                # Use Ray for distributed computation
                @ray.remote
                def distributed_graph_operation(path: str, op: str, params: Dict[str, Any]):
                    # This would use actual Anant graph operations
                    return {"status": "success", "operation": op, "result": "computed"}
                
                future = distributed_graph_operation.remote(storage_path, operation, parameters)
                result = await asyncio.get_event_loop().run_in_executor(None, ray.get, future)
            else:
                # Fallback to local computation
                result = {"status": "success", "operation": operation, "note": "local_computation"}
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing graph query: {e}")
            return {"status": "error", "message": str(e)}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize connections and services"""
    global pg_pool, redis_client, ray_initialized
    
    try:
        # Initialize PostgreSQL connection pool
        pg_pool = await asyncpg.create_pool(
            config.postgres_url,
            min_size=5,
            max_size=20
        )
        logger.info("✅ PostgreSQL connection pool initialized")
        
        # Initialize Redis
        redis_client = redis.from_url(config.redis_url)
        await redis_client.ping()
        logger.info("✅ Redis connection established")
        
        # Initialize Ray (optional for distributed computing)
        try:
            if not ray.is_initialized():
                ray.init(address=config.ray_address, ignore_reinit_error=True)
            ray_initialized = True
            logger.info("✅ Ray cluster connection established")
        except Exception as e:
            logger.warning(f"⚠️  Ray initialization failed: {e}")
            ray_initialized = False
        
        # Ensure Parquet storage directory exists
        Path(config.parquet_base_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Parquet storage ready: {config.parquet_base_path}")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections"""
    global pg_pool, redis_client, ray_initialized
    
    if pg_pool:
        await pg_pool.close()
    
    if redis_client:
        await redis_client.close()
    
    if ray_initialized:
        ray.shutdown()

# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "postgres": pg_pool is not None,
            "redis": redis_client is not None,
            "ray": ray_initialized
        }
    }

@app.post("/graphs", response_model=Dict[str, Any])
async def create_graph(
    request: GraphCreateRequest,
    user = Depends(get_current_user)
):
    """Create a new graph registration"""
    try:
        # Generate storage path
        storage_path = f"{config.parquet_base_path}/{request.name}_{uuid4().hex[:8]}.parquet"
        
        # Create registry entry
        graph_data = GraphRegistryEntry(
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            graph_type=request.graph_type,
            owner_id=UUID(user["id"]),
            is_public=request.is_public,
            storage_path=storage_path,
            tags=request.tags,
            properties=request.properties
        )
        
        graph_id = await GraphRegistryDB.create_graph(graph_data)
        
        # Initialize empty Parquet file (using Anant's approach)
        Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
        
        return {
            "graph_id": str(graph_id),
            "name": request.name,
            "storage_path": storage_path,
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"Error creating graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graphs", response_model=List[Dict[str, Any]])
async def list_graphs(
    graph_type: Optional[str] = None,
    is_public: Optional[bool] = None,
    limit: int = 100,
    user = Depends(get_current_user)
):
    """List graphs in the registry"""
    try:
        graphs = await GraphRegistryDB.list_graphs(
            owner_id=UUID(user["id"]) if not is_public else None,
            graph_type=graph_type,
            is_public=is_public,
            limit=limit
        )
        return graphs
        
    except Exception as e:
        logger.error(f"Error listing graphs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graphs/{graph_id}", response_model=Dict[str, Any])
async def get_graph(graph_id: UUID, user = Depends(get_current_user)):
    """Get graph registry information"""
    try:
        graph = await GraphRegistryDB.get_graph(graph_id)
        if not graph:
            raise HTTPException(status_code=404, detail="Graph not found")
        
        return graph
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graphs/{graph_id}/stats", response_model=GraphStats)
async def get_graph_stats(graph_id: UUID, user = Depends(get_current_user)):
    """Get or compute graph statistics"""
    try:
        graph = await GraphRegistryDB.get_graph(graph_id)
        if not graph:
            raise HTTPException(status_code=404, detail="Graph not found")
        
        # Compute fresh statistics from Parquet files
        stats = await GraphOperations.compute_graph_stats(graph["storage_path"])
        
        # Update registry with computed stats
        await GraphRegistryDB.update_graph_stats(graph_id, stats)
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graphs/{graph_id}/query", response_model=Dict[str, Any])
async def query_graph(
    graph_id: UUID,
    request: GraphQueryRequest,
    user = Depends(get_current_user)
):
    """Execute a query against the graph using distributed computing"""
    start_time = datetime.utcnow()
    
    try:
        graph = await GraphRegistryDB.get_graph(graph_id)
        if not graph:
            raise HTTPException(status_code=404, detail="Graph not found")
        
        # Execute query using Anant's native operations
        result = await GraphOperations.execute_graph_query(
            graph["storage_path"], 
            request.operation, 
            request.parameters
        )
        
        # Record query for analytics
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        await GraphRegistryDB.record_query(
            graph_id, UUID(user["id"]), request.operation, 
            execution_time, result.get("status") == "success"
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/registry/stats", response_model=Dict[str, Any])
async def get_registry_stats(user = Depends(get_current_user)):
    """Get overall registry statistics"""
    try:
        async with pg_pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_graphs,
                    COUNT(*) FILTER (WHERE is_public = true) as public_graphs,
                    COUNT(DISTINCT owner_id) as total_users,
                    COUNT(DISTINCT graph_type) as graph_types,
                    SUM(node_count) as total_nodes,
                    SUM(edge_count) as total_edges,
                    SUM(file_size_bytes) as total_storage_bytes
                FROM graph_registry 
                WHERE status = 'active'
            """)
        
        return dict(stats)
        
    except Exception as e:
        logger.error(f"Error getting registry stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "registry_server:app",
        host="0.0.0.0",
        port=8080,
        reload=config.debug,
        log_level="info"
    )