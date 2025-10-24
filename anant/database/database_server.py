#!/usr/bin/env python3
"""
Anant Integration Database Server
================================

FastAPI server providing REST API access to the production database infrastructure.
Runs on port 8079 by default.
"""

import sys
import asyncio
import uvicorn
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Mock the logger for standalone operation
class SimpleLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")

sys.modules['anant_integration.core'] = type('MockModule', (), {'logger': SimpleLogger()})()

# Database imports
from anant_integration.database.config import DatabaseConfig
from anant_integration.database.connection import DatabaseManager

# Create FastAPI app
app = FastAPI(
    title="Anant Integration Database API",
    description="Production database infrastructure for the Anant Integration Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global database manager
db_manager = None


# Pydantic models for API
class UserCreate(BaseModel):
    username: str
    email: str
    full_name: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    created_at: datetime

class ConceptCreate(BaseModel):
    name: str
    namespace: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    properties: Dict[str, Any] = {}

class ConceptResponse(BaseModel):
    id: str
    user_id: str
    name: str
    namespace: Optional[str]
    type: Optional[str]
    description: Optional[str]
    properties: Dict[str, Any]
    created_at: datetime

class GraphCreate(BaseModel):
    name: str
    description: Optional[str] = None
    graph_type: str = "hypergraph"
    metadata: Dict[str, Any] = {}

class GraphResponse(BaseModel):
    id: str
    user_id: str
    name: str
    description: Optional[str]
    graph_type: str
    status: str
    version: int
    is_public: bool
    metadata: Dict[str, Any]
    created_at: datetime

class SystemStatus(BaseModel):
    status: str
    timestamp: datetime
    database: Dict[str, Any]
    cache: Dict[str, Any]
    vector_store: Dict[str, Any]


# Dependency to get database manager
async def get_db_manager() -> DatabaseManager:
    global db_manager
    if db_manager is None:
        config = DatabaseConfig()
        db_manager = DatabaseManager(config)
    return db_manager


# Health check endpoints
@app.get("/health", response_model=SystemStatus)
async def health_check(db: DatabaseManager = Depends(get_db_manager)):
    """System health check"""
    
    # Test database connections
    postgres_healthy = await db.health_check_postgres()
    redis_healthy = await db.health_check_redis()
    chroma_healthy = await db.health_check_chroma()
    
    return SystemStatus(
        status="healthy" if all([postgres_healthy, redis_healthy, chroma_healthy]) else "degraded",
        timestamp=datetime.utcnow(),
        database={
            "type": "PostgreSQL",
            "status": "healthy" if postgres_healthy else "error",
            "url": "localhost:5432/anant_meta"
        },
        cache={
            "type": "Redis", 
            "status": "healthy" if redis_healthy else "error",
            "url": "localhost:6379/0"
        },
        vector_store={
            "type": "ChromaDB",
            "status": "healthy" if chroma_healthy else "error",
            "url": "localhost:8000"
        }
    )


@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "Anant Integration Database API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# Database info endpoints
@app.get("/database/info")
async def database_info(db: DatabaseManager = Depends(get_db_manager)):
    """Get database information"""
    
    try:
        async with await db.get_postgres_session() as session:
            from sqlalchemy import text
            
            # Get database stats
            result = await session.execute(text("""
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes
                FROM pg_stat_user_tables 
                ORDER BY tablename
            """))
            
            table_stats = []
            for row in result:
                table_stats.append({
                    "schema": row[0],
                    "table": row[1],
                    "inserts": row[2],
                    "updates": row[3],
                    "deletes": row[4]
                })
            
            # Get database size
            size_result = await session.execute(text("""
                SELECT pg_size_pretty(pg_database_size('anant_meta')) as size
            """))
            db_size = size_result.scalar()
            
            return {
                "database": "anant_meta",
                "size": db_size,
                "tables": len(table_stats),
                "table_stats": table_stats
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database info error: {str(e)}")


# User management endpoints
@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate, db: DatabaseManager = Depends(get_db_manager)):
    """Create a new user"""
    
    try:
        async with await db.get_postgres_session() as session:
            from sqlalchemy import text
            
            user_id = uuid4()
            await session.execute(text("""
                INSERT INTO users (id, username, email, password_hash, full_name)
                VALUES (:id, :username, :email, :password, :full_name)
            """), {
                "id": user_id,
                "username": user.username,
                "email": user.email,
                "password": f"hashed_{user.password}",  # In production, use proper hashing
                "full_name": user.full_name
            })
            await session.commit()
            
            # Return created user
            result = await session.execute(text("""
                SELECT id, username, email, full_name, is_active, is_admin, created_at
                FROM users WHERE id = :id
            """), {"id": user_id})
            
            row = result.fetchone()
            return UserResponse(
                id=str(row[0]),
                username=row[1],
                email=row[2],
                full_name=row[3],
                is_active=row[4],
                is_admin=row[5],
                created_at=row[6]
            )
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"User creation failed: {str(e)}")


@app.get("/users/", response_model=List[UserResponse])
async def list_users(limit: int = 10, offset: int = 0, db: DatabaseManager = Depends(get_db_manager)):
    """List users with pagination"""
    
    try:
        async with await db.get_postgres_session() as session:
            from sqlalchemy import text
            
            result = await session.execute(text("""
                SELECT id, username, email, full_name, is_active, is_admin, created_at
                FROM users 
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """), {"limit": limit, "offset": offset})
            
            users = []
            for row in result:
                users.append(UserResponse(
                    id=str(row[0]),
                    username=row[1],
                    email=row[2], 
                    full_name=row[3],
                    is_active=row[4],
                    is_admin=row[5],
                    created_at=row[6]
                ))
            
            return users
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User listing failed: {str(e)}")


@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, db: DatabaseManager = Depends(get_db_manager)):
    """Get user by ID"""
    
    try:
        async with await db.get_postgres_session() as session:
            from sqlalchemy import text
            
            result = await session.execute(text("""
                SELECT id, username, email, full_name, is_active, is_admin, created_at
                FROM users WHERE id = :id
            """), {"id": user_id})
            
            row = result.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="User not found")
            
            return UserResponse(
                id=str(row[0]),
                username=row[1],
                email=row[2],
                full_name=row[3],
                is_active=row[4],
                is_admin=row[5],
                created_at=row[6]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User retrieval failed: {str(e)}")


# Concept management endpoints
@app.post("/concepts/", response_model=ConceptResponse)
async def create_concept(concept: ConceptCreate, user_id: str, db: DatabaseManager = Depends(get_db_manager)):
    """Create a new concept"""
    
    try:
        async with await db.get_postgres_session() as session:
            from sqlalchemy import text
            import json
            
            concept_id = uuid4()
            await session.execute(text("""
                INSERT INTO concepts (id, user_id, name, namespace, type, description, properties)
                VALUES (:id, :user_id, :name, :namespace, :type, :description, :properties)
            """), {
                "id": concept_id,
                "user_id": user_id,
                "name": concept.name,
                "namespace": concept.namespace,
                "type": concept.type,
                "description": concept.description,
                "properties": json.dumps(concept.properties)
            })
            await session.commit()
            
            # Return created concept
            result = await session.execute(text("""
                SELECT id, user_id, name, namespace, type, description, properties, created_at
                FROM concepts WHERE id = :id
            """), {"id": concept_id})
            
            row = result.fetchone()
            return ConceptResponse(
                id=str(row[0]),
                user_id=str(row[1]),
                name=row[2],
                namespace=row[3],
                type=row[4],
                description=row[5],
                properties=row[6] if row[6] else {},
                created_at=row[7]
            )
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Concept creation failed: {str(e)}")


@app.get("/concepts/", response_model=List[ConceptResponse])
async def list_concepts(limit: int = 10, offset: int = 0, db: DatabaseManager = Depends(get_db_manager)):
    """List concepts with pagination"""
    
    try:
        async with await db.get_postgres_session() as session:
            from sqlalchemy import text
            
            result = await session.execute(text("""
                SELECT id, user_id, name, namespace, type, description, properties, created_at
                FROM concepts 
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """), {"limit": limit, "offset": offset})
            
            concepts = []
            for row in result:
                concepts.append(ConceptResponse(
                    id=str(row[0]),
                    user_id=str(row[1]),
                    name=row[2],
                    namespace=row[3],
                    type=row[4],
                    description=row[5],
                    properties=row[6] if row[6] else {},
                    created_at=row[7]
                ))
            
            return concepts
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Concept listing failed: {str(e)}")


# Cache operations
@app.get("/cache/{key}")
async def get_cache(key: str, db: DatabaseManager = Depends(get_db_manager)):
    """Get value from cache"""
    
    try:
        redis_client = await db.get_redis_client()
        value = await redis_client.get(f"anant:{key}")
        
        if value:
            return {"key": key, "value": value.decode(), "found": True}
        else:
            return {"key": key, "value": None, "found": False}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache retrieval failed: {str(e)}")


@app.post("/cache/{key}")
async def set_cache(key: str, value: str, ttl: int = 3600, db: DatabaseManager = Depends(get_db_manager)):
    """Set value in cache"""
    
    try:
        redis_client = await db.get_redis_client()
        await redis_client.setex(f"anant:{key}", ttl, value)
        
        return {"key": key, "value": value, "ttl": ttl, "success": True}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache setting failed: {str(e)}")


# Vector operations
@app.post("/vectors/search")
async def search_vectors(query: str, collection: str = "default", limit: int = 5, db: DatabaseManager = Depends(get_db_manager)):
    """Search vectors in ChromaDB"""
    
    try:
        chroma_client = await db.get_chroma_client()
        
        # Get or create collection
        coll = chroma_client.get_or_create_collection(collection)
        
        # Search
        results = coll.query(
            query_texts=[query],
            n_results=limit
        )
        
        return {
            "query": query,
            "collection": collection,
            "results": {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "ids": results["ids"][0] if results["ids"] else []
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")


@app.post("/vectors/add")
async def add_vectors(
    documents: List[str], 
    collection: str = "default", 
    metadatas: Optional[List[Dict[str, Any]]] = None,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Add documents to vector collection"""
    
    try:
        chroma_client = await db.get_chroma_client()
        
        # Get or create collection
        coll = chroma_client.get_or_create_collection(collection)
        
        # Generate IDs
        ids = [str(uuid4()) for _ in documents]
        
        # Add documents
        coll.add(
            documents=documents,
            metadatas=metadatas or [{"added_at": str(datetime.utcnow())} for _ in documents],
            ids=ids
        )
        
        return {
            "collection": collection,
            "added": len(documents),
            "ids": ids,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector addition failed: {str(e)}")


# Server startup and shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize database connections on startup"""
    global db_manager
    
    print("🚀 Starting Anant Integration Database Server...")
    print("📊 Initializing database connections...")
    
    try:
        config = DatabaseConfig()
        db_manager = DatabaseManager(config)
        
        # Test connections
        postgres_healthy = await db_manager.health_check_postgres()
        redis_healthy = await db_manager.health_check_redis()
        chroma_healthy = await db_manager.health_check_chroma()
        
        print(f"  ✅ PostgreSQL: {'Connected' if postgres_healthy else 'Failed'}")
        print(f"  ✅ Redis: {'Connected' if redis_healthy else 'Failed'}")
        print(f"  ✅ ChromaDB: {'Connected' if chroma_healthy else 'Failed'}")
        
        if postgres_healthy:
            print("🎉 Database server ready!")
        else:
            print("⚠️ Database server started with degraded functionality")
            
    except Exception as e:
        print(f"❌ Startup failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections on shutdown"""
    global db_manager
    
    print("🔒 Shutting down database connections...")
    
    if db_manager:
        await db_manager.close_all()
        print("✅ Database connections closed")


if __name__ == "__main__":
    print("🚀 Anant Integration Database Server")
    print("=" * 40)
    
    # Server configuration
    HOST = "0.0.0.0"
    PORT = 8079
    
    print(f"🌐 Starting server on http://{HOST}:{PORT}")
    print(f"📖 API Documentation: http://localhost:{PORT}/docs")
    print(f"🔍 Health Check: http://localhost:{PORT}/health")
    
    # Run server
    uvicorn.run(
        "database_server:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info"
    )