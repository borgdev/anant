#!/usr/bin/env python3
"""
Standalone Database Server
==========================

FastAPI server for the database API without complex dependencies.
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
from pydantic import BaseModel

# Import database config and connection directly
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field

from sqlalchemy import create_engine, text
import asyncpg
import redis.asyncio as redis
import chromadb


# Database configuration
class DatabaseConfig(BaseSettings):
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT") 
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="anant_meta", env="POSTGRES_DB")
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Additional configuration for server
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8079, env="API_PORT")
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields in .env file
    
    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def postgres_async_url(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


# Simple database manager
class SimpleDatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._redis_client = None
        self._chroma_client = None
    
    async def get_postgres_connection(self):
        return await asyncpg.connect(
            user=self.config.postgres_user,
            password=self.config.postgres_password,
            database=self.config.postgres_db,
            host=self.config.postgres_host,
            port=self.config.postgres_port
        )
    
    async def get_redis_client(self):
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.config.redis_url)
        return self._redis_client
    
    def get_chroma_client(self):
        if self._chroma_client is None:
            self._chroma_client = chromadb.PersistentClient(path="./chroma_data")
        return self._chroma_client
    
    async def health_check_postgres(self) -> bool:
        try:
            conn = await self.get_postgres_connection()
            await conn.fetchval("SELECT 1")
            await conn.close()
            return True
        except:
            return False
    
    async def health_check_redis(self) -> bool:
        try:
            client = await self.get_redis_client()
            await client.ping()
            return True
        except:
            return False
    
    def health_check_chroma(self) -> bool:
        try:
            client = self.get_chroma_client()
            client.heartbeat()
            return True
        except:
            return False


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

# Global instances
config = DatabaseConfig()
db_manager = SimpleDatabaseManager(config)


# Pydantic models
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

class SystemStatus(BaseModel):
    status: str
    timestamp: datetime
    database: Dict[str, Any]
    cache: Dict[str, Any]
    vector_store: Dict[str, Any]


# API endpoints
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


@app.get("/health", response_model=SystemStatus)
async def health_check():
    """System health check"""
    
    postgres_healthy = await db_manager.health_check_postgres()
    redis_healthy = await db_manager.health_check_redis()
    chroma_healthy = db_manager.health_check_chroma()
    
    return SystemStatus(
        status="healthy" if all([postgres_healthy, redis_healthy, chroma_healthy]) else "degraded",
        timestamp=datetime.utcnow(),
        database={
            "type": "PostgreSQL",
            "status": "healthy" if postgres_healthy else "error",
            "url": f"{config.postgres_host}:{config.postgres_port}/{config.postgres_db}"
        },
        cache={
            "type": "Redis", 
            "status": "healthy" if redis_healthy else "error",
            "url": f"{config.redis_host}:{config.redis_port}/{config.redis_db}"
        },
        vector_store={
            "type": "ChromaDB",
            "status": "healthy" if chroma_healthy else "error",
            "path": "./chroma_data"
        }
    )


@app.get("/database/info")
async def database_info():
    """Get database information"""
    
    try:
        conn = await db_manager.get_postgres_connection()
        
        # Get database stats
        result = await conn.fetch("""
            SELECT 
                schemaname,
                tablename,
                COALESCE(n_tup_ins, 0) as inserts,
                COALESCE(n_tup_upd, 0) as updates,
                COALESCE(n_tup_del, 0) as deletes
            FROM pg_stat_user_tables 
            ORDER BY tablename
        """)
        
        table_stats = []
        for row in result:
            table_stats.append({
                "schema": row["schemaname"],
                "table": row["tablename"],
                "inserts": row["inserts"],
                "updates": row["updates"],
                "deletes": row["deletes"]
            })
        
        # Get database size
        db_size = await conn.fetchval("SELECT pg_size_pretty(pg_database_size($1))", config.postgres_db)
        
        await conn.close()
        
        return {
            "database": config.postgres_db,
            "size": db_size,
            "tables": len(table_stats),
            "table_stats": table_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database info error: {str(e)}")


@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate):
    """Create a new user"""
    
    try:
        conn = await db_manager.get_postgres_connection()
        
        user_id = uuid4()
        await conn.execute("""
            INSERT INTO users (id, username, email, password_hash, full_name)
            VALUES ($1, $2, $3, $4, $5)
        """, user_id, user.username, user.email, f"hashed_{user.password}", user.full_name)
        
        # Return created user
        row = await conn.fetchrow("""
            SELECT id, username, email, full_name, is_active, is_admin, created_at
            FROM users WHERE id = $1
        """, user_id)
        
        await conn.close()
        
        return UserResponse(
            id=str(row["id"]),
            username=row["username"],
            email=row["email"],
            full_name=row["full_name"],
            is_active=row["is_active"],
            is_admin=row["is_admin"],
            created_at=row["created_at"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"User creation failed: {str(e)}")


@app.get("/users/", response_model=List[UserResponse])
async def list_users(limit: int = 10, offset: int = 0):
    """List users with pagination"""
    
    try:
        conn = await db_manager.get_postgres_connection()
        
        result = await conn.fetch("""
            SELECT id, username, email, full_name, is_active, is_admin, created_at
            FROM users 
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
        """, limit, offset)
        
        users = []
        for row in result:
            users.append(UserResponse(
                id=str(row["id"]),
                username=row["username"],
                email=row["email"], 
                full_name=row["full_name"],
                is_active=row["is_active"],
                is_admin=row["is_admin"],
                created_at=row["created_at"]
            ))
        
        await conn.close()
        return users
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User listing failed: {str(e)}")


@app.get("/cache/{key}")
async def get_cache(key: str):
    """Get value from cache"""
    
    try:
        client = await db_manager.get_redis_client()
        value = await client.get(f"anant:{key}")
        
        if value:
            return {"key": key, "value": value.decode(), "found": True}
        else:
            return {"key": key, "value": None, "found": False}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache retrieval failed: {str(e)}")


@app.post("/cache/{key}")
async def set_cache(key: str, value: str, ttl: int = 3600):
    """Set value in cache"""
    
    try:
        client = await db_manager.get_redis_client()
        await client.setex(f"anant:{key}", ttl, value)
        
        return {"key": key, "value": value, "ttl": ttl, "success": True}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache setting failed: {str(e)}")


@app.post("/vectors/search")
async def search_vectors(query: str, collection: str = "default", limit: int = 5):
    """Search vectors in ChromaDB"""
    
    try:
        client = db_manager.get_chroma_client()
        
        # Get or create collection
        coll = client.get_or_create_collection(collection)
        
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
async def add_vectors(documents: List[str], collection: str = "default", metadatas: Optional[List[Dict[str, Any]]] = None):
    """Add documents to vector collection"""
    
    try:
        client = db_manager.get_chroma_client()
        
        # Get or create collection
        coll = client.get_or_create_collection(collection)
        
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


if __name__ == "__main__":
    print("🚀 Anant Integration Database Server")
    print("=" * 40)
    
    # Load configuration from .env file
    config = DatabaseConfig()
    
    print(f"🌐 Starting server on http://{config.api_host}:{config.api_port}")
    print(f"📖 API Documentation: http://localhost:{config.api_port}/docs")
    print(f"🔍 Health Check: http://localhost:{config.api_port}/health")
    print(f"🗄️  Database: {config.postgres_db}@{config.postgres_host}:{config.postgres_port}")
    print(f"🔴 Redis: {config.redis_host}:{config.redis_port}/{config.redis_db}")
    
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        reload=False,
        log_level="info"
    )