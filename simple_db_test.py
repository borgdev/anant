#!/usr/bin/env python3
"""
Simple Database Configuration Test
=================================

Direct test of database configuration without full package imports.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Direct imports without going through package __init__
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field

from typing import Optional, Dict, Any


class DatabaseConfig(BaseSettings):
    """Database configuration for production deployment"""
    
    # Environment
    environment: str = Field(default="production", env="ANANT_ENV")
    debug: bool = Field(default=False, env="ANANT_DEBUG")
    
    # PostgreSQL Configuration
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="anant_meta", env="POSTGRES_DB")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # ChromaDB Configuration
    chroma_host: str = Field(default="localhost", env="CHROMA_HOST")
    chroma_port: int = Field(default=8000, env="CHROMA_PORT")
    chroma_persist_directory: str = Field(default="./chroma_data", env="CHROMA_PERSIST_DIR")
    
    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL"""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


def test_database_config():
    """Test database configuration"""
    print("🔧 Testing Database Configuration...")
    
    try:
        config = DatabaseConfig()
        
        print(f"📊 Database Configuration:")
        print(f"  Environment: {config.environment}")
        print(f"  PostgreSQL: {config.postgres_host}:{config.postgres_port}/{config.postgres_db}")
        print(f"  PostgreSQL URL: {config.postgres_url}")
        print(f"  Redis: {config.redis_host}:{config.redis_port}/{config.redis_db}")
        print(f"  Redis URL: {config.redis_url}")
        print(f"  ChromaDB: {config.chroma_host}:{config.chroma_port}")
        print(f"  ChromaDB Data: {config.chroma_persist_directory}")
        
        print("\n✅ Database configuration loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


async def test_postgres_connection():
    """Test PostgreSQL connection"""
    print("\n🐘 Testing PostgreSQL connection...")
    
    try:
        import asyncpg
        
        config = DatabaseConfig()
        
        conn = await asyncpg.connect(
            user=config.postgres_user,
            password=config.postgres_password,
            database=config.postgres_db,
            host=config.postgres_host,
            port=config.postgres_port
        )
        
        # Test basic query
        version = await conn.fetchval('SELECT version()')
        print(f"✅ PostgreSQL connected! Version: {version[:50]}...")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False


async def test_redis_connection():
    """Test Redis connection"""
    print("\n🔴 Testing Redis connection...")
    
    try:
        import redis.asyncio as redis
        
        config = DatabaseConfig()
        
        client = redis.from_url(config.redis_url)
        
        # Test basic operations
        await client.ping()
        await client.set("test_key", "test_value")
        value = await client.get("test_key")
        await client.delete("test_key")
        
        print(f"✅ Redis connected! Test value: {value.decode() if value else 'None'}")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}")
        return False


def test_chromadb_connection():
    """Test ChromaDB connection"""
    print("\n🌈 Testing ChromaDB connection...")
    
    try:
        import chromadb
        
        config = DatabaseConfig()
        
        # Create client
        client = chromadb.PersistentClient(path=config.chroma_persist_directory)
        
        # Test basic operations
        collection = client.get_or_create_collection("test_collection")
        collection.add(
            documents=["Test document"],
            ids=["test_id"],
            metadatas=[{"test": "metadata"}]
        )
        
        results = collection.query(
            query_texts=["Test"],
            n_results=1
        )
        
        # Cleanup
        client.delete_collection("test_collection")
        
        print(f"✅ ChromaDB connected! Test results: {len(results['documents'][0])} documents")
        return True
        
    except Exception as e:
        print(f"⚠️ ChromaDB connection failed: {e}")
        return False


async def main():
    """Main test function"""
    print("🚀 Anant Integration - Simple Database Test")
    print("=" * 50)
    
    success_count = 0
    total_tests = 4
    
    # Test configuration
    if test_database_config():
        success_count += 1
    
    # Test PostgreSQL
    if await test_postgres_connection():
        success_count += 1
    
    # Test Redis  
    if await test_redis_connection():
        success_count += 1
    
    # Test ChromaDB
    if test_chromadb_connection():
        success_count += 1
    
    print(f"\n📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ All database tests passed!")
        return True
    else:
        print("⚠️ Some database tests failed - but that's okay for development")
        return False


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)