#!/usr/bin/env python3
"""
Database Test Script
===================

Simple script to test database connectivity without full integration platform initialization.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Simple logger for testing
class SimpleLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")

# Mock the logger import
sys.modules['anant_integration.core'] = type('MockModule', (), {'logger': SimpleLogger()})()

from anant_integration.database.config import DatabaseConfig
from anant_integration.database.connection import DatabaseManager


async def test_database_connections():
    """Test database connections"""
    print("🔧 Testing Database Connections...")
    
    # Create configuration
    config = DatabaseConfig()
    
    print(f"📊 Database Configuration:")
    print(f"  PostgreSQL: {config.postgres_host}:{config.postgres_port}/{config.postgres_db}")
    print(f"  Redis: {config.redis_host}:{config.redis_port}/{config.redis_db}")
    print(f"  ChromaDB: {config.chroma_host}:{config.chroma_port}")
    
    # Create database manager
    db_manager = DatabaseManager(config)
    
    # Test PostgreSQL
    print("\n🐘 Testing PostgreSQL connection...")
    try:
        postgres_healthy = await db_manager.health_check_postgres()
        if postgres_healthy:
            print("✅ PostgreSQL connection successful")
        else:
            print("❌ PostgreSQL connection failed")
    except Exception as e:
        print(f"❌ PostgreSQL error: {e}")
    
    # Test Redis
    print("\n🔴 Testing Redis connection...")
    try:
        redis_healthy = await db_manager.health_check_redis()
        if redis_healthy:
            print("✅ Redis connection successful")
        else:
            print("❌ Redis connection failed")
    except Exception as e:
        print(f"⚠️ Redis error: {e}")
    
    # Test ChromaDB
    print("\n🌈 Testing ChromaDB connection...")
    try:
        chroma_healthy = await db_manager.health_check_chroma()
        if chroma_healthy:
            print("✅ ChromaDB connection successful")
        else:
            print("❌ ChromaDB connection failed")
    except Exception as e:
        print(f"⚠️ ChromaDB error: {e}")
    
    # Close connections
    await db_manager.close_all()
    print("\n🔒 Database connections closed")


if __name__ == "__main__":
    print("🚀 Anant Integration - Database Connection Test")
    print("=" * 50)
    
    try:
        asyncio.run(test_database_connections())
        print("\n✅ Database test completed!")
    except Exception as e:
        print(f"\n❌ Database test failed: {e}")
        sys.exit(1)