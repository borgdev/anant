"""
Database service
Manages PostgreSQL database connections and sessions
"""

import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from ..config import settings


class Base(DeclarativeBase):
    """Base class for database models"""
    pass


class DatabaseService:
    """Service for managing database connections"""
    
    def __init__(self):
        self.engine = None
        self.async_session = None
        self.connected = False
        
    async def initialize(self) -> bool:
        """Initialize database connection"""
        try:
            print("🔧 Initializing database connection...")
            
            # Create async engine
            self.engine = create_async_engine(
                settings.DATABASE_URL,
                echo=settings.DEBUG,
                pool_pre_ping=True,
                pool_recycle=300
            )
            
            # Create session maker
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                assert result.scalar() == 1
            
            self.connected = True
            print("✅ Database connection established")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            self.connected = False
            return False
    
    async def shutdown(self):
        """Shutdown database connections"""
        try:
            if self.engine:
                await self.engine.dispose()
                print("🔌 Database connection closed")
        except Exception as e:
            print(f"⚠️ Error during database shutdown: {e}")
        finally:
            self.connected = False
    
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self.connected and self.engine is not None
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session"""
        if not self.is_connected():
            raise Exception("Database not connected")
        
        async with self.async_session() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def health_check(self) -> dict:
        """Perform database health check"""
        if not self.is_connected():
            return {
                "healthy": False,
                "status": "disconnected",
                "error": "Database not connected"
            }
        
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(
                    text("SELECT current_database(), current_user, version()")
                )
                db_info = result.fetchone()
                
                return {
                    "healthy": True,
                    "status": "connected",
                    "database": db_info[0] if db_info else "unknown",
                    "user": db_info[1] if db_info else "unknown", 
                    "version": db_info[2].split()[0] if db_info else "unknown"
                }
        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "error": str(e)
            }
    
    def get_status(self) -> dict:
        """Get database service status"""
        return {
            "connected": self.connected,
            "database_url": settings.DATABASE_URL.split("@")[1] if "@" in settings.DATABASE_URL else "unknown",
            "engine_pool_size": self.engine.pool.size() if self.engine and hasattr(self.engine.pool, 'size') else 0
        }


# Dependency to get database session
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    from ..core.lifecycle import app_lifecycle
    
    if not app_lifecycle.database_service.is_connected():
        raise Exception("Database service not available")
    
    async for session in app_lifecycle.database_service.get_session():
        yield session