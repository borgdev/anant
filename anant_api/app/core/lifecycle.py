"""
Application lifecycle management
Handles startup, shutdown, and service initialization
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI

from ..services.ray_service import RayService  
from ..services.anant_service import AnantService
from ..services.database_service import DatabaseService
from ..config import settings


class AppLifecycle:
    """Manages application lifecycle events"""
    
    def __init__(self):
        self.ray_service = RayService()
        self.anant_service = AnantService()
        self.database_service = DatabaseService()
        self.startup_time = None
        
    async def startup(self) -> bool:
        """Initialize all services on startup"""
        print("🚀 Starting Anant Graph API with Ray cluster integration...")
        self.startup_time = time.time()
        
        # Initialize database connection first
        database_connected = await self.database_service.initialize()
        
        # Initialize Ray cluster connection
        ray_connected = await self.ray_service.initialize()
        
        if ray_connected:
            # Initialize Anant graph system with Ray support
            anant_ready = await self.anant_service.initialize(
                ray_enabled=True,
                ray_service=self.ray_service
            )
            
            if anant_ready and database_connected:
                print("✅ Anant Graph API startup complete - All systems ready")
                return True
            elif anant_ready:
                print("⚠️ Anant Graph API startup with degraded service - Database unavailable")
                return False
            else:
                print("⚠️ Anant Graph API startup with degraded service - Anant graph unavailable")
                return False
        else:
            print("⚠️ Anant Graph API startup with degraded service - Ray cluster unavailable")
            return False
    
    async def shutdown(self):
        """Cleanup all services on shutdown"""
        print("🛑 Shutting down Anant Graph API...")
        
        # Shutdown Anant service first
        await self.anant_service.shutdown()
        
        # Then shutdown Ray service
        await self.ray_service.shutdown()
        
        # Finally shutdown database
        await self.database_service.shutdown()
        
        print("✅ Anant Graph API shutdown complete")
    
    def is_ready(self) -> bool:
        """Check if all services are ready"""
        return (
            self.ray_service.is_connected() and 
            self.anant_service.is_initialized() and
            self.database_service.is_connected()
        )
    
    def get_status(self) -> dict:
        """Get comprehensive status of all services"""
        return {
            "startup_time": self.startup_time,
            "uptime_seconds": time.time() - self.startup_time if self.startup_time else 0,
            "ready": self.is_ready(),
            "ray_service": self.ray_service.get_status(),
            "anant_service": self.anant_service.get_status(),
            "database_service": self.database_service.get_status()
        }


# Global lifecycle manager
app_lifecycle = AppLifecycle()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager
    """
    # Startup
    await app_lifecycle.startup()
    
    # Application is ready
    yield
    
    # Shutdown
    await app_lifecycle.shutdown()