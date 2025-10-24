"""
Anant graph service
Manages Anant graph system integration
"""

import asyncio
from typing import Optional, Dict, Any

from ..config import settings
from .ray_service import RayService

# Import Anant graph functionality with fallback
try:
    from anant_api.anant import initialize_anant_graph, get_anant_graph
except ImportError:
    # Fallback for Docker deployment
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    try:
        from anant_api.anant import initialize_anant_graph, get_anant_graph
    except ImportError:
        # Mock functions if Anant is not available
        def initialize_anant_graph(**kwargs):
            class MockAnantGraph:
                def get_status(self): return {"status": "mock", "initialized": True}
                def health_check(self): return {"healthy": True, "metrics": {}}
                def get_detailed_status(self): return {"status": "mock_detailed", "capabilities": []}
                def cleanup(self): pass
            return MockAnantGraph()
        
        def get_anant_graph():
            return None


class AnantService:
    """Service for managing Anant graph system"""
    
    def __init__(self):
        self.initialized = False
        self.anant_graph = None
        self.last_error: Optional[str] = None
        self.ray_enabled = False
        
    async def initialize(self, ray_enabled: bool = False, ray_service: Optional[RayService] = None) -> bool:
        """Initialize Anant graph system"""
        try:
            print("🔧 Initializing Anant graph system...")
            
            self.ray_enabled = ray_enabled and ray_service and ray_service.is_connected()
            
            # Initialize Anant graph with Ray support
            self.anant_graph = await asyncio.to_thread(
                initialize_anant_graph,
                ray_enabled=self.ray_enabled,
                config=settings
            )
            
            self.initialized = True
            self.last_error = None
            print("✅ Anant graph system initialized successfully")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            print(f"❌ Failed to initialize Anant graph system: {e}")
            self.initialized = False
            return False
    
    async def shutdown(self):
        """Shutdown Anant graph system gracefully"""
        try:
            print("🛑 Shutting down Anant graph system...")
            if self.anant_graph:
                # Cleanup Anant graph resources
                await asyncio.to_thread(self.anant_graph.cleanup)
        except Exception as e:
            print(f"⚠️ Error during Anant shutdown: {e}")
        finally:
            self.initialized = False
            self.anant_graph = None
    
    def is_initialized(self) -> bool:
        """Check if Anant graph is initialized"""
        return self.initialized and self.anant_graph is not None
    
    async def get_graph_status(self) -> Dict[str, Any]:
        """Get Anant graph status"""
        if not self.is_initialized():
            return {
                "initialized": False,
                "error": self.last_error or "Not initialized"
            }
        
        try:
            status = await asyncio.to_thread(self.anant_graph.get_status)
            return {
                "initialized": True,
                "status": status,
                "capabilities": [
                    "Hierarchical Knowledge Graphs",
                    "Multi-Modal Analysis", 
                    "Distributed Processing" if self.ray_enabled else "Local Processing",
                    "Real-time Graph Operations"
                ]
            }
        except Exception as e:
            return {
                "initialized": True,
                "error": str(e)
            }
    
    async def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed Anant graph status"""
        if not self.is_initialized():
            raise Exception("Anant graph system not initialized")
        
        try:
            detailed_status = await asyncio.to_thread(self.anant_graph.get_detailed_status)
            return {
                "anant_graph": detailed_status,
                "ray_integration": self.ray_enabled,
                "initialization_time": detailed_status.get("initialization_time"),
                "capabilities": {
                    "hierarchical_graphs": True,
                    "multi_modal_analysis": True,
                    "distributed_processing": self.ray_enabled,
                    "real_time_operations": True,
                    "knowledge_graphs": True
                }
            }
        except Exception as e:
            raise Exception(f"Failed to get detailed status: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Anant graph"""
        if not self.is_initialized():
            return {
                "healthy": False,
                "status": "not_initialized",
                "error": self.last_error
            }
        
        try:
            health = await asyncio.to_thread(self.anant_graph.health_check)
            return {
                "healthy": health.get("healthy", False),
                "status": "ready",
                "metrics": health.get("metrics", {})
            }
        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "error": str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get Anant service status"""
        return {
            "initialized": self.initialized,
            "ray_enabled": self.ray_enabled,
            "last_error": self.last_error,
            "anant_version": settings.ANANT_VERSION
        }
    
    def get_graph_instance(self):
        """Get the Anant graph instance for advanced operations"""
        return self.anant_graph if self.is_initialized() else None