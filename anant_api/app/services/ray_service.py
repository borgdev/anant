"""
Ray cluster service
Manages Ray cluster connection and operations
"""

import asyncio
import ray
from typing import Optional, Dict, Any

from ..config import settings


class RayService:
    """Service for managing Ray cluster connections"""
    
    def __init__(self):
        self.connected = False
        self.connection_attempts = 0
        self.last_error: Optional[str] = None
        
    async def initialize(self) -> bool:
        """Initialize Ray cluster connection"""
        max_retries = settings.RAY_RETRY_ATTEMPTS
        retry_delay = settings.RAY_RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                self.connection_attempts += 1
                
                if not ray.is_initialized():
                    ray_kwargs = {
                        "namespace": settings.RAY_NAMESPACE,
                        "ignore_reinit_error": True,
                    }
                    
                    if settings.RAY_ADDRESS:
                        ray_kwargs["address"] = settings.RAY_ADDRESS
                    
                    # Initialize Ray in async context
                    await asyncio.to_thread(ray.init, **ray_kwargs)
                    
                    # Verify connection
                    cluster_resources = await asyncio.to_thread(ray.cluster_resources)
                    
                    self.connected = True
                    self.last_error = None
                    
                    print(f"✅ Connected to Ray cluster (attempt {attempt + 1})")
                    print(f"📊 Cluster resources: {cluster_resources}")
                    return True
                    
            except Exception as e:
                self.last_error = str(e)
                print(f"❌ Failed to connect to Ray cluster (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    print("🔄 Starting local Ray cluster as fallback")
                    try:
                        await asyncio.to_thread(ray.init)
                        self.connected = True
                        self.last_error = None
                        return True
                    except Exception as e2:
                        self.last_error = str(e2)
                        print(f"💥 Failed to start local Ray cluster: {e2}")
                        self.connected = False
                        return False
        
        self.connected = False
        return False
    
    async def shutdown(self):
        """Shutdown Ray connection gracefully"""
        try:
            if ray.is_initialized():
                await asyncio.to_thread(ray.shutdown)
                print("🔌 Ray cluster connection closed")
        except Exception as e:
            print(f"⚠️ Error during Ray shutdown: {e}")
        finally:
            self.connected = False
    
    def is_connected(self) -> bool:
        """Check if Ray is connected"""
        return self.connected and ray.is_initialized()
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get detailed cluster information"""
        if not self.is_connected():
            return {
                "connected": False,
                "error": self.last_error,
                "connection_attempts": self.connection_attempts
            }
        
        try:
            # Use asyncio.to_thread for all Ray calls
            resources, available, nodes = await asyncio.gather(
                asyncio.to_thread(ray.cluster_resources),
                asyncio.to_thread(ray.available_resources),
                asyncio.to_thread(ray.nodes)
            )
            
            alive_nodes = sum(1 for node in nodes if node.get("Alive", False))
            
            # Calculate utilization
            cpu_total = resources.get("CPU", 0)
            cpu_available = available.get("CPU", 0)
            cpu_usage = ((cpu_total - cpu_available) / cpu_total * 100) if cpu_total > 0 else 0
            
            memory_total = resources.get("memory", 0)
            memory_available = available.get("memory", 0)
            memory_usage = ((memory_total - memory_available) / memory_total * 100) if memory_total > 0 else 0
            
            return {
                "connected": True,
                "cluster_overview": {
                    "total_nodes": len(nodes),
                    "alive_nodes": alive_nodes,
                    "dead_nodes": len(nodes) - alive_nodes,
                    "cpu_utilization_percent": round(cpu_usage, 2),
                    "memory_utilization_percent": round(memory_usage, 2),
                    "cluster_health": "healthy" if alive_nodes > 0 else "unhealthy"
                },
                "resources": {
                    "total": resources,
                    "available": available,
                    "utilization": {
                        "cpu_percent": round(cpu_usage, 2),
                        "memory_percent": round(memory_usage, 2)
                    }
                },
                "nodes": nodes,
                "ray_version": ray.__version__
            }
        except Exception as e:
            return {
                "connected": True,
                "error": f"Failed to get cluster info: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get Ray service status"""
        return {
            "connected": self.connected,
            "ray_initialized": ray.is_initialized() if self.connected else False,
            "connection_attempts": self.connection_attempts,
            "last_error": self.last_error,
            "ray_version": ray.__version__ if self.connected else None
        }