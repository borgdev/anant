"""
FastAPI Application with Ray Scaling
====================================

Main FastAPI application that leverages Ray for distributed processing
and scaling across multiple workers.
"""

import os
import ray
import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import configuration
from config import config

# Import sub-applications
from apps.analytics import analytics_app
from apps.ml_service import ml_app
from apps.data_processing import data_app
from apps.monitoring import monitoring_app


# Global state for Ray cluster
class RayClusterState:
    def __init__(self):
        self.connected = False
        self.connection_attempts = 0
        self.last_connection_error: Optional[str] = None
        
    def is_connected(self) -> bool:
        return self.connected and ray.is_initialized()


ray_state = RayClusterState()


async def initialize_ray() -> bool:
    """Initialize Ray connection with async support."""
    max_retries = config.RAY_RETRY_ATTEMPTS
    retry_delay = config.RAY_RETRY_DELAY
    
    for attempt in range(max_retries):
        try:
            ray_state.connection_attempts += 1
            
            if not ray.is_initialized():
                ray_kwargs = config.get_ray_init_kwargs()
                
                # Use asyncio.to_thread for Ray initialization in async context
                await asyncio.to_thread(ray.init, **ray_kwargs)
                
                # Verify connection
                cluster_resources = await asyncio.to_thread(ray.cluster_resources)
                
                ray_state.connected = True
                ray_state.last_connection_error = None
                
                print(f"✅ Connected to Ray cluster (attempt {attempt + 1})")
                print(f"📊 Cluster resources: {cluster_resources}")
                return True
                
        except Exception as e:
            ray_state.last_connection_error = str(e)
            print(f"❌ Failed to connect to Ray cluster (attempt {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                print("🔄 Starting local Ray cluster as fallback")
                try:
                    await asyncio.to_thread(ray.init)
                    ray_state.connected = True
                    ray_state.last_connection_error = None
                    return True
                except Exception as e2:
                    ray_state.last_connection_error = str(e2)
                    print(f"💥 Failed to start local Ray cluster: {e2}")
                    ray_state.connected = False
                    return False
    
    ray_state.connected = False
    return False


async def shutdown_ray():
    """Shutdown Ray connection gracefully."""
    try:
        if ray.is_initialized():
            await asyncio.to_thread(ray.shutdown)
            print("🔌 Ray cluster connection closed")
    except Exception as e:
        print(f"⚠️ Error during Ray shutdown: {e}")
    finally:
        ray_state.connected = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI app startup and shutdown.
    This replaces the deprecated @app.on_event decorators.
    """
    # Startup
    print("🚀 Starting FastAPI application with Ray scaling...")
    
    # Initialize Ray cluster connection
    ray_connected = await initialize_ray()
    
    if ray_connected:
        print("✅ Application startup complete - Ray cluster ready")
    else:
        print("⚠️ Application startup complete - Ray cluster unavailable")
    
    # Application is ready
    yield
    
    # Shutdown
    print("🛑 Shutting down FastAPI application...")
    await shutdown_ray()
    print("✅ Application shutdown complete")


# Create main FastAPI app with modern lifespan management
app = FastAPI(
    title=config.APP_NAME,
    description="FastAPI application with Ray-based distributed processing",
    version=config.APP_VERSION,
    lifespan=lifespan,  # Modern lifespan management
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=config.CORS_CREDENTIALS,
    allow_methods=config.CORS_METHODS,
    allow_headers=config.CORS_HEADERS,
)

# Mount sub-applications
app.mount("/analytics", analytics_app)
app.mount("/ml", ml_app)
app.mount("/data", data_app)
app.mount("/monitoring", monitoring_app)


@app.get("/", response_model=Dict[str, Any])
async def root() -> Dict[str, Any]:
    """Root endpoint with cluster information."""
    cluster_info = {}
    
    if ray_state.is_connected():
        try:
            # Use asyncio.to_thread for Ray calls in async context
            cluster_resources = await asyncio.to_thread(ray.cluster_resources)
            available_resources = await asyncio.to_thread(ray.available_resources)
            nodes = await asyncio.to_thread(ray.nodes)
            
            cluster_info = {
                "cluster_resources": cluster_resources,
                "available_resources": available_resources,
                "nodes": len(nodes),
                "ray_version": ray.__version__,
                "connection_status": "connected"
            }
        except Exception as e:
            cluster_info = {
                "error": f"Failed to get cluster info: {str(e)}",
                "connection_status": "error"
            }
    else:
        cluster_info = {
            "connection_status": "disconnected",
            "last_error": ray_state.last_connection_error,
            "connection_attempts": ray_state.connection_attempts
        }
    
    return {
        "message": f"Welcome to {config.APP_NAME}",
        "version": config.APP_VERSION,
        "status": "running",
        "ray_cluster": cluster_info,
        "sub_apps": [
            {
                "name": "Analytics", 
                "path": "/analytics", 
                "description": "Distributed analytics and reporting with Polars",
                "endpoints": ["/analytics/", "/analytics/compute", "/analytics/reports", "/analytics/realtime"]
            },
            {
                "name": "ML Service", 
                "path": "/ml", 
                "description": "Machine Learning training and inference",
                "endpoints": ["/ml/", "/ml/train", "/ml/predict", "/ml/models"]
            },
            {
                "name": "Data Processing", 
                "path": "/data", 
                "description": "ETL and data processing pipelines with Polars",
                "endpoints": ["/data/", "/data/process", "/data/pipeline", "/data/generate"]
            },
            {
                "name": "Monitoring", 
                "path": "/monitoring", 
                "description": "System and cluster monitoring",
                "endpoints": ["/monitoring/", "/monitoring/metrics", "/monitoring/health", "/monitoring/dashboard"]
            }
        ],
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json"
        },
        "features": {
            "ray_version": ray.__version__ if ray_state.is_connected() else "disconnected",
            "polars_support": True,
            "async_processing": True,
            "distributed_computing": ray_state.is_connected()
        }
    }


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> JSONResponse:
    """Health check endpoint with proper HTTP status codes."""
    health_status = {
        "status": "healthy", 
        "timestamp": time.time(),
        "version": config.APP_VERSION
    }
    
    # Check Ray cluster health
    if ray_state.is_connected():
        try:
            nodes = await asyncio.to_thread(ray.nodes)
            alive_nodes = sum(1 for node in nodes if node.get("Alive", False))
            
            health_status.update({
                "ray_status": "connected",
                "cluster_nodes": len(nodes),
                "alive_nodes": alive_nodes,
                "cluster_healthy": alive_nodes > 0,
                "connection_attempts": ray_state.connection_attempts
            })
        except Exception as e:
            health_status.update({
                "ray_status": "error",
                "ray_error": str(e),
                "cluster_healthy": False
            })
    else:
        health_status.update({
            "ray_status": "disconnected",
            "cluster_healthy": False,
            "last_error": ray_state.last_connection_error,
            "connection_attempts": ray_state.connection_attempts
        })
    
    # Determine overall health and HTTP status
    if not health_status.get("cluster_healthy", False):
        health_status["status"] = "degraded"
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status
        )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=health_status
    )


@app.get("/cluster/status")
async def cluster_status() -> Dict[str, Any]:
    """Get detailed cluster status with modern async patterns."""
    if not ray_state.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Ray cluster not connected"
        )
    
    try:
        # Use asyncio.to_thread for all Ray calls
        resources, available, nodes = await asyncio.gather(
            asyncio.to_thread(ray.cluster_resources),
            asyncio.to_thread(ray.available_resources),
            asyncio.to_thread(ray.nodes)
        )
        
        # Calculate utilization
        cpu_total = resources.get("CPU", 0)
        cpu_available = available.get("CPU", 0)
        cpu_usage = ((cpu_total - cpu_available) / cpu_total * 100) if cpu_total > 0 else 0
        
        memory_total = resources.get("memory", 0)
        memory_available = available.get("memory", 0)
        memory_usage = ((memory_total - memory_available) / memory_total * 100) if memory_total > 0 else 0
        
        alive_nodes = sum(1 for node in nodes if node.get("Alive", False))
        
        return {
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
            "ray_version": ray.__version__,
            "connection_info": {
                "connected": ray_state.connected,
                "connection_attempts": ray_state.connection_attempts,
                "last_error": ray_state.last_connection_error
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to get cluster status: {str(e)}"
        )


@app.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get application configuration (non-sensitive)."""
    return {
        "app_name": config.APP_NAME,
        "app_version": config.APP_VERSION,
        "debug": config.DEBUG,
        "ray_namespace": config.RAY_NAMESPACE,
        "worker_counts": {
            "compute_workers": config.COMPUTE_WORKERS,
            "ml_trainers": config.ML_TRAINERS,
            "ml_inference_servers": config.ML_INFERENCE_SERVERS,
            "data_processors": config.DATA_PROCESSORS
        },
        "timeouts": {
            "ml_training": config.ML_TRAINING_TIMEOUT,
            "ml_inference": config.ML_INFERENCE_TIMEOUT,
            "data_processing": config.DATA_PROCESSING_TIMEOUT
        },
        "features": {
            "polars_data_processing": True,
            "async_endpoints": True,
            "modern_fastapi": True,
            "ray_integration": ray_state.is_connected()
        }
    }


@app.post("/cluster/scale")
async def scale_workers(worker_type: str, count: int) -> Dict[str, Any]:
    """Scale specific worker types (for demonstration)."""
    if not ray_state.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Ray cluster not connected"
        )
    
    if count < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Worker count must be non-negative"
        )
    
    # This is a simplified example - in practice, you'd implement proper worker management
    return {
        "message": f"Scaling {worker_type} workers to {count}",
        "worker_type": worker_type,
        "target_count": count,
        "current_cluster_nodes": len(await asyncio.to_thread(ray.nodes)) if ray_state.is_connected() else 0,
        "note": "This is a demonstration endpoint - actual scaling depends on Ray cluster configuration"
    }


@app.post("/cluster/reconnect")
async def reconnect_ray() -> Dict[str, Any]:
    """Reconnect to Ray cluster."""
    if ray_state.is_connected():
        return {
            "message": "Already connected to Ray cluster",
            "status": "connected",
            "connection_attempts": ray_state.connection_attempts
        }
    
    success = await initialize_ray()
    
    return {
        "message": "Ray reconnection attempt completed",
        "status": "connected" if success else "failed",
        "success": success,
        "connection_attempts": ray_state.connection_attempts,
        "last_error": ray_state.last_connection_error
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


if __name__ == "__main__":
    uvicorn_config = config.get_uvicorn_kwargs()
    uvicorn.run(
        "main:app",
        **uvicorn_config
    )