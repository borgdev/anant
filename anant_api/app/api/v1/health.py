"""
Health check endpoints
System health, readiness, and liveness checks
"""

import time
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from ...core.lifecycle import app_lifecycle
from ...config import settings

router = APIRouter()


@router.get("/", status_code=status.HTTP_200_OK)
async def health_check() -> JSONResponse:
    """Main health check endpoint with proper HTTP status codes"""
    health_status = {
        "status": "healthy", 
        "timestamp": time.time(),
        "version": settings.APP_VERSION,
        "service": "anant_graph_api"
    }
    
    # Get service statuses
    lifecycle_status = app_lifecycle.get_status()
    
    # Check Ray cluster health
    ray_status = lifecycle_status.get("ray_service", {})
    health_status.update({
        "ray_status": "connected" if ray_status.get("connected") else "disconnected",
        "ray_error": ray_status.get("last_error"),
        "connection_attempts": ray_status.get("connection_attempts", 0)
    })
    
    # Check Anant graph health
    anant_status = lifecycle_status.get("anant_service", {})
    health_status.update({
        "anant_status": "ready" if anant_status.get("initialized") else "not_initialized",
        "anant_error": anant_status.get("last_error")
    })
    
    # Check database health
    database_status = lifecycle_status.get("database_service", {})
    health_status.update({
        "database_status": "connected" if database_status.get("connected") else "disconnected",
        "database_url": database_status.get("database_url", "unknown")
    })
    
    # Determine overall health and HTTP status
    overall_healthy = app_lifecycle.is_ready()
    
    if not overall_healthy:
        health_status["status"] = "degraded"
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status
        )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=health_status
    )


@router.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes/Docker"""
    ready = app_lifecycle.is_ready()
    
    return JSONResponse(
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "ready": ready,
            "timestamp": time.time(),
            "services": app_lifecycle.get_status()
        }
    )


@router.get("/live")
async def liveness_check():
    """Liveness probe for Kubernetes/Docker"""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "alive": True,
            "timestamp": time.time(),
            "uptime": app_lifecycle.get_status().get("uptime_seconds", 0)
        }
    )


@router.get("/detailed")
async def detailed_health():
    """Detailed health information for monitoring"""
    try:
        # Get detailed health from Anant service
        anant_health = await app_lifecycle.anant_service.health_check()
        
        # Get Ray cluster info
        ray_info = await app_lifecycle.ray_service.get_cluster_info()
        
        # Get database health
        database_health = await app_lifecycle.database_service.health_check()
        
        return {
            "overall_status": "healthy" if app_lifecycle.is_ready() else "degraded",
            "timestamp": time.time(),
            "services": {
                "anant_graph": anant_health,
                "ray_cluster": ray_info,
                "database": database_health
            },
            "system": app_lifecycle.get_status()
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "overall_status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
        )