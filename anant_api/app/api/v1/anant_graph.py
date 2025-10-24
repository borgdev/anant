"""
Anant graph management endpoints
Core Anant graph operations and status
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Dict, Any

from ...core.lifecycle import app_lifecycle

router = APIRouter()


@router.get("/status")
async def anant_status() -> Dict[str, Any]:
    """Get detailed Anant graph system status"""
    if not app_lifecycle.anant_service.is_initialized():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Anant graph system not initialized"
        )
    
    try:
        detailed_status = await app_lifecycle.anant_service.get_detailed_status()
        return detailed_status
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to get Anant status: {str(e)}"
        )


@router.get("/health")
async def anant_health():
    """Get Anant graph health check"""
    try:
        health = await app_lifecycle.anant_service.health_check()
        
        if not health.get("healthy", False):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=health
            )
        
        return health
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to check Anant health: {str(e)}"
        )


@router.get("/capabilities")
async def anant_capabilities():
    """Get Anant graph capabilities"""
    if not app_lifecycle.anant_service.is_initialized():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Anant graph system not initialized"
        )
    
    try:
        status_info = await app_lifecycle.anant_service.get_graph_status()
        return {
            "capabilities": status_info.get("capabilities", []),
            "ray_enabled": app_lifecycle.anant_service.ray_enabled,
            "version": app_lifecycle.anant_service.get_status().get("anant_version"),
            "distributed_processing": app_lifecycle.anant_service.ray_enabled
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to get Anant capabilities: {str(e)}"
        )