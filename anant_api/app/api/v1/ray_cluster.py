"""
Ray cluster management endpoints
Ray cluster status, monitoring, and operations
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any

from ...core.lifecycle import app_lifecycle

router = APIRouter()


@router.get("/status")
async def cluster_status() -> Dict[str, Any]:
    """Get detailed Ray cluster status"""
    if not app_lifecycle.ray_service.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Ray cluster not connected"
        )
    
    try:
        cluster_info = await app_lifecycle.ray_service.get_cluster_info()
        return cluster_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to get cluster status: {str(e)}"
        )


@router.get("/nodes")
async def cluster_nodes():
    """Get information about Ray cluster nodes"""
    if not app_lifecycle.ray_service.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Ray cluster not connected"
        )
    
    try:
        cluster_info = await app_lifecycle.ray_service.get_cluster_info()
        return {
            "nodes": cluster_info.get("nodes", []),
            "cluster_overview": cluster_info.get("cluster_overview", {}),
            "total_nodes": cluster_info.get("cluster_overview", {}).get("total_nodes", 0),
            "alive_nodes": cluster_info.get("cluster_overview", {}).get("alive_nodes", 0)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to get cluster nodes: {str(e)}"
        )


@router.get("/resources")
async def cluster_resources():
    """Get Ray cluster resource information"""
    if not app_lifecycle.ray_service.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Ray cluster not connected"
        )
    
    try:
        cluster_info = await app_lifecycle.ray_service.get_cluster_info()
        return {
            "resources": cluster_info.get("resources", {}),
            "utilization": cluster_info.get("cluster_overview", {}),
            "ray_version": cluster_info.get("ray_version")
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to get cluster resources: {str(e)}"
        )