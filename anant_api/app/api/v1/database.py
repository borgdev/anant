"""
Database management endpoints
Database status, health checks, and connection info
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.lifecycle import app_lifecycle
from ...services.database_service import get_db

router = APIRouter()


@router.get("/status")
async def database_status() -> Dict[str, Any]:
    """Get database connection status"""
    if not app_lifecycle.database_service.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Database not connected"
        )
    
    try:
        health_info = await app_lifecycle.database_service.health_check()
        status_info = app_lifecycle.database_service.get_status()
        
        return {
            "connected": True,
            "health": health_info,
            "connection_info": status_info,
            "service_status": "operational"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to get database status: {str(e)}"
        )


@router.get("/health")
async def database_health():
    """Get detailed database health check"""
    try:
        health = await app_lifecycle.database_service.health_check()
        
        if not health.get("healthy", False):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=health
            )
        
        return health
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to check database health: {str(e)}"
        )


@router.get("/test-connection")
async def test_database_connection(db: AsyncSession = Depends(get_db)):
    """Test database connection with a simple query"""
    try:
        from sqlalchemy import text
        result = await db.execute(text("SELECT 1 as test_value"))
        test_result = result.scalar()
        
        return {
            "connection_test": "passed",
            "test_query": "SELECT 1",
            "result": test_result,
            "status": "operational"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection test failed: {str(e)}"
        )


@router.get("/info")
async def database_info():
    """Get database configuration information"""
    if not app_lifecycle.database_service.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Database not connected"
        )
    
    try:
        health = await app_lifecycle.database_service.health_check()
        status_info = app_lifecycle.database_service.get_status()
        
        return {
            "database_info": {
                "name": health.get("database", "unknown"),
                "user": health.get("user", "unknown"),
                "version": health.get("version", "unknown"),
                "status": health.get("status", "unknown")
            },
            "connection_info": {
                "connected": status_info.get("connected", False),
                "database_url": status_info.get("database_url", "unknown"),
                "pool_size": status_info.get("engine_pool_size", 0)
            },
            "service_status": "operational" if health.get("healthy") else "degraded"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to get database info: {str(e)}"
        )