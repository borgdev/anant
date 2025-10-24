"""
Main API router
Combines all API endpoints
"""

from fastapi import APIRouter

from . import health, ray_cluster, anant_graph, database, auth

# Create main API router
api_router = APIRouter()

# Include sub-routers
api_router.include_router(
    health.router, 
    prefix="/health", 
    tags=["Health"]
)

api_router.include_router(
    ray_cluster.router, 
    prefix="/ray", 
    tags=["Ray Cluster"]
)

api_router.include_router(
    anant_graph.router, 
    prefix="/anant", 
    tags=["Anant Graph"]
)

api_router.include_router(
    database.router, 
    prefix="/database", 
    tags=["Database"]
)

api_router.include_router(
    auth.router, 
    prefix="/auth", 
    tags=["Authentication"]
)