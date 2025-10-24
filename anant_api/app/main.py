"""
Anant Graph API - Main Application
==================================

Clean, enterprise-grade FastAPI application with proper separation of concerns.
"""

import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import core components
from .config import settings
from .core.lifecycle import lifespan, app_lifecycle
from .api.v1.api import api_router

# Import middleware
from .middleware.auth import AuthenticationMiddleware


def create_application() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title=settings.APP_NAME,
        description="Anant Graph API - Distributed graph processing and knowledge management",
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None
    )

    # Add middleware
    setup_middleware(app)
    
    # Add routes
    setup_routes(app)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """Configure application middleware"""
    
    # Security middleware
    app.add_middleware(AuthenticationMiddleware)
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware
    if settings.TRUSTED_HOSTS:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.TRUSTED_HOSTS
        )


def setup_routes(app: FastAPI) -> None:
    """Configure application routes"""
    
    # Include API v1 routes
    app.include_router(
        api_router, 
        prefix="/api/v1"
    )
    
    # Root endpoint
    @app.get("/")
    async def root():
        """API root endpoint with system information"""
        return {
            "message": f"Welcome to {settings.APP_NAME}",
            "version": settings.APP_VERSION,
            "status": "ready" if app_lifecycle.is_ready() else "starting",
            "documentation": {
                "swagger_ui": "/docs" if settings.DEBUG else "disabled",
                "redoc": "/redoc" if settings.DEBUG else "disabled",
                "openapi_json": "/openapi.json" if settings.DEBUG else "disabled"
            },
            "api": {
                "v1": "/api/v1",
                "health": "/api/v1/health",
                "ray_cluster": "/api/v1/ray", 
                "anant_graph": "/api/v1/anant",
                "database": "/api/v1/database",
                "authentication": "/api/v1/auth"
            },
            "features": {
                "ray_integration": app_lifecycle.ray_service.is_connected(),
                "anant_graph": app_lifecycle.anant_service.is_initialized(),
                "database_connection": app_lifecycle.database_service.is_connected(),
                "distributed_computing": True,
                "authentication": True,
                "real_time_processing": True
            },
            "system": app_lifecycle.get_status()
        }
    
    # System info endpoint
    @app.get("/info")
    async def system_info():
        """Detailed system information"""
        return {
            "app": {
                "name": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "debug": settings.DEBUG,
                "environment": settings.ENVIRONMENT
            },
            "services": app_lifecycle.get_status(),
            "config": {
                "ray_address": settings.RAY_ADDRESS,
                "ray_namespace": settings.RAY_NAMESPACE,
                "anant_version": settings.ANANT_VERSION
            }
        }


def setup_exception_handlers(app: FastAPI) -> None:
    """Configure global exception handlers"""
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled errors"""
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
                "timestamp": time.time(),
                "path": str(request.url),
                "method": request.method
            }
        )


# Create the application instance
app = create_application()


if __name__ == "__main__":
    """Run the application directly"""
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )