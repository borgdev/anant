"""
Anant Graph API Configuration
============================

Environment configuration for Anant Graph API with Ray integration.
"""

import os
from typing import Optional, List, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import field_validator, Field
import secrets


class Settings(BaseSettings):
    """Application configuration with Pydantic validation."""
    
    # Application settings
    APP_NAME: str = "Anant Graph API"
    APP_VERSION: str = "1.0.0"
    ANANT_VERSION: str = "2024.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8088
    RELOAD: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Ray settings
    RAY_ADDRESS: Optional[str] = None  # Will be set by environment
    RAY_NAMESPACE: str = "anant_graph"
    RAY_CONNECT_TIMEOUT: int = 30
    RAY_RETRY_ATTEMPTS: int = 3
    RAY_RETRY_DELAY: float = 5.0
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["*"]
    
    # Security settings
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    TRUSTED_HOSTS: Optional[List[str]] = None
    
    # Database settings
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "anant_secure_2024"
    POSTGRES_DB: str = "anant_enterprise"
    DATABASE_URL: Optional[str] = None
    
    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info) -> str:
        if isinstance(v, str):
            return v
        values = info.data if hasattr(info, 'data') else {}
        # Build simple connection URL
        return (
            f"postgresql+asyncpg://"
            f"{values.get('POSTGRES_USER', 'postgres')}:"
            f"{values.get('POSTGRES_PASSWORD', 'anant_secure_2024')}@"
            f"{values.get('POSTGRES_HOST', 'postgres')}:"
            f"{values.get('POSTGRES_PORT', 5432)}/"
            f"{values.get('POSTGRES_DB', 'anant_enterprise')}"
        )
    
    # Redis settings
    REDIS_URL: str = "redis://:anant_cluster_2024@cache:6379/0"
    REDIS_TTL: int = 3600
    
    # Anant Graph settings
    ANANT_DATA_PATH: str = "/app/data"
    ANANT_STORAGE_BACKEND: str = "ray"
    ANANT_ENABLE_DISTRIBUTED: bool = True
    ANANT_CACHE_SIZE: int = 10000
    ANANT_BATCH_SIZE: int = 1000
    
    # Graph processing settings
    GRAPH_WORKERS: int = 4
    ANALYTICS_WORKERS: int = 2
    KNOWLEDGE_WORKERS: int = 2
    
    # Performance settings
    MAX_GRAPH_NODES: int = 1000000
    MAX_GRAPH_EDGES: int = 5000000
    COMPUTATION_TIMEOUT: int = 300
    QUERY_TIMEOUT: int = 60
    
    # Monitoring settings
    ENABLE_MONITORING: bool = True
    METRICS_COLLECTION_INTERVAL: int = 30
    HEALTH_CHECK_INTERVAL: int = 10
    
    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return bool(v)
    
    @field_validator("RELOAD", mode="before")
    @classmethod
    def parse_reload(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return bool(v)
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            if v.strip() == "*":
                return ["*"]
            elif v.startswith('[') and v.endswith(']'):
                # Handle JSON-like format
                import json
                try:
                    return json.loads(v)
                except:
                    return ["*"]
            else:
                # Handle comma-separated format
                return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v if v else ["*"]
    
    @field_validator("TRUSTED_HOSTS", mode="before")
    @classmethod
    def parse_trusted_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    def get_ray_init_kwargs(self) -> Dict[str, Any]:
        """Get Ray initialization kwargs."""
        kwargs = {
            "namespace": self.RAY_NAMESPACE,
            "ignore_reinit_error": True,
        }
        
        if self.RAY_ADDRESS:
            kwargs["address"] = self.RAY_ADDRESS
        
        return kwargs
    
    def get_uvicorn_kwargs(self) -> Dict[str, Any]:
        """Get Uvicorn server kwargs."""
        return {
            "host": self.HOST,
            "port": self.PORT,
            "reload": self.RELOAD,
            "log_level": self.LOG_LEVEL.lower(),
        }
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        # Don't parse env vars as JSON for certain fields
        env_nested_delimiter = "__"


class DevelopmentSettings(Settings):
    """Development configuration."""
    DEBUG: bool = True
    RELOAD: bool = True
    LOG_LEVEL: str = "DEBUG"
    RAY_ADDRESS: Optional[str] = None  # Use local Ray for development


class ProductionSettings(Settings):
    """Production configuration."""
    DEBUG: bool = False
    RELOAD: bool = False
    LOG_LEVEL: str = "INFO"
    TRUSTED_HOSTS: Optional[List[str]] = ["localhost", "127.0.0.1", "anant-ray-head"]


class TestingSettings(Settings):
    """Testing configuration."""
    DEBUG: bool = True
    DATABASE_URL: Optional[str] = "postgresql+asyncpg://anant_dev:dev_password@postgres-dev:5432/anant_test"
    RAY_ADDRESS: Optional[str] = None
    GRAPH_WORKERS: int = 1
    ANALYTICS_WORKERS: int = 1
    KNOWLEDGE_WORKERS: int = 1


def get_settings() -> Settings:
    """Get configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Export the active configuration
settings = get_settings()