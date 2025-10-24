"""
Environment Configuration
========================

Simple environment configuration for FastAPI + Ray application.
"""

import os
from typing import Optional, Dict, Any


class Config:
    """Application configuration."""
    
    # Application settings
    APP_NAME = os.getenv("APP_NAME", "FastAPI Ray Cluster")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Server settings
    HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
    PORT = int(os.getenv("FASTAPI_PORT", "8080"))
    RELOAD = os.getenv("FASTAPI_RELOAD", "false").lower() == "true"
    
    # Ray settings
    RAY_ADDRESS = os.getenv("RAY_ADDRESS", None)
    RAY_NAMESPACE = os.getenv("RAY_NAMESPACE", "fastapi")
    
    # Ray cluster connection settings
    RAY_CONNECT_TIMEOUT = int(os.getenv("RAY_CONNECT_TIMEOUT", "30"))
    RAY_RETRY_ATTEMPTS = int(os.getenv("RAY_RETRY_ATTEMPTS", "3"))
    RAY_RETRY_DELAY = float(os.getenv("RAY_RETRY_DELAY", "5.0"))
    
    # Worker settings
    COMPUTE_WORKERS = int(os.getenv("COMPUTE_WORKERS", "3"))
    ML_TRAINERS = int(os.getenv("ML_TRAINERS", "2"))
    ML_INFERENCE_SERVERS = int(os.getenv("ML_INFERENCE_SERVERS", "2"))
    DATA_PROCESSORS = int(os.getenv("DATA_PROCESSORS", "3"))
    
    # CORS settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    CORS_CREDENTIALS = os.getenv("CORS_CREDENTIALS", "true").lower() == "true"
    CORS_METHODS = os.getenv("CORS_METHODS", "*").split(",")
    CORS_HEADERS = os.getenv("CORS_HEADERS", "*").split(",")
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Monitoring settings
    ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
    METRICS_COLLECTION_INTERVAL = int(os.getenv("METRICS_COLLECTION_INTERVAL", "30"))
    
    # Analytics settings
    ANALYTICS_CACHE_SIZE = int(os.getenv("ANALYTICS_CACHE_SIZE", "1000"))
    ANALYTICS_BATCH_SIZE = int(os.getenv("ANALYTICS_BATCH_SIZE", "100"))
    
    # ML settings
    ML_MODEL_CACHE_SIZE = int(os.getenv("ML_MODEL_CACHE_SIZE", "10"))
    ML_TRAINING_TIMEOUT = int(os.getenv("ML_TRAINING_TIMEOUT", "300"))
    ML_INFERENCE_TIMEOUT = int(os.getenv("ML_INFERENCE_TIMEOUT", "30"))
    
    # Data processing settings
    DATA_CHUNK_SIZE = int(os.getenv("DATA_CHUNK_SIZE", "1000"))
    DATA_PROCESSING_TIMEOUT = int(os.getenv("DATA_PROCESSING_TIMEOUT", "120"))
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    
    @classmethod
    def get_ray_init_kwargs(cls) -> Dict[str, Any]:
        """Get Ray initialization kwargs."""
        kwargs = {
            "namespace": cls.RAY_NAMESPACE,
            "ignore_reinit_error": True,
        }
        
        if cls.RAY_ADDRESS:
            kwargs["address"] = cls.RAY_ADDRESS
        
        return kwargs
    
    @classmethod
    def get_uvicorn_kwargs(cls) -> Dict[str, Any]:
        """Get Uvicorn server kwargs."""
        return {
            "host": cls.HOST,
            "port": cls.PORT,
            "reload": cls.RELOAD,
            "log_level": cls.LOG_LEVEL.lower(),
        }


# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    RELOAD = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    RELOAD = False
    LOG_LEVEL = "INFO"


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    RAY_ADDRESS = None  # Use local Ray for testing
    COMPUTE_WORKERS = 1
    ML_TRAINERS = 1
    DATA_PROCESSORS = 1


def get_config() -> Config:
    """Get configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()


# Export the active configuration
config = get_config()