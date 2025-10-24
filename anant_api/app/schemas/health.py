"""
Health check schemas
Pydantic models for health endpoints
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"


class ServiceStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected" 
    ERROR = "error"
    READY = "ready"
    NOT_INITIALIZED = "not_initialized"


class BasicHealthResponse(BaseModel):
    status: HealthStatus
    timestamp: float
    version: str
    service: str = "anant_graph_api"


class DetailedHealthResponse(BaseModel):
    overall_status: HealthStatus
    timestamp: float
    services: Dict[str, Any]
    system: Dict[str, Any]


class ReadinessResponse(BaseModel):
    ready: bool
    timestamp: float
    services: Dict[str, Any]


class LivenessResponse(BaseModel):
    alive: bool = True
    timestamp: float
    uptime: float