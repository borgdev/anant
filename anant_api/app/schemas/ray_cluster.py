"""
Ray cluster schemas
Pydantic models for Ray cluster endpoints
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class ClusterOverview(BaseModel):
    total_nodes: int
    alive_nodes: int
    dead_nodes: int
    cpu_utilization_percent: float
    memory_utilization_percent: float
    cluster_health: str


class ResourceUtilization(BaseModel):
    cpu_percent: float
    memory_percent: float


class ClusterResources(BaseModel):
    total: Dict[str, Any]
    available: Dict[str, Any]
    utilization: ResourceUtilization


class ClusterStatusResponse(BaseModel):
    connected: bool
    cluster_overview: Optional[ClusterOverview] = None
    resources: Optional[ClusterResources] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    ray_version: Optional[str] = None
    error: Optional[str] = None


class NodesResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    cluster_overview: ClusterOverview
    total_nodes: int
    alive_nodes: int


class ResourcesResponse(BaseModel):
    resources: ClusterResources
    utilization: ClusterOverview
    ray_version: Optional[str] = None