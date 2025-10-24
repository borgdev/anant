"""
Anant graph schemas
Pydantic models for Anant graph endpoints
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class AnantCapabilities(BaseModel):
    hierarchical_graphs: bool = True
    multi_modal_analysis: bool = True
    distributed_processing: bool
    real_time_operations: bool = True
    knowledge_graphs: bool = True


class AnantStatusResponse(BaseModel):
    anant_graph: Dict[str, Any]
    ray_integration: bool
    initialization_time: Optional[float] = None
    capabilities: AnantCapabilities


class AnantHealthResponse(BaseModel):
    healthy: bool
    status: str
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AnantCapabilitiesResponse(BaseModel):
    capabilities: List[str]
    ray_enabled: bool
    version: Optional[str] = None
    distributed_processing: bool